// DeepSeek-V4.1-Flash routed experts at prefill (TF_DSV41_EXPERT_PREFILL=1): TensorFold 0.6.0's grouped EXL3 expert
// GEMM (tensorfold/cuda/exl3/experts_grouped.cuh, any width, mul1 for this checkpoint) with each decoded trellis tile
// applied to MTL member tiles (MTL x 16 member rows) instead of one: upstream's kernel decodes the expert's trellis
// once per 16 members, so a prefill chunk (2,048 rows x top-6: ~32 members an expert, all 2,048 for the shared
// expert) decodes every tile 2-128 times; here 1-32 times (MTL 4). The decode (shuffles, 64-bit merges, the mul1
// hash with dp4a, half2 fma) is most of the kernel's instructions at prefill, the mma a small part.
//
// Kept from grouped_kernel<CB, NT, 4, PF, LO, HI>, per output element (so Z is upstream's, bit for bit, for every row
// count, every member grouping and every setting below):
//   - the K ranges: K split `split` of SK, warp w of W = 4 runs k tiles [split * KT / SK + w * PW, + PW) ascending;
//   - one mma.m16n8k16 (f16 -> f32) per (k tile, column tile, n8 half) into the element's accumulator from +0.0, its
//     A operand load_pair's values (X[row][k + 2t ..], 0 for a dead row), its B operand upstream's decode_tile<CB, K2>
//     of the tile's words (load_words: word lane + 32 l, 0 past the tile). An mma's element (i, j) depends on row i of
//     A, column j of B and its accumulator only, so which member tile or lane group a row sits in changes no bit;
//   - the warps' partials through shared memory added in warp order; Z stored for live member rows only, at upstream's
//     address (mat, split, pair row, column);
//   - the exits: u >= ucount, an empty member group; the K2 switch and its __trap outside the instance's [LO, HI].
//   decode_tile / mma16816 / load_pair / load_words / Fmt / LaneMap are upstream's (#include, not copied).
//
// Changed (work placement only): a program owns MTL member tiles (grid.z = mats * SK * ceil(maxm / (16 MTL))) and NT
// column tiles; for each k tile a warp decodes each of its NT tiles once and issues the 2 mma of every live member
// tile with it; the reduction runs tile by tile through one red[W][16][NT * 16] buffer (upstream's size at this NT).

#include "experts_grouped.cuh"

namespace dsv41_x3pf {

using tf_exl3x::decode_tile;
using tf_exl3x::Fmt;
using tf_exl3x::LaneMap;
using tf_exl3x::load_pair;
using tf_exl3x::load_words;
using tf_exl3x::mma16816;

constexpr int W = 4;   // warps a CTA: upstream's default_config, which fixes the K ranges

template <int CB, int K2, int NT, int MTL>
__device__ __forceinline__ void pf_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                         const half* const (&x0)[MTL], const half* const (&x1)[MTL],
                                         const bool (&ok0)[MTL], const bool (&ok1)[MTL], int live, int lane,
                                         float (&acc)[MTL][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t nx[NT][LW];                                         // the next k tile's words (one in flight)
    if (nkt > 0)
#pragma unroll
        for (int i = 0; i < NT; ++i) load_words<K2>(nx[i], tp + i * TW, lane);

    for (int it = 0; it < nkt; ++it) {
        uint32_t w[NT][LW];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int l = 0; l < LW; ++l) w[i][l] = nx[i][l];
        if (it + 1 < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(nx[i], tp + (size_t)(it + 1) * kstride + i * TW, lane);
        const int k = (kt0 + it) * 16;
        uint32_t a[MTL][4];
#pragma unroll
        for (int mt = 0; mt < MTL; ++mt) {
            const bool on = mt < live;
            a[mt][0] = load_pair(x0[mt] + k, on && ok0[mt]);
            a[mt][1] = load_pair(x1[mt] + k, on && ok1[mt]);
            a[mt][2] = load_pair(x0[mt] + k + 8, on && ok0[mt]);
            a[mt][3] = load_pair(x1[mt] + k + 8, on && ok1[mt]);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile<CB, K2>(w[i], map, lane, b0, b1);
#pragma unroll
            for (int mt = 0; mt < MTL; ++mt) {
                if (mt < live) {                                 // CTA-uniform
                    mma16816(acc[mt][i][0], a[mt], b0);
                    mma16816(acc[mt][i][1], a[mt], b1);
                }
            }
        }
    }
}

// Program (u, n block, (mat * SK + split) * MG + member group): MTL of grouped_kernel's member tiles.
template <int CB, int NT, int MTL, int LO, int HI>
__global__ void __launch_bounds__(W * 32) pf_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MG = (maxm + 16 * MTL - 1) / (16 * MTL);
    const int mg = blockIdx.z % MG;
    const int split = (blockIdx.z / MG) % SK;
    const int mat = blockIdx.z / MG / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[MTL * 16];
    for (int i = threadIdx.x; i < MTL * 16; i += W * 32) {
        const int m = mg * MTL * 16 + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                                  // members come first: an empty group
    int live = 0;                                                // member tiles with members (a prefix)
#pragma unroll
    for (int mt = 0; mt < MTL; ++mt)
        if (rows_sh[mt * 16] >= 0) live = mt + 1;

    const half* x0[MTL];
    const half* x1[MTL];
    bool ok0[MTL], ok1[MTL];
#pragma unroll
    for (int mt = 0; mt < MTL; ++mt) {
        const int r0 = rows_sh[mt * 16 + g], r1 = rows_sh[mt * 16 + g + 8];
        ok0[mt] = r0 >= 0;
        ok1[mt] = r1 >= 0;
        x0[mt] = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
        x1[mt] = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    float acc[MTL][NT][2][4];
#pragma unroll
    for (int mt = 0; mt < MTL; ++mt)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[mt][i][h][c] = 0.f;

    switch (k2) {
#define DSV41_X3PF_CASE(K2_)                                                                                     \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            pf_tiles<CB, K2_, NT, MTL>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, live, lane, acc);       \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        DSV41_X3PF_CASE(2)
        DSV41_X3PF_CASE(3)
        DSV41_X3PF_CASE(4)
        DSV41_X3PF_CASE(5)
        DSV41_X3PF_CASE(6)
        DSV41_X3PF_CASE(7)
        DSV41_X3PF_CASE(8)
        DSV41_X3PF_CASE(9)
        DSV41_X3PF_CASE(10)
        DSV41_X3PF_CASE(11)
        DSV41_X3PF_CASE(12)
        DSV41_X3PF_CASE(13)
        DSV41_X3PF_CASE(14)
        DSV41_X3PF_CASE(15)
        DSV41_X3PF_CASE(16)
#undef DSV41_X3PF_CASE
        default:
            __trap();
    }

    // grouped_kernel's reduction, one member tile at a time: the warps' partials added in warp order
    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int mt = 0; mt < MTL; ++mt) {
        if (mt >= live) break;                                   // CTA-uniform
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[mt][i][h][0];
                red[warp][g][col + 1] = acc[mt][i][h][1];
                red[warp][g + 8][col] = acc[mt][i][h][2];
                red[warp][g + 8][col + 1] = acc[mt][i][h][3];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[mt * 16 + row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        __syncthreads();
    }
}

struct Args {
    const half* x0;
    const half* x1;
    const int64_t* tp0;
    const int64_t* tp1;
    const int* k2_0;
    const int* k2_1;
    const int* uids;
    const int* ucount;
    const int* members;
    float* z;
    int K, N, P, SK, maxm, slots, nexp_max, mats;
};

template <int CB, int NT, int MTL, int LO, int HI>
void launch(const Args& a, cudaStream_t stream) {
    const int MG = (a.maxm + 16 * MTL - 1) / (16 * MTL);
    dim3 grid((unsigned)a.nexp_max, (unsigned)(a.N / (16 * NT)), (unsigned)(a.mats * a.SK * MG));
    pf_kernel<CB, NT, MTL, LO, HI><<<grid, W * 32, 0, stream>>>(a.x0, a.x1, a.tp0, a.tp1, a.k2_0, a.k2_1, a.uids,
                                                                 a.ucount, a.members, a.z, a.K, a.N, a.P, a.SK,
                                                                 a.maxm, a.slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The instances: mul1; widths 8..8 (DSpark's 4-bit experts) and 2..10 (routed 2/3-bit + the shared expert at 4/5
// bits); (nt, mtl) = (4, 4) (64 members a program) and (8, 2).
template <int NT, int MTL>
bool dispatch_ranges(const Args& a, cudaStream_t stream, int lo, int hi) {
    if (lo == 8 && hi == 8) launch<2, NT, MTL, 8, 8>(a, stream);
    else if (lo >= 2 && hi <= 10) launch<2, NT, MTL, 2, 10>(a, stream);
    else return false;
    return true;
}

}  // namespace dsv41_x3pf

void dsv41_x3pf_grouped_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0,
                             const at::Tensor& TP1, const at::Tensor& B0, const at::Tensor& B1,
                             const at::Tensor& uids, const at::Tensor& ucount, const at::Tensor& members, at::Tensor& Z,
                             int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK, int64_t slots, int64_t cb,
                             int64_t nt, int64_t mtl, int64_t lo, int64_t hi) {
    using namespace dsv41_x3pf;
    TORCH_CHECK(cb == 2, "x3pf: built for the mul1 codebook (cb 2) only");
    TORCH_CHECK(K % (16 * SK * W) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    TORCH_CHECK(members.size(0) == uids.size(0), "one member row a distinct-expert slot");
    Args a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats;
    auto stream = at::cuda::getCurrentCUDAStream();
    bool ok = false;
    if (nt == 4 && mtl == 4) ok = dispatch_ranges<4, 4>(a, stream, (int)lo, (int)hi);
    else if (nt == 8 && mtl == 2) ok = dispatch_ranges<8, 2>(a, stream, (int)lo, (int)hi);
    TORCH_CHECK(ok, "x3pf: unsupported (nt, mtl, lo, hi) = (", nt, ", ", mtl, ", ", lo, ", ", hi, ")");
}
