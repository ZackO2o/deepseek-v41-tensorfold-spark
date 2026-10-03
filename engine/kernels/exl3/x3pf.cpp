// Bindings of x3pf.cu: upstream exl3 ``grouped`` (tensorfold/cuda/exl3/experts.cpp, the same arguments and Z) with
// each decoded trellis tile applied to mtl member tiles: nt column tiles a program, mtl member tiles a program.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void dsv41_x3pf_grouped_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                             const at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                             int64_t, int64_t, int64_t, int64_t, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

void grouped(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
             const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
             const at::Tensor& members, at::Tensor Z, int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK,
             int64_t slots, int64_t cb, int64_t nt, int64_t mtl, int64_t lo, int64_t hi) {
    check(X0, at::kHalf, "X0");
    check(X1, at::kHalf, "X1");
    check(TP0, at::kLong, "TP0");
    check(TP1, at::kLong, "TP1");
    check(B0, at::kInt, "B0");
    check(B1, at::kInt, "B1");
    check(uids, at::kInt, "uids");
    check(ucount, at::kInt, "ucount");
    check(members, at::kInt, "members");
    check(Z, at::kFloat, "Z");
    TORCH_CHECK(Z.numel() >= mats * SK * P * N, "Z too small");
    TORCH_CHECK(X0.numel() >= P * K && X1.numel() >= P * K, "X too small");
    c10::cuda::CUDAGuard guard(X0.device());
    dsv41_x3pf_grouped_cuda(X0, X1, TP0, TP1, B0, B1, uids, ucount, members, Z, mats, K, N, P, SK, slots, cb, nt, mtl,
                            lo, hi);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped", &grouped);
}
