#!/usr/bin/env python3
"""Copy the staged serving layer (engine/serving) and its tests (tests/serving) into the TensorFold work tree on
branch dsv41-060: ``src/tensorfold/families/deepseek_v41/cuda/<module>.py`` and ``tests/test_dsv41_serving_*.py``
(+ ``tests/dsv41_serving_fakes.py``, ``tests/fixtures/dsv41_encoding_ref.py``). Prints the paths it wrote (add exactly
those). The work tree is shared: this never runs git.

    python3 scripts/port_serving.py [<engine checkout>]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("protocol", "pool", "state", "sessions", "sessdisk", "plan", "rounds", "batch", "memory", "encoding",
           "dsml", "structured", "app", "stack")
TESTS = ("batch", "memory", "template", "dsml", "app")
PKG = "tensorfold.families.deepseek_v41.cuda"
TORCH_TESTS = ("batch", "memory", "app")


def _test_source(text: str, name: str) -> str:
    text = text.replace("from engine.serving import", f"from {PKG} import")
    text = re.sub(r"from engine\.serving\.(\w+) import", rf"from {PKG}.\1 import", text)
    text = text.replace("from fake_forward import", "from dsv41_serving_fakes import")
    text = text.replace("from test_template import", "from test_dsv41_serving_template import")
    text = text.replace('Path(__file__).parent / "data" / "encoding_dsv41_ref.py"',
                        'Path(__file__).parent / "fixtures" / "dsv41_encoding_ref.py"')
    text = text.replace("    from engine.kernels.csa2 import rows\n",
                        f'    rows = pytest.importorskip("{PKG}.csa2.rows")\n')
    head = "from dsv41_serving_fakes import model_dir_fixture, topo_fixture  # noqa: F401  (fixtures)\n"
    if name in TORCH_TESTS:
        text = text.replace("import pytest\n", 'import pytest\n\npytest.importorskip("torch")\n', 1)
    last = max(m.end() for m in re.finditer(r"^(?:from \S+ import [^\n(]*|import \S+)[^\n]*\n", text, re.M))
    return text[:last] + head + text[last:]


def main() -> None:
    tf = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "<engine checkout>").expanduser()
    dst = tf / "src/tensorfold/families/deepseek_v41/cuda"
    if not dst.is_dir():
        raise SystemExit(f"{dst}: no deepseek_v41 family (branch dsv41-060?)")
    wrote = []
    for m in MODULES:
        src = (ROOT / "engine/serving" / f"{m}.py").read_text()
        if "engine.kernels" in src or "from engine." in src:
            raise SystemExit(f"{m}.py still imports from the staging repo")
        (dst / f"{m}.py").write_text(src)
        wrote.append(dst / f"{m}.py")
    conf = (ROOT / "tests/serving/conftest.py").read_text()
    fake = (ROOT / "tests/serving/fake_forward.py").read_text()
    fake = fake.replace("from engine.serving.", f"from {PKG}.")
    keep = conf[conf.index("TEXT = {"):conf.index("def pytest_configure")].rstrip() + "\n"
    keep = keep.replace('str(ROOT / ".cache" / "dsv41-tok")', 'str(Path.home() / ".cache" / "dsv41-tok")')
    keep = keep.replace("    from engine.serving import topology\n", f"    from {PKG} import topology\n")
    fake = fake.replace("from __future__ import annotations\n\n",
                        "from __future__ import annotations\n\nimport os\nfrom pathlib import Path\n\n", 1)
    fake = fake.replace("import numpy as np\nimport torch\n", "import numpy as np\nimport pytest\nimport torch\n", 1)
    imports_end = fake.index("M64 = ")
    fakes = fake[:imports_end] + keep + "\n\n" + fake[imports_end:]
    fakes = fakes.replace('"""A fake V4.1 forward for the serving tests',
                          '"""The serving tests\' fixtures (topology, tokenizer folder) and a fake V4.1 forward', 1)
    (tf / "tests/dsv41_serving_fakes.py").write_text(fakes)
    wrote.append(tf / "tests/dsv41_serving_fakes.py")
    for t in TESTS:
        src = (ROOT / "tests/serving" / f"test_{t}.py").read_text()
        out = tf / "tests" / f"test_dsv41_serving_{t}.py"
        out.write_text(_test_source(src, t))
        wrote.append(out)
    (tf / "tests/fixtures").mkdir(exist_ok=True)
    ref = tf / "tests/fixtures/dsv41_encoding_ref.py"
    ref.write_text((ROOT / "tests/serving/data/encoding_dsv41_ref.py").read_text())
    wrote.append(ref)
    for p in wrote:
        print(p.relative_to(tf))


if __name__ == "__main__":
    main()
