"""HCCL customer-scenario compile matrix (real env, no NPU device needed).

Layer 2 (installed-product gate): runs against the REAL installed torch_npu
and REAL CANN.  The source-tree stage (packaging + framework-chain orders on a
staged package) lives in test/test_hccl_header_compatibility.py as layer 1.

Layout contract under test (mirrors the ACL endgame, PR !45706): real headers
at BOTH the public include root and the legacy deep path; the deep path is
deprecated.  See handoff 18-hccl-header-compat.md for the full story.

  * MUST PASS   : shallow <hccl/hccl.h> in any -I order (except T alone: the
                  legacy dir carries no acl headers); both framework orders
                  (NPUStream.h first / hccl_comm.h first; <hccl/hccl.h>
                  comes in transitively via HcclInterface) under CANN-first --
                  the C_N cells are the clarified requirement acceptance.
  * SKIPPED     : cells outside the acceptance contract -- N/T-first with
                  hccl_comm.h (a CANN-only header cannot coexist with the
                  bundled set under that -I order; customers put CANN first),
                  framework without N or C, shallow with T alone.
  * NOT COVERED : the legacy deep path (deprecated, deliberately not tested,
                  same policy as the ACL test_cpp_compile.py).

Compile-only (-fsyntax-only): ABI compatibility and CANN/torch_npu version
pairing are explicitly OUT OF SCOPE.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
for path in ("", _REPO_ROOT):
    while path in sys.path:
        sys.path.remove(path)

import torch  # noqa: F401
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests

PASS, SKIP = "pass", "skip"


def _npu_include():
    return os.path.join(os.path.dirname(os.path.realpath(torch_npu.__file__)), "include")


def _cann_include():
    bases = [os.environ.get(var) for var in
             ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_AICPU_PATH")]
    bases = [b for b in bases if b]
    bases.append("/usr/local/Ascend/ascend-toolkit/latest")
    for base in bases:
        for sub in ("", "aarch64-linux", "x86_64-linux"):
            inc = os.path.join(base, sub, "include")
            if os.path.isfile(os.path.join(inc, "hccl", "hccl.h")):
                return inc
    return None


def _cann_has_comm():
    inc = _cann_include()
    return bool(inc) and os.path.isfile(os.path.join(inc, "hccl", "hccl_comm.h"))


_FRAMEWORK_STYLES = ("framework", "framework_comm_first")
_STYLES_WITH_COMM = _FRAMEWORK_STYLES


def _hccl_third_party_dir():
    return os.path.join(_npu_include(), "third_party", "hccl", "inc")


def _torch_dirs():
    torch_inc = os.path.join(os.path.dirname(os.path.realpath(torch.__file__)), "include")
    import sysconfig
    return [torch_inc,
            os.path.join(torch_inc, "torch", "csrc", "api", "include"),
            sysconfig.get_path("include")]


def _cxx_std():
    v = torch.__version__.split("+", 1)[0].split(".")
    major = int(v[0]) if v and v[0].isdigit() else 0
    minor = int(v[1]) if len(v) > 1 and v[1].isdigit() else 0
    return "c++20" if (major, minor) >= (2, 14) else "c++17"


def _compile(src, i_order):
    with tempfile.TemporaryDirectory() as d:
        cpp = Path(d) / "t.cpp"
        cpp.write_text(src)
        args = ["g++", "-std=" + _cxx_std(), "-fsyntax-only"]
        args += ["-I" + x for x in i_order]
        args += [str(cpp)]
        proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stderr or "")


def _dedup_errors(err):
    errs = [ln for ln in err.splitlines() if "error:" in ln or "fatal error:" in ln]
    seen, msgs = [], set()
    for ln in errs:
        m = ln.split("error:", 1)[1].strip() if "error:" in ln else ln
        if m not in msgs:
            msgs.add(m)
            seen.append(ln)
    snippet = "\n".join(seen[:10])
    if len(errs) > len(seen):
        snippet += "\n... (%d error lines, %d distinct)" % (len(errs), len(seen))
    return snippet or err[-500:]


BODY = ("void probe(){HcclCommConfig c{};"
        "HcclCommConfigInit(&c);"
        "HcclComm comm=nullptr;(void)comm;}")

# The legacy deep path is intentionally NOT covered (deprecated, same as the
# ACL test): shallow + framework match the customer scenarios; the shallow
# T-first PASS cells double as the layout-change detector.
INCLUDES = {
    "shallow": ('#include <hccl/hccl.h>', []),
    # Two framework orders = the clarified requirement scenario: NPUStream.h
    # and hccl/hccl_comm.h in either source order (the C_N cells are the
    # acceptance under the customer's -I).
    "framework": ('#include <torch/extension.h>\n'
                  '#include "torch_npu/csrc/core/npu/NPUStream.h"\n'
                  '#include <hccl/hccl_comm.h>', None),
    "framework_comm_first": ('#include <torch/extension.h>\n'
                             '#include <hccl/hccl_comm.h>\n'
                             '#include "torch_npu/csrc/core/npu/NPUStream.h"', None),
}

CONFIGS = {
    "N": lambda: [_npu_include()],
    "T": lambda: [_hccl_third_party_dir()],
    "C": lambda: [_cann_include()],
    "N_T": lambda: [_npu_include(), _hccl_third_party_dir()],
    "T_N": lambda: [_hccl_third_party_dir(), _npu_include()],
    "N_C": lambda: [_npu_include(), _cann_include()],
    "C_N": lambda: [_cann_include(), _npu_include()],
    "T_C": lambda: [_hccl_third_party_dir(), _cann_include()],
    "C_T": lambda: [_cann_include(), _hccl_third_party_dir()],
    "C_N_T": lambda: [_cann_include(), _npu_include(), _hccl_third_party_dir()],
    "C_T_N": lambda: [_cann_include(), _hccl_third_party_dir(), _npu_include()],
    "N_T_C": lambda: [_npu_include(), _hccl_third_party_dir(), _cann_include()],
    "T_N_C": lambda: [_hccl_third_party_dir(), _npu_include(), _cann_include()],
    "N_C_T": lambda: [_npu_include(), _cann_include(), _hccl_third_party_dir()],
    "T_C_N": lambda: [_hccl_third_party_dir(), _cann_include(), _npu_include()],
}


def _classify(style, cfg):
    first = cfg.split("_")[0]
    has_n = "N" in cfg
    has_c = "C" in cfg
    if style in _FRAMEWORK_STYLES:
        if not has_n:
            return SKIP, "NPUStream.h resolves through torch_npu include root (N)"
        if not has_c:
            return SKIP, "hccl/hccl_comm.h only exists in CANN"
        if first == "C":
            return PASS, "contract: CANN-first (customer configuration)"
        return SKIP, ("N/T-first + <hccl/hccl_comm.h>: a CANN-only header cannot "
                      "coexist with the bundled header set under this -I order; "
                      "out of the acceptance contract (customers put CANN first)")
    if cfg == "T":
        return SKIP, "legacy dir carries no acl headers; <acl/acl.h> needs N or C"
    return PASS, "contract: real headers at both roots, any -I order"


class TestHcclHeaderMatrix(TestCase):
    def _run(self, style, cfg):
        verdict, reason = _classify(style, cfg)
        if verdict == SKIP:
            self.skipTest(reason)
        if "C" in cfg and _cann_include() is None:
            self.skipTest("CANN not found")
        if style in _STYLES_WITH_COMM and not _cann_has_comm():
            self.skipTest(
                "CANN has no hccl/hccl_comm.h (header layout changed, e.g. "
                "pre-9.2.0 or a future refactor): re-evaluate these styles "
                "against the requirement; shallow/framework umbrella cells "
                "below still cover the stable <hccl/hccl.h> contract")
        inc_line, extra = INCLUDES[style]
        if extra is None:
            extra = _torch_dirs()
        i_order = [d for d in CONFIGS[cfg]() if d] + extra
        ok, err = _compile(inc_line + "\n" + BODY, i_order)
        if verdict == PASS:
            self.assertTrue(ok, (
                "\nHCCL contract cell FAILED (must pass).\n"
                "  include style : %s\n  -I order      : %s (%s)\n%s"
            ) % (style, cfg, reason, _dedup_errors(err)))


def _make(style, cfg):
    def m(self):
        self._run(style, cfg)
    m.__name__ = "test__%s__%s" % (style, cfg)
    return m


for _style in INCLUDES:
    for _cfg in CONFIGS:
        setattr(TestHcclHeaderMatrix,
                "test__%s__%s" % (_style, _cfg),
                _make(_style, _cfg))


if __name__ == "__main__":
    run_tests()
