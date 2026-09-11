import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
for path in ("",):
    while path in sys.path:
        sys.path.remove(path)

import torch  # noqa: F401
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests

# NOTE on coverage -- a real but currently UNCAUGHT failure mode:
# A torch_npu header may directly reference a struct/type that only exists in
# the NEWEST CANN it was built against. When a customer pairs that torch_npu
# with an OLDER system CANN (and CANN is placed in front in -I), the
# transitive ACL headers resolve to the older CANN, which lacks that
# referenced definition -> compile error. This is the same "version skew"
# family as the deep-path CANN mixing, just triggered through a torch_npu
# header's own dependency on a new CANN symbol rather than through the deep
# path. The current test env does NOT exhibit it (the torch_npu-vs-CANN skew
# here happens not to surface in symbols the framework headers directly
# reference), but it exists in the wild -- watch for it when the CANN gap
# widens or torch_npu starts using newer CANN APIs.
#
# This runs against the REAL env, so outcomes depend on the current CANN-vs-
# torch_npu version relationship (same / CANN-newer / CANN-older):
#   * stub-truncation failures (T-first, 报错二) are version-INDEPENDENT and
#     surface under any version relationship;
#   * version-skew-sensitive failures (deep-path CANN mixing [disabled above],
#     and the torch_npu-header-depends-on-new-CANN-symbol case) only manifest
#     under a skewed relationship -- a same-version env MASKS them, a mismatched
#     one reveals them. Re-run after switching CANN versions to observe how
#     behavior shifts across version relationships.


def _npu_include():
    return os.path.join(os.path.dirname(os.path.realpath(torch_npu.__file__)), "include")


def _cann_include():
    base = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not base:
        return None
    inc = os.path.join(base, "include")
    return inc if os.path.isdir(os.path.join(inc, "acl")) else None


def _third_party_dir():
    return os.path.join(_npu_include(), "third_party", "acl", "inc")


def _torch_dirs():
    """Extra -I needed to compile a torch_npu framework header (torch + python)."""
    torch_inc = os.path.join(os.path.dirname(os.path.realpath(torch.__file__)), "include")
    import sysconfig
    py_inc = sysconfig.get_path("include")
    return [torch_inc, os.path.join(torch_inc, "torch", "csrc", "api", "include"), py_inc]


def _cxx_std():
    """C++ standard for the installed torch: 2.14+ requires c++20, earlier c++17.
    If the toolchain can't deliver it (e.g. GCC 10 tops out below true C++20),
    framework tests fail loudly with torch's own '#error C++XX required' --
    a clear toolchain signal, not an ACL bug."""
    v = torch.__version__.split("+", 1)[0].split(".")
    major = int(v[0]) if v and v[0].isdigit() else 0
    minor = int(v[1]) if len(v) > 1 and v[1].isdigit() else 0
    return "c++20" if (major, minor) >= (2, 14) else "c++17"


def _compile(include_line, i_order, body="aclrtStream s = nullptr;\n"):
    """Compile a literal customer source against the REAL CANN + REAL torch_npu.

    No fabrication: real installed versions, real -I order, literal #include lines.
    Returns (ok, stderr). g++ writes both the -H include tree and the diagnostics
    to stderr (stdout is empty for -fsyntax-only); stderr is returned separately so
    callers can grep real 'error:' lines instead of trusting a tail that may be
    buried under the (huge, for framework style) -H tree.
    """
    src = include_line + "\n" + body
    with tempfile.TemporaryDirectory() as d:
        cpp = Path(d) / "t.cpp"
        cpp.write_text(src)
        args = ["g++", "-std=" + _cxx_std(), "-fsyntax-only", "-H"]
        args += ["-I" + x for x in i_order]
        args += [str(cpp)]
        proc = subprocess.run(args, capture_output=True, text=True)
    return proc.returncode == 0, (proc.stderr or "")


# Customer include styles (literal, exactly as a customer writes them).
# Each value: (include_line, extra_fixed_includes) where extra_fixed_includes are
# dirs always appended (e.g. torch/python for framework headers; do not carry acl/).
INCLUDES = {
    "shallow": ("#include <acl/acl_rt.h>", []),
    # Deep path is INTENTIONALLY DISABLED: torch_npu already migrated its source
    # to the shallow form (PR 795468cbc / 26.1.1) and the deep path is deprecated.
    # A customer still using it hits the CANN-mixing error (deep path pins the
    # first-level header to torch_npu while transitive deps get stolen by a
    # front-placed CANN) -> it is NOT a "must pass" scenario. Re-enable manually
    # only to characterize that legacy failure.
    # "deep": ('#include "third_party/acl/inc/acl/acl_rt.h"', []),
    "framework": (
        # Any torch_npu header that transitively pulls ACL works here as the probe
        # (it exercises torch_npu's own transitive include hygiene). NPUStream.h
        # is the default representative; swap in e.g. NPUEvent.h / NPUGraph.h /
        # CalcuOpUtil.h / AclInterface.h / ... as needed.
        '#include <torch/extension.h>\n#include "torch_npu/csrc/core/npu/NPUStream.h"',
        None,  # lazily resolved to _torch_dirs() per-run
    ),
}

# Realistic customer -I configurations: all subsets+orders of N/T/C.
# N = torch_npu/include, T = third_party/acl/inc, C = cann/include.
# N and T are both inside the torch_npu package, so they usually appear together;
# C (system CANN) is external and most commonly placed either FIRST or LAST
# relative to the torch_npu dirs (both orders covered below).
CONFIGS = {
    # single
    "N": lambda: [_npu_include()],
    "T": lambda: [_third_party_dir()],
    "C": lambda: [_cann_include()],
    # pair
    "N_T": lambda: [_npu_include(), _third_party_dir()],
    "T_N": lambda: [_third_party_dir(), _npu_include()],
    "N_C": lambda: [_npu_include(), _cann_include()],
    "C_N": lambda: [_cann_include(), _npu_include()],
    "T_C": lambda: [_third_party_dir(), _cann_include()],
    "C_T": lambda: [_cann_include(), _third_party_dir()],
    # triple (C-first and C-last are the common customer patterns)
    "C_N_T": lambda: [_cann_include(), _npu_include(), _third_party_dir()],
    "C_T_N": lambda: [_cann_include(), _third_party_dir(), _npu_include()],
    "N_T_C": lambda: [_npu_include(), _third_party_dir(), _cann_include()],
    "T_N_C": lambda: [_third_party_dir(), _npu_include(), _cann_include()],
    "N_C_T": lambda: [_npu_include(), _cann_include(), _third_party_dir()],
    "T_C_N": lambda: [_third_party_dir(), _cann_include(), _npu_include()],
}


class TestAclCustomerScenarios(TestCase):
    """Customer real-usage guard.

    Every realistic customer scenario (literal #include style x real -I order)
    is compiled against the REAL installed CANN and REAL installed torch_npu --
    no version fabrication, no path swapping. A failure is a real customer
    breakage: triage whether the usage is reasonable (fix torch_npu) or
    unreasonable (fix the customer script).
    """

    def _run(self, inc_kind, cfg_kind):
        if "C" in cfg_kind and _cann_include() is None:
            self.skipTest("CANN not found")
        # framework needs N (torch_npu/include) to resolve torch_npu headers;
        # configs without N are meaningless for it.
        if inc_kind == "framework" and "N" not in cfg_kind:
            self.skipTest("framework needs N to resolve torch_npu headers")
        inc_line, extra = INCLUDES[inc_kind]
        if extra is None:  # framework: resolve torch/python dirs lazily
            extra = _torch_dirs()
        i_order = CONFIGS[cfg_kind]() + extra
        ok, err = _compile(inc_line, i_order)
        if not ok:
            errs = [ln for ln in err.splitlines()
                    if "error:" in ln or "fatal error:" in ln]
            if errs:
                # dedupe by message text (the part after 'error:') so many
                # identical lines (e.g. 138x 'aclrtStream has not been
                # declared' across hccl.h params) collapse to one, and each
                # distinct error type shows once with its first file:line.
                seen, msgs = [], set()
                for ln in errs:
                    m = ln.split("error:", 1)[1].strip() if "error:" in ln else ln
                    if m not in msgs:
                        msgs.add(m)
                        seen.append(ln)
                snippet = "\n".join(seen[:20])
                if len(errs) > len(seen):
                    snippet += "\n... (共 %d 条 error,去重 %d 种,仅列前 %d 种)" % (
                        len(errs), len(seen), min(len(seen), 20))
            else:
                snippet = err[-700:]
        else:
            snippet = ""
        self.assertTrue(
            ok,
            msg=(
                f"\nCustomer scenario FAILED to compile.\n"
                f"  include style : {inc_kind}  ({inc_line})\n"
                f"  -I order      : {cfg_kind}\n"
                f"  -> a real customer hitting this would see a compile error.\n"
                f"--- errors ---\n{snippet}"
            ),
        )


def _make(inc_kind, cfg_kind):
    def m(self):
        self._run(inc_kind, cfg_kind)
    m.__name__ = "test_%s__%s" % (inc_kind, cfg_kind)
    return m


# Generate one test method per (include style x -I config) -- each reports
# independently so every broken customer scenario surfaces for triage.
for _inc in INCLUDES:
    for _cfg in CONFIGS:
        setattr(TestAclCustomerScenarios,
                "test_%s__%s" % (_inc, _cfg),
                _make(_inc, _cfg))


if __name__ == "__main__":
    run_tests()
