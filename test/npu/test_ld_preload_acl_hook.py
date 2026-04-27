# Integration test: verify that LD_PRELOAD-injected overrides of ACL symbols
# loaded via FunctionLoader (e.g. aclrtMallocAlign32) are picked up.
#
# Requires: an NPU environment with libascendcl available, a C compiler.
# Run: python test/npu/test_ld_preload_acl_hook.py

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


HOOK_SRC = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int aclError;
typedef int aclrtMemMallocPolicy;

__attribute__((visibility("default")))
aclError aclrtMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    const char *flag = getenv("ACL_HOOK_FLAG_FILE");
    if (flag != NULL) {
        FILE *fp = fopen(flag, "w");
        if (fp != NULL) {
            fprintf(fp, "hit\n");
            fclose(fp);
        }
    }
    typedef aclError (*real_fn_t)(void **, size_t, aclrtMemMallocPolicy);
    static real_fn_t real = NULL;
    if (real == NULL) {
        real = (real_fn_t)dlsym(RTLD_NEXT, "aclrtMallocAlign32");
    }
    if (real == NULL) {
        return -1;
    }
    return real(devPtr, size, policy);
}
"""


# Minimal Python payload: allocate a tensor on NPU, which eventually exercises
# aclrtMallocAlign32 via the caching allocator's large-block path.
PAYLOAD_SRC = textwrap.dedent(
    """
    import sys
    import torch
    import torch_npu
    torch.npu.set_device(0)
    t = torch.empty(1024 * 1024, dtype=torch.float32, device='npu')
    del t
    torch.npu.empty_cache()
    sys.exit(0)
    """
)


def _compile_hook(tmpdir, cc):
    src_path = os.path.join(tmpdir, "hook.c")
    so_path = os.path.join(tmpdir, "libacl_hook_test.so")
    with open(src_path, "w") as f:
        f.write(HOOK_SRC)
    subprocess.check_call(
        [cc, "-shared", "-fPIC", "-O2", "-o", so_path, src_path, "-ldl"]
    )
    return so_path


def _compile_unrelated(tmpdir, cc):
    src_path = os.path.join(tmpdir, "other.c")
    so_path = os.path.join(tmpdir, "libunrelated_hook.so")
    with open(src_path, "w") as f:
        f.write("int unrelated_noop(void) { return 0; }\n")
    subprocess.check_call(
        [cc, "-shared", "-fPIC", "-O2", "-o", so_path, src_path]
    )
    return so_path


def _run_payload(env):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(PAYLOAD_SRC)
        script = f.name
    try:
        return subprocess.run(
            [sys.executable, script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        os.unlink(script)


@unittest.skipIf(
    not torch_npu.npu.is_available(), "NPU not available; skipping LD_PRELOAD test"
)
class TestLdPreloadAclHook(TestCase):
    @classmethod
    def setUpClass(cls):
        cc = os.environ.get("CC", "gcc")
        if shutil.which(cc) is None:
            raise unittest.SkipTest(
                "C compiler '{}' not found; skipping LD_PRELOAD test".format(cc)
            )
        cls._cc = cc
        cls._tmpdir = tempfile.mkdtemp(prefix="acl_hook_")
        try:
            cls._hook_so = _compile_hook(cls._tmpdir, cc)
            cls._unrelated_so = _compile_unrelated(cls._tmpdir, cc)
        except Exception:
            shutil.rmtree(cls._tmpdir, ignore_errors=True)
            raise

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _base_env(self):
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        return env

    def test_without_preload_behaves_unchanged(self):
        """No LD_PRELOAD: allocation succeeds, hook never invoked."""
        flag = os.path.join(self._tmpdir, "flag_no_preload")
        env = self._base_env()
        env["ACL_HOOK_FLAG_FILE"] = flag
        result = _run_payload(env)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(
            os.path.exists(flag), "hook must not be called when LD_PRELOAD is unset"
        )

    def test_with_preload_hook_is_invoked(self):
        """LD_PRELOAD set to hook .so: hook is called for aclrtMallocAlign32."""
        flag = os.path.join(self._tmpdir, "flag_preload")
        env = self._base_env()
        env["LD_PRELOAD"] = self._hook_so
        env["ACL_HOOK_FLAG_FILE"] = flag
        result = _run_payload(env)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(
            os.path.exists(flag),
            "hook must be called when LD_PRELOAD overrides aclrtMallocAlign32",
        )

    def test_preload_without_symbol_falls_back(self):
        """LD_PRELOAD set to a .so that does NOT define ACL symbols:
        allocation must still succeed via fallback to libascendcl."""
        flag = os.path.join(self._tmpdir, "flag_unrelated")
        env = self._base_env()
        env["LD_PRELOAD"] = self._unrelated_so
        env["ACL_HOOK_FLAG_FILE"] = flag
        result = _run_payload(env)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(
            os.path.exists(flag),
            "unrelated preload must not cause the hook flag to be set",
        )

    def test_multiple_preload_sos(self):
        """LD_PRELOAD with multiple .so (hook first): hook still wins."""
        flag = os.path.join(self._tmpdir, "flag_multi")
        env = self._base_env()
        env["LD_PRELOAD"] = "{}:{}".format(self._hook_so, self._unrelated_so)
        env["ACL_HOOK_FLAG_FILE"] = flag
        result = _run_payload(env)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(flag), "first-loaded preload must win")


if __name__ == "__main__":
    run_tests()
