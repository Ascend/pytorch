# Owner(s): ["module: inductor"]
"""Install latest akg/mfusion wheels, then smoke-test TORCHINDUCTOR_USE_AKG=1.

Repository: https://repo.mindspore.cn/mindspore/akg/newest/
  - akg     -> scheduler/{arch}/
  - mfusion -> fusion/{arch}/

When akg/torch_mlir is installed, verify AkgScheduling and akg_auto_fallback.
When akg is missing, verify the same entry falls back to NpuMlirScheduling and
mlir_auto_fallback, matching npu_inductor_plugin.register_mlir_codegen_backend().

Also reuses test_torch_mlir.TestAdd pointwise cases for end-to-end smoke coverage.

Usage:
  python test_akg_torch_mlir.py                         # try install akg, then run tests
  python test_akg_torch_mlir.py --insecure              # bypass SSL verification
  python test_akg_torch_mlir.py --download-only         # download wheels only (fail on error)
  python test_akg_torch_mlir.py --arch aarch64 --py-tag cp311

If wheel download/install fails (e.g. network), tests still run using MLIR fallback.
"""

from __future__ import annotations

import sys

# Install-only flags must be stripped before importing common_utils (it reads sys.argv at import).
_INSTALL_FLAGS = frozenset({"--download-only", "--insecure"})
_INSTALL_VALUE_FLAGS = frozenset({"--arch", "--py-tag", "--download-dir"})
SAVED_INSTALL_ARGV: list[str] = []


def _strip_install_cli_args() -> None:
    """Remove install-only flags from sys.argv; save them for install_akg_wheels()."""
    global SAVED_INSTALL_ARGV
    install_tokens: list[str] = []
    test_argv: list[str] = []
    index = 1
    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg in _INSTALL_FLAGS:
            install_tokens.append(arg)
            index += 1
        elif arg in _INSTALL_VALUE_FLAGS and index + 1 < len(sys.argv):
            install_tokens.extend([arg, sys.argv[index + 1]])
            index += 2
        else:
            test_argv.append(arg)
            index += 1
    SAVED_INSTALL_ARGV = install_tokens
    sys.argv = [sys.argv[0], *test_argv]


if __name__ == "__main__":
    _strip_install_cli_args()

import argparse
import hashlib
import http.client
import importlib
import logging
import os
import platform
import re
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import torch
from torch._inductor.codegen.common import device_codegens
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import run_tests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AKG wheel download / install (from install_akg.py)
# ---------------------------------------------------------------------------

BASE_URL = "https://repo.mindspore.cn/mindspore/akg/newest/"
PACKAGES = {"akg": "scheduler/{arch}/", "mfusion": "fusion/{arch}/"}
ARCH_MAP = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
PIP_INDEX = "https://mirrors.aliyun.com/pypi/simple"
TRUSTED_HOSTS = ("repo.mindspore.cn", "mirrors.aliyun.com", "pypi.org", "files.pythonhosted.org")
MAX_RETRIES, RETRY_DELAY_SEC, CHUNK_SIZE = 3, 3, 1024 * 1024
_NET_ERRORS = (
    urllib.error.URLError,
    urllib.error.ContentTooShortError,
    http.client.HTTPException,
    TimeoutError,
    ConnectionResetError,
    BrokenPipeError,
    OSError,
)

_ssl_ctx: ssl.SSLContext | None = None
_insecure = False


class IntegrityError(Exception):
    pass


def _with_retry(label: str, fn: Callable[[], object], errors: tuple[type[BaseException], ...] = _NET_ERRORS):
    last_error: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except RuntimeError:
            raise
        except errors as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            logger.info("%s failed (attempt %d/%d)", label, attempt, MAX_RETRIES)
            time.sleep(RETRY_DELAY_SEC)
    raise RuntimeError(f"{label} failed after {MAX_RETRIES} attempts") from last_error


def _open(url: str, timeout: int):
    kwargs = {"timeout": timeout, "context": _ssl_ctx} if _ssl_ctx else {"timeout": timeout}
    try:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "test_akg_torch_mlir.py/1.0"}), **kwargs
        )
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise RuntimeError(
                "SSL certificate verification failed. Retry with: python test_akg_torch_mlir.py --insecure"
            ) from exc
        raise


def fetch(url: str, timeout: int = 120) -> bytes:
    def _read() -> bytes:
        with _open(url, timeout) as resp:
            return resp.read()

    return _with_retry(f"fetch {url}", _read)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_expected_hash(sha256_url: str) -> str | None:
    try:
        return fetch(sha256_url).decode().strip().split()[0]
    except urllib.error.HTTPError:
        return None


def download(url: str, dest: Path, sha256_url: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = _fetch_expected_hash(sha256_url)
    if not expected:
        logger.warning("No checksum file found; skipping verification for %s", dest.name)
    elif dest.exists():
        if _sha256(dest) == expected:
            logger.info("%s already present and verified, skipping download", dest.name)
            return
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")

    def _once() -> None:
        with _open(url, 600) as resp, tmp.open("wb") as out:
            nbytes = 0
            while chunk := resp.read(CHUNK_SIZE):
                out.write(chunk)
                nbytes += len(chunk)
            if resp.length is not None and nbytes != resp.length:
                raise IntegrityError(f"incomplete download for {dest.name}")
        if expected and _sha256(tmp) != expected:
            raise IntegrityError(f"SHA256 mismatch for {dest.name}")
        tmp.replace(dest)
        logger.info("Downloaded %s (%.2f MiB)", dest.name, dest.stat().st_size / 1024 / 1024)

    _with_retry(f"download {dest.name}", _once, (*_NET_ERRORS, IntegrityError))
    if expected:
        logger.info("SHA256 verified for %s", dest.name)


def resolve_target(arch_arg: str | None, py_tag_arg: str | None) -> tuple[str, str]:
    arch = arch_arg or ARCH_MAP.get(platform.machine().lower())
    if not arch:
        raise RuntimeError(f"Unsupported machine type {platform.machine()!r}; use --arch")
    version = sys.version_info[:2]
    if py_tag_arg:
        return arch, py_tag_arg
    if version not in {(3, 10), (3, 11), (3, 12)}:
        raise RuntimeError(f"No wheels for Python {version[0]}.{version[1]}; use 3.10-3.12 or --py-tag")
    return arch, f"cp{version[0]}{version[1]}"


def pick_wheel(subdir: str, package: str, py_tag: str, arch: str) -> str:
    token = f"linux_{arch}"
    wheels = sorted(
        name
        for name in (
            href.split("/")[-1]
            for href in re.findall(r'href="([^"]+\.whl)"', fetch(BASE_URL + subdir).decode("utf-8", "replace"))
            if not href.endswith(".whl.sha256")
        )
        if name.startswith(f"{package}-") and f"-{py_tag}-{py_tag}-" in name and token in name
    )
    if not wheels:
        raise RuntimeError(f"No {package} wheel found for py={py_tag}, arch={arch}")
    return wheels[-1]


def pip_install(wheels: list[Path]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "--extra-index-url", PIP_INDEX]
    if _insecure:
        for host in TRUSTED_HOSTS:
            cmd.extend(["--trusted-host", host])
    cmd.extend(map(str, wheels))
    logger.info("Running: %s", " ".join(cmd))
    _with_retry("pip install", lambda: subprocess.run(cmd, check=True), (subprocess.CalledProcessError,))


def cleanup_downloads(download_dir: Path) -> None:
    if not download_dir.exists():
        return
    for path in download_dir.glob("*.whl*"):
        path.unlink(missing_ok=True)
    if not any(download_dir.iterdir()):
        download_dir.rmdir()
    logger.info("Removed downloaded wheels from %s", download_dir.resolve())


def log_release_info() -> None:
    try:
        text = fetch(BASE_URL + "release_info.yaml").decode()
        info = {
            k: (m.group(1)[:12] if k == "commit_id" else m.group(1))
            for k in ("branch", "commit_id", "date")
            if (m := re.search(rf"{k}:\s*(\S+)", text))
        }
        if info:
            logger.info(
                "Latest build: branch=%s, commit=%s, date=%s",
                info.get("branch"),
                info.get("commit_id"),
                info.get("date"),
            )
    except (urllib.error.URLError, RuntimeError):
        logger.warning("Failed to read release_info.yaml", exc_info=True)


def install_akg_wheels(args: argparse.Namespace) -> bool:
    """Download and optionally install akg/mfusion wheels.

    Returns True when wheels are installed successfully.
    Returns False when install fails (network, checksum, pip, etc.).
    Raises on failure when --download-only is set.
    """
    global _ssl_ctx, _insecure

    if args.insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        _ssl_ctx, _insecure = ctx, True
        logger.warning("HTTPS verification disabled; wheel SHA256 verification remains enabled")

    arch, py_tag = resolve_target(args.arch, args.py_tag)
    logger.info("Repository: %s", BASE_URL)
    logger.info("Target arch=%s, py_tag=%s", arch, py_tag)
    log_release_info()

    wheels: list[Path] = []
    try:
        for package, subdir_tpl in PACKAGES.items():
            subdir = subdir_tpl.format(arch=arch)
            name = pick_wheel(subdir, package, py_tag, arch)
            dest = args.download_dir / name
            logger.info("Processing package %s: %s", package, name)
            download(BASE_URL + subdir + name, dest, BASE_URL + subdir + name + ".sha256")
            wheels.append(dest)
        if args.download_only:
            logger.info("Wheels saved to %s", args.download_dir.resolve())
            return True
        pip_install(wheels)
        logger.info("Installation completed")
        return True
    except Exception:
        if args.download_only:
            raise
        logger.warning(
            "AKG wheel installation failed; continuing with MLIR fallback tests",
            exc_info=True,
        )
        return False
    finally:
        if not args.download_only:
            cleanup_downloads(args.download_dir)


def _build_install_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arch", choices=["x86_64", "aarch64"])
    parser.add_argument("--py-tag")
    parser.add_argument("--download-dir", type=Path, default=Path.cwd() / "akg_wheels")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--insecure", action="store_true", help="Disable HTTPS certificate verification")
    return parser


# ---------------------------------------------------------------------------
# AKG / MLIR smoke tests
# ---------------------------------------------------------------------------

AKG_ENV = {
    "TORCHINDUCTOR_NPU_BACKEND": "mlir",
    "TORCHINDUCTOR_USE_AKG": "1",
}

HAS_AKG_STACK = False
EXPECTED_SCHEDULING = ""
EXPECTED_COMPILE_API = ""
EXPECTED_SCHEDULING_CLS = None
TestAkgTorchMlir = None


def _set_akg_env() -> dict[str, str | None]:
    original = {name: os.environ.get(name) for name in AKG_ENV}
    os.environ.update(AKG_ENV)
    return original


def _restore_env(original: dict[str, str | None]) -> None:
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@contextmanager
def _temporary_akg_env():
    original = _set_akg_env()
    try:
        yield
    finally:
        _restore_env(original)


def _bootstrap_tests() -> None:
    """Probe akg stack and import torch_npu._inductor after optional wheel install."""
    global HAS_AKG_STACK, EXPECTED_SCHEDULING, EXPECTED_COMPILE_API
    global EXPECTED_SCHEDULING_CLS, TestAkgTorchMlir

    try:
        importlib.import_module("akg")
        importlib.import_module("torch_mlir")
        HAS_AKG_STACK = True
    except ImportError:
        HAS_AKG_STACK = False

    EXPECTED_SCHEDULING = "AkgScheduling" if HAS_AKG_STACK else "NpuMlirScheduling"
    EXPECTED_COMPILE_API = "akg_auto_fallback" if HAS_AKG_STACK else "mlir_auto_fallback"

    # Backend selection is resolved when torch_npu._inductor is first imported.
    with _temporary_akg_env():
        importlib.import_module("torch_npu._inductor")
        from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.akg import AkgScheduling
        from torch_npu._inductor.ascend_npu_ir.ascend_npu_ir.npu.codegen.mlir import NpuMlirScheduling

        # isort: off
        import test_torch_mlir as torch_mlir_tests
        # isort: on

    EXPECTED_SCHEDULING_CLS = AkgScheduling if HAS_AKG_STACK else NpuMlirScheduling

    class _TestAkgTorchMlir(torch_mlir_tests.TestAdd):
        """Verify AKG enablement, or MLIR fallback when akg packages are unavailable."""

        @staticmethod
        def _fused_op_calc(first_element, second_element):
            return (first_element + second_element) * second_element

        def setUp(self):
            self._original_akg_env = _set_akg_env()
            try:
                super().setUp()
            except Exception:
                _restore_env(self._original_akg_env)
                raise

        def tearDown(self):
            try:
                super().tearDown()
            finally:
                _restore_env(self._original_akg_env)

        def test_backend_scheduling_registered(self):
            scheduling = getattr(device_codegens.get("npu"), "scheduling", None)
            self.assertIs(
                scheduling,
                EXPECTED_SCHEDULING_CLS,
                f"expected {EXPECTED_SCHEDULING} when HAS_AKG_STACK={HAS_AKG_STACK}",
            )

        def test_fused_kernel_compile_path(self):
            """Fused multi-op subgraphs should use the selected backend compile API."""
            shape = torch_mlir_tests.TestUtils._pointwise_demo_shapes[0]
            dtype = "float32"
            x = self._generate_tensor(shape, dtype)
            y = self._generate_tensor(shape, dtype)
            expected = self._fused_op_calc(x, y)

            compiled = torch.compile(self._fused_op_calc)
            result, codes = run_and_get_code(compiled, x, y)

            self.assertEqual(expected, result)
            self.assertIn(
                EXPECTED_COMPILE_API,
                codes[0],
                f"expected {EXPECTED_COMPILE_API} when HAS_AKG_STACK={HAS_AKG_STACK}",
            )

    TestAkgTorchMlir = _TestAkgTorchMlir
    globals()["TestAkgTorchMlir"] = TestAkgTorchMlir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    install_args = _build_install_parser().parse_args(SAVED_INSTALL_ARGV)

    try:
        installed = install_akg_wheels(install_args)
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return 130
    except Exception:
        logger.exception("Download failed")
        return 1

    if install_args.download_only:
        return 0 if installed else 1

    _bootstrap_tests()
    run_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
else:
    _bootstrap_tests()
