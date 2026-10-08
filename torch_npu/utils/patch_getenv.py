import os
import logging

_seen = set()
_installed = False

_orig_getenv = os.getenv
loggerEnv = logging.getLogger("torch_npu.env")


def _log_once(key: str, val):
    if key in _seen:
        return
    _seen.add(key)
    loggerEnv.info("get env %s = %s", key, val)


def _patched_getenv(key, default=None):
    # Read os.environ[key] instead of the saved os.getenv: os.getenv resolves
    # environ.get at call time and would re-enter this patch; __getitem__ is
    # not patched, so this is exactly one real lookup per call.
    try:
        val = os.environ[key]
    except KeyError:
        return default
    if val != "":
        _log_once(key, val)
    return val


def _should_install() -> bool:
    # Imported before _add_logging_module() configures the torch_npu.env
    # logger, so gate on the raw env vars (same as the ACL log gating);
    # without them the INFO log is never observable.
    if os.environ.get("TORCH_NPU_LOGS") is not None:
        return True
    if os.environ.get("TORCH_LOGS") is not None:
        return True
    return loggerEnv.isEnabledFor(logging.INFO)


def _install():
    global _installed
    if _installed:
        return
    os.getenv = _patched_getenv
    os.environ.get = _patched_getenv
    _installed = True


# patch on import, only when the log is observable
if _should_install():
    _install()
