# securec headers for FXRT

Source: https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/libboundscheck/libboundscheck-v1.1.16.tar.gz

Version: libboundscheck v1.1.16

Archive SHA256: aee8368ef04a42a499edd5bfebce529e7f32dd138bfed383d316e48af4e45d2c

This directory contains only the unmodified upstream headers needed by the
converter and its build-time link stubs. No implementation or upstream build
is included. The corresponding libraries are supplied by CANN at runtime.

| Local file | Upstream file |
| --- | --- |
| `include/securec.h` | `include/securec.h` |
| `include/securectype.h` | `include/securectype.h` |

When updating these headers, use one pinned upstream version, preserve its
license and header notices, and update this source record together with the
files. Keep headers from other repositories in their own directories.
