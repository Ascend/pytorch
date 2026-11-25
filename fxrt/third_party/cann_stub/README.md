# CANN link-only stubs

The converter wheel is linked against the C and C++ stub definitions in this
directory. They use the pinned GE and CANN headers prepared under
`third_party`, and are built using the converter's C++17 and libstdc++ ABI=0
settings. The compiler produces the C++ symbols, including the required virtual
tables and RTTI. No installed CANN libraries or GE source tree are needed to
build the wheel.

Every stub function traps if called. The private implementation types completed
in a few files exist only to compile constructor/destructor stubs; their layouts
are not runtime implementations. These DSOs must never be installed or used to
run a model. CMake does not install them. `check_link.py` checks converter
imports and rejects stub markers or stub runtime search paths in the artifacts.

The wheel retains CANN SONAME dependencies; running it requires compatible real
CANN libraries. The converter does not directly link protobuf; GE may still
use it internally. C++ symbol names alone do not guarantee C++ ABI compatibility
between the pinned headers, compiler settings and runtime libraries.

When adding an imported API, add its matching declaration and trapping
definition to the source for the library that provides it. Compile against the
same pinned headers as the converter. For a new library, add its target and
SONAME to `CMakeLists.txt`. Rebuild and inspect the missing-symbol diagnostics
from `check_link.py`; also check against the intended real CANN runtime, since
the link-only check cannot prove runtime ABI or provider compatibility.
