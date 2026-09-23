"""
Custom operator loading module for FXRT.
"""

import os
import subprocess
import shutil
import hashlib
import stat
import time
import platform
import ctypes
import sysconfig
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import fxrt
from fxrt._logging import get_logger

logger = get_logger(__name__)

class FileLock:
    """Context manager for file locking with timeout."""
    def __init__(self, build_dir: Path, wait_seconds=0.1):
        self.lock_file = build_dir / "build.lock"
        self.wait_seconds = wait_seconds
        self.fd = None

    def try_lock(self):
        """Attempt to acquire the lock file."""
        try:
            mode = stat.S_IRUSR | stat.S_IWUSR
            self.fd = os.open(self.lock_file, os.O_EXCL | os.O_CREAT, mode)
            return True
        except FileExistsError:
            return False

    def wait(self):
        """Wait until the lock file is released."""
        while self.lock_file.exists():
            time.sleep(self.wait_seconds)

    def release_lock(self):
        """Release the lock file."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.lock_file.unlink(missing_ok=True)

class CustomOpLoader:
    """Custom operator loader that handles compilation and loading of C++ sources."""

    def __init__(self):
        self._loaded_libraries = {}  # Cache for loaded libraries
        self._compiled_libraries = {}  # Cache for compiled libraries
        self.fxrt_path = os.path.dirname(os.path.realpath(fxrt.__file__))
        self.fxrt_lib_path = os.path.join(self.fxrt_path, 'lib')
        self.fxrt_include_path = os.path.join(self.fxrt_path, 'include')
        self.debug_mode = False  # Set to True to enable debug symbols

    def _get_cache_dir(self) -> Path:
        """
        Get the cache directory path and ensure it exists.
        """
        cache_dir = Path.cwd() / ".fxrt_custom_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_build_dir(self, name: str, build_directory: Optional[str]) -> Path:
        """
        Get the build directory path.
        """
        if build_directory:
            build_dir = Path(build_directory) / f"{name}"
        else:
            build_dir = self._get_cache_dir() / f"{name}"
        build_dir.mkdir(parents=True, exist_ok=True)
        return build_dir

    def _get_lib_name(self, base_name: str, compile_hash: str) -> str:
        """
        Generate library name with embedded hash.
        """
        return f"{base_name}_{compile_hash[:8]}"

    def _calculate_compile_hash(self, name: str, sources: List[str], extra_cflags: List[str],
                                extra_ldflags: List[str], extra_include_paths: List[str],
                                build_directory: Optional[str], backend: str) -> str:
        """
        Calculate hash for compilation parameters to detect changes.
        """
        # Create a hash object
        hash_obj = hashlib.sha256()
        hash_obj.update(name.encode('utf-8'))

        # Add sources with their modification times
        for source in sorted(sources):
            source_path = Path(source).absolute()
            if source_path.exists():
                hash_obj.update(str(source_path).encode('utf-8'))
                hash_obj.update(str(source_path.stat().st_mtime).encode('utf-8'))
                hash_obj.update(str(source_path.stat().st_size).encode('utf-8'))
            else:
                hash_obj.update(str(source_path).encode('utf-8'))

        # Add compilation parameters
        hash_obj.update(str(sorted(extra_cflags)).encode('utf-8'))
        hash_obj.update(str(sorted(extra_ldflags)).encode('utf-8'))
        hash_obj.update(str(sorted(extra_include_paths)).encode('utf-8'))
        hash_obj.update(str(build_directory).encode('utf-8'))
        hash_obj.update(backend.encode('utf-8'))

        return hash_obj.hexdigest()

    def _get_platform_lib_extension(self) -> str:
        """Get the dynamic library extension for the current platform."""
        system = platform.system()
        if system == "Windows":
            return ".dll"
        if system == "Darwin":  # macOS
            return ".dylib"
        return ".so"

    def _get_ascend_environment(self) -> Dict[str, Any]:
        """Get Ascend environment configuration."""
        ascend_path = os.environ.get('ASCEND_CUSTOM_PATH', '/usr/local/Ascend/ascend-toolkit')

        return {
            'ASCEND_PATH': ascend_path,
            'INCLUDE_DIRS': [
                f"{ascend_path}/latest/include/",
                f"{ascend_path}/latest/lib64/",
                f"{ascend_path}/latest/aarch64-linux/include/experiment"
            ],
            'LIB_DIRS': [
                f"{ascend_path}/latest/lib64/"
            ],
            'LIBS': ["ascendcl", "fxrt", "ops_ascend_aclnn_common"]
        }

    def _get_cpu_environment(self) -> Dict[str, Any]:
        """Get CPU environment configuration."""
        return {
            'INCLUDE_DIRS': [],
            'LIB_DIRS': [],
            'LIBS': ["fxrt"]
        }

    def _get_backend_environment(self, backend: str = "Ascend") -> Dict[str, Any]:
        """Get environment configuration based on backend."""
        if backend.lower() == "ascend":
            return self._get_ascend_environment()
        if backend.lower() == "cpu":
            return self._get_cpu_environment()

        logger.warning("Unknown backend '%s', using Ascend as default", backend)
        return self._get_ascend_environment()

    def _check_validate_sources(self, sources: List[str]) -> bool:
        """Validate source files exist and are readable."""
        for source in sources:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source}")

            if not source_path.is_file():
                raise ValueError(f"Source path is not a file: {source}")

            if not os.access(source_path, os.R_OK):
                raise RuntimeError(f"Source file is not readable: {source}")

    def _check_ninja_available(self) -> bool:
        """Check if required build tools are available."""
        try:
            subprocess.check_output(['ninja', '--version'])
        except Exception as e:
            raise RuntimeError("Ninja is required to load c++ extensions") from e

    def _create_ninja_build_file(self, sources: List[str], build_dir: Path, extra_cflags: List[str],
                                 extra_ldflags: List[str], extra_include_paths: List[str],
                                 lib_name: str, backend: str = "Ascend") -> str:
        """Generate build.ninja content for the custom operator."""
        # Copy source files to build directory
        dst_sources = []
        for source in sources:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source}")

            dst_source = build_dir / source_path.name
            if source_path != dst_source:
                shutil.copy(source_path, dst_source)
            dst_sources.append(str(dst_source))

        backend_env = self._get_backend_environment(backend)

        # Combine include paths
        all_include_paths = [self.fxrt_include_path, os.path.join(self.fxrt_include_path, 'src')]
        thirdparty_path = os.path.join(self.fxrt_include_path, 'third_party')
        if os.path.isdir(thirdparty_path):
            for subdir in os.listdir(thirdparty_path):
                all_include_paths.append(os.path.join(thirdparty_path, subdir))
        all_include_paths += extra_include_paths
        all_include_paths += backend_env['INCLUDE_DIRS']
        include_flags = ' '.join([f'-I{path}' for path in all_include_paths])

        # Combine library directories
        all_lib_dirs = [self.fxrt_lib_path] + backend_env['LIB_DIRS']
        lib_dir_flags = ' '.join([f'-L{path}' for path in all_lib_dirs])

        # Combine libraries
        all_libs = backend_env['LIBS']
        all_libs = [' '.join([f'-l{lib}' for lib in backend_env['LIBS']])]
        all_libs += [' '.join(extra_ldflags)]
        all_libs += ["-shared -Wl,-z,relro,-z,now,-z,noexecstack"]
        all_libs += ["-Wl,--disable-new-dtags"]
        all_libs += [f"-Wl,--rpath,'{self.fxrt_lib_path}' -s"]
        lib_flags = ' '.join(all_libs)

        # Combine compiler flags
        python_include_path = sysconfig.get_path('include', scheme='posix_prefix')
        python_includes = [python_include_path] if python_include_path else []
        all_cflags = [f'-isystem {include}' for include in python_includes]
        all_cflags += ['-std=c++17', '-fstack-protector-all', '-fPIC', '-pie', '-shared'] + extra_cflags
        try:
            # pylint: disable=import-outside-toplevel
            import torch
            abi_flag = int(getattr(getattr(torch, '_C', None), '_GLIBCXX_USE_CXX11_ABI', 0))
            all_cflags += [f'-D_GLIBCXX_USE_CXX11_ABI={abi_flag}']
        except (ImportError, TypeError):
            all_cflags += ['-D_GLIBCXX_USE_CXX11_ABI=0']
        if self.debug_mode:
            all_cflags += ['-g']
        else:
            all_cflags += ['-O2']
        if backend.lower() == "ascend":
            all_cflags += ['-DASCEND_CUSTOM_OP']
        cflags_str = ' '.join(all_cflags)

        # Get platform-specific library extension
        lib_ext = self._get_platform_lib_extension()

        # Object files
        obj_files = []
        for source in dst_sources:
            obj_name = Path(source).stem + '.o'
            obj_files.append(obj_name)
        obj_files_str = ' '.join(obj_files)

        # Create ninja build file content with proper formatting
        ninja_parts = []

        # Add header and variables
        ninja_parts.append("# Ninja build file for FXRT custom operator")
        ninja_parts.append("# Generated automatically by fxrt.ops.load()")
        ninja_parts.append("")
        ninja_parts.append("# Variables")
        ninja_parts.append("cxx = g++")
        ninja_parts.append(f"cflags = {cflags_str} {include_flags}")
        ninja_parts.append(f"ldflags = {lib_dir_flags} {lib_flags}")
        ninja_parts.append(f"lib_name = {lib_name}")
        ninja_parts.append(f"lib_ext = {lib_ext}")
        ninja_parts.append("")

        # Add rules
        ninja_parts.append("# Rules")
        ninja_parts.append("rule cxx")
        ninja_parts.append("  command = $cxx $cflags -c $in -o $out")
        ninja_parts.append("  description = CXX $out")
        ninja_parts.append("")
        ninja_parts.append("rule link")
        ninja_parts.append("  command = $cxx $in -o $out $ldflags")
        ninja_parts.append("  description = LINK $out")
        ninja_parts.append("")

        # Add build object files section
        ninja_parts.append("# Build object files")
        for source in dst_sources:
            obj_name = Path(source).stem + '.o'
            ninja_parts.append(f"build {obj_name}: cxx {source}")
        ninja_parts.append("")

        # Add library linking rule
        ninja_parts.append("# Build shared library")
        ninja_parts.append(f"build $lib_name$lib_ext: link {obj_files_str}")
        ninja_parts.append("")

        # Join all parts with newlines
        ninja_content = '\n'.join(ninja_parts)

        ninja_file = build_dir / "build.ninja"
        ninja_file.write_text(ninja_content)
        return ninja_file

    def _run_ninja_build(self, build_dir: Path, lib_name: str) -> None:
        """Run Ninja build in the specified directory."""
        # Build with ninja
        build_cmd = ["ninja", "-v"]
        env = os.environ.copy()
        lib_ext = self._get_platform_lib_extension()
        lib_file = build_dir / f"{lib_name}{lib_ext}"
        log_file = build_dir / ".build.log"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                subprocess.run(build_cmd, cwd=build_dir, env=env, stdout=f, stderr=f, check=True)
        except subprocess.CalledProcessError as e:
            with open(log_file, "r", encoding="utf-8") as f:
                build_log = f.read()
            msg = f"Error building extension {lib_name}: {build_log}"
            if lib_file.exists():
                lib_file.unlink()
            raise RuntimeError(msg) from e

        if not lib_file.exists():
            raise FileNotFoundError(f"Compiled library not found: {lib_file}")
        logger.info("Successfully compiled library: %s", lib_file)
        return str(lib_file)

    def _compile_sources(self, name: str, sources: Union[str, List[str]],
                         extra_cflags: Optional[List[str]] = None,
                         extra_ldflags: Optional[List[str]] = None,
                         extra_include_paths: Optional[List[str]] = None,
                         backend: str = "Ascend",
                         build_directory: Optional[str] = None) -> Optional[str]:
        """Compile C++ sources into a dynamic library."""
        if isinstance(sources, str):
            sources = [sources]

        if extra_cflags is None:
            extra_cflags = []
        if extra_ldflags is None:
            extra_ldflags = []
        if extra_include_paths is None:
            extra_include_paths = []

        self._check_validate_sources(sources)
        self._check_ninja_available()

        build_dir = self._get_build_dir(name, build_directory)
        compile_hash = self._calculate_compile_hash(
            name, sources, extra_cflags, extra_ldflags,
            extra_include_paths, build_directory, backend
        )
        # Check cache
        if compile_hash in self._compiled_libraries:
            lib_path = self._compiled_libraries[compile_hash]
            logger.info("Using cached library: %s", lib_path)
            return lib_path

        lib_name = self._get_lib_name(name, compile_hash)
        lib_ext = self._get_platform_lib_extension()
        lib_file = build_dir / f"{lib_name}{lib_ext}"
        file_lock = FileLock(build_dir)
        if file_lock.try_lock():
            try:
                self._create_ninja_build_file(
                    sources, build_dir, extra_cflags, extra_ldflags,
                    extra_include_paths, lib_name, backend
                )
                lib_file = self._run_ninja_build(build_dir, lib_name)
                self._compiled_libraries[compile_hash] = lib_file
            finally:
                file_lock.release_lock()
        else:
            file_lock.wait()
        return lib_file

    def _load_library(self, lib_path: str) -> bool:
        """Load the compiled dynamic library."""
        try:
            lib = ctypes.CDLL(lib_path)
            logger.info("Successfully loaded library: %s", lib_path)
            self._loaded_libraries[lib_path] = lib
            return True
        except OSError as e:
            raise RuntimeError(f"Failed to load library {lib_path}: {e}") from e

    def load_library(self, lib_path: str) -> bool:
        """
        Load a pre-compiled dynamic library containing custom operators.

        Args:
            lib_path: Path to the dynamic library file (.so, .dll, .dylib)

        Returns:
            bool: True if successful, False otherwise

        Example:
            # Load a pre-compiled library
            success = fxrt.ops.load_library("/path/to/my_custom_ops.so")
        """
        try:
            # Validate library path
            lib_path_obj = Path(lib_path)
            if not lib_path_obj.exists():
                logger.error("Library file not found: %s", lib_path)
                return False

            if not lib_path_obj.is_file():
                logger.error("Library path is not a file: %s", lib_path)
                return False

            # Check if library is already loaded
            if lib_path in self._loaded_libraries:
                logger.info("Library already loaded: %s", lib_path)
                return True

            # Load the library
            return self._load_library(lib_path)

        except OSError:
            logger.exception("Failed to load library %s", lib_path)
            return False

    # pylint: disable=redefined-builtin
    def compile(self, name: str, sources: Union[str, List[str]],
                extra_cflags: Optional[List[str]] = None,
                extra_ldflags: Optional[List[str]] = None,
                extra_include_paths: Optional[List[str]] = None,
                backend: str = "Ascend",
                debug_mode: bool = False,
                build_directory: Optional[str] = None) -> Optional[str]:
        """
        Compile C++ source files into a dynamic library without loading it.

        Args:
            name: Custom name for the compiled library
            sources: Path to C++ source file(s) containing custom operators
            extra_cflags: Additional compiler flags (e.g., ["-O2", "-std=c++17"])
            extra_ldflags: Additional linker flags (e.g., ["-lmy_lib"])
            extra_include_paths: Additional include paths (e.g., ["/path/to/headers"])
            backend: Target backend for compilation ("Ascend" or "CPU")
            debug_mode: Whether to compile with debug symbols (enables -g flag)
            build_directory: Custom directory for build artifacts. If None, uses cache_dir as build directory (optional)

        Returns:
            Optional[str]: Path to the compiled dynamic library if successful, None otherwise

        Example:
            lib_path = fxrt.ops.compile(
                name="my_custom_op",
                sources="my_custom_op.cpp",
                extra_cflags=["-O2", "-std=c++17"],
                extra_ldflags=["-lascendcl"],
                extra_include_paths=["/path/to/includes"])
        """
        try:
            # Set debug mode
            self.debug_mode = debug_mode

            # Compile sources and return the library path
            return self._compile_sources(name, sources, extra_cflags,
                                         extra_ldflags, extra_include_paths,
                                         backend, build_directory)

        except (OSError, subprocess.SubprocessError):
            logger.exception("Failed to compile custom operators")
            return None

    def load(self, name: str, sources: Union[str, List[str]],
             extra_cflags: Optional[List[str]] = None,
             extra_ldflags: Optional[List[str]] = None,
             extra_include_paths: Optional[List[str]] = None,
             backend: str = "Ascend",
             debug_mode: bool = False,
             build_directory: Optional[str] = None) -> bool:
        """
        Load custom operators from C++ source files.

        Args:
            name: Custom name for the compiled library
            sources: Path to C++ source file(s) containing custom operators
            extra_cflags: Additional compiler flags (e.g., ["-O2", "-std=c++17"])
            extra_ldflags: Additional linker flags (e.g., ["-lmy_lib"])
            extra_include_paths: Additional include paths (e.g., ["/path/to/headers"])
            backend: Target backend for compilation ("Ascend" or "CPU")
            debug_mode: Whether to compile with debug symbols (enables -g flag)
            build_directory: Custom directory for build artifacts. If None, uses cache_dir as build directory (optional)

        Returns:
            bool: True if successful, False otherwise

        Example:
            success = fxrt.ops.load(
                name="my_custom_lib",
                sources=["my_custom_op.cpp"],
                build_directory="/tmp/my_build",
                extra_cflags=["-O2", "-std=c++17"],
                extra_ldflags=["-lascendcl"],
                extra_include_paths=["/path/to/includes"])
        """
        try:
            self.debug_mode = debug_mode
            lib_path = self._compile_sources(name, sources, extra_cflags,
                                             extra_ldflags, extra_include_paths,
                                             backend, build_directory)
            if lib_path is None:
                return False
            return self.load_library(lib_path)
        except (OSError, subprocess.SubprocessError):
            logger.exception("Failed to load custom operators")
            return False

# Global loader instance
_loader = CustomOpLoader()

# pylint: disable=redefined-builtin
def compile(name: str, sources: Union[str, List[str]], extra_cflags: Optional[List[str]] = None,
            extra_ldflags: Optional[List[str]] = None,
            extra_include_paths: Optional[List[str]] = None,
            backend: str = "Ascend",
            debug_mode: bool = False,
            build_directory: Optional[str] = None) -> Optional[str]:
    """
    Compile C++ source files into a dynamic library without loading it.

    This function compiles C++ source files containing custom operators
    and returns the path to the compiled dynamic library. It handles all the complexity of Ninja
    build configuration and compilation, but does not load the library into the runtime.

    Args:
        name: Custom name for the compiled library
        sources: Path to C++ source file(s) containing custom operators.
                 Can be a single string or list of strings.
        extra_cflags: Additional compiler flags (e.g., ["-O2", "-std=c++17"])
        extra_ldflags: Additional linker flags (e.g., ["-lmy_lib"])
        extra_include_paths: Additional include paths (e.g., ["/path/to/headers"])
        backend: Target backend for compilation ("Ascend" or "CPU")
        debug_mode: Whether to compile with debug symbols (enables -g flag)
        build_directory: Custom directory for build artifacts. If None, uses cache_dir as build directory (optional)

    Returns:
        Optional[str]: Path to the compiled dynamic library if successful, None otherwise

    Example:
        import fxrt

        # Compile a single source file for Ascend backend
        lib_path = fxrt.ops.compile("my_custom_op.cpp", backend="Ascend")
        # Compile multiple source files
        lib_path = fxrt.ops.compile(["op1.cpp", "op2.cpp"])
        # Compile with custom name and build directory
        lib_path = fxrt.ops.compile("my_custom_op.cpp",
                          name="my_custom_lib",
                          build_directory="/tmp/my_build")
    """
    return _loader.compile(name, sources, extra_cflags,
                           extra_ldflags, extra_include_paths, backend,
                           debug_mode, build_directory)

def load(name: str, sources: Union[str, List[str]], extra_cflags: Optional[List[str]] = None,
         extra_ldflags: Optional[List[str]] = None,
         extra_include_paths: Optional[List[str]] = None,
         backend: str = "Ascend",
         debug_mode: bool = False,
         build_directory: Optional[str] = None) -> bool:
    """
    Load custom operators from C++ source files.

    Args:
        name: Custom name for the compiled library
        sources: Path to C++ source file(s) containing custom operators
        extra_cflags: Additional compiler flags (e.g., ["-O2", "-std=c++17"])
        extra_ldflags: Additional linker flags (e.g., ["-lmy_lib"])
        extra_include_paths: Additional include paths (e.g., ["/path/to/headers"])
        backend: Target backend for compilation ("Ascend" or "CPU")
        debug_mode: Whether to compile with debug symbols (enables -g flag)
        build_directory: Custom directory for build artifacts. If None, uses cache_dir as build directory (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    return _loader.load(name, sources, extra_cflags,
                        extra_ldflags, extra_include_paths, backend,
                        debug_mode, build_directory)


def load_library(lib_path: str) -> bool:
    """
    Load a pre-compiled dynamic library containing custom operators.

    Args:
        lib_path: Path to the dynamic library file (.so, .dll, .dylib)

    Returns:
        bool: True if successful, False otherwise
    """
    return _loader.load_library(lib_path)
