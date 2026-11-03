# PyTorch Contribution Guide

Thank you for considering contributing to PyTorch! We welcome contributions of all forms, including bug fixes, feature enhancements, documentation improvements, and so on. Whether you are an experienced developer or a first-time open source participant, your help is invaluable.

## Project Introduction

PyTorch is a deep learning framework distribution based on the Ascend NPU. It is deeply optimized and adapted for Huawei Ascend NPUs. This project provides API compatibility with official PyTorch and fully utilizes the computing power of Ascend chips.

### Project Architecture

```text
pytorch
├── docs/                           # Project documents
├── ci/                             # CI build scripts
├── tools/                          # Development tools
├── cmake/                          # CMake configuration
├── torch_npu/                      # Core NPU adaptation module
│   ├── csrc/                       # C++ backend implementation
│   ├── distributed/                # Distributed Python interface
│   ├── _inductor/                  # Inductor backend adaptation
│   ├── dynamo/                     # Dynamo compiler adaptation
│   ├── npu/                        # NPU Python interface
│   ├── profiler/                   # Profiling Python interface
│   ├── _afd/                       # AFD Python interface
│   ├── _logging/                   # Logging module Python interface
│   ├── asd/                        # Asynchronous detection tool
│   ├── contrib/                    # Contributed extension modules
│   ├── onnx/                       # ONNX adaptation
│   └── optim/                      # Optimizer adaptation
├── third_party/                    # Third-party dependencies
├── torchnpugen/                    # Code generation tool
├── examples/                       # Sample code
└── test/                           # Test cases
```

### Core Modules

| Module                     | Description                                                                                                                              |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `torch_npu/csrc/core/npu`  | NPU core components: event management (NPUEvent), stream management (NPUStream), graph execution (NPUGraph), device guard (NPUGuard), and memory management |
| `torch_npu/csrc/aten`      | NPU backend of ATen operators: operator registration, dispatch, and implementation adaptation                                              |
| `torch_npu/csrc/framework` | Operator command framework: OpCommand, kernel dispatch, and operator builders                                                              |
| `torch_npu/npu/aclnn`      | Python interfaces of ACLNN operators: wrappers of the AscendCL NPU operator library                                                        |
| `torch_npu/npu/amp`        | Automatic mixed precision: GradScaler and FP16/BF16 support                                                                                |
| `torchnpugen`              | Code generation tool: automatic differentiation code generation and code templates                                                        |
| `examples`                 | Sample code: distributed communication, model inference, and ResNet samples                                                                |
| `third_party/op-plugin`    | Operator plugin: custom operator implementations and PyTorch operator overrides                                                            |
| `test/npu`                 | NPU function tests: device management, memory allocation, and operator tests                                                               |

## How to Contribute

We look forward to your participation! Every contribution drives the progress of PyTorch:

- **Report issues**: Report bugs or submit feature suggestions to help us identify and resolve problems
- **Contribute code**: Submit code fixes or new features to participate in project development directly
- **Improve documentation**: Improve documents or supplement missing content to make the project more readable
- **Review code**: Review pull requests to help improve code quality
- **Spread the word**: Share the project on blogs or social media, and give the repository a star ⭐

## Contribution Scenarios

This project welcomes contributions of all forms. We look forward to your participation!

### 1. Requirements and Feature Suggestions

If you have suggestions for new features or ideas for performance optimization, submit an issue to discuss them with the community.

**Issue type**: requirement or feature suggestion

**Content to include**:

- **Feature background**: the problem that the feature solves and the value that it brings to users
- **Feature description**: describe the suggested feature in detail
- **Design scheme**: the technical approach, design of key modules, and relationships between upstream and downstream components
- **Expected benefits**: feature objectives, performance metrics, and accuracy

### 2. Bug Reports and Fixes

If you find a bug or a document problem, we sincerely welcome your feedback and fix suggestions.

**Bug report format**:

- **Environment information**: PyTorch version, OS, Python version, CANN version, and so on
- **Problem description**: add labels so that the issue is highlighted on the issue dashboard
- **Reproduction steps**: describe how to reproduce the problem in as much detail as possible
- **Expected behavior**: describe the behavior that you expect
- **Special notes for reviewers**: any special circumstances, if applicable

**Fix process**:

1. Find the corresponding bug description in the issue
2. Comment `/assign` to claim the task
3. Create a branch for the fix
4. Submit a pull request

### 3. Community Support

If you can resolve issues raised by others, share your solutions in the issues.

## Contribution Process

### Contributor License Agreement

Sign the CLA before you submit code to the PyTorch community for the first time.

For individual contributors, refer to the [ICLA online document](https://www.mindspore.cn/icla) for details.

### Development and Testing

1. **Fork the repository**: On the GitCode platform, click **Fork** in the upper right corner of the repository to clone it to your personal account

2. **Clone the repository to your local machine**:

   ```bash
   git clone https://gitcode.com/<your-username>/pytorch.git
   cd pytorch
   ```

3. **Create a development branch**:

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

4. **Develop code**: Follow the **[coding standards](#coding-standards)**

5. **Test the code**: Run tests to ensure that the code functions properly

6. **CI gate checks**: Run CI checks to ensure that the code passes compilation, static checks, and UT tests

7. **Submit a pull request**: Submit the PR and wait for code review

8. **Community review**: If the change involves patches, header file macros, API interfaces, and so on, submit it for community review

### Code Merge Review Requirements

The following types of changes require community review:

- **Patch replacement**: patch replacement of native PyTorch interfaces
- **Header file macro updates**: adding or modifying macro definitions
- **API interface changes**: adding, modifying, or deleting public APIs
- **Core component changes**: modifications to core modules such as memory management and device management

## Coding Standards

Follow these styles to make PyTorch easy to develop, review, and maintain.

### Coding Guidelines

- **Python**: We recommend the [PEP 8 coding style](https://pep8.org/)
- **C++**: We recommend the [Google C++ Style Guide](http://google.github.io/styleguide/cppguide.html)

For code checks, refer to [Local Static Checks](#local-static-checks).

### Unit Test Guidelines

- **Python**: We recommend [pytest](http://pytest.org/en/latest/)
- **C++**: We recommend the [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md)

The design intent of a test case should be reflected in its descriptive name.

### Refactoring Guidelines

We encourage developers to refactor code to eliminate code smells. All code must comply with the coding style and test style requirements.

## Practical Guide

### Environment Setup and Compilation

**Build**:

```bash
# Install dependencies and compile
bash ci/build.sh --python=3.10

# Build for a specified PyTorch version (supports 2.10.0 / 2.11.0 / 2.12.0)
# The corresponding PyTorch version must be installed in the environment
bash ci/build.sh --python=3.10 --torch=2.10.0

# Alternatively, use CMake to compile manually
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Compilation Acceleration Tips

#### Building with Ninja

By default, CMake uses the Makefile generator. Installing the Ninja build system significantly accelerates compilation.

The `setup.py` file of this project automatically detects whether Ninja is installed in the system. If the environment variable `CMAKE_GENERATOR` is set to `ninja` or the `ninja` command is available in `PATH`, Ninja is automatically used as the build system.

```bash
pip install ninja
```

After Ninja is installed, compilation takes effect automatically without additional configuration. If you have compiled the project before, perform a cleanup once after installing Ninja:

```bash
python setup.py clean
```

#### Using the Mold Linker

In a development loop where you frequently modify a single file and recompile, linking time dominates. The system linker (GNU `ld`) shipped with most Linux distributions is slow. Using a faster linker significantly improves the build experience.

The `CMakeLists.txt` file of this project has built-in linker auto-detection logic. It detects the mold linker first and automatically enables it (`-fuse-ld=mold`) if the linker exists.

```bash
sudo apt install mold
# Alternatively, install from source: https://github.com/rui314/mold
```

Recompile after installation for the setting to take effect automatically. To confirm that the linker is enabled correctly, check whether the link options in the compilation output contain `-fuse-ld=mold`.

#### Using CCache

Even though dependency tracking is based on file modification times, files are recompiled in many scenarios. Using ccache effectively avoids redundant compilation and saves substantial time.

The `CMakeLists.txt` file of this project has built-in ccache auto-detection logic. ccache is enabled automatically after installation. We recommend that you adjust the ccache configuration (such as the cache directory, cache size, and compression) based on your environment for the best results:

```bash
sudo apt install ccache
# Alternatively
sudo yum install ccache
```

To verify that ccache takes effect, perform two full compilations in sequence. The second compilation should be significantly faster than the first. If ccache does not take effect, check whether the `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` variables in `build/CMakeCache.txt` contain ccache:

```cmake
//C compiler launcher
CMAKE_C_COMPILER_LAUNCHER:PATH=/usr/bin/ccache

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:PATH=/usr/bin/ccache
```

#### Compiling Only Required Targets

If you only need to rebuild `torch_npu.so`, specify the target directly in the build directory to avoid a full build:

```bash
cd build && ninja torch_npu
```

If Ninja is not installed, replace `ninja` with `make`.

### Local Static Checks

The project uses [lintrunner](https://github.com/suo/lintrunner) for static checks. You can run locally the same check items as CI, including Python code style (Flake8, Ruff, PYFMT), C++ formatting (ClangFormat, ClangTidy), spelling checks (Codespell), and so on.

#### Installing Dependencies

```bash
# Install lintrunner and uv (required by some linters)
pip install lintrunner
pip install uv
```

#### Initialization (Execute Once During First Use or Update)

```bash
# Download the external binary tools required by lintrunner (clang-format, clang-tidy, and so on)
lintrunner init
```

#### Running Static Checks

```bash
# Check the changed files in the current workspace and the HEAD commit (workspace + HEAD)
lintrunner

# Run only specified check items
lintrunner --take FLAKE8,RUFF,PYFMT,SPACES,TABS,NEWLINE

# Automatically fix issues that can be auto-fixed (formatter-type linters; for example, ignore PYREFLY)
lintrunner --skip PYREFLY -a

# Check only the changed files in the current workspace
git diff --name-only HEAD | xargs lintrunner
```

> **Tip**: The `--take` parameter specifies only some check items to run. Common items are as follows:
>
> | Code          | Description                                                                    |
> |---------------|--------------------------------------------------------------------------------|
> | `FLAKE8`      | Python syntax and style checks                                                 |
> | `RUFF`        | Fast Python linting and import sorting                                         |
> | `PYFMT`       | Python code formatting (usort + ruff-format)                                   |
> | `CLANGFORMAT` | C++ code formatting                                                            |
> | `CLANGTIDY`   | C++ static analysis                                                            |
> | `SPACES`      | Trailing space checks                                                          |
> | `TABS`        | Tab character checks                                                           |
> | `NEWLINE`     | End-of-file newline checks                                                     |
> | `CODESPELL`   | Spelling checks. If a result is a false positive, add the word to `tools/linter/dictionary.txt` in lexicographic order and check again |

For more commands, refer to the [lintrunner wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).

### PR Merge Requirements

**Merge checklist** (For detailed requirements, refer to the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md)):

- [ ] The code compiles successfully
- [ ] Static checks pass (CppLint, CppCheck, and so on)
- [ ] UT test cases pass
- [ ] The code style complies with standards (PEP 8, Google C++ Style)
- [ ] Commit messages comply with standards (Conventional Commits)
- [ ] The PR title correctly uses type labels (feat, fix, refactor, docs, test, and so on)
- [ ] Code comments are complete and error logs are correctly recorded
- [ ] The code implementation performs checks such as return value and null pointer checks

### Function Verification Guide

**Test case locations**:

- `test/npu/` - NPU function tests
- `test/nn/` - Neural network layer tests
- `test/distributed/` - Distributed tests
- `test/dynamo/` - Compiler tests

**Run tests** (For details, refer to the [test document](./test/README.md)):

```bash
# Install test dependencies
pip3 install -r test/requirements.txt

# Complete test files
cd test
bash get_synchronized_files.sh

# Run a single test file
python test_autocast.py

# Alternatively, use run_test.py
python run_test.py -i test_autocast

# Run a specified test case
python test_autocast.py -v -k test_autocast_nn_fp32

# Run all UTs
cd ..
python ci/access_control_test.py --all
```

### CI Gate Exception Handling

CI gate exceptions mainly include the following types. Resolve them based on the related prompts:

- **Compilation exception**: Check the cause of the compilation failure, resolve the problem, and compile again
- **Static check exception**: Locate and resolve problems in the code based on the prompts (such as code style and potential bugs)
- **UT test failure**: Locate the failed test cases based on the prompts and check the causes

### AI-Assisted Development

The PyTorch NPU project encourages the use of AI-assisted development and documentation development to improve contribution efficiency. We provide the official Ascend agent-skills repository, which contains a series of AI agent skill configurations for the Ascend ecosystem. These configurations help you make better use of AI coding assistants during development.

- **agent-skills repository**: [https://gitcode.com/Ascend/agent-skills](https://gitcode.com/Ascend/agent-skills)
- The repository provides commonly used skill templates and tools for Ascend chip scenarios. You can use them for code generation, problem diagnosis, performance analysis, and other scenarios.
- The skills in the repository are continuously updated. Contributions of new skills and improvement suggestions for existing skills are welcome.

Note the following when using AI-assisted development:

- AI-generated code still requires human review to ensure code quality, security, and correctness.
- Follow the project [coding standards](#coding-standards) and [unit test guidelines](#unit-test-guidelines).
- Submitted code must pass CI gate checks (compilation, static checks, UT tests, and so on).

### Documentation Development

#### Document Storage

The documents of this project are in Markdown format and stored in the `docs/zh/` directory of the repository. They are hosted on the GitCode platform together with the code.

> **Note**: Documents are maintained in long-stable version branches, such as `v2.7.1`. To view or modify documents, switch to the corresponding long-stable version branch first.

Documents mainly include the following categories:

- **Installation guide** (`installation_guide/`): environment preparation, source code compilation, pip installation, and other instructions
- **Quick start** (`quick_start/`): quick start tutorials
- **Native API document** (`native_apis/`): native API support of each PyTorch version
- **Framework feature guide** (`framework_feature_guide_pytorch/`): NPU graph mode, Inductor, memory optimization, and other feature descriptions
- **Environment variable reference** (`environment_variable_reference/`): descriptions of NPU-related environment variables
- **Troubleshooting** (`troubleshooting/`): common problems and error code analysis
- **Security note** (`SECURITYNOTE.md`): security-related descriptions
- **Contribution guide** (`CONTRIBUTING.md`): this document

#### How to Submit Documents

The document submission process is the same as the code submission process. For details, refer to [Contribution Process](#contribution-process):

1. Fork the repository and create a branch locally.
2. Add or modify the corresponding Markdown files in the `docs/zh/` directory.
3. Note the following when writing documents:
   - Use clear and accurate Chinese expressions.
   - Ensure that code samples are runnable.
   - Follow the format and style of existing documents.
4. Submit a pull request and describe the document changes in the PR description.

#### CI Document Checks

After you submit a document pull request, the CI gate automatically performs the following checks on the changed Markdown files:

- **Newline check (NEWLINE)**: ensures that the file ends with exactly one newline and contains no redundant blank lines
- **Trailing space check (SPACES)**: ensures that no line ends with extra spaces
- **Tab character check (TABS)**: ensures that the file uses spaces instead of tab characters for indentation
- **Spelling check (CODESPELL)**: checks English spelling errors using the codespell tool

## Submitting a Pull Request

1. **Push code to the remote repository**:

   ```bash
   git add .
   git status
   git commit -m "Your commit title"
   git commit -s --amend  # Add a detailed description
   git push origin {new_branch_name}
   ```

2. **Create a pull request**

Create a pull request on GitCode and complete the form based on the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md):

- Merge source
- Modification scheme
- Document changes
- Interface changes
- Function verification
- CheckList

After confirming that the information is complete and accurate, submit the pull request and wait for code review.

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, secure, and inclusive environment for all participants:

- **Respect differences**: respect different views and experiences, and embrace diverse cultures
- **Open mind**: accept constructive criticism, and keep learning and improving
- **Focus on contributions**: focus on what benefits the community most and drive project development
- **Empathy**: show empathy to other community members and help each other

### Communication Channels

We provide multiple communication channels for community interaction:

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**: for reporting bugs and submitting feature suggestions
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**: for code review and discussion

### Issue Consultation

We warmly welcome every developer to participate in community discussions. We look forward to growing together with you:

- **Found an unresolved issue**: comment in the issue to share your solution
- **Found a long-unhandled issue**: perform a pre-check before resolving it to avoid duplicated work
- **Resolved your own reported issue**: share your solution so that the community can learn and improve together

If you have any questions, join the community discussions at any time. We look forward to your wonderful contributions!
