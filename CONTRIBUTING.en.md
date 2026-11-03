# PyTorch Contribution Guide

Thank you for considering contributing to PyTorch! We welcome contributions of all forms, including bug fixes, feature enhancements, documentation improvements, and so on. Whether you are an experienced developer or a first-time open source contributor, your help is invaluable.

## Project Introduction

PyTorch is a deep learning framework distribution based on the Ascend NPU. It is deeply optimized and adapted for Huawei Ascend NPUs. This project provides API compatibility with the official PyTorch and fully exploits the computing capability of Ascend chips.

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

### Core Module Description

| Module                     | Description                                                                                     |
|----------------------------|--------------------------------------------------------------------------------------------------|
| `torch_npu/csrc/core/npu`  | Core NPU components: event management (NPUEvent), stream management (NPUStream), graph execution (NPUGraph), device guard (NPUGuard), and memory management |
| `torch_npu/csrc/aten`      | ATen operator NPU backend: operator registration, scheduling, and implementation adaptation      |
| `torch_npu/csrc/framework` | Operator command framework: OpCommand, kernel scheduling, and operator builders                  |
| `torch_npu/npu/aclnn`      | ACLNN operator Python interface: encapsulation of the AscendCL NPU operator library              |
| `torch_npu/npu/amp`        | Automatic mixed precision: GradScaler and FP16/BF16 support                                      |
| `torchnpugen`              | Code generation tool: automatic differentiation code generation and code templates               |
| `examples`                 | Sample code: distributed communication, model inference, and ResNet samples                      |
| `third_party/op-plugin`    | Operator plugin: custom operator implementation and PyTorch operator override                    |
| `test/npu`                 | NPU function tests: device management, memory allocation, and operator testing                   |

## How to Contribute

We look forward to your participation! Every contribution is an important force driving PyTorch forward:

- **Report issues**: Report bugs or submit feature suggestions to help us identify and resolve problems.
- **Contribute code**: Submit code fixes or new feature implementations to participate in project development directly.
- **Improve documents**: Improve documents or supplement missing content to enhance project readability.
- **Review code**: Review pull requests to help improve code quality.
- **Share the project**: Share the project on blogs and social media, and give the repository a ⭐.

## Contribution Scenarios

This project warmly welcomes contributions of all forms. We look forward to your participation!

### 1. Requirements and Feature Suggestions

If you have suggestions for new features or ideas for performance optimization, we sincerely invite you to submit an issue for in-depth community discussion.

**Issue type**: requirement/feature suggestion

**Content to include**:

- **Feature background**: what problem the feature solves and what value it brings to users
- **Feature description**: a detailed description of the suggested feature
- **Design scheme**: technical ideas, key module designs, and relationships between upstream and downstream components
- **Expected benefits**: feature goals, performance metrics, and accuracy

### 2. Bug Report and Fix

If you find a bug or a document problem, we sincerely welcome your feedback and fix suggestions.

**Bug report format**:

- **Environment information**: PyTorch version, OS, Python version, CANN version, and so on
- **Problem description**: add labels so that the problem is highlighted on the issue dashboard
- **Reproduction steps**: describe in as much detail as possible how to reproduce the problem
- **Expected behavior**: describe the behavior you expect
- **Special notes for reviewers**: any special circumstances, if any

**Fix process**:

1. Find the corresponding bug description in an issue.
2. Comment `/assign` to claim the task.
3. Create a branch for the fix.
4. Submit a pull request.

### 3. Community Building Support

If you are able to resolve issues raised by others, we look forward to your sharing the solutions in the issues.

## Contribution Process

### Contributor License Agreement

Before submitting code to the PyTorch community for the first time, you need to sign the [CLA](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/individual).

### Development and Testing

1. **Fork the repository**: On the GitCode platform, click "Fork" in the upper right corner of the repository to clone the repository to your personal account.

2. **Clone the repository to the local host**:

   ```bash
   git clone https://gitcode.com/<your-username>/pytorch.git
   cd pytorch
   ```

3. **Create a development branch**:

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

4. **Develop code**: Follow **[Coding Standards](#coding-standards)**.

5. **Test code**: Run tests to ensure that the code works properly.

6. **Run CI checks**: Run CI checks to ensure that the code passes compilation, static checks, and UT tests.

7. **Submit a pull request**: Submit a PR and wait for code review.

8. **Community review**: If the update involves patches, macros in header files, or API interfaces, submit it for community review.

### Code Merge Review Requirements

The following types of changes require community review:

- **Patch replacement**: patch replacements of native PyTorch interfaces
- **Macro update in header files**: addition or modification of macro definitions
- **API interface changes**: addition, modification, or deletion of public APIs
- **Core component changes**: changes to core modules such as memory management and device management

## Coding Standards

Follow these styles to make PyTorch easy to develop, review, and maintain.

### Coding Guidelines

- **Python**: It is recommended to use the [PEP 8 coding style](https://pep8.org/).
- **C++**: It is recommended to use the [Google C++ Style Guide](http://google.github.io/styleguide/cppguide.html).

To perform code checks, refer to [Local Static Checks](#local-static-checks).

### Unit Test Guidelines

- **Python**: It is recommended to use [pytest](http://pytest.org/en/latest/).
- **C++**: It is recommended to use the [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md).

The design intent of a test case should be reflected in its comment name.

### Refactoring Guidelines

We encourage developers to refactor code to eliminate code smells. All code should comply with the coding style and test style requirements.

## Practical Guide

### Environment Setup and Compilation

**Build and compile**:

```bash
# Install dependencies and compile
bash ci/build.sh --python=3.10

# Build for a specified PyTorch version (2.10.0, 2.11.0, or 2.12.0)
# The corresponding PyTorch version must have been installed in the environment
bash ci/build.sh --python=3.10 --torch=2.10.0

# Alternatively, use CMake to compile manually
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Compilation Acceleration Tips

#### Building with Ninja

By default, CMake uses the Makefile generator. Installing the Ninja build system can significantly speed up compilation.

The `setup.py` of this project automatically detects whether Ninja is installed in the system. If the `CMAKE_GENERATOR` environment variable is set to `ninja`, or the `ninja` command is available in `PATH`, Ninja is automatically used as the build system.

```bash
pip install ninja
```

After Ninja is installed, compilation takes effect automatically without extra configuration. If the project has been compiled before, perform a cleanup once after installing Ninja:

```bash
python setup.py clean
```

#### Using the Mold Linker

In a development loop where a single file is frequently modified and recompiled, linking time dominates. The system linker (GNU `ld`) that ships with most Linux distributions is slow. Using a faster linker can significantly improve the build experience.

The `CMakeLists.txt` of this project has built-in linker auto-detection logic. The mold linker is detected first and automatically enabled if present (`-fuse-ld=mold`).

```bash
sudo apt install mold
# Alternatively, install from source: https://github.com/rui314/mold
```

After installation, recompile and the setting takes effect automatically. To confirm that the linker is correctly enabled, check whether the link options in the compilation output contain `-fuse-ld=mold`.

#### Using CCache

Even when dependency tracking is based on file modification time, files are recompiled in many scenarios. Using ccache effectively avoids redundant compilation and saves a lot of time.

The `CMakeLists.txt` of this project has built-in ccache auto-detection logic, so ccache is automatically enabled after installation. It is recommended to adjust the ccache configuration (such as the cache directory, cache size, and compression) according to your environment for the best effect:

```bash
sudo apt install ccache
# Alternatively
sudo yum install ccache
```

To verify that ccache takes effect, run two full compilations in a row. The second one should be significantly faster than the first one. If it does not take effect, check whether the `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` variables in `build/CMakeCache.txt` contain ccache:

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

The project uses [lintrunner](https://github.com/suo/lintrunner) to perform static checks. It supports running check items locally that are fully consistent with CI, including Python code style (Flake8, Ruff, and PYFMT), C++ formatting (ClangFormat and ClangTidy), spelling checks (Codespell), and so on.

#### Installing Dependencies

```bash
# Install lintrunner and uv (required by some linters)
pip install lintrunner
pip install uv
```

#### Initialization (perform once upon first use or update)

```bash
# Download the external binary tools required by lintrunner (clang-format, clang-tidy, and so on)
lintrunner init
```

#### Performing Static Checks

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

> **Note**: The `--take` parameter specifies running only some check items. Common items are as follows:
>
> | Code          | Description                                                                            |
> |---------------|------------------------------------------------------------------------------------------|
> | `FLAKE8`      | Python syntax and style check                                                           |
> | `RUFF`        | Fast Python lint and import sorting                                                     |
> | `PYFMT`       | Python code formatting (usort + ruff-format)                                            |
> | `CLANGFORMAT` | C++ code formatting                                                                     |
> | `CLANGTIDY`   | C++ static analysis                                                                     |
> | `SPACES`      | Trailing space check                                                                    |
> | `TABS`        | Tab character check                                                                     |
> | `NEWLINE`     | End-of-file newline check                                                               |
> | `CODESPELL`   | Spelling check. If a word is a false positive, add it to `tools/linter/dictionary.txt` in alphabetical order and check again |

For more execution commands, refer to the [lintrunner wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).

### PR Merge Requirements

**Merge checklist** (for details, refer to the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md)):

- [ ] The code is successfully compiled.
- [ ] Static checks pass (CppLint, CppCheck, and so on).
- [ ] UT test cases pass.
- [ ] The code style complies with the standards (PEP 8 and Google C++ Style).
- [ ] Commit messages are standardized (Conventional Commits).
- [ ] PR titles correctly use type labels (feat, fix, refactor, docs, test, and so on).
- [ ] Code comments are complete and error logs are correctly recorded.
- [ ] The code implementation performs checks on return values, null pointers, and so on.

### Function Verification Guide

**Test case locations**:

- `test/npu/`: NPU function tests
- `test/nn/`: network layer tests
- `test/distributed/`: distributed tests
- `test/dynamo/`: compiler tests

**Running tests** (for details, refer to the [test document](./test/README.md)):

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

### CI Anomaly Handling

CI anomalies mainly include the following types. Resolve them according to the related prompts:

- **Compilation anomaly**: check the cause of the compilation failure, resolve the problem, and compile again.
- **Static check anomaly**: locate and resolve the problems in the code according to the prompts (such as code style and potential bugs).
- **UT test failure**: locate the failed test cases according to the prompts and check the causes.

### AI-Assisted Development

The PyTorch NPU project encourages AI-assisted development and documentation development to improve contribution efficiency. We provide the official Ascend agent-skills repository, which contains a series of AI agent skill configurations for the Ascend ecosystem. These configurations help you better use AI coding assistants during development.

- **agent-skills repository**: [https://gitcode.com/Ascend/agent-skills](https://gitcode.com/Ascend/agent-skills)
- The repository provides commonly used skill templates and tools for Ascend chip scenarios, which can be used for code generation, problem diagnosis, performance profiling, and other scenarios.
- The skills in the repository are continuously updated. You are welcome to contribute new skills or suggest improvements to existing skills.

Pay attention to the following when using AI-assisted development:

- AI-generated code still requires manual review to ensure code quality, security, and correctness.
- Follow the project's [Coding Standards](#coding-standards) and [Unit Test Guidelines](#unit-test-guidelines).
- The submitted code must pass CI checks (compilation, static checks, UT tests, and so on).

### Documentation Development

#### Document Hosting

Documents of this project are in Markdown format and stored in the `docs/zh/` directory of the repository. They are hosted on the GitCode platform together with the code.

> **NOTE**: Documents are hosted in long-term support version branches, such as `v2.7.1`. If you need to view or modify documents, switch to the corresponding long-term support version branch.

Documents mainly include the following categories:

- **Software installation** (`installation_guide/`): environment preparation, source code compilation, pip installation, and so on.
- **Quick start** (`quick_start/`): quick start tutorials.
- **Native APIs** (`native_apis/`): native API support of each PyTorch version.
- **Framework features** (`framework_feature_guide_pytorch/`): features such as the NPU graph mode, Inductor, and memory optimization.
- **Environment variables** (`environment_variable_reference/`): descriptions of NPU-related environment variables.
- **Troubleshooting** (`troubleshooting/`): common problems and error code analysis.
- **Security note** (`SECURITYNOTE.md`): security-related description.
- **Contribution guide** (`CONTRIBUTING.md`): this document.

#### How to Submit Documents

The document submission process is the same as the code submission process. Refer to [Contribution Process](#contribution-process):

1. Fork the repository and create a local branch.
2. Add or modify the corresponding Markdown file in the `docs/zh/` directory.
3. Note the following when writing documents:
   - Use clear and accurate Chinese expressions.
   - Ensure that code samples are runnable.
   - Follow the format and style of existing documents.
4. Submit a pull request and describe the document changes in the PR description.

#### CI Document Checks

After a document pull request is submitted, the CI gate automatically performs the following checks on the changed Markdown files:

- **Newline check (NEWLINE)**: ensures that the file ends with exactly one newline character and contains no redundant blank lines.
- **Trailing space check (SPACES)**: ensures that no line ends with redundant spaces.
- **Tab check (TABS)**: ensures that files are indented with spaces instead of tabs.
- **Spelling check (CODESPELL)**: checks English spelling errors through the codespell tool.

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

Create a pull request on GitCode and completely fill in the following items according to the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md):

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

- **Respect differences**: respect different opinions and experiences, and embrace diverse cultures.
- **Be open-minded**: accept constructive criticism, and keep learning and improving.
- **Focus on contributions**: focus on what is best for the community and drive the project forward.
- **Show empathy**: show empathy to other community members and help each other.

### Communication Channels

We provide multiple communication channels for you to participate in community interaction:

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**: for reporting bugs and submitting feature suggestions
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**: for code review and discussion

### Question Consultation

We warmly welcome every developer to actively participate in community discussions! We look forward to growing together with you:

- **Find an unresolved issue**: welcome to comment on the issue and show your solution.
- **Encounter a long-standing unresolved issue**: perform a precheck before solving it to avoid duplicated work.
- **Solve a problem you reported**: also share your solution so that the community can learn and progress together.

If you have any questions, feel free to communicate and discuss in the community. We look forward to your wonderful contributions!
