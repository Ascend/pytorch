# PyTorch Contribution Guide

Thank you for considering contributing to PyTorch. We welcome all kinds of contributions, including bug fixes, feature enhancements, documentation improvements, and so on. Whether you are an experienced developer or a first-time participant in an open-source project, your help is highly valuable.

## Project Introduction

PyTorch is a deep learning framework distribution based on the Ascend NPU, deeply optimized and adapted for Huawei Ascend NPUs. This project provides API compatibility with the official PyTorch and fully leverages the computing power of Ascend chips.

### Project Architecture

```text
pytorch
├── docs/                           # Project documentation
├── ci/                             # CI build scripts
├── tools/                          # Development tools
├── cmake/                          # CMake configuration
├── torch_npu/                      # NPU core adaptation modules
│   ├── csrc/                       # C++ backend implementation
│   ├── distributed/                # Distributed Python interfaces
│   ├── _inductor/                  # Inductor backend adaptation
│   ├── dynamo/                     # Dynamo compiler adaptation
│   ├── npu/                        # NPU Python interfaces
│   ├── profiler/                   # Profiling Python interfaces
│   ├── _afd/                       # AFD Python interfaces
│   ├── _logging/                   # Logging module Python interfaces
│   ├── asd/                        # Asynchronous detection tool
│   ├── contrib/                    # Contributed extension modules
│   ├── onnx/                       # ONNX adaptation
│   └── optim/                      # Optimizer adaptation
├── third_party/                    # Third-party dependencies
├── torchnpugen/                    # Code generation tool
├── examples/                       # Example code
└── test/                           # Test cases
```

### Core Module Description

| Module                         | Description                                                                       |
|----------------------------|--------------------------------------------------------------------------|
| `torch_npu/csrc/core/npu`  | NPU core components: event management (NPUEvent), stream management (NPUStream), graph execution (NPUGraph), device guard (NPUGuard), memory management |
| `torch_npu/csrc/aten`      | ATen operator NPU backend: operator registration, dispatch, implementation adaptation                                              |
| `torch_npu/csrc/framework` | Operator command framework: OpCommand, Kernel dispatch, operator builder                                         |
| `torch_npu/npu/aclnn`      | ACLNN operator Python interfaces: AscendCL NPU operator library encapsulation                                    |
| `torch_npu/npu/amp`        | Automatic mixed precision: GradScaler, FP16/BF16 support                                           |
| `torchnpugen`              | Code generation tool: automatic differentiation code generation, code templates                                                     |
| `examples`                 | Example code: distributed communication, model inference, ResNet examples                                                |
| `third_party/op-plugin`    | Operator plugin: custom operator implementation, PyTorch operator override                                                |
| `test/npu`                 | NPU functional tests: device management, memory allocation, operator tests                                                  |

## Ways to Contribute

We eagerly look forward to your participation. Every contribution is an important force driving PyTorch forward:

- **Report issues**: Report bugs or submit feature suggestions to help us discover and resolve problems
- **Contribute code**: Submit code fixes or new feature implementations to directly participate in project development
- **Improve documentation**: Improve the documentation or supplement missing content to enhance the readability of the project
- **Review code**: Review pull requests to help improve code quality
- **Share and promote**: Share the project on blogs or social media, and give the repository a star

## Contribution Scenarios

This project warmly welcomes all kinds of contributions. We look forward to your participation.

### 1. Requirements and Feature Suggestions

If you have new feature suggestions or performance optimization ideas, we warmly invite you to submit an Issue for in-depth discussion with the community.

**Issue Type**: Requirement/Feature Suggestion

**Content to Include**:

- **Feature background**: What problem this feature solves and what value it brings to users
- **Feature description**: Describe the suggested feature in detail
- **Design proposal**: Technical approach, key module design, and upstream and downstream component relationships
- **Expected benefits**: Feature objectives, performance metrics, and accuracy performance

### 2. Bug Reporting and Fixing

If you find bugs or documentation issues, we sincerely welcome your feedback and fixing suggestions.

**Bug Report Format**:

- **Environment information**: PyTorch version, OS, Python version, CANN version, and so on
- **Problem description**: Add labels to highlight the issue on the dashboard
- **Reproduction steps**: Describe in as much detail as possible how to reproduce the problem
- **Expected behavior**: Describe the behavior you expect
- **Special notes for reviewers**: Include any special circumstances

**Fixing process**:

1. Find the corresponding bug description in the Issue
2. Comment `/assign` to claim the task
3. Create a branch for the fix
4. Submit a Pull Request

### 3. Community Co-Building

If you can help solve problems raised by others, we eagerly look forward to your sharing your solutions in the Issue.

## Contribution Workflow

### Contributor License Agreement

You need to sign the [CLA](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/individual) before submitting code to the PyTorch community for the first time.

### Development and Testing

1. **Fork the repository**: On the GitCode platform, click the "Fork" button at the top right of the repository page to clone the repository to your personal account

2. **Clone to local**:

   ```bash
   git clone https://gitcode.com/<your-username>/pytorch.git
   cd pytorch
   ```

3. **Create a development branch**:

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

4. **Develop code**: Follow the **[code standards](#code-standards)**

5. **Test the code**: Run tests to ensure the code functions properly

6. **Gate check**: Run CI checks to ensure the code passes compilation, static checks, and UT tests

7. **Submit a Pull Request**: Submit a PR and wait for code review

8. **Community review**: If the update involves patches, header file macros, API interfaces, and so on, a community review must be submitted

### Code Merge Review Requirements

The following types of modifications require community review:

- **Patch replacement**: Patch replacement of PyTorch native interfaces
- **Header file macro updates**: Adding or modifying macro definitions
- **API interface changes**: Adding, modifying, or deleting public APIs
- **Core component changes**: Modifications to core modules such as memory management and device management

## Code Standards

Follow these styles to make PyTorch easy to develop, review, and maintain.

### Coding Guide

- **Python**: It is advised to use the [PEP 8 coding style](https://pep8.org/)
- **C++**: It is advised to use the [Google C++ Style Guide](http://google.github.io/styleguide/cppguide.html)

To run code checks, refer to [Local Static Check](#local-static-check).

### Unit Testing Guide

- **Python**: It is advised to use [pytest](http://pytest.org/en/latest/)
- **C++**: It is advised to use the [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md)

The design intent of a test case should be reflected through its comment name.

### Refactoring Guide

We encourage developers to refactor code to eliminate code smells. All code should meet the requirements of coding style and testing style.

## Hands-on Guide

### Environment Setup and Build

**Build and compilation**:

```bash
# Install dependencies and build
bash ci/build.sh --python=3.10

# Build for a specified PyTorch version (supports 2.10.0 / 2.11.0 / 2.12.0)
# The corresponding PyTorch version must be installed in the environment
bash ci/build.sh --python=3.10 --torch=2.10.0

# Or manually build using CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Tips for Build Acceleration

#### Using Ninja Build

By default, CMake uses the Makefile generator. Installing the Ninja build system can significantly speed up compilation.

The `setup.py` of this project automatically detects whether Ninja is installed in the system. If the `CMAKE_GENERATOR` environment variable is set to `ninja`, or the `ninja` command is available in the `PATH`, Ninja is automatically used as the build system.

```bash
pip install ninja
```

After installing Ninja, the compilation takes effect automatically without additional configuration. If you have compiled before, you need to perform a cleanup once after installing Ninja:

```bash
python setup.py clean
```

#### Using the Mold Linker

In a development loop where a single file is frequently modified and recompiled, linking time dominates. The system linker (GNU `ld`) included with most Linux distributions is slow, and using a faster linker can significantly improve the build experience.

The `CMakeLists.txt` of this project has built-in linker auto-detection logic: it prioritizes detecting the mold linker and automatically enables it if present (`-fuse-ld=mold`).

```bash
sudo apt install mold
# Or install from source: https://github.com/rui314/mold
```

After installation, recompilation takes effect automatically. To confirm whether the linker is correctly enabled, check whether the link options in the compilation output include `-fuse-ld=mold`.

#### Using CCache

Even if dependency tracking is based on file modification time, files are still repeatedly compiled in many scenarios. Using ccache can effectively avoid repeated compilation and save a lot of time.

The `CMakeLists.txt` of this project has built-in ccache auto-detection logic, which is enabled automatically after ccache is installed. However, you are advised to adjust the ccache configuration (such as the cache directory, cache size, and compression) based on your own environment for optimal results:

```bash
sudo apt install ccache
# or
sudo yum install ccache
```

Verify whether ccache is effective: run two consecutive full compilations, and the second one should be noticeably faster than the first. If it is not effective, check whether the `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` variables in `build/CMakeCache.txt` contain ccache:

```cmake
//C compiler launcher
CMAKE_C_COMPILER_LAUNCHER:PATH=/usr/bin/ccache

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:PATH=/usr/bin/ccache
```

#### Building Only the Required Target

If you only need to rebuild `torch_npu.so`, you can directly specify the target in the build directory to avoid a full build:

```bash
cd build && ninja torch_npu
```

If Ninja is not installed, replace `ninja` with `make`.

### Local Static Check

The project uses [lintrunner](https://github.com/suo/lintrunner) for static checks. It supports running check items fully consistent with CI locally, including Python code style (Flake8, Ruff, PYFMT), C++ formatting (ClangFormat, ClangTidy), spell checking (Codespell), and so on.

#### Installing Dependencies

```bash
# Install lintrunner and uv (required by some linters)
pip install lintrunner
pip install uv
```

#### Initialization (Run Once on First Use or Update)

```bash
# Download the external binary tools required by lintrunner (clang-format, clang-tidy, and so on)
lintrunner init
```

#### Running Static Checks

```bash
# Check the incremental changes of files in the current workspace and the HEAD commit (workspace + HEAD)
lintrunner

# Run only the specified check items
lintrunner --take FLAKE8,RUFF,PYFMT,SPACES,TABS,NEWLINE

# Automatically fix auto-fixable issues (formatter-type linters, such as ignoring PYREFLY)
lintrunner --skip PYREFLY -a

# Only check the incremental changes of files modified in the current workspace
git diff --name-only HEAD | xargs lintrunner
```

> **Tip**: The `--take` parameter specifies that only some check items are run. Common items are as follows:
>
> | Code            | Description                                                             |
> |---------------|----------------------------------------------------------------|
> | `FLAKE8`      | Python syntax and style check                                                 |
> | `RUFF`        | Python fast lint and import sorting                                     |
> | `PYFMT`       | Python code formatting (usort + ruff-format)                              |
> | `CLANGFORMAT` | C++ code formatting                                                      |
> | `CLANGTIDY`   | C++ static analysis                                                       |
> | `SPACES`      | Trailing whitespace check                                                         |
> | `TABS`        | Tab character check                                                       |
> | `NEWLINE`     | End-of-file newline check                                                       |
> | `CODESPELL`   | Spell check. If it is a false positive, add the false positive word to `tools/linter/dictionary.txt` in lexicographical order and run the check again |

For more execution commands, refer to the [lintrunner wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).

### PR Merge Requirements

**Merge checklist** (for detailed requirements, refer to the [PR Template](../../.gitcode/PULL_REQUEST_TEMPLATE.md)):

- [ ] Code compiles successfully
- [ ] Static checks pass (CppLint, CppCheck, and so on)
- [ ] UT test cases pass
- [ ] Code style complies with the specifications (PEP 8, Google C++ Style)
- [ ] Commit messages comply with the specifications (Conventional Commits)
- [ ] PR titles correctly use type tags (feat, fix, refactor, docs, test, and so on)
- [ ] Code comments are complete and error logs are correctly recorded
- [ ] Code implementation performs validation such as return value and null pointer checks

### Functional Verification Guide

**Test case locations**:

- `test/npu/` - NPU functional tests
- `test/nn/` - Network layer tests
- `test/distributed/` - Distributed tests
- `test/dynamo/` - Compiler tests

**Running tests** (for details, refer to the [test documentation](../../test/README.md)):

```bash
# Install test dependencies
pip3 install -r test/requirements.txt

# Complete the test files
cd test
bash get_synchronized_files.sh

# Run a single test file
python test_autocast.py

# Or use run_test.py
python run_test.py -i test_autocast

# Run specified test cases
python test_autocast.py -v -k test_autocast_nn_fp32

# Run all UT tests
cd ..
python ci/access_control_test.py --all
```

### Gate Exception Handling

Gate exceptions mainly include the following types. Resolve them according to the related prompts:

- **Compilation exception**: Check the cause of the code compilation failure, resolve the problem, and then recompile
- **Static check exception**: Find and resolve the problems in the code according to the prompts (such as code style and potential bugs)
- **UT test failure**: Find the test cases that fail to pass according to the prompts and check the cause

### AI-Assisted Development

The PyTorch NPU project encourages the use of AI to assist in development and documentation to improve contribution efficiency. We provide the official Ascend agent-skills repository, which contains a series of AI Agent Skill configurations suitable for the Ascend ecosystem, helping you better use AI coding assistants during development.

- **agent-skills repository**: [https://gitcode.com/Ascend/agent-skills](https://gitcode.com/Ascend/agent-skills)
- This repository provides common Skill templates and tools for Ascend chip scenarios, which can be used in scenarios such as code generation, problem diagnosis, and performance analysis.
- The skills in the repository are continuously updated, and contributions of new Skills or suggestions for improving existing Skills are welcome.

Note the following when using AI-assisted development:

- AI-generated code still requires manual review to ensure code quality, security, and correctness.
- Follow the [code standards](#code-standards) and [unit testing guide](#unit-testing-guide) of the project.
- Submitted code must pass gate checks (compilation, static checks, UT tests, and so on).

### Documentation Development Instructions

#### Documentation Hosting Method

The documentation of this project is in Markdown format, stored in the `docs/en/` directory of the repository, and hosted on the GitCode platform together with the code.

> **Note**: The documentation is hosted in the branch of the long-term stable version, such as `v2.7.1`. If you need to view or modify the documentation, switch to the corresponding long-term stable version branch.

The documentation mainly includes the following categories:

- **Software installation** (`installation_guide/`): Instructions on environment preparation, source code compilation, pip installation, and so on.
- **Quick start** (`quick_start/`): A quick start tutorial.
- **Native APIs** (`native_apis/`): Support status of PyTorch native APIs for each version.
- **Framework features** (`framework_feature_guide_pytorch/`): Descriptions of features such as NPU graph mode, Inductor, and memory optimization.
- **Environment variables** (`environment_variable_reference/`): Descriptions of NPU-related environment variables.
- **Troubleshooting** (`troubleshooting/`): Analysis of common issues and error codes.
- **Security statement** (`SECURITYNOTE.md`): Security-related instructions.
- **Contribution guide** (`CONTRIBUTING.md`): This document.

#### How to Submit Documentation

The submission process for documentation is the same as that for code. Refer to [Contribution Workflow](#contribution-workflow):

1. Fork the repository and create a branch locally.
2. Add or modify the corresponding Markdown files in the `docs/en/` directory.
3. When writing documentation, note the following:
   - Use clear and accurate Chinese expressions.
   - Ensure that code examples are runnable.
   - Follow the format and style of the existing documentation.
4. Submit a Pull Request and describe the documentation changes in the PR description.

#### CI Documentation Check

After submitting a Pull Request for documentation, the CI gate automatically performs the following checks on the changed Markdown files:

- **Newline check (NEWLINE)**: Ensures that the file has exactly one newline at the end and does not contain extra blank lines.
- **Trailing space check (SPACES)**: Ensures that no line ends with extra spaces.
- **Tab check (TABS)**: Ensures that the file uses space indentation instead of tabs.
- **Spell check (CODESPELL)**: Checks English spelling errors using the codespell tool.

## Submitting a Pull Request

1. **Push code to the remote repository**:

   ```bash
   git add .
   git status
   git commit -m "Your commit title"
   git commit -s --amend  # Add detailed description
   git push origin {new_branch_name}
   ```

2. **Create a Pull Request**

Create a Pull Request on GitCode and fill it out completely according to the [PR Template](../../.gitcode/PULL_REQUEST_TEMPLATE.md):

- Merge source
- Modification plan
- Documentation changes
- Interface changes
- Functional verification
- Checklist

After confirming that the information is complete and accurate, submit the Pull Request and wait for code review.

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe, and inclusive environment for all participants:

- **Respect differences**: Respect different viewpoints and experiences, and embrace diverse cultures
- **Open mindset**: Accept constructive criticism, and continuously learn and improve
- **Focus on contribution**: Focus on what is most beneficial to the community and drive the project forward
- **Empathy**: Show empathy to other community members and help each other

### Communication Channels

We provide multiple communication channels to facilitate your participation in community interaction:

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**: Used to report bugs and propose feature suggestions
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**: Used for code review and discussion

### Questions and Inquiries

We warmly welcome every developer to actively participate in community discussions. We look forward to growing together with you:

- **Found an unresolved issue**: Welcome to comment on the Issue and share your solution
- **Encountered an issue that has been unresolved for a long time**: You are advised to perform a pre-check before solving it to avoid duplicated work
- **Successfully resolved an issue you reported**: Please also share your solution so that the community can learn and improve together

If you have any questions, feel free to discuss them in the community. We look forward to your outstanding contributions.
