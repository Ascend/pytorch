# Ascend Extension for PyTorch Contribution Guide

Thank you for your interest in contributing to PyTorch. We welcome all kinds of contributions, including bug fixes, new features, documentation improvements, and more. Whether you're a seasoned open-source contributor or just getting started, your help is greatly valued and appreciated.

## Project Introduction

This is an Ascend NPU‑based distribution of PyTorch, deeply optimized for Huawei Ascend NPUs. It offers API compatibility with the official PyTorch and fully leverages the computing capabilities of Ascend chips.

### Project Architecture

```text
pytorch
├── docs/                           # Project documentation
├── ci/                             # CI build scripts
├── tools/                          # Development tools
├── cmake/                          # CMake configuration
├── torch_npu/                      # NPU main adaptation modules
│   ├── csrc/                       # C++ backend implementation
│   ├── distributed/                # Distributed Python interfaces
│   ├── _inductor/                  # Inductor backend adaptation
│   ├── dynamo/                     # Dynamo compiler adaptation
│   ├── npu/                        # Python interface for the NPU 
│   ├── profiler/                   # Python interface for performance analysis 
│   ├── _afd/                       # Python interface for the AFD 
│   ├── _logging/                   # Python Interface for the logging module 
│   ├── asd/                        # Asynchronous detection tool
│   ├── contrib/                    # Contributed extensions
│   ├── onnx/                       # ONNX adaptation
│   └── optim/                      # Optimizer adaptation
├── third_party/                    # Third-party dependencies
├── torchnpugen/                    # Code generation tools
├── examples/                       # Code examples
└── test/                           # Test cases
```

### Main Modules

| Module | Description |
| --- | --- |
| `torch_npu/csrc/core/npu` | NPU core components: event management (NPUEvent), stream management (NPUStream), graph execution (NPUGraph), device guard (NPUGuard), memory management |
| `torch_npu/csrc/aten` | ATen operator NPU backend: operator registration, dispatch, implementation adaptation |
| `torch_npu/csrc/framework` | Operator command framework: OpCommand, Kernel dispatch, operator builder |
| `torch_npu/npu/aclnn` | ACLNN operator Python interface: AscendCL NPU operator library encapsulation |
| `torch_npu/npu/amp` | Automatic mixed precision: with support for GradScaler, FP16, and BF16 |
| `torchnpugen` | Code generation tool: automatic differentiation code generation, code templates |
| `examples` | Example code: distributed communication, model inference, ResNet examples |
| `third_party/op-plugin` | Operator plugin: custom operator implementation, PyTorch operator override |
| `test/npu` | NPU functional tests: device management, memory allocation, operator tests |

## Ways to Contribute

We eagerly look forward to your participation! Every contribution is a vital force driving PyTorch forward:

- **Report Issues**: Submit bug reports or feature suggestions to help us identify problems and plan improvements.
- **Commit Code**: Submit pull requests with bug fixes or new features to directly participate in project development.
- **Improve Documentation**: Help enhance or supplement existing documentation to make the project easier to understand and use.
- **Review Code**: Participate in pull request reviews to help maintain and improve code quality.
- **Share and Promote**: Spread the word by sharing the project on blogs or social media, and don't forget to give us a ⭐.

## Contribution Scenarios

We welcome all types of contributions and look forward to growing this project together with you.

### 1. Requirements and Suggestions on Features

If you have ideas for new features or performance optimizations, we warmly invite you to open an Issue to discuss them with the community.

**Issue Type**: Requirement/Feature Suggestion

**Required Content**:

- **Feature Background**: Explain what problem this feature solves, and what value it brings to users.
- **Feature Description**: Describe the proposed feature in detail.
- **Design Proposal**: Provide the technical approach, key module design, and relationships between upstream and downstream components.
- **Expected Benefits**: Clarify the goals, performance metrics, and accuracy performance.

### 2. Bug Reporting and Fixing

If you discover a bug or a documentation issue, we sincerely welcome your feedback and fixing suggestions.

**Bug Report Template**:

- **Environment**: PyTorch version, OS, Python version, CANN version, etc.
- **Problem Description**: Add labels to help highlight this issue on the dashboard.
- **Reproduction Steps**: Provide step-by-step instructions to reproduce the issue.
- **Expected Behavior**: Describe what you expected to happen.
- **Notes for Reviewers**: Include any special context or background information reviewers should know (if any).

**Bug Fix Workflow**:

1. Find the corresponding bug report in the issues.
2. Comment `/assign` to claim the task.
3. Create a branch for the fix.
4. Submit a Pull Request.

### 3. Community Co-Building

If you can help solve issues others have raised, we would like to see your solutions shared in the issue threads.

## Contribution Workflow

### Contributor License Agreement

The CLA is required before you commit code to the PyTorch community for the first time.

For individual contributors, please refer to the [ICLA Online Document](https://www.mindspore.cn/icla) for details.

### Development and Testing

1. **Fork Repository**: On the GitCode platform, click the **Fork** button at the top right of the repository page to copy it to your own account.

2. **Clone to Local**:

   ```bash
   git clone https://gitcode.com/<your-username>/pytorch.git
   cd pytorch
   ```

3. **Create Development Branch**:

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

4. **Develop Code**: Follow the **[Code Standards](#code-standards)**.

5. **Perform Code Testing**: Run tests to ensure the code functions properly.

6. **Perform Gate Check**: Run CI checks to ensure the code passes the build, static check, and unit test.

7. **Submit Pull Request**: Submit a PR and wait for code review.

8. **Submit for Community Review**: If the update involves patches, header file macros, APIs, and more, a community review must be submitted.

### Code Merge Review Requirements

The following types of modifications require community review:

- **Patch Replacement**: Patch replacement of PyTorch native APIs
- **Header File Macro Updates**: Adding or modifying macro definitions
- **API Changes**: Adding, modifying, or deleting public APIs
- **Core Component Changes**: Modifications to core modules such as memory management and device management

## Code Standards

Follow these styles to make PyTorch easy to develop, review, and maintain.

### Coding Guide

- **Python**: The [PEP 8 coding style](https://pep8.org/) is recommended.
- **C++**: The [Google C++ Style Guide](http://google.github.io/styleguide/cppguide.html) is recommended.

To perform code checks, refer to [Local Static Check](#local-static-check).

### Unit Testing Guide

- **Python**: The [pytest](http://pytest.org/en/latest/) is recommended.
- **C++**: The [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md) is recommended.

The design intent of a test case should be reflected through its comment name.

### Refactoring Guide

We encourage developers to refactor code to eliminate code smells. All code should meet the requirements of coding style and testing style.

## Practical Guide

### Environment Setup and Build

**Build and Compilation**:

```bash
# Install dependencies and build
bash ci/build.sh --python=3.10

# Build for a specific PyTorch version (supports 2.10.0 / 2.11.0 / 2.12.0)
# The corresponding version of PyTorch should be installed in the environment
bash ci/build.sh --python=3.10 --torch=2.10.0

# Or manually build using CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Tips for Build Acceleration

#### Using Ninja Build

By default, CMake uses the Makefile generator. Installing the Ninja build system can significantly speed up the build process.

The project's `setup.py` will automatically detect whether Ninja is installed in the system: if the `CMAKE_GENERATOR` environment variable is set to `ninja`, or the `ninja` command is available in the `PATH`, Ninja will be automatically used as the build system.

```bash
pip install ninja
```

After installing Ninja, the build will take effect automatically without additional configuration. If you have built previously, you need to perform a cleanup first after installing Ninja:

```bash
python setup.py clean
```

#### Using the Mold Linker

In a development cycle where a single file is frequently modified and recompiled, linking time can dominate. The system linker (GNU `ld`) included with most Linux distributions is slow, and using a faster linker can significantly improve the build experience.

The project's `CMakeLists.txt` has built-in automatic Linker detection logic: it prioritizes detecting the mold linker and automatically enables it if found (`-fuse-ld=mold`).

```bash
sudo apt install mold
# Or install from source: https://github.com/rui314/mold
```

After Installation, re-build takes effect automatically. To confirm whether the linker is correctly enabled, check if the link options in the build output include `-fuse-ld=mold`.

#### Using CCache

Even if dependency tracking is based on file modification time, there are still many scenarios where files are recompiled. Using ccache can effectively avoid redundant compilation and save a significant amount of time.

The project's `CMakeLists.txt` has built-in ccache auto-detection logic, which will be automatically enabled after installing ccache. However, it is recommended to adjust the ccache configuration (such as cache directory, cache size, compression, etc.) according to your own environment for optimal results:

```bash
sudo apt install ccache
# or
sudo yum install ccache
```

Verify whether ccache is effective: Execute two consecutive full builds, and the second one should be significantly faster than the first. If it is not effective, check whether the `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` variables in `build/CMakeCache.txt` contain ccache:

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

The project uses [lintrunner](https://github.com/suo/lintrunner) for static checks, supporting locally running checks that are fully consistent with CI, including Python code style (Flake8, Ruff, PYFMT), C++ format (ClangFormat, ClangTidy), spell check (Codespell), etc.

#### Installing Dependencies

```bash
# Install lintrunner and uv (required by some linters)
pip install lintrunner
pip install uv
```

#### Initialization (Upon First Use or Updating)

```bash
# Download the external binary tools required by lintrunner (clang-format, clang-tidy, etc.)
lintrunner init
```

#### Running Static Check

```bash
# Check the current workspace changes and the file delta of the HEAD commit (Workspace + HEAD)
lintrunner

# Run only specified check items
lintrunner --take FLAKE8,RUFF,PYFMT,SPACES,TABS,NEWLINE

# Automatically fix auto-fixable issues (formatter-type linters, such as ignoring PYREFLY)
lintrunner --skip PYREFLY -a

# Only check the incremental changes of files modified in the current workspace
git diff --name-only HEAD | xargs lintrunner
```

> **Tip**: The `--take` parameter can specify running only certain check items. Common items are as follows:
>
> | Code | Description |
> | --- | --- |
> | `FLAKE8` | Python syntax and style check |
> | `RUFF` | Python fast lint and import sorting |
> | `PYFMT` | Python code formatting (usort + ruff-format) |
> | `CLANGFORMAT` | C++ code formatting |
> | `CLANGTIDY` | C++ static analysis |
> | `SPACES` | Trailing whitespace check |
> | `TABS` | Tab character check |
> | `NEWLINE` | End-of-file newline check |
> | `CODESPELL` | Spell check. If a false positive occurs, add the false positive word to `tools/linter/dictionary.txt` in lexicographical order and recheck |

For more execution commands, refer to the [lintrunner wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).

### PR Merge Requirements

**Merge checklist** (For detailed requirements, refer to the [PR Template](../../.gitcode/PULL_REQUEST_TEMPLATE.md)):

- [ ] Code build
- [ ] Static check (CppLint, CppCheck, etc.)
- [ ] Unit test
- [ ] Code style compliance (PEP 8, Google C++ Style)
- [ ] Commit convention compliance
- [ ] Correct use of PR title type tags (feat, fix, refactor, docs, test, etc.)
- [ ] Complete code comments and correct error logs
- [ ] Verification of return values and null pointers in code implementation

### Functional Verification Guide

**Test Case Locations**:

- `test/npu/` - NPU functional tests
- `test/nn/` - Network layer tests
- `test/distributed/` - Distributed tests
- `test/dynamo/` - Compiler tests

**Running Tests** (for details, see the [test documentation](../../test/README.md)):

```bash
# Install test dependencies
pip3 install -r test/requirements.txt

# Complete the test file
cd test
bash get_synchronized_files.sh

# Run a single test file
python test_autocast.py

# Or use run_test.py
python run_test.py -i test_autocast

# Run Specified Test Cases
python test_autocast.py -v -k test_autocast_nn_fp32

# Run full unit tests
cd ..
python ci/access_control_test.py --all
```

### Gatekeeper Exception Handling

Gatekeeper exceptions mainly include the following types. Please resolve them according to the relevant prompts:

- **Build Exception**: Check the cause of the code build failure, resolve the issue, and then rebuild.
- **Static Check Exception**: Follow the prompts to find and resolve issues in the code (such as code style, potential bugs, etc.)
- **Unit Test Failure**: Follow the prompts to find the failed test cases and investigate the cause

### AI-Assisted R&D Support

The PyTorch NPU project encourages the use of AI to assist in R&D and documentation development to improve contribution efficiency. We provide the official Ascend agent-skills repository, which contains a series of AI agent skill configurations suitable for the Ascend ecosystem, helping you better utilize AI coding assistants during development.

- **agent-skills Repository**: [https://gitcode.com/Ascend/agent-skills](https://gitcode.com/Ascend/agent-skills)
- This repository provides commonly used skill templates and tools for Ascend chip scenarios, which can be used for code generation, problem diagnosis, performance analysis, and other scenarios.
- The skills in the repository are continuously updated, and contributions of new skills or suggestions for improving existing skills are welcome.

Note when using AI for R&D support:

- AI-generated code still requires manual review to ensure code quality, security, and correctness.
- Follow the project's [Code Standards](#code-standards) and [Unit Testing Guide](#unit-testing-guide).
- Submitted code must pass gate checks (build, static check, UT, etc.).

### Notes for Documentation Development

#### Documentation Hosting Method

The documentation for this project is in Markdown format, stored in the `docs/en/` directory of the repository, and hosted on the GitCode platform along with the code.

> **Note**: The documentation is hosted in a long-term stable version branch, such as `v2.7.1`. If you need to view or modify the documentation, switch to the corresponding long-term stable version branch.

The following types of documents are included mainly:

- **Installation Guide** (`installation_guide/`): Instructions on environment setup, source code build, pip installation, etc.
- **Quick Start** (`quick_start/`): A quick start tutorial.
- **Native API Documentation** (`native_apis/`): Support status of native PyTorch APIs across different versions.
- **Framework Feature Guide** (`framework_feature_guide_pytorch/`): Descriptions of features such as NPU graph mode, Inductor, and memory optimization.
- **Environment Variable Reference** (`environment_variable_reference/`): Description of NPU-related environment variables.
- **Troubleshooting** (`troubleshooting/`): Analysis of common issues and error codes.
- **Security Statement** (`SECURITYNOTE.md`): Security-related instructions.
- **Contribution Guide** (`CONTRIBUTING.md`): Instructions on how to contribute to this project.

#### How to Submit Documentation

The submission process for documentation is the same as for code. Refer to the [Contribution Workflow](#contribution-workflow):

1. Fork the Repository and create a branch locally.
2. Add or modify the corresponding Markdown files in the `docs/en/` directory.
3. When writing documentation, note the following:
   - Use clear and accurate Chinese expressions.
   - Ensure code examples are executable.
   - Follow the format and style of existing documentation.
4. Submit a Pull Request and describe the documentation changes in the PR description.

#### CI Documentation Check

After submitting a PR for documentation, the CI gate will automatically perform the following checks on the changed Markdown files:

- **Newlines (NEWLINE)**: Ensures there is exactly one newline at the end of the file and that the file does not contain extra blank lines.
- **Trailing spaces (SPACES)**: Ensures there are no trailing spaces at the end of each line.
- **Tabs (TABS)**: Ensures spaces are used for indentation in the file instead of tabs.
- **Spelling (CODESPELL)**: Checks for English spelling errors using the codespell tool.

## Submitting a Pull Request

1. **Push Code to Remote Repository**:

   ```bash
   git add .
   git status
   git commit -m "Your commit title"
   git commit -s --amend  # Add detailed descriptiontailed description
   git push origin {new_branch_name}
   ```

2. **Create a Pull Request**

Create a Pull Request on GitCode, and fill it out completely according to the [PR template](../../.gitcode/PULL_REQUEST_TEMPLATE.md):

- Merge Source
- Modification Plan
- Documentation Changes
- API Changes
- Functional Verification
- Checklist

After confirming the information is complete and accurate, submit a PR and wait for code review.

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe, and inclusive environment for all participants:

- **Respect Differences**: Respect different viewpoints and experiences, and embrace diverse cultures
- **Open Mindset**: Accept constructive criticism, and continuously learn and improve
- **Focus on Contribution**: Focus on what is best for the community, and drive the project forward
- **Empathy**: Show empathy towards other community members, and help each other

### Communication Channels

We provide multiple communication channels for you to engage with the community:

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**: For reporting bugs and suggesting features
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**: For code review and discussion

### Questions and Inquiries

We warmly welcome every developer to actively participate in community discussions! We look forward to growing together with you:

- **Found an unresolved issue**: Feel free to comment on the Issue and showcase your solution
- **Encountered a long-standing unresolved issue**: It is recommended to perform a pre-check before solving to avoid duplicated effort
- **Successfully resolved an issue you reported**: Share your solution so the community can learn and progress together

If you have any questions, feel free to discuss them in the community at any time. We look forward to your outstanding contributions.
