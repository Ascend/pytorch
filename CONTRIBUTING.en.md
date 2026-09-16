# TorchNPU Contribution Guide

Thank you for considering contributing to TorchNPU! We welcome contributions in any form, including bug fixes, feature enhancements, documentation improvements, and more. Whether you're an experienced developer or a first-time open source project, your help is invaluable.

## Project Introduction

TorchNPU is a deep learning adaptation plug-in based on Ascend NPU. It is optimized and adapted to Huawei Ascend NPU. This project provides API compatibility with the upstream PyTorch and fully utilizes the computing capability of the Ascend chip.

### Project Architecture

```text
pytorch
├── docs/                           # Project Documentation
├── ci/                             # Script for building the CI.
├── tools/                          # Development tool
├── cmake/                          # CMake Configuration
├── torch_npu/                      # NPU Core Adaptation Module
│   ├── csrc/                       # C++ backend implementation
│   ├── distributed/                # Distributed Python interface
│   ├── _inductor/                  # Inductor backend adaptation
│   ├── dynamo/                     # Dynamo Compiler Adaptation
│   ├── npu/                        # NPU Python interface
│   ├── profiler/                   # Performance Analysis Python Interface
│   ├── _afd/                       # AFD Python Interface
│   ├── _logging/                   # Python interface of the log module
│   ├── asd/                        # Asynchronous detection tool
│   ├── contrib/                    # Extended Modules Contributed
│   ├── onnx/                       # ONNX adaptation
│   └── optim/                      # Optimizer adaptation
├── third_party/                    # Third-Party Dependency
├── torchnpugen/                    # Code generation tool
├── examples/                       # Sample Code
└── test/                           # Test Case
```

### Core Module Description

| Module                     | Description                                                                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `torch_npu/csrc/core/npu`  | NPU core components: event management (NPUEvent), flow management (NPUStream), graph execution (NPUGraph), device guard (NPUGuard), and memory management |
| `torch_npu/csrc/aten`      | ATen operator NPU backend: operator registration, scheduling, and adaptation                                                                              |
| `torch_npu/csrc/framework` | Operator command framework: OpCommand, Kernel scheduling, and operator builder                                                                            |
| `torch_npu/npu/aclnn`      | ACLNN operator Python interface: AscendCL NPU operator library encapsulation                                                                              |
| `torch_npu/npu/amp`        | Automatic blending precision: GradScaler, FP16/BF16 support                                                                                               |
| `torchnpugen`              | Code generation tools: automatic differential code generation, code templates                                                                             |
| `examples`                 | Sample code: distributed communication, model inference, ResNet example                                                                                   |
| `third_party/op-plugin`    | Operator plug-in: custom operator implementation and TorchNPU operator coverage                                                                          |
| `test/npu`                 | NPU function test: device management, memory allocation, and operator test                                                                                |

## Contribution Overview

### Contribution Modes

We look forward to your joining! Every contribution is an important driving force for TorchNPU's progress:

- **Report issues**: Report bugs or submit feature suggestions to help us discover and solve problems
- **Submit proposals**: Propose technical solutions or architectural designs to guide project evolution
- **Contribute code**: Submit code fixes or new feature implementations, directly participating in project development
- **Improve documentation**: Improve documentation or supplement missing content to enhance project readability
- **Code review**: Review Pull Requests to help improve code quality
- **Assist the community**: Answer questions and share solutions to help more developers
- **Share and promote**: Share the project on blogs and social media to expand community influence

### Communication Channels

We provide multiple communication channels for you to engage in community interaction:

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**: For reporting bugs and proposing feature suggestions
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**: For code review and discussion
- **[SIG](https://gitcode.com/Ascend/community/tree/master/AscendForPyTorch/sigs)**: Community collaboration organizations divided by technical area, responsible for code review and technical decisions

### Contribution Scenarios

Submitting an Issue is the first step in contributing. When [creating an Issue](https://gitcode.com/Ascend/pytorch/issues/create/choose), please select the corresponding template:

#### Bug Report

When you find a bug, please use the 🐛 Bug report template to submit an Issue. The template includes the following required fields:

- **Environment information**: Operating system, Ascend hardware information, CANN software version, and installed software versions
- **🐛 Problem description**: Clearly and concisely describe the specific phenomenon and occurrence scenario of the bug; if reproducible, provide a **minimal reproducible example** (stripped of irrelevant code and business dependencies, directly runnable), and paste the error message including the **full traceback**

> ⚠️ Please desensitize: Before submitting, remove sensitive information such as API keys, passwords, private addresses, and personal data; use placeholders such as `<TOKEN>` if necessary.

#### Feature Request

When you have new feature suggestions or performance optimization ideas, please use the ✨ Feature request template to submit. The template includes the following fields:

- **💻 Requirement background, current status, expected feature content, specific design proposal, and test plan** (required): Explain the requirement scenario, why current capabilities are insufficient, the proposed feature content and design approach; relevant Issue/PR links may be attached
- **Alternative solutions**: Describe other implementation approaches you have considered
- **Additional notes**: Other relevant notes or screenshots

#### RFC (Request for Comments)

For major architectural changes or technical proposals, please use the 📜 RFC (Request for Comments) template to initiate an RFC, which will be advanced after community discussion and review. The RFC template is a structured document with the following main sections:

- **Overview**: Introduction, motivation, goals and non-goals
- **Use case analysis**: Scenario use cases, key performance indicators, security/privacy and DFX requirements
- **Design proposal**: Overall proposal, technology selection, functional and performance design, security/privacy and DFX design, programming and invocation design (including interface definitions)
- **Test design**: Unit tests, integration tests, end-to-end tests
- **Drawbacks and risks**: Breaking changes, performance regression, implementation cost, compatibility, etc.
- **Prior art**: Similar designs referenced from other projects/communities
- **Unresolved questions**: Open questions awaiting community discussion and decision

## Contribution Process

To ensure smooth contributions, please follow this process:

### Step 1: Sign the CLA

Before you submit code to the TorchNPU community for the first time, you need to sign the CLA (Contributor License Agreement).

- **Individual contributors**: please refer to the [ICLA online document](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/individual)
- **Corporate contributors**: please refer to the [CCLA online document](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/employee)

### Step 2: Submit an Issue

Before starting development, please submit an Issue describing the problem you want to solve or the feature you propose, so that the community can understand and discuss it. For the detailed format requirements of Issues, please refer to [Contribution Scenarios](#contribution-scenarios).

### Step 3: Community Discussion

After submitting an Issue, please discuss it in the community and confirm the proposal is feasible before starting development.

#### Claim a Task

If you want to fix a bug, please comment `/assign` under the corresponding Issue to claim the task and avoid duplicate work by multiple people.

#### SIG Review

The following types of changes require community review through the SIG (Special Interest Group) mechanism:

- **Patch replacement**: Patch replacement of TorchNPU native interfaces
- **Header file macro update**: Add or modify macro definitions
- **API interface change**: Add, modify, or delete public APIs
- **Core component change**: Modifications to core modules such as memory management and device management
- **RFC proposal**: Major architectural changes or technical proposals that require initiating an RFC to solicit community input

A SIG is a community collaboration organization divided by technical area, responsible for code review and technical decisions. For the detailed SIG list, please refer to the [Ascend community SIG page](https://gitcode.com/Ascend/community/tree/master/AscendForPyTorch/sigs). For changes involving the above types, please declare community review through the SIG mechanism in the PR.

### Step 4: Fork and Clone

1. **Fork the repository**: On the GitCode platform, click the "Fork" button in the upper right corner of the repository to create a copy under your personal account

2. **Clone to local**:

   ```bash
   git clone https://gitcode.com/<your-username>/pytorch.git
   cd pytorch
   ```

3. **Create a development branch**:

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

### Step 5: Code Development

Please follow the coding styles below to make TorchNPU easy to develop, review, and maintain.

#### Coding Guide

- **Python**: recommended [PEP 8 Coding Style](https://pep8.org/)
- **C++**: recommended [Google C++ Coding Guide](http://google.github.io/styleguide/cppguide.html)

#### Refactoring Guide

Developers are encouraged to refactor the code to eliminate code smells. All code should conform to the coding style and test style requirements.

#### Environment Setup

For detailed installation guidance (including dependencies and environment requirements for each version), please refer to the [Ascend community installation guide](https://www.hiascend.com/document/detail/zh/Pytorch/master/installguide/ref/docs/zh/installation_guide/references/building_from_source.md).

Quick build commands:

```bash
# Install dependencies and compile
bash ci/build.sh --python=3.10

# Build for a specified PyTorch version (supporting version 2.13 and later. The available version is the version.txt file.)
# --The value of torch indicates the version of the package to be built, which can contain the post number. (For example, 2.13.0.post1 indicates the post build of the 2.13 main line.)
# The PyTorch corresponding to major.minor has been installed.
bash ci/build.sh --python=3.10 --torch=2.14.0

# Or manually compile with CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

To speed up compilation, the project has built-in automatic detection for Ninja, Mold linker, and CCache. For usage, see [Build Acceleration](docs/en/installation_guide/references/build_acceleration.md).

### Step 6: Local Static Check

After completing code development, please run static checks locally to ensure code style and quality meet the requirements. For details, see [Appendix A: Local Static Check](#appendix-a-local-static-check).

### Building with Clang

For instructions on compiling with Clang, please refer to [Building with Clang](docs/en/installation_guide/references/building_with_clang.md)

### Step 7: Local Functional Verification

Please write and run unit tests to ensure the code functions correctly. If new features are added, corresponding test cases must be added. For details, see [Appendix B: Unit Tests and Functional Verification](#appendix-b-unit-tests-and-functional-verification).

### Step 8: Submit a Pull Request

1. **Push code to the remote repository**:

   ```bash
   git add .
   git status
   git commit -m "feat(module): brief description of changes"
   git commit -s --amend  # Add a detailed description
   git push origin {new_branch_name}
   ```

   > **Note**: One PR per commit is the recommended practice for this project. If multiple incremental commits are produced during development, please use `git rebase -i` to squash them into a single commit before pushing, to keep the commit history clean.

2. **Create a Pull Request**

Create a Pull Request on GitCode, filling it out completely according to the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md):

- Merge source
- Modification proposal
- Documentation changes
- Interface changes
- Functional verification
- Checklist

After confirming the information is complete and accurate, submit the Pull Request and wait for code review.

### Step 9: Trigger Gate Checks

After submitting the PR, type `compile` in the PR comment area to trigger the gate checks. For more bot command details, refer to the [Ascend community bot usage guide](https://gitcode.com/Ascend/infrastructure/blob/master/docs/robot/robot%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md).

Gate check failures mainly include the following types. Please resolve them according to the related prompts:

- **Compilation error**: Check the cause of the code compilation failure, resolve the issue, and recompile
- **Static check exception**: Follow the prompts to find and resolve issues in the code (such as code style, potential bugs, etc.)
- **UT test failure**: Locate the failing test cases according to the prompts and investigate the cause

### Step 10: Code Review and Merge

#### PR Merge Requirements

**Merge checklist** (for detailed requirements, see the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md)):

- [ ] Code compiles successfully
- [ ] Static checks pass (CppLint, CppCheck, etc.)
- [ ] UT test cases pass
- [ ] Code style complies with specifications (PEP 8, Google C++ Style)
- [ ] Commit message is standardized (Conventional Commits compliant)
- [ ] PR title correctly uses type labels (feat, fix, refactor, docs, test, etc.)
- [ ] Code comments are complete and error logs are recorded correctly
- [ ] Code implementation includes return value and null pointer validation

#### Code Review Process

1. **Reviewer review**: After the reviewer of the corresponding module reviews the code, type `/lgtm` in the PR comment area to indicate approval
2. **Committer approval**: After the committer of the corresponding module reviews and approves, type `/approve` in the PR comment area to indicate consent to merge
3. **Automatic merge**: After the PR meets the merge conditions, the bot will automatically merge the code

> **Tip**: The complete list of reviewers and committers can be viewed by clicking the corresponding link in the first comment of the PR.

## Advanced Guide

### AI-Assisted R&D

The TorchNPU project encourages the use of AI-assisted R&D and document development to improve contribution efficiency. We provide the official Ascend agent-skills repository, which contains a series of AI Agent Skill configurations for the Ascend ecosystem, helping you better utilize AI coding assistants during development.

- **agent-skills repository**: [https://gitcode.com/Ascend/agent-skills](https://gitcode.com/Ascend/agent-skills)
- This repository provides Skill templates and tools commonly used in Ascend chip scenarios, applicable to code generation, problem diagnosis, performance analysis, and more.
- The skills in the repository are continuously updated. New skills are welcome, as are improvement suggestions for existing skills.

When using AI-assisted R&D, please note:

- AI-generated code still requires manual review to ensure code quality, security, and correctness.
- Follow the project's [Coding Guide](#step-5-code-development) and [Appendix B: Unit Tests and Functional Verification](#appendix-b-unit-tests-and-functional-verification).
- Submitted code must pass gate checks (compilation, static checks, UT tests, etc.).

### Document Development Notes

#### Document Hosting

This project's documentation is in Markdown format, stored in the `docs/en/` directory of the repository, and hosted on the GitCode platform together with the code.

> **Note**: Documentation is hosted on long-term stable version branches, such as `v2.7.1`. If you need to view or modify documentation, please switch to the corresponding long-term stable version branch.

The documentation mainly includes the following categories:

- **Installation Guide** (`installation_guide/`): Environment preparation, source compilation, pip installation, etc.
- **Quick Start** (`quick_start/`): Quick start tutorials.
- **Native API Documentation** (`native_apis/`): TorchNPU native API support status for each version.
- **Framework Feature Guide** (`framework_feature_guide_pytorch/`): NPU graph mode, Inductor, memory optimization, and other feature descriptions.
- **Environment Variable Reference** (`environment_variable_reference/`): NPU-related environment variable descriptions.
- **Troubleshooting** (`troubleshooting/`): Common problems and error code analysis.
- **Security Statement** (`SECURITYNOTE.md`): Security-related notes.
- **Contribution Guide** (`CONTRIBUTING.en.md`): This document.

#### How to Submit Documentation

The documentation submission process is the same as the code submission process. Please refer to [Contribution Process](#contribution-process):

1. Fork the repository and create a local branch.
2. Add or modify the corresponding Markdown file in the `docs/en/` directory.
3. When writing documentation, note:
   - Use clear and accurate English expressions.
   - Code examples must be runnable.
   - Follow the format and style of existing documentation.
4. Submit a Pull Request and describe the documentation changes in the PR description.

#### CI Documentation Checks

After a documentation Pull Request is submitted, the CI gate will automatically perform the following checks on the changed Markdown files:

- **Newline check (NEWLINE)**: Ensure the file has exactly one newline at the end and no redundant blank lines.
- **Trailing space check (SPACES)**: Ensure no extra spaces at the end of each line.
- **Tab check (TABS)**: Ensure spaces are used for indentation instead of tabs (Tab).
- **Spell check (CODESPELL)**: Check English spelling errors using the codespell tool.

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe, and inclusive environment for all participants:

- **Respect differences**: Respect different viewpoints and experiences, and embrace multiculturalism
- **Open mind**: Accept constructive criticism, and keep learning and improving
- **Focus on contributions**: Focus on what is most beneficial to the community and drive project development
- **Empathy**: Show empathy to other community members and help each other

## Question Consultation

We warmly welcome every developer to actively participate in community discussions! Looking forward to growing together with you:

- **Unresolved issues found**: Feel free to comment in Issues to showcase your solution
- **Long-unhandled issues**: Pre-checking before solving is recommended to avoid duplicate work
- **Successfully solved your own reported issue**: Please also share your solution so the community can learn and progress together

If you have any questions, feel free to discuss in the community at any time. We look forward to your wonderful contributions!

## Appendix A: Local Static Check

The project uses [lintrunner](https://github.com/suo/lintrunner) for static checks, supporting running check items locally that are fully consistent with CI, including Python code style (Flake8, Ruff, PYFMT), C++ format (ClangFormat, ClangTidy), and spell check (Codespell).

### Install Dependencies

```bash
# Install lintrunner and uv (required by some linters)
pip install lintrunner
pip install uv
```

### Initialize (run once on first use or update)

```bash
# Download external binary tools required by lintrunner (clang-format, clang-tidy, etc.)
lintrunner init
```

### Run Static Checks

```bash
# Check file deltas of current workspace changes and HEAD commit (workspace + HEAD)
lintrunner

# Run only specified check items
lintrunner --take FLAKE8,RUFF,PYFMT,SPACES,TABS,NEWLINE

# Automatically fix auto-fixable issues (formatter linters; skip PYREFLY if needed)
lintrunner --skip PYREFLY -a

# Check only file deltas of current workspace changes
git diff --name-only HEAD | xargs lintrunner
```

> **Tip**: The `--take` parameter specifies running only some check items. Common items:
>
> | Code            | Description                                                                                                                                     |
> |-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
> | `FLAKE8`        | Python syntax and style check                                                                                                                    |
> | `RUFF`          | Python fast lint and import sorting                                                                                                             |
> | `PYFMT`         | Python code formatting (usort + ruff-format)                                                                                                     |
> | `CLANGFORMAT`   | C++ code formatting                                                                                                                              |
> | `CLANGTIDY`     | C++ static analysis                                                                                                                              |
> | `SPACES`        | Trailing space check                                                                                                                             |
> | `TABS`          | Tab character check                                                                                                                              |
> | `NEWLINE`       | End-of-file newline check                                                                                                                        |
> | `CODESPELL`     | Spell check. If a false positive occurs, add the word in lexicographic order to `tools/linter/dictionary.txt` and re-check |

For more commands, see the [lintrunner wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).

## Appendix B: Unit Tests and Functional Verification

### Unit Test Guide

- **Python**: recommended [pytest](http://pytest.org/en/latest/)
- **C++**: recommended [Googletest Primer](https://github.com/google/googletest/blob/main/docs/primer.md)

The design intent of a test case should be reflected by its annotation name.

### Functional Verification Guide

**Test case locations**:

- `test/npu/` - NPU functional tests
- `test/nn/` - Network layer tests
- `test/distributed/` - Distributed tests
- `test/dynamo/` - Compiler tests

**Run tests** (for details, see the [test documentation](./test/README.md)):

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

# Run a specified test case
python test_autocast.py -v -k test_autocast_nn_fp32

# Run the full UT
cd ..
python ci/access_control_test.py --all
```
