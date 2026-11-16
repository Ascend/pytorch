# TorchNPU Contribution Guide

Thank you for considering contributing to TorchNPU! We welcome contributions of any form, including bug fixes, feature enhancements, and documentation improvements. Whether you are an experienced developer or participating in an open-source project for the first time, your help is invaluable.

## Project Introduction

TorchNPU is a deep learning adapter plugin based on Ascend NPU, deeply optimized and adapted for Huawei Ascend NPUs. This project is API-compatible with upstream PyTorch and fully unleashes the compute capabilities of Ascend chips.

### Project Architecture

```text
pytorch
├── docs/                           # Project documentation
├── ci/                             # CI build scripts
├── tools/                          # Development tools
├── cmake/                          # CMake configuration
├── torch_npu/                      # NPU core adaptation module
│   ├── csrc/                       # C++ backend implementation
│   ├── distributed/                # Distributed Python interface
│   ├── _inductor/                  # Inductor backend adaptation
│   ├── dynamo/                     # Dynamo compiler adaptation
│   ├── npu/                        # NPU Python interface
│   ├── profiler/                   # Performance analysis Python interface
│   ├── _afd/                       # AFD Python interface
│   ├── _logging/                   # Log module Python interface
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

| Module                     | Description                                                                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `torch_npu/csrc/core/npu`  | NPU core components: event management (NPUEvent), stream management (NPUStream), graph execution (NPUGraph), device guard (NPUGuard), and memory management |
| `torch_npu/csrc/aten`      | ATen operator NPU backend: operator registration, scheduling, and implementation adaptation                                                                              |
| `torch_npu/csrc/framework` | Operator command framework: OpCommand, Kernel scheduling, and operator builder                                                                             |
| `torch_npu/npu/aclnn`      | ACLNN operator Python interface: encapsulation of the AscendCL NPU operator library                                                                        |
| `torch_npu/npu/amp`        | Automatic mixed precision: GradScaler, FP16/BF16 support                                                                                                   |
| `torchnpugen`              | Code generation tool: automatic differentiation code generation and code templates                                                                        |
| `examples`                 | Example code: distributed communication, model inference, and ResNet examples                                                                              |
| `third_party/op-plugin`    | Operator plugin: custom operator implementation and TorchNPU operator coverage                                                                             |
| `test/npu`                 | NPU functional tests: device management, memory allocation, and operator tests                                                                              |

## Contribution Overview

### Contribution Modes

We look forward to your joining! Every contribution is an important driving force for the progress of TorchNPU:

- **Report issues**: Report bugs or submit feature suggestions to help us discover and solve problems
- **Submit proposals**: Propose technical solutions or architectural designs to guide the evolution of the project
- **Contribute code**: Submit code fixes or new feature implementations to directly participate in project development
- **Improve documentation**: Improve documentation or supplement missing content to enhance project readability
- **Code review**: Review Pull Requests to help improve code quality
- **Assist the community**: Answer others' questions and share solutions to help more developers
- **Share and promote**: Share the project on blogs and social media to expand community influence

### Communication Channels

We provide multiple communication channels for you to interact with the community:

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**: For reporting bugs and proposing feature suggestions
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**: For code review and discussion
- **[SIG](https://gitcode.com/Ascend/community/tree/master/AscendForPyTorch/sigs)**: Community collaboration organizations divided by technical area, responsible for code review and technical decisions

### Contribution Scenarios

Submitting an Issue is the first step of contribution. When [creating an Issue](https://gitcode.com/Ascend/pytorch/issues/create/choose), select the corresponding template:

#### Bug Report

When you find a bug, use the 🐛 Bug Report template to submit an Issue. The template includes the following required fields:

- **Environment information**: Operating system, Ascend hardware information, CANN software version, and installed versions of the corresponding software
- **🐛 Problem description**: Clearly and concisely describe the specific symptom of the bug and the scenario in which it occurs. If the bug is reproducible, provide a **minimal reproducible example** (excluding irrelevant code and business dependencies, directly runnable), and paste the error information with the **complete stack trace (Full Traceback)**

> ⚠️ Please desensitize the information: Before submission, remove sensitive information such as API keys, passwords, private addresses, and personal data. Use placeholders such as `<TOKEN>` when necessary.

#### Feature Request

When you have new feature suggestions or performance optimization ideas, use the ✨ Feature Request template to submit an Issue. The template includes the following fields:

- **💻 Requirement background, current status, expected feature content, specific design solution, and test solution** (required): Describe the requirement scenario, why the current capabilities cannot meet the requirement, the proposed feature content and design approach. Related Issue/PR links can be attached.
- **Alternative solutions**: Describe other implementation approaches you have considered
- **Additional notes**: Other related notes or screenshots

#### Request for Comments (RFC)

For major architectural changes or technical proposals, use the 📜 RFC template to initiate an RFC, which will be advanced after community discussion and review. The RFC template is a structured document with the following main sections:

- **Overview**: Introduction, motivation, goals, and non-goals
- **Use case analysis**: Scenario use cases, key performance indicators, and security, privacy, and DFX requirements
- **Design solution**: Overall solution, technology selection, functional and performance design, security, privacy, and DFX design, and programming and invocation design (including interface definitions)
- **Test design**: Unit tests, integration tests, and end-to-end tests
- **Drawbacks and risks**: Breaking changes, performance regression, implementation cost, compatibility, etc.
- **Prior art**: Similar designs drawn from other projects or communities
- **Unresolved questions**: Open questions awaiting community discussion and decision

## Contribution Process

To ensure smooth contribution, follow the process below:

### Step 1: Sign the CLA

Before you submit code to the TorchNPU community for the first time, you need to sign the CLA (Contributor License Agreement).

- **Individual contributors**: See the [ICLA online document](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/individual)
- **Corporate contributors**: See the [CCLA online document](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/employee)

### Step 2: Submit an Issue

Before starting development, submit an Issue describing the problem you want to solve or the feature you suggest, so that the community can understand and discuss it. For the detailed format requirements of Issues, see [Contribution Scenarios](#contribution-scenarios).

### Step 3: Community Discussion

After submitting the Issue, discuss it in the community and confirm the solution is feasible before starting development.

#### Claim a Task

If you want to fix a bug, comment `/assign` under the corresponding Issue to claim the task and avoid duplicate development by multiple people.

#### SIG Review

The following types of changes require community review through the Special Interest Group (SIG) mechanism:

- **Patch replacement**: Patch replacement of TorchNPU native APIs
- **Header file macro update**: Add or modify macro definitions
- **API changes**: Add, modify, or delete public APIs
- **Core component changes**: Modifications to core modules such as memory management and device management
- **RFC proposals**: Major architectural changes or technical solutions that require initiating an RFC to solicit community opinions

A SIG is a community collaboration organization divided by technical area, responsible for code review and technical decisions. For the detailed SIG list, see the [Ascend community SIG page](https://gitcode.com/Ascend/community/tree/master/AscendForPyTorch/sigs). For changes involving the preceding types, declare community review through the SIG mechanism in the PR.

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

Follow the coding styles below to make TorchNPU easy to develop, review, and maintain.

#### Coding Guide

- **Python**: The [PEP 8 coding style](https://pep8.org/) is recommended
- **C++**: The [Google C++ coding guide](http://google.github.io/styleguide/cppguide.html) is recommended

#### Refactoring Guide

We encourage developers to refactor code to eliminate code smells. All code should conform to the coding style and test style requirements.

#### Environment Setup

For detailed installation guidance (including dependencies and environment requirements for each version), see the [Ascend community installation guide](https://www.hiascend.com/document/detail/en/Pytorch/master/installguide/ref/docs/en/installation_guide/references/building_from_source.md).

Quick build commands:

```bash
# Install dependencies and compile
bash ci/build.sh --python=3.10

# Build for a specified PyTorch version (2.13 and later supported; the available versions are defined in version.txt)
# The value of --torch is the package version to build, which can carry a post number (for example, 2.13.0.post1 indicates a post build of the 2.13 mainline)
# The PyTorch of the corresponding major.minor version must be installed in the environment
bash ci/build.sh --python=3.10 --torch=2.14.0

# Or compile manually with CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

To speed up compilation, the project has built-in automatic detection of Ninja, the Mold linker, and CCache. For usage, see [Build Acceleration](docs/en/installation_guide/references/build_acceleration.md).

### Step 6: Local Static Check

After completing code development, run static checks locally to ensure the code style and quality meet the requirements. For details, see [Appendix A: Local Static Check](#appendix-a-local-static-check).

### Building with Clang

For instructions on compiling with Clang, see [Building with Clang](docs/en/installation_guide/references/building_with_clang.md).

### Step 7: Local Functional Verification

Write and run unit tests to ensure the code functions correctly. If new features are added, corresponding test cases must be added. For details, see [Appendix B: Unit Tests and Functional Verification](#appendix-b-unit-tests-and-functional-verification).

### Step 8: Submit a Pull Request

1. **Push code to the remote repository**:

   ```bash
   git add .
   git status
   git commit -m "feat(module): brief description of changes"
   git commit -s --amend  # Add a detailed description
   git push origin {new_branch_name}
   ```

   > **Note**: One commit per PR is the recommended practice for this project. If multiple incremental commits are generated during development, use `git rebase -i` to squash them into a single commit before pushing, so as to keep the commit history clear.

2. **Create a Pull Request**

Create a Pull Request on GitCode and fill it out completely according to the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md):

- Merge source
- Modification solution
- Documentation changes
- Interface changes
- Functional verification
- CheckList

After confirming that the information is complete and accurate, submit the Pull Request and wait for code review.

### Step 9: Trigger Gate Checks

After submitting the PR, enter `compile` in the PR comment area to trigger gate checks. For more bot commands, see the [Ascend community bot usage guide](https://gitcode.com/Ascend/infrastructure/blob/master/docs/robot/robot%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md).

Gate check exceptions mainly include the following types. Resolve them according to the related prompts:

- **Compilation exception**: Check the cause of the compilation failure, resolve the issue, and recompile
- **Static check exception**: Locate and resolve the issue in the code according to the prompts (such as code style and potential bugs)
- **UT test failure**: Locate the failing test cases according to the prompts and investigate the cause

### Step 10: Code Review and Merge

#### PR Merge Requirements

**Merge checklist** (for detailed requirements, see the [PR template](./.gitcode/PULL_REQUEST_TEMPLATE.md)):

- [ ] Code compiles successfully
- [ ] Static checks pass (CppLint, CppCheck, etc.)
- [ ] UT test cases pass
- [ ] Code style complies with specifications (PEP 8, Google C++ Style)
- [ ] Commit messages are standardized (Conventional Commits compliant)
- [ ] PR titles correctly use type labels (feat, fix, refactor, docs, test, etc.)
- [ ] Code comments are complete and error logs are correctly recorded
- [ ] Code implementation includes return value and null pointer validation

#### Code Review Process

1. **Reviewer review**: After the reviewer of the corresponding module reviews the code, enter `/lgtm` in the PR comment area to indicate approval
2. **Committer approval**: After the committer of the corresponding module approves the review, enter `/approve` in the PR comment area to consent to the merge
3. **Automatic merge**: After the PR meets the merge conditions, the bot will automatically merge the code

> **Tip**: The complete list of reviewers and committers can be viewed by clicking the corresponding links in the first comment of the PR.

## Advanced Guide

### AI-Assisted R&D

The TorchNPU project encourages the use of AI-assisted R&D and document development to improve contribution efficiency. We provide the official Ascend agent-skills repository, which contains a series of AI agent skill configurations applicable to the Ascend ecosystem, helping you better utilize AI coding assistants in development.

- **agent-skills repository**: [https://gitcode.com/Ascend/agent-skills](https://gitcode.com/Ascend/agent-skills)
- This repository provides skill templates and tools commonly used in Ascend chip scenarios, applicable to code generation, problem diagnosis, performance analysis, and other scenarios.
- The skills in the repository are continuously updated. Contributions of new skills and improvement suggestions for existing skills are welcome.

When using AI-assisted R&D, note the following:

- AI-generated code still requires manual review to ensure code quality, security, and correctness.
- Follow the project's [Coding Guide](#step-5-code-development) and [Appendix B: Unit Tests and Functional Verification](#appendix-b-unit-tests-and-functional-verification).
- Submitted code must pass gate checks (compilation, static checks, UT tests, etc.).

### Document Development Notes

#### Document Hosting

This project's documentation is in Markdown format, stored in the `docs/en/` directory of the repository, and hosted on the GitCode platform together with the code.

> **Note**: Documentation is hosted on long-term stable version branches, such as `v2.7.1`. To view or modify the documentation, switch to the corresponding long-term stable version branch.

The documentation mainly includes the following categories:

- **Installation Guide** (`installation_guide/`): Environment preparation, source compilation, pip installation, and other instructions.
- **Quick Start** (`quick_start/`): Quick start tutorial.
- **Native API Documentation** (`native_apis/`): TorchNPU native API support status for each version.
- **Framework Feature Guide** (`framework_feature_guide_pytorch/`): NPU graph mode, Inductor, memory optimization, and other feature descriptions.
- **Environment Variable Reference** (`environment_variable_reference/`): Descriptions of NPU-related environment variables.
- **Troubleshooting** (`troubleshooting/`): Common problems and error code analysis.
- **Security Statement** (`SECURITYNOTE.en.md`): Security-related notes.
- **Contribution Guide** (`CONTRIBUTING.en.md`): This document.

#### How to Submit Documentation

The documentation submission process is the same as the code submission process. See [Contribution Process](#contribution-process):

1. Fork the repository and create a local branch.
2. Add or modify the corresponding Markdown file in the `docs/en/` directory.
3. When writing documentation, note the following:
   - Use clear and accurate English expressions.
   - Code examples must be runnable.
   - Follow the format and style of the existing documentation.
4. Submit a Pull Request and describe the documentation changes in the PR description.

#### CI Documentation Checks

After a documentation Pull Request is submitted, the CI gate automatically performs the following checks on the changed Markdown files:

- **Newline check (NEWLINE)**: Ensure the file ends with exactly one newline and contains no redundant blank lines.
- **Trailing space check (SPACES)**: Ensure no extra spaces at the end of each line.
- **Tab check (TABS)**: Ensure spaces are used for indentation instead of tabs (Tab).
- **Spell check (CODESPELL)**: Check English spelling errors using the codespell tool.

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe, and inclusive environment for all participants:

- **Respect differences**: Respect different viewpoints and experiences, and embrace multiculturalism
- **Open mind**: Accept constructive criticism and keep learning and improving
- **Focus on contributions**: Focus on what benefits the community most and drive project development
- **Empathy**: Show empathy to other community members and help each other

## Questions

We warmly welcome every developer to actively participate in community discussions! We look forward to growing together with you:

- **Unresolved issues found**: Feel free to comment in the Issue to present your solution
- **Long-pending issues encountered**: Pre-check before solving is recommended to avoid duplicate work
- **Successfully solved your own reported issue**: Please also share your solution so that the community can learn and progress together

If you have any questions, you are always welcome to discuss them in the community. We look forward to your wonderful contributions!

## Appendix A: Local Static Check

The project uses [lintrunner](https://github.com/suo/lintrunner) for static checks. It supports running the same check items locally as in CI, including Python code style (Flake8, Ruff, PYFMT), C++ format (ClangFormat, ClangTidy), and spell check (Codespell).

### Install Dependencies

```bash
# Install lintrunner and uv (required by some linters)
pip install lintrunner
pip install uv
```

### Initialize (Run Once upon First Use or Update)

```bash
# Download the external binary tools required by lintrunner (clang-format, clang-tidy, etc.)
lintrunner init
```

### Run Static Checks

```bash
# Check the file delta of the current workspace changes and HEAD commits (workspace + HEAD)
lintrunner

# Run only specified check items
lintrunner --take FLAKE8,RUFF,PYFMT,SPACES,TABS,NEWLINE

# Automatically fix auto-fixable issues (formatter-class linters; skip PYREFLY if needed)
lintrunner --skip PYREFLY -a

# Check only the file delta of the current workspace changes
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
> | `CODESPELL`     | Spell check. If a word is a false positive, add it in lexicographic order to `tools/linter/dictionary.txt` and re-check |

For more execution commands, see the [lintrunner wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).

### About Cloud Code Checks in the Pipeline (codecheck_pre-commit)

The `codecheck_pre-commit` task in the PR pipeline is not the local pre-commit/lintrunner. It submits the PR to a cloud code check service that runs its own rule set (including security rules). Therefore, note the following two points:

1. **Passing local checks does not mean this task passes**: The cloud rule set is not distributed with the repository and cannot be pre-checked locally. For example, the security rule `G.EDV.04` prohibits the use of `shell=True` in `subprocess` (severity: critical), and this rule is not in the `.flake8`/lintrunner configuration.
2. **How to view failure details**: The "Details" link in the PR comment requires logging in to the pipeline platform. External contributors can locate the problem as follows: after logging in to GitCode, open the pipeline run page, enter the `codecheck_pre-commit` task log, find the line starting with `openlibing_url:`, and the `entryCheckDashCode` link in it is the check details page, where you can see the rule name, level, file, and line number.

After fixing the issues, push a new commit and comment `compile` under the PR to re-trigger the pipeline (available to the PR creator; `retry [task name]` and `stop` are also available). Note: commenting `compile` immediately after pushing may cause the pipeline to bind to an expired commit (with the prompt "the commitId for the pipeline run cannot be obtained", and all tasks staying in QUEUED for a long time). In this case, comment `stop`, wait about one minute, and then comment `compile` again to recover.

## Appendix B: Unit Tests and Functional Verification

### Unit Test Guide

- **Python**: [pytest](http://pytest.org/en/latest/) is recommended
- **C++**: [Googletest Primer](https://github.com/google/googletest/blob/main/docs/primer.md) is recommended

The design intent of a test case should be reflected by its comment name.

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
