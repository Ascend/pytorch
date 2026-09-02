# TorchNPU Contribution Guide

Thank you for considering contributing to TorchNPU! We welcome contributions in any form, including bug fixes, feature enhancements, documentation improvements, and more. Whether you're an experienced developer or a first-time open source project, your help is invaluable.

## Project Introduction

TorchNPU is a deep learning adaptation plug-in based on Ascend NPU. It is optimized and adapted to Huawei Ascend NPU. This project provides API compatibility with the upstream PyTorch and fully utilizes the computing capability of the Ascend chip.

### Project Architecture

```text
pytorch
├── docs/                           #Project Documentation
├── ci/                             #Script for building the CI.
├── tools/                          #Development tool
├── cmake/                          #CMake Configuration
├── torch_npu/                      #NPU Core Adaptation Module
│   ├── csrc/                       #C++ backend implementation
│   ├── distributed/                #Distributed Python interface
│   ├── _inductor/                  #Inductor backend adaptation
│   ├── dynamo/                     #Dynamo Compiler Adaptation
│   ├── npu/                        #NPU Python interface
│   ├── profiler/                   #Performance Analysis Python Interface
│   ├── _afd/                       #AFD Python Interface
│   ├── _logging/                   #Python interface of the log module
│   ├── asd/                        #Asynchronous detection tool
│   ├── contrib/                    #Extended Modules Contributed
│   ├── onnx/                       #ONNX adaptation
│   └── optim/                      #Optimizer adaptation
├── third_party/                    #Third-Party Dependency
├── torchnpugen/                    #Code generation tool
├── examples/                       #Sample Code
└── test/                           #Test Case
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
| `third_party/op-plugin`    | Operator plug-in: custom operator implementation and PyTorch operator coverage                                                                            |
| `test/npu`                 | NPU function test: device management, memory allocation, and operator test                                                                                |

## Contribution Mode

We look forward to your joining! Each contribution is an important driving force behind TorchNPU's progress:

- **Feedback on issues**: Report bugs or submit feature suggestions to help us find and solve problems
- **Contribute code**: Submit code fixing or new function implementation, and directly participate in project development.
- **Complete the document**: Improve the document or supplement the missing content to improve the project readability.
- **Code review**: Review Pull requests to help improve code quality.
- **Share and communicate**: Share projects on blogs and social media, and point to the warehouse.

## Contribution Scenario

This project warmly welcomes all forms of contributions and looks forward to your participation!

### 1. Requirements and Function Suggestions

If you have new feature suggestions or performance optimization ideas, we invite you to submit an issue for further discussion in the community.

**Issue type: requirement/function suggestion**

**Contents to be included:**

- **Function background**: What problems can this function solve and what benefits can it bring to users?
- **Function description**: Describe the recommended functions in detail.
- **Design solution**: technical idea, key module design, and relationship between upstream and downstream components.
- **Expected benefits**: function objectives, performance indicators, and precision performance.

### 2. Bug feedback and fixing

If you find a bug or documentation problem, we sincerely welcome your feedback and suggestions for fixing it.

**Bug report format:**

- **Environment information**: PyTorch version, OS, Python version, and CANN version.
- **Problem Description**: Add a tag to highlight on the problem dashboard.
- **Reproduction procedure**: Describe how to reproduce the problem in detail.
- **Expected Behavior**: Describe the behavior you expect to occur.
- **Special Note to Reviewer**: If there are any special circumstances.

**Repair process:**

1. Find the corresponding bug description in the Issue.
2. Comment on the`/assign`Claim this task
3. Creating a Branch for Repair
4. Submit a Pull Request.

### 3. Assistance in community building

If you have the ability to solve the problem others have raised, we look forward to sharing your solution in the Issue.

## Contribution process

### Contributor License Agreement

Before you submit your code to the Ascend for PyTorch community for the first time, you need to sign [CLA](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/individual).

### Development and test

1. **Fork repository**: On the GitCode platform, click Fork in the upper right corner of the repository to clone the repository to the personal account.
2. **Clone to local:**
    
    ```bash
    git clone https://gitcode.com/<your-username>/pytorch.git
    cd pytorch
    ```

3. **To create a development branch:**
    
    ```bash
    git checkout -b {new_branch_name} origin/master
    ```

4. **Code development**: Please follow the [Code Specifications](#code-specifications).
5. **Code test**: Run tests to ensure that the code functions properly.
6. **Access control check**: Run the CI check to ensure that the code passes the compilation, static check, and UT test.
7. **Submit a Pull Request**: Submit a PR and wait for the code review.
8. **Community review**: If patches, header file macros, and API interface updates are involved, submit the updates to the community for review.

### Code Incorporation Review Requirements

The following types of modifications require community review:

- **Patch replacement**: Replace the patch of the PyTorch native interface.
- **Header file macro update**: Add or modify macro definitions.
- **API interface change**: adding, modifying, or deleting public APIs.
- **Core component changes**: Core modules such as memory management and device management are modified.

## Code Specifications

Follow these styles to make the Torch NPU easy to develop, review, and maintain.

### Coding Guide

- **Python**: recommended [PEP 8 Coding Style](https://pep8.org/).
- **C++**: recommended [Google C++ Coding Guide](http://google.github.io/styleguide/cppguide.html).

Check the code. For details, see the [Local static check](#local-static-check).

### Unit Test Guide

- **Python**: recommended [pytest](http://pytest.org/en/latest/).
- **C++**: recommended [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md).

The design intent of a test case should be reflected by its annotation name.

### Refactoring Guide

Developers are encouraged to refactor the code to eliminate code odors. All code should conform to the coding style and test style requirements.

## hands-on guide

### Environment setup and compilation

**Compilation and building:**

```bash
#Installing and Compiling the Dependency (If --torch is not specified, the installed version of PyTorch in the environment is used.)
bash ci/build.sh --python=3.10

#Build for the specified PyTorch version (supporting version 2.13 and later. The available version is the version.txt file.)
#--The value of torch indicates the version of the package to be built, which can contain the post number. (For example, 2.13.0.post1 indicates the post build of the 2.13 main line.)
#The PyTorch corresponding to major.minor has been installed.
bash ci/build.sh --python=3.10 --torch=2.14.0

#Or manually compile with CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Compilation Acceleration Skills

#### Build with Ninja

By default, CMake uses the Makefile generator. Installing the Ninja build system can significantly speed up compilation.

This project`setup.py`Automatically detects whether Ninja is installed on the system: If the environment variable`CMAKE_GENERATOR`Set to`ninja`, or`ninja`Ordered in the`PATH`available in, will automatically use Ninja as the build system.

```bash
pip install ninja
```

After the Ninja is installed, the compilation takes effect automatically. No additional configuration is required. If the Ninja has been compiled, perform the following operations after installing the Ninja:

```bash
python setup.py clean
```

#### Using Mold Linker

In a development cycle where individual files are frequently modified and recompiled, link time dominates. The system linker (GNU) that comes with most Linux distributions `ld` Slower, using a faster linker can significantly improve the build experience.

This project's`CMakeLists.txt`Built-in linker automatic detection logic: The mold linker is detected first, and if it exists, the mold linker is automatically enabled.`-fuse-ld=mold`.

```bash
sudo apt install mold
#Alternatively, install the software from the source code: https://github.com/rui314/mold
```

After the installation, recompile the software automatically takes effect. To confirm that the linker is enabled correctly, check whether the link option in the compilation output contains`-fuse-ld=mold`.

#### Using the CCache

Even if dependency tracking is based on file modification time, there are many scenarios where files are repeatedly compiled. Using ccache can effectively avoid repeated compilation and save a lot of time.

This project's`CMakeLists.txt`The built-in ccache automatic detection logic is automatically enabled after the ccache is installed. However, you are advised to adjust the ccache configuration (such as the cache directory, cache size, and compression) based on the environment to obtain the best results.

```bash
sudo apt install ccache
#or the
sudo yum install ccache
```

Check whether the ccache takes effect by performing two complete compilations in a row. The second compilation should be significantly faster than the first compilation. If not, check the`build/CMakeCache.txt`medium`CMAKE_C_COMPILER_LAUNCHER`And to the`CMAKE_CXX_COMPILER_LAUNCHER`Whether the variable contains ccache:

```cmake
//C compiler launcher
CMAKE_C_COMPILER_LAUNCHER:PATH=/usr/bin/ccache

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:PATH=/usr/bin/ccache
```

#### Compile only the required targets

If you just have to rebuild`torch_npu.so`, you can directly specify the target in the build directory to avoid full build.

```bash
cd build && ninja torch_npu
```

If Ninja is not installed, the`ninja`Replace with`make`That's it.

### Local static check

Project Use [lintrunner](https://github.com/suo/lintrunner) Perform static check. Check items that are the same as CI can be run locally, including Python code style (Flake8, Ruff, and PYFMT), C++ format (ClangFormat and ClangTidy), and spelling check (Codespell).

#### Installation Dependency

```bash
#Install lintrunner and UV (required by some linters).
pip install lintrunner
pip install uv
```

#### Initialize (once at first use or update)

```bash
#Download the external binary tools required by the lintrunner, such as clang-format and clang-tidy.
lintrunner init
```

#### Perform static checks

```bash
#Checks the current workspace changes and the file delta committed by HEAD (Workspace + HEAD)
lintrunner

#Run only specified checks
lintrunner --take FLAKE8,RUFF,PYFMT,SPACES,TABS,NEWLINE

#Automatically fix problems that can be automatically fixed (formatter class linter, e.g., ignore PYREFLY)
lintrunner --skip PYREFLY -a

#Check only file deltas for changes in the current workspace
git diff --name-only HEAD | xargs lintrunner
```

> **Hint:**`--take`You can specify the parameters to run only some check items. The common check items are as follows:
> 
> | Code          | Description                                                                                                                                     |
> | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
> | `FLAKE8`      | Python Syntax and Style Check                                                                                                                   |
> | `RUFF`        | Python fast lint and import sorting                                                                                                             |
> | `PYFMT`       | Python code formatting (usort + ruff-format)                                                                                                    |
> | `CLANGFORMAT` | C++ code formatting                                                                                                                             |
> | `CLANGTIDY`   | C++ static analysis                                                                                                                             |
> | `SPACES`      | Line end blank check                                                                                                                            |
> | `TABS`        | Tab character check                                                                                                                             |
> | `NEWLINE`     | Newline check at the end of a file                                                                                                              |
> | `CODESPELL`   | Spell check, and if it's a false positive, add the false positive words in lexicographic order to`tools/linter/dictionary.txt`And then re-check |

For more commands, see the [lintrunner wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).

### PR Incorporation Requirements

**Checklist** (For details, see the PR template):

- \[\] Code compilation is successful.
- \[\] Static check (such as CppLint and CppCheck) passed.
- \[\] UT test case passed.
- \[\] The code style complies with the specifications (PEP 8 and Google C++ Style).
- \[\] Specification for submission of information (conventional Commits compliance)
- \[\] PR Header Correct use of type labels (feat, fix, refactor, docs, test, etc.)
- \[\] The code comments are complete and error logs are recorded correctly.
- \[\] The code implementation verifies the return value and null pointer.

### Function Verification Guide

**Test Case Location:**

- `test/npu/`\- NPU function test
- `test/nn/`\- Network layer test
- `test/distributed/`\- Distributed test
- `test/dynamo/`\- Compiler test

**Run the test** (for details, see the Test Document):

```bash
#Installation test dependency
pip3 install -r test/requirements.txt

#Complete the test file.
cd test
bash get_synchronized_files.sh

#Running a Single Test File
python test_autocast.py

#or run_test.py
python run_test.py -i test_autocast

#Running a Specified Test Case
python test_autocast.py -v -k test_autocast_nn_fp32

#Running the full UT
cd ..
python ci/access_control_test.py --all
```

### Troubleshooting Access Control Exceptions

The access control exceptions include the following. Rectify the faults according to the related prompts:

- **Compilation error**: Check the cause of the code compilation failure, rectify the fault, and then recompile the code.
- **Static check exception**: Follow the prompt to locate and solve the problems (such as the code style and potential bugs) in the code.
- **If the UT test fails**: Locate the items that fail to pass the test and locate the cause.

### AI-assisted R&D

The TorchNPU project encourages AI-assisted R&D and document development to improve contribution efficiency. we provides the official Ascend agent-skills repository, which contains a series of AI agent skills applicable to the Ascend ecosystem, helping you better use AI coding assistants during development.

- **agent-skills repository:**https://gitcode.com/Ascend/agent-skills
- This repository provides skill templates and tools that are commonly used in the Ascend chip scenario, which can be used in scenarios such as code generation, problem diagnosis, and performance analysis.
- The skills in the warehouse are continuously updated. New skills are welcome to contribute or propose improvement suggestions for existing skills.

When using AI to assist R&D, pay attention to the following points:

- AI-generated code still needs to be manually reviewed to ensure code quality, security, and correctness.
- Following the project's [Code Specifications](#code-specifications)And to the [Unit Test Guide](#unit-test-guide).
- The submitted code must pass the access control check (compilation, static check, and UT test).

### Document Development Description

#### Document bearing mode

The documents of this project are in Markdown format and stored in the`docs/zh/`Directory, hosted with the code on the GitCode platform.

> **Note: The document is carried in the branch of the long-term stable version, such as**`v2.7.1`. If you need to view or modify the document, switch to the corresponding long-term stable version branch and perform the operation.

The document mainly includes the following categories:

- **Installation Guide** (`installation_guide/`): environment preparation, source code compilation, and pip installation.
- **Quick Start** (`quick_start/`): Quick Start Tutorial.
- **Native API Documentation** (`native_apis/`): PyTorch native APIs supported by each version.
- **Framework Feature Guide** (`framework_feature_guide_pytorch/`): describes the NPU diagram mode, Inductor, and memory optimization features.
- **Environment Variable Reference** (`environment_variable_reference/`): describes the environment variables related to the NPU.
- **Troubleshooting** (`troubleshooting/`): FAQs and error code analysis.
- **Safety Statement** (`SECURITYNOTE.md`): safety-related description.
- **Contribution Guide** (`CONTRIBUTING.md`): This document.

#### How to submit a document

The document submission process is the same as the code submission process. For details, see the [Contribution process](#contribution-process):

1. Fork repository and create a local branch.
2. In the `docs/zh/` Add or modify the corresponding Markdown file in the directory.
3. Note the following when writing the document:
    
    - Use clear and accurate Chinese expressions.
    - The code sample must be runable.
    - Follows the format and style of the existing document.
4. Submit a pull request and describe the document changes in the PR description.

#### Check the CI document

After the Pull Request is submitted, the CI access control system automatically checks the changed Markdown file as follows:

- **NEWLINE**: Ensure that the file contains only one newline character at the end of the file and that the file does not contain redundant blank lines.
- **Trailing Space Check (SPACES)**: Ensure that there are no extra spaces at the end of each line.
- **Tab Check (TABS)**: Ensure that the file is indented with spaces instead of tabs (Tab).
- **CodeSPELL**: Use the codespell tool to check spelling errors.

## Submit a Pull Request

1. **Push code to the remote repository:**
    
    ```bash
    git add .
    git status
    git commit -m "Your commit title"
    git commit -s --amend  #Added detailed description.
    git push origin {new_branch_name}
    ```

2. **Creating a Pull Request**

Create a Pull Request on the GitCode based on the PR Template Complete:

- Incorporation Source
- Modification scheme
- Document change
- Interface Changes
- Function Verification
- CheckList

Submit a Pull Request after confirming that the information is complete and correct, and wait for code review.

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe and inclusive environment for all participants:

- **Respect for differences**: respect for different views and experiences and inclusiveness of multiculturalism.
- **Open mind**: Accept constructive criticism and keep learning and improving.
- **Focus on contributions**: Focus on what is most beneficial to the community and promote project development.
- **Empathy**: Empathy for other community members and help each other.

### Communication channel

We provide you with multiple communication channels to engage in community interactions:

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**: used to report bugs and provide function suggestions.
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**: for code review and discussion.

### Question consultation

We warmly welcome every developer to participate in the community discussion! Looking forward to growing with you:

- **Unresolved issues found**: Feel free to comment on the Issue to demonstrate your solution.
- **Problems that have not been handled for a long time**: It is recommended to perform pre-check before solving the problems to avoid repeated work.
- **Successfully solved the problems you reported**: Please also share your solution so that the community can learn and progress together.

If you have any questions, please feel free to exchange and discuss in the community and look forward to your wonderful contributions!
