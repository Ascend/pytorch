# PyTorch 贡献指南

感谢您考虑为 PyTorch 做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都是非常宝贵的。

## 项目介绍

PyTorch 是基于 Ascend NPU 的深度学习框架发行版，针对华为昇腾 NPU 进行了深度优化适配。本项目提供与 PyTorch 官方的 API 兼容性，并充分发挥昇腾芯片的计算能力。

### 项目架构

```text
pytorch
├── docs/                           # 项目文档
├── ci/                             # CI 构建脚本
├── tools/                          # 开发工具
├── cmake/                          # CMake 配置
├── torch_npu/                      # NPU 核心适配模块
│   ├── csrc/                       # C++ 后端实现
│   ├── distributed/                # 分布式 Python 接口
│   ├── _inductor/                  # Inductor 后端适配
│   ├── dynamo/                     # Dynamo 编译器适配
│   ├── npu/                        # NPU Python 接口
│   ├── profiler/                   # 性能分析 Python 接口
│   ├── _afd/                       # AFD Python 接口
│   ├── _logging/                   # 日志模块 Python 接口
│   ├── asd/                        # 异步检测工具
│   ├── contrib/                    # 贡献的扩展模块
│   ├── onnx/                       # ONNX 适配
│   └── optim/                      # 优化器适配
├── third_party/                    # 第三方依赖
├── torchnpugen/                    # 代码生成工具
├── examples/                       # 示例代码
└── test/                           # 测试用例
```

### 核心模块说明

| 模块                         | 说明                                                                       |
|----------------------------|--------------------------------------------------------------------------|
| `torch_npu/csrc/core/npu`  | NPU 核心组件：事件管理(NPUEvent)、流管理(NPUStream)、图执行(NPUGraph)、设备守卫(NPUGuard)、内存管理 |
| `torch_npu/csrc/aten`      | ATen 算子 NPU 后端：算子注册、调度、实现适配                                              |
| `torch_npu/csrc/framework` | 算子命令框架：OpCommand、Kernel 调度、算子构建器                                         |
| `torch_npu/npu/aclnn`      | ACLNN 算子 Python 接口：AscendCL NPU 算子库封装                                    |
| `torch_npu/npu/amp`        | 自动混合精度：GradScaler、FP16/BF16 支持                                           |
| `torchnpugen`              | 代码生成工具：自动微分代码生成、代码模板                                                     |
| `examples`                 | 示例代码：分布式通信、模型推理、ResNet 示例                                                |
| `third_party/op-plugin`    | 算子插件：自定义算子实现、PyTorch 算子覆盖                                                |
| `test/npu`                 | NPU 功能测试：设备管理、内存分配、算子测试                                                  |

## 贡献方式

我们热情期待您的加入！每一个贡献都是推动 PyTorch 进步的重要力量：

- **反馈问题**：报告 Bug 或提交功能建议，帮助我们发现并解决问题
- **贡献代码**：提交代码修复或新功能实现，直接参与项目开发
- **完善文档**：改进文档或补充缺失内容，提升项目可读性
- **代码审查**：审查 Pull Request，帮助提升代码质量
- **分享传播**：在博客、社交媒体上分享项目，给仓库点个 ⭐

## 贡献场景

本项目热烈欢迎各种形式的贡献，期待您的参与！

### 一、需求与功能建议

如果您有新功能建议或性能优化想法，我们热情邀请您提交 Issue 与社区深入讨论。

**Issue 类型**：需求/功能建议

**需要包含的内容**：

- **功能背景**：该功能解决什么问题、能为用户带来什么价值
- **功能描述**：详细描述建议的功能
- **设计方案**：技术思路、关键模块设计、上下游组件关系
- **预期收益**：功能目标、性能指标、精度表现

### 二、Bug 反馈与修复

如果您发现 Bug 或文档问题，我们真诚欢迎您的反馈和修复建议。

**Bug Report 格式**：

- **环境信息**：PyTorch 版本、OS、Python 版本、CANN 版本等
- **问题描述**：添加标签以便在问题仪表板上突出显示
- **复现步骤**：尽可能详细地描述如何重现问题
- **预期行为**：描述您预期发生的行为
- **给审稿人的特别说明**：如有任何特殊情况

**修复流程**：

1. 在 Issue 中找到对应的 Bug 描述
2. 评论 `/assign` 认领该任务
3. 创建分支进行修复
4. 提交 Pull Request

### 三、协助社区建设

如果您有能力解决他人提出的问题，我们热烈期待您在 Issue 中分享您的解决方案。

## 贡献流程

### 贡献者许可协议

在您第一次向 PyTorch 社区提交代码之前，需要签署 CLA。

对于个人贡献者，详细信息请参考 [ICLA 在线文档](https://www.mindspore.cn/icla)。

### 开发与测试

1. **Fork 仓库**：在 GitCode 平台点击仓库右上角 "Fork" 按钮，将仓库克隆到个人账户

2. **克隆到本地**：

   ```bash
   git clone https://gitcode.com/<your-username>/pytorch.git
   cd pytorch
   ```

3. **创建开发分支**：

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

4. **代码开发**：请遵循 **[代码规范](#代码规范)**

5. **代码测试**：运行测试确保代码功能正常

6. **门禁检查**：运行 CI 检查，确保代码通过编译、静态检查、UT 测试

7. **提交 Pull Request**：提交 PR 并等待代码审查

8. **社区评审**：如果涉及 patch、头文件宏、API 接口等更新，需提交社区评审

### 代码合入评审要求

以下类型的修改需要社区评审：

- **Patch 替换**：对 PyTorch 原生接口的 patch 替换
- **头文件宏更新**：新增或修改宏定义
- **API 接口变更**：新增、修改或删除公共 API
- **核心组件变更**：内存管理、设备管理等核心模块的修改

## 代码规范

请遵循这些风格，使 PyTorch 易于开发、审查和维护。

### 编码指南

- **Python**：建议使用 [PEP 8 编码样式](https://pep8.org/)
- **C++**：建议使用 [Google C++ 编码指南](http://google.github.io/styleguide/cppguide.html)

执行代码检查，可参照[本地静态检查](#本地静态检查)。

### 单元测试指南

- **Python**：建议使用 [pytest](http://pytest.org/en/latest/)
- **C++**：建议使用 [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md)

测试用例的设计意图应该通过它的注释名称来反映。

### 重构指南

我们鼓励开发人员重构代码以消除代码异味。所有的代码都应该符合编码风格和测试风格的需求。

## 实操指南

### 环境搭建与编译

**编译构建**：

```bash
# 安装依赖并编译
bash ci/build.sh --python=3.10

# 针对指定的 PyTorch 版本构建（支持 2.10.0 / 2.11.0 / 2.12.0）
# 要求环境中已安装对应版本的 PyTorch
bash ci/build.sh --python=3.10 --torch=2.10.0

# 或使用 CMake 手动编译
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 编译加速技巧

#### 使用 Ninja 构建

默认情况下，CMake 使用 Makefile 生成器。安装 Ninja 构建系统可以显著加快编译速度。

本项目 `setup.py` 会自动检测系统中是否安装了 Ninja：如果环境变量 `CMAKE_GENERATOR` 设置为 `ninja`，或者 `ninja` 命令在 `PATH` 中可用，将自动使用 Ninja 作为构建系统。

```bash
pip install ninja
```

安装 Ninja 后，编译即可自动生效，无需额外配置。如果之前已经编译过，安装 Ninja 后需要先执行一次清理：

```bash
python setup.py clean
```

#### 使用 Mold 链接器

在频繁修改单个文件并重新编译的开发循环中，链接时间会占据主导。大多数 Linux 发行版自带的系统链接器（GNU `ld`）速度较慢，使用更快的链接器可以显著改善构建体验。

本项目的 `CMakeLists.txt` 已内置链接器自动检测逻辑：优先检测 mold 链接器，若存在则自动启用（`-fuse-ld=mold`）。

```bash
sudo apt install mold
# 或从源码安装：https://github.com/rui314/mold
```

安装后重新编译即可自动生效。若需确认链接器是否正确启用，可检查编译输出中的链接选项是否包含 `-fuse-ld=mold`。

#### 使用 CCache

即使依赖跟踪基于文件修改时间，仍有许多场景下文件会被重复编译。使用 ccache 可以有效避免重复编译，节省大量时间。

本项目的 `CMakeLists.txt` 已内置 ccache 自动检测逻辑，安装 ccache 后即可自动启用。但建议根据自身环境调整 ccache 配置（如缓存目录、缓存大小、压缩等）以获得最佳效果：

```bash
sudo apt install ccache
# 或
sudo yum install ccache
```

验证 ccache 是否生效：连续执行两次完整编译，第二次应明显快于第一次。如果未生效，可检查 `build/CMakeCache.txt` 中的 `CMAKE_C_COMPILER_LAUNCHER` 和 `CMAKE_CXX_COMPILER_LAUNCHER` 变量是否包含 ccache：

```cmake
//C compiler launcher
CMAKE_C_COMPILER_LAUNCHER:PATH=/usr/bin/ccache

//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:PATH=/usr/bin/ccache
```

#### 仅编译所需目标

如果只需重新构建 `torch_npu.so`，可以在 build 目录下直接指定目标，避免全量构建：

```bash
cd build && ninja torch_npu
```

如果未安装 Ninja，将 `ninja` 替换为 `make` 即可。

### 本地静态检查

项目使用 [lintrunner](https://github.com/suo/lintrunner) 进行静态检查，支持在本地运行与 CI 完全一致的检查项，包括 Python 代码风格（Flake8、Ruff、PYFMT）、C++ 格式（ClangFormat、ClangTidy）、拼写检查（Codespell）等。

#### 安装依赖

```bash
# 安装 lintrunner 及 uv（部分 linter 需要）
pip install lintrunner
pip install uv
```

#### 初始化（首次使用或更新时执行一次）

```bash
# 下载 lintrunner 所需的外部二进制工具（clang-format、clang-tidy 等）
lintrunner init
```

#### 执行静态检查

```bash
# 检查当前工作区改动和HEAD提交的文件增量（工作区 + HEAD）
lintrunner

# 仅运行指定检查项
lintrunner --take FLAKE8,RUFF,PYFMT,SPACES,TABS,NEWLINE

# 自动修复可自动修复的问题（formatter 类 linter, 如忽略PYREFLY）
lintrunner --skip PYREFLY -a

# 仅检查当前工作区改动的文件增量
git diff --name-only HEAD | xargs lintrunner
```

> **提示**：`--take` 参数可指定只运行部分检查项，常用项如下：
>
> | 代码            | 说明                                                             |
> |---------------|----------------------------------------------------------------|
> | `FLAKE8`      | Python 语法与风格检查                                                 |
> | `RUFF`        | Python 快速 lint 与 import 排序                                     |
> | `PYFMT`       | Python 代码格式化（usort + ruff-format）                              |
> | `CLANGFORMAT` | C++ 代码格式化                                                      |
> | `CLANGTIDY`   | C++ 静态分析                                                       |
> | `SPACES`      | 行尾空格检查                                                         |
> | `TABS`        | Tab 字符检查                                                       |
> | `NEWLINE`     | 文件末尾换行检查                                                       |
> | `CODESPELL`   | 拼写检查, 如果是误报可以将误报词按照字典序添加至 `tools/linter/dictionary.txt` 后再重新检查 |

更多执行命令可参照[lintrunner wiki](https://github.com/pytorch/pytorch/wiki/lintrunner)。

### PR 合入要求

**合入检查清单**（详细要求参考 [PR 模板](./.gitcode/PULL_REQUEST_TEMPLATE.md)）：

- [ ] 代码编译通过
- [ ] 静态检查通过（CppLint、CppCheck 等）
- [ ] UT 测试用例通过
- [ ] 代码风格符合规范（PEP 8、Google C++ Style）
- [ ] 提交信息规范（符合 Conventional Commits）
- [ ] PR 标题正确使用类型标签（feat、fix、refactor、docs、test 等）
- [ ] 代码注释完备，正确记录错误日志
- [ ] 代码实现进行了返回值、空指针等校验

### 功能验证指导

**测试用例位置**：

- `test/npu/` - NPU 功能测试
- `test/nn/` - 网络层测试
- `test/distributed/` - 分布式测试
- `test/dynamo/` - 编译器测试

**运行测试**（详细说明参考 [测试文档](./test/README.md)）：

```bash
# 安装测试依赖
pip3 install -r test/requirements.txt

# 补全测试文件
cd test
bash get_synchronized_files.sh

# 运行单个测试文件
python test_autocast.py

# 或使用 run_test.py
python run_test.py -i test_autocast

# 运行指定用例
python test_autocast.py -v -k test_autocast_nn_fp32

# 运行全量 UT
cd ..
python ci/access_control_test.py --all
```

### 门禁异常处理

门禁异常主要包含如下几种，请根据相关提示解决：

- **编译异常**：请检查代码编译失败的原因，解决问题后重新编译
- **静态检查异常**：请依照提示查找代码中的问题并解决（如代码风格、潜在 Bug 等）
- **UT 测试未通过**：请根据提示查找测试用例不通过项并检查原因

### AI辅助研发

PyTorch NPU 项目鼓励使用 AI 辅助研发与文档开发，以提升贡献效率。我们提供了昇腾官方的 agent-skills 仓库，其中包含一系列适用于昇腾生态的 AI Agent Skill 配置，可帮助您在开发中更好地利用 AI 编码助手。

- **agent-skills 仓库**：[https://gitcode.com/Ascend/agent-skills](https://gitcode.com/Ascend/agent-skills)
- 该仓库提供了昇腾芯片场景下常用的 Skill 模板和工具，可用于代码生成、问题诊断、性能分析等场景。
- 仓库中的 skills 持续更新中，同时欢迎贡献新的 Skill 或对现有 Skill 提出改进建议。

使用 AI 辅助研发时请注意：

- AI 生成的代码仍需人工审查，确保代码质量、安全性和正确性。
- 遵循项目的[代码规范](#代码规范)和[单元测试指南](#单元测试指南)。
- 提交的代码需通过门禁检查（编译、静态检查、UT 测试等）。

### 文档开发说明

#### 文档承载方式

本项目的文档采用 Markdown 格式，存放于仓库的 `docs/zh/` 目录下，随代码一同托管在 GitCode 平台。

> **注意**：文档承载在长稳版本的分支中，如 `v2.7.1`。如果您需要查看或修改文档，请切换到对应的长稳版本分支进行操作。

文档主要包含以下类目：

- **安装指南**（`installation_guide/`）：环境准备、源码编译、pip 安装等说明。
- **快速入门**（`quick_start/`）：快速上手教程。
- **原生 API 文档**（`native_apis/`）：各版本 PyTorch 原生 API 支持情况。
- **框架特性指南**（`framework_feature_guide_pytorch/`）：NPU 图模式、Inductor、内存优化等特性说明。
- **环境变量参考**（`environment_variable_reference/`）：NPU 相关环境变量说明。
- **故障排除**（`troubleshooting/`）：常见问题及错误码分析。
- **安全声明**（`SECURITYNOTE.md`）：安全相关说明。
- **贡献指南**（`CONTRIBUTING.md`）：本文档。

#### 如何提交文档

文档的提交流程与代码提交一致，请参考[贡献流程](#贡献流程)：

1. Fork 仓库并在本地创建分支。
2. 在 `docs/zh/` 目录下新增或修改对应的 Markdown 文件。
3. 编写文档时注意：
   - 使用清晰、准确的中文表述。
   - 代码示例需确保可运行。
   - 遵循现有文档的格式和风格。
4. 提交 Pull Request，并在 PR 描述中说明文档变更内容。

#### CI 文档检查

提交文档的 Pull Request 后，CI 门禁会自动对变更的 Markdown 文件进行以下检查：

- **换行符检查（NEWLINE）**：确保文件末尾有且仅有一个换行符，且文件不包含多余的空行。
- **尾随空格检查（SPACES）**：确保每行末尾没有多余的空格。
- **制表符检查（TABS）**：确保文件中使用空格缩进而非制表符（Tab）。
- **拼写检查（CODESPELL）**：通过 codespell 工具检查英文拼写错误。

## 提交 Pull Request

1. **推送代码到远程仓库**：

   ```bash
   git add .
   git status
   git commit -m "Your commit title"
   git commit -s --amend  # 添加详细描述
   git push origin {new_branch_name}
   ```

2. **创建 Pull Request**

在 GitCode 上创建 Pull Request，根据 [PR 模板](./.gitcode/PULL_REQUEST_TEMPLATE.md) 完整填写：

- 合入来源
- 修改方案
- 资料变更
- 接口变更
- 功能验证
- CheckList

确认信息完整准确后提交 Pull Request，等待代码审查。

## 社区准则

### 行为准则

我们致力于为所有参与者提供一个友好、安全、包容的环境：

- **尊重差异**：尊重不同的观点和经验，包容多元文化
- **开放心态**：接受建设性的批评，持续学习和进步
- **聚焦贡献**：关注对社区最有利的事情，推动项目发展
- **同理心**：对其他社区成员表示同理心，互帮互助

### 沟通渠道

我们为您提供多种沟通渠道，方便您参与社区互动：

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**：用于报告 Bug、提出功能建议
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**：用于代码审查和讨论

### 问题咨询

我们热烈欢迎每一位开发者积极参与社区讨论！期待与您共同成长：

- **发现未解决的问题**：欢迎在 Issue 中发表评论，展示您的解决方案
- **遇到长期未处理的问题**：建议在解决前进行预检查，避免重复工作
- **成功解决了自己报告的问题**：也请分享您的解决方案，让社区一起学习和进步

有任何疑问，随时欢迎在社区中交流讨论，期待您的精彩贡献！
