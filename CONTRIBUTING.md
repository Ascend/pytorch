# TorchNPU 贡献指南

感谢您考虑为 TorchNPU 做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都是非常宝贵的。

## 项目介绍

TorchNPU 是基于 Ascend NPU 的深度学习适配插件，针对华为昇腾 NPU 进行了深度优化适配。本项目提供与上游 PyTorch 的 API 兼容性，并充分发挥昇腾芯片的计算能力。

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
| `third_party/op-plugin`    | 算子插件：自定义算子实现、TorchNPU 算子覆盖                                               |
| `test/npu`                 | NPU 功能测试：设备管理、内存分配、算子测试                                                  |

## 贡献概览

### 贡献方式

我们热情期待您的加入！每一个贡献都是推动 TorchNPU 进步的重要力量：

- **反馈问题**：报告 Bug 或提交功能建议，帮助我们发现并解决问题
- **提交方案**：提出技术方案或架构设计，为项目演进提供方向指引
- **贡献代码**：提交代码修复或新功能实现，直接参与项目开发
- **完善文档**：改进文档或补充缺失内容，提升项目可读性
- **代码审查**：审查 Pull Request，帮助提升代码质量
- **协助社区**：解答他人问题、分享解决方案，帮助更多开发者
- **分享传播**：在博客、社交媒体上分享项目，扩大社区影响力

### 沟通渠道

我们为您提供多种沟通渠道，方便您参与社区互动：

- **[Issues](https://gitcode.com/Ascend/pytorch/issues)**：用于报告 Bug、提出功能建议
- **[Pull Requests](https://gitcode.com/Ascend/pytorch/pulls)**：用于代码审查和讨论
- **[SIG](https://gitcode.com/Ascend/community/tree/master/AscendForPyTorch/sigs)**：按技术领域划分的社区协作组织，负责代码评审和技术决策

### 贡献场景

提交 Issue 是参与贡献的第一步。在 [创建 Issue](https://gitcode.com/Ascend/pytorch/issues/create/choose) 时，请选择对应的模板：

#### Bug Report（缺陷报告）

发现 Bug 时，请使用 🐛 Bug report 模板提交 Issue，模板包含以下必填项：

- **环境信息**：操作系统、昇腾硬件信息、CANN 软件版本、已安装的对应软件版本
- **🐛 问题描述**：清晰简洁地描述 Bug 的具体现象与出现场景；若可复现，请提供**最小可复现示例**（剔除无关代码与业务依赖、可直接运行），并粘贴包含**完整堆栈（Full Traceback）**的报错信息

> ⚠️ 请注意脱敏：提交前移除 API 密钥、密码、私有地址、个人数据等敏感信息，必要时用 `<TOKEN>` 等占位符替代。

#### Feature Request（需求建议）

有新功能建议或性能优化想法时，请使用 ✨ Feature request 模板提交，模板包含以下字段：

- **💻 需求背景、当前现状、期望实现的功能内容、具体的设计方案、以及测试方案**（必填）：说明需求场景、当前无法满足的原因、提议的功能内容与设计思路，可附相关 Issue/PR 链接
- **替代方案**：描述你考虑过的其他实现方案
- **补充说明**：其他相关说明或截图

#### RFC（方案提案）

如有较大的架构变更或技术方案，请使用 📜 RFC（Request for Comments）模板发起 RFC，经社区讨论和评审后推进实施。RFC 模板为结构化文档，主要章节包括：

- **概述**：简介、动机、目标与非目标
- **用例分析**：场景用例、关键性能指标、安全隐私及 DFX 要求
- **方案设计**：总体方案、技术选型、功能与性能设计、安全隐私与 DFX 设计、编程与调用设计（含接口定义）
- **测试设计**：单元测试、集成测试、端到端测试
- **缺点和风险**：Breaking Change、性能回退、实现成本、兼容性等
- **现有技术**：参考其他项目/社区的类似设计
- **未解决问题**：待社区讨论决策的开放问题

## 贡献流程

为确保贡献顺利进行，请按照以下流程操作：

### 第一步：签署 CLA

在您第一次向 TorchNPU 社区提交代码之前，需要签署 CLA（Contributor License Agreement）。

- **个人贡献者**：请参考 [ICLA 在线文档](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/individual)
- **企业贡献者**：请参考 [CCLA 在线文档](https://clasign.osinfra.cn/sign-cla/690ca9ddf91c03dee6082ab1/employee)

### 第二步：提交 Issue

在开始开发之前，请先提交 Issue 描述您要解决的问题或建议的功能，以便社区了解和讨论。Issue 的详细格式要求请参考[贡献场景](#贡献场景)。

### 第三步：社区讨论

提交 Issue 后，请在社区中进行讨论，确认方案可行后再开始开发。

#### 认领任务

如果您要修复某个 Bug，请在对应 Issue 下评论 `/assign` 认领该任务，避免多人重复开发。

#### SIG 评审

以下类型的修改需要通过 SIG（Special Interest Group）机制进行社区评审：

- **Patch 替换**：对 TorchNPU 原生接口的 patch 替换
- **头文件宏更新**：新增或修改宏定义
- **API 接口变更**：新增、修改或删除公共 API
- **核心组件变更**：内存管理、设备管理等核心模块的修改
- **RFC 提案**：涉及重大架构变更或技术方案，需先发起 RFC 征集社区意见

SIG 是按技术领域划分的社区协作组织，负责代码评审和技术决策。详细的 SIG 列表请参考 [Ascend 社区 SIG 页面](https://gitcode.com/Ascend/community/tree/master/AscendForPyTorch/sigs)。涉及以上类型的修改，请在 PR 中通过 SIG 机制申报社区评审。

### 第四步：Fork 与克隆

1. **Fork 仓库**：在 GitCode 平台点击仓库右上角 "Fork" 按钮，在您的个人账号下创建一份仓库副本

2. **克隆到本地**：

   ```bash
   git clone https://gitcode.com/<your-username>/pytorch.git
   cd pytorch
   ```

3. **创建开发分支**：

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

### 第五步：代码开发

请遵循以下编码风格，使 TorchNPU 易于开发、审查和维护。

#### 编码指南

- **Python**：建议使用 [PEP 8 编码样式](https://pep8.org/)
- **C++**：建议使用 [Google C++ 编码指南](http://google.github.io/styleguide/cppguide.html)

#### 重构指南

我们鼓励开发人员重构代码以消除代码异味。所有的代码都应该符合编码风格和测试风格的需求。

#### 环境搭建

详细的安装指导（包括各版本依赖、环境要求等）请参考[昇腾社区安装指南](https://www.hiascend.com/document/detail/zh/Pytorch/master/installguide/ref/docs/zh/installation_guide/references/building_from_source.md)。

快速构建命令：

```bash
# 安装依赖并编译
bash ci/build.sh --python=3.10

# 针对指定的 PyTorch 版本构建（支持 2.13 及以上，可用版本以 version.txt 为准）
# --torch 的值即要构建的包版本，可带 post 号（如 2.13.0.post1 表示 2.13 主线的 post 构建）
# 要求环境中已安装对应 major.minor 的 PyTorch
bash ci/build.sh --python=3.10 --torch=2.14.0

# 或使用 CMake 手动编译
mkdir build && cd build
cmake ..
make -j$(nproc)
```

如需加快编译速度，项目已内置 Ninja、Mold 链接器、CCache 的自动检测逻辑，使用方法请参见[编译加速](docs/zh/installation_guide/references/build_acceleration.md)。

### 第六步：本地静态检查

完成代码开发后，请在本地运行静态检查，确保代码风格和质量符合要求。详细操作请参考 [附录 A：本地静态检查](#附录-a本地静态检查)。

### 使用 Clang 编译

使用 Clang 进行编译的方法，请参见[使用 Clang 编译](docs/zh/installation_guide/references/building_with_clang.md)。

### 第七步：本地功能验证

请编写并运行单元测试，确保代码功能正确。如有新增功能，需同步新增对应的测试用例。详细操作请参考 [附录 B：单元测试与功能验证](#附录-b单元测试与功能验证)。

### 第八步：提交 Pull Request

1. **推送代码到远程仓库**：

   ```bash
   git add .
   git status
   git commit -m "feat(module): 简要描述变更内容"
   git commit -s --amend  # 添加详细描述
   git push origin {new_branch_name}
   ```

   > **说明**：一个 PR 对应一个提交是本项目的推荐做法。若开发过程中产生了多个增量提交，请在推送前使用 `git rebase -i` 将它们压缩为单个提交，保持提交历史清晰。

2. **创建 Pull Request**

在 GitCode 上创建 Pull Request，根据 [PR 模板](./.gitcode/PULL_REQUEST_TEMPLATE.md) 完整填写：

- 合入来源
- 修改方案
- 资料变更
- 接口变更
- 功能验证
- CheckList

确认信息完整准确后提交 Pull Request，等待代码审查。

### 第九步：触发门禁检查

提交 PR 后，在 PR 评论区输入 `compile` 即可触发门禁检查。更多 Bot 命令说明请参考 [Ascend 社区 Bot 使用指南](https://gitcode.com/Ascend/infrastructure/blob/master/docs/robot/robot%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)。

门禁异常主要包含如下几种，请根据相关提示解决：

- **编译异常**：请检查代码编译失败的原因，解决问题后重新编译
- **静态检查异常**：请依照提示查找代码中的问题并解决（如代码风格、潜在 Bug 等）
- **UT 测试未通过**：请根据提示查找测试用例不通过项并检查原因

### 第十步：代码审查与合入

#### PR 合入要求

**合入检查清单**（详细要求参考 [PR 模板](./.gitcode/PULL_REQUEST_TEMPLATE.md)）：

- [ ] 代码编译通过
- [ ] 静态检查通过（CppLint、CppCheck 等）
- [ ] UT 测试用例通过
- [ ] 代码风格符合规范（PEP 8、Google C++ Style）
- [ ] 提交信息规范（符合 Conventional Commits）
- [ ] PR 标题正确使用类型标签（feat、fix、refactor、docs、test 等）
- [ ] 代码注释完备，正确记录错误日志
- [ ] 代码实现进行了返回值、空指针等校验

#### 代码审查流程

1. **Reviewer 审查**：对应模块的 reviewer 审查代码后，在 PR 评论区输入 `/lgtm` 表示认可
2. **Committer 审批**：对应模块的 committer 审查通过后，在 PR 评论区输入 `/approve` 表示同意合入
3. **自动合入**：PR 满足合入条件后，机器人会自动合入代码

> **提示**：完整的 reviewer 和 committer 名单可以在 PR 的第一条评论中点击对应链接查看。

## 进阶指南

### AI 辅助研发

TorchNPU 项目鼓励使用 AI 辅助研发与文档开发，以提升贡献效率。我们提供了昇腾官方的 agent-skills 仓库，其中包含一系列适用于昇腾生态的 AI Agent Skill 配置，可帮助您在开发中更好地利用 AI 编码助手。

- **agent-skills 仓库**：[https://gitcode.com/Ascend/agent-skills](https://gitcode.com/Ascend/agent-skills)
- 该仓库提供了昇腾芯片场景下常用的 Skill 模板和工具，可用于代码生成、问题诊断、性能分析等场景。
- 仓库中的 skills 持续更新中，同时欢迎贡献新的 Skill 或对现有 Skill 提出改进建议。

使用 AI 辅助研发时请注意：

- AI 生成的代码仍需人工审查，确保代码质量、安全性和正确性。
- 遵循项目的[编码指南](#第五步代码开发)和[附录 B：单元测试与功能验证](#附录-b单元测试与功能验证)。
- 提交的代码需通过门禁检查（编译、静态检查、UT 测试等）。

### 文档开发说明

#### 文档承载方式

本项目的文档采用 Markdown 格式，存放于仓库的 `docs/zh/` 目录下，随代码一同托管在 GitCode 平台。

> **注意**：文档承载在长稳版本的分支中，如 `v2.7.1`。如果您需要查看或修改文档，请切换到对应的长稳版本分支进行操作。

文档主要包含以下类目：

- **安装指南**（`installation_guide/`）：环境准备、源码编译、pip 安装等说明。
- **快速入门**（`quick_start/`）：快速上手教程。
- **原生 API 文档**（`native_apis/`）：各版本 TorchNPU 原生 API 支持情况。
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

## 社区准则

### 行为准则

我们致力于为所有参与者提供一个友好、安全、包容的环境：

- **尊重差异**：尊重不同的观点和经验，包容多元文化
- **开放心态**：接受建设性的批评，持续学习和进步
- **聚焦贡献**：关注对社区最有利的事情，推动项目发展
- **同理心**：对其他社区成员表示同理心，互帮互助

## 问题咨询

我们热烈欢迎每一位开发者积极参与社区讨论！期待与您共同成长：

- **发现未解决的问题**：欢迎在 Issue 中发表评论，展示您的解决方案
- **遇到长期未处理的问题**：建议在解决前进行预检查，避免重复工作
- **成功解决了自己报告的问题**：也请分享您的解决方案，让社区一起学习和进步

有任何疑问，随时欢迎在社区中交流讨论，期待您的精彩贡献！

## 附录 A：本地静态检查

项目使用 [lintrunner](https://github.com/suo/lintrunner) 进行静态检查，支持在本地运行与 CI 完全一致的检查项，包括 Python 代码风格（Flake8、Ruff、PYFMT）、C++ 格式（ClangFormat、ClangTidy）、拼写检查（Codespell）等。

### 安装依赖

```bash
# 安装 lintrunner 及 uv（部分 linter 需要）
pip install lintrunner
pip install uv
```

### 初始化（首次使用或更新时执行一次）

```bash
# 下载 lintrunner 所需的外部二进制工具（clang-format、clang-tidy 等）
lintrunner init
```

### 执行静态检查

```bash
# 检查当前工作区改动和HEAD提交的文件增量（工作区 + HEAD）
lintrunner

# 仅运行指定检查项
lintrunner --take FLAKE8,RUFF,PYFMT,SPACES,TABS,NEWLINE

# 自动修复可自动修复的问题（formatter 类 linter，如需忽略 PYREFLY）
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

## 附录 B：单元测试与功能验证

### 单元测试指南

- **Python**：建议使用 [pytest](http://pytest.org/en/latest/)
- **C++**：建议使用 [Googletest Primer](https://github.com/google/googletest/blob/main/docs/primer.md)

测试用例的设计意图应该通过它的注释名称来反映。

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
