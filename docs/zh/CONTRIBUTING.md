# PyTorch贡献指南

感谢您考虑为 PyTorch做出贡献!我们欢迎任何形式的贡献,包括错误修复、功能增强、文档改进等,甚至只是反馈。无论您是经验丰富的开发者还是第一次参与开源项目,您的帮助都是非常宝贵的。

您可以通过多种方式支持本项目:

- 通过[Issues](https://gitcode.com/Ascend/pytorch/issues)反馈问题。
- 建议或实现新功能。
- 改进或扩展文档。
- 审查Pull Request并协助其他贡献者。
- 传播项目:在博客文章、社交媒体上分享PyTorch,或给仓库点个⭐。

## 寻找可贡献的问题  

您可以通过查看[Issues列表](https://gitcode.com/Ascend/pytorch/issues)了解项目的发展计划和路线图。

## 贡献流程

- [贡献者许可协议](#贡献者许可协议)
- [开发与测试](#开发与测试)

### 贡献者许可协议

在您第一次向 PyTorch 社区提交代码之前,需要签署 CLA。

对于个人贡献者,详细信息请参考[ICLA 在线文档](https://www.mindspore.cn/icla)。

### 开发与测试

1. 先在GitCode平台点击仓库右上角"Fork"按钮,将仓库克隆到个人账户

2. 克隆到本地:

    ```bash
    git clone https://gitcode.com/<your-username>/pytorch.git
    cd pytorch

    ```

3. 在个人仓库进行代码开发请遵循[代码规范](#代码规范)。

4. 代码测试，请参见 [代码测试](https://gitcode.com/Ascend/pytorch/blob/master/test/README.md)。

5. [门禁异常处理](#门禁异常处理)。

6. [提交Pull Request](#提交Pull Request)。

7. [报告问题](#报告问题)。

#### 代码规范

请遵循这些风格,以使 PyTorch 易于开发、审查和维护。

- 编码指南

    请在PyTorch社区使用规统一的编码分格,python建议的编码风格是[PEP 8编码样式](https://pep8.org/),C++编码所建议的风格是  [Google C++编码指南](http://google.github.io/styleguide/cppguide.html)  。可以使用[CppLint](https://github.com/cpplint/cpplint),[CppCheck](http://cppcheck.sourceforge.net/),[CMakeLint](https://github.com/cmake-lint/cmake-lint),[CodeSpell](https://github.com/codespell-project/codespell),  [Lizard](http://www.lizard.ws/),[ShellCheck](https://github.com/koalaman/shellcheck)和[pylint](https://pylint.org/)检查代码的格式,建议在您的IDE中安装这些插件。

- 单元测试指南

    请在PyTorch社区使用统一的单元测试风格,  Python中建议的单元测试风格是[pytest](http://www.pytest.org/en/latest/),C++单元测试所建议的风格是  [Googletest Primer](https://github.com/google/googletest/blob/master/docs/primer.md)  。测试用例的设计意图应该通过它的注释名称来反映。

- 重构指南

    我们鼓励开发人员重构我们的代码以消除[代码异味](https://en.wikipedia.org/wiki/Code_smell)。所有的代码都应该符合编码风格和测试风格的需求,重构代码也不例外。当您收到警告时,您必须重构要合并的代码。

#### 门禁异常处理

门禁异常主要包含如下几种,请根据相关提示解决异常问题。

- 编译异常

    请检查代码编译失败的原因,解决问题后重新编译即可。

- 静态检查异常(代码Bug、代码漏洞、代码异味)

    请依照提示查找代码中的异常并解决。

- UT测试未通过

    请根据提示,查找测试用例不通过项并检查原因,解决后再测试。

#### 提交Pull Request<a id="提交Pull Request"></a>

1. 本地创建分支。

    为了避免多个分支之间的不一致,建议创建新的分支进行开发：

    ```bash
    git checkout -b {new_branch_name} origin/master
    ```

    以master分支为例,PyTorch可能会根据需要创建版本分支和下游开发分支,请先修复上游的bug。然后就可以随意更改代码了。

2. 将代码推送到远程仓库。

    更新代码后,您需要以正式的方式推送更新:

    ```bash
    git add .
    git status # Check the update status
    git commit -m "Your commit title"
    git commit -s --amend #Add the concrete description of your commit
    git push origin {new_branch_name}
    ```

3. 创建Pull Request

   在GitCode上创建Pull Request
   根据`.gitcode/PULL_REQUEST_TEMPLATE.md`中的规范模板,完整填写:
   - 合入来源
   - 修改方案
   - 资料变更
   - 接口变更
   - 功能验证
   - CheckList
   确认信息完整准确后提交Pull Request,等待代码审查
   
#### 报告问题

为项目做出贡献的一个好方法是在遇到问题时发送详细报告。我们总是很感激写得很好、彻底的错误报告,并会由此感谢您！

报告问题时,请参考以下格式:

- 您使用的是什么版本的环境 (pytorch、os、python 等)?
- 这是错误报告还是功能请求?
- 什么样的问题,添加标签以在问题仪表板上突出显示。
- 发生了什么?
- 您预计会发生什么?
- 如何重现它?(尽可能最小和精确。)
- 给审稿人的特别说明?

问题咨询:

- 如果您发现一个未解决的问题,而这正是您要解决的问题,请对该问题发表一些评论,告诉其他人您将负责它。
- 如果问题已打开一段时间,建议贡献者在解决该问题之前进行预检查。
- 如果您解决了自己报告的问题,则还需要在关闭该问题之前让其他人知道。

## 社区准则

### 行为准则

   我们致力于为所有参与者提供一个友好、安全和包容的环境。参与本项目即表示您同意:

   - 尊重不同的观点和经验
   - 接受建设性的批评
   - 关注对社区最有利的事情
   - 对其他社区成员表示同理心

### 沟通渠道

- **Issues**:用于报告Bug、提出功能建议和讨论技术问题
- **Pull Requests**:用于代码审查和讨论具体实现
