# 调用样例

完成了算子适配开发后，即可实现C++ extensions的方式调用自定义算子。下文以自定义Add算子为例，介绍注册算子开发过程以及算子适配开发过程。具体样例可参考[LINK](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch/CppExtensionInvocation)。

1.  完成自定义算子工程创建、算子开发及编译部署流程，具体可参考《[CANN Ascend C算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)》。
2.  下载示例代码。

    ```
    # 下载样例代码
    git clone https://gitee.com/ascend/samples.git
    # 进入代码目录
    cd samples/operator/ascendc/0_introduction/1_add_frameworklaunch/CppExtensionInvocation
    ```

3.  完成算子适配，具体可参考[适配开发](./adaptation_description_single.md)章节。
4.  执行如下命令编译并安装wheel包。

    ```Python
    python3 setup.py build bdist_wheel
    pip install dist/*.whl --force-reinstall
    ```

5.  执行测试用例并查看验证结果。
    -   eager模式：

        执行如下命令：

        ```
        python3 test_add_custom.py
        ```

        返回如下回显，表示执行成功。

        ```
        Ran 2 tests in 12.524s
         
        OK
        ```

    -   torch.compile模式：

        执行如下命令：

        ```
        python3 test_add_custom_graph.py
        ```

        返回如下回显，表示执行成功。

        ```
        Ran 1 test in 9.125s
         
        OK
        ```

