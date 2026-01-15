# 调用样例

完成了算子适配开发后，即可实现C++ extensions的方式调用自定义算子。下文介绍注册算子开发过程以及算子适配开发过程。昇腾提供了add\_custom和matmul\_leakyrelu\_custom两个kernel算子适配样例供开发者参考，具体可查看[LINK](https://gitcode.com/ascend/op-plugin/tree/master/examples/cpp_extension)。

1.  完成自定义算子工程创建、算子开发及编译部署流程，具体可参考《[CANN Ascend C算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0002.html)》。
2.  下载示例代码。

    ```
    # 下载样例代码
    git clone https://gitcode.com/ascend/op-plugin.git
    cd op-plugin
    git checkout master
    # 进入代码目录
    cd examples/cpp_extension
    ```

3.  完成算子适配，具体可参考[适配开发](./adaptation_description_kernel.md)。
4.  执行如下命令编译并安装wheel包。

    ```Python
    python setup.py bdist_wheel
    cd dist
    pip install *.whl
    ```

5.  执行测试用例并查看验证结果。

    执行如下命令：

    ```Python
    cd test
    python test.py
    ```

    返回如下回显，表示执行成功。

    ```Python
    Ran 2 tests
     
    OK
    ```

