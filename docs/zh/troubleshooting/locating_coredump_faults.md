# coredump问题定位

当应用程序运行过程中发生异常，即屏幕中显示了“Segmentation fault”字样，则表示出现了coredump。如果屏幕中没有显示“Segmentation fault”，没有Python堆栈，plog日志中也没有ERROR级别的日志，且有进程收到11号信号，则也是属于出现了coredump。

## 获取core文件

1.  设置生成coredump文件。

    执行如下命令查看当前设置：

    ```
    ulimit -c
    ```

    如果为unlimited，则表示生成coredump文件的大小设置为无限制，此时如果进程崩溃就会生成coredump文件。

    如果为0，发生异常时不会保存coredump文件，需要执行如下命令进行配置：

    ```
    ulimit -c unlimited
    ```

2.  设置coredump文件存储位置和名称。

    ```
    # 临时修改生成的coredump文件的名称
    sysctl -w kernel.core_pattern=core-%e.%p.%h.%t
    # 设置coredump生成目录
    echo "/{path_to_coredump}/core.%t.%e.%p" >/proc/sys/kernel/core_pattern
    ```
    命令中`%e.%p.%h.%t`或`%t.%e.%p`为文件名称的变量，`{path_to_coredump}`为生成目录，可自行设置。

3.  生成coredump文件。

    运行模型脚本，若模型报错、进程崩溃，即可生成coredump文件。

> [!NOTE]
> -   默认情况下，未开启core文件配置，不会生成coredump文件。
> -   如果为容器场景，请确保coredump文件产生的路径挂载在容器外，容器销毁后该文件仍然存在。

## gdb调试core文件

gdb命令行调试主要针对coredump场景，执行目录下会生成coredump文件，使用gdb调试该文件并打印堆栈，方法如下：

1.  参考[GDB官方文档](https://sourceware.org/gdb/)安装GDB。
2.  调试coredump文件。

    执行如下命令进入gdb模式，调试coredump文件。

    ```
    gdb python3 core*.*    # coredump文件名请根据用户自己的设置修改
    ```

    执行命令后，gdb工具会将发生异常的代码、其所在的函数、文件名和所在文件的行数打印到屏幕，方便定位问题。

    > [!NOTE]  
    > 调试环境需与生成coredump文件的环境一致。例如，容器场景需进入对应的容器内进行调试。

3.  通过如下命令查看堆栈。

    ```
    (gdb) bt        # 查看堆栈
    (gdb) thread apply all bt     # 查看所有进程的堆栈
    ```

    根据回显中的函数堆栈找到出错的位置。

    **图 1**  查看堆栈  
    ![](../figures/viewing_stack.png "查看堆栈")

