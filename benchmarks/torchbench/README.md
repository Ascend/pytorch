# Torchbenchmark

## 简介

为了评测图模式能力，pytorch社区在CI HUD中提供了图模式应用在torchbench仓库和huggingface仓库的多个模型的加速效果。参考社区的实现，benchmarks提供了对部分指定模型的支持。用户可按照此README进行NPU图模式测试。

## Torchbench

## 准备环境

1. 安装requirements.txt中的依赖包

    ```shell
    pip install -r requirements.txt
    ```

2. 下载pytorch/benchmark源码，并切换至指定commit id

    ```shell
    git clone  https://github.com/pytorch/benchmark.git --depth=1
    cd benchmark
    git remote set-branches origin '9910b31cc17d175a781412fd9ca6f18a4ee04610'
    git fetch --depth 1 origin 9910b31cc17d175a781412fd9ca6f18a4ee04610
    git checkout 9910b31cc17d175a781412fd9ca6f18a4ee04610
    cd ..
    ```

## 准备数据集

1. 部分模型需要少量数据集，名单如下

    ```text
    INPUT_TARBALLS:
        # index file for S3 storage of the input data
        - pytorch_stargan_inputs.tar.gz
        - LearningToPaint_inputs.tar.gz
        - speech_transformer_inputs.tar.gz
    ```

2. 下载数据集可以使用下方URL

    ```text
    # 直接访问，即可下载数据集
    https://ossci-datasets.s3.amazonaws.com/torchbench/data/pytorch_stargan_inputs.tar.gz
    https://ossci-datasets.s3.amazonaws.com/torchbench/data/LearningToPaint_inputs.tar.gz
    https://ossci-datasets.s3.amazonaws.com/torchbench/data/speech_transformer_inputs.tar.gz
    ```

3. 下载后的数据集放到任意目录下，然后使用环境变量`TORCHBENCH_DATA_PATH`指定数据集目录，运行测试脚本

    ```shell
    # 创建数据集./dataset目录
    mkdir ./dataset

    # 将数据集移动到./dataset目录下后，解压数据集
    tar -xvzf pytorch_stargan_inputs.tar.gz
    tar -xvzf LearningToPaint_inputs.tar.gz
    tar -xvzf speech_transformer_inputs.tar.gz

    # 使用环境变量，指定数据集目录
    export TORCHBENCH_DATA_PATH="./dataset"
    ```

## 运行测试

1. 精度和端到端总时间验证

    ```shell
    # 不使能--only，默认运行torchbench_models_list.txt目录下的所有模型
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --iterations 50

    # 使能--only，运行指定模型
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --only BERT_pytorch --iterations 50
    ```

    执行上述命令后，会在终端界面分别打印出模型执行（eager模式和图模式）的单步"端到端时间"、单步loss、"端到端时间"平均值以及eager模式和图模式精度比较的结果（pass_accuracy or fail_accuracy）

2. 编译总时间验证

    ```shell
    # 添加参数--dump-compile-time，会在模型测试结束后输出编译时间的测量结果
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --iterations 50 --dump-compile-time
    ```

    执行上述命令后，会在终端界面打印出图模式下的算子编译时间

3. 算子总时间验证

    算子时间验证需要开启profile工具，添加参数`--enable-profiler`，profile结果默认输出路径为`./profile`，用户指定输出路径可使用参数`--prof-output-path`。

    ```shell
    # 不指定profile输出路径
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --iterations 50 --enable-profiler

    # 指定profile输出路径
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --iterations 50 --enable-profiler --prof-output-path 'your/path/for/profile/output/'
    ```

    `./profile`下的目录结构示例如下，`step_trace_time.csv`文件中记录了模型执行（eager模式和图模式）的单步算子时间

    ```text
    ./profile
    ├── BERT_pytorch
    │   ├── compile
    │   │   └── localhost.localdomain_xxxx_ascend_pt
    │   │       └── ASCEND_PROFILER_OUTPUT
    |   |           └── step_trace_time.csv
    │   └── eager
    │       └── localhost.localdomain_xxxx_ascend_pt
    │           └── ASCEND_PROFILER_OUTPUT
    |               └── step_trace_time.csv
    | ······
    ```

4. NPU图模式后端指定

    当前NPU图模式后端通过`--npu-backend`参数指定，支持`mlir`、`dvm`、`akg`、`triton`（triton-ascend）三种模式，不显示指定会默认选择三种模式中端到端时间加速比最大的图模式后端，使用示例如下

    ```shell
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only BERT_pytorch --iterations 50
    ```

5. Aclgraph使能关闭与模式指定

    当前NPU图模式后端在静态shape的条件下默认开启aclgraph，如果需要关闭aclgraph，可添加参数`--disable-aclgraph`。示例如下

    ```shell
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only BERT_pytorch --iterations 50 --disable-aclgraph
    ```

    aclgraph支持两种模式，分别是`max-autotune`和`reduce-overhead`，默认是`max-autotune`模式。用户可通过参数`--aclgraph-mode`指定aclgraph模式。示例如下

    ```shell
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only BERT_pytorch --iterations 50 --aclgraph-mode reduce-overhead
    ```

6. 动态shape指定

    当前NPU图模式后端默认执行静态shape，用户可用过参数`--dynamic-shapes`开启动态shape模式，或者可以通过参数`--dynamic-batch-only`仅针对输入数据的batch轴开启动态shape。示例如下

    ```shell
    # 开启动态shape模式
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only BERT_pytorch --iterations 50 --dynamic-shapes

    # 仅针对输入数据的batch轴开启动态shape
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only BERT_pytorch --iterations 50 --dynamic-batch-only
    ```

7. eager与aot\_eager精度对比定位

    ```shell
    # 不使能--only，默认运行torchbench_models_list.txt目录下的所有模型
    python3 torchbench.py --precision-checker --train --float32 --backend aot_eager

    # 使能--only，运行指定模型
    python3 torchbench.py --precision-checker --train --float32 --backend aot_eager --only BERT_pytorch
    ```

    执行上述命令之后，会在终端界面打印出网络模型各模块的精度对比结果。当前工具仅支持eager与aot_eager模式对比。

## 结果展示

1. 执行下述命令，测试精度、编译时间、端到端总时间，并保存日志

    ```shell
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --iterations 50 --dump-compile-time 2>&1 | tee models.log
    ```

2. 执行下述命令，打开profile，获取算子总时间

    ```shell
    python3 torchbench.py --accuracy --cold-start-latency --train --float32 --backend inductor --iterations 50 --enable-profiler
    ```

3. 执行下述命令，获得模型的测试结果，默认输出到`./analysis.xlsx`文件中

    ```shell
    python3 extract_log.py --log_file models.log
    ```

4. `./analysis.xlsx`文件的基本内容如下，展示了模型名称，图模式与eager模式的精度比较结果，算子编译时间（ms），eager模式的端到端时间（ms），图模式的端到端时间（ms），端到端时间加速比，eager模式的算子时间（ms），图模式的算子时间（ms），算子时间加速比

| model_name | accuracy | op_compile_time | eager_E2E_avg_time | compile_E2E_avg_time | E2E_speed_up_rate | eager_OP_avg_time | compile_OP_avg_time | OP_speed_up_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BERT_pytorch | pass_accuracy | 41055.906 | 48.49 | 20.26 | 2.393385982 | 20.399543 | 17.4955115 | 1.16598723 |

## HuggingFace

## 准备环境

1. 安装requirements.txt中的依赖包

    ```shell
    pip install -r requirements.txt
    ```

## 运行测试

1. NPU图模式后端指定

    当前NPU图模式后端通过`--npu-backend`参数指定，支持`mlir`、`dvm`、`triton`（triton-ascend）三种模式，使用示例如下

    ```shell
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only AlbertForMaskedLM --iterations 50
    ```

2. 精度和端到端总时间验证

    ```shell
    # 不使能--only，默认运行huggingface_models_list.txt目录下的所有模型
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --iterations 50

    # 使能--only，运行指定模型
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only AlbertForMaskedLM --iterations 50
    ```

    执行上述命令后，会在终端界面分别打印出模型执行（eager模式和图模式）的单步"端到端时间"、单步loss、"端到端时间"平均值以及eager模式和图模式精度比较的结果（pass_accuracy or fail_accuracy）

3. 编译总时间验证

    ```shell
    # 添加参数--dump-compile-time，会在模型测试结束后输出编译时间的测量结果
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --iterations 50 --dump-compile-time
    ```

    执行上述命令后，会在终端界面打印出图模式下的算子编译时间。

4. 算子总时间验证

    算子时间验证需要开启profile工具，添加参数`--enable-profiler`，profile结果默认输出路径为`./profile`，用户指定输出路径可使用参数`--prof-output-path`。

    ```shell
    # 不指定profile输出路径
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --iterations 50 --enable-profiler

    # 指定profile输出路径
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --iterations 50 --enable-profiler --prof-output-path 'your/path/for/profile/output/'
    ```

    `./profile`下的目录结构示例如下，`step_trace_time.csv`文件中记录了模型执行（eager模式和图模式）的单步算子时间

    ```text
    ./profile
    ├── AlbertForMaskedLM
    │   ├── compile
    │   │   └── localhost.localdomain_xxxx_ascend_pt
    │   │       └── ASCEND_PROFILER_OUTPUT
    |   |           └── step_trace_time.csv
    │   └── eager
    │       └── localhost.localdomain_xxxx_ascend_pt
    │           └── ASCEND_PROFILER_OUTPUT
    |               └── step_trace_time.csv
    | ······
    ```

5. Aclgraph使能关闭与模式指定

    当前NPU图模式后端在静态shape的条件下默认开启aclgraph，如果需要关闭aclgraph，可添加参数`--disable-aclgraph`。示例如下

    ```shell
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only AlbertForMaskedLM --iterations 50 --disable-aclgraph
    ```

    aclgraph支持两种模式，分别是`max-autotune`和`reduce-overhead`，默认是`max-autotune`模式。用户可通过参数`--aclgraph-mode`指定aclgraph模式。示例如下

    ```shell
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --only AlbertForMaskedLM --iterations 50 --aclgraph-mode reduce-overhead
    ```

## 结果展示

1. 执行下述命令，测试精度、编译时间、端到端总时间，并保存日志

    ```shell
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --iterations 50 --dump-compile-time 2>&1 | tee models.log
    ```

2. 执行下述命令，打开profile，获取算子总时间

    ```shell
    python3 huggingface.py --accuracy --cold-start-latency --train --float32 --backend inductor --npu-backend mlir --iterations 50 --enable-profiler
    ```

3. 执行下述命令，获得模型的测试结果，默认输出到`./analysis.xlsx`文件中

    ```shell
    python3 extract_log.py --log_file models.log
    ```

4. `./analysis.xlsx`文件的基本内容如下，展示了模型名称，图模式与eager模式的精度比较结果，算子编译时间（ms），eager模式的端到端时间（ms），图模式的端到端时间（ms），端到端时间加速比，eager模式的算子时间（ms），图模式的算子时间（ms），算子时间加速比

| model_name | accuracy | op_compile_time | eager_E2E_avg_time | compile_E2E_avg_time | E2E_speed_up_rate | eager_OP_avg_time | compile_OP_avg_time | OP_speed_up_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AlbertForMaskedLM | pass_accuracy | 58395.435 | 150.77 | 139.67 | 1.07794325 | 147.834943 | 136.9055115 | 1.0798731 |
