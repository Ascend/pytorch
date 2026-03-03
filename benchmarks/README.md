# Torchbenchmark

## 简介

为了评测图模式能力，pytorch社区在CI HUD中提供了图模式应用在torchbench仓库的多个模型的加速效果。参考社区的实现，benchmarks提供了对部分指定模型的支持。用户可按照此README进行NPU图模式测试。

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
    ```
    INPUT_TARBALLS:
        # index file for S3 storage of the input data
        - pytorch_stargan_inputs.tar.gz
        - LearningToPaint_inputs.tar.gz
        - speech_transformer_inputs.tar.gz
    ```

2. 下载数据集可以使用下方URL
    ```
    https://ossci-datasets.s3.amazonaws.com/torchbench/data/<TARBALL_NAME>
    ```
    例如，下载pytorch_stargan_inputs数据集，可以使用
    ```
    https://ossci-datasets.s3.amazonaws.com/torchbench/data/pytorch_stargan_inputs.tar.gz
    ```

3. 下载后的数据集放到指定目录下，以pytorch_stargan_inputs.tar.gz为例
    ```shell
    # 创建.data/目录
    cd ./benchmark/torchbenchmark/data/
    mkdir .data/

    # 将数据集移动到./benchmark/torchbenchmark/data/.data目录下
    cd ../../../
    cp your/path/to/dataset/pytorch_stargan_inputs.tar.gz ./benchmark/torchbenchmark/data/.data
    
    # 解压数据集
    cd ./benchmark/torchbenchmark/data/
    tar -xvzf ./benchmark/torchbenchmark/data/.data/pytorch_stargan_inputs.tar.gz
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
    ```
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