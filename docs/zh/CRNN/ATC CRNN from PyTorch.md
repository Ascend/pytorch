# CRNN
-   [交付件基本信息](#交付件基本信息.md)
-   [概述](#概述.md)
-   [推理环境准备](#推理环境准备.md)
-   [快速上手](#快速上手.md)
-   [版本说明](#版本说明.md)
<h2 id="交付件基本信息.md">交付件基本信息</h2>

发布者（Publisher）：Huawei

应用领域（Application Domain）：OCR

版本（Version）：1.2

修改时间（Modified）：2021.04.15

大小（Size）：34MB, 17MB

框架（Framework）：PyTorch\_1.6.0

模型格式（Model Format）：onnx, om

精度（Precision）：FP16

处理器（Processor）：昇腾310

应用级别（Categories）：Official

描述（Description）：基于PyTorch CRNN模型，使用ATC工具转换得到的，可以在昇腾AI设备上运行的离线模型

<h2 id="概述.md">概述</h2>

CRNN网络是一种用于在图像画面上识别和分辨出字符串的网络模型，模型的输入为一张3通道RGB图片，尺寸为\(32, 100, 3\)。CRNN的采取backbone网络提取图像特征。由于backbone部分得到的Feature Map是不能直接进行RNN预测的，我们需要进行一步Map-to-Seq操作，CNN Feature Map的N\*C\*H\*W转化为\(NH\)\*W\*C，同时依然将W维作为RNN中的时序维度。这之后使用BLSTM结构对其进行预测，并最终使用CTC找到最高概率的序列并输出。

-   参考论文：[Baoguang Shi, Xiang Bai, Cong Yao "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" arXiv:1507.05717](https://arxiv.org/abs/1507.05717)

-   参考实现：

    ```
    url=https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec.git
    branch=master
    commit_id=90c83db3f06d364c4abd115825868641b95f6181
    ```


-   适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/modelzoo.git
    branch=master
    commit_id=9706793659e5788fcd0b8800f4aec34918de7c02
    code_path=built-in/PyTorch/Official/cv/image_classification/CRNN_for_PyTorch
    ```


通过Git获取对应commit\_id的代码方法如下：

```
git clone {repository_url}        # 克隆仓库的代码
cd {repository_name}              # 切换到模型的代码仓目录
git checkout {branch}             # 切换到对应分支
git reset --hard {commit_id}      # 代码设置到对应的commit_id
cd {code_path}                    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

## 输入输出数据<a name="section1677131162117"></a>

-   输入数据

    <a name="table7424170132318"></a>
    <table><thead align="left"><tr id="row245619010232"><th class="cellrowborder" valign="top" width="16.85%" id="mcps1.1.5.1.1"><p id="p12217164043818"><a name="p12217164043818"></a><a name="p12217164043818"></a><span id="ph8217114015387"><a name="ph8217114015387"></a><a name="ph8217114015387"></a>输入数据</span></p>
    </th>
    <th class="cellrowborder" valign="top" width="26.63%" id="mcps1.1.5.1.2"><p id="p10217154016381"><a name="p10217154016381"></a><a name="p10217154016381"></a><span id="ph721754033819"><a name="ph721754033819"></a><a name="ph721754033819"></a>大小</span></p>
    </th>
    <th class="cellrowborder" valign="top" width="26.31%" id="mcps1.1.5.1.3"><p id="p15217540103814"><a name="p15217540103814"></a><a name="p15217540103814"></a><span id="ph52178403387"><a name="ph52178403387"></a><a name="ph52178403387"></a>数据类型</span></p>
    </th>
    <th class="cellrowborder" valign="top" width="30.209999999999997%" id="mcps1.1.5.1.4"><p id="p112172040153811"><a name="p112172040153811"></a><a name="p112172040153811"></a><span id="ph52171840163812"><a name="ph52171840163812"></a><a name="ph52171840163812"></a>数据排布格式</span></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row345611014234"><td class="cellrowborder" valign="top" width="16.85%" headers="mcps1.1.5.1.1 "><p id="p99212034185613"><a name="p99212034185613"></a><a name="p99212034185613"></a>input</p>
    </td>
    <td class="cellrowborder" valign="top" width="26.63%" headers="mcps1.1.5.1.2 "><p id="p104564062317"><a name="p104564062317"></a><a name="p104564062317"></a>batchsize x 1 x 32 x 100</p>
    </td>
    <td class="cellrowborder" valign="top" width="26.31%" headers="mcps1.1.5.1.3 "><p id="p1456401236"><a name="p1456401236"></a><a name="p1456401236"></a>FLOAT32</p>
    </td>
    <td class="cellrowborder" valign="top" width="30.209999999999997%" headers="mcps1.1.5.1.4 "><p id="p1447015471315"><a name="p1447015471315"></a><a name="p1447015471315"></a>NCHW</p>
    </td>
    </tr>
    </tbody>
    </table>

-   输出数据

    <a name="table12507194019110"></a>
    <table><thead align="left"><tr id="row155081140913"><th class="cellrowborder" valign="top" width="16.85%" id="mcps1.1.5.1.1"><p id="p1673295793810"><a name="p1673295793810"></a><a name="p1673295793810"></a><span id="ph157328571389"><a name="ph157328571389"></a><a name="ph157328571389"></a>输出数据</span></p>
    </th>
    <th class="cellrowborder" valign="top" width="26.63%" id="mcps1.1.5.1.2"><p id="p137321657133818"><a name="p137321657133818"></a><a name="p137321657133818"></a><span id="ph147326579382"><a name="ph147326579382"></a><a name="ph147326579382"></a>大小</span></p>
    </th>
    <th class="cellrowborder" valign="top" width="26.31%" id="mcps1.1.5.1.3"><p id="p27321057143817"><a name="p27321057143817"></a><a name="p27321057143817"></a><span id="ph973295710384"><a name="ph973295710384"></a><a name="ph973295710384"></a>数据类型</span></p>
    </th>
    <th class="cellrowborder" valign="top" width="30.209999999999997%" id="mcps1.1.5.1.4"><p id="p8732105713814"><a name="p8732105713814"></a><a name="p8732105713814"></a><span id="ph3732185743818"><a name="ph3732185743818"></a><a name="ph3732185743818"></a>数据排布格式</span></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1250854016111"><td class="cellrowborder" valign="top" width="16.85%" headers="mcps1.1.5.1.1 "><p id="p1250834016112"><a name="p1250834016112"></a><a name="p1250834016112"></a>output1</p>
    </td>
    <td class="cellrowborder" valign="top" width="26.63%" headers="mcps1.1.5.1.2 "><p id="p150817406118"><a name="p150817406118"></a><a name="p150817406118"></a>26 x batchsize x 37</p>
    </td>
    <td class="cellrowborder" valign="top" width="26.31%" headers="mcps1.1.5.1.3 "><p id="p1050811401016"><a name="p1050811401016"></a><a name="p1050811401016"></a>FLOAT32</p>
    </td>
    <td class="cellrowborder" valign="top" width="30.209999999999997%" headers="mcps1.1.5.1.4 "><p id="p45090401619"><a name="p45090401619"></a><a name="p45090401619"></a>ND</p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="推理环境准备.md">推理环境准备</h2>

-   本样例配套的CANN版本为[3.2.0](https://www.hiascend.com/software/cann/commercial)  。
-   硬件环境和运行环境准备请参见《[CANN 软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade)》。
-   该模型需要以下依赖。

    **表 1**  依赖列表

    <a name="table681174220258"></a>
    <table><thead align="left"><tr id="row6811342102512"><th class="cellrowborder" valign="top" width="39.79%" id="mcps1.2.3.1.1"><p id="p39023462613"><a name="p39023462613"></a><a name="p39023462613"></a>依赖名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="60.209999999999994%" id="mcps1.2.3.1.2"><p id="p168144242520"><a name="p168144242520"></a><a name="p168144242520"></a>版本</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1881164217259"><td class="cellrowborder" valign="top" width="39.79%" headers="mcps1.2.3.1.1 "><p id="p1357310203218"><a name="p1357310203218"></a><a name="p1357310203218"></a>ONNX</p>
    </td>
    <td class="cellrowborder" valign="top" width="60.209999999999994%" headers="mcps1.2.3.1.2 "><p id="p17311499269"><a name="p17311499269"></a><a name="p17311499269"></a>1.6.0</p>
    </td>
    </tr>
    <tr id="row208112424254"><td class="cellrowborder" valign="top" width="39.79%" headers="mcps1.2.3.1.1 "><p id="p189133419262"><a name="p189133419262"></a><a name="p189133419262"></a>Pytorch</p>
    </td>
    <td class="cellrowborder" valign="top" width="60.209999999999994%" headers="mcps1.2.3.1.2 "><p id="p158119421254"><a name="p158119421254"></a><a name="p158119421254"></a>1.6.0</p>
    </td>
    </tr>
    <tr id="row988052516339"><td class="cellrowborder" valign="top" width="39.79%" headers="mcps1.2.3.1.1 "><p id="p13175271338"><a name="p13175271338"></a><a name="p13175271338"></a>opencv-python</p>
    </td>
    <td class="cellrowborder" valign="top" width="60.209999999999994%" headers="mcps1.2.3.1.2 "><p id="p288115252334"><a name="p288115252334"></a><a name="p288115252334"></a>4.5.1</p>
    </td>
    </tr>
    <tr id="row1882442172510"><td class="cellrowborder" valign="top" width="39.79%" headers="mcps1.2.3.1.1 "><p id="p1391133419264"><a name="p1391133419264"></a><a name="p1391133419264"></a>torchvision</p>
    </td>
    <td class="cellrowborder" valign="top" width="60.209999999999994%" headers="mcps1.2.3.1.2 "><p id="p38254292518"><a name="p38254292518"></a><a name="p38254292518"></a>0.7.0</p>
    </td>
    </tr>
    </tbody>
    </table>

    >![](figures/icon-note.gif) **说明：** 
    >请用户根据自己的运行环境自行安装所需依赖。


<h2 id="快速上手.md">快速上手</h2>

## 获取源码<a name="section05971153175817"></a>

1.  单击“立即下载”，下载源码包。
2.  上传源码包到服务器任意目录并解压（如：/home/HwHiAiUser）。

    ```
    ├── atc_crnn.sh                         //onnx模型转换om模型脚本
    ├── benchmark.aarch64                   //离线推理工具（适用ARM架构）
    ├── benchmark.x86_64                    //离线推理工具（适用x86架构）
    ├── crnn_final_bs1.om                   //用于推理的离线模型
    ├── crnn_npu_dy.onnx                    //onnx格式的模型文件
    ├── crnn.py                             //crnn模型文件
    ├── get_info.py                         //生成推理输入的数据集二进制info文件
    ├── gpu_checkpoint_12_acc_0.7923.pth    //训练后的权重文件
    ├── parse_testdata.py                   //数据集预处理脚本
    ├── postpossess_CRNN_pytorch.py         //benchmark验证推理结果脚本
    ├── pth2onnx.py                         //用于转换pth模型文件到onnx模型文件
    ├── acl_env_arm.sh                      //设置ACL环境变量（适用ARM架构）
    ├── acl_env_x86.sh                      //设置ACL环境变量（适用x86架构）
    ├── acl_net.py                          //acl接口文件
    ├── post_CRNN_pytorch_acl.py            //推理及验证精度脚本
    └── ReadMe.md
    ```

    >![](figures/icon-note.gif) **说明：** 
    >benchmark离线推理工具使用请参见《[CANN V100R020C20 推理benchmark工具用户指南](https://support.huawei.com/enterprise/zh/doc/EDOC1100180792)》。


## 准备数据集<a name="section77496151595"></a>

1.  获取原始数据集。

    本模型支持多种开源OCR mdb数据集（例如IIIT5K\_lmdb），请用户自行准备好图片数据集，IIIT5K\_lmdb验证集目录参考。

    ```
    ├── IIIT5K_lmdb           # 验证数据集
         ├── data.mdb         # 数据文件
         └── lock.mdb         # 锁文件
    ```

    上传数据集到服务器任意目录并解压（如：_/home/HwHiAiUser/dataset_）。

2.  数据预处理。

    OCR使用的数据集一般是Imdb或者mdb格式，昇腾ModelZoo PyTorch模型统一提供基于ACL接口编写的Benchmark工具，用于om模型推理，所以需要先把mdb格式的文件通过脚本处理成二进制文件。使用parse\_testdata.py脚本将数据集转换为二进制文件。

    1.  修改脚本参数。

        执行命令。

        ```
        vim parse_testdata.py
        ```

        修改脚本中test\_dir和output\_bin参数值。

        ```
        test_dir = '/home/HwHiAiUser/dataset/IIIT5K_lmdb' # 修改为实际数据实际存放路径
        output_bin = './input_bin/'     # 修改为二进制输出路径
        ```

    2.  执行parse\_testdata.py脚本。

        ```
        python3.7 parse_testdata.py
        ```

        执行成功后，二进制文件生成在_./input\_bin_文件夹下，标签数据label.txt生成在当前目录下。



## 模型推理<a name="section1039612712591"></a>

1.  模型转换。

    使用PyTorch将模型权重文件.pth转换为.onnx文件，再使用ATC工具将.onnx文件转为离线推理模型文件.om文件。

    1.  获取权重文件。

        从源码中获取gpu\_checkpoint\_12\_acc\_0.7923.pth文件。

    2.  导出onnx文件。

        pth2onnx.py脚本将.pth文件转换为.onnx文件，执行如下命令。

        ```
        python3.7 pth2onnx.py ./gpu_checkpoint_12_acc_0.7923.pth ./crnn_npu_dy.onnx
        ```

        第一个参数为输入权重文件路径，第二个参数为输出onnx文件路径。该脚本导出动态batch轴的onnx模型，方便部署，用户可以根据需要修改。

        运行成功后，在当前目录生成crnn\_npu\_dy.onnx模型文件。

        >![](figures/icon-notice.gif) **须知：** 
        >使用ATC工具将.onnx文件转换为.om文件，需要.onnx算子版本需为11。在pth2onnx.py脚本中torch.onnx.export方法中的输入参数opset\_version的值需为11，请勿修改。

    3.  使用ATC工具将ONNX模型转OM模型。
        1.  修改atc\_crnn.sh脚本，通过ATC工具使用脚本完成转换，具体的脚本示例如下。

            ```
            # 配置环境变量
            export PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/atc/bin:$PATH
            export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/atc/python/site-packages/:$PYTHONPATH
            export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/atc/lib64:$LD_LIBRARY_PATH
            export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
            export DUMP_GE_GRAPH=2
            export SLOG_PRINT_TO_STDOUT=1
            export REPEAT_TUNE=true
            
            # 使用atc工具进行模型转换
            /usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=$1 --auto_tune_mode="GA,RL" --framework=5 --output=$2 --input_format=NCHW --input_shape="actual_input_1:1,1,32,100" --log=info --soc_version=Ascend310
            ```

            参数说明：

            -   --model：为ONNX模型文件。
            -   --framework：5代表ONNX模型。
            -   --output：输出的OM模型。
            -   --input\_format：输入数据的格式。
            -   --input\_shape：输入数据的shape。
            -   --auto\_tune\_mode：使用auto tune工具。
            -   --log：日志级别。
            -   --soc\_version：处理器型号。

            >![](figures/icon-note.gif) **说明：** 
            >脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《CANN V100R020C20 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/doc/EDOC1100180777/6dfa6beb)》。
            >**Auto Tune**工具在“RL”模式需要安装TensorFlow框架。详细介绍请参见《CANN V100R020C20 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/doc/EDOC1100180777/6dfa6beb)》中
            >“Auto Tune工具使用指南”章节。

        2.  执行atc\_crnn.sh脚本，将.onnx文件转为离线推理模型文件.om文件。

            添加执行权限。

            ```
            chmod u+x atc_crnn.sh
            ```

            执行脚本。

            ```
            ./atc_crnn.sh ./crnn_npu_dy.onnx crnn_final_bs1
            ```

            第一个参数为待转换的onnx模型，第二个参数为输出om文件的名称。运行成功后生成crnn\_final\_bs1.om。



2. 开始推理验证。

   1. 设置环境变量。

      -   x86环境。  
          
          ```
          source acl_env_x86.sh
          ```
          
      -   arm环境。
          
          ```
          source acl_env_arm.sh 
          ```
      
      
      ​      请根据《[CANN 软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-upgrade)》手册中Ascend-cann-toolkit_\_\{version\}_\_linux-_\{arch\}_.run实际安装位置修改对应环境变量脚本。
      
   2. 执行推理并测试精度

      运行脚本post\_CRNN\_pytorch\_acl.py进行精度测试，精度会打印并保存在json文件中。运行post\_CRNN\_pytorch\_acl.py脚本前请确保acl\_net.py和该脚本在同一目录中。

      ```
   python3.7.5 post_CRNN_pytorch_acl.py --pre_dir ./input_bin --batchsize 1 --modelpath crnn_final_bs1.om --label label.txt --json_output_file result
      ```

      参数说明：

      - --pre\_dir：预处理脚本处理后的文件夹。
      - --batchsize：推理batchsize。
      - --modelpath：模型路径。
      - --label：标签文件。
      - --json\_output\_file：保存结果的json文件名称。
      
      >![](figures/icon-note.gif) **说明：** 
      >
      >精度测试时，暂不支持使用benchmark推理，该步骤使用脚本进行测试。