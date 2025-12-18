# 报错信息分析说明

用户在模型训练异常时可能产生较多的错误信息，包括Python调用栈和异常信息、原生框架报错信息、torch\_npu错误码、CANN软件错误码以及plog日志等，本节主要对于定位错误来源进行分析指导。

> [!NOTE]
> - 定位报错的主要原则：关注第一条报错信息。  
> - CANN软件相关报错的详细介绍请参见《[CANN 故障处理](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/troubleshooting/troubleshooting_0001.html)》。

