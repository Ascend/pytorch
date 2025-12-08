# 安装说明

为使用PyTorch框架的开发者提供昇腾AI处理器的超强算力，昇腾开发Ascend Extension for PyTorch（即torch\_npu插件）用于适配PyTorch框架。

本文主要向用户介绍如何快速完成PyTorch框架、Ascend Extension for PyTorch（即torch\_npu插件）以及扩展模块的安装。

## 安装方案

本文档包含物理机、容器、虚拟机场景下，安装驱动、固件、<term>CANN</term>软件、PyTorch框架和torch\_npu插件的方案，部署架构如[图1](#安装方案图)所示。

**图 1**  安装方案<a id="安装方案图"></a>  
![](figures/installation_scheme.png "安装方案图")

## 硬件配套和支持的操作系统

**表 1**  产品硬件支持列表

|产品|是否支持（训练场景）|
|--|:-:|
|<term>Atlas A3 训练系列产品</term>|√|
|<term>Atlas A3 推理系列产品</term>|x|
|<term>Atlas A2 训练系列产品</term>|√|
|<term>Atlas A2 推理系列产品</term>|x|
|<term>Atlas 200I/500 A2 推理产品</term>|x|
|<term>Atlas 推理系列产品</term>|x|
|<term>Atlas 训练系列产品</term>|√|


-   各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。
-   各硬件产品对应虚拟机部署场景支持的操作系统请参考《CANN 软件安装指南》的“安装说明”章节（商用版）或“安装说明”章节（社区版）。
-   各硬件产品对应容器部署场景支持的操作系统请参考《CANN 软件安装指南》的“安装说明”章节（商用版）或“安装说明”章节（社区版）。

## 安装方式

本手册提供了二进制软件包、源码编译以及二进制软件包（abi1版本）安装方式，用户可根据实际需求自行选择安装PyTorch框架和torch\_npu插件的方式，不要求两者安装方式统一。

