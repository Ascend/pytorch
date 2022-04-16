# PyTorch Operator Development Guide

- [PyTorch Operator Development Guide](#pytorch-operator-development-guide)
    - [Overview](#overview)
  - [Operator Development Process](#operator-development-process)
  - [Operator Development Preparations](#operator-development-preparations)
    - [Setting Up the Environment](#setting-up-the-environment)
    - [Looking Up Operators](#looking-up-operators)
  - [Operator Adaptation](#operator-adaptation)
    - [Prerequisites](#prerequisites)
    - [Obtaining the PyTorch Source Code](#obtaining-the-pytorch-source-code)
    - [Registering an Operator](#registering-an-operator)
      - [Overview](#overview-1)
      - [Registering an Operator for PyTorch 1.8.1](#registering-an-operator-for-pytorch-181)
        - [Registering an Operator](#registering-an-operator-1)
        - [Examples](#examples)
    - [Developing an Operator Adaptation Plugin](#developing-an-operator-adaptation-plugin)
      - [Overview](#overview-2)
      - [Introduction to the npu_native_functions.yaml File](#introduction-to-the-npu_native_functionsyaml-file)
      - [Adaptation Plugin Implementation](#adaptation-plugin-implementation)
      - [Examples](#examples-1)
    - [Compiling and Installing the PyTorch Plugin](#compiling-and-installing-the-pytorch-plugin)
      - [Compiling the PyTorch Plugin](#compiling-the-pytorch-plugin)
      - [Installing the PyTorch Plugin](#installing-the-pytorch-plugin)
  - [Operator Function Verification](#operator-function-verification)
    - [Overview](#overview-3)
      - [Introduction](#introduction)
      - [Test Cases and Records](#test-cases-and-records)
    - [Implementation](#implementation)
      - [Introduction](#introduction-1)
      - [Procedure](#procedure)
  - [FAQs](#faqs)
    - [Pillow==5.3.0 Installation Failed](#pillow530-installation-failed)
    - [pip3.7 install torchvision Installation Failed](#pip37-install-torchvision-installation-failed)
    - ["torch 1.5.0xxxx" and "torchvision" Do Not Match When torch-\*.whl Is Installed](#torch-150xxxx-and-torchvision-do-not-match-when-torch-whl-is-installed)
    - [How Do I View Test Run Logs?](#how-do-i-view-test-run-logs)
    - [Why Cannot the Custom TBE Operator Be Called?](#why-cannot-the-custom-tbe-operator-be-called)
    - [How Do I Determine Whether the TBE Operator Is Correctly Called for PyTorch Adaptation?](#how-do-i-determine-whether-the-tbe-operator-is-correctly-called-for-pytorch-adaptation)
    - [PyTorch Compilation Fails and the Message "error: ld returned 1 exit status" Is Displayed](#pytorch-compilation-fails-and-the-message-error-ld-returned-1-exit-status-is-displayed)
    - [PyTorch Compilation Fails and the Message "error: call of overload...." Is Displayed](#pytorch-compilation-fails-and-the-message-error-call-of-overload-is-displayed)
  - [Appendixes](#appendixes)
    - [Installing CMake](#installing-cmake)
    - [Exporting a Custom Operator](#exporting-a-custom-operator)



### Overview

To enable the PyTorch deep learning framework to run on Ascend AI Processors, you need to use Tensor Boost Engine (TBE) to customize the framework operators.

## Operator Development Process

PyTorch operator development includes TBE operator development and operator adaptation to the PyTorch framework.

1.  TBE operator development: If an operator on your network is incompatible with the  Ascend AI Software Stack, you need to develop a TBE operator and then adapt it to the PyTorch framework.

    For details about the TBE operator development process and methods, see the _CANN TBE Custom Operator Development Guide_.

2.  Operator adaptation to the PyTorch framework: If a TBE operator has been developed and is compatible with the Ascend AI Software Stack, you can directly adapt it to the PyTorch framework.

    The following figure shows the operator adaptation process in the PyTorch framework.

    **Figure 1** Operator adaptation process in the PyTorch framework<a name="en-us_topic_0000001105032530_fig1981905141719"></a>  
    ![](figures/operator-adaptation-process-in-the-pytorch-framework.png "operator-adaptation-process-in-the-pytorch-framework")


**Table 1** Description of the operator development process

<a name="en-us_topic_0000001105032530_en-us_topic_0228422310_table131083578318"></a>
<table><thead align="left"><tr id="en-us_topic_0000001105032530_en-us_topic_0228422310_row210905703113"><th class="cellrowborder" valign="top" width="6.811326262527976%" id="mcps1.2.5.1.1"><p id="en-us_topic_0000001105032530_en-us_topic_0228422310_p41091857143113"><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p41091857143113"></a><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p41091857143113"></a>No.</p>
</th>
<th class="cellrowborder" valign="top" width="17.865135740001946%" id="mcps1.2.5.1.2"><p id="en-us_topic_0000001105032530_en-us_topic_0228422310_p1710955713112"><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p1710955713112"></a><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p1710955713112"></a>Procedure</p>
</th>
<th class="cellrowborder" valign="top" width="55.55123090396029%" id="mcps1.2.5.1.3"><p id="en-us_topic_0000001105032530_en-us_topic_0228422310_p26391719183320"><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p26391719183320"></a><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p26391719183320"></a>Description</p>
</th>
<th class="cellrowborder" valign="top" width="19.772307093509777%" id="mcps1.2.5.1.4"><p id="en-us_topic_0000001105032530_en-us_topic_0228422310_p13109155719317"><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p13109155719317"></a><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p13109155719317"></a>Reference</p>
</th>
</tr>
</thead>
<tbody><tr id="en-us_topic_0000001105032530_row1381016124918"><td class="cellrowborder" valign="top" width="6.811326262527976%" headers="mcps1.2.5.1.1 "><p id="en-us_topic_0000001105032530_p1181015128915"><a name="en-us_topic_0000001105032530_p1181015128915"></a><a name="en-us_topic_0000001105032530_p1181015128915"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="17.865135740001946%" headers="mcps1.2.5.1.2 "><p id="en-us_topic_0000001105032530_p1881012121799"><a name="en-us_topic_0000001105032530_p1881012121799"></a><a name="en-us_topic_0000001105032530_p1881012121799"></a>Set up the environment.</p>
</td>
<td class="cellrowborder" valign="top" width="55.55123090396029%" headers="mcps1.2.5.1.3 "><p id="en-us_topic_0000001105032530_p1381018121891"><a name="en-us_topic_0000001105032530_p1381018121891"></a><a name="en-us_topic_0000001105032530_p1381018121891"></a>Set up the development and operating environments required for operator development, execution, and verification.</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="19.772307093509777%" headers="mcps1.2.5.1.4 "><p id="en-us_topic_0000001105032530_p1498205181013"><a name="en-us_topic_0000001105032530_p1498205181013"></a><a name="en-us_topic_0000001105032530_p1498205181013"></a><a href="#operator-development-preparationsmd">Operator Development Preparations</a></p>
</td>
</tr>
<tr id="en-us_topic_0000001105032530_row194671091290"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="en-us_topic_0000001105032530_p3467169594"><a name="en-us_topic_0000001105032530_p3467169594"></a><a name="en-us_topic_0000001105032530_p3467169594"></a>2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="en-us_topic_0000001105032530_p1346749990"><a name="en-us_topic_0000001105032530_p1346749990"></a><a name="en-us_topic_0000001105032530_p1346749990"></a>Look up operators.</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="en-us_topic_0000001105032530_p1767111499129"><a name="en-us_topic_0000001105032530_p1767111499129"></a><a name="en-us_topic_0000001105032530_p1767111499129"></a>View the list of supported TBE operators and list of operators adapted to PyTorch.</p>
<a name="en-us_topic_0000001105032530_ul03431749101318"></a><a name="en-us_topic_0000001105032530_ul03431749101318"></a><ul id="en-us_topic_0000001105032530_ul03431749101318"><li>List of operators supported by <span id="en-us_topic_0000001105032530_ph1748571571010"><a name="en-us_topic_0000001105032530_ph1748571571010"></a><a name="en-us_topic_0000001105032530_ph1748571571010"></a>Ascend AI Processors</span> and detailed specifications and constraints of the supported operators</li><li>List of operators adapted to PyTorch</li></ul>
</td>
</tr>
<tr id="en-us_topic_0000001105032530_en-us_topic_0228422310_row411025743119"><td class="cellrowborder" valign="top" width="6.811326262527976%" headers="mcps1.2.5.1.1 "><p id="en-us_topic_0000001105032530_p156991054952"><a name="en-us_topic_0000001105032530_p156991054952"></a><a name="en-us_topic_0000001105032530_p156991054952"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="17.865135740001946%" headers="mcps1.2.5.1.2 "><p id="en-us_topic_0000001105032530_en-us_topic_0228422310_p3110657203110"><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p3110657203110"></a><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p3110657203110"></a>Obtain the PyTorch source code.</p>
</td>
<td class="cellrowborder" valign="top" width="55.55123090396029%" headers="mcps1.2.5.1.3 "><p id="en-us_topic_0000001105032530_en-us_topic_0228422310_p381282212"><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p381282212"></a><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p381282212"></a>Obtain the PyTorch source code from the Ascend Community.</p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="19.772307093509777%" headers="mcps1.2.5.1.4 "><p id="en-us_topic_0000001105032530_p10679152717175"><a name="en-us_topic_0000001105032530_p10679152717175"></a><a name="en-us_topic_0000001105032530_p10679152717175"></a><a href="#operator-development-preparationsmd">Operator Adaptation</a></p>
</td>
</tr>
<tr id="en-us_topic_0000001105032530_row1184984391512"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="en-us_topic_0000001105032530_p1054075616153"><a name="en-us_topic_0000001105032530_p1054075616153"></a><a name="en-us_topic_0000001105032530_p1054075616153"></a>4</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="en-us_topic_0000001105032530_p1463045415151"><a name="en-us_topic_0000001105032530_p1463045415151"></a><a name="en-us_topic_0000001105032530_p1463045415151"></a>Register an operator.</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="en-us_topic_0000001105032530_p1634748161614"><a name="en-us_topic_0000001105032530_p1634748161614"></a><a name="en-us_topic_0000001105032530_p1634748161614"></a>Distribute the operator to the Ascend AI Processor.</p>
</td>
</tr>
<tr id="en-us_topic_0000001105032530_en-us_topic_0228422310_row252634054913"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="en-us_topic_0000001105032530_p55407561152"><a name="en-us_topic_0000001105032530_p55407561152"></a><a name="en-us_topic_0000001105032530_p55407561152"></a>5</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="en-us_topic_0000001105032530_p116302054131518"><a name="en-us_topic_0000001105032530_p116302054131518"></a><a name="en-us_topic_0000001105032530_p116302054131518"></a>Develop the operator adaptation layer.</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="en-us_topic_0000001105032530_p8583195119173"><a name="en-us_topic_0000001105032530_p8583195119173"></a><a name="en-us_topic_0000001105032530_p8583195119173"></a>Develop the operator adaptation layer to map the attributes of operators based on third-party frameworks to those of the operators adapted to Ascend AI Processors.</p>
</td>
</tr>
<tr id="en-us_topic_0000001105032530_en-us_topic_0228422310_row17721543154917"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="en-us_topic_0000001105032530_p125402056121515"><a name="en-us_topic_0000001105032530_p125402056121515"></a><a name="en-us_topic_0000001105032530_p125402056121515"></a>6</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="en-us_topic_0000001105032530_p963085451515"><a name="en-us_topic_0000001105032530_p963085451515"></a><a name="en-us_topic_0000001105032530_p963085451515"></a>Compile and install the PyTorch framework.</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="en-us_topic_0000001105032530_p1463431151811"><a name="en-us_topic_0000001105032530_p1463431151811"></a><a name="en-us_topic_0000001105032530_p1463431151811"></a>Compile and adapt the developed PyTorch source code, and install the compiled source package.</p>
</td>
</tr>
<tr id="en-us_topic_0000001105032530_en-us_topic_0228422310_row162484372491"><td class="cellrowborder" valign="top" width="6.811326262527976%" headers="mcps1.2.5.1.1 "><p id="en-us_topic_0000001105032530_p454075611155"><a name="en-us_topic_0000001105032530_p454075611155"></a><a name="en-us_topic_0000001105032530_p454075611155"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="17.865135740001946%" headers="mcps1.2.5.1.2 "><p id="en-us_topic_0000001105032530_p11630135413155"><a name="en-us_topic_0000001105032530_p11630135413155"></a><a name="en-us_topic_0000001105032530_p11630135413155"></a>Verify the operator functions.</p>
</td>
<td class="cellrowborder" valign="top" width="55.55123090396029%" headers="mcps1.2.5.1.3 "><p id="en-us_topic_0000001105032530_en-us_topic_0228422310_p4952132615216"><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p4952132615216"></a><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p4952132615216"></a>Verify the operator functions in the real-world hardware environment.</p>
</td>
<td class="cellrowborder" valign="top" width="19.772307093509777%" headers="mcps1.2.5.1.4 "><p id="en-us_topic_0000001105032530_en-us_topic_0228422310_p20908934557"><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p20908934557"></a><a name="en-us_topic_0000001105032530_en-us_topic_0228422310_p20908934557"></a><a href="#operator-function-verificationmd">Operator Function Verification</a></p>
</td>
</tr>
</tbody>
</table>

## Operator Development Preparations
### Setting Up the Environment

-   The development or operating environment of CANN has been installed. For details, see the _CANN Software Installation Guide_.
-   Python 3.7.5 or 3.8 has been installed.
-   CMake 3.12.0 or later has been installed. For details, see [Installing CMake](#installing-cmake).
-   GCC 7.3.0 or later has been installed. For details about how to install and use GCC 7.3.0, see "Installing GCC 7.3.0" in the  _CANN Software Installation Guide_.
-   The Git tool has been installed. To install Git for Ubuntu and CentOS, run the following commands:
    -   Ubuntu and EulerOS

        ```
        apt-get install patch
        apt-get install git
        ```

    -   CentOS

        ```
        yum install patch
        yum install git
        ```



### Looking Up Operators

During operator development, you can query the list of operators supported by Ascend AI Processors and the list of operators adapted to PyTorch. Develop or adapt operators to PyTorch based on the query result.

-   If an operator is not supported by the Ascend AI Processor, develop a TBE operator and adapt the operator to the PyTorch framework.
-   If an operator is supported by the Ascend AI Processor but has not been adapted to the PyTorch framework, you only need to adapt the operator to the PyTorch framework.
-   If an operator has been adapted to the PyTorch framework, you can directly use the operator without development or adaptation.

The following describes how to query the operators supported by Ascend AI Processors as well as operators adapted to PyTorch.

-   You can query the operators supported by Ascend AI Processors and the corresponding operator constraints in either of the following modes:
    -   For operator development on the command line, you can perform offline query. For details, see the _CANN Operator List \(Ascend 910\)_.
    -   For operator development using MindStudio, you can perform online query on MindStudio. For details, see "Supported Operators and Models" in the _MindStudio User Guide_.

-   For the list of operators adapted to PyTorch, see the *[PyTorch API Support](https://gitee.com/ascend/pytorch/blob/master/docs/en/PyTorch%20API%20Support.md)*.

## Operator Adaptation

### Prerequisites

-   The development and operating environments have been set up, and related dependencies have been installed. For details, see [Setting Up the Environment](#setting-up-the-environment).
-   TBE operators have been developed and deployed. For details, see the _CANN TBE Custom Operator Development Guide_.

### Obtaining the PyTorch Source Code

For details about how to obtain the PyTorch source code of PyTorch 1.8.1, perform steps described in "Installing the PyTorch Framework" in the *[PyTorch Installation Guide](https://gitee.com/ascend/pytorch/blob/master/docs/en/PyTorch%20Installation%20Guide/PyTorch%20Installation%20Guide.md)*. The full code adapted to Ascend AI Processors is generated in the **pytorch/pytorch_v1.8.1** directory. The PyTorch operator is also adapted and developed in this directory.

### Registering an Operator

#### Overview

Currently, the NPU adaptation dispatch principle is as follows: The NPU operator is directly dispatched as the NPU adaptation function without being processed by the common function of the framework. That is, the operator execution call stack contains only the function call of the NPU adaptation and does not contain the common function of the framework. During compilation, the PyTorch framework generates the calling description of the middle layer of the new operator based on the definition in **native\_functions.yaml** and the type and device dispatch principle defined in the framework. For NPUs, the description is generated in **build/aten/src/ATen/NPUType.cpp**.


#### Registering an Operator for PyTorch 1.8.1

##### Registering an Operator

1.  Open the **native\_functions.yaml** file.

    The **native\_functions.yaml** file defines all operator function prototypes, including function names and parameters. Each operator function supports dispatch information of different hardware platforms. The file is in the **pytorch/aten/src/ATen/native/native\_functions.yaml** directory.

2.  Determine the functions to be dispatched.
    -   Existing operator in the YAML file

        Dispatch all functions related to the operator to be adapted.

    -   Custom operator that does not exist in the YAML file

        The YAML file does not contain the operator information. Therefore, you need to manually add related functions, including the function names, parameters, and return types. For details about how to add a rule, see **pytorch/aten/src/ATen/native/README.md**.

        ```
        - func: operator name (input parameter information) -> return type
        ```



##### Examples

The following uses the torch.add\(\) operator as an example to describe how to register an operator.

1.  Open the **native\_functions.yaml** file.
2.  Search for related functions.

    Search for **add** in the YAML file and find **func** that describes the add operator. The add operator is a built-in operator of PyTorch. Therefore, you do not need to manually add **func**. If the operator is a custom operator, you need to manually add **func**.

3.  Determine the function description related to operator name and type.
    -   Dispatch description of **add.Tensor**

        ```
        - func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
          structured_delegate: add.out
          variants: function, method
          dispatch:
            SparseCPU, SparseCUDA: add_sparse
            MkldnnCPU: mkldnn_add
        ```

    -   Dispatch description of **add.Scalar**

        ```
        - func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
          variants: function, method
          dispatch:
            DefaultBackend: add
        ```

    -   Dispatch description of **add\_.Tensor**

        ```
        - func: add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
          variants: method
          structured_delegate: add.out
          dispatch:
            SparseCPU, SparseCUDA: add_sparse_
            MkldnnCPU: mkldnn_add_
        ```

    -   Dispatch description of **add\_.Scalar**

        ```
        - func: add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
          variants: method
          dispatch:
            DefaultBackend: add_
        ```

    -   Dispatch description of **add.out**

        ```
        - func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
          structured: True
          structured_inherits: TensorIteratorBase
          dispatch:
            CPU, CUDA: add_out
            SparseCPU: add_out_sparse_cpu
            SparseCUDA: add_out_sparse_cuda
            MkldnnCPU: mkldnn_add_out
        ```



### Developing an Operator Adaptation Plugin

#### Overview

You can develop an operator adaptation plugin to convert the formats of the input parameters, output parameters, and attributes of the PyTorch native operators so that the obtained formats are the same as the formats of the input parameters, output parameters, and attributes of the TBE operators. The PyTorch source code that is adapted to Ascend AI Processors provides methods related to adaptation association, type conversion and discrimination, and dynamic shape processing for users.

#### Introduction to the npu_native_functions.yaml File

```
backend: NPU     # Backend type
cpp_namespace: at_npu::native     # Namespace of the development operator in the plugin
supported:     # Supported operators aligned with PyTorch Native Functions
  - add.Tensor
  - add.Scalar
  - slow_conv3d.out
  - slow_conv3d_forward.output
  - slow_conv3d_forward
  - convolution
  - _convolution
  - _convolution_nogroup
  - addcdiv
  - addcdiv_
  - addcdiv.out

autograd:       # Supported operators that are aligned with PyTorch Native Functions and inherited from Function and have forward and reverse operations
  - maxpool2d

custom:     # Custom operators. The operator format definition needs to be provided.
  - func: npu_dtype_cast(Tensor self, ScalarType dtype) -> Tensor
    variants: function, method
  - func: npu_dtype_cast_(Tensor(a!) self, Tensor src) -> Tensor(a!)
    variants: method
  - func: npu_alloc_float_status(Tensor self) -> Tensor
    variants: function, method
  - func: npu_get_float_status(Tensor self) -> Tensor
    variants: function, method
 
custom_autograd:    # Custom operators inherited from Function
  - func: npu_convolution(Tensor input, Tensor weight, Tensor? bias, ...) -> Tensor
```

The official **native_functions.yaml** file defines the operator definitions and distribution details of PyTorch Native Functions. To adapt to the officially defined operators on the NPU devices, you only need to register the NPU distribution. The format of each function can be obtained by parsing the official .yaml file based on the supported and autograd operators. The corresponding function declaration, registration, and distribution can be automatically completed. Therefore, during operator porting and development, you only need to pay attention to the implementation details. For custom operators, no specific operator definitions are available. Therefore, you need to define the operators in the **npu_native_functions.yaml** file to perform structured parsing on the operators to implement automatic registration and Python API binding.

#### Adaptation Plugin Implementation

1.  Register the operators.

    Add operator information based on the description in the **npu_native_functions.yaml** file.

2.  Create an adaptation plugin file.

    The NPU TBE operator adaptation file is stored in the  pytorch/torch_npu/csrc/aten/ops** directory and is named in the upper camel case. The file name is in the format of  _operator name_  + **KernelNpu.cpp**, for example, **AddKernelNpu.cpp**.

3.  Introduce the dependency header files.

    The PyTorch source code that is adapted to Ascend AI Processors provides common tools in **torch_npu/csrc/framework/utils** for users.

    >![](public_sys-resources/icon-note.gif) **NOTE:** 
    >For details about the functions and usage of the tools, see the header files and source code.

4.  Define the main adaptation function of the operator.

    Determine the main adaptation function for custom operators based on the dispatch function in the registered operator.

5.  Implement the main adaptation functions.

    Implement the operator's main adaptation function and construct the corresponding input, output, and attributes based on the TBE operator prototype.


#### Examples

The following uses the torch.add() operator as an example to describe the development process of operator adaptation..

1. Register the operator.

   Add the torch.add() operator to the corresponding location in the **npu_native_functions.yaml** file for automatic declaration and registration.

   ```
   supported:       # Supported operators aligned with PyTorch Native Functions
     add.Tensor
     add_.Tensor
     add.out
     add.Scaler
     add_.Scaler
   ```

   Format reference: *[Operator Porting and Development Guide](https://gitee.com/ascend/pytorch/wikis/%E7%AE%97%E5%AD%90%E8%BF%81%E7%A7%BB%E5%92%8C%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97)* 

2. Create an adaptation plugin file.

   Create the **AddKernelNpu.cpp** adaptation file in **pytorch/torch_npu/csrc/aten/ops**.

3. Introduce the dependency header files.

   ```
   #include <ATen/Tensor.h>
   #include <c10/util/SmallVector.h>
   
   #include "torch_npu/csrc/core/npu/register/OptionsManager.h"
   #include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
   #include "torch_npu/csrc/framework/utils/OpAdapter.h"
   #include "torch_npu/csrc/aten/NPUNativeFunctions.h"
   ```

   >![](public_sys-resources/icon-note.gif) **NOTE:** 
   >**CalcuOpUtil.h** contains functions for type conversion and discrimination. 
   >**OpAdapter.h** contains header files associated with adaptation.

4. Define the operator adaptation main functions.

   ```
   at::Tensor NPUNativeFunctions::add(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
   at::Tensor NPUNativeFunctions::add(const at::Tensor &self, at::Scalar other, at::Scalar alpha)
   at::Tensor &NPUNativeFunctions::add_(at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
   at::Tensor &NPUNativeFunctions::add_(at::Tensor &self, at::Scalar other, at::Scalar alpha)
   at::Tensor &NPUNativeFunctions::add_out(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha,at::Tensor &result)
   ```

   > ![](public_sys-resources/icon-note.gif) **NOTE:** 
   >
   > **NPUNativeFunctions** is the namespace constraint that needs to be added for operator definitions.

5. Implement the adaptation main functions.

   1. add implementation

      ```
       // When the input parameters are Tensor and Tensor:
             at::Tensor NPUNativeFunctions::add(const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
             {
               alpha_check_npu(self.scalar_type(), alpha);
               if ((!(self.is_contiguous() && other.is_contiguous())) &&
                   (NpuUtils::check_5d_5d_match(self) ||
                   NpuUtils::check_5d_5d_match(other)) &&
                   check_size(self, other))
               {
                 int64_t c0_len = 16;
                 at::Tensor self_use = stride_add_tensor_get(self);
                 at::Scalar self_c1_offset(
                   self.storage_offset() / (self.size(2) * self.size(3) * c0_len));
                 at::Tensor other_use = stride_add_tensor_get(other);
                 at::Scalar other_c1_offset(
                   other.storage_offset() / (other.size(2) * other.size(3) * c0_len));
                 at::Scalar stride_len(self.size(1) / c0_len);
                 at::Tensor result = NPUNativeFunctions::npu_stride_add(
                   self_use, other_use, self_c1_offset, other_c1_offset, stride_len);
                 return result;
               }
               // calculate the output size
               at::Tensor outputTensor = add_dest_output(self, other);
               auto outputSize = broadcast_ops_npu_output_size(self, other);
             
               // construct the output tensor of the NPU
               at::Tensor result = OpPreparation::ApplyTensorWithFormat(
                   outputSize,
             	  outputTensor.options(),
             	  CalcuOpUtil::get_tensor_npu_format(outputTensor));
             
               // calculate the output result of the NPU
               add_out_npu_nocheck(result, self, other, alpha);
             
               return result;
             }
                 
             // When the input parameters are Tensor and Scalar:
             at::Tensor NPUNativeFunctions::add(const at::Tensor &self, at::Scalar other, at::Scalar alpha)
             {
               alpha_check_npu(self.scalar_type(), alpha);
               // calculate the output size
               auto outputSize = input_same_output_size(self);
               // construct the output tensor of the NPU
               at::Tensor result = OpPreparation::ApplyTensorWithFormat(
                   outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));
             
               // calculate the output result of the NPU
               adds_out_npu_nocheck(result, self, other, alpha);
             
               return result;
             }
      
      ```

   2. add\_implementation (In the local operation scenario, the return value is **self**.)

      ```
      // When the input parameters are Tensor and Tensor:
             at::Tensor &NPUNativeFunctions::add_(at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
             {
               c10::SmallVector<at::Tensor, N> inputs = {self, other};
               c10::SmallVector<at::Tensor, N> outputs = {self};
               CalcuOpUtil::check_memory_over_laps(inputs, outputs);
             
               if (!NpuUtils::check_match(&self))
               {
                 at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
                 at::Tensor result = add_out_npu_nocheck(contiguousSelf, contiguousSelf, other, alpha);
                 NpuUtils::format_fresh_view(self, result);
               }
               else
               {
                 add_out_npu_nocheck(self, self, other, alpha);
               }
             
               return self;
             }
             
             // When the input parameters are Tensor and Scalar:
             at::Tensor &NPUNativeFunctions::add_(at::Tensor &self, at::Scalar other, at::Scalar alpha)
             {
               if (!NpuUtils::check_match(&self))
               {
                 at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
                 at::Tensor result = adds_out_npu_nocheck(contiguousSelf, contiguousSelf, other, alpha);
                 NpuUtils::format_fresh_view(self, result);
               }
               else
               {
                 adds_out_npu_nocheck(self, self, other, alpha);
               }
             
               return self;
             }
      
      ```

   3.  add\_out implementation (scenario where the input parameter result is a return value)

       ```
       at::Tensor &NPUNativeFunctions::add_out(
               const at::Tensor &self,
               const at::Tensor &other,
               at::Scalar alpha,
               at::Tensor &result)
           {
             bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
       
             at::Tensor outputTensor;
             if (not isSelfWrapped)
             {
               outputTensor = self;
             }
             else
             {
               outputTensor = other;
             }
             auto outputSize = broadcast_ops_npu_output_size(self, other);
             OpPreparation::CheckOut(
                 {self},
                 result,
                 CalcuOpUtil::get_tensor_npu_format(result),
                 outputTensor.scalar_type(),
                 outputSize);
       
             OpPipeWithDefinedOut pipe;
             return pipe.CheckMemory({self, other}, {result})
                 .Func([&self, &other, &alpha](at::Tensor &result)
                       { add_out_npu_nocheck(result, self, other, alpha); })
                 .Call(result);
           }
       ```


>![](public_sys-resources/icon-note.gif) **NOTE:** 
>For details about the implementation code of **AddKernelNpu.cpp**, see the **pytorch/torch_npu/csrc/aten/ops/AddKernelNpu.cpp** document.

### Compiling and Installing the PyTorch Plugin

#### Compiling the PyTorch Plugin

1.  Go to the PyTorch working directory **pytorch**.
2.  Install the dependency.

    ```
    pip3 install -r requirements.txt
    ```

3.  Compile and generate the binary installation package of the PyTorch plugin.

    ```
    bash ci/build.sh --python=3.7
    or
    bash ci/build.sh --python=3.8
    or
    bash ci/build.sh --python=3.9
    ```
    Specify the Python version in the environment for compilation. After the compilation is successful, the binary package **torch_npu\*.whl** is generated in the **pytorch/dist** directory, for example, **torch_npu-1.8.1rc1-cp37-cp37m-linux_x86_64.whl**.


#### Installing the PyTorch Plugin

Go to the **pytorch/dist** directory and run the following command to install PyTorch:

```
pip3 install --upgrade torch_npu-1.8.1rc1-cp37-cp37m-linux_{arch}.whl
```

_**\{arch\}**_  indicates the architecture information. The value can be **aarch64** or **x86\_64**.

>![](public_sys-resources/icon-note.gif) **NOTE:** 
>--upgrade: Uninstall the PyTorch plugin software package installed in the environment and then perform the update installation. You can run the following command to check whether the PyTorch plugin has been installed in the environment:
>**pip3 list | grep torch_npu**

After the code has been modified, you need to re-compile and re-install the PyTorch plugin.

## Operator Function Verification

### Overview

#### Introduction

After operator adaptation is complete, you can run the PyTorch operator adapted to Ascend AI Processor to verify the operator running result.

Operator verification involves all deliverables generated during operator development, including the implementation files, operator prototype definitions, operator information library, and operator plugins. This section describes only the verification method.

#### Test Cases and Records

Use the PyTorch frontend to construct the custom operator function and run the function to verify the custom operator functions.

The test cases and test tools are provided in the **pytorch/test/test_network_ops**  directory at **https://gitee.com/ascend/pytorch**.

### Implementation

#### Introduction

This section describes how to test the functions of a PyTorch operator.

#### Procedure

1.  Set environment variables.

    ```
    # Set environment variables. The details are as follows (the root user is used as an example and the installation path is the default path):
    usr/local/Ascend/ascend-toolkit/set_env.sh 
    # Set environment variables. The details are as follows (a non-root user is used as an example and the installation path is the default path):
    ${HOME}/Ascend/ascend-toolkit/set_env.sh
    ```

2.  Compile test scripts. Take the add operator as an example. Compile the test script file **test\_add.py** in the **pytorch/test/test\_network\_ops** directory.
    ```
    # Import the dependency library.
    import torch
    import torch_npu
    import numpy as np
    
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import create_common_tensor
    
    # Define the add test case class.
    class TestAdd(TestCase):
    
        # Define the functions to execute the add operator on the CPU and NPU.
        def cpu_op_exec(self, input1, input2):
            output = torch.add(input1, input2, alpha = 1)
            output = output.numpy()
            return output
        def npu_op_exec_new(self, input1, input2):
            output = torch.add(input1, input2, alpha = 1)
            output = output.to("cpu")
            output = output.numpy()
            return output
    
        # Define a general function for the add scenario. This function is used to input data and compare the compute results of the CPU and NPU.
        def add_result(self, shape_format):
            for item in shape_format:
                cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
                cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
                if cpu_input1.dtype == torch.float16:
                    cpu_input1 = cpu_input1.to(torch.float32)
                    cpu_input2 = cpu_input2.to(torch.float32)                
                cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
                npu_output = self.npu_op_exec_new(npu_input1, npu_input2)
                cpu_output = cpu_output.astype(npu_output.dtype)            
                self.assertRtolEqual(cpu_output, npu_output)
    
        # Define a test case for a specific add scenario. The test case function must start with test_.
        def test_add_shape_format_fp32_2d(self):
            format_list = [0, 3, 29]
            shape_format = [
                [np.float32, i, [5, 256]]  for i in format_list 
            ]
            self.add_result(shape_format)
    
    if __name__ == "__main__":
        run_tests()
    ```

3.  Execute the test case script.

    Go to the directory where **test\_add.py** is located, and run the following command:

    ```
    python3.7 test_add.py
    ```


## FAQs

### Pillow==5.3.0 Installation Failed

#### Symptom

**Pillow==5.3.0** installation failed.

#### Possible Cause

Necessary dependencies are missing, such as libjpeg, python-devel, zlib-devel, and libjpeg-turbo-devel.

#### Solutions

Run the following command to install the required dependencies:

```
apt-get install libjpeg python-devel  zlib-devel  libjpeg-turbo-devel  # Ubunntu, EulerOS
yum install libjpeg python-devel  zlib-devel  libjpeg-turbo-devel      # CentOS
```

### pip3.7 install torchvision Installation Failed

#### Symptom

**pip3.7 install torchvision** installation failed.

#### Possible Cause

The versions of PyTorch and TorchVision do not match.

#### Solutions

Run the following command:

```
pip3.7 install torchvision --no-deps
```

### "torch 1.5.0xxxx" and "torchvision" Do Not Match When torch-\*.whl Is Installed

#### Symptom

During the installation of **torch-**_\*_**.whl**, the message "ERROR: torchvision 0.6.0 has requirement torch==1.5.0, but you'll have torch 1.5.0a0+1977093 which is incompatible" is displayed.

![](figures/en-us_image_0000001144082048.png)

However, the installation is successful.

#### Possible Cause

When the PyTorch is installed, the version check is automatically triggered. The version of the torchvision installed in the environment is 0.6.0. During the check, it is found that the version of the **torch-**_\*_**.whl** is inconsistent with the required version 1.5.0. As a result, an error message is displayed.

#### Solutions

This problem has no impact on the actual result, and no action is required.

### How Do I View Test Run Logs?

When an error message is displayed during the test, but the reference information is insufficient, how can we view more detailed run logs?

Output the logs to the screen and redirect them to a specified text file.

1.  Set the environment variable to display the logs of the current user on the screen.

    ```
    # Log printing
    export SLOG_PRINT_TO_STDOUT=1
    export ASCEND_GLOBAL_LOG_LEVEL=1
    #0: debug; 1: info level; 2: warning 3: error
    ```
    After the setting is complete, run the test case to output related logs to the screen. To facilitate viewing and backtracking, you are advised to perform [2](#en-us_topic_0000001125315889_li168732325719) as required.

2.  Redirect the logs to a specified file when a test case is executed.

    ```
    python3.7 test_add.py > test_log.txt
    ```


### Why Cannot the Custom TBE Operator Be Called?

#### Symptom

The custom TBE operator has been developed and adapted to PyTorch. However, the newly developed operator cannot be called during test case execution.

#### Possible Cause

-   The environment variables are not set correctly.
-   An error occurs in the YAML file. As a result, the operator is not correctly dispatched.
-   The implementation of the custom TBE operator is incorrect. As a result, the operator cannot be called.

#### Solutions

1.  Set the operating environment by referring to [Operator Function Verification](#operator-function-verification). Pay special attention to the following settings:

    ```
    . /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh 
    ```

2.  Check whether the dispatch configuration of the corresponding operator in the YAML file is correct and complete.
3.  Analyze and check the code implementation. The recommended methods are as follows:
    1.  Modify the operator adaptation implementation in PyTorch so that **test\_add.py** can call the TBE operator in the custom operator package.

        "pytorch/aten/src/ATen/native/npu/AddKernelNpu.cpp"

        ![](figures/en-us_image_0000001144082088.png)

    2.  After the compilation and installation steps are complete, call **python3.7 test\_add.py** to perform the test.

        ```
        Run the cd command to go to the directory where test_add.py is stored and call
        test_add.py
        to perform the test.
        ```

        There should be no error in this step. The log added in **add** should be displayed. If an error occurs, check the code to ensure that no newly developed code affects the test.

    3.  Combine the newly developed custom TBE operator into CANN. Add logs to the operator entry as the running identifier.
    4.  After the compilation and installation of CANN are complete, call **python3.7.5 test\_add.py** to perform the test.

        >![](public_sys-resources/icon-note.gif) **NOTE:** 
        >According to the design logic of Ascend, the priority of the custom operator package is higher than that of the built-in operator package. During operator loading, the system preferentially loads the operators in the custom operator package. During the process, if the operator information file in the custom operator package fails to be parsed, the custom operator package is skipped and no operator in the custom operator package is loaded or scheduled.
        >-   If an error occurs in this step or the log added in **add** is not displayed, the newly developed custom TBE operator is incorrect, which affects the loading of the custom operator package. You are advised to **check whether the operator information library definition in the newly developed custom TBE operator is correct**.
        >-   If this step is correct, **the operator information library definition in the newly developed custom TBE operator does not affect the running**.

    5.  Call  **python3.7.5** _xxx_**\_testcase.py**  to perform the test.

        >![](public_sys-resources/icon-note.gif) **NOTE:** 
        >-   If the logs added to the newly developed custom TBE operator are displayed on the screen, the newly developed operator is scheduled.
        >-   If the logs added to the newly developed custom TBE operator are not displayed on the screen, the problem may occur in PyTorch adaptation. In this case, you need to check the implementation code of PyTorch adaptation. Most of the problems are due to the incorrect adaptation of input and output of  _xxxx_**KernelNpu.cpp**.



### How Do I Determine Whether the TBE Operator Is Correctly Called for PyTorch Adaptation?

Both the custom and built-in operators are stored in the installation directory as .py source code after installation. Therefore, you can edit the source code and add logs at the API entry to print the input parameters, and determine whether the input parameters are correct.

>![](public_sys-resources/icon-caution.gif) **CAUTION:** 
>This operation may cause risks. You are advised to back up the file to be modified before performing this operation. If the files are not backed up and cannot be restored after being damaged, contact technical support.

The following uses the **zn\_2\_nchw** operator in the built-in operator package as an example:

1.  Open the installation directory of the operator package in the user directory.

    ```
    cd ~/.local/Ascend/opp/op_impl/built-in/ai_core/tbe/impl
    ll
    ```

    The .py source code file of the corresponding operator is read-only, that is, the file cannot be edited.

    ![](figures/en-us_image_0000001190081791.png)

2.  Modify the attributes of the .py source code file of the operator and add the write permission.

    ```
    sudo chmod +w zn_2_nchw.py
    ll
    ```

    ![](figures/en-us_image_0000001190081803.png)

3.  Open the .py source code file of the operator, add logs, save the file, and exit.

    ```
    vi zn_2_nchw.py
    ```

    ![](figures/en-us_image_0000001190201951.png)

    In the preceding example, only an identifier is added. In actual commissioning, you can add the input parameters to be printed.

4.  Call and execute the test case to analyze the input parameter information.
5.  After the test analysis is complete, open the .py source code file of the operator again, delete the added logs, save the file, and exit.
6.  Modify the attributes of the .py source code file of the operator and remove the write permission.

    ```
    sudo chmod -w zn_2_nchw.py
    ```

    ![](figures/en-us_image_0000001144082072.png)


### PyTorch Compilation Fails and the Message "error: ld returned 1 exit status" Is Displayed

#### Symptom

PyTorch compilation fails and the message "error: ld returned 1 exit status" is displayed.

![](figures/en-us_image_0000001190201973.png)

#### Possible Cause

According to the log analysis, the possible cause is that the adaptation function implemented in _xxxx_**KernelNpu.cpp** does not match the dispatch implementation API parameters required by the PyTorch framework operator. In the preceding example, the function is **binary\_cross\_entropy\_npu**. Open the corresponding _xxxx_**KernelNpu.cpp** file and find the adaptation function.

![](figures/en-us_image_0000001144241896.png)

In the implementation, the type of the last parameter is **int**, which does not match the required **long**.

#### Solutions

Modify the adaptation function implemented in  _xxxx_**KernelNpu.cpp**. In the preceding example, change the type of the last parameter in the  **binary_cross_entropy_npu**  function to **int64_t** (use **int64\t** instead of **long** in the .cpp file).

### PyTorch Compilation Fails and the Message "error: call of overload...." Is Displayed

#### Symptom

PyTorch compilation fails and the message "error: call of overload...." is displayed.

![](figures/en-us_image_0000001144082056.png)

![](figures/en-us_image_0000001190201935.png)

#### Possible Cause

According to the log analysis, the error is located in line 30 in the _xxxx_**KernelNpu.cpp**  file, indicating that the **NPUAttrDesc** parameter is invalid. In the preceding example, the function is **binary\_cross\_entropy\_attr**. Open the corresponding _xxxx_**KernelNpu.cpp**  file and find the adaptation function.

![](figures/en-us_image_0000001144082064.png)

In the implementation, the type of the second input parameter of **NPUAttrDesc** is **int**, which does not match the definition of **NPUAttrDesc**.

#### Solutions

1. Replace the incorrect code line in the **binary\_cross\_entropy\_attr\(\)** function with the code in the preceding comment.

2. Change the input parameter type of **binary\_cross\_entropy\_attr\(\)** to **int64\_t**.

## Appendixes

### Installing CMake

The following describes how to upgrade CMake to 3.12.1.

1.  Obtain the CMake software package.

    ```
    wget https://cmake.org/files/v3.12/cmake-3.12.1.tar.gz --no-check-certificate
    ```

2.  Decompress the package and go to the software package directory.

    ```
    tar -xf cmake-3.12.1.tar.gz
    cd cmake-3.12.1/
    ```

3.  Run the configuration, compilation, and installation commands.

    ```
    ./configure --prefix=/usr/local/cmake
    make && make install
    ```

4.  Set the soft link.

    ```
    ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
    ```

5.  Check whether CMake has been installed.

    ```
    cmake --version
    ```

    If the message "cmake version 3.12.1" is displayed, the installation is successful.


### Exporting a Custom Operator

#### Overview

A PyTorch model contains a custom operator. You can export the custom operator as an ONNX single-operator model, which can be easily ported to other AI frameworks. Three types of custom operator export are available: NPU-adapted TBE operator export, C++ operator export, and pure Python operator export.

#### Prerequisites

You have installed the PyTorch framework.

#### TBE Operator Export

A TBE operator can be exported using either of the following methods:

Method 1:

1.  Define and register an operator.

    ```
    # Define an operator.
    @parse_args('v', 'v', 'f', 'i', 'i', 'i', 'i')
    def symbolic_npu_roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode):
        args = [input, rois]
        kwargs = {"spatial_scale_f": spatial_scale,
                "pooled_height_i": pooled_height,
                "pooled_width_i": pooled_width,
                "sample_num_i": sample_num,
                "roi_end_mode_i": roi_end_mode}
    
        return g.op('torch::npu_roi_align',*args, **kwargs)
    
    # Register the operator.
    import torch.onnx.symbolic_registry as sym_registry
    def register_onnx_sym_npu_roi_align():
          sym_registry.register_op('npu_roi_align', symbolic_npu_roi_align, '', 11)   
              
    register_onnx_sym_npu_roi_align()
    ```

2.  Customize a model.

    ```
    # Define a model.
    class CustomModel_npu_op(torch.nn.Module):
        def __init__(self,a,b):
            super(CustomModel_npu_op, self).__init__()
    
            self.weight = Parameter(torch.Tensor(8,10,1024))    
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
        def forward(self, a, b, d):  
            spatial_scale=d[0].item()
            pooled_height=d[1].item()
            pooled_width=d[2].item()
            sample_num=d[3].item()
            roi_end_mode=d[4].item()
            rtn = torch.npu_roi_align(a, self.weight, spatial_scale, pooled_height, pooled_width, sample_num,roi_end_mode)
    
            return rtn
    ```

3.  Export the ONNX file.

    ```
    # Define an export function.
    def do_export(model, inputs, f, *args, **kwargs):
        out = torch.onnx._export(model, inputs, f, verbose=True, export_params=True, do_constant_folding=True,*args, **kwargs)
    
    # Initialize the input.
    """
    Initialize the input parameters a, b, and h1 of the model. For details, see the detailed code.
    """
    
    # Export the ONNX file.
    model = CustomModel_npu_op(a,b)
    model = model.npu()
    model.eval()
    do_export(model, (a, b, h1), f, input_names=["intput"]+["","","","","","","npu_roi_align.weight"],opset_version=11)
    ```


Method 2:

1.  Define a method class.

    ```
    # Implement the operator method class and symbol export method.
    class CustomClassOp_Func_npu_roi_align(Function):
        @staticmethod
        def forward(ctx, input, rois, spatial_scale, pooled_height, pooled_width , sample_num, roi_end_mode):
            rtn = torch.npu_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode)
            return rtn
    
        @staticmethod
        def symbolic(g, input, rois, spatial_scale, pooled_height, pooled_width , sample_num, roi_end_mode):
            args = [input, rois]
            kwargs = {"spatial_scale_f": spatial_scale,
                        "pooled_height_i": pooled_height,
                        "pooled_width_i": pooled_width,
                        "sample_num_i": sample_num,
                        "roi_end_mode_i": roi_end_mode}
            return g.op('torch::npu_roi_align',*args, **kwargs)
    ```

2.  Customize an operator model.

    ```
    # Implement an operator model.
    class NpuOp_npu_roi_align_Module(torch.nn.Module):
        def __init__(self):
            super(NpuOp_npu_roi_align_Module, self).__init__()
    
            self.spatial_scale = torch.randn(10, dtype=torch.float32, requires_grad=False,device="cpu")[0].item()
            self.pooled_height = 2
            self.pooled_width = 0
            self.sample_num = 1
            self.roi_end_mode = 1
    
            self.weight = Parameter(torch.Tensor(8,10,1024))
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
            self.func = CustomClassOp_Func_npu_roi_align.apply
            self.test_npu_op=1
    
        def forward(self, input):
            rtn = self.func(input, self.weight, self.spatial_scale, self.pooled_height, self.pooled_width, self.sample_num, self.roi_end_mode)
            return rtn
    ```

3.  Customize a model.

    ```
    # Create a custom model.
    class CustomModel_Module_op(torch.nn.Module):
        def __init__(self,a,b):
            super(CustomModel_Module_op, self).__init__()
            self.npu_roi_align = NpuOp_npu_roi_align_Module()
        #@staticmethod
        def forward(self, a):
            rtn = self.npu_roi_align(a) 
            return rtn
    ```

4.  Export the ONNX file.

    ```
    # Build data.
    a = torch.randn(5, 10, 1024, dtype=torch.float32, requires_grad=True,device=rnddata_device)
    b = torch.randn(10, 10, 1024, dtype=torch.float32, requires_grad=True,device=rnddata_device)
    
    # Instantiate the model.
    model = CustomModel_Module_op(a,b)
    model = model.npu()
    model.eval()
    a = a.to('npu:6')
    b = b.to('npu:6')
    
    # Export the ONNX file.
    do_export(model, a, f=ONNX_NPU_OP_MODULE_FILENAME, input_names=["intput"]+["npu_roi_align.weight"],opset_version=11)
    ```


>![](public_sys-resources/icon-note.gif) **NOTE:** 
>For details about the implementation code, see [test\_custom\_ops\_npu\_demo.py](https://gitee.com/ascend/pytorch/blob/master/test/test_npu/test_onnx/torch.onnx/custom_ops_demo/test_custom_ops_npu_demo.py). If you do not have the permission to obtain the code, contact Huawei technical support to join the **Ascend** organization.

#### C++ Operator Export

1.  Customize an operator.

    ```
    import torch
    import torch.utils.cpp_extension
    # Define a C++ operator.
    def test_custom_add():    
        op_source = """    
        #include <torch/script.h>    
    
        torch::Tensor custom_add(torch::Tensor self, torch::Tensor other) {
            return self + other;    
        }
        static auto registry = 
            torch::RegisterOperators("custom_namespace::custom_add",&custom_add);
        """
        torch.utils.cpp_extension.load_inline(
            name="custom_add",
            cpp_sources=op_source,
            is_python_module=False,
            verbose=True,
        )
    
    test_custom_add()
    ```

2.  Register the custom operator.

    ```
    # Define the operator registration method and register the operator.
    from torch.onnx import register_custom_op_symbolic
    
    def symbolic_custom_add(g, self, other):
        return g.op('custom_namespace::custom_add', self, other)
    
    register_custom_op_symbolic('custom_namespace::custom_add', symbolic_custom_add, 9)
    ```

3.  Build a model.

    ```
    # Build an operator model.
    class CustomAddModel(torch.nn.Module):
        def forward(self, a, b):
            return torch.ops.custom_namespace.custom_add(a, b)
    ```

4.  Export the operator as an ONNX model.

    ```
    # Export the operator as an ONNX model.
    def do_export(model, inputs, *args, **kwargs):
        out = torch.onnx._export(model, inputs, "custom_demo.onnx", *args, **kwargs)
    
    x = torch.randn(2, 3, 4, requires_grad=False)
    y = torch.randn(2, 3, 4, requires_grad=False)
    model = CustomAddModel()
    do_export(model, (x, y), opset_version=11)
    ```


>![](public_sys-resources/icon-note.gif) **NOTE:** 
>For details about the implementation code, see [test\_custom\_ops\_demo.py](https://gitee.com/ascend/pytorch/blob/master/test/test_npu/test_onnx/torch.onnx/custom_ops_demo/test_custom_ops_demo.py). If you do not have the permission to obtain the code, contact Huawei technical support to join the **Ascend** organization.

#### Pure Python Operator Export

1.  Customize an operator.

    ```
    import torch
    import torch.onnx.symbolic_registry as sym_registry
    
    import torch.utils.cpp_extension
    import torch.nn as nn
    import torch.nn.modules as Module
    from torch.autograd import Function
    import numpy as np
    
    from torch.nn.parameter import Parameter
    import math
    from torch.nn  import init
    
    # Define an operator class method.
    class CustomClassOp_Add_F(Function):
        @staticmethod
        def forward(ctx, input1,input2):
            rtn = torch.add(input1,input2)
            return torch.add(input1,rtn)
    
        @staticmethod
        def symbolic(g,input1,input2):
            rtn = g.op("Custom::CustomClassOp_Add", input1, input2,test_attr1_i=1,test_attr2_f=1.0)
            rtn = g.op("ATen::CustomClassOp_Add", input1, rtn)
            rtn = g.op("C10::CustomClassOp_Add", rtn, input2)
            #erro doman: rtn = g.op("onnx::CustomClassOp_Add", input1, input2)
    
            return rtn
    ```

2.  Build a model.

    ```
    # Register the operator and build a model.
    class CustomClassOp_Add(torch.nn.Module):
        def __init__(self):
            super(CustomClassOp_Add, self).__init__()
            self.add = CustomClassOp_Add_F.apply
    
            #graph(%0 : Float(1, 8, 10, 1024),
            #      %1 : Float(8, 10, 1024))
            self.weight = Parameter(torch.Tensor(8,10,1024))
    
            #%1 : Float(8, 10, 1024) = onnx::Constant[value=<Tensor>]()
            self.weight1 = torch.Tensor(8,10,1024)
    
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        def forward(self, input):
            rtn = torch.add(self.weight1, self.weight)
    
            rtn = self.add(self.weight, rtn)
            rtn1 = self.add(self.weight, self.weight1)
            rtn1 = self.add(self.weight1,rtn1) 
            rtn = self.add(rtn,rtn1)
    
            return rtn
    ```

3.  Export the operator as an ONNX model.

    ```
    ONNX_FILE_NAME = "./custom_python_module_demo.onnx"
    def do_export(model, inputs, *args, **kwargs):
        out = torch.onnx._export(model, inputs, ONNX_FILE_NAME, verbose=True,keep_initializers_as_inputs=True, *args, **kwargs)
    
    def test_class_export():
        model = CustomModel()
        model.eval()
        input_x_shape = [1, 8, 10, 1024]
        input = torch.randn(input_x_shape)
        output = model(input)
        do_export(model, input, opset_version=11)
    
    # Export the operator as an ONNX model.
    test_class_export()
    ```


>![](public_sys-resources/icon-note.gif) **NOTE:** 
>For details about the implementation code, see [test\_custom\_ops\_python\_module.py](https://gitee.com/ascend/pytorch/blob/master/test/test_npu/test_onnx/torch.onnx/custom_ops_demo/test_custom_ops_python_module.py). If you do not have the permission to obtain the code, contact Huawei technical support to join the **Ascend** organization.

