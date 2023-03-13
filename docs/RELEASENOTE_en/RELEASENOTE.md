# FrameworkPTAdapter 3.0.RC2 Release Notes

- [FrameworkPTAdapter 3.0.RC2 Release Notes](#frameworkptadapter-30rc2-release-notes)
  - [FrameworkPTAdapter 3.0.RC2](#frameworkptadapter-30rc2)
    - [Before You Start](#before-you-start)
    - [New Features](#new-features)
    - [Modified Features](#modified-features)
    - [Resolved Issues](#resolved-issues)
    - [Known Issues](#known-issues)
    - [Compatibility](#compatibility)
  - [FrameworkPTAdapter 3.0.RC1](#frameworkptadapter-30rc1)
    - [Before You Start](#before-you-start-1)
    - [New Features](#new-features-1)
    - [Modified Features](#modified-features-1)
    - [Resolved Issues](#resolved-issues-1)
    - [Known Issues](#known-issues-1)
    - [Compatibility](#compatibility-1)
  - [FrameworkPTAdapter 2.0.4](#frameworkptadapter-204)
    - [Before You Start](#before-you-start-2)
    - [New Features](#new-features-2)
    - [Modified Features](#modified-features-2)
    - [Resolved Issues](#resolved-issues-2)
    - [Known Issues](#known-issues-2)
    - [Compatibility](#compatibility-2)
  - [FrameworkPTAdapter 2.0.3](#frameworkptadapter-203)
    - [Before You Start](#before-you-start-3)
    - [New Features](#new-features-3)
    - [Modified Features](#modified-features-3)
    - [Resolved Issues](#resolved-issues-3)
    - [Known Issues](#known-issues-3)
    - [Compatibility](#compatibility-3)
  - [FrameworkPTAdapter 2.0.2](#frameworkptadapter-202)
    - [Before You Start](#before-you-start-4)
    - [New Features](#new-features-4)
    - [Modified Features](#modified-features-4)
    - [Resolved Issues](#resolved-issues-4)
    - [Known Issues](#known-issues-4)
    - [Compatibility](#compatibility-4)


## FrameworkPTAdapter 3.0.RC2


### Before You Start

This framework is modified based on the open-source PyTorch 1.5.0 and 1.8.1 developed by Facebook, inherits native PyTorch features, and uses NPUs for dynamic image training. Models are adapted by operator granularity, code can be reused, and current networks can be ported and used on NPUs with only device types or data types modified.

PyTorch 1.8.1 uses plug-in adaptation and is completely decoupled from the native PyTorch framework. The current functions and performance of PyTorch 1.8.1 are basically the same as those of PyTorch 1.5.0, providing good development experience for backend operator adaptation.
AOE-based tuning is supported.

### New Features

**Table 1** Features supported by PyTorch

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.459999999999999%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>Level-1 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="26.27%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>Level-2 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="63.27%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>Description</p>
</th>
</tr>
</thead>
<tbody>
<tr id="row7979351559"><td class="cellrowborder" rowspan="2" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>PyTorch 1.5.0 features adapted to NPUs</p>
</td>
</tr>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a> Basic framework functions</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a> Fixed a few bugs.</p>
</td>
</tr>
<tr id="row7979351559"><td class="cellrowborder" rowspan="3" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>PyTorch 1.8.1 features adapted to NPUs</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>Decoupled as plugins</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>Decoupled the torch_npu plugin from the native PyTorch framework completely.</p>
</td>
<tr>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>Performance optimization</p>
</td> 
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>Optimized the single-operator delivery and collective communication functions.</p>
</td>
</tr>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>API satisfaction</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>Added more APIs. (For details, see the API list.)</p>
</td>
</tr>
</tbody>
</table>

### Modified Features

N/A

### Resolved Issues

N/A

### Known Issues

<a name="table1969972073016"></a>

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>Known Issue</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>Data type support</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPUs do not support the input or output of the inf/nan data of the float16 type</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>Data format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>Dimensions cannot be reduced when the format larger than 4D is used.</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>Restrictions on collective communication</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>The graphs executed on different devices in a training job must be the same.</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>Allocation at only 1, 2, 4, or 8 processors is supported.</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>Only the int8, int32, float16, and float32 data types are supported.</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex function</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>In the current version, Apex is implemented mainly using Python, and the customized optimization of CUDA kernel in Apex is not supported.</p>
</td>
</tr>
</tbody>
</table>

### Compatibility

Atlas 800 (model 9010): CentOS 7.6, Ubuntu 18.04/2.04, BC-Linux 7.6, Debian 9.9, Debian 10, openEuler 20.03 LTS

Atlas 800 (model 9000): CentOS 7.6, Ubuntu 18.04/2.04, EulerOS 2.8/2.10, Kylin V10, BC-Linux 7.6, openEuler 20.03 LTS, UOS 20 1020e

## FrameworkPTAdapter 3.0.RC1
### Before You Start

This framework is modified based on the open-source PyTorch 1.5.0 and 1.8.1 developed by Facebook, inherits native PyTorch features, and uses NPUs for dynamic image training. Models are adapted by operator granularity, code can be reused, and current networks can be ported and used on NPUs with only device types or data types modified.

PyTorch 1.8.1 adopts the plugin adaptation mode and inherits the features of PyTorch 1.5.0. Their functions are basically the same, but PyTorch 1.8.1 provides better development experience for backend operator adaptation. It supports AOE-based tuning.

### New Features

**Table 1** Features supported by PyTorch

<a name="t76c34275cbb74753970f7c5a9eb594fa"></a>

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.459999999999999%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>Level-1 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="26.27%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>Level-2 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="63.27%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>Description</p>
</th>
</tr>
</thead>
<tbody>
<tr id="row7979351559"><td class="cellrowborder" rowspan="2" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>PyTorch 1.5.0 features adapted to NPUs</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>AOE-based tuning</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>Supported the AOE auto tuning tool to improve the model performance.</p>
</td>
</tr>
<tr id="row13971435754"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p79831331353"><a name="p79831331353"></a><a name="p79831331353"></a>Basic framework functions</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a> Added the function of adapted operator development. For details, see the API list.</p>
</td>
</tr>
<tr id="row7979351559"><td class="cellrowborder" rowspan="3" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>PyTorch 1.8.1 features adapted to NPUs</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>Plugin decoupling</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>Decoupled the NPU-adapted code and ported it to the **torch_npu** plugin to decouple the Ascend-adapted code from the native PyTorch code.</p>
</td>
<tr>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>AOE-based tuning</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>Supported the AOE auto tuning tool to improve the model performance.</p>
</td>
</tr>
<tr id="row109719353511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18983183315515"><a name="p18983183315515"></a><a name="p18983183315515"></a>Improvement of framework API satisfaction</p>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p836221112812"><a name="p836221112812"></a><a name="p836221112812"></a>Supported APIs that have been adapted to Ascend PyTorch 1.5.0. For details, see the API list.</p>
</td>
</tr>
</tbody>
</table>





### Modified Features

N/A

### Resolved Issues

N/A

### Known Issues

<a name="table1969972073016"></a>

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>Known Issue</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>Symptom</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>Data type support</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPUs do not support the input or output of the inf/nan data of the float16 type.</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>Data format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>Dimensions cannot be reduced when the format larger than 4D is used.</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>Restrictions on collective communication</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>The graphs executed on different devices in a training job must be the same.</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>Allocation at only 1, 2, 4, or 8 processors is supported.</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>Only the int8, int32, float16, and float32 data types are supported.</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex function</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>In the current version, Apex is implemented mainly using Python, and the customized optimization of CUDA kernel in Apex is not supported.</p>
</td>
</tr>
</tbody>
</table>



### Compatibility

Atlas 800 (model 9010): CentOS 7.6, Ubuntu 18.04/2.04, BC-Linux 7.6, Debian 9.9, Debian 10, openEuler 20.03 LTS

Atlas 800 (model 9000): CentOS 7.6, Ubuntu 18.04/2.04, EulerOS 2.8/2.10, Kylin V10, BC-Linux 7.6, openEuler 20.03 LTS, UOS 20 1020e

## FrameworkPTAdapter 2.0.4
### Before You Start

This framework is modified based on the open-source PyTorch 1.5.0 and 1.8.1 developed by Facebook, inherits native PyTorch features, and uses NPUs for dynamic image training. Models are adapted by operator granularity, code can be reused, and current networks can be ported and used on NPUs with only device types or data types modified.

PyTorch 1.8.1 inherits the features of PyTorch 1.5.0. Their functions are basically the same, but PyTorch 1.8.1 provides better development experience for backend operator adaptation. Currently, PyTorch 1.8.1 supports only the ResNet-50 network model.
### New Features

**Table 1** Features supported by PyTorch

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.459999999999999%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>Level-1 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="26.27%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>Level-2 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="63.27%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row7979351559"><td class="cellrowborder" rowspan="3" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>PyTorch 1.5.0 features adapted to NPUs</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>Optimization of Model Accuracy Analyzer</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>Supported mapping between IR and TBE operators and enabled the NPU dump data to be loaded to the GPU side for comparison.</p>
</td>
</tr>
<tr id="row109719353511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18983183315515"><a name="p18983183315515"></a><a name="p18983183315515"></a>E2E prof</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p836221112812"><a name="p836221112812"></a><a name="p836221112812"></a>Integrated the profile data obtained by the native PyTorch Profiling tool and CANN prof tool to implement end-to-end model and operator performance analysis.</p>
</td>
</tr>
<tr id="row13971435754"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p79831331353"><a name="p79831331353"></a><a name="p79831331353"></a>Basic framework functions</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a>Added the function of adapted operator development. For details, see the operator list.</p>
</td>
</tr>
    <tr id="row7979351559"><td class="cellrowborder" rowspan="2" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>PyTorch 1.8.1 features adapted to NPUs</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>AMP</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>Supported the native training with automatic mixed precision (AMP) of PyTorch.</p>
</td>
</tr>
<tr id="row109719353511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18983183315515"><a name="p18983183315515"></a><a name="p18983183315515"></a>Profiling</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p836221112812"><a name="p836221112812"></a><a name="p836221112812"></a>Supported the native profiling function of PyTorch.</p>
</td>
</tr>
    <tr id="row7979351559"><td class="cellrowborder" rowspan="2" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>OS compatibility</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>OS compatibility</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>Supported Ubuntu 20.04 (x86 and ARM) and EulerOS 2.10 (ARM).</p>
</td>
<tr id="row13971435754"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p79831331353"><a name="p79831331353"></a><a name="p79831331353"></a>Python version compatibility</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a>Supported compilation and use of Python 3.9 (only in PyTorch 1.8.1).</p>
</td>
</tr>
</tbody>
</table>

### Modified Features

N/A
### Resolved Issues

N/A
### Known Issues

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>Known Issue</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>Data type support</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPUs do not support the input or output of the inf/nan data of the float16 type.</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>Data format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>Dimensions cannot be reduced when the format larger than 4D is used.</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>Restrictions on collective communication</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>The graphs executed on different devices in a training job must be the same.</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>Allocation at only 1, 2, 4, or 8 processors is supported.</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>Only the int8, int32, float16, and float32 data types are supported.</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex function</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>In the current version, Apex is implemented mainly using Python, and the customized optimization of CUDA kernel in Apex is not supported.</p>
</td>
</tr>
</tbody>
</table>

### Compatibility

Atlas 800 (model 9010): CentOS 7.6, Ubuntu 18.04/2.04, BC-Linux 7.6, Debian 9.9, Debian 10, openEuler 20.03 LTS

Atlas 800 (model 9000): CentOS 7.6, Ubuntu 18.04/2.04, EulerOS 2.8/2.10, Kylin V10, BC-Linux 7.6, openEuler 20.03 LTS, UOS 20 1020e
## FrameworkPTAdapter 2.0.3
### Before You Start

This framework is modified based on the open-source PyTorch 1.5.0 developed by Facebook, inherits native PyTorch features, and uses NPUs for dynamic image training. Models are adapted by operator granularity, code can be reused, and current networks can be ported and used on NPUs with only device types or data types modified.

PyTorch 1.8.1 is supported by this version and later, and this version inherits the features of PyTorch 1.5.0 and provides the same functions, except for the Profiling tool. In addition, it optimizes the backend operator adaptation. Currently, PyTorch 1.8.1 supports only the ResNet-50 network model.
### New Features

**Table  1**  Features supported by  PyTorch

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.459999999999999%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>Level-1 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="26.27%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>Level-2 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="63.27%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row7979351559"><td class="cellrowborder" rowspan="5" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p69836331553"><a name="p69836331553"></a><a name="p69836331553"></a>PyTorch features adapted to NPUs</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p149831333357"><a name="p149831333357"></a><a name="p149831333357"></a>PyTorch 1.8.1</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p1398313336511"><a name="p1398313336511"></a><a name="p1398313336511"></a>Added PyTorch 1.8.1. Currently, only the ResNet-50 network is supported, including the training scenario for distributed data parallel (DDP).</p>
</td>
</tr>
<tr id="row109719353511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18983183315515"><a name="p18983183315515"></a><a name="p18983183315515"></a>Python 3.8</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p836221112812"><a name="p836221112812"></a><a name="p836221112812"></a>Supported compilation and use of Python 3.8.</p>
</td>
</tr>
<tr id="row13971435754"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p79831331353"><a name="p79831331353"></a><a name="p79831331353"></a>Operator overflow/underflow detection tool</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p109832331055"><a name="p109832331055"></a><a name="p109832331055"></a>Supported IR-level operator overflow/underflow detection in the PyTorch framework. When an AI Core operator overflow/underflow occurs, the IR information is displayed.</p>
</td>
</tr>
<tr id="row185381431133610"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p953883153611"><a name="p953883153611"></a><a name="p953883153611"></a>OS compatibility</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p13538203153618"><a name="p13538203153618"></a><a name="p13538203153618"></a>Supported UOS 20 1020e ARM.</p>
</td>
</tr>
<tr id="row91681125173610"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2168172519364"><a name="p2168172519364"></a><a name="p2168172519364"></a>Basic framework functions</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1416815259367"><a name="p1416815259367"></a><a name="p1416815259367"></a>Added the function of adapted operator development. For details, see the operator list.</p>
</td>
</tr>
<tr id="row11970351050"><td class="cellrowborder" rowspan="26" valign="top" width="10.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p298314333514"><a name="p298314333514"></a><a name="p298314333514"></a>Model training</p>
</td>
<td class="cellrowborder" valign="top" width="26.27%" headers="mcps1.2.4.1.2 "><p id="p19984193312511"><a name="p19984193312511"></a><a name="p19984193312511"></a>CenterFace</p>
</td>
<td class="cellrowborder" valign="top" width="63.27%" headers="mcps1.2.4.1.3 "><p id="p7517351258"><a name="p7517351258"></a><a name="p7517351258"></a>-</p>
</td>
</tr>
<tr id="row9965351254"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1984033458"><a name="p1984033458"></a><a name="p1984033458"></a>PCBU</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p298411338513"><a name="p298411338513"></a><a name="p298411338513"></a>-</p>
</td>
</tr>
<tr id="row14966351959"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p198414331957"><a name="p198414331957"></a><a name="p198414331957"></a>Net++</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1798413331756"><a name="p1798413331756"></a><a name="p1798413331756"></a>-</p>
</td>
</tr>
<tr id="row20966351454"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p4984183317518"><a name="p4984183317518"></a><a name="p4984183317518"></a>FCN8S</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1698453311519"><a name="p1698453311519"></a><a name="p1698453311519"></a>-</p>
</td>
</tr>
<tr id="row19653517518"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1298483319512"><a name="p1298483319512"></a><a name="p1298483319512"></a>OSNetRetinaFace</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1798423312519"><a name="p1798423312519"></a><a name="p1798423312519"></a>-</p>
</td>
</tr>
<tr id="row39619351751"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1298414331513"><a name="p1298414331513"></a><a name="p1298414331513"></a>PSPnet</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10984633254"><a name="p10984633254"></a><a name="p10984633254"></a>-</p>
</td>
</tr>
<tr id="row496335654"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19984433351"><a name="p19984433351"></a><a name="p19984433351"></a>EDSR</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p898473314517"><a name="p898473314517"></a><a name="p898473314517"></a>-</p>
</td>
</tr>
<tr id="row17953357517"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p14984123312515"><a name="p14984123312515"></a><a name="p14984123312515"></a>Tsm</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p159841033855"><a name="p159841033855"></a><a name="p159841033855"></a>-</p>
</td>
</tr>
<tr id="row16951435551"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19984933457"><a name="p19984933457"></a><a name="p19984933457"></a>pnasnet5large</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p89846335512"><a name="p89846335512"></a><a name="p89846335512"></a>-</p>
</td>
</tr>
<tr id="row2095113520514"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1798416331952"><a name="p1798416331952"></a><a name="p1798416331952"></a>Gaitset</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p098417336516"><a name="p098417336516"></a><a name="p098417336516"></a>-</p>
</td>
</tr>
<tr id="row89516351511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p11985153312516"><a name="p11985153312516"></a><a name="p11985153312516"></a>fcn</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p169858331155"><a name="p169858331155"></a><a name="p169858331155"></a>-</p>
</td>
</tr>
<tr id="row5953353513"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p18985103311514"><a name="p18985103311514"></a><a name="p18985103311514"></a>Albert</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2098513334519"><a name="p2098513334519"></a><a name="p2098513334519"></a>-</p>
</td>
</tr>
<tr id="row119533516513"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p598511336511"><a name="p598511336511"></a><a name="p598511336511"></a>AdvancedEast</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p119850331513"><a name="p119850331513"></a><a name="p119850331513"></a>-</p>
</td>
</tr>
<tr id="row39511356512"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p698553311512"><a name="p698553311512"></a><a name="p698553311512"></a>ReidStrongBaseline</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1198514331659"><a name="p1198514331659"></a><a name="p1198514331659"></a>-</p>
</td>
</tr>
<tr id="row129463518517"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p99851633158"><a name="p99851633158"></a><a name="p99851633158"></a>Fast-scnn</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1298514331359"><a name="p1298514331359"></a><a name="p1298514331359"></a>-</p>
</td>
</tr>
<tr id="row9942035757"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p109854331151"><a name="p109854331151"></a><a name="p109854331151"></a>RDN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1198583317511"><a name="p1198583317511"></a><a name="p1198583317511"></a>-</p>
</td>
</tr>
<tr id="row79415357517"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6985833351"><a name="p6985833351"></a><a name="p6985833351"></a>SRFlow</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p29851033350"><a name="p29851033350"></a><a name="p29851033350"></a>-</p>
</td>
</tr>
<tr id="row12947357514"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p149859334518"><a name="p149859334518"></a><a name="p149859334518"></a>MGN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19851833850"><a name="p19851833850"></a><a name="p19851833850"></a>-</p>
</td>
</tr>
<tr id="row3941735855"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p898533316518"><a name="p898533316518"></a><a name="p898533316518"></a>Roberta</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p998553318518"><a name="p998553318518"></a><a name="p998553318518"></a>-</p>
</td>
</tr>
<tr id="row1093335856"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1985533451"><a name="p1985533451"></a><a name="p1985533451"></a>RegNetY</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p59851033355"><a name="p59851033355"></a><a name="p59851033355"></a>-</p>
</td>
</tr>
<tr id="row49373518516"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p19986173311512"><a name="p19986173311512"></a><a name="p19986173311512"></a>VoVNet-39</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p798683310515"><a name="p798683310515"></a><a name="p798683310515"></a>-</p>
</td>
</tr>
<tr id="row5930351357"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p189862336515"><a name="p189862336515"></a><a name="p189862336515"></a>RegNetX</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1398618331651"><a name="p1398618331651"></a><a name="p1398618331651"></a>-</p>
</td>
</tr>
<tr id="row169312351655"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p149863337513"><a name="p149863337513"></a><a name="p149863337513"></a>RefineNet</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p2098615331959"><a name="p2098615331959"></a><a name="p2098615331959"></a>-</p>
</td>
</tr>
<tr id="row11931235957"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1198616331657"><a name="p1198616331657"></a><a name="p1198616331657"></a>RefineDet</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1698611332518"><a name="p1698611332518"></a><a name="p1698611332518"></a>-</p>
</td>
</tr>
<tr id="row189215359511"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p16986183319512"><a name="p16986183319512"></a><a name="p16986183319512"></a>AlignedReID</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1898715331055"><a name="p1898715331055"></a><a name="p1898715331055"></a>-</p>
</td>
</tr>
<tr id="row89213351858"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p119878330511"><a name="p119878330511"></a><a name="p119878330511"></a>FaceBoxes</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p129878331158"><a name="p129878331158"></a><a name="p129878331158"></a>-</p>
</td>
</tr>
</tbody>
</table>

### Modified Features

N/A
### Resolved Issues

N/A
### Known Issues

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>Known Issue</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>Data type support</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPUs do not support the input or output of the inf/nan data of the float16 type.</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>Data format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>Dimensions cannot be reduced when the format larger than 4D is used.</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>Restrictions on collective communication</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>The graphs executed on different devices in a training job must be the same.</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>Allocation at only 1, 2, 4, or 8 processors is supported.</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>Only the int8, int32, float16, and float32 data types are supported.</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex function</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>In the current version, Apex is implemented mainly using Python, and the customized optimization of CUDA kernel in Apex is not supported.</p>
</td>
</tr>
</tbody>
</table>

### Compatibility

Atlas 800 (model 9010): CentOS 7.6, Ubuntu 18.04, BC-Linux 7.6, Debian 9.9, Debian 10, openEuler 20.03 LTS

Atlas 800 (model 9000): CentOS 7.6, Euler 2.8, Kylin v10, BC-Linux 7.6, openEuler 20.03 LTS, UOS 20 1020e

## FrameworkPTAdapter 2.0.2
### Before You Start

This framework is modified based on the open-source PyTorch 1.5.0 primarily developed by Facebook, inherits native PyTorch features, and uses NPUs for dynamic image training. Models are adapted by operator granularity, code can be reused, and current networks can be ported and used on NPUs with only device types or data types modified.
### New Features

**Table  1**  Features supported by  PyTorch

<table><thead align="left"><tr id="r0c10e7163bf54fe8816ab5ca2d77ccc4"><th class="cellrowborder" valign="top" width="10.489999999999998%" id="mcps1.2.4.1.1"><p id="a7888762cf8294977b7d114b1c898d1bd"><a name="a7888762cf8294977b7d114b1c898d1bd"></a><a name="a7888762cf8294977b7d114b1c898d1bd"></a>Level-1 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="26.3%" id="mcps1.2.4.1.2"><p id="a4581ffde4a5f455faadfba144243a9d4"><a name="a4581ffde4a5f455faadfba144243a9d4"></a><a name="a4581ffde4a5f455faadfba144243a9d4"></a>Level-2 Feature</p>
</th>
<th class="cellrowborder" valign="top" width="63.21%" id="mcps1.2.4.1.3"><p id="a2a1562364b09433a83133fa10b3cf2b3"><a name="a2a1562364b09433a83133fa10b3cf2b3"></a><a name="a2a1562364b09433a83133fa10b3cf2b3"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row171322953713"><td class="cellrowborder" rowspan="7" valign="top" width="10.489999999999998%" headers="mcps1.2.4.1.1 "><p id="p10237517181"><a name="p10237517181"></a><a name="p10237517181"></a>Adapted training models</p>
</td>
<td class="cellrowborder" valign="top" width="26.3%" headers="mcps1.2.4.1.2 "><p id="p6134294377"><a name="p6134294377"></a><a name="p6134294377"></a>YOLOv4</p>
</td>
<td class="cellrowborder" valign="top" width="63.21%" headers="mcps1.2.4.1.3 "><p id="p314533811397"><a name="p314533811397"></a><a name="p314533811397"></a>-</p>
</td>
</tr>
<tr id="row15990182233714"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p399019224374"><a name="p399019224374"></a><a name="p399019224374"></a>YOLOv3</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1714553816396"><a name="p1714553816396"></a><a name="p1714553816396"></a>-</p>
</td>
</tr>
<tr id="row5301255373"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1530325123719"><a name="p1530325123719"></a><a name="p1530325123719"></a>DB</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5146193819390"><a name="p5146193819390"></a><a name="p5146193819390"></a>-</p>
</td>
</tr>
<tr id="row274203413712"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p197423343379"><a name="p197423343379"></a><a name="p197423343379"></a>RFCN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15828941153919"><a name="p15828941153919"></a><a name="p15828941153919"></a>-</p>
</td>
</tr>
<tr id="row78671918163714"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p188671318183713"><a name="p188671318183713"></a><a name="p188671318183713"></a>CRNN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p10828241123917"><a name="p10828241123917"></a><a name="p10828241123917"></a>-</p>
</td>
</tr>
<tr id="row16912221171"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p6913132676"><a name="p6913132676"></a><a name="p6913132676"></a>Densenset161</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p209131221474"><a name="p209131221474"></a><a name="p209131221474"></a>-</p>
</td>
</tr>
<tr id="row1016073314719"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p3160153320717"><a name="p3160153320717"></a><a name="p3160153320717"></a>Densenset191</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1516012331173"><a name="p1516012331173"></a><a name="p1516012331173"></a>-</p>
</td>
</tr>
<tr id="row9627165910386"><td class="cellrowborder" rowspan="4" valign="top" width="10.489999999999998%" headers="mcps1.2.4.1.1 "><p id="p1561535993811"><a name="p1561535993811"></a><a name="p1561535993811"></a>PyTorch features adapted to NPUs</p>
</td>
<td class="cellrowborder" valign="top" width="26.3%" headers="mcps1.2.4.1.2 "><p id="p136151659163819"><a name="p136151659163819"></a><a name="p136151659163819"></a>Basic framework functions</p>
</td>
<td class="cellrowborder" valign="top" width="63.21%" headers="mcps1.2.4.1.3 "><p id="p1661535983813"><a name="p1661535983813"></a><a name="p1661535983813"></a>Added the function of adapted operator development. For details, see the operator list.</p>
</td>
</tr>
<tr id="row7627155917380"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1561525916383"><a name="p1561525916383"></a><a name="p1561525916383"></a>Model Accuracy Analyzer</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19616115953816"><a name="p19616115953816"></a><a name="p19616115953816"></a>Added the Model Accuracy Analyzer and supported training accuracy demarcation.</p>
</td>
</tr>
<tr id="row46269593383"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p0616559163819"><a name="p0616559163819"></a><a name="p0616559163819"></a>Ascend 710 AI Processor</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p186161459193813"><a name="p186161459193813"></a><a name="p186161459193813"></a>Supported the online inference on Ascend 710 AI Processors.</p>
</td>
</tr>
<tr id="row76261059153817"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p11616175953812"><a name="p11616175953812"></a><a name="p11616175953812"></a>OS compatibility</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p8935115971714"><a name="p8935115971714"></a><a name="p8935115971714"></a>Supported Ubuntu 18.04.5 and openEuler 20.03 LTS.</p>
</td>
</tr>
</tbody>
</table>

### Modified Features

N/A
### Resolved Issues

N/A
### Known Issues

<table><thead align="left"><tr id="row3699162017307"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.1.3.1.1"><p id="p16992020153010"><a name="p16992020153010"></a><a name="p16992020153010"></a>Known Issue</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.1.3.1.2"><p id="p269919203308"><a name="p269919203308"></a><a name="p269919203308"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="row9699142003011"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1769932017300"><a name="p1769932017300"></a><a name="p1769932017300"></a>Data type support</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p13699152010301"><a name="p13699152010301"></a><a name="p13699152010301"></a>NPUs do not support the input or output of the inf/nan data of the float16 type.</p>
</td>
</tr>
<tr id="row146991520153016"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p156997200308"><a name="p156997200308"></a><a name="p156997200308"></a>Data format</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p10699182020308"><a name="p10699182020308"></a><a name="p10699182020308"></a>Dimensions cannot be reduced when the format larger than 4D is used.</p>
</td>
</tr>
<tr id="row11121205610549"><td class="cellrowborder" rowspan="3" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p1647216219558"><a name="p1647216219558"></a><a name="p1647216219558"></a>Restrictions on collective communication</p>
<p id="p0465121912402"><a name="p0465121912402"></a><a name="p0465121912402"></a></p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p3116115695415"><a name="p3116115695415"></a><a name="p3116115695415"></a>The graphs executed on different devices in a training job must be the same.</p>
</td>
</tr>
<tr id="row51211656105411"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p1311616560541"><a name="p1311616560541"></a><a name="p1311616560541"></a>Allocation at only 1, 2, 4, or 8 processors is supported.</p>
</td>
</tr>
<tr id="row8647195765419"><td class="cellrowborder" valign="top" headers="mcps1.1.3.1.1 "><p id="p2064225716544"><a name="p2064225716544"></a><a name="p2064225716544"></a>Only the int8, int32, float16, and float32 data types are supported.</p>
</td>
</tr>
<tr id="row4646195719548"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.1.3.1.1 "><p id="p4642195718541"><a name="p4642195718541"></a><a name="p4642195718541"></a>Apex function</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.1.3.1.2 "><p id="p864205725412"><a name="p864205725412"></a><a name="p864205725412"></a>In the current version, Apex is implemented mainly using Python, and the customized optimization CUDA kernel in Apex is not supported.</p>
</td>
</tr>
</tbody>
</table>

### Compatibility

Atlas 800 (model 9010): CentOS 7.6/Ubuntu 18.04/BC-Linux 7.6/Debian 9.9/Debian 10/openEuler 20.03 LTS

Atlas 800 (model 9000): CentOS 7.6/EulerOS 2.8/Kylin v10/BC-Linux 7.6/openEuler 20.03 LTS

