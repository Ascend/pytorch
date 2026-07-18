# Release Notes

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-14T00:44:37.555Z pushedAt=2026-06-14T07:51:04.582Z -->

## Version Mapping

### Product Version Information

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph4778145519911">Ascend Extension for PyTorch</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">26.0.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">Official Release</p>
</td>
</tr>
<tr id="row754461214611"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p155445122062">Release Date</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p135443128613">April 2026</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">Maintenance Period</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">For details, see the <a href="https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/README.md">Branch Maintenance Strategy</a></p>
</td>
</tr>
</tbody>
</table>

### Related Product Version Mapping

Ascend Extension for PyTorch branch names follow the naming convention **\{PyTorch version\}-\{Ascend version\}**, where the former indicates the compatible PyTorch version and the latter indicates the extension version. See the table below for the detailed mapping.

|CANN Version|PyTorch Version|Ascend Extension for PyTorch Version|Ascend Extension for PyTorch Code Branch Name|Ascend Extension for PyTorch Installation Package Version|Python Version|
|--|--|--|--|--|--|
|Commercial Edition: 9.0.0<br>Community Edition: 9.0.0|2.7.1|26.0.0|v2.7.1-26.0.0|2.7.1.post4|Python3.9.*x*, Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|Commercial Edition: 9.0.0<br>Community Edition: 9.0.0|2.8.0|26.0.0|v2.8.0-26.0.0|2.8.0.post4|Python3.9.*x*, Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|Commercial Edition: 9.0.0<br>Community Edition: 9.0.0|2.9.0|26.0.0|v2.9.0-26.0.0|2.9.0.post2|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|Commercial Edition: 9.0.0<br>Community Edition: 9.0.0|2.10.0|26.0.0|v2.10.0-26.0.0|2.10.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|

## Version Compatibility

|Ascend Extension for PyTorch Version|CANN Version|Related Product Version Compatibility|
|--|--|--|
|26.0.0|CANN 9.0.0<br>CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1|-|
|7.3.0|CANN 8.5.0<br>CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0|Driving SDK 7.3.0 is compatible with CANN 8.1.RC1, CANN 8.2.RC1, CANN 8.3.RC1, CANN 8.5.0|
|7.2.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|Driving SDK 7.2.RC1 is compatible with CANN 8.1.RC1, CANN 8.2.RC1, CANN 8.3.RC1|
|7.1.0|CANN 8.3.RC1<br>CANN 8.2.RC1<br>CANN 8.1.RC1<br>CANN 8.0.0<br>CANN 8.0.RC3<br>CANN 8.0.RC2|Driving SDK 7.1.RC1 is compatible with CANN 8.0.0, CANN 8.1.RC1, and CANN 8.2.RC1|

## Release Notes

### New Features

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1">Component</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">Description</th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3">Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="6" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1"><span>Ascend Extension for PyTorch</span> (torch-npu)</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">P2P communication supports group-based dispatch.</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">batch_isend_irecv supports group-based dispatch.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">DTensor strategy expansion.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Added support for multiple operator splitting types, increasing operator coverage to over 99%.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Cross-stream memory deferred release optimization.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">When memory is sufficient, reduces the overhead of querying cross-stream memory release status, shortening runtime and improving execution performance.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Event pool management for D2D copy.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Implements pooled management of events in cross-device copy operations, reducing frequent creation overhead and improving performance.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Enhanced event capabilities, supporting cross-process and cross-device usage.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Supports event synchronization in cross-process shared memory and cross-device copy scenarios, improving overall performance.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Host allocator supports background thread optimization for query time.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Uses a background thread to periodically query event completion status, improving overall memory allocation performance.</td>
    </tr>
  </tbody>
</table>

### Removed Features

None

### API Changes

This section covers API changes, including additions, modifications, deprecations, and removals. These changes reflect only code-level updates. Documentation improvements, such as language, formatting, links, or other non-code enhancements, are not included here.

- Added: New APIs introduced in this version
- Modified: APIs that have changed compared to the previous version
- Deprecated: APIs that are no longer evolving, may be removed one year after the deprecation announcement
- Deleted: APIs that have been removed in this version

**Table 1** Summary of API changes for Ascend Extension for PyTorch

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="11.53%" id="mcps1.2.6.1.1">Changed Version</th>
      <th class="cellrowborder" valign="top" width="37.68%" id="mcps1.2.6.1.2">Class Name/API Prototype</th>
      <th class="cellrowborder" valign="top" width="15.22%" id="mcps1.2.6.1.3">Class/API Category</th>
      <th class="cellrowborder" valign="top" width="15.32%" id="mcps1.2.6.1.4">Change Category</th>
      <th class="cellrowborder" valign="top" width="20.25%" id="mcps1.2.6.1.5">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="18" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.NpuGraphOpHandler</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">&#8226; torch_npu functionality refactored<br>&#8226; Not dependent on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_block_sparse_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dense_lightning_indexer_grad_kl_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dense_lightning_indexer_softmax_lse</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_floyd_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention_v3</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_matmul_gelu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.save_npugraph_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 or later</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_attention_to_ffn</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_attention_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Not dependent on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_recurrent_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.matmul.cube_math_type</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 or later</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_swiglu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.3.0RC1 or later</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_clipped_swiglu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Not dependent on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">&#8226; Added optional parameters: dropout_mask, seed, offset<br>&#8226; Not dependent on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.8.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">Changes are the same as v2.7.1</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.9.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">Changes are the same as v2.7.1</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.10.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">Changes are the same as v2.7.1</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
> Ascend Extension for PyTorch has added support for some APIs and features. For details, see [Ascend Extension for PyTorch Custom API Reference](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/en/custom_APIs/overview.md) or [PyTorch Native API Support](../native_apis/pytorch_2-10-0/overview.md).

### Resolved Issues

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">Issue Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">Symptom</strong>: In specific scenarios, an NN process issue may occur, causing OOM on the networks of two ICs (compute instances).</p>
<p id="p15983141916104"><strong id="b1598312196108">Impact</strong>: NN process errors may occur in some environments.</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">Severity Level</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">Minor</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">Root Cause Analysis</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">The get_device_properties API is obtained by calling the cudaGetDeviceProperties API, and multiple APIs are called on the NPU to obtain the corresponding properties. Among them, the device memory information depends on aclGetMemInfo, but this API must be bound to a context before it can be called, resulting in continuous occupation of card 0 memory.</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">Solution</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019">The get_device_properties API is aligned with the aclrtGetMenInfo API.</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">Modification Impact</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">After the fix, the get_device_properties API is aligned with the aclrtGetMenInfo API, eliminating the potential risk of resource occupation.</p>
</td>
</tr>
</tbody>
</table>

### Known Issues

None

## Upgrade Impact

### Impact of the Upgrade on the Current System

- Impact on services

    The software version upgrade process will cause service interruption.

- Impact on network communication

    No impact on communication.

### Post-Upgrade Impact on the Current System

None

## Version Mapping Documents

|Document Name|Description|Release Notes|
|---|---|---|
|[Ascend Extension for PyTorch Software Installation Guide](../installation_guide/installation_description.md)|Provides instructions for installing the PyTorch framework training environment on Ascend devices, as well as upgrade and uninstallation operations.|&#8226; Added adaptation for PyTorch 2.10.0.<br>&#8226; Added support for Python 3.13.|
|[PyTorch Model Migration and Tuning Guide](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/pytorch_model_migration_fine_tuning/overview.md)|Includes guidance on model migration and tuning, precision issue location, and performance problem resolution, along with a library of common model examples.|Added adaptation for PyTorch 2.10.0.|
|[PyTorch Feature Guide](../framework_feature_guide_pytorch/overview.md)|Leverages the powerful computing capabilities of Ascend AI processors through Ascend Extension for PyTorch, creating a series of unique features in areas such as memory optimization, error location, and high-performance computing.|&#8226; Added content related to "NPUGraph".<br>&#8226; Updated content related to "Operator Adaptation Development Based on C++ Extension".<br>&#8226; Optimized content related to "Operator Adaptation Development Based on OpPlugin".|
|[PyTorch Graph Mode Usage Guide (TorchAir)](https://gitcode.com/Ascend/torchair/blob/26.0.0/docs/en/overview.md)|As a graph mode capability extension library for Ascend Extension for PyTorch, it provides an Ascend device-friendly torch.compile graph mode backend, enabling graph mode inference acceleration and optimization for PyTorch networks on Ascend NPUs.|&#8226; Added graph mode for the npugraph_ex backend, enabling single capture and multiple execution of tasks through the Capture&Replay approach. The original reduce-overhead mode, which configured the graph editing backend via config.mode, will no longer evolve and is no longer recommended.<br>&#8226; Enhanced GE mode features, including specifying the scope of dump operators and calibrating the SuperKernel scope within graphs.|
|[Ascend Extension for PyTorch Custom API Reference](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/en/custom_APIs/overview.md)|Provides function prototypes, feature descriptions, parameter descriptions, and call examples for Ascend Extension for PyTorch custom APIs.|&#8226; Added adaptation for PyTorch 2.10.0.<br>&#8226; For specific API Changes, refer to [API Changes](#api-changes).|
|[PyTorch Native API Support](../native_apis/pytorch_2-10-0/overview.md)|Provides the support status of native APIs for PyTorch versions 2.10.0/2.9.0/2.8.0/2.7.1 on Ascend devices.|Added the PyTorch 2.10.0 native API support list.|
|[Supported Suites and Third-Party Libraries](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.0.0/en/supported_suites_and_third_party_libraries/supported_suites_and_third_party_libraries.md)|Introduces model suites and acceleration libraries supported by Ascend devices, third-party libraries natively supported by Ascend, and Ascend self-developed plugins.|No new additions.|
|[Environment Variable Reference](../environment_variable_reference/env_variable_list.md)|Environment variables that can be used during Ascend Extension for PyTorch training and online inference.|&#8226; Added "TORCH_NPU_USE_COMPATIBLE_IMPL".<br>&#8226; Added "TORCH_NPU_LOGS".<br>&#8226; Added "TORCH_NPU_LOGS_FILTER".<br>&#8226; Added "TORCH_NPU_DEVICE_CAPABILITY".<br>&#8226; Added "TORCH_TRANSFER_TO_NPU".<br>&#8226; Added "TORCHINDUCTOR_NPU_BACKEND".<br>&#8226; Added "INDUCTOR_ASCEND_CHECK_ACCURACY".|

## Virus Scan and Vulnerability Patch List

### Virus Scan Results

|Antivirus Software|Antivirus Software Version|Virus Database Version|Scan Time|Scan Results|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-04-01 08:00:00.0|2026-04-02|Virus-free and Malware-free|
|Kaspersky|12.0.0.6672|2026-04-02 10:05:00.0|2026-04-02|Virus-free and Malware-free|
|Bitdefender|7.5.1.200224|7.100588|2026-04-02|Virus-free and Malware-free|

### Vulnerability Patch List

No vulnerability fixes in this version.
