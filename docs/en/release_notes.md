# Version Notes

## Key Features

- Supports the alltoallvc communication operator, optimizing alltoall communication performance.
- Supports initializing communicators through hierarchical link establishment, optimizing performance.
- Supports Inductor's automatic operator fusion based on the Ascend C IR.
- Inherits DeviceAllocator and supports enabling accelerator memory and graph-related interfaces in Eager mode.
- Starting from this version, the Torch-MLIR compiler in the Inductor backend is deprecated, affecting the *Torch.compile* and *Environment Variables* documents. For the specific changes, refer to [Related Documents for This Version](#related-documents-for-this-version).
- Starting from this version, the compilation optimization functions provided based on the BiSheng compiler are deprecated and can be replaced with [Clang](../zh/installation_guide/references/building_with_clang.md).

## Version Compatibility Notes

### Product Version Information

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph4778145519911">TorchNPU</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">26.2.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">Official Version</p>
</td>
</tr>
<tr id="row754461214611"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p155445122062">Release Date</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p135443128613">October 2026</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">Maintenance Period</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">Refer to <a href="https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/README.zh.md#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5">Branch Maintenance Policy</a></p>
</td>
</tr>
</tbody>
</table>

### Version Compatibility Notes for Related Products

The version compatibility table of firmware and drivers is related to all Ascend hardware and CANN versions. For specific selection, refer to [CANN Version Notes](https://gitcode.com/cann/release-management/blob/master/9.2.0/release-notes.md).

To expand TorchNPU capabilities, for the version requirements of the self-developed plugins provided by Ascend, refer to [Compatible Software Libraries](./user_guide/libraries.md).

TorchNPU code branch names follow the naming rule **\{PyTorch version\}-\{TorchNPU version\}**, where the former is the PyTorch version matched by TorchNPU. The detailed matching is shown in the following table:

|TorchNPU Code Branch Name|PyTorch Version|TorchNPU Version|TorchNPU Installation Package Version|CANN Version|Python Version|Triton-Ascend Version|
|--|--|--|--|--|--|--|
|v2.7.1-26.2.0|2.7.1|26.2.0|2.7.1.post11|9.2.0|Python3.9.*x*, Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|v3.2.2|
|v2.9.0-26.2.0|2.9.0|26.2.0|2.9.0.post9|9.2.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*, Python3.14.*x*|No matching version|
|v2.10.0-26.2.0|2.10.0|26.2.0|2.10.0.post7|9.2.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*, Python3.14.*x*|v3.6.0|
|v2.11.0-26.2.0|2.11.0|26.2.0|2.11.0.post3|9.2.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*, Python3.14.*x*|No matching version|
|v2.12.0-26.2.0|2.12.0|26.2.0|2.12.0.post3|9.2.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*, Python3.14.*x*|No matching version|
|v2.13.0-26.2.0|2.13.0|26.2.0|2.13.0|9.2.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*, Python3.14.*x*|v3.6.0|
|v2.14.0-26.2.0|2.14.0|26.2.0|2.14.0|9.2.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*, Python3.14.*x*|No matching version|

## Version Compatibility Description

> [!NOTE]
>
> In the table, "Y" indicates compatibility.

<p style="display:none">
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg .tg-rhr9{font-weight:bold;text-align:center;vertical-align:middle}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>
</p>
<table class="tg"><thead>
  <tr>
    <th class="tg-rhr9" rowspan="2">TorchNPU</th>
    <th class="tg-amwm" colspan="4">CANN Version</th>
  </tr>
  <tr>
    <th class="tg-c3ow">8.5.X</th>
    <th class="tg-c3ow">9.0.X</th>
    <th class="tg-c3ow">9.1.X</th>
    <th class="tg-c3ow">9.2.X</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">7.3.X</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
  </tr>
  <tr>
    <td class="tg-baqh">26.0.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
    <tr>
    <td class="tg-baqh">26.1.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
  <tr>
    <td class="tg-baqh">26.2.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
</tbody>
</table>

## Update Description

### Added Feature Descriptions

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1">Component</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">Feature</th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3">Feature Description</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">Adaptation Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="19" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1">TorchNPU</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">DTensor completion.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Full coverage of sharding strategies for math ops; adds the linspace factory function on DeviceMesh.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.14.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">c10d architecture standardization.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">The backend discovery mechanism is refactored to dynamic registration through Python entrypoints.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.14.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Communication fault tolerance.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">NCCL2 adds the reconfigure capability, supporting communicator rebuilding for fault recovery.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.14.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Simplified compilation configuration.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Supports globally setting the default compile backend.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.14.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Expanded operator coverage.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">uniform enters the compiler decomposition path; shallow_copy_data_ is added to support cross-device tensor.data= assignment (the full stack from Dynamo to Inductor).</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.14.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Enhanced FSDP2 sharding flexibility.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Supports independent per-parameter sharding on different DeviceMeshes; fully_shard can explicitly declare the data parallel dimension on the SPMD mesh through DataParallelMeshDims.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.13.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">FSDP2 chunked loss。</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">The norm/head parts are computed in the forward pass, reducing the peak memory usage during large model training.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.13.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Expanded quantization support.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">The AOTInductor C shim layer supports MXFP4.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.13.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Recompilation control.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Adds recompile_limit to limit the number of recompilations of a single function at the call level.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.13.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Expanded operator coverage.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Inplace custom operators support compile; the scan output dimension uses movedim; BatchLinearLHSFusion recognizes more linear calls.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.13.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Mempool community feature follow-up.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">The architecture follows the community mainline, fixes the memory leak issue, supports a separate ID control mechanism, and catches up with the features newly added in the community.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.13.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Eager mode APIs support torch.accelerator.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Inherits DeviceAllocator and supports enabling accelerator memory and graph-related interfaces in Eager mode.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Added adaptation for PyTorch 2.13.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">torch.bitwise_right_shift and torch.bitwise_left_shift are supported on the NPU.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">torch.bitwise_right_shift and torch.bitwise_left_shift support computation on the NPU, avoiding fallback to the CPU.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Adapted for PyTorch 2.12.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Supports the Inductor Ascend C backend.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Supports Inductor's automatic operator fusion based on the Ascend C IR.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Adapted for PyTorch 2.11.0 and later versions</td>
    </tr>
      <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Standardized memory optimization configuration.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Refactors memory configuration items, reusing community common configurations implemented through a registration mechanism; supports dynamic configuration through the PYTORCH_ALLOC_CONF environment variable.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Adapted for PyTorch 2.10.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Supports the alltoallvc communication operator.</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">Supports the alltoallvc communication operator, optimizing alltoall communication performance.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">&#8226; Adapted for PyTorch 2.7.1<br>&#8226; Adapted for PyTorch 2.9.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Communicators support initialization through hierarchical link establishment.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Supports initializing communicators through hierarchical link establishment, optimizing performance.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">&#8226; Adapted for PyTorch 2.7.1<br>&#8226; Adapted for PyTorch 2.9.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Communication operators support symmetric memory.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Supports memory window registration; key communication operators support symmetric memory, improving data transmission efficiency.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">&#8226; Adapted for PyTorch 2.7.1<br>&#8226; Adapted for PyTorch 2.9.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Supports Batch consistency configuration.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">The torch_npu.npu.set_deterministic_level(level) interface adds enumeration value 3 to specify consistency at the Batch level; the get interface returns the current level. Both deterministic computation and Batch consistency support dynamic configuration.</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">&#8226; Adapted for PyTorch 2.7.1<br>&#8226; Adapted for PyTorch 2.9.0 and later versions</td>
    </tr>
  </tbody>
</table>

### Key Feature Changes

This version inherits all features of the 26.1.X version of this product.

### Interface Change Description <a id="interface-change-description"></a>

The interface change description in this section covers added, modified, deprecated, and deleted interfaces. Interface changes only reflect modifications at the code level and do not include optimizations or improvements to the document itself in terms of language, format, links, and so on.

- Added: indicates interfaces added in this version.
- Modified: indicates that the interface has been modified compared with the previous version.
- Deprecated: indicates that the interface stops evolving from the version in which the deprecation is announced and may be removed one year after the announcement.
- Deleted: indicates that the interface has been removed in this version.

**Table 1** TorchNPU Interface Change Summary

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="11.53%" id="mcps1.2.6.1.1">Changed Version</th>
      <th class="cellrowborder" valign="top" width="37.68%" id="mcps1.2.6.1.2">Class Name/API Prototype</th>
      <th class="cellrowborder" valign="top" width="15.22%" id="mcps1.2.6.1.3">Class/API Category</th>
      <th class="cellrowborder" valign="top" width="15.32%" id="mcps1.2.6.1.4">Change Category</th>
      <th class="cellrowborder" valign="top" width="20.25%" id="mcps1.2.6.1.5">Change Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="216" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.batch_norm_reduce</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.distributed.all_to_all_vc</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_quant_gmm_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_cast</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_dynamic_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_all_gather_quant_mm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_all_to_all_quant_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_anti_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_apply_adam_w</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dual_level_quant_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_block_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_mx_quant_with_dual_axis</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_cross_entropy_loss_with_max_sum</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_linear_cross_entropy_loss_with_max_sum_backward</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_linear_online_max_sum</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gelu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_dynamic_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_add</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_add_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_swiglu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kronecker_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kv_rmsnorm_rope_cache_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_token_permute</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_token_permute_with_routing_map</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">&#8226; On <term>Atlas A2 products</term>/<term>Atlas A3 products</term>, depends on CANN 8.5.0 and later versions<br>&#8226; On <term>Ascend 950PR&950DT</term>, depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_token_unpermute</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_token_unpermute_with_routing_map</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">&#8226; On <term>Atlas A2 products</term>/<term>Atlas A3 products</term>, depends on CANN 8.5.0 and later versions<br>&#8226; On <term>Ascend 950PR&950DT</term>, depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_gmm_alltoallv</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_matmul_all_to_all</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_mm_reduce_scatter</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>   
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_dynamic_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_quant_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scatter_pa_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_swiglu_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_weight_quant_preprocess</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.get_task_queue_enable</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_task_queue_enable</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.NPUPluggableAllocator</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.change_current_allocator</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.distributed.reduce_scatter_tensor_uneven</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.empty_with_swapped_memory</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.erase_stream</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.npu.get_device_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.npu.get_stream_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.matmul_checksum</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.ExternalEvent</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.SyncLaunchStream</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.aclnn.allow_hf32</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.are_compatible_impl_enabled</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.conv.allow_hf32</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.empty_virt_addr_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.graph_task_group_begin</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.graph_task_group_end</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.graph_task_update_begin</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.graph_task_update_end</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.host_empty_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.matmul.allow_hf32</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.matmul.cube_math_type</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx.mark</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx.mstx_range</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx.range_end</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx.range_start</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.mstx</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_deterministic_level</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_op_timeout_ms</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.use_compatible_impl</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_dynamic_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_all_gather_base_mm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_alltoallv_gmm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_anti_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_block_sparse_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_chunk_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_clipped_swiglu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_convert_weight_to_int4pack</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dense_lightning_indexer_grad_kl_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dense_lightning_indexer_softmax_lse</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dequant_swiglu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_block_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_quant_asymmetric</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fast_gelu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_infer_attention_score</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_infer_attention_score_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gelu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gmm_alltoallv</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_group_norm_silu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_group_norm_swish</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_finalize_routing</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_interleave_rope</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kv_quant_sparse_flash_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_kv_rmsnorm_rope_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mm_all_reduce_base</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mm_reduce_scatter_base</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_compute_expert_tokens</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_combine</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_combine_add_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_combine_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_dispatch</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_distribute_dispatch_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_finalize_routing</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_gating_top_k</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_gating_top_k_softmax</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_init_routing</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_init_routing_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_moe_update_expert</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mrope</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.NpuGraphOpHandler</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_scatter</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_scatter_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quantize</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_recurrent_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rotary_mul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rotate_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.2.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scatter_nd_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scatter_nd_update_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scatter_pa_kv_cache</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sim_exponential_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sparse_lightning_indexer_grad_kl_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_trans_quant_param</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_transpose_batchmatmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_weight_quant_batchmatmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedAdadelta</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedAdam</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedAdamP</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedAdamW</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedBertAdam</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedLamb</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedOptimizerBase</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedRMSprop</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedRMSpropTF</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.optim.NpuFusedSGD</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.AiCMetrics</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.ExportType</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.ProfilerAction</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.ProfilerActivity</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.ProfilerLevel</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler._ExperimentalConfig</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler._KinetoProfile</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.dynamic_profile.init</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.dynamic_profile.start</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.dynamic_profile.step</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.profile.disable_profiler_in_child_thread</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.profile.enable_profiler_in_child_thread</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.profile</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.profiler.analyse</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.schedule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.supported_activities</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.supported_ai_core_metrics</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.supported_export_type</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.profiler.supported_profiler_level</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.npu.reset_stream_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.scatter_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.scatter_update_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.npu.set_device_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.npu.set_stream_limit</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>    
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.get_cann_version</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.reset_thread_affinity</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.set_thread_affinity</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.distributed.ProcessGroupHCCL</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch.distributed.is_hccl_available</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu._npu_dropout</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.distributed.all_gather_into_tensor_uneven</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.distributed.reinit_process_group</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.jit.optimize</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.aclnn.version</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.check_uce_in_memory</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr> 
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.clear_npu_overflow_flag</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.config.allow_internal_format</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.get_amp_supported_dtype</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.get_autocast_dtype</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.get_mm_bmm_format_nd</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.is_autocast_enabled</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.is_jit_compile_false</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.restart_device</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_aoe</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_autocast_dtype</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_autocast_enabled</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_mm_bmm_format_nd</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.stop_device</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.stress_detect</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.utils.is_support_inf_nan</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.utils.npu_check_overflow</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_deformable_conv2d</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_linear</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_swiglu</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.FlopsCounter</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.get_part_combined_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.is_combined_tensor_valid</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.npu_combine_tensors</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.utils.save_async</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_cross_entropy_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_swiglu_quant_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.2.0</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_format_cast</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_format_cast_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.set_compile_mode</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.enable_deterministic_with_backward</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.disable_deterministic_with_backward</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.Event().recorded_time()</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.ExternalEvent().record()</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.ExternalEvent().reset()</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu.ExternalEvent().wait()</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_matmul_all_to_all</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_attention_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_all_to_all_matmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
   <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scaled_masked_softmax</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom Interface</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.9.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">The changes are the same as those in v2.7.1.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.10.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">The changes are the same as those in v2.7.1.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.11.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">The changes are the same as those in v2.7.1.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.12.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">The changes are the same as those in v2.7.1.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.13.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">The changes are the same as those in v2.7.1.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.14.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">The changes are the same as those in v2.7.1.</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> TorchNPU adds support for some APIs and features. For details, refer to [Custom API](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/overview.md) or [Native API](../zh/api/native_api/pytorch_2-14-0/overview.md).

### Resolved Issues

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">Issue Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">Symptom</strong>: When the multi_stream_lazy_reclaim feature is enabled, random Core Dumps occur.</p>
<p id="p15983141916104"><strong id="b1598312196108">Impact</strong>: The multi_stream_lazy_reclaim feature is unavailable.</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">Severity Level</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">Critical</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">Root Cause Analysis</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">In specific scenarios, free_block() in process_events() calls try_merge_blocks(), which may release params.block early after it has been obtained by get_free_block() but before it is marked as allocated. As a result, subsequent code accesses the freed memory (Use-After-Free), triggering a Core Dump.</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">Solution</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019">1. After get_free_block() returns a Block, immediately mark its status as allocated.<br>2. Move the logic for determining sum > kLazyQuerySize in process_events() to before the get_free_block() call.</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">Impact of the Change</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">After the fix, the feature works normally, and the Core Dump issue is fixed.</p>
</td>
</tr>
</tbody>
</table>

### Known Issues

None

## Upgrade Impact

### Impact on the Existing System During Upgrade

- Impact on services

    Service interruption may occur during the software version upgrade.

- Impact on network communication

    No impact on communication.

### Impact on the Existing System After Upgrade

None

## Related Documents for This Version <a id="related-documents-for-this-version"></a>

|Document Name|Description|Update Description|
|---|---|---|
|[Installation Guide](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=182&ids=89dda9ba9de741349efa03687a487678%2C202%2C106%2C1%2C6%2C177%2C)|Describes how to install the PyTorch framework training environment on Ascend devices, as well as operations such as upgrade and uninstallation.|&#8226; Added adaptation for PyTorch 2.13.0 and PyTorch 2.14.0.<br>&#8226; Added documentation about compilation acceleration and Clang compilation.|
|[TorchNPU Overview](../zh/user_guide/product_overview.md)|The TorchNPU plugin is a deep learning adaptation framework based on Ascend. It enables the Ascend NPU to support the PyTorch framework and provides PyTorch framework users with the powerful computing power of Ascend AI processors.|No updates |
|[Quick Start](../zh/user_guide/quick_start.md)|Provides a simple model migration example that uses the simplest automatic migration method to help users quickly experience the process of migrating GPU model scripts to the Ascend NPU.|Optimized the Quick Start document.|
|[Torch.compile](../zh/user_guide/torch_compile/overview.md)|Significantly accelerates model training and inference tasks through the approach of "dynamic graph capture + static graph optimization + efficient code generation".|&#8226; Added the Ascend C compiler to the Inductor backend.<br>&#8226; Added a link to the official PyTorch reference documentation.<br>&#8226; Deleted content about the Torch-MLIR compiler from the Inductor section|
|[Compatible Software Libraries](./user_guide/libraries.md)|Provides compatible software libraries that extend the capabilities of TorchNPU.|Added the Triton-Ascend component.|
|[Troubleshooting](../zh/user_guide/troubleshooting/troubleshooting_process.md)|Uses various abnormal fault phenomena that developers may encounter during inference and training as the entry point, and provides self-service problem locating and handling methods to help developers quickly locate and resolve faults.|No updates|
|[Native API](../zh/api/native_api/pytorch_2-12-0/overview.md)|Describes the support of the native APIs of PyTorch 2.12.0/2.11.0/2.10.0/2.9.0/2.7.1 on Ascend devices.|&#8226; Added the native API support lists for PyTorch 2.13.0 and PyTorch 2.14.0.<br>&#8226; Aligned the content format with the PyTorch community. |
|[Custom API](https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/en/custom_APIs/overview.md)|Provides function prototypes, function descriptions, parameter descriptions, and usage examples of TorchNPU custom APIs.|&#8226; Added adaptation for PyTorch 2.13.0 and PyTorch 2.14.0.<br>&#8226; For specific interface changes, refer to [Interface Change Description](#interface-change-description).|
|[Environment Variables](../zh/api/environment_variable/env_variable_list.md)|Environment variables that can be used during TorchNPU training and online inference.|&#8226; Added a mapping table of PyTorch environment variables.<br>&#8226; Adjusted part of the directory structure.<br>&#8226; Deleted content about the Torch-MLIR compiler from the TORCHINDUCTOR_NPU_BACKEND section.<br>&#8226; Deleted content about the Torch-MLIR compiler from the (beta) INDUCTOR_ASCEND_CHECK_ACCURACY section.<br>&#8226; Removed the TORCHINDUCTOR_USE_AKG section.<br>&#8226; Removed the NPU_INDUCTOR_FALLBACK_LIST section.|
|[Memory Management](../zh/developer_notes/memory_management/memory_resource_overview.md)|TorchNPU has built a complete memory management system that deeply integrates PyTorch’s native memory management mechanisms and provides multiple unique optimization capabilities tailored to Ascend NPU hardware features.|The content is independent and optimized.|
|[Distributed](../zh/developer_notes/distributed/distributed_overview.md)|Describes the supported core features, including key capabilities such as parallel strategies, sharding primitives, communication strategies, distributed startup, and fault tolerance, and explains how each feature is used on NPUs and the similarities and differences compared with native PyTorch.|&#8226; The Distributed section is displayed independently.<br>&#8226; Added an overview section.<br>&#8226; Added the parallel strategy feature.<br>&#8226; Added the communication strategy feature.<br>&#8226; Refactored the communicator parameters. |
|[Operator Dispatch](../zh/developer_notes/operator_dispatch/automatic_core_binding.md)|Comprehensively improves TorchNPU dispatch and program running performance through automatic core binding, Stream-level TaskQueue parallel dispatch, and compilation optimization.|&#8226; The Operator Dispatch section is displayed independently.<br>&#8226; Added the taskqueue feature.|
|[NPUGraph](../zh/developer_notes/npugraph.md)|NPUGraph is a static graph capture technology used in Eager Mode (single-operator execution mode). It defines and encapsulates a series of NPU kernels as a unit (that is, an operation graph), and starts multiple NPU operations through a single CPU operation, thereby reducing startup overhead.|The content is independent and optimized.|
|[Fault Diagnosis](../zh/developer_notes/fault_diagnosis/feature_value_detection.md)|Without affecting the training performance and accuracy of large models, uses silent data error feature value detection technology based on communication flows to quickly and stably identify accuracy problems.|The content is independent and optimized.|
|[Custom Operator Adaptation Development](../zh/developer_notes/custom_operator_adaptation/opplugin_operator_adaptation/adaptation_overview_opplugin.md)|Writes and calls custom operators based on the OpPlugin plugin or C++ extensions.|The content is independent and optimized.|
|[TorchAir](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/en/overview.md)|As the graph mode capability extension library of Ascend TorchNPU, provides a torch.compile graph mode backend friendly to Ascend devices, implementing graph mode inference acceleration and optimization of PyTorch networks on the Ascend NPU.|&#8226; Added the feature of importing custom operators into graphs with zero deliverables.<br>&#8226; Added the function of skipping recapture verification before each replay.<br>&#8226; Enhanced the SuperKernel fusion optimization function.|
|[Security Statement](../zh/reference/security_statement.md)|Provides the software versions, system hardening requirements, security configurations (data storage, debugging interfaces, running environments, and so on), permission configurations, and firewall settings of the TorchNPU, OpPlugin, TorchAir, and Ascend Extension for TensorPipe components.|Routine update.|

## Virus Scan Results

|Antivirus Software Name|Antivirus Software Version|Virus Database Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-09-19 08:00:00.0|2026-09-20|No viruses or malicious code|
|Kaspersky|12.0.0.6672|2026-09-20 10:13:00.0|2026-09-20|No viruses or malicious code|
|Bitdefender|7.5.1.200224|7.101048|2026-09-20|No viruses or malicious code|

## Vulnerability Patching List

None

## Revision History

|Document|Release Date|Change Description|
|--|--|--|
|01|2026-10-15|First official release|
