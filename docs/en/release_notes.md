# Release Notes

## Version Compatibility Notes

### Product Version Information

<table><tbody><tr id="row135479428341"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.1.1"><p id="p125478428345">Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.1.1 "><p id="p3547142103415"><span id="ph4778145519911">TorchNPU</span></p>
</td>
</tr>
<tr id="row11547114203412"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.2.1"><p id="p17547142103418">Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.2.1 "><p id="p2547184216342"><span id="ph1414342615376">26.1.0</span></p>
</td>
</tr>
<tr id="row854711422349"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.3.1"><p id="p354754216341">Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">Official Version</p>
</td>
</tr>
<tr id="row754461214611"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p155445122062">Release Date</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p135443128613">July 2026</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">Maintenance Period</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">Refer to <a href="https://gitcode.com/Ascend/pytorch/blob/master/README.md#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5">Branch Maintenance Policy</a></p>
</td>
</tr>
</tbody>
</table>

### Version Compatibility Notes for Related Products

The version compatibility table of firmware and drivers is related to all Ascend hardware and CANN versions. For specific selection, refer to [CANN Version Notes](https://gitcode.com/cann/release-management/blob/master/9.1.0/release-notes.md).

To expand TorchNPU capabilities, for the version requirements of the self-developed plugins provided by Ascend, refer to [Compatible Software Libraries](./user_guide/libraries.md).

TorchNPU code branch names follow the naming rule **\{PyTorch version\}-\{TorchNPU version\}**, where the former is the PyTorch version matched by TorchNPU. The detailed matching is shown in the following table:

|TorchNPU Code Branch Name|PyTorch Version|TorchNPU Version|TorchNPU Installation Package Version|CANN Version|Python Version|
|--|--|--|--|--|--|
|v2.7.1-26.1.0|2.7.1|26.1.0|2.7.1.post8|9.1.0|Python3.9.*x*, Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|v2.9.0-26.1.0|2.9.0|26.1.0|2.9.0.post6|9.1.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|v2.10.0-26.1.0|2.10.0|26.1.0|2.10.0.post4|9.1.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|v2.11.0-26.1.0|2.11.0|26.1.0|2.11.0|9.1.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|
|v2.12.0-26.1.0|2.12.0|26.1.0|2.12.0|9.1.0|Python3.10.*x*, Python3.11.*x*, Python3.12.*x*, Python3.13.*x*|

## Version Compatibility Description

> [!NOTE]
>
> In the table, "Y" means compatible.

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
    <th class="tg-amwm" colspan="3">CANN Version</th>
  </tr>
  <tr>
    <th class="tg-c3ow">8.5.X</th>
    <th class="tg-c3ow">9.0.X</th>
    <th class="tg-c3ow">9.1.X</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">7.3.X</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
    <td class="tg-c3ow">Y</td>
  </tr>
  <tr>
    <td class="tg-baqh">26.0.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
    <tr>
    <td class="tg-baqh">26.1.X</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
    <td class="tg-baqh">Y</td>
  </tr>
</tbody>
</table>

## Update Description

### New Features

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1">Component</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">Feature</th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3">Feature Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="6" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1">TorchNPU</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">Eager mode enables DVM graph-free operator fusion.</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">Eager mode supports device virtual machine (DVM) operator fusion, reducing scheduling and execution overhead and accelerating network execution.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">LibTorch Stable ABI is supported on the NPU.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">The NPU has implemented the LibTorch Stable ABI capability aligned with CUDA.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Key computing APIs support heterogeneous input.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Key computing APIs support computation with a scalar tensor on the CPU and a non-scalar tensor on the NPU as input, improving API consistency.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Supports operator compilation in scenarios where input tensors are dynamic on all axes.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">In graph mode, supports operator compilation in scenarios where input tensors are dynamic on all axes (such as [-1,-1,-1,-1]), improving the generalization and performance of graph mode in dynamic shape scenarios.</td>
    </tr>
    <tr>
      <!-- npu="950" id1 -->
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Supports the <term>Ascend 950DT products</term>.</td>
      <!-- end id1 -->
      <!-- npu="950" id2 -->
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Adapts the existing TorchNPU capabilities to <term>Ascend 950DT products</term>.</td>
      <!-- end id2 -->
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">The original Ascend Extension for PyTorch and torch_npu are uniformly renamed to TorchNPU.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Except for the import name and the Whl package name prefix, which remain unchanged, all other parts are changed to TorchNPU.</td>
    </tr>
  </tbody>
</table>

### Deleted Features

None

### API Change Description

The API change description in this section covers added, modified, deprecated, and deleted APIs. API changes only reflect modifications at the code level and do not include optimizations or improvements to the document itself in terms of language, format, links, and so on.

- Added: indicates APIs added in this version.
- Modified: indicates that the API has been modified compared with the previous version.
- Deprecated: indicates that the API stops evolving from the version in which the deprecation is announced and may be removed one year after the announcement.
- Deleted: indicates that the API has been removed in this version.

**Table 1** TorchNPU API Change Summary

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
      <td class="cellrowborder" rowspan="36" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.print_npugraph_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_chunk_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rotate_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_apply_rotary_pos_emb</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Does not depend on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_quant_matmul_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_alltoallv_quant_gmm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_dynamic_dual_level_mx_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fused_causal_conv1d</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_dynamic_block_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_masked_causal_conv1d</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mhc_post</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mhc_pre</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mhc_sinkhorn</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_transpose_quant_batchmatmul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.save_npugraph_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_rms_norm_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_scaled_masked_softmax</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_cross_entropy_loss</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_attention_update</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_fusion_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gelu_mul</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_top_k_top_p</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_gather_sparse_index</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_swiglu_quant</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_rms_norm</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">(beta) torch_npu.npu.finalize_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">(beta) torch_npu.npu.init_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">(beta) torch_npu.npu.set_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.apply_rotary_pos_emb</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">(beta) torch_npu.npu_ciou</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">(beta) torch_npu.npu_iou</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_swiglu_quant_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mla_prolog_v3</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 and later versions</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_quant_lightning_indexer</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_sparse_flash_attention</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
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
  </tbody>
</table>

> [!NOTE]
>
> TorchNPU adds support for some APIs and features. For details, refer to the [Custom API](https://gitcode.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/overview.md) or the [Native API](../en/api/native_api/pytorch_2-12-0/overview.md).

<!-- npu="950" id5 -->
### Incompatible Change Descriptions of the Ascend 950DT Products
<!-- end id5 -->

<table>
  <thead align="left">
    <tr>
      <th>Component</th>
      <th>Change Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">TorchNPU</td>
      <!-- npu="950,A3,910b" id3 -->
      <td>Due to the architecture change of <term>Ascend 950DT products</term>, some operators and communication APIs have been adjusted. Therefore, when calling related APIs, pay attention to the differences in API constraints between <term>Ascend 950DT products</term> and <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>. For details, refer to <a href="../en/api/native_api/pytorch_2-12-0/overview.md">Native API</a>.</td>
      <!-- end id3 -->
    </tr>
    <tr>
      <!-- npu="950" id4 -->
      <td><term>Ascend 950DT products</term> currently supports only the Triton and DVM modes in the Inductor backend compiler and does not currently support the MLIR mode. For details, refer to <a href="../en/user_guide/torch_compile/compilation_backend/inductor.md">Inductor</a>.</td>
      <!-- end id4 -->
    </tr>
  </tbody>
</table>

### Resolved Issues

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">Issue Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">Symptom</strong>: During the custom adaptation of the GDN operator, the aclnn_extension module did not pass the stream parameter.</p>
<p id="p15983141916104"><strong id="b1598312196108">Impact</strong>: The execution order between operators was abnormal, and the data synchronization mechanism did not take effect correctly.</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">Severity Level</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">General</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">Root Cause Analysis</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">The memory allocation and release in EXEC_NPU_CMD_V1_EXT and EXEC_NPU_CMD_V2_EXT were not matched and were not consistent with the corresponding EXEC_NPU_CMD_V1 and EXEC_NPU_CMD_V2, resulting in memory allocation or release problems and therefore accuracy problems.</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">Solution</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019">The memory (including workspace) allocation and release in EXEC_NPU_CMD_V1_EXT and EXEC_NPU_CMD_V2_EXT are corrected to be consistent with EXEC_NPU_CMD_V1 and EXEC_NPU_CMD_V2.</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">Impact of the Change</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">After the fix, the whole-network accuracy of models using the GDN operator is normal.</p>
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

## Related Documents for This Version

|Document Name|Description|Update Description|
|---|---|---|
|[Installation Guide](../en/installation_guide/building_from_source.md)|Describes how to install the PyTorch framework training environment on Ascend devices, as well as operations such as upgrade and uninstallation.|&#8226; Added adaptation for PyTorch 2.11.0 and PyTorch 2.12.0.<br>&#8226; Added the software installation FAQ.<br>&#8226; Added content about supporting <term>Ascend 950DT products</term>. |
|[TorchNPU Overview](../en/user_guide/product_overview.md)|The TorchNPU plugin is a deep learning adaptation framework based on Ascend. It enables the Ascend NPU to support the PyTorch framework and provides PyTorch framework users with the powerful computing power of Ascend AI processors.|&#8226; Updated content about the software architecture.<br>&#8226; Added content about the initialization process during the startup phase of the TorchNPU plugin.<br>&#8226; Added content about supporting <term>Ascend 950DT products</term>. |
|[Quick Start](../en/user_guide/quick_start.md)|Provides a simple model migration example that uses the simplest automatic migration method to help users quickly experience the process of migrating GPU model scripts to the Ascend NPU.|Added content about supporting <term>Ascend 950DT products</term>. |
|[Torch.compile](../en/user_guide/torch_compile/overview.md)|Significantly accelerates model training and inference tasks through the approach of "dynamic graph capture + static graph optimization + efficient code generation".|The content is independent and optimized.|
|[Compatible Software Libraries](./user_guide/libraries.md)|Compatible software libraries that extend the capabilities of TorchNPU.|&#8226; Only the "self-developed plugins provided by Ascend" part in the original Suites and Third-Party Libraries Support List is retained.<br>&#8226; Added the HyperParallel and AKG components.|
|[Troubleshooting](../en/user_guide/troubleshooting/troubleshooting_process.md)|Uses various abnormal fault phenomena that developers may encounter during inference and training as the entry point, and provides self-service problem locating and handling methods to help developers quickly locate and resolve faults.|Added content about "accuracy exceptions after using the NZ format".|
|[Native API](../en/api/native_api/pytorch_2-12-0/overview.md)|Describes the support of the native APIs of PyTorch 2.12.0/2.11.0/2.10.0/2.9.0/2.7.1 on Ascend devices.|&#8226; Added the native API support lists for PyTorch 2.11.0 and PyTorch 2.12.0.<br>&#8226; Added content about supporting <term>Ascend 950DT products</term>. |
|[Custom API](https://gitcode.com/Ascend/op-plugin/blob/master/docs/en/custom_APIs/overview.md)|Provides function prototypes, function descriptions, parameter descriptions, and usage examples of TorchNPU custom APIs.|&#8226; Added adaptation for PyTorch 2.11.0 and PyTorch 2.12.0.<br>&#8226; Added content about supporting <term>Ascend 950DT products</term>.<br>&#8226; For specific API changes, refer to [API Change Description](#api-change-description).|
|[Environment Variables](../en/api/environment_variable/env_variable_list.md)|Environment variables that can be used during TorchNPU training and online inference.|&#8226; Added "TORCHINDUCTOR_USE_AKG".<br>&#8226; Added "(beta) TORCHINDUCTOR_ENABLE_MFUSION".<br>&#8226; Added "TORCH_NPU_LAZY_FUSION".<br>&#8226; Added "TORCH_HCCL_BLOCKING_WAIT".<br>&#8226; Added content about supporting <term>Ascend 950DT products</term>.|
|[Framework Features](../en/developer_notes/memory_management/memory_resource_overview.md)|Based on TorchNPU, provides the powerful computing power of Ascend AI processors and creates a series of unique features in aspects such as memory optimization, error locating, and high-performance computing.|Updated content about "torch_npu.npu.NPUGraph".|
|[TorchAir](https://gitcode.com/Ascend/torchair/blob/master/docs/en/overview.md)|As the graph mode capability extension library of Ascend TorchNPU, provides a torch.compile graph mode backend friendly to Ascend devices, implementing graph mode inference acceleration and optimization of PyTorch networks on the Ascend NPU.|&#8226; npugraph_ex function enhancement: Added support for the SuperKernel fusion optimization function, the force_recapture function, graph capture security policy configuration, and so on.<br>&#8226; GE graph mode function enhancement: Extended the npu_stream_switch API to support specifying concurrency policies and so on.<br>&#8226; Added content about supporting <term>Ascend 950DT products</term>.|
|[Security Statement](../en/reference/security_statement.md)|Provides the software versions, system hardening requirements, security configurations (data storage, debugging APIs, running environments, and so on), permission configurations, and firewall settings of the TorchNPU, OpPlugin, TorchAir, and Ascend Extension for TensorPipe components.|Routine update.|

## Virus Scan and Vulnerability Patching List

### Virus Scan Results

|Antivirus Software Name|Antivirus Software Version|Virus Database Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-07-05 08:00:00.0|2026-07-06|No viruses or malicious code|
|Kaspersky|12.0.0.6672|2026-07-06 10:03:00.0|2026-07-06|No viruses or malicious code|
|Bitdefender|7.5.1.200224|7.101156|2026-07-06|No viruses or malicious code|

### Vulnerability Patching List

No vulnerabilities were fixed in this version.
