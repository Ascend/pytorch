# Release Notes

## Version Mapping

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
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.3.1 "><p id="p2547114214349">Official Release</p>
</td>
</tr>
<tr id="row754461214611"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.4.1"><p id="p155445122062">Release Date</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.4.1 "><p id="p135443128613">July 2026</p>
</td>
</tr>
<tr id="row954744243418"><th class="firstcol" valign="top" width="26.25%" id="mcps1.1.3.5.1"><p id="p15471742193419">Maintenance Period</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%" headers="mcps1.1.3.5.1 "><p id="p1154734212344">For details, see the <a href="https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/README.zh.md#%E5%88%86%E6%94%AF%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5">Branch Maintenance Strategy</a></p>
</td>
</tr>
</tbody>
</table>

### Related Product Version Mapping

The version mapping tables of firmware and drivers are related to all Ascend hardware and CANN versions. For the specific selection, see [CANN Release Notes](https://gitcode.com/cann/release-management/blob/master/9.1.0/release-notes.md).

TorchNPU code branch names follow the naming convention **\{PyTorch version\}-\{TorchNPU version\}**, where the former is the PyTorch version matched by TorchNPU. The detailed mapping is as follows:

|TorchNPU Code Branch Name|PyTorch Version|TorchNPU Version|TorchNPU Installation Package Version|CANN Version|Python Version|
|--|--|--|--|--|--|
|v2.7.1-26.1.0|2.7.1|26.1.0|2.7.1.post8|9.1.0|Python3.9.*x*、Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.9.0-26.1.0|2.9.0|26.1.0|2.9.0.post6|9.1.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.10.0-26.1.0|2.10.0|26.1.0|2.10.0.post4|9.1.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.11.0-26.1.0|2.11.0|26.1.0|2.11.0|9.1.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|
|v2.12.0-26.1.0|2.12.0|26.1.0|2.12.0|9.1.0|Python3.10.*x*、Python3.11.*x*、Python3.12.*x*、Python3.13.*x*|

## Version Compatibility

> [!NOTE]
>
> "Y" in the table indicates compatibility.

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

## Update Notes

### New Features

<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1">Component</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">Feature</th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="6" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1">TorchNPU</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">The former Ascend Extension for PyTorch and torch_npu are uniformly renamed to TorchNPU.</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">Except for the import name and the Whl package name prefix, which remain unchanged, all other parts are changed to TorchNPU.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Added support for the <term>Ascend 950DT</term> product.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">The existing TorchNPU capabilities are adapted to <term>Ascend 950DT</term>.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Eager mode enables DVM graphless fusion operators.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Eager mode supports DVM (Device Virtual Machine) operator fusion, reducing scheduling and execution overhead and accelerating network execution.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">LibTorch Stable ABI is supported on the NPU.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">The NPU has implemented LibTorch Stable ABI capabilities aligned with CUDA.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Key computation APIs support heterogeneous inputs.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">Key computation APIs support computation with scalar tensors on the CPU and non-scalar tensors on the NPU as inputs, improving API consistency.</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">Supports operator compilation for input tensors with dynamic all axes.</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">In graph mode, operator compilation for input tensors with dynamic all axes (for example, [-1,-1,-1,-1]) is supported, improving the generalization and performance of graph mode in dynamic shape scenarios.</td>
    </tr>
  </tbody>
</table>

### Removed Features

None

### API Changes

This section covers API changes, including additions, modifications, deprecations, and removals. API changes only reflect code-level modifications and do not include optimization improvements to the documentation itself in terms of language, formatting, links, and so on.

- Added: APIs added in this version.
- Modified: APIs modified compared with the previous version.
- Deprecated: APIs that stop evolving from the version in which the deprecation is announced and may be removed one year after the announcement.
- Deleted: APIs removed in this version.

**Table 1** Summary of TorchNPU API changes

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
      <td class="cellrowborder" rowspan="35" valign="top" width="11.53%" headers="mcps1.2.6.1.1">v2.7.1</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.print_npugraph_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 or later</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_chunk_gated_delta_rule</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 or later</td>
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
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Not dependent on a specific CANN version</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_add_quant_matmul_</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Added</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 or later</td>
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
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 or later</td>
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
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 or later</td>
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
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 or later</td>
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
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 or later</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.save_npugraph_tensor</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 8.5.0 or later</td>
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
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu.finalize_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu.init_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu.set_dump</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu_ciou</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">（beta）torch_npu.npu_iou</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.1.0</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_grouped_matmul_swiglu_quant_v2</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 or later</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">torch_npu.npu_mla_prolog_v3</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.2">Custom API</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.3">Modified</td>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.4">Depends on CANN 9.0.0 or later</td>
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
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">Changes are the same as v2.7.1</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.10.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">Changes are the same as v2.7.1</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.11.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">Changes are the same as v2.7.1</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top" headers="mcps1.2.6.1.1">v2.12.0</td>
      <td class="cellrowborder" colspan="4" valign="top" headers="mcps1.2.6.1.2 mcps1.2.6.1.3 mcps1.2.6.1.4 mcps1.2.6.1.5">Changes are the same as v2.7.1</td>
    </tr>
  </tbody>
</table>

> [!NOTE]  
> TorchNPU has added support for some APIs and features. For details, see [Custom API](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/en/custom_APIs/overview.md) or [Native API](../native_apis/pytorch_2-12-0/overview.md).

### Non-Compatible Changes for Ascend 950DT

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
      <td>Due to the architecture change of <term>Ascend 950DT</term>, some operators and communication interfaces have been adjusted. Therefore, when calling the related APIs, pay attention to the differences in interface constraints between <term>Ascend 950DT</term> and <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>. For details, see <a href="../native_apis/pytorch_2-12-0/overview.md">Native API</a>.</td>
    </tr>
    <tr>
      <td><term>Ascend 950DT</term> currently supports only the Triton and DVM modes in the Inductor backend compiler, and does not support the MLIR mode. For details, see <a href="../torch_compile/pytorch_inductor_desc.md">Inductor</a>.</td>
    </tr>
  </tbody>
</table>

### Resolved Issues

<table><tbody><tr id="row098217197105"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.1.1"><p id="p109824198109">Issue Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.1.1 "><p id="p9982131912103"><strong id="b59839199105">Symptom</strong>: During the custom adaptation of the GDN operator, the aclnn_extension module does not pass the stream parameter.</p>
<p id="p15983141916104"><strong id="b1598312196108">Impact</strong>: This causes abnormal execution order between operators, and the data synchronization mechanism fails to take effect correctly.</p>
</td>
</tr>
<tr id="row1298311191102"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.2.1"><p id="p109831119201013">Severity Level</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.2.1 "><p id="p18983019161017">Minor</p>
</td>
</tr>
<tr id="row598371901017"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.3.1"><p id="p19833192101">Root Cause Analysis</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.3.1 "><p id="p1798319199103">In EXEC_NPU_CMD_V1_EXT and EXEC_NPU_CMD_V2_EXT, memory allocation and release are not paired, and are not consistent with the corresponding EXEC_NPU_CMD_V1 and EXEC_NPU_CMD_V2. This causes memory allocation or release problems and therefore precision issues.</p>
</td>
</tr>
<tr id="row1298318191109"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.4.1"><p id="p1798321961013">Solution</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.4.1 "><p id="p119831219181019">Correct the memory (including workspace) allocation and release in EXEC_NPU_CMD_V1_EXT and EXEC_NPU_CMD_V2_EXT to be consistent with EXEC_NPU_CMD_V1 and EXEC_NPU_CMD_V2.</p>
</td>
</tr>
<tr id="row1198341919103"><th class="firstcol" valign="top" width="14.469999999999999%" id="mcps1.1.3.5.1"><p id="p9983219181017">Modification Impact</p>
</th>
<td class="cellrowborder" valign="top" width="85.53%" headers="mcps1.1.3.5.1 "><p id="p15983119101017">After the fix, the end-to-end precision of models using the GDN operator is normal.</p>
</td>
</tr>
</tbody>
</table>

### Known Issues

None

## Upgrade Impact

### Impact of the Upgrade on the Current System

- Impact on services

    The software version upgrade process causes service interruption.

- Impact on network communication

    No impact on communication.

### Post-Upgrade Impact on the Current System

None

## Version Mapping Documents

|Document Name|Description|Update Notes|
|---|---|---|
|[Software Installation](../installation_guide/installation_description.md)|Provides operations such as installing the PyTorch framework training environment on Ascend devices, as well as upgrading and uninstalling.|&#8226; Added adaptation for PyTorch 2.11.0 and PyTorch 2.12.0.<br>&#8226; Added the software installation FAQ.<br>&#8226; Added content related to <term>Ascend 950DT</term> support. |
|[TorchNPU Overview](../overview/product_overview.md)|TorchNPU is a deep learning adaptation framework based on Ascend. It enables Ascend NPU to support the PyTorch framework and provides PyTorch users with the exceptional computing power of Ascend AI processors.|&#8226; Updated the software architecture content.<br>&#8226; Added content about the initialization process during the startup of the TorchNPU plugin.<br>&#8226; Added content related to <term>Ascend 950DT</term> support. |
|[Quick Start](../quick_start/quick_start.md)|Provides a simple model migration example that uses the simplest automatic migration method, helping users quickly experience the process of migrating GPU model scripts to Ascend NPU.|Added content related to <term>Ascend 950DT</term> support. |
|[Torch.compile](../torch_compile/pytorch_compilation_mode.md)|Significantly accelerates model training and inference tasks through "dynamic graph capture + static graph optimization + efficient code generation".| &#8226; Content is independent and optimized.<br>&#8226; Added content related to <term>Ascend 950DT</term> support.|
|[Companion Software Libraries](https://gitcode.com/Ascend/docs/blob/master/FrameworkPTAdapter/26.1.0/zh/supported_suites_and_third_party_libraries/supported_suites_and_third_party_libraries.md)|Companion software libraries that provide extended capabilities for TorchNPU.|&#8226; Only the "Ascend in-house plugins" part of the original *Supported Suites and Third-Party Libraries* is retained.<br>&#8226; Added the HyperParallel and AKG components.|
|[Fault Handling](../troubleshooting/troubleshooting_process.md)|Starting from the abnormal fault phenomena that developers may encounter during inference and training, provides self-service problem location and problem handling methods to help developers quickly locate and resolve faults.|Added content related to "precision anomaly after using the NZ format".|
|[Native API](../native_apis/pytorch_2-12-0/overview.md)|Provides the support status of native APIs for PyTorch 2.12.0/2.11.0/2.10.0/2.9.0/2.7.1 on Ascend devices.|&#8226; Added the native API support lists for PyTorch 2.11.0 and PyTorch 2.12.0.<br>&#8226; Added content related to <term>Ascend 950DT</term> support. |
|[Custom API](https://gitcode.com/Ascend/op-plugin/blob/26.1.0/docs/zh/custom_APIs/overview.md)|Provides the function prototypes, feature descriptions, parameter descriptions, and call examples of TorchNPU custom APIs.|&#8226; Added adaptation for PyTorch 2.11.0 and PyTorch 2.12.0.<br>&#8226; Added content related to <term>Ascend 950DT</term> support.<br>&#8226; For specific API changes, see [API Changes](#api-changes).|
|[Environment Variables](../environment_variable_reference/env_variable_list.md)|Environment variables that can be used during TorchNPU training and online inference.|&#8226; Added "TORCHINDUCTOR_USE_AKG".<br>&#8226; Added "（beta）TORCHINDUCTOR_ENABLE_MFUSION".<br>&#8226; Added "TORCH_NPU_LAZY_FUSION".<br>&#8226; Added "TORCH_HCCL_BLOCKING_WAIT".<br>&#8226; Added content related to <term>Ascend 950DT</term> support.|
|[Framework Features](../framework_feature_guide_pytorch/overview.md)|Based on TorchNPU, provides the powerful computing power of Ascend AI processors and builds a series of unique features in areas such as memory optimization, error location, and high-performance computing.|&#8226; Updated content related to "torch_npu.npu.NPUGraph".<br>&#8226; Added content related to <term>Ascend 950DT</term> support.|
|[TorchAir](https://gitcode.com/Ascend/torchair/blob/26.1.0/docs/zh/overview.md)|As a graph mode capability extension library of Ascend TorchNPU, provides an Ascend device-friendly torch.compile graph mode backend, implementing graph mode inference acceleration and optimization of PyTorch networks on Ascend NPUs.|&#8226; Enhanced the npugraph_ex function: added support for the SuperKernel fusion optimization function, the force_recapture function, graph capture security policy configuration, and so on.<br>&#8226; Enhanced the GE graph mode functions: extended the npu_stream_switch interface to support specifying concurrency policies, and so on.<br>&#8226; Added content related to <term>Ascend 950DT</term> support.|
|[Security Statement](../security_statement/security_statement.md)|Provides the software versions, system hardening requirements, security configurations (data storage, debugging interfaces, running environments, and so on), permission configurations, and firewall settings of the TorchNPU, OpPlugin, TorchAir, and Ascend Extension for TensorPipe components.|Routine update.|

## Virus Scan and Vulnerability Patch List

### Virus Scan Results

|Antivirus Software Name|Antivirus Software Version|Virus Database Version|Scan Time|Scan Results|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-07-05 08:00:00.0|2026-07-06|No viruses, no malware|
|Kaspersky|12.0.0.6672|2026-07-06 10:03:00.0|2026-07-06|No viruses, no malware|
|Bitdefender|7.5.1.200224|7.101156|2026-07-06|No viruses, no malware|

### Vulnerability Patch List

No vulnerability fixes in this version.
