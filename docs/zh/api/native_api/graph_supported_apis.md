# 支持入图的API清单

昇腾支持入图的API列表请参见[表1](#昇腾支持入图的API列表)。

> [!NOTE]
>
> - 标注“支持入图”的API，表示该接口可在`torch.compile`编译后的计算图中正常使用，能够被图捕获并参与优化。关于`torch.compile`的说明请参考[torch.compile](../../user_guide/torch_compile/_menu_torch_compile.md)。
> - 本文档展示的列表均为已验证支持入图的API。部分未验证的API可能也具备支持入图的能力，在昇腾NPU环境中请谨慎使用。我们将持续验证更多API并定期更新本文档。
> - 标注星号（*）的API存在产品或PyTorch版本限制，详见表格下方说明。

**表 1**  昇腾支持入图的API列表<a id="昇腾支持入图的API列表"></a>

| API名称                                               |
|-----------------------------------------------------|
| Tensor.abs                                          |
| Tensor.add                                          |
| Tensor.addmm                                        |
| Tensor.all                                          |
| Tensor.any                                          |
| Tensor.argmax                                       |
| Tensor.argmin                                       |
| Tensor.bitwise\_and                                 |
| Tensor.bitwise\_not                                 |
| Tensor.bitwise\_or                                  |
| Tensor.bmm                                          |
| Tensor.clamp                                        |
| Tensor.copy\_                                       |
| Tensor.cos                                          |
| Tensor.cumsum                                       |
| Tensor.div                                          |
| Tensor.eq                                           |
| Tensor.exponential\_                                |
| Tensor.fill\_                                       |
| Tensor.gather                                       |
| Tensor.ge                                           |
| Tensor.gt                                           |
| Tensor.histc                                        |
| Tensor.index\_add                                   |
| Tensor.index\_fill                                  |
| Tensor.index\_put                                   |
| Tensor.index\_select                                |
| Tensor.le                                           |
| Tensor.log2                                         |
| Tensor.logical\_and\_                               |
| Tensor.logical\_not                                 |
| Tensor.lt                                           |
| Tensor.masked\_fill                                 |
| Tensor.matmul                                       |
| Tensor.max                                          |
| Tensor.maximum                                      |
| Tensor.mean                                         |
| Tensor.min                                          |
| Tensor.minimum                                      |
| Tensor.mm                                           |
| Tensor.mul                                          |
| Tensor.ne                                           |
| Tensor.neg                                          |
| Tensor.pow                                          |
| Tensor.reciprocal                                   |
| Tensor.remainder                                    |
| Tensor.repeat                                       |
| Tensor.repeat\_interleave                           |
| Tensor.roll                                         |
| Tensor.rsqrt                                        |
| Tensor.scatter\_                                    |
| Tensor.scatter\_add                                 |
| Tensor.sigmoid                                      |
| Tensor.sign                                         |
| Tensor.sin                                          |
| Tensor.softmax                                      |
| Tensor.sort                                         |
| Tensor.sqrt                                         |
| Tensor.sub                                          |
| Tensor.sum                                          |
| Tensor.to                                           |
| Tensor.topk                                         |
| Tensor.tril                                         |
| Tensor.triu                                         |
| Tensor.uniform\_                                    |
| Tensor.where                                        |
| torch.\_fused\_adamw\_                              |
| torch.\_log\_softmax                                |
| torch.\_softmax\_backward\_data                     |
| torch.abs                                           |
| torch.add                                           |
| torch.addmm                                         |
| torch.all                                           |
| torch.any                                           |
| torch.arange                                        |
| torch.argmax                                        |
| torch.argmin                                        |
| torch.bitwise\_and                                  |
| torch.bitwise\_not                                  |
| torch.bitwise\_or                                   |
| torch.bmm                                           |
| torch.cat                                           |
| torch.clamp                                         |
| torch.constant\_pad\_nd                             |
| torch.convolution                                   |
| torch.cos                                           |
| torch.cumsum                                        |
| torch.device<sup>*</sup>                                        |
| torch.distributed.\_all\_gather\_base               |
| torch.distributed.\_reduce\_scatter\_base           |
| torch.distributed.all\_gather                       |
| torch.distributed.all\_gather\_into\_tensor         |
| torch.distributed.all\_reduce                       |
| torch.distributed.all\_to\_all\_single              |
| torch.distributed.get\_process\_group\_ranks        |
| torch.distributed.get\_rank                         |
| torch.distributed.get\_world\_size                  |
| torch.distributed.init\_process\_group              |
| torch.distributed.is\_gloo\_available               |
| torch.distributed.reduce\_op                        |
| torch.distributed.reduce\_scatter\_tensor           |
| torch.distributed.ReduceOp                          |
| torch.div                                           |
| torch.dropout                                       |
| torch.embedding                                     |
| torch.eq                                            |
| torch.exp                                           |
| torch.fill\_                                        |
| torch.floor\_divide                                 |
| torch.gather                                        |
| torch.ge                                            |
| torch.gt                                            |
| torch.index                                         |
| torch.index\_add                                    |
| torch.index\_fill                                   |
| torch.index\_put                                    |
| torch.index\_select                                 |
| torch.le                                            |
| torch.linalg.vector\_norm                           |
| torch.log                                           |
| torch.logical\_and\_                                |
| torch.logical\_not                                  |
| torch.lt                                            |
| torch.masked\_fill                                  |
| torch.matmul                                        |
| torch.max                                           |
| torch.maximum                                       |
| torch.mean                                          |
| torch.min                                           |
| torch.minimum                                       |
| torch.mm                                            |
| torch.mul                                           |
| torch.native\_dropout                               |
| torch.ne                                            |
| torch.neg                                           |
| torch.nn.functional.cross\_entropy<sup>*</sup>                  |
| torch.nn.functional.layer\_norm                     |
| torch.nn.functional.mse\_loss                       |
| torch.nn.functional.one\_hot                        |
| torch.nn.functional.scaled\_dot\_product\_attention<sup>*</sup> |
| torch.nn.functional.silu                            |
| torch.nn.functional.softmax                         |
| torch.nn.GELU                                       |
| torch.no\_grad                                      |
| torch.normal                                        |
| torch.pow                                           |
| torch.reciprocal                                    |
| torch.relu                                          |
| torch.remainder                                     |
| torch.roll                                          |
| torch.rsqrt                                         |
| torch.rsub                                          |
| torch.scatter                                       |
| torch.scatter\_add                                  |
| torch.sigmoid                                       |
| torch.sign                                          |
| torch.sin                                           |
| torch.softmax                                       |
| torch.sort                                          |
| torch.sqrt                                          |
| torch.stack                                         |
| torch.sub                                           |
| torch.sum                                           |
| torch.threshold                                     |
| torch.topk                                          |
| torch.tril                                          |
| torch.triu                                          |
| torch.where                                         |
| torch.zeros                                         |
| torch\_npu.npu\_add\_rms\_norm                      |
| torch\_npu.npu\_all\_gather\_base\_mm               |
| torch\_npu.npu\_dtype\_cast                         |
| torch\_npu.npu\_fusion\_attention                   |
| torch\_npu.npu\_mm\_reduce\_scatter\_base           |
| torch\_npu.npu\_moe\_token\_permute                 |
| torch\_npu.npu\_moe\_token\_unpermute               |
| torch\_npu.npu\_rms\_norm                           |
| torch\_npu.npu\_rotary\_mul                         |
| torch\_npu.npu\_swiglu                              |
| torch\_npu.npu\_grouped\_matmul                     |
| torch.\_foreach\_copy\_                             |

特殊说明：

<!-- npu="910b" id1 -->
- \* `torch.device`：在<term>Atlas A2训练系列产品</term>上，PyTorch版本为2.7.1时不支持入图。
<!-- end id1 -->

<!-- npu="A3" id2 -->
- \* `torch.device`：在<term>Atlas A3训练系列产品</term>上，PyTorch版本为2.7.1时不支持入图。
<!-- end id2 -->

<!-- npu="910b" id3 -->
- \* `torch.nn.functional.cross_entropy`：在<term>Atlas A2训练系列产品</term>上，PyTorch版本为2.7.1时不支持入图。
<!-- end id3 -->

<!-- npu="A3" id4 -->
- \* `torch.nn.functional.cross_entropy`：在<term>Atlas A3训练系列产品</term>上，PyTorch版本为2.7.1时不支持入图。
<!-- end id4 -->

<!-- npu="950" id5 -->
- \* `torch.nn.functional.scaled_dot_product_attention`：在<term>Ascend 950DT系列产品</term>上，PyTorch版本为2.7.1、2.9.0、2.10.0、2.11.0、2.12.0、2.13.0、2.14.0时不支持入图。
<!-- end id5 -->
