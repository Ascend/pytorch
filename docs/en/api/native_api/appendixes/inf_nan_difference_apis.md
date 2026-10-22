# Description of Differences in inf/nan Value Handling

In TorchNPU, some APIs handle the special values `inf`/`nan` differently from the native PyTorch implementation, which may lead to inconsistent computation results. This document mainly lists two types of difference scenarios: behavior differences when the input is abnormal (the input contains `inf`/`nan`), and differences in the `inf`/`nan` output when numeric overflow occurs.

**APIs with differences in abnormal input handling**

When the input Tensor contains `inf`/`nan` values, the output of TorchNPU differs from that of native PyTorch. For the specific API list, see [Table 1](#list-of-apis-with-inf-nan-processing-differences).

**Table 1**  List of APIs whose processing differs from the native implementation when the input contains inf/nan<a id="list-of-apis-with-inf-nan-processing-differences"></a>

| API Name |
| --- |
| Tensor.sign |
| torch.sign |
| torch.linalg.svd |
| torch.linalg.qr |

**Scenarios where outputting inf/nan differs when numeric overflow occurs**

When the input itself does not contain `inf`/`nan` but numeric overflow occurs during computation, the following differences may occur:

1. The native output is `nan`, while the NPU output is ±`inf`;
2. The native output is ±`inf`, while the NPU output is `nan`;
3. Both the native and NPU outputs are `inf`, but with opposite signs.

The above differences are caused by different underlying hardware computation mechanisms. They are hardware behavior differences rather than precision issues.
