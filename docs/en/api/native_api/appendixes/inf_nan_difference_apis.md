# Description of Differences in inf/nan Value Handling

In TorchNPU, some APIs process the special values `inf`/`nan` differently from the native PyTorch implementation, which may cause inconsistent computation results. This document mainly covers two types of differences: behavioral differences when processing abnormal inputs (inputs containing `inf`/`nan`), and differences in outputting `inf`/`nan` when numerical overflow occurs.

**List of APIs with differences in abnormal input processing**

When the input Tensor contains `inf`/`nan` values, the output results of TorchNPU differ from those of native PyTorch. For the specific API list, see [Table 1](#apis-differing-on-inf-nan-input).

**Table 1**  List of APIs whose processing differs from native when the input contains inf/nan<a id="apis-differing-on-inf-nan-input"></a>

| API Name |
| --- |
| Tensor.sign |
| torch.sign |
| torch.linalg.svd |
| torch.linalg.qr |

**Scenarios where outputting inf/nan differs upon numerical overflow**

When the input itself does not contain `inf`/`nan` but numerical overflow occurs during computation, the following differences may arise:

1. The native output is `nan`, while the NPU output is ±`inf`.
2. The native output is ±`inf`, while the NPU output is `nan`.
3. Both the native and NPU outputs are `inf`, but with opposite signs.

These differences are caused by different underlying hardware computation mechanisms. They are hardware behavior differences, not precision issues.
