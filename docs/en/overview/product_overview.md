# Ascend Extension for PyTorch

<!-- md-trans-meta sourceCommit=e6dd39e7131a89f72cf49d80d53002e4cc645bbf translatedAt=2026-07-09T08:28:50.150Z pushedAt=2026-07-09T08:44:08.245Z -->

Ascend Extension for PyTorch is a deep learning adaptation framework based on Ascend, enabling Ascend NPUs to support the PyTorch framework and providing PyTorch users with the superior computing power of Ascend AI processors.

[Project source code repository](https://gitcode.com/Ascend/pytorch).

## Overall Architecture

The overall architecture of Ascend Extension for PyTorch is shown below.

**Figure 1** Overall architecture of Ascend Extension for PyTorch  
![figure1](../figures/architecture_torch_npu.png "Overall architecture of Ascend Extension for PyTorch")

- Ascend Extension for PyTorch (the torch_npu plugin): An Ascend PyTorch adaptation plugin that inherits open-source PyTorch features and is deeply optimized for the Ascend AI processor series, enabling users to perform model training and tuning based on the PyTorch framework.
- PyTorch native library/third-party library adaptation: Adapts and supports PyTorch native libraries and mainstream third-party libraries, complementing ecosystem capabilities and improving the ease of use of the Ascend platform.

## Key Features

- Ascend AI processor adaptation: Based on open-source PyTorch, adapted for Ascend AI processors, providing native Python APIs.
- Basic framework features: PyTorch dynamic graphs, automatic differentiation, profiling, optimizers, and more.
- Custom operator development: Supports adding custom operators within the PyTorch framework.
- Distributed training: Supports native distributed data parallel training, including collective communication primitives such as Broadcast and AllReduce for single-node multi-device and multi-node multi-device scenarios.
- Model inference: Supports exporting standard ONNX models, which can be converted into offline inference models using offline conversion tools.

## More Information

For more information about Ascend Extension for PyTorch, see the online course: [Ascend Extension for PyTorch](https://www.hiascend.com/edu/courses?activeTab=Ascend+Extension+for+PyTorch).
