# Introduction to Compilation Optimization Technology

> [!NOTE]
>
> In the current version, compilation optimization is available only as a trial feature. This functionality may be adjusted or improved in subsequent versions. During use, you are advised to pay attention to subsequent version updates.

## Compilation Technology: LTO Link-Time Optimization

LTO is a mature compilation optimization technology that has been widely adopted in the industry. It reduces call overhead through cross-file function inlining, eliminates redundant code via cross-file function specialization and constant propagation, and also enables cross-language optimization. These capabilities deliver considerable performance gains. From a top-down analysis perspective, LTO is effective for both front-end and back-end bottlenecks. LTO optimization is categorized into FullLTO and ThinLTO. ThinLTO is a more recent link-time optimization technique that offers better runtime performance than FullLTO, significantly reducing the time and memory consumption of link-time optimization.

**Figure 1**  LTO optimization schematic diagram  
<img src="../figures/comp_opt_intro_fig_01.png" height="465.5" width="399.8911">

## Compilation Technology: PGO Feedback Optimization

PGO (Profile-Guided Optimization) is a compiler optimization technology. It collects performance data during program execution and uses this data during the compilation phase to optimize program performance. PGO requires two compilation passes. In the first compilation, instrumentation is inserted into the application code, and by running typical use cases and workloads, information about the execution frequency of functions and branches in the application code is collected. In the second compilation, further optimization is performed based on the runtime statistics to generate a high-performance application. The PGO feedback optimization technology delivers significant results in data- and computation-intensive scenarios with high front-end bottlenecks, such as databases and distributed storage, achieving performance improvements of 10-30%. It can effectively reduce computation time and resource consumption, improve application performance, significantly lower operational costs, and enhance user experience.

## Introduction to Compilation Optimization Solutions

By applying the LTO and PGO compilation optimization technologies of the Bisheng compiler and compiling the three components, Python, PyTorch, and TorchNPU, you can effectively improve program performance.

Due to the Pybind11 framework, compatibility conflicts may exist among the relevant compilation optimization packages. You can refer to the following table to choose partial or full compilation optimization for the packages. The subsequent compilation optimization guidance uses the Bisheng compiler as an example.

**Table 1** Compatibility mapping

|Python|PyTorch|TorchNPU|Compatible|
|--|--|--|--|
|gcc (default)|gcc (default)|gcc (default)|Yes|
|gcc (default)|gcc (default)|Bisheng|No|
|gcc (default)|Bisheng|gcc (default)|No|
|gcc (default)|Bisheng|Bisheng|Yes|
|Bisheng|gcc (default)|gcc (default)|Yes|
|Bisheng|gcc (default)|Bisheng|No|
|Bisheng|Bisheng|gcc (default)|No|
|Bisheng|Bisheng|Bisheng|Yes|
