# SHUT\_DOWN\_FX\_PASS\_LIST

## Function Description

This environment variable specifies the list of `torch_npu` graph optimization FX passes to be disabled. The passes in the list are skipped at the registration stage (they do not participate in graph optimization for inference and training), which is used to avoid issues introduced by specific passes or to compare and verify the pass effects.

- Set to `all`: disables all custom FX passes.
- Set to a comma-separated list of pass names: the passes in the list are not registered.

> [!NOTE]
>
> - This environment variable must be set before importing `torch_npu`.
> - The pass name is the function name at registration. For the specific naming conventions, refer to [ascend_graph_pass.py](../../../../../torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py).
> - This variable appends to the default disabled list and does not enable passes that are disabled by default.

This environment variable is provided by TorchNPU; PyTorch has no directly corresponding environment variable.

## Configuration Example

Disabling a specified pass:

```bash
export SHUT_DOWN_FX_PASS_LIST="view_fold_pass,fold_cat"
```

Disabling all custom passes:

```bash
export SHUT_DOWN_FX_PASS_LIST="all"
```

## Usage Constraints

None

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="950" id2 -->
- <term>Ascend 950DT products</term>
<!-- end id2 -->
