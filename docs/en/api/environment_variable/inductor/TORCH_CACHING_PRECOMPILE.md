# TORCH\_CACHING\_PRECOMPILE

## Function Description

This environment variable enables the experimental feature of automatic caching precompile. When enabled, `torch.compile` automatically serializes Dynamo cache entries and backend compilation artifacts into the global Precompile Cache during compilation, and automatically loads and reuses them in subsequent runs, thereby skipping repeated compilation and speeding up subsequent compilation.

- Default value "0": disables automatic caching precompile.
- Set to "1": enables automatic caching precompile.

When enabled, the system performs the following:

- **Automatic loading at compile time**: when `torch.compile` is called, existing compilation caches are automatically searched for and loaded from DynamoCache.
- **Automatic saving after compilation**: after compilation completes, Dynamo cache entries and backend artifacts (such as the AOTAutograd cache) are automatically written to DynamoCache.
- **Guard filtering**: automatically discards non-serializable guards (such as `ID_MATCH`, `CLOSURE_MATCH`, `WEAKREF_ALIVE`, and `DICT_VERSION`) to ensure that caches can be reused across processes.
- **Works with `torch.compiler.save_cache_artifacts()`**: when `save_cache_artifacts()` is called, the cache entries in `PrecompileContext` are saved as well.

> [!NOTE]
>
> - This environment variable must be configured before the process starts; changes during process runtime do not take effect.
> - This is an experimental feature, suitable for scenarios where the same model needs to be compiled multiple times (such as multi-epoch training), and can significantly reduce subsequent compilation time.
> - After it is enabled, some guards are discarded, which may relax the cache hit conditions; pay attention to the correctness of compilation results when using it.
> - It is recommended to enable it together with `torch.compiler.save_cache_artifacts()`/`load_cache_artifacts()`.

This variable corresponds to PyTorch's [TORCH_CACHING_PRECOMPILE](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/config.py), with the same configuration method.

## Configuration Example

Enabling automatic caching precompile:

```bash
export TORCH_CACHING_PRECOMPILE=1
```

## Usage Constraints

None

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>
<!-- end id3 -->
<!-- npu="950" id4 -->
- <term>Ascend 950DT products</term>
<!-- end id4 -->
