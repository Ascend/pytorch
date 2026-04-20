# USE_STORE_IN_CAT

## 功能描述
用于控制编译模式Inductor后端下，针对torch.cat接口参与算子融合的行为，当前默认为False。

## 配置示例
使用triton-ascend的扩展接口insert_slice与extract_slice处理cat场景的融合，用户不感知，用于性能调优与错误规避。
```bash
export USE_STORE_IN_CAT=False
```
使用triton-ascend的社区接口store处理cat场景的融合。
```bash
export USE_STORE_IN_CAT=True
```


## 使用约束
当INDUCTOR_INDIRECT_MEMORY_MODE环境变量配置为"simt_only",或"simd_simt_mix"时，始终为True。

## 支持的型号
-   <term>Atlas A5 系列产品</term>