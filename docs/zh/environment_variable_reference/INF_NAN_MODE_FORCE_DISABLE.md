# INF\_NAN\_MODE\_FORCE\_DISABLE

## 功能描述

<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>，通过此环境变量可强制关闭INF\_NAN模式。关闭INF\_NAN模式会导致计算过程中产生的Inf和NaN被转换成对应数据类型的最大值和0值，使后续运算结果出现差异，所以进行了强制拦截。若一定要关闭INF\_NAN模式，则需要配置该环境变量为“1”即强制关闭INF\_NAN模式，关闭后需注意Inf和NaN值的变化。

-   1：强制关闭INF\_NAN模式，开启饱和模式。<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>强制关闭INF\_NAN模式后需注意Inf和NaN值的变化。
-   0：不对INF\_NAN模式做处理，<term>Atlas A2 训练系列产品</term>/<term>Atlas A3 训练系列产品</term>关闭INF\_NAN模式会被拦截报错。默认值为0。

针对<term>Atlas 训练系列产品</term>/<term>Atlas 推理系列产品</term>/<term>Atlas 200I/500 A2 推理产品</term>，该环境变量不生效。

## 配置示例

```
export INF_NAN_MODE_FORCE_DISABLE=1
```

## 使用约束

无

## 支持的型号

-   <term>Atlas A2 训练系列产品</term>
-   <term>Atlas A3 训练系列产品</term>

