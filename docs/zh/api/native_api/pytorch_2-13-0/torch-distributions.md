# torch.distributions

> [!NOTE]
>
> - API的**支持情况**中，✔表示API支持在对应硬件环境上运行，✘表示暂不支持。<br>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.13/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://pytorch.org/docs/2.13/distributions.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Distribution](#distribution)
- [ExponentialFamily](#exponentialfamily)
- [Bernoulli](#bernoulli)
- [Beta](#beta)
- [Binomial](#binomial)
- [Categorical](#categorical)
- [Cauchy](#cauchy)
- [Chi2](#chi2)
- [ContinuousBernoulli](#continuousbernoulli)
- [Dirichlet](#dirichlet)
- [Exponential](#exponential)
- [FisherSnedecor](#fishersnedecor)
- [Gamma](#gamma)
- [Geometric](#geometric)
- [Gumbel](#gumbel)
- [HalfCauchy](#halfcauchy)
- [HalfNormal](#halfnormal)
- [Independent](#independent)
- [Kumaraswamy](#kumaraswamy)
- [LKJCholesky](#lkjcholesky)
- [Laplace](#laplace)
- [LogNormal](#lognormal)
- [LowRankMultivariateNormal](#lowrankmultivariatenormal)
- [MixtureSameFamily](#mixturesamefamily)
- [Multinomial](#multinomial)
- [MultivariateNormal](#multivariatenormal)
- [NegativeBinomial](#negativebinomial)
- [Normal](#normal)
- [OneHotCategorical](#onehotcategorical)
- [Pareto](#pareto)
- [Poisson](#poisson)
- [RelaxedBernoulli](#relaxedbernoulli)
- [LogitRelaxedBernoulli](#logitrelaxedbernoulli)
- [RelaxedOneHotCategorical](#relaxedonehotcategorical)
- [StudentT](#studentt)
- [TransformedDistribution](#transformeddistribution)
- [Uniform](#uniform)
- [VonMises](#vonmises)
- [Weibull](#weibull)
- [Wishart](#wishart)
- [KL Divergence](#kl-divergence)
- [Transforms](#transforms)
- [Constraints](#constraints)
- [Constraint Registry](#constraint-registry)

</div>

<div style="display:none;">

## &#8203;torch.distributions

</div>

## Distribution

### <code><i>class</i></code> torch.distributions.distribution.Distribution

<div style="margin-left: 2em">

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">batch_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.batch_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.batch_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">event_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.event_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.event_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">set_default_validate_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.set_default_validate_args](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.set_default_validate_args)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.stddev](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.stddev)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.distribution.Distribution.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## ExponentialFamily

### <code><i>class</i></code> torch.distributions.exp_family.ExponentialFamily

<div style="margin-left: 2em">

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exp_family.ExponentialFamily.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exp_family.ExponentialFamily.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Bernoulli

### <code><i>class</i></code> torch.distributions.bernoulli.Bernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.has_enumerate_support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.has_enumerate_support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.param_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.param_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.bernoulli.Bernoulli.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Beta

### <code><i>class</i></code> torch.distributions.beta.Beta

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">concentration0()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.concentration0](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.concentration0)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">concentration1()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.concentration1](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.concentration1)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.beta.Beta.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Binomial

### <code><i>class</i></code> torch.distributions.binomial.Binomial

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.has_enumerate_support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.has_enumerate_support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.param_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.param_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.binomial.Binomial.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Categorical

### <code><i>class</i></code> torch.distributions.categorical.Categorical

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.has_enumerate_support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.has_enumerate_support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.param_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.param_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.categorical.Categorical.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Cauchy

### <code><i>class</i></code> torch.distributions.cauchy.Cauchy

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.cauchy.Cauchy.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Chi2

### <code><i>class</i></code> torch.distributions.chi2.Chi2

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.chi2.Chi2](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.chi2.Chi2)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.chi2.Chi2.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.chi2.Chi2.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">df()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.chi2.Chi2.df](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.chi2.Chi2.df)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.chi2.Chi2.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.chi2.Chi2.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## ContinuousBernoulli

### <code><i>class</i></code> torch.distributions.continuous_bernoulli.ContinuousBernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.param_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.param_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.stddev](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.stddev)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Dirichlet

### <code><i>class</i></code> torch.distributions.dirichlet.Dirichlet

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.dirichlet.Dirichlet.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Exponential

### <code><i>class</i></code> torch.distributions.exponential.Exponential

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.stddev](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.stddev)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.exponential.Exponential.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## FisherSnedecor

### <code><i>class</i></code> torch.distributions.fishersnedecor.FisherSnedecor

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Gamma

### <code><i>class</i></code> torch.distributions.gamma.Gamma

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gamma.Gamma.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Geometric

### <code><i>class</i></code> torch.distributions.geometric.Geometric

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.geometric.Geometric.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Gumbel

### <code><i>class</i></code> torch.distributions.gumbel.Gumbel

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.stddev](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.stddev)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.gumbel.Gumbel.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## HalfCauchy

### <code><i>class</i></code> torch.distributions.half_cauchy.HalfCauchy

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.scale](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.scale)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_cauchy.HalfCauchy.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## HalfNormal

### <code><i>class</i></code> torch.distributions.half_normal.HalfNormal

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.scale](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.scale)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.half_normal.HalfNormal.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Independent

### <code><i>class</i></code> torch.distributions.independent.Independent

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.has_enumerate_support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.has_enumerate_support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.independent.Independent.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Kumaraswamy

### <code><i>class</i></code> torch.distributions.kumaraswamy.Kumaraswamy

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## LKJCholesky

### <code><i>class</i></code> torch.distributions.lkj_cholesky.LKJCholesky

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Laplace

### <code><i>class</i></code> torch.distributions.laplace.Laplace

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.stddev](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.stddev)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.laplace.Laplace.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## LogNormal

### <code><i>class</i></code> torch.distributions.log_normal.LogNormal

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">loc()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.loc](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.loc)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.scale](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.scale)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.log_normal.LogNormal.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## LowRankMultivariateNormal

### <code><i>class</i></code> torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.covariance_matrix](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.covariance_matrix)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.precision_matrix](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.precision_matrix)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.scale_tril](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.scale_tril)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## MixtureSameFamily

### <code><i>class</i></code> torch.distributions.mixture_same_family.MixtureSameFamily

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">component_distribution()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.component_distribution](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.component_distribution)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mixture_distribution()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.mixture_distribution](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.mixture_distribution)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Multinomial

### <code><i>class</i></code> torch.distributions.multinomial.Multinomial

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.param_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.param_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">total_count()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.total_count](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.total_count)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multinomial.Multinomial.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## MultivariateNormal

### <code><i>class</i></code> torch.distributions.multivariate_normal.MultivariateNormal

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： `dim`需小于等于8192

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.scale_tril](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.scale_tril)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## NegativeBinomial

### <code><i>class</i></code> torch.distributions.negative_binomial.NegativeBinomial

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.param_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.param_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Normal

### <code><i>class</i></code> torch.distributions.normal.Normal

<div style="margin-left: 2em">

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.stddev](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.stddev)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.normal.Normal.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## OneHotCategorical

### <code><i>class</i></code> torch.distributions.one_hot_categorical.OneHotCategorical

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.has_enumerate_support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.has_enumerate_support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.param_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.param_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Pareto

### <code><i>class</i></code> torch.distributions.pareto.Pareto

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.pareto.Pareto)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.pareto.Pareto.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.pareto.Pareto.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.pareto.Pareto.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.pareto.Pareto.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.pareto.Pareto.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.pareto.Pareto.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.pareto.Pareto.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Poisson

### <code><i>class</i></code> torch.distributions.poisson.Poisson

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.poisson.Poisson.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## RelaxedBernoulli

### <code><i>class</i></code> torch.distributions.relaxed_bernoulli.RelaxedBernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">temperature()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## LogitRelaxedBernoulli

### <code><i>class</i></code> torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.param_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.param_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## RelaxedOneHotCategorical

### <code><i>class</i></code> torch.distributions.relaxed_categorical.RelaxedOneHotCategorical

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">temperature()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## StudentT

### <code><i>class</i></code> torch.distributions.studentT.StudentT

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.studentT.StudentT.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## TransformedDistribution

### <code><i>class</i></code> torch.distributions.transformed_distribution.TransformedDistribution

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## Uniform

### <code><i>class</i></code> torch.distributions.uniform.Uniform

<div style="margin-left: 2em">

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.cdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.cdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.icdf](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.icdf)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.stddev](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.stddev)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.uniform.Uniform.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## VonMises

### <code><i>class</i></code> torch.distributions.von_mises.VonMises

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.sample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.sample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.von_mises.VonMises.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Weibull

### <code><i>class</i></code> torch.distributions.weibull.Weibull

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.weibull.Weibull)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.weibull.Weibull.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.weibull.Weibull.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.weibull.Weibull.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.weibull.Weibull.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.weibull.Weibull.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.weibull.Weibull.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.weibull.Weibull.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Wishart

### <code><i>class</i></code> torch.distributions.wishart.Wishart

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.arg_constraints](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.arg_constraints)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.covariance_matrix](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.covariance_matrix)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.entropy](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.entropy)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.expand](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.expand)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.has_rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.has_rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.log_prob](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.log_prob)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.mean](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.mean)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.mode](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.mode)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.precision_matrix](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.precision_matrix)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.rsample](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.rsample)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.scale_tril](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.scale_tril)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.support](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.support)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.variance](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.wishart.Wishart.variance)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>

## KL Divergence

### torch.distributions.kl.kl_divergence

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kl.kl_divergence](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.kl.kl_divergence)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

## Transforms

### <code><i>class</i></code> torch.distributions.transforms.AbsTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.AbsTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.AbsTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

**限制与说明**： <term>Ascend 950DT</term>：不支持complex64，complex128

</div>

### <code><i>class</i></code> torch.distributions.transforms.AffineTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.AffineTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.AffineTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.CatTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.CatTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.CatTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.CorrCholeskyTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.CorrCholeskyTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.CorrCholeskyTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.ExpTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.ExpTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.ExpTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.LowerCholeskyTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.LowerCholeskyTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.LowerCholeskyTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.PositiveDefiniteTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.PositiveDefiniteTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.PositiveDefiniteTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.PowerTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.PowerTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.PowerTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.ReshapeTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.ReshapeTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.ReshapeTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.SigmoidTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.SigmoidTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.SigmoidTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.SoftplusTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.SoftplusTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.SoftplusTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.TanhTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.TanhTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.TanhTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.StackTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.StackTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.StackTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.SoftmaxTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.SoftmaxTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.SoftmaxTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.StickBreakingTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.StickBreakingTransform](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.StickBreakingTransform)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### <code><i>class</i></code> torch.distributions.transforms.Transform

<div style="margin-left: 2em">

> <font size="3">inv()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.inv](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.Transform.inv)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">sign()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.sign](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.Transform.sign)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">log_abs_det_jacobian()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.log_abs_det_jacobian](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.Transform.log_abs_det_jacobian)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">forward_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.forward_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.Transform.forward_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

> <font size="3">inverse_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.inverse_shape](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.transforms.Transform.inverse_shape)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

</div>

## Constraints

### torch.distributions.constraints.dependent_property

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.dependent_property](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.constraints.dependent_property)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributions.constraints.greater_than

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.greater_than](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.constraints.greater_than)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributions.constraints.less_than

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.less_than](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.constraints.less_than)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributions.constraints.multinomial

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.multinomial](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.constraints.multinomial)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

### torch.distributions.constraints.cat

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.cat](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.constraints.cat)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

### torch.distributions.constraints.stack

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.stack](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.constraints.stack)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✔ |

</div>

## Constraint Registry

### <code><i>class</i></code> torch.distributions.constraint_registry.ConstraintRegistry

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraint_registry.ConstraintRegistry](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

> <font size="3">register()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraint_registry.ConstraintRegistry.register](https://pytorch.org/docs/2.13/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry.register)

**支持情况**：

| 硬件 | 是否支持 |
| ---- | :----: |
| <term>Atlas A2 训练系列产品</term> | ✔ |
| <term>Atlas A3 训练系列产品</term> | ✔ |
| <term>Ascend 950DT</term> | ✘ |

</div>

</div>
