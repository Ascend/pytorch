# torch.distributions

> [!NOTE]   
> 如果API没有"限制与说明"，说明此API和原生API支持度保持一致。<br>

## 目录

- [base API](#base-api)
- [ExponentialFamily](#exponentialfamily)
- [Bernoulli](#bernoulli)
- [MixtureSameFamily](#mixturesamefamily)
- [HalfCauchy](#halfcauchy)
- [Chi2](#chi2)
- [Geometric](#geometric)
- [TransformedDistribution](#transformeddistribution)
- [HalfNormal](#halfnormal)
- [Pareto](#pareto)
- [RelaxedBernoulli](#relaxedbernoulli)
- [Gumbel](#gumbel)
- [KLDivergence](#kldivergence)
- [ConstraintRegistry](#constraintregistry)
- [Transforms](#transforms)

## base API

### _`class`_ torch.distributions.bernoulli.Bernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.entropy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.expand)

**是否支持**：是

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.has_enumerate_support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.has_enumerate_support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.logits)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.param_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.param_shape)

**是否支持**：是

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.probs)

**是否支持**：是

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.sample)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.bernoulli.Bernoulli.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.bernoulli.Bernoulli.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.beta.Beta

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">concentration0()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.concentration0](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.concentration0)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">concentration1()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.concentration1](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.concentration1)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.entropy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.beta.Beta.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.beta.Beta.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.binomial.Binomial

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.entropy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.expand)

**是否支持**：是

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.has_enumerate_support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.has_enumerate_support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.log_prob)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.logits)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.param_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.param_shape)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.probs)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.binomial.Binomial.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.binomial.Binomial.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.chi2.Chi2

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.chi2.Chi2](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.chi2.Chi2)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.chi2.Chi2.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.chi2.Chi2.arg_constraints)

**是否支持**：是

</div>

> <font size="3">df()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.chi2.Chi2.df](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.chi2.Chi2.df)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.chi2.Chi2.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.chi2.Chi2.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.continuous_bernoulli.ContinuousBernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.arg_constraints)

**是否支持**：是

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.cdf)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.icdf)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.log_prob)

**是否支持**：是

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.logits)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.param_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.param_shape)

**是否支持**：是

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.probs)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.rsample)

**是否支持**：是

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.sample)

**是否支持**：是

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.stddev](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.stddev)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.continuous_bernoulli.ContinuousBernoulli.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.dirichlet.Dirichlet

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.expand)

**是否支持**：是

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.has_rsample)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.log_prob)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.mode)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.rsample)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.dirichlet.Dirichlet.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.dirichlet.Dirichlet.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.exponential.Exponential

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.arg_constraints)

**是否支持**：是

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.cdf)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.expand)

**是否支持**：是

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.icdf)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.mode)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.stddev](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.stddev)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exponential.Exponential.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exponential.Exponential.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.fishersnedecor.FisherSnedecor

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.arg_constraints)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.has_rsample)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.fishersnedecor.FisherSnedecor.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.independent.Independent

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.has_enumerate_support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.has_enumerate_support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.rsample)

**是否支持**：是

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.independent.Independent.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.independent.Independent.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.kumaraswamy.Kumaraswamy

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.entropy)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kumaraswamy.Kumaraswamy.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.lkj_cholesky.LKJCholesky

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lkj_cholesky.LKJCholesky.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.laplace.Laplace

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.arg_constraints)

**是否支持**：是

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.cdf)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.icdf)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.mode)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.stddev](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.stddev)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.laplace.Laplace.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.laplace.Laplace.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.covariance_matrix](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.covariance_matrix)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.entropy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.precision_matrix](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.precision_matrix)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.scale_tril](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.scale_tril)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.mixture_same_family.MixtureSameFamily

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.cdf)

**是否支持**：是

</div>

> <font size="3">component_distribution()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.component_distribution](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.component_distribution)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mixture_distribution()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.mixture_distribution](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.mixture_distribution)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.mixture_same_family.MixtureSameFamily.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.multinomial.Multinomial

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.logits)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.param_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.param_shape)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.probs)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">total_count()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.total_count](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.total_count)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multinomial.Multinomial.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multinomial.Multinomial.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.multivariate_normal.MultivariateNormal

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： dim需小于等于8192

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.entropy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.scale_tril](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.scale_tril)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.multivariate_normal.MultivariateNormal.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.negative_binomial.NegativeBinomial

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.arg_constraints)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.logits)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.param_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.param_shape)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.probs)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.sample)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.negative_binomial.NegativeBinomial.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.one_hot_categorical.OneHotCategorical

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.has_enumerate_support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.has_enumerate_support)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.log_prob)

**是否支持**：是

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.logits)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.mode)

**是否支持**：是

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.param_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.param_shape)

**是否支持**：是

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.probs)

**是否支持**：是

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.sample)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.one_hot_categorical.OneHotCategorical.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.poisson.Poisson

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson.arg_constraints)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson.sample)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.poisson.Poisson.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.poisson.Poisson.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.studentT.StudentT

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.entropy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.expand)

**是否支持**：是

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.mode)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.studentT.StudentT.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.studentT.StudentT.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.uniform.Uniform

<div style="margin-left: 2em">

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.cdf)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.icdf)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.mode)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.rsample)

**是否支持**：是

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.stddev](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.stddev)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.uniform.Uniform.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.uniform.Uniform.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.von_mises.VonMises

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.von_mises.VonMises.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.von_mises.VonMises.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.wishart.Wishart

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.arg_constraints)

**是否支持**：是

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.covariance_matrix](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.covariance_matrix)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.entropy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.precision_matrix](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.precision_matrix)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.scale_tril](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.scale_tril)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.wishart.Wishart.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.wishart.Wishart.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.transforms.AbsTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.AbsTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.AbsTransform)

**是否支持**：是

**限制与说明**： <term>Ascend 950DT</term>：不支持complex64，complex128

</div>

### _`class`_ torch.distributions.transforms.AffineTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.AffineTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.AffineTransform)

**是否支持**：是

</div>

### _`class`_ torch.distributions.transforms.CatTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.CatTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.CatTransform)

**是否支持**：是

</div>

### _`class`_ torch.distributions.transforms.CorrCholeskyTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.CorrCholeskyTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.CorrCholeskyTransform)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributions.transforms.ExpTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.ExpTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.ExpTransform)

**是否支持**：是

</div>

### _`class`_ torch.distributions.transforms.LowerCholeskyTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.LowerCholeskyTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.LowerCholeskyTransform)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributions.transforms.PositiveDefiniteTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.PositiveDefiniteTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.PositiveDefiniteTransform)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributions.transforms.PowerTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.PowerTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.PowerTransform)

**是否支持**：是

</div>

### _`class`_ torch.distributions.transforms.ReshapeTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.ReshapeTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.ReshapeTransform)

**是否支持**：是

</div>

### _`class`_ torch.distributions.transforms.SigmoidTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.SigmoidTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.SigmoidTransform)

**是否支持**：是

</div>

### _`class`_ torch.distributions.transforms.SoftplusTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.SoftplusTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.SoftplusTransform)

**是否支持**：是

</div>

### _`class`_ torch.distributions.transforms.TanhTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.TanhTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.TanhTransform)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributions.transforms.StackTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.StackTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.StackTransform)

**是否支持**：是

</div>

### torch.distributions.constraints.dependent_property

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.dependent_property](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.constraints.dependent_property)

**是否支持**：是

</div>

### torch.distributions.constraints.greater_than

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.greater_than](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.constraints.greater_than)

**是否支持**：是

</div>

### torch.distributions.constraints.less_than

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.less_than](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.constraints.less_than)

**是否支持**：是

</div>

### torch.distributions.constraints.multinomial

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.multinomial](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.constraints.multinomial)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## ExponentialFamily

### _`class`_ torch.distributions.distribution.Distribution

<div style="margin-left: 2em">

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.arg_constraints)

**是否支持**：是

</div>

> <font size="3">batch_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.batch_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.batch_shape)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">event_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.event_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.event_shape)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.mode)

**是否支持**：是

</div>

> <font size="3">set_default_validate_args()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.set_default_validate_args](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.set_default_validate_args)

**是否支持**：是

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.stddev](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.stddev)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.distribution.Distribution.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.distribution.Distribution.variance)

**是否支持**：是

</div>

</div>

## Bernoulli

### _`class`_ torch.distributions.exp_family.ExponentialFamily

<div style="margin-left: 2em">

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.exp_family.ExponentialFamily.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.exp_family.ExponentialFamily.entropy)

**是否支持**：是

</div>

</div>

## MixtureSameFamily

### _`class`_ torch.distributions.categorical.Categorical

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.expand)

**是否支持**：是

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.has_enumerate_support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.has_enumerate_support)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.log_prob)

**是否支持**：是

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.logits)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.mode)

**是否支持**：是

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.param_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.param_shape)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.probs)

**是否支持**：是

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.categorical.Categorical.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.categorical.Categorical.variance)

**是否支持**：是

</div>

</div>

## HalfCauchy

### _`class`_ torch.distributions.cauchy.Cauchy

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.arg_constraints)

**是否支持**：是

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.cdf)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.expand)

**是否支持**：是

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.icdf)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.log_prob)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.rsample)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.cauchy.Cauchy.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.cauchy.Cauchy.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Chi2

### _`class`_ torch.distributions.gamma.Gamma

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.cdf)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.has_rsample)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.log_prob)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.mode)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gamma.Gamma.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gamma.Gamma.variance)

**是否支持**：是

</div>

</div>

## Geometric

### _`class`_ torch.distributions.geometric.Geometric

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.entropy)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.logits)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.probs)

**是否支持**：是

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.sample)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.geometric.Geometric.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.geometric.Geometric.variance)

**是否支持**：是

</div>

</div>

## TransformedDistribution

### _`class`_ torch.distributions.gumbel.Gumbel

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.stddev](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.stddev)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.gumbel.Gumbel.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.gumbel.Gumbel.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.half_cauchy.HalfCauchy

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.arg_constraints)

**是否支持**：是

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.cdf)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.icdf)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.scale](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.scale)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_cauchy.HalfCauchy.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_cauchy.HalfCauchy.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.half_normal.HalfNormal

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.arg_constraints)

**是否支持**：是

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.cdf)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.has_rsample)

**是否支持**：是

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.icdf)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.log_prob)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.scale](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.scale)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.half_normal.HalfNormal.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.half_normal.HalfNormal.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.log_normal.LogNormal

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.entropy)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">loc()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.loc](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.loc)

**是否支持**：是

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.mode)

**是否支持**：是

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.scale](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.scale)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.log_normal.LogNormal.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.log_normal.LogNormal.variance)

**是否支持**：是

</div>

</div>

### _`class`_ torch.distributions.relaxed_bernoulli.RelaxedBernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.arg_constraints)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.has_rsample)

**是否支持**：是

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits)

**是否支持**：是

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">temperature()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.relaxed_categorical.RelaxedOneHotCategorical

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical)

**是否支持**：是

**限制与说明**： 可能回退至CPU执行

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.arg_constraints)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.expand)

**是否支持**：是

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits)

**是否支持**：是

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.support)

**是否支持**：是

</div>

> <font size="3">temperature()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

### _`class`_ torch.distributions.weibull.Weibull

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.weibull.Weibull)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.weibull.Weibull.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.weibull.Weibull.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.weibull.Weibull.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.weibull.Weibull.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.weibull.Weibull.mode)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.weibull.Weibull.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.weibull.Weibull.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.weibull.Weibull.variance)

**是否支持**：是

</div>

</div>

## HalfNormal

### _`class`_ torch.distributions.normal.Normal

<div style="margin-left: 2em">

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.arg_constraints)

**是否支持**：是

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.cdf)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.has_rsample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.icdf)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.mean)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.mode)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.rsample)

**是否支持**：是

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.stddev](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.stddev)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.normal.Normal.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.normal.Normal.variance)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Pareto

### _`class`_ torch.distributions.pareto.Pareto

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.pareto.Pareto)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.pareto.Pareto.arg_constraints)

**是否支持**：是

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.entropy](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.pareto.Pareto.entropy)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.pareto.Pareto.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.mean](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.pareto.Pareto.mean)

**是否支持**：是

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.mode](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.pareto.Pareto.mode)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.pareto.Pareto.support)

**是否支持**：是

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.pareto.Pareto.variance](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.pareto.Pareto.variance)

**是否支持**：是

</div>

</div>

## RelaxedBernoulli

### _`class`_ torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.expand)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.logits](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.logits)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.param_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.param_shape)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.probs](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.probs)

**是否支持**：是

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.rsample)

**是否支持**：是

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.support)

**是否支持**：是

</div>

</div>

## Gumbel

### _`class`_ torch.distributions.transformed_distribution.TransformedDistribution

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution)

**是否支持**：是

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.arg_constraints](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.arg_constraints)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.cdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.cdf)

**是否支持**：是

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.expand](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.expand)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.has_rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.has_rsample)

**是否支持**：是

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.icdf](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf)

**是否支持**：是

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.log_prob](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.log_prob)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.rsample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.rsample)

**是否支持**：是

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.sample](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.sample)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transformed_distribution.TransformedDistribution.support](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.support)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## KLDivergence

### torch.distributions.kl.kl_divergence

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.kl.kl_divergence](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.kl.kl_divergence)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

## ConstraintRegistry

### _`class`_ torch.distributions.transforms.SoftmaxTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.SoftmaxTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.SoftmaxTransform)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

### _`class`_ torch.distributions.constraint_registry.ConstraintRegistry

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraint_registry.ConstraintRegistry](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

> <font size="3">register()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraint_registry.ConstraintRegistry.register](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry.register)

**是否支持**：是，暂不支持<term>Ascend 950DT</term>

</div>

</div>

## Transforms

### _`class`_ torch.distributions.transforms.StickBreakingTransform

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.StickBreakingTransform](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.StickBreakingTransform)

**是否支持**：是

</div>

### _`class`_ torch.distributions.transforms.Transform

<div style="margin-left: 2em">

> <font size="3">inv()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.inv](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.Transform.inv)

**是否支持**：是

</div>

> <font size="3">sign()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.sign](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.Transform.sign)

**是否支持**：是

</div>

> <font size="3">log_abs_det_jacobian()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.log_abs_det_jacobian](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.Transform.log_abs_det_jacobian)

**是否支持**：是

</div>

> <font size="3">forward_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.forward_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.Transform.forward_shape)

**是否支持**：是

</div>

> <font size="3">inverse_shape()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.transforms.Transform.inverse_shape](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.transforms.Transform.inverse_shape)

**是否支持**：是

</div>

</div>

### torch.distributions.constraints.cat

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.cat](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.constraints.cat)

**是否支持**：是

</div>

### torch.distributions.constraints.stack

<div style="margin-left: 2em">

**原生文档**：[torch.distributions.constraints.stack](https://pytorch.org/docs/2.11/distributions.html#torch.distributions.constraints.stack)

**是否支持**：是

</div>
