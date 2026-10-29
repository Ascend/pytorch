# torch.distributions

> [!NOTE]
>
> - If an API has "Constraints & Notes", it means that the support level of the API on the Ascend NPU differs from the native version. Refer to the specific notes to ensure compatibility with the Ascend NPU platform.
> - Some APIs exist in the [PyTorch Community Documentation](https://pytorch.org/docs/2.12/) but are not included in this support list. These APIs have not been verified yet. Use them with caution. We will continue the verification work and update the document upon completion.
> - Product support scope: This document only provides verified product information. Unverified products are temporarily excluded.
> - The modules listed in this directory are consistent with the native documentation. For module-related notes, refer to the native documentation [LINK](https://pytorch.org/docs/2.12/distributions.html).

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>Contents</b></font></div>

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

**Native Documentation**: [torch.distributions.distribution.Distribution.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.arg_constraints)

**Support Status**:

<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id3 -->

</div>

> <font size="3">batch_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.distribution.Distribution.batch_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.batch_shape)

**Support Status**:

<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id6 -->

</div>

> <font size="3">event_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.distribution.Distribution.event_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.event_shape)

**Support Status**:

<!-- npu="910b" id7 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id9 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.distribution.Distribution.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.mean)

**Support Status**:

<!-- npu="910b" id10 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id12 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.distribution.Distribution.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.mode)

**Support Status**:

<!-- npu="910b" id13 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id15 -->

</div>

> <font size="3">set_default_validate_args()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.distribution.Distribution.set_default_validate_args](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.set_default_validate_args)

**Support Status**:

<!-- npu="910b" id16 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id18 -->

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.distribution.Distribution.stddev](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.stddev)

**Support Status**:

<!-- npu="910b" id19 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id21 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.distribution.Distribution.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.support)

**Support Status**:

<!-- npu="910b" id22 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id24 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.distribution.Distribution.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.distribution.Distribution.variance)

**Support Status**:

<!-- npu="910b" id25 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id27 -->

</div>

</div>

## ExponentialFamily

### <code><i>class</i></code> torch.distributions.exp_family.ExponentialFamily

<div style="margin-left: 2em">

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exp_family.ExponentialFamily.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exp_family.ExponentialFamily.entropy)

**Support Status**:

<!-- npu="910b" id28 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id30 -->

</div>

</div>

## Bernoulli

### <code><i>class</i></code> torch.distributions.bernoulli.Bernoulli

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli)

**Support Status**:

<!-- npu="910b" id31 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id33 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.arg_constraints)

**Support Status**:

<!-- npu="910b" id34 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id36 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.entropy)

**Support Status**:

<!-- npu="910b" id37 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id39 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.expand)

**Support Status**:

<!-- npu="910b" id40 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id42 -->

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.has_enumerate_support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.has_enumerate_support)

**Support Status**:

<!-- npu="910b" id43 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id45 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.log_prob)

**Support Status**:

<!-- npu="910b" id46 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id48 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.logits)

**Support Status**:

<!-- npu="910b" id49 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id51 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.mean)

**Support Status**:

<!-- npu="910b" id52 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id54 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.mode)

**Support Status**:

<!-- npu="910b" id55 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id57 -->

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.param_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.param_shape)

**Support Status**:

<!-- npu="910b" id58 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id60 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.probs)

**Support Status**:

<!-- npu="910b" id61 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id63 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.sample)

**Support Status**:

<!-- npu="910b" id64 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id66 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.support)

**Support Status**:

<!-- npu="910b" id67 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id69 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.bernoulli.Bernoulli.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.bernoulli.Bernoulli.variance)

**Support Status**:

<!-- npu="910b" id70 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id72 -->

</div>

</div>

## Beta

### <code><i>class</i></code> torch.distributions.beta.Beta

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta)

**Support Status**:

<!-- npu="910b" id73 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id75 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.arg_constraints)

**Support Status**:

<!-- npu="910b" id76 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id78 -->

</div>

> <font size="3">concentration0()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.concentration0](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.concentration0)

**Support Status**:

<!-- npu="910b" id79 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id81 -->

</div>

> <font size="3">concentration1()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.concentration1](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.concentration1)

**Support Status**:

<!-- npu="910b" id82 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id84 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.entropy)

**Support Status**:

<!-- npu="910b" id85 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id87 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.expand)

**Support Status**:

<!-- npu="910b" id88 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id90 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.has_rsample)

**Support Status**:

<!-- npu="910b" id91 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id93 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.log_prob)

**Support Status**:

<!-- npu="910b" id94 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id96 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.mean)

**Support Status**:

<!-- npu="910b" id97 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id99 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.mode)

**Support Status**:

<!-- npu="910b" id100 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id102 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.rsample)

**Support Status**:

<!-- npu="910b" id103 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id105 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.support)

**Support Status**:

<!-- npu="910b" id106 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id108 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.beta.Beta.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.beta.Beta.variance)

**Support Status**:

<!-- npu="910b" id109 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id111 -->

</div>

</div>

## Binomial

### <code><i>class</i></code> torch.distributions.binomial.Binomial

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial)

**Support Status**:

<!-- npu="910b" id112 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id114 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.arg_constraints)

**Support Status**:

<!-- npu="910b" id115 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id117 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.entropy)

**Support Status**:

<!-- npu="910b" id118 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id120 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.expand)

**Support Status**:

<!-- npu="910b" id121 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id123 -->

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.has_enumerate_support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.has_enumerate_support)

**Support Status**:

<!-- npu="910b" id124 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id126 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.log_prob)

**Support Status**:

<!-- npu="910b" id127 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id129 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.logits)

**Support Status**:

<!-- npu="910b" id130 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id132 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.mean)

**Support Status**:

<!-- npu="910b" id133 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id135 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.mode)

**Support Status**:

<!-- npu="910b" id136 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id138 -->

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.param_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.param_shape)

**Support Status**:

<!-- npu="910b" id139 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id141 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.probs)

**Support Status**:

<!-- npu="910b" id142 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id144 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.sample)

**Support Status**:

<!-- npu="910b" id145 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id147 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.support)

**Support Status**:

<!-- npu="910b" id148 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id150 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.binomial.Binomial.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.binomial.Binomial.variance)

**Support Status**:

<!-- npu="910b" id151 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id153 -->

</div>

</div>

## Categorical

### <code><i>class</i></code> torch.distributions.categorical.Categorical

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical)

**Support Status**:

<!-- npu="910b" id154 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id156 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.arg_constraints)

**Support Status**:

<!-- npu="910b" id157 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id159 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.entropy)

**Support Status**:

<!-- npu="910b" id160 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id162 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.expand)

**Support Status**:

<!-- npu="910b" id163 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id165 -->

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.has_enumerate_support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.has_enumerate_support)

**Support Status**:

<!-- npu="910b" id166 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id168 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.log_prob)

**Support Status**:

<!-- npu="910b" id169 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id171 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.logits)

**Support Status**:

<!-- npu="910b" id172 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id174 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.mean)

**Support Status**:

<!-- npu="910b" id175 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id177 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.mode)

**Support Status**:

<!-- npu="910b" id178 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id180 -->

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.param_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.param_shape)

**Support Status**:

<!-- npu="910b" id181 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id183 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.probs)

**Support Status**:

<!-- npu="910b" id184 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id186 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.sample)

**Support Status**:

<!-- npu="910b" id187 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id189 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.support)

**Support Status**:

<!-- npu="910b" id190 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id192 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.categorical.Categorical.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.categorical.Categorical.variance)

**Support Status**:

<!-- npu="910b" id193 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id195 -->

</div>

</div>

## Cauchy

### <code><i>class</i></code> torch.distributions.cauchy.Cauchy

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy)

**Support Status**:

<!-- npu="910b" id196 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id198 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.arg_constraints)

**Support Status**:

<!-- npu="910b" id199 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id201 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.cdf)

**Support Status**:

<!-- npu="910b" id202 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id204 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.entropy)

**Support Status**:

<!-- npu="910b" id205 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id207 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.expand)

**Support Status**:

<!-- npu="910b" id208 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id210 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.has_rsample)

**Support Status**:

<!-- npu="910b" id211 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id213 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.icdf)

**Support Status**:

<!-- npu="910b" id214 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id216 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.log_prob)

**Support Status**:

<!-- npu="910b" id217 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id219 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.mean)

**Support Status**:

<!-- npu="910b" id220 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id222 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.mode)

**Support Status**:

<!-- npu="910b" id223 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id225 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.rsample)

**Support Status**:

<!-- npu="910b" id226 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id228 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.support)

**Support Status**:

<!-- npu="910b" id229 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id231 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.cauchy.Cauchy.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.cauchy.Cauchy.variance)

**Support Status**:

<!-- npu="910b" id232 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id234 -->

</div>

</div>

## Chi2

### <code><i>class</i></code> torch.distributions.chi2.Chi2

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.chi2.Chi2](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.chi2.Chi2)

**Support Status**:

<!-- npu="910b" id235 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id237 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.chi2.Chi2.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.chi2.Chi2.arg_constraints)

**Support Status**:

<!-- npu="910b" id238 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id240 -->

</div>

> <font size="3">df()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.chi2.Chi2.df](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.chi2.Chi2.df)

**Support Status**:

<!-- npu="910b" id241 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id243 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.chi2.Chi2.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.chi2.Chi2.expand)

**Support Status**:

<!-- npu="910b" id244 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id246 -->

</div>

</div>

## ContinuousBernoulli

### <code><i>class</i></code> torch.distributions.continuous_bernoulli.ContinuousBernoulli

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli)

**Support Status**:

<!-- npu="910b" id247 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id249 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.arg_constraints)

**Support Status**:

<!-- npu="910b" id250 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id252 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.cdf)

**Support Status**:

<!-- npu="910b" id253 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id255 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.entropy)

**Support Status**:

<!-- npu="910b" id256 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id258 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.expand)

**Support Status**:

<!-- npu="910b" id259 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id261 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.has_rsample)

**Support Status**:

<!-- npu="910b" id262 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id264 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.icdf)

**Support Status**:

<!-- npu="910b" id265 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id267 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.log_prob)

**Support Status**:

<!-- npu="910b" id268 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id270 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.logits)

**Support Status**:

<!-- npu="910b" id271 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id273 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.mean)

**Support Status**:

<!-- npu="910b" id274 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id276 -->

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.param_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.param_shape)

**Support Status**:

<!-- npu="910b" id277 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id277 -->
<!-- npu="A3" id278 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id278 -->
<!-- npu="950" id279 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id279 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.probs)

**Support Status**:

<!-- npu="910b" id280 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id282 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.rsample)

**Support Status**:

<!-- npu="910b" id283 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id283 -->
<!-- npu="A3" id284 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id284 -->
<!-- npu="950" id285 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id285 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.sample)

**Support Status**:

<!-- npu="910b" id286 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id288 -->

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.stddev](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.stddev)

**Support Status**:

<!-- npu="910b" id289 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id291 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.support)

**Support Status**:

<!-- npu="910b" id292 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id294 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.continuous_bernoulli.ContinuousBernoulli.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.continuous_bernoulli.ContinuousBernoulli.variance)

**Support Status**:

<!-- npu="910b" id295 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id297 -->

</div>

</div>

## Dirichlet

### <code><i>class</i></code> torch.distributions.dirichlet.Dirichlet

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet)

**Support Status**:

<!-- npu="910b" id298 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id298 -->
<!-- npu="A3" id299 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id299 -->
<!-- npu="950" id300 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id300 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.arg_constraints)

**Support Status**:

<!-- npu="910b" id301 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id301 -->
<!-- npu="A3" id302 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id302 -->
<!-- npu="950" id303 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id303 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.entropy)

**Support Status**:

<!-- npu="910b" id304 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id306 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.expand)

**Support Status**:

<!-- npu="910b" id307 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id309 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.has_rsample)

**Support Status**:

<!-- npu="910b" id310 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id310 -->
<!-- npu="A3" id311 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id311 -->
<!-- npu="950" id312 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id312 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.log_prob)

**Support Status**:

<!-- npu="910b" id313 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id315 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.mean)

**Support Status**:

<!-- npu="910b" id316 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id318 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.mode)

**Support Status**:

<!-- npu="910b" id319 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id319 -->
<!-- npu="A3" id320 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id320 -->
<!-- npu="950" id321 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id321 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.rsample)

**Support Status**:

<!-- npu="910b" id322 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id324 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.support)

**Support Status**:

<!-- npu="910b" id325 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id327 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.dirichlet.Dirichlet.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.dirichlet.Dirichlet.variance)

**Support Status**:

<!-- npu="910b" id328 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id330 -->

</div>

</div>

## Exponential

### <code><i>class</i></code> torch.distributions.exponential.Exponential

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential)

**Support Status**:

<!-- npu="910b" id331 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id333 -->

**Constraints & Notes**:

<!-- npu="950" id1433 -->
- On <term>Ascend 950DT products</term>, the random number generation behavior is consistent with the GPU implementation, and the generated results differ from those of <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>.
<!-- end id1433 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.arg_constraints)

**Support Status**:

<!-- npu="910b" id334 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id336 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.cdf)

**Support Status**:

<!-- npu="910b" id337 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id339 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.entropy)

**Support Status**:

<!-- npu="910b" id340 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id340 -->
<!-- npu="A3" id341 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id341 -->
<!-- npu="950" id342 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id342 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.expand)

**Support Status**:

<!-- npu="910b" id343 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id345 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.has_rsample)

**Support Status**:

<!-- npu="910b" id346 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id348 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.icdf)

**Support Status**:

<!-- npu="910b" id349 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id351 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.log_prob)

**Support Status**:

<!-- npu="910b" id352 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id352 -->
<!-- npu="A3" id353 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id353 -->
<!-- npu="950" id354 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id354 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.mean)

**Support Status**:

<!-- npu="910b" id355 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id357 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.mode)

**Support Status**:

<!-- npu="910b" id358 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id360 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.rsample)

**Support Status**:

<!-- npu="910b" id361 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id363 -->

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.stddev](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.stddev)

**Support Status**:

<!-- npu="910b" id364 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id364 -->
<!-- npu="A3" id365 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id365 -->
<!-- npu="950" id366 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id366 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.support)

**Support Status**:

<!-- npu="910b" id367 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id367 -->
<!-- npu="A3" id368 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id368 -->
<!-- npu="950" id369 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id369 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.exponential.Exponential.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.exponential.Exponential.variance)

**Support Status**:

<!-- npu="910b" id370 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id370 -->
<!-- npu="A3" id371 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id371 -->
<!-- npu="950" id372 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id372 -->

</div>

</div>

## FisherSnedecor

### <code><i>class</i></code> torch.distributions.fishersnedecor.FisherSnedecor

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor)

**Support Status**:

<!-- npu="910b" id373 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id373 -->
<!-- npu="A3" id374 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id374 -->
<!-- npu="950" id375 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id375 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.arg_constraints)

**Support Status**:

<!-- npu="910b" id376 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id376 -->
<!-- npu="A3" id377 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id377 -->
<!-- npu="950" id378 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id378 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.expand)

**Support Status**:

<!-- npu="910b" id379 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id379 -->
<!-- npu="A3" id380 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id380 -->
<!-- npu="950" id381 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id381 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.has_rsample)

**Support Status**:

<!-- npu="910b" id382 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id382 -->
<!-- npu="A3" id383 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id383 -->
<!-- npu="950" id384 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id384 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.log_prob)

**Support Status**:

<!-- npu="910b" id385 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id385 -->
<!-- npu="A3" id386 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id386 -->
<!-- npu="950" id387 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id387 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.mean)

**Support Status**:

<!-- npu="910b" id388 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id388 -->
<!-- npu="A3" id389 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id389 -->
<!-- npu="950" id390 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id390 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.mode)

**Support Status**:

<!-- npu="910b" id391 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id391 -->
<!-- npu="A3" id392 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id392 -->
<!-- npu="950" id393 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id393 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.rsample)

**Support Status**:

<!-- npu="910b" id394 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id394 -->
<!-- npu="A3" id395 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id395 -->
<!-- npu="950" id396 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id396 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.support)

**Support Status**:

<!-- npu="910b" id397 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id397 -->
<!-- npu="A3" id398 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id398 -->
<!-- npu="950" id399 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id399 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.fishersnedecor.FisherSnedecor.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.fishersnedecor.FisherSnedecor.variance)

**Support Status**:

<!-- npu="910b" id400 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id400 -->
<!-- npu="A3" id401 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id401 -->
<!-- npu="950" id402 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id402 -->

</div>

</div>

## Gamma

### <code><i>class</i></code> torch.distributions.gamma.Gamma

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma)

**Support Status**:

<!-- npu="910b" id403 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id403 -->
<!-- npu="A3" id404 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id404 -->
<!-- npu="950" id405 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id405 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.arg_constraints)

**Support Status**:

<!-- npu="910b" id406 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id406 -->
<!-- npu="A3" id407 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id407 -->
<!-- npu="950" id408 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id408 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.cdf)

**Support Status**:

<!-- npu="910b" id409 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id409 -->
<!-- npu="A3" id410 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id410 -->
<!-- npu="950" id411 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id411 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.entropy)

**Support Status**:

<!-- npu="910b" id412 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id412 -->
<!-- npu="A3" id413 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id413 -->
<!-- npu="950" id414 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id414 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.expand)

**Support Status**:

<!-- npu="910b" id415 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id415 -->
<!-- npu="A3" id416 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id416 -->
<!-- npu="950" id417 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id417 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.has_rsample)

**Support Status**:

<!-- npu="910b" id418 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id418 -->
<!-- npu="A3" id419 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id419 -->
<!-- npu="950" id420 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id420 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.log_prob)

**Support Status**:

<!-- npu="910b" id421 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id421 -->
<!-- npu="A3" id422 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id422 -->
<!-- npu="950" id423 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id423 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.mean)

**Support Status**:

<!-- npu="910b" id424 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id424 -->
<!-- npu="A3" id425 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id425 -->
<!-- npu="950" id426 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id426 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.mode)

**Support Status**:

<!-- npu="910b" id427 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id427 -->
<!-- npu="A3" id428 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id428 -->
<!-- npu="950" id429 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id429 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.rsample)

**Support Status**:

<!-- npu="910b" id430 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id430 -->
<!-- npu="A3" id431 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id431 -->
<!-- npu="950" id432 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id432 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.support)

**Support Status**:

<!-- npu="910b" id433 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id433 -->
<!-- npu="A3" id434 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id434 -->
<!-- npu="950" id435 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id435 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gamma.Gamma.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gamma.Gamma.variance)

**Support Status**:

<!-- npu="910b" id436 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id436 -->
<!-- npu="A3" id437 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id437 -->
<!-- npu="950" id438 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id438 -->

</div>

</div>

## Geometric

### <code><i>class</i></code> torch.distributions.geometric.Geometric

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric)

**Support Status**:

<!-- npu="910b" id439 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id439 -->
<!-- npu="A3" id440 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id440 -->
<!-- npu="950" id441 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id441 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.arg_constraints)

**Support Status**:

<!-- npu="910b" id442 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id442 -->
<!-- npu="A3" id443 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id443 -->
<!-- npu="950" id444 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id444 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.entropy)

**Support Status**:

<!-- npu="910b" id445 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id445 -->
<!-- npu="A3" id446 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id446 -->
<!-- npu="950" id447 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id447 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.expand)

**Support Status**:

<!-- npu="910b" id448 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id448 -->
<!-- npu="A3" id449 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id449 -->
<!-- npu="950" id450 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id450 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.log_prob)

**Support Status**:

<!-- npu="910b" id451 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id451 -->
<!-- npu="A3" id452 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id452 -->
<!-- npu="950" id453 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id453 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.logits)

**Support Status**:

<!-- npu="910b" id454 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id454 -->
<!-- npu="A3" id455 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id455 -->
<!-- npu="950" id456 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id456 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.mean)

**Support Status**:

<!-- npu="910b" id457 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id457 -->
<!-- npu="A3" id458 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id458 -->
<!-- npu="950" id459 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id459 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.mode)

**Support Status**:

<!-- npu="910b" id460 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id460 -->
<!-- npu="A3" id461 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id461 -->
<!-- npu="950" id462 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id462 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.probs)

**Support Status**:

<!-- npu="910b" id463 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id463 -->
<!-- npu="A3" id464 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id464 -->
<!-- npu="950" id465 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id465 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.sample)

**Support Status**:

<!-- npu="910b" id466 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id466 -->
<!-- npu="A3" id467 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id467 -->
<!-- npu="950" id468 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id468 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.support)

**Support Status**:

<!-- npu="910b" id469 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id469 -->
<!-- npu="A3" id470 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id470 -->
<!-- npu="950" id471 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id471 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.geometric.Geometric.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.geometric.Geometric.variance)

**Support Status**:

<!-- npu="910b" id472 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id472 -->
<!-- npu="A3" id473 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id473 -->
<!-- npu="950" id474 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id474 -->

</div>

</div>

## Gumbel

### <code><i>class</i></code> torch.distributions.gumbel.Gumbel

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel)

**Support Status**:

<!-- npu="910b" id475 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id475 -->
<!-- npu="A3" id476 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id476 -->
<!-- npu="950" id477 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id477 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.arg_constraints)

**Support Status**:

<!-- npu="910b" id478 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id478 -->
<!-- npu="A3" id479 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id479 -->
<!-- npu="950" id480 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id480 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.entropy)

**Support Status**:

<!-- npu="910b" id481 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id481 -->
<!-- npu="A3" id482 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id482 -->
<!-- npu="950" id483 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id483 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.expand)

**Support Status**:

<!-- npu="910b" id484 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id484 -->
<!-- npu="A3" id485 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id485 -->
<!-- npu="950" id486 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id486 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.log_prob)

**Support Status**:

<!-- npu="910b" id487 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id487 -->
<!-- npu="A3" id488 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id488 -->
<!-- npu="950" id489 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id489 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.mean)

**Support Status**:

<!-- npu="910b" id490 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id490 -->
<!-- npu="A3" id491 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id491 -->
<!-- npu="950" id492 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id492 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.mode)

**Support Status**:

<!-- npu="910b" id493 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id493 -->
<!-- npu="A3" id494 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id494 -->
<!-- npu="950" id495 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id495 -->

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.stddev](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.stddev)

**Support Status**:

<!-- npu="910b" id496 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id496 -->
<!-- npu="A3" id497 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id497 -->
<!-- npu="950" id498 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id498 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.support)

**Support Status**:

<!-- npu="910b" id499 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id499 -->
<!-- npu="A3" id500 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id500 -->
<!-- npu="950" id501 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id501 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.gumbel.Gumbel.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.gumbel.Gumbel.variance)

**Support Status**:

<!-- npu="910b" id502 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id502 -->
<!-- npu="A3" id503 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id503 -->
<!-- npu="950" id504 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id504 -->

</div>

</div>

## HalfCauchy

### <code><i>class</i></code> torch.distributions.half_cauchy.HalfCauchy

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy)

**Support Status**:

<!-- npu="910b" id505 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id505 -->
<!-- npu="A3" id506 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id506 -->
<!-- npu="950" id507 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id507 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.arg_constraints)

**Support Status**:

<!-- npu="910b" id508 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id508 -->
<!-- npu="A3" id509 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id509 -->
<!-- npu="950" id510 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id510 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.cdf)

**Support Status**:

<!-- npu="910b" id511 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id511 -->
<!-- npu="A3" id512 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id512 -->
<!-- npu="950" id513 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id513 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.entropy)

**Support Status**:

<!-- npu="910b" id514 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id514 -->
<!-- npu="A3" id515 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id515 -->
<!-- npu="950" id516 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id516 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.expand)

**Support Status**:

<!-- npu="910b" id517 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id517 -->
<!-- npu="A3" id518 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id518 -->
<!-- npu="950" id519 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id519 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.has_rsample)

**Support Status**:

<!-- npu="910b" id520 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id520 -->
<!-- npu="A3" id521 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id521 -->
<!-- npu="950" id522 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id522 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.icdf)

**Support Status**:

<!-- npu="910b" id523 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id523 -->
<!-- npu="A3" id524 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id524 -->
<!-- npu="950" id525 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id525 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.log_prob)

**Support Status**:

<!-- npu="910b" id526 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id526 -->
<!-- npu="A3" id527 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id527 -->
<!-- npu="950" id528 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id528 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.mean)

**Support Status**:

<!-- npu="910b" id529 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id529 -->
<!-- npu="A3" id530 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id530 -->
<!-- npu="950" id531 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id531 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.mode)

**Support Status**:

<!-- npu="910b" id532 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id532 -->
<!-- npu="A3" id533 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id533 -->
<!-- npu="950" id534 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id534 -->

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.scale](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.scale)

**Support Status**:

<!-- npu="910b" id535 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id535 -->
<!-- npu="A3" id536 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id536 -->
<!-- npu="950" id537 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id537 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.support)

**Support Status**:

<!-- npu="910b" id538 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id538 -->
<!-- npu="A3" id539 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id539 -->
<!-- npu="950" id540 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id540 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_cauchy.HalfCauchy.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_cauchy.HalfCauchy.variance)

**Support Status**:

<!-- npu="910b" id541 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id541 -->
<!-- npu="A3" id542 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id542 -->
<!-- npu="950" id543 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id543 -->

</div>

</div>

## HalfNormal

### <code><i>class</i></code> torch.distributions.half_normal.HalfNormal

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal)

**Support Status**:

<!-- npu="910b" id544 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id544 -->
<!-- npu="A3" id545 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id545 -->
<!-- npu="950" id546 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id546 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.arg_constraints)

**Support Status**:

<!-- npu="910b" id547 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id547 -->
<!-- npu="A3" id548 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id548 -->
<!-- npu="950" id549 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id549 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.cdf)

**Support Status**:

<!-- npu="910b" id550 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id550 -->
<!-- npu="A3" id551 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id551 -->
<!-- npu="950" id552 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id552 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.entropy)

**Support Status**:

<!-- npu="910b" id553 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id553 -->
<!-- npu="A3" id554 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id554 -->
<!-- npu="950" id555 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id555 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.expand)

**Support Status**:

<!-- npu="910b" id556 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id556 -->
<!-- npu="A3" id557 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id557 -->
<!-- npu="950" id558 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id558 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.has_rsample)

**Support Status**:

<!-- npu="910b" id559 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id559 -->
<!-- npu="A3" id560 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id560 -->
<!-- npu="950" id561 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id561 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.icdf)

**Support Status**:

<!-- npu="910b" id562 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id562 -->
<!-- npu="A3" id563 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id563 -->
<!-- npu="950" id564 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id564 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.log_prob)

**Support Status**:

<!-- npu="910b" id565 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id565 -->
<!-- npu="A3" id566 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id566 -->
<!-- npu="950" id567 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id567 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.mean)

**Support Status**:

<!-- npu="910b" id568 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id568 -->
<!-- npu="A3" id569 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id569 -->
<!-- npu="950" id570 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id570 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.mode)

**Support Status**:

<!-- npu="910b" id571 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id571 -->
<!-- npu="A3" id572 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id572 -->
<!-- npu="950" id573 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id573 -->

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.scale](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.scale)

**Support Status**:

<!-- npu="910b" id574 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id574 -->
<!-- npu="A3" id575 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id575 -->
<!-- npu="950" id576 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id576 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.support)

**Support Status**:

<!-- npu="910b" id577 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id577 -->
<!-- npu="A3" id578 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id578 -->
<!-- npu="950" id579 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id579 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.half_normal.HalfNormal.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.half_normal.HalfNormal.variance)

**Support Status**:

<!-- npu="910b" id580 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id580 -->
<!-- npu="A3" id581 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id581 -->
<!-- npu="950" id582 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id582 -->

</div>

</div>

## Independent

### <code><i>class</i></code> torch.distributions.independent.Independent

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent)

**Support Status**:

<!-- npu="910b" id583 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id583 -->
<!-- npu="A3" id584 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id584 -->
<!-- npu="950" id585 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id585 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.arg_constraints)

**Support Status**:

<!-- npu="910b" id586 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id586 -->
<!-- npu="A3" id587 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id587 -->
<!-- npu="950" id588 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id588 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.entropy)

**Support Status**:

<!-- npu="910b" id589 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id589 -->
<!-- npu="A3" id590 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id590 -->
<!-- npu="950" id591 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id591 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.expand)

**Support Status**:

<!-- npu="910b" id592 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id592 -->
<!-- npu="A3" id593 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id593 -->
<!-- npu="950" id594 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id594 -->

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.has_enumerate_support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.has_enumerate_support)

**Support Status**:

<!-- npu="910b" id595 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id595 -->
<!-- npu="A3" id596 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id596 -->
<!-- npu="950" id597 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id597 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.has_rsample)

**Support Status**:

<!-- npu="910b" id598 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id598 -->
<!-- npu="A3" id599 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id599 -->
<!-- npu="950" id600 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id600 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.log_prob)

**Support Status**:

<!-- npu="910b" id601 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id601 -->
<!-- npu="A3" id602 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id602 -->
<!-- npu="950" id603 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id603 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.mean)

**Support Status**:

<!-- npu="910b" id604 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id604 -->
<!-- npu="A3" id605 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id605 -->
<!-- npu="950" id606 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id606 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.mode)

**Support Status**:

<!-- npu="910b" id607 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id607 -->
<!-- npu="A3" id608 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id608 -->
<!-- npu="950" id609 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id609 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.rsample)

**Support Status**:

<!-- npu="910b" id610 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id610 -->
<!-- npu="A3" id611 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id611 -->
<!-- npu="950" id612 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id612 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.sample)

**Support Status**:

<!-- npu="910b" id613 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id613 -->
<!-- npu="A3" id614 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id614 -->
<!-- npu="950" id615 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id615 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.support)

**Support Status**:

<!-- npu="910b" id616 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id616 -->
<!-- npu="A3" id617 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id617 -->
<!-- npu="950" id618 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id618 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.independent.Independent.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.independent.Independent.variance)

**Support Status**:

<!-- npu="910b" id619 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id619 -->
<!-- npu="A3" id620 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id620 -->
<!-- npu="950" id621 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id621 -->

</div>

</div>

## Kumaraswamy

### <code><i>class</i></code> torch.distributions.kumaraswamy.Kumaraswamy

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy)

**Support Status**:

<!-- npu="910b" id622 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id622 -->
<!-- npu="A3" id623 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id623 -->
<!-- npu="950" id624 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id624 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.arg_constraints)

**Support Status**:

<!-- npu="910b" id625 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id625 -->
<!-- npu="A3" id626 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id626 -->
<!-- npu="950" id627 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id627 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.entropy)

**Support Status**:

<!-- npu="910b" id628 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id628 -->
<!-- npu="A3" id629 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id629 -->
<!-- npu="950" id630 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id630 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.expand)

**Support Status**:

<!-- npu="910b" id631 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id631 -->
<!-- npu="A3" id632 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id632 -->
<!-- npu="950" id633 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id633 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.has_rsample)

**Support Status**:

<!-- npu="910b" id634 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id634 -->
<!-- npu="A3" id635 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id635 -->
<!-- npu="950" id636 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id636 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.mean)

**Support Status**:

<!-- npu="910b" id637 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id637 -->
<!-- npu="A3" id638 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id638 -->
<!-- npu="950" id639 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id639 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.mode)

**Support Status**:

<!-- npu="910b" id640 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id640 -->
<!-- npu="A3" id641 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id641 -->
<!-- npu="950" id642 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id642 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.support)

**Support Status**:

<!-- npu="910b" id643 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id643 -->
<!-- npu="A3" id644 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id644 -->
<!-- npu="950" id645 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id645 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kumaraswamy.Kumaraswamy.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kumaraswamy.Kumaraswamy.variance)

**Support Status**:

<!-- npu="910b" id646 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id646 -->
<!-- npu="A3" id647 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id647 -->
<!-- npu="950" id648 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id648 -->

</div>

</div>

## LKJCholesky

### <code><i>class</i></code> torch.distributions.lkj_cholesky.LKJCholesky

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lkj_cholesky.LKJCholesky](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky)

**Support Status**:

<!-- npu="910b" id649 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id649 -->
<!-- npu="A3" id650 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id650 -->
<!-- npu="950" id651 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id651 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints)

**Support Status**:

<!-- npu="910b" id652 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id652 -->
<!-- npu="A3" id653 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id653 -->
<!-- npu="950" id654 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id654 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lkj_cholesky.LKJCholesky.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.expand)

**Support Status**:

<!-- npu="910b" id655 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id655 -->
<!-- npu="A3" id656 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id656 -->
<!-- npu="950" id657 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id657 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lkj_cholesky.LKJCholesky.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.log_prob)

**Support Status**:

<!-- npu="910b" id658 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id658 -->
<!-- npu="A3" id659 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id659 -->
<!-- npu="950" id660 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id660 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lkj_cholesky.LKJCholesky.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.sample)

**Support Status**:

<!-- npu="910b" id661 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id661 -->
<!-- npu="A3" id662 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id662 -->
<!-- npu="950" id663 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id663 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lkj_cholesky.LKJCholesky.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lkj_cholesky.LKJCholesky.support)

**Support Status**:

<!-- npu="910b" id664 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id664 -->
<!-- npu="A3" id665 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id665 -->
<!-- npu="950" id666 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id666 -->

</div>

</div>

## Laplace

### <code><i>class</i></code> torch.distributions.laplace.Laplace

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace)

**Support Status**:

<!-- npu="910b" id667 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id667 -->
<!-- npu="A3" id668 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id668 -->
<!-- npu="950" id669 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id669 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.arg_constraints)

**Support Status**:

<!-- npu="910b" id670 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id670 -->
<!-- npu="A3" id671 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id671 -->
<!-- npu="950" id672 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id672 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.cdf)

**Support Status**:

<!-- npu="910b" id673 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id673 -->
<!-- npu="A3" id674 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id674 -->
<!-- npu="950" id675 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id675 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.entropy)

**Support Status**:

<!-- npu="910b" id676 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id676 -->
<!-- npu="A3" id677 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id677 -->
<!-- npu="950" id678 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id678 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.expand)

**Support Status**:

<!-- npu="910b" id679 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id679 -->
<!-- npu="A3" id680 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id680 -->
<!-- npu="950" id681 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id681 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.has_rsample)

**Support Status**:

<!-- npu="910b" id682 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id682 -->
<!-- npu="A3" id683 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id683 -->
<!-- npu="950" id684 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id684 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.icdf)

**Support Status**:

<!-- npu="910b" id685 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id685 -->
<!-- npu="A3" id686 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id686 -->
<!-- npu="950" id687 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id687 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.log_prob)

**Support Status**:

<!-- npu="910b" id688 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id688 -->
<!-- npu="A3" id689 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id689 -->
<!-- npu="950" id690 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id690 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.mean)

**Support Status**:

<!-- npu="910b" id691 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id691 -->
<!-- npu="A3" id692 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id692 -->
<!-- npu="950" id693 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id693 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.mode)

**Support Status**:

<!-- npu="910b" id694 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id694 -->
<!-- npu="A3" id695 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id695 -->
<!-- npu="950" id696 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id696 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.rsample)

**Support Status**:

<!-- npu="910b" id697 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id697 -->
<!-- npu="A3" id698 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id698 -->
<!-- npu="950" id699 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id699 -->

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.stddev](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.stddev)

**Support Status**:

<!-- npu="910b" id700 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id700 -->
<!-- npu="A3" id701 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id701 -->
<!-- npu="950" id702 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id702 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.support)

**Support Status**:

<!-- npu="910b" id703 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id703 -->
<!-- npu="A3" id704 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id704 -->
<!-- npu="950" id705 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id705 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.laplace.Laplace.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.laplace.Laplace.variance)

**Support Status**:

<!-- npu="910b" id706 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id706 -->
<!-- npu="A3" id707 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id707 -->
<!-- npu="950" id708 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id708 -->

</div>

</div>

## LogNormal

### <code><i>class</i></code> torch.distributions.log_normal.LogNormal

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal)

**Support Status**:

<!-- npu="910b" id709 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id709 -->
<!-- npu="A3" id710 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id710 -->
<!-- npu="950" id711 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id711 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.arg_constraints)

**Support Status**:

<!-- npu="910b" id712 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id712 -->
<!-- npu="A3" id713 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id713 -->
<!-- npu="950" id714 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id714 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.entropy)

**Support Status**:

<!-- npu="910b" id715 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id715 -->
<!-- npu="A3" id716 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id716 -->
<!-- npu="950" id717 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id717 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.expand)

**Support Status**:

<!-- npu="910b" id718 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id718 -->
<!-- npu="A3" id719 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id719 -->
<!-- npu="950" id720 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id720 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.has_rsample)

**Support Status**:

<!-- npu="910b" id721 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id721 -->
<!-- npu="A3" id722 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id722 -->
<!-- npu="950" id723 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id723 -->

</div>

> <font size="3">loc()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.loc](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.loc)

**Support Status**:

<!-- npu="910b" id724 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id724 -->
<!-- npu="A3" id725 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id725 -->
<!-- npu="950" id726 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id726 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.mean)

**Support Status**:

<!-- npu="910b" id727 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id727 -->
<!-- npu="A3" id728 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id728 -->
<!-- npu="950" id729 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id729 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.mode)

**Support Status**:

<!-- npu="910b" id730 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id730 -->
<!-- npu="A3" id731 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id731 -->
<!-- npu="950" id732 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id732 -->

</div>

> <font size="3">scale()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.scale](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.scale)

**Support Status**:

<!-- npu="910b" id733 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id733 -->
<!-- npu="A3" id734 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id734 -->
<!-- npu="950" id735 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id735 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.support)

**Support Status**:

<!-- npu="910b" id736 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id736 -->
<!-- npu="A3" id737 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id737 -->
<!-- npu="950" id738 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id738 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.log_normal.LogNormal.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.log_normal.LogNormal.variance)

**Support Status**:

<!-- npu="910b" id739 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id739 -->
<!-- npu="A3" id740 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id740 -->
<!-- npu="950" id741 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id741 -->

</div>

</div>

## LowRankMultivariateNormal

### <code><i>class</i></code> torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal)

**Support Status**:

<!-- npu="910b" id742 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id742 -->
<!-- npu="A3" id743 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id743 -->
<!-- npu="950" id744 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id744 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.arg_constraints)

**Support Status**:

<!-- npu="910b" id745 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id745 -->
<!-- npu="A3" id746 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id746 -->
<!-- npu="950" id747 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id747 -->

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.covariance_matrix](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.covariance_matrix)

**Support Status**:

<!-- npu="910b" id748 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id748 -->
<!-- npu="A3" id749 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id749 -->
<!-- npu="950" id750 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id750 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.entropy)

**Support Status**:

<!-- npu="910b" id751 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id751 -->
<!-- npu="A3" id752 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id752 -->
<!-- npu="950" id753 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id753 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.expand)

**Support Status**:

<!-- npu="910b" id754 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id754 -->
<!-- npu="A3" id755 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id755 -->
<!-- npu="950" id756 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id756 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.has_rsample)

**Support Status**:

<!-- npu="910b" id757 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id757 -->
<!-- npu="A3" id758 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id758 -->
<!-- npu="950" id759 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id759 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.log_prob)

**Support Status**:

<!-- npu="910b" id760 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id760 -->
<!-- npu="A3" id761 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id761 -->
<!-- npu="950" id762 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id762 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mean)

**Support Status**:

<!-- npu="910b" id763 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id763 -->
<!-- npu="A3" id764 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id764 -->
<!-- npu="950" id765 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id765 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mode)

**Support Status**:

<!-- npu="910b" id766 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id766 -->
<!-- npu="A3" id767 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id767 -->
<!-- npu="950" id768 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id768 -->

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.precision_matrix](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.precision_matrix)

**Support Status**:

<!-- npu="910b" id769 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id769 -->
<!-- npu="A3" id770 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id770 -->
<!-- npu="950" id771 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id771 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.rsample)

**Support Status**:

<!-- npu="910b" id772 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id772 -->
<!-- npu="A3" id773 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id773 -->
<!-- npu="950" id774 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id774 -->

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.scale_tril](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.scale_tril)

**Support Status**:

<!-- npu="910b" id775 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id775 -->
<!-- npu="A3" id776 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id776 -->
<!-- npu="950" id777 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id777 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.support)

**Support Status**:

<!-- npu="910b" id778 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id778 -->
<!-- npu="A3" id779 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id779 -->
<!-- npu="950" id780 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id780 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.variance)

**Support Status**:

<!-- npu="910b" id781 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id781 -->
<!-- npu="A3" id782 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id782 -->
<!-- npu="950" id783 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id783 -->

</div>

</div>

## MixtureSameFamily

### <code><i>class</i></code> torch.distributions.mixture_same_family.MixtureSameFamily

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily)

**Support Status**:

<!-- npu="910b" id784 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id784 -->
<!-- npu="A3" id785 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id785 -->
<!-- npu="950" id786 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id786 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.arg_constraints)

**Support Status**:

<!-- npu="910b" id787 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id787 -->
<!-- npu="A3" id788 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id788 -->
<!-- npu="950" id789 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id789 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.cdf)

**Support Status**:

<!-- npu="910b" id790 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id790 -->
<!-- npu="A3" id791 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id791 -->
<!-- npu="950" id792 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id792 -->

</div>

> <font size="3">component_distribution()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.component_distribution](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.component_distribution)

**Support Status**:

<!-- npu="910b" id793 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id793 -->
<!-- npu="A3" id794 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id794 -->
<!-- npu="950" id795 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id795 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.expand)

**Support Status**:

<!-- npu="910b" id796 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id796 -->
<!-- npu="A3" id797 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id797 -->
<!-- npu="950" id798 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id798 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.has_rsample)

**Support Status**:

<!-- npu="910b" id799 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id799 -->
<!-- npu="A3" id800 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id800 -->
<!-- npu="950" id801 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id801 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.log_prob)

**Support Status**:

<!-- npu="910b" id802 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id802 -->
<!-- npu="A3" id803 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id803 -->
<!-- npu="950" id804 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id804 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.mean)

**Support Status**:

<!-- npu="910b" id805 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id805 -->
<!-- npu="A3" id806 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id806 -->
<!-- npu="950" id807 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id807 -->

</div>

> <font size="3">mixture_distribution()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.mixture_distribution](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.mixture_distribution)

**Support Status**:

<!-- npu="910b" id808 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id808 -->
<!-- npu="A3" id809 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id809 -->
<!-- npu="950" id810 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id810 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.sample)

**Support Status**:

<!-- npu="910b" id811 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id811 -->
<!-- npu="A3" id812 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id812 -->
<!-- npu="950" id813 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id813 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.support)

**Support Status**:

<!-- npu="910b" id814 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id814 -->
<!-- npu="A3" id815 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id815 -->
<!-- npu="950" id816 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id816 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.mixture_same_family.MixtureSameFamily.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.mixture_same_family.MixtureSameFamily.variance)

**Support Status**:

<!-- npu="910b" id817 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id817 -->
<!-- npu="A3" id818 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id818 -->
<!-- npu="950" id819 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id819 -->

</div>

</div>

## Multinomial

### <code><i>class</i></code> torch.distributions.multinomial.Multinomial

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial)

**Support Status**:

<!-- npu="910b" id820 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id820 -->
<!-- npu="A3" id821 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id821 -->
<!-- npu="950" id822 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id822 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.arg_constraints)

**Support Status**:

<!-- npu="910b" id823 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id823 -->
<!-- npu="A3" id824 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id824 -->
<!-- npu="950" id825 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id825 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.entropy)

**Support Status**:

<!-- npu="910b" id826 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id826 -->
<!-- npu="A3" id827 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id827 -->
<!-- npu="950" id828 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id828 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.expand)

**Support Status**:

<!-- npu="910b" id829 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id829 -->
<!-- npu="A3" id830 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id830 -->
<!-- npu="950" id831 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id831 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.log_prob)

**Support Status**:

<!-- npu="910b" id832 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id832 -->
<!-- npu="A3" id833 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id833 -->
<!-- npu="950" id834 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id834 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.logits)

**Support Status**:

<!-- npu="910b" id835 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id835 -->
<!-- npu="A3" id836 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id836 -->
<!-- npu="950" id837 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id837 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.mean)

**Support Status**:

<!-- npu="910b" id838 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id838 -->
<!-- npu="A3" id839 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id839 -->
<!-- npu="950" id840 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id840 -->

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.param_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.param_shape)

**Support Status**:

<!-- npu="910b" id841 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id841 -->
<!-- npu="A3" id842 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id842 -->
<!-- npu="950" id843 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id843 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.probs)

**Support Status**:

<!-- npu="910b" id844 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id844 -->
<!-- npu="A3" id845 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id845 -->
<!-- npu="950" id846 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id846 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.sample)

**Support Status**:

<!-- npu="910b" id847 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id847 -->
<!-- npu="A3" id848 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id848 -->
<!-- npu="950" id849 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id849 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.support)

**Support Status**:

<!-- npu="910b" id850 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id850 -->
<!-- npu="A3" id851 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id851 -->
<!-- npu="950" id852 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id852 -->

</div>

> <font size="3">total_count()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.total_count](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.total_count)

**Support Status**:

<!-- npu="910b" id853 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id853 -->
<!-- npu="A3" id854 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id854 -->
<!-- npu="950" id855 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id855 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multinomial.Multinomial.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multinomial.Multinomial.variance)

**Support Status**:

<!-- npu="910b" id856 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id856 -->
<!-- npu="A3" id857 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id857 -->
<!-- npu="950" id858 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id858 -->

</div>

</div>

## MultivariateNormal

### <code><i>class</i></code> torch.distributions.multivariate_normal.MultivariateNormal

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal)

**Support Status**:

<!-- npu="910b" id859 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id859 -->
<!-- npu="A3" id860 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id860 -->
<!-- npu="950" id861 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id861 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.arg_constraints)

**Support Status**:

<!-- npu="910b" id862 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id862 -->
<!-- npu="A3" id863 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id863 -->
<!-- npu="950" id864 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id864 -->

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix)

**Support Status**:

<!-- npu="910b" id865 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id865 -->
<!-- npu="A3" id866 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id866 -->
<!-- npu="950" id867 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id867 -->

**Constraints & Notes**: `dim` must be less than or equal to 8192

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.entropy)

**Support Status**:

<!-- npu="910b" id868 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id868 -->
<!-- npu="A3" id869 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id869 -->
<!-- npu="950" id870 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id870 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.expand)

**Support Status**:

<!-- npu="910b" id871 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id871 -->
<!-- npu="A3" id872 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id872 -->
<!-- npu="950" id873 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id873 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.has_rsample)

**Support Status**:

<!-- npu="910b" id874 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id874 -->
<!-- npu="A3" id875 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id875 -->
<!-- npu="950" id876 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id876 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.log_prob)

**Support Status**:

<!-- npu="910b" id877 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id877 -->
<!-- npu="A3" id878 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id878 -->
<!-- npu="950" id879 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id879 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.mean)

**Support Status**:

<!-- npu="910b" id880 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id880 -->
<!-- npu="A3" id881 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id881 -->
<!-- npu="950" id882 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id882 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.mode)

**Support Status**:

<!-- npu="910b" id883 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id883 -->
<!-- npu="A3" id884 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id884 -->
<!-- npu="950" id885 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id885 -->

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix)

**Support Status**:

<!-- npu="910b" id886 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id886 -->
<!-- npu="A3" id887 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id887 -->
<!-- npu="950" id888 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id888 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.rsample)

**Support Status**:

<!-- npu="910b" id889 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id889 -->
<!-- npu="A3" id890 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id890 -->
<!-- npu="950" id891 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id891 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.scale_tril](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.scale_tril)

**Support Status**:

<!-- npu="910b" id892 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id892 -->
<!-- npu="A3" id893 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id893 -->
<!-- npu="950" id894 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id894 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.support)

**Support Status**:

<!-- npu="910b" id895 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id895 -->
<!-- npu="A3" id896 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id896 -->
<!-- npu="950" id897 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id897 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.multivariate_normal.MultivariateNormal.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.multivariate_normal.MultivariateNormal.variance)

**Support Status**:

<!-- npu="910b" id898 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id898 -->
<!-- npu="A3" id899 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id899 -->
<!-- npu="950" id900 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id900 -->

</div>

</div>

## NegativeBinomial

### <code><i>class</i></code> torch.distributions.negative_binomial.NegativeBinomial

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial)

**Support Status**:

<!-- npu="910b" id901 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id901 -->
<!-- npu="A3" id902 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id902 -->
<!-- npu="950" id903 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id903 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.arg_constraints)

**Support Status**:

<!-- npu="910b" id904 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id904 -->
<!-- npu="A3" id905 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id905 -->
<!-- npu="950" id906 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id906 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.expand)

**Support Status**:

<!-- npu="910b" id907 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id907 -->
<!-- npu="A3" id908 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id908 -->
<!-- npu="950" id909 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id909 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.log_prob)

**Support Status**:

<!-- npu="910b" id910 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id910 -->
<!-- npu="A3" id911 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id911 -->
<!-- npu="950" id912 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id912 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.logits)

**Support Status**:

<!-- npu="910b" id913 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id913 -->
<!-- npu="A3" id914 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id914 -->
<!-- npu="950" id915 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id915 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.mean)

**Support Status**:

<!-- npu="910b" id916 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id916 -->
<!-- npu="A3" id917 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id917 -->
<!-- npu="950" id918 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id918 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.mode)

**Support Status**:

<!-- npu="910b" id919 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id919 -->
<!-- npu="A3" id920 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id920 -->
<!-- npu="950" id921 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id921 -->

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.param_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.param_shape)

**Support Status**:

<!-- npu="910b" id922 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id922 -->
<!-- npu="A3" id923 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id923 -->
<!-- npu="950" id924 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id924 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.probs)

**Support Status**:

<!-- npu="910b" id925 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id925 -->
<!-- npu="A3" id926 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id926 -->
<!-- npu="950" id927 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id927 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.sample)

**Support Status**:

<!-- npu="910b" id928 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id928 -->
<!-- npu="A3" id929 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id929 -->
<!-- npu="950" id930 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id930 -->

**Constraints & Notes**: May fall back to CPU execution

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.support)

**Support Status**:

<!-- npu="910b" id931 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id931 -->
<!-- npu="A3" id932 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id932 -->
<!-- npu="950" id933 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id933 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.negative_binomial.NegativeBinomial.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.negative_binomial.NegativeBinomial.variance)

**Support Status**:

<!-- npu="910b" id934 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id934 -->
<!-- npu="A3" id935 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id935 -->
<!-- npu="950" id936 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id936 -->

</div>

</div>

## Normal

### <code><i>class</i></code> torch.distributions.normal.Normal

<div style="margin-left: 2em">

**Constraints & Notes**:

<!-- npu="950" id1434 -->
- On <term>Ascend 950DT products</term>, the random number generation behavior is consistent with the GPU implementation, and the generated results differ from those of <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>.
<!-- end id1434 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.arg_constraints)

**Support Status**:

<!-- npu="910b" id937 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id937 -->
<!-- npu="A3" id938 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id938 -->
<!-- npu="950" id939 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id939 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.cdf)

**Support Status**:

<!-- npu="910b" id940 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id940 -->
<!-- npu="A3" id941 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id941 -->
<!-- npu="950" id942 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id942 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.entropy)

**Support Status**:

<!-- npu="910b" id943 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id943 -->
<!-- npu="A3" id944 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id944 -->
<!-- npu="950" id945 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id945 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.expand)

**Support Status**:

<!-- npu="910b" id946 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id946 -->
<!-- npu="A3" id947 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id947 -->
<!-- npu="950" id948 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id948 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.has_rsample)

**Support Status**:

<!-- npu="910b" id949 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id949 -->
<!-- npu="A3" id950 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id950 -->
<!-- npu="950" id951 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id951 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.icdf)

**Support Status**:

<!-- npu="910b" id952 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id952 -->
<!-- npu="A3" id953 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id953 -->
<!-- npu="950" id954 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id954 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.log_prob)

**Support Status**:

<!-- npu="910b" id955 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id955 -->
<!-- npu="A3" id956 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id956 -->
<!-- npu="950" id957 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id957 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.mean)

**Support Status**:

<!-- npu="910b" id958 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id958 -->
<!-- npu="A3" id959 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id959 -->
<!-- npu="950" id960 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id960 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.mode)

**Support Status**:

<!-- npu="910b" id961 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id961 -->
<!-- npu="A3" id962 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id962 -->
<!-- npu="950" id963 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id963 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.rsample)

**Support Status**:

<!-- npu="910b" id964 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id964 -->
<!-- npu="A3" id965 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id965 -->
<!-- npu="950" id966 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id966 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.sample)

**Support Status**:

<!-- npu="910b" id967 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id967 -->
<!-- npu="A3" id968 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id968 -->
<!-- npu="950" id969 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id969 -->

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.stddev](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.stddev)

**Support Status**:

<!-- npu="910b" id970 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id970 -->
<!-- npu="A3" id971 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id971 -->
<!-- npu="950" id972 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id972 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.support)

**Support Status**:

<!-- npu="910b" id973 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id973 -->
<!-- npu="A3" id974 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id974 -->
<!-- npu="950" id975 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id975 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.normal.Normal.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.normal.Normal.variance)

**Support Status**:

<!-- npu="910b" id976 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id976 -->
<!-- npu="A3" id977 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id977 -->
<!-- npu="950" id978 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id978 -->

</div>

</div>

## OneHotCategorical

### <code><i>class</i></code> torch.distributions.one_hot_categorical.OneHotCategorical

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical)

**Support Status**:

<!-- npu="910b" id979 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id979 -->
<!-- npu="A3" id980 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id980 -->
<!-- npu="950" id981 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id981 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.arg_constraints)

**Support Status**:

<!-- npu="910b" id982 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id982 -->
<!-- npu="A3" id983 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id983 -->
<!-- npu="950" id984 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id984 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.entropy)

**Support Status**:

<!-- npu="910b" id985 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id985 -->
<!-- npu="A3" id986 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id986 -->
<!-- npu="950" id987 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id987 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.expand)

**Support Status**:

<!-- npu="910b" id988 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id988 -->
<!-- npu="A3" id989 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id989 -->
<!-- npu="950" id990 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id990 -->

</div>

> <font size="3">has_enumerate_support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.has_enumerate_support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.has_enumerate_support)

**Support Status**:

<!-- npu="910b" id991 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id991 -->
<!-- npu="A3" id992 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id992 -->
<!-- npu="950" id993 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id993 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.log_prob)

**Support Status**:

<!-- npu="910b" id994 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id994 -->
<!-- npu="A3" id995 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id995 -->
<!-- npu="950" id996 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id996 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.logits)

**Support Status**:

<!-- npu="910b" id997 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id997 -->
<!-- npu="A3" id998 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id998 -->
<!-- npu="950" id999 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id999 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.mean)

**Support Status**:

<!-- npu="910b" id1000 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1000 -->
<!-- npu="A3" id1001 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1001 -->
<!-- npu="950" id1002 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1002 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.mode)

**Support Status**:

<!-- npu="910b" id1003 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1003 -->
<!-- npu="A3" id1004 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1004 -->
<!-- npu="950" id1005 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1005 -->

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.param_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.param_shape)

**Support Status**:

<!-- npu="910b" id1006 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1006 -->
<!-- npu="A3" id1007 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1007 -->
<!-- npu="950" id1008 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1008 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.probs)

**Support Status**:

<!-- npu="910b" id1009 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1009 -->
<!-- npu="A3" id1010 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1010 -->
<!-- npu="950" id1011 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1011 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.sample)

**Support Status**:

<!-- npu="910b" id1012 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1012 -->
<!-- npu="A3" id1013 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1013 -->
<!-- npu="950" id1014 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1014 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.support)

**Support Status**:

<!-- npu="910b" id1015 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1015 -->
<!-- npu="A3" id1016 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1016 -->
<!-- npu="950" id1017 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1017 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.one_hot_categorical.OneHotCategorical.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.one_hot_categorical.OneHotCategorical.variance)

**Support Status**:

<!-- npu="910b" id1018 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1018 -->
<!-- npu="A3" id1019 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1019 -->
<!-- npu="950" id1020 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1020 -->

</div>

</div>

## Pareto

### <code><i>class</i></code> torch.distributions.pareto.Pareto

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.pareto.Pareto](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.pareto.Pareto)

**Support Status**:

<!-- npu="910b" id1021 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1021 -->
<!-- npu="A3" id1022 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1022 -->
<!-- npu="950" id1023 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1023 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.pareto.Pareto.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.pareto.Pareto.arg_constraints)

**Support Status**:

<!-- npu="910b" id1024 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1024 -->
<!-- npu="A3" id1025 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1025 -->
<!-- npu="950" id1026 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1026 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.pareto.Pareto.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.pareto.Pareto.entropy)

**Support Status**:

<!-- npu="910b" id1027 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1027 -->
<!-- npu="A3" id1028 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1028 -->
<!-- npu="950" id1029 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1029 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.pareto.Pareto.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.pareto.Pareto.expand)

**Support Status**:

<!-- npu="910b" id1030 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1030 -->
<!-- npu="A3" id1031 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1031 -->
<!-- npu="950" id1032 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1032 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.pareto.Pareto.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.pareto.Pareto.mean)

**Support Status**:

<!-- npu="910b" id1033 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1033 -->
<!-- npu="A3" id1034 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1034 -->
<!-- npu="950" id1035 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1035 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.pareto.Pareto.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.pareto.Pareto.mode)

**Support Status**:

<!-- npu="910b" id1036 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1036 -->
<!-- npu="A3" id1037 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1037 -->
<!-- npu="950" id1038 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1038 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.pareto.Pareto.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.pareto.Pareto.support)

**Support Status**:

<!-- npu="910b" id1039 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1039 -->
<!-- npu="A3" id1040 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1040 -->
<!-- npu="950" id1041 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1041 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.pareto.Pareto.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.pareto.Pareto.variance)

**Support Status**:

<!-- npu="910b" id1042 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1042 -->
<!-- npu="A3" id1043 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1043 -->
<!-- npu="950" id1044 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1044 -->

</div>

</div>

## Poisson

### <code><i>class</i></code> torch.distributions.poisson.Poisson

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson)

**Support Status**:

<!-- npu="910b" id1045 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1045 -->
<!-- npu="A3" id1046 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1046 -->
<!-- npu="950" id1047 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1047 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson.arg_constraints)

**Support Status**:

<!-- npu="910b" id1048 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1048 -->
<!-- npu="A3" id1049 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1049 -->
<!-- npu="950" id1050 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1050 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson.expand)

**Support Status**:

<!-- npu="910b" id1051 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1051 -->
<!-- npu="A3" id1052 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1052 -->
<!-- npu="950" id1053 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1053 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson.log_prob)

**Support Status**:

<!-- npu="910b" id1054 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1054 -->
<!-- npu="A3" id1055 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1055 -->
<!-- npu="950" id1056 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1056 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson.mean)

**Support Status**:

<!-- npu="910b" id1057 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1057 -->
<!-- npu="A3" id1058 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1058 -->
<!-- npu="950" id1059 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1059 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson.mode)

**Support Status**:

<!-- npu="910b" id1060 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1060 -->
<!-- npu="A3" id1061 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1061 -->
<!-- npu="950" id1062 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1062 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson.sample)

**Support Status**:

<!-- npu="910b" id1063 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1063 -->
<!-- npu="A3" id1064 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1064 -->
<!-- npu="950" id1065 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1065 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson.support)

**Support Status**:

<!-- npu="910b" id1066 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1066 -->
<!-- npu="A3" id1067 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1067 -->
<!-- npu="950" id1068 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1068 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.poisson.Poisson.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.poisson.Poisson.variance)

**Support Status**:

<!-- npu="910b" id1069 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1069 -->
<!-- npu="A3" id1070 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1070 -->
<!-- npu="950" id1071 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1071 -->

</div>

</div>

## RelaxedBernoulli

### <code><i>class</i></code> torch.distributions.relaxed_bernoulli.RelaxedBernoulli

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.RelaxedBernoulli](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli)

**Support Status**:

<!-- npu="910b" id1072 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1072 -->
<!-- npu="A3" id1073 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1073 -->
<!-- npu="950" id1074 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1074 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.RelaxedBernoulli.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.arg_constraints)

**Support Status**:

<!-- npu="910b" id1075 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1075 -->
<!-- npu="A3" id1076 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1076 -->
<!-- npu="950" id1077 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1077 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand)

**Support Status**:

<!-- npu="910b" id1078 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1078 -->
<!-- npu="A3" id1079 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1079 -->
<!-- npu="950" id1080 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1080 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.RelaxedBernoulli.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.has_rsample)

**Support Status**:

<!-- npu="910b" id1081 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1081 -->
<!-- npu="A3" id1082 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1082 -->
<!-- npu="950" id1083 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1083 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits)

**Support Status**:

<!-- npu="910b" id1084 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1084 -->
<!-- npu="A3" id1085 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1085 -->
<!-- npu="950" id1086 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1086 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs)

**Support Status**:

<!-- npu="910b" id1087 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1087 -->
<!-- npu="A3" id1088 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1088 -->
<!-- npu="950" id1089 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1089 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.RelaxedBernoulli.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.support)

**Support Status**:

<!-- npu="910b" id1090 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1090 -->
<!-- npu="A3" id1091 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1091 -->
<!-- npu="950" id1092 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1092 -->

</div>

> <font size="3">temperature()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature)

**Support Status**:

<!-- npu="910b" id1093 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1093 -->
<!-- npu="A3" id1094 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1094 -->
<!-- npu="950" id1095 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1095 -->

</div>

</div>

## LogitRelaxedBernoulli

### <code><i>class</i></code> torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli)

**Support Status**:

<!-- npu="910b" id1096 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1096 -->
<!-- npu="A3" id1097 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1097 -->
<!-- npu="950" id1098 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1098 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.arg_constraints)

**Support Status**:

<!-- npu="910b" id1099 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1099 -->
<!-- npu="A3" id1100 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1100 -->
<!-- npu="950" id1101 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1101 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.expand)

**Support Status**:

<!-- npu="910b" id1102 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1102 -->
<!-- npu="A3" id1103 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1103 -->
<!-- npu="950" id1104 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1104 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.log_prob)

**Support Status**:

<!-- npu="910b" id1105 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1105 -->
<!-- npu="A3" id1106 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1106 -->
<!-- npu="950" id1107 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1107 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.logits)

**Support Status**:

<!-- npu="910b" id1108 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1108 -->
<!-- npu="A3" id1109 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1109 -->
<!-- npu="950" id1110 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1110 -->

</div>

> <font size="3">param_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.param_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.param_shape)

**Support Status**:

<!-- npu="910b" id1111 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1111 -->
<!-- npu="A3" id1112 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1112 -->
<!-- npu="950" id1113 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1113 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.probs)

**Support Status**:

<!-- npu="910b" id1114 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1114 -->
<!-- npu="A3" id1115 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1115 -->
<!-- npu="950" id1116 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1116 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.rsample)

**Support Status**:

<!-- npu="910b" id1117 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1117 -->
<!-- npu="A3" id1118 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1118 -->
<!-- npu="950" id1119 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1119 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.support)

**Support Status**:

<!-- npu="910b" id1120 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1120 -->
<!-- npu="A3" id1121 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1121 -->
<!-- npu="950" id1122 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1122 -->

</div>

</div>

## RelaxedOneHotCategorical

### <code><i>class</i></code> torch.distributions.relaxed_categorical.RelaxedOneHotCategorical

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical)

**Support Status**:

<!-- npu="910b" id1123 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1123 -->
<!-- npu="A3" id1124 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1124 -->
<!-- npu="950" id1125 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1125 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.arg_constraints)

**Support Status**:

<!-- npu="910b" id1126 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1126 -->
<!-- npu="A3" id1127 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1127 -->
<!-- npu="950" id1128 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1128 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.expand)

**Support Status**:

<!-- npu="910b" id1129 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1129 -->
<!-- npu="A3" id1130 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1130 -->
<!-- npu="950" id1131 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1131 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.has_rsample)

**Support Status**:

<!-- npu="910b" id1132 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1132 -->
<!-- npu="A3" id1133 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1133 -->
<!-- npu="950" id1134 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1134 -->

</div>

> <font size="3">logits()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits)

**Support Status**:

<!-- npu="910b" id1135 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1135 -->
<!-- npu="A3" id1136 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1136 -->
<!-- npu="950" id1137 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1137 -->

</div>

> <font size="3">probs()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs)

**Support Status**:

<!-- npu="910b" id1138 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1138 -->
<!-- npu="A3" id1139 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1139 -->
<!-- npu="950" id1140 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1140 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.support)

**Support Status**:

<!-- npu="910b" id1141 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1141 -->
<!-- npu="A3" id1142 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1142 -->
<!-- npu="950" id1143 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1143 -->

</div>

> <font size="3">temperature()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature)

**Support Status**:

<!-- npu="910b" id1144 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1144 -->
<!-- npu="A3" id1145 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1145 -->
<!-- npu="950" id1146 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1146 -->

</div>

</div>

## StudentT

### <code><i>class</i></code> torch.distributions.studentT.StudentT

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT)

**Support Status**:

<!-- npu="910b" id1147 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1147 -->
<!-- npu="A3" id1148 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1148 -->
<!-- npu="950" id1149 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1149 -->

**Constraints & Notes**: May fall back to CPU execution

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.arg_constraints)

**Support Status**:

<!-- npu="910b" id1150 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1150 -->
<!-- npu="A3" id1151 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1151 -->
<!-- npu="950" id1152 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1152 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.entropy)

**Support Status**:

<!-- npu="910b" id1153 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1153 -->
<!-- npu="A3" id1154 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1154 -->
<!-- npu="950" id1155 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1155 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.expand)

**Support Status**:

<!-- npu="910b" id1156 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1156 -->
<!-- npu="A3" id1157 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1157 -->
<!-- npu="950" id1158 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1158 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.has_rsample)

**Support Status**:

<!-- npu="910b" id1159 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1159 -->
<!-- npu="A3" id1160 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1160 -->
<!-- npu="950" id1161 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1161 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.log_prob)

**Support Status**:

<!-- npu="910b" id1162 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1162 -->
<!-- npu="A3" id1163 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1163 -->
<!-- npu="950" id1164 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1164 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.mean)

**Support Status**:

<!-- npu="910b" id1165 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1165 -->
<!-- npu="A3" id1166 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1166 -->
<!-- npu="950" id1167 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1167 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.mode)

**Support Status**:

<!-- npu="910b" id1168 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1168 -->
<!-- npu="A3" id1169 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1169 -->
<!-- npu="950" id1170 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1170 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.rsample)

**Support Status**:

<!-- npu="910b" id1171 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1171 -->
<!-- npu="A3" id1172 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1172 -->
<!-- npu="950" id1173 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1173 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.support)

**Support Status**:

<!-- npu="910b" id1174 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1174 -->
<!-- npu="A3" id1175 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1175 -->
<!-- npu="950" id1176 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1176 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.studentT.StudentT.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.studentT.StudentT.variance)

**Support Status**:

<!-- npu="910b" id1177 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1177 -->
<!-- npu="A3" id1178 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1178 -->
<!-- npu="950" id1179 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1179 -->

</div>

</div>

## TransformedDistribution

### <code><i>class</i></code> torch.distributions.transformed_distribution.TransformedDistribution

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution)

**Support Status**:

<!-- npu="910b" id1180 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1180 -->
<!-- npu="A3" id1181 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1181 -->
<!-- npu="950" id1182 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1182 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.arg_constraints)

**Support Status**:

<!-- npu="910b" id1183 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1183 -->
<!-- npu="A3" id1184 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1184 -->
<!-- npu="950" id1185 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1185 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.cdf)

**Support Status**:

<!-- npu="910b" id1186 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1186 -->
<!-- npu="A3" id1187 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1187 -->
<!-- npu="950" id1188 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1188 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.expand)

**Support Status**:

<!-- npu="910b" id1189 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1189 -->
<!-- npu="A3" id1190 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1190 -->
<!-- npu="950" id1191 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1191 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.has_rsample)

**Support Status**:

<!-- npu="910b" id1192 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1192 -->
<!-- npu="A3" id1193 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1193 -->
<!-- npu="950" id1194 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1194 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.icdf)

**Support Status**:

<!-- npu="910b" id1195 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1195 -->
<!-- npu="A3" id1196 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1196 -->
<!-- npu="950" id1197 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1197 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.log_prob)

**Support Status**:

<!-- npu="910b" id1198 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1198 -->
<!-- npu="A3" id1199 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1199 -->
<!-- npu="950" id1200 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1200 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.rsample)

**Support Status**:

<!-- npu="910b" id1201 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1201 -->
<!-- npu="A3" id1202 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1202 -->
<!-- npu="950" id1203 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1203 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.sample)

**Support Status**:

<!-- npu="910b" id1204 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1204 -->
<!-- npu="A3" id1205 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1205 -->
<!-- npu="950" id1206 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1206 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transformed_distribution.TransformedDistribution.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution.support)

**Support Status**:

<!-- npu="910b" id1207 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1207 -->
<!-- npu="A3" id1208 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1208 -->
<!-- npu="950" id1209 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1209 -->

</div>

</div>

## Uniform

### <code><i>class</i></code> torch.distributions.uniform.Uniform

<div style="margin-left: 2em">

**Constraints & Notes**:

<!-- npu="950" id1435 -->
- On <term>Ascend 950DT products</term>, the random number generation behavior is consistent with the GPU implementation, and the generated results differ from those of <term>Atlas A2 training products</term> and <term>Atlas A3 training products</term>.
<!-- end id1435 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.arg_constraints)

**Support Status**:

<!-- npu="910b" id1210 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1210 -->
<!-- npu="A3" id1211 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1211 -->
<!-- npu="950" id1212 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1212 -->

</div>

> <font size="3">cdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.cdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.cdf)

**Support Status**:

<!-- npu="910b" id1213 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1213 -->
<!-- npu="A3" id1214 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1214 -->
<!-- npu="950" id1215 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1215 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.entropy)

**Support Status**:

<!-- npu="910b" id1216 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1216 -->
<!-- npu="A3" id1217 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1217 -->
<!-- npu="950" id1218 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1218 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.expand)

**Support Status**:

<!-- npu="910b" id1219 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1219 -->
<!-- npu="A3" id1220 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1220 -->
<!-- npu="950" id1221 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1221 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.has_rsample)

**Support Status**:

<!-- npu="910b" id1222 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1222 -->
<!-- npu="A3" id1223 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1223 -->
<!-- npu="950" id1224 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1224 -->

</div>

> <font size="3">icdf()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.icdf](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.icdf)

**Support Status**:

<!-- npu="910b" id1225 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1225 -->
<!-- npu="A3" id1226 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1226 -->
<!-- npu="950" id1227 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1227 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.log_prob)

**Support Status**:

<!-- npu="910b" id1228 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1228 -->
<!-- npu="A3" id1229 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1229 -->
<!-- npu="950" id1230 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1230 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.mean)

**Support Status**:

<!-- npu="910b" id1231 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1231 -->
<!-- npu="A3" id1232 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1232 -->
<!-- npu="950" id1233 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1233 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.mode)

**Support Status**:

<!-- npu="910b" id1234 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1234 -->
<!-- npu="A3" id1235 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1235 -->
<!-- npu="950" id1236 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1236 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.rsample)

**Support Status**:

<!-- npu="910b" id1237 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1237 -->
<!-- npu="A3" id1238 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1238 -->
<!-- npu="950" id1239 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1239 -->

</div>

> <font size="3">stddev()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.stddev](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.stddev)

**Support Status**:

<!-- npu="910b" id1240 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1240 -->
<!-- npu="A3" id1241 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1241 -->
<!-- npu="950" id1242 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1242 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.support)

**Support Status**:

<!-- npu="910b" id1243 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1243 -->
<!-- npu="A3" id1244 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1244 -->
<!-- npu="950" id1245 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1245 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.uniform.Uniform.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.uniform.Uniform.variance)

**Support Status**:

<!-- npu="910b" id1246 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1246 -->
<!-- npu="A3" id1247 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1247 -->
<!-- npu="950" id1248 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1248 -->

</div>

</div>

## VonMises

### <code><i>class</i></code> torch.distributions.von_mises.VonMises

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises)

**Support Status**:

<!-- npu="910b" id1249 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1249 -->
<!-- npu="A3" id1250 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1250 -->
<!-- npu="950" id1251 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1251 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.arg_constraints)

**Support Status**:

<!-- npu="910b" id1252 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1252 -->
<!-- npu="A3" id1253 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1253 -->
<!-- npu="950" id1254 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1254 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.expand)

**Support Status**:

<!-- npu="910b" id1255 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1255 -->
<!-- npu="A3" id1256 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1256 -->
<!-- npu="950" id1257 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1257 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.has_rsample)

**Support Status**:

<!-- npu="910b" id1258 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1258 -->
<!-- npu="A3" id1259 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1259 -->
<!-- npu="950" id1260 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1260 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.log_prob)

**Support Status**:

<!-- npu="910b" id1261 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1261 -->
<!-- npu="A3" id1262 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1262 -->
<!-- npu="950" id1263 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1263 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.mean)

**Support Status**:

<!-- npu="910b" id1264 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1264 -->
<!-- npu="A3" id1265 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1265 -->
<!-- npu="950" id1266 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1266 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.mode)

**Support Status**:

<!-- npu="910b" id1267 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1267 -->
<!-- npu="A3" id1268 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1268 -->
<!-- npu="950" id1269 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1269 -->

</div>

> <font size="3">sample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.sample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.sample)

**Support Status**:

<!-- npu="910b" id1270 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1270 -->
<!-- npu="A3" id1271 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1271 -->
<!-- npu="950" id1272 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1272 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.support)

**Support Status**:

<!-- npu="910b" id1273 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1273 -->
<!-- npu="A3" id1274 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1274 -->
<!-- npu="950" id1275 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1275 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.von_mises.VonMises.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.von_mises.VonMises.variance)

**Support Status**:

<!-- npu="910b" id1276 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1276 -->
<!-- npu="A3" id1277 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1277 -->
<!-- npu="950" id1278 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1278 -->

</div>

</div>

## Weibull

### <code><i>class</i></code> torch.distributions.weibull.Weibull

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.weibull.Weibull](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.weibull.Weibull)

**Support Status**:

<!-- npu="910b" id1279 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1279 -->
<!-- npu="A3" id1280 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1280 -->
<!-- npu="950" id1281 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1281 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.weibull.Weibull.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.weibull.Weibull.arg_constraints)

**Support Status**:

<!-- npu="910b" id1282 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1282 -->
<!-- npu="A3" id1283 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1283 -->
<!-- npu="950" id1284 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1284 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.weibull.Weibull.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.weibull.Weibull.entropy)

**Support Status**:

<!-- npu="910b" id1285 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1285 -->
<!-- npu="A3" id1286 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1286 -->
<!-- npu="950" id1287 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1287 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.weibull.Weibull.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.weibull.Weibull.expand)

**Support Status**:

<!-- npu="910b" id1288 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1288 -->
<!-- npu="A3" id1289 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1289 -->
<!-- npu="950" id1290 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1290 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.weibull.Weibull.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.weibull.Weibull.mean)

**Support Status**:

<!-- npu="910b" id1291 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1291 -->
<!-- npu="A3" id1292 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1292 -->
<!-- npu="950" id1293 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1293 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.weibull.Weibull.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.weibull.Weibull.mode)

**Support Status**:

<!-- npu="910b" id1294 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1294 -->
<!-- npu="A3" id1295 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1295 -->
<!-- npu="950" id1296 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1296 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.weibull.Weibull.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.weibull.Weibull.support)

**Support Status**:

<!-- npu="910b" id1297 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1297 -->
<!-- npu="A3" id1298 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1298 -->
<!-- npu="950" id1299 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1299 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.weibull.Weibull.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.weibull.Weibull.variance)

**Support Status**:

<!-- npu="910b" id1300 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1300 -->
<!-- npu="A3" id1301 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1301 -->
<!-- npu="950" id1302 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1302 -->

</div>

</div>

## Wishart

### <code><i>class</i></code> torch.distributions.wishart.Wishart

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart)

**Support Status**:

<!-- npu="910b" id1303 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1303 -->
<!-- npu="A3" id1304 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1304 -->
<!-- npu="950" id1305 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1305 -->

> <font size="3">arg_constraints()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.arg_constraints](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.arg_constraints)

**Support Status**:

<!-- npu="910b" id1306 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1306 -->
<!-- npu="A3" id1307 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1307 -->
<!-- npu="950" id1308 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1308 -->

</div>

> <font size="3">covariance_matrix()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.covariance_matrix](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.covariance_matrix)

**Support Status**:

<!-- npu="910b" id1309 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1309 -->
<!-- npu="A3" id1310 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1310 -->
<!-- npu="950" id1311 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1311 -->

</div>

> <font size="3">entropy()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.entropy](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.entropy)

**Support Status**:

<!-- npu="910b" id1312 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1312 -->
<!-- npu="A3" id1313 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1313 -->
<!-- npu="950" id1314 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1314 -->

</div>

> <font size="3">expand()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.expand](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.expand)

**Support Status**:

<!-- npu="910b" id1315 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1315 -->
<!-- npu="A3" id1316 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1316 -->
<!-- npu="950" id1317 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1317 -->

</div>

> <font size="3">has_rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.has_rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.has_rsample)

**Support Status**:

<!-- npu="910b" id1318 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1318 -->
<!-- npu="A3" id1319 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1319 -->
<!-- npu="950" id1320 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1320 -->

</div>

> <font size="3">log_prob()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.log_prob](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.log_prob)

**Support Status**:

<!-- npu="910b" id1321 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1321 -->
<!-- npu="A3" id1322 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1322 -->
<!-- npu="950" id1323 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1323 -->

</div>

> <font size="3">mean()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.mean](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.mean)

**Support Status**:

<!-- npu="910b" id1324 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1324 -->
<!-- npu="A3" id1325 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1325 -->
<!-- npu="950" id1326 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1326 -->

</div>

> <font size="3">mode()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.mode](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.mode)

**Support Status**:

<!-- npu="910b" id1327 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1327 -->
<!-- npu="A3" id1328 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1328 -->
<!-- npu="950" id1329 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1329 -->

</div>

> <font size="3">precision_matrix()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.precision_matrix](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.precision_matrix)

**Support Status**:

<!-- npu="910b" id1330 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1330 -->
<!-- npu="A3" id1331 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1331 -->
<!-- npu="950" id1332 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1332 -->

</div>

> <font size="3">rsample()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.rsample](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.rsample)

**Support Status**:

<!-- npu="910b" id1333 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1333 -->
<!-- npu="A3" id1334 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1334 -->
<!-- npu="950" id1335 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1335 -->

</div>

> <font size="3">scale_tril()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.scale_tril](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.scale_tril)

**Support Status**:

<!-- npu="910b" id1336 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1336 -->
<!-- npu="A3" id1337 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1337 -->
<!-- npu="950" id1338 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1338 -->

</div>

> <font size="3">support()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.support](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.support)

**Support Status**:

<!-- npu="910b" id1339 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1339 -->
<!-- npu="A3" id1340 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1340 -->
<!-- npu="950" id1341 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1341 -->

</div>

> <font size="3">variance()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.wishart.Wishart.variance](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.wishart.Wishart.variance)

**Support Status**:

<!-- npu="910b" id1342 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1342 -->
<!-- npu="A3" id1343 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1343 -->
<!-- npu="950" id1344 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1344 -->

</div>

</div>

## KL Divergence

### torch.distributions.kl.kl_divergence

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.kl.kl_divergence](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.kl.kl_divergence)

**Support Status**:

<!-- npu="910b" id1345 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1345 -->
<!-- npu="A3" id1346 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1346 -->
<!-- npu="950" id1347 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1347 -->

</div>

## Transforms

### <code><i>class</i></code> torch.distributions.transforms.AbsTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.AbsTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.AbsTransform)

**Support Status**:

<!-- npu="910b" id1348 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1348 -->
<!-- npu="A3" id1349 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1349 -->
<!-- npu="950" id1350 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1350 -->

<!-- npu="950" id1432 -->
**Constraints & Notes**: <term>Ascend 950DT products</term>: complex64 and complex128 are not supported.
<!-- end id1432 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.AffineTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.AffineTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.AffineTransform)

**Support Status**:

<!-- npu="910b" id1351 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1351 -->
<!-- npu="A3" id1352 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1352 -->
<!-- npu="950" id1353 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1353 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.CatTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.CatTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.CatTransform)

**Support Status**:

<!-- npu="910b" id1354 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1354 -->
<!-- npu="A3" id1355 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1355 -->
<!-- npu="950" id1356 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1356 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.CorrCholeskyTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.CorrCholeskyTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.CorrCholeskyTransform)

**Support Status**:

<!-- npu="910b" id1357 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1357 -->
<!-- npu="A3" id1358 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1358 -->
<!-- npu="950" id1359 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1359 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.ExpTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.ExpTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.ExpTransform)

**Support Status**:

<!-- npu="910b" id1360 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1360 -->
<!-- npu="A3" id1361 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1361 -->
<!-- npu="950" id1362 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1362 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.LowerCholeskyTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.LowerCholeskyTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.LowerCholeskyTransform)

**Support Status**:

<!-- npu="910b" id1363 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1363 -->
<!-- npu="A3" id1364 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1364 -->
<!-- npu="950" id1365 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1365 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.PositiveDefiniteTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.PositiveDefiniteTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.PositiveDefiniteTransform)

**Support Status**:

<!-- npu="910b" id1366 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1366 -->
<!-- npu="A3" id1367 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1367 -->
<!-- npu="950" id1368 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1368 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.PowerTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.PowerTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.PowerTransform)

**Support Status**:

<!-- npu="910b" id1369 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1369 -->
<!-- npu="A3" id1370 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1370 -->
<!-- npu="950" id1371 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1371 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.ReshapeTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.ReshapeTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.ReshapeTransform)

**Support Status**:

<!-- npu="910b" id1372 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1372 -->
<!-- npu="A3" id1373 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1373 -->
<!-- npu="950" id1374 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1374 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.SigmoidTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.SigmoidTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.SigmoidTransform)

**Support Status**:

<!-- npu="910b" id1375 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1375 -->
<!-- npu="A3" id1376 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1376 -->
<!-- npu="950" id1377 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1377 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.SoftplusTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.SoftplusTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.SoftplusTransform)

**Support Status**:

<!-- npu="910b" id1378 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1378 -->
<!-- npu="A3" id1379 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1379 -->
<!-- npu="950" id1380 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1380 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.TanhTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.TanhTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.TanhTransform)

**Support Status**:

<!-- npu="910b" id1381 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1381 -->
<!-- npu="A3" id1382 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1382 -->
<!-- npu="950" id1383 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1383 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.StackTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.StackTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.StackTransform)

**Support Status**:

<!-- npu="910b" id1384 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1384 -->
<!-- npu="A3" id1385 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1385 -->
<!-- npu="950" id1386 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1386 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.SoftmaxTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.SoftmaxTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.SoftmaxTransform)

**Support Status**:

<!-- npu="910b" id1387 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1387 -->
<!-- npu="A3" id1388 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1388 -->
<!-- npu="950" id1389 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1389 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.StickBreakingTransform

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.StickBreakingTransform](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.StickBreakingTransform)

**Support Status**:

<!-- npu="910b" id1390 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1390 -->
<!-- npu="A3" id1391 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1391 -->
<!-- npu="950" id1392 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1392 -->

</div>

### <code><i>class</i></code> torch.distributions.transforms.Transform

<div style="margin-left: 2em">

> <font size="3">inv()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.Transform.inv](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.Transform.inv)

**Support Status**:

<!-- npu="910b" id1393 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1393 -->
<!-- npu="A3" id1394 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1394 -->
<!-- npu="950" id1395 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1395 -->

</div>

> <font size="3">sign()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.Transform.sign](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.Transform.sign)

**Support Status**:

<!-- npu="910b" id1396 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1396 -->
<!-- npu="A3" id1397 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1397 -->
<!-- npu="950" id1398 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1398 -->

</div>

> <font size="3">log_abs_det_jacobian()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.Transform.log_abs_det_jacobian](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.Transform.log_abs_det_jacobian)

**Support Status**:

<!-- npu="910b" id1399 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1399 -->
<!-- npu="A3" id1400 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1400 -->
<!-- npu="950" id1401 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1401 -->

</div>

> <font size="3">forward_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.Transform.forward_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.Transform.forward_shape)

**Support Status**:

<!-- npu="910b" id1402 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1402 -->
<!-- npu="A3" id1403 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1403 -->
<!-- npu="950" id1404 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1404 -->

</div>

> <font size="3">inverse_shape()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.transforms.Transform.inverse_shape](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.transforms.Transform.inverse_shape)

**Support Status**:

<!-- npu="910b" id1405 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1405 -->
<!-- npu="A3" id1406 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1406 -->
<!-- npu="950" id1407 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1407 -->

</div>

</div>

## Constraints

### torch.distributions.constraints.dependent_property

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.constraints.dependent_property](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.constraints.dependent_property)

**Support Status**:

<!-- npu="910b" id1408 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1408 -->
<!-- npu="A3" id1409 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1409 -->
<!-- npu="950" id1410 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1410 -->

</div>

### torch.distributions.constraints.greater_than

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.constraints.greater_than](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.constraints.greater_than)

**Support Status**:

<!-- npu="910b" id1411 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1411 -->
<!-- npu="A3" id1412 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1412 -->
<!-- npu="950" id1413 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1413 -->

</div>

### torch.distributions.constraints.less_than

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.constraints.less_than](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.constraints.less_than)

**Support Status**:

<!-- npu="910b" id1414 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1414 -->
<!-- npu="A3" id1415 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1415 -->
<!-- npu="950" id1416 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1416 -->

</div>

### torch.distributions.constraints.multinomial

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.constraints.multinomial](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.constraints.multinomial)

**Support Status**:

<!-- npu="910b" id1417 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1417 -->
<!-- npu="A3" id1418 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1418 -->
<!-- npu="950" id1419 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1419 -->

</div>

### torch.distributions.constraints.cat

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.constraints.cat](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.constraints.cat)

**Support Status**:

<!-- npu="910b" id1420 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1420 -->
<!-- npu="A3" id1421 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1421 -->
<!-- npu="950" id1422 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1422 -->

</div>

### torch.distributions.constraints.stack

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.constraints.stack](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.constraints.stack)

**Support Status**:

<!-- npu="910b" id1423 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1423 -->
<!-- npu="A3" id1424 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1424 -->
<!-- npu="950" id1425 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1425 -->

</div>

## Constraint Registry

### <code><i>class</i></code> torch.distributions.constraint_registry.ConstraintRegistry

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.constraint_registry.ConstraintRegistry](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry)

**Support Status**:

<!-- npu="910b" id1426 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1426 -->
<!-- npu="A3" id1427 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1427 -->
<!-- npu="950" id1428 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1428 -->

> <font size="3">register()</font>

<div style="margin-left: 2em">

**Native Documentation**: [torch.distributions.constraint_registry.ConstraintRegistry.register](https://pytorch.org/docs/2.12/distributions.html#torch.distributions.constraint_registry.ConstraintRegistry.register)

**Support Status**:

<!-- npu="910b" id1429 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1429 -->
<!-- npu="A3" id1430 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1430 -->
<!-- npu="950" id1431 -->
- <term>Ascend 950DT products</term>: Not Supported
<!-- end id1431 -->

</div>

</div>
