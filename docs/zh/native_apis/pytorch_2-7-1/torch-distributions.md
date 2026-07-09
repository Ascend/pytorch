# torch.distributions

> [!NOTE]   
> 若API“是否支持”为“是”，“限制与说明”为“-”，说明此API和原生API支持度保持一致。

|API名称|是否支持|限制与说明|
|--|--|--|
|torch.distributions.distribution.Distribution.arg_constraints|是|-|
|torch.distributions.distribution.Distribution.batch_shape|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.distribution.Distribution.event_shape|是|-|
|torch.distributions.distribution.Distribution.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.distribution.Distribution.mode|是|-|
|torch.distributions.distribution.Distribution.set_default_validate_args|是|-|
|torch.distributions.distribution.Distribution.stddev|是|-|
|torch.distributions.distribution.Distribution.support|是|-|
|torch.distributions.distribution.Distribution.variance|是|-|
|torch.distributions.exp_family.ExponentialFamily.entropy|是|-|
|torch.distributions.bernoulli.Bernoulli|是|-|
|torch.distributions.bernoulli.Bernoulli.arg_constraints|是|-|
|torch.distributions.bernoulli.Bernoulli.entropy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.bernoulli.Bernoulli.expand|是|-|
|torch.distributions.bernoulli.Bernoulli.has_enumerate_support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.bernoulli.Bernoulli.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.bernoulli.Bernoulli.logits|是|-|
|torch.distributions.bernoulli.Bernoulli.mean|是|-|
|torch.distributions.bernoulli.Bernoulli.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.bernoulli.Bernoulli.param_shape|是|-|
|torch.distributions.bernoulli.Bernoulli.probs|是|-|
|torch.distributions.bernoulli.Bernoulli.sample|是|-|
|torch.distributions.bernoulli.Bernoulli.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.bernoulli.Bernoulli.variance|是|-|
|torch.distributions.beta.Beta|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.concentration0|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.concentration1|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.entropy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.beta.Beta.variance|是|-|
|torch.distributions.binomial.Binomial|是|可能回退至CPU执行|
|torch.distributions.binomial.Binomial.arg_constraints|是|-|
|torch.distributions.binomial.Binomial.entropy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.expand|是|-|
|torch.distributions.binomial.Binomial.has_enumerate_support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.log_prob|是|可能回退至CPU执行|
|torch.distributions.binomial.Binomial.logits|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.param_shape|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.probs|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.binomial.Binomial.variance|是|-|
|torch.distributions.categorical.Categorical|是|-|
|torch.distributions.categorical.Categorical.arg_constraints|是|-|
|torch.distributions.categorical.Categorical.entropy|是|-|
|torch.distributions.categorical.Categorical.expand|是|-|
|torch.distributions.categorical.Categorical.has_enumerate_support|是|-|
|torch.distributions.categorical.Categorical.log_prob|是|-|
|torch.distributions.categorical.Categorical.logits|是|-|
|torch.distributions.categorical.Categorical.mean|是|-|
|torch.distributions.categorical.Categorical.mode|是|-|
|torch.distributions.categorical.Categorical.param_shape|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.categorical.Categorical.probs|是|-|
|torch.distributions.categorical.Categorical.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.categorical.Categorical.support|是|-|
|torch.distributions.categorical.Categorical.variance|是|-|
|torch.distributions.cauchy.Cauchy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.cauchy.Cauchy.arg_constraints|是|-|
|torch.distributions.cauchy.Cauchy.cdf|是|-|
|torch.distributions.cauchy.Cauchy.entropy|是|-|
|torch.distributions.cauchy.Cauchy.expand|是|-|
|torch.distributions.cauchy.Cauchy.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.cauchy.Cauchy.icdf|是|-|
|torch.distributions.cauchy.Cauchy.log_prob|是|可能回退至CPU执行|
|torch.distributions.cauchy.Cauchy.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.cauchy.Cauchy.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.cauchy.Cauchy.rsample|是|可能回退至CPU执行|
|torch.distributions.cauchy.Cauchy.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.cauchy.Cauchy.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.chi2.Chi2|是|可能回退至CPU执行|
|torch.distributions.chi2.Chi2.arg_constraints|是|-|
|torch.distributions.chi2.Chi2.df|是|-|
|torch.distributions.chi2.Chi2.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.arg_constraints|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.cdf|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.entropy|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.icdf|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.log_prob|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.logits|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.param_shape|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.probs|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.rsample|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.sample|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.stddev|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.support|是|-|
|torch.distributions.continuous_bernoulli.ContinuousBernoulli.variance|是|-|
|torch.distributions.dirichlet.Dirichlet|是|可能回退至CPU执行|
|torch.distributions.dirichlet.Dirichlet.arg_constraints|是|-|
|torch.distributions.dirichlet.Dirichlet.entropy|是|-|
|torch.distributions.dirichlet.Dirichlet.expand|是|-|
|torch.distributions.dirichlet.Dirichlet.has_rsample|是|-|
|torch.distributions.dirichlet.Dirichlet.log_prob|是|可能回退至CPU执行|
|torch.distributions.dirichlet.Dirichlet.mean|是|-|
|torch.distributions.dirichlet.Dirichlet.mode|是|-|
|torch.distributions.dirichlet.Dirichlet.rsample|是|可能回退至CPU执行|
|torch.distributions.dirichlet.Dirichlet.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.dirichlet.Dirichlet.variance|是|-|
|torch.distributions.exponential.Exponential|是|-|
|torch.distributions.exponential.Exponential.arg_constraints|是|-|
|torch.distributions.exponential.Exponential.cdf|是|-|
|torch.distributions.exponential.Exponential.entropy|是|-|
|torch.distributions.exponential.Exponential.expand|是|-|
|torch.distributions.exponential.Exponential.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.exponential.Exponential.icdf|是|-|
|torch.distributions.exponential.Exponential.log_prob|是|-|
|torch.distributions.exponential.Exponential.mean|是|-|
|torch.distributions.exponential.Exponential.mode|是|-|
|torch.distributions.exponential.Exponential.rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.exponential.Exponential.stddev|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.exponential.Exponential.support|是|-|
|torch.distributions.exponential.Exponential.variance|是|-|
|torch.distributions.fishersnedecor.FisherSnedecor|是|可能回退至CPU执行|
|torch.distributions.fishersnedecor.FisherSnedecor.arg_constraints|是|-|
|torch.distributions.fishersnedecor.FisherSnedecor.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.fishersnedecor.FisherSnedecor.has_rsample|是|-|
|torch.distributions.fishersnedecor.FisherSnedecor.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.fishersnedecor.FisherSnedecor.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.fishersnedecor.FisherSnedecor.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.fishersnedecor.FisherSnedecor.rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.fishersnedecor.FisherSnedecor.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.fishersnedecor.FisherSnedecor.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.gamma.Gamma|是<br>暂不支持<term>Ascend 950DT</term>|可能回退至CPU执行|
|torch.distributions.gamma.Gamma.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.gamma.Gamma.cdf|是|可能回退至CPU执行|
|torch.distributions.gamma.Gamma.entropy|是|-|
|torch.distributions.gamma.Gamma.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.gamma.Gamma.has_rsample|是|-|
|torch.distributions.gamma.Gamma.log_prob|是|可能回退至CPU执行|
|torch.distributions.gamma.Gamma.mean|是|-|
|torch.distributions.gamma.Gamma.mode|是|-|
|torch.distributions.gamma.Gamma.rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.gamma.Gamma.support|是|-|
|torch.distributions.gamma.Gamma.variance|是|-|
|torch.distributions.geometric.Geometric|是|-|
|torch.distributions.geometric.Geometric.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.geometric.Geometric.entropy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.geometric.Geometric.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.geometric.Geometric.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.geometric.Geometric.logits|是|-|
|torch.distributions.geometric.Geometric.mean|是|-|
|torch.distributions.geometric.Geometric.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.geometric.Geometric.probs|是|-|
|torch.distributions.geometric.Geometric.sample|是|-|
|torch.distributions.geometric.Geometric.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.geometric.Geometric.variance|是|-|
|torch.distributions.gumbel.Gumbel|是|-|
|torch.distributions.gumbel.Gumbel.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.gumbel.Gumbel.entropy|是|-|
|torch.distributions.gumbel.Gumbel.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.gumbel.Gumbel.log_prob|是|-|
|torch.distributions.gumbel.Gumbel.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.gumbel.Gumbel.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.gumbel.Gumbel.stddev|是|-|
|torch.distributions.gumbel.Gumbel.support|是|-|
|torch.distributions.gumbel.Gumbel.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_cauchy.HalfCauchy|是|可能回退至CPU执行|
|torch.distributions.half_cauchy.HalfCauchy.arg_constraints|是|-|
|torch.distributions.half_cauchy.HalfCauchy.cdf|是|-|
|torch.distributions.half_cauchy.HalfCauchy.entropy|是|-|
|torch.distributions.half_cauchy.HalfCauchy.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_cauchy.HalfCauchy.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_cauchy.HalfCauchy.icdf|是|-|
|torch.distributions.half_cauchy.HalfCauchy.log_prob|是|-|
|torch.distributions.half_cauchy.HalfCauchy.mean|是|-|
|torch.distributions.half_cauchy.HalfCauchy.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_cauchy.HalfCauchy.scale|是|-|
|torch.distributions.half_cauchy.HalfCauchy.support|是|-|
|torch.distributions.half_cauchy.HalfCauchy.variance|是|-|
|torch.distributions.half_normal.HalfNormal|是|可能回退至CPU执行|
|torch.distributions.half_normal.HalfNormal.arg_constraints|是|-|
|torch.distributions.half_normal.HalfNormal.cdf|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_normal.HalfNormal.entropy|是|-|
|torch.distributions.half_normal.HalfNormal.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_normal.HalfNormal.has_rsample|是|-|
|torch.distributions.half_normal.HalfNormal.icdf|是|-|
|torch.distributions.half_normal.HalfNormal.log_prob|是|-|
|torch.distributions.half_normal.HalfNormal.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_normal.HalfNormal.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_normal.HalfNormal.scale|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_normal.HalfNormal.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.half_normal.HalfNormal.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.independent.Independent|是|-|
|torch.distributions.independent.Independent.arg_constraints|是|-|
|torch.distributions.independent.Independent.entropy|是|-|
|torch.distributions.independent.Independent.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.independent.Independent.has_enumerate_support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.independent.Independent.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.independent.Independent.log_prob|是|-|
|torch.distributions.independent.Independent.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.independent.Independent.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.independent.Independent.rsample|是|-|
|torch.distributions.independent.Independent.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.independent.Independent.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.independent.Independent.variance|是|-|
|torch.distributions.kumaraswamy.Kumaraswamy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.kumaraswamy.Kumaraswamy.arg_constraints|是|-|
|torch.distributions.kumaraswamy.Kumaraswamy.entropy|是|可能回退至CPU执行|
|torch.distributions.kumaraswamy.Kumaraswamy.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.kumaraswamy.Kumaraswamy.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.kumaraswamy.Kumaraswamy.mean|是|-|
|torch.distributions.kumaraswamy.Kumaraswamy.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.kumaraswamy.Kumaraswamy.support|是|-|
|torch.distributions.kumaraswamy.Kumaraswamy.variance|是|-|
|torch.distributions.lkj_cholesky.LKJCholesky|是|可能回退至CPU执行|
|torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lkj_cholesky.LKJCholesky.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lkj_cholesky.LKJCholesky.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lkj_cholesky.LKJCholesky.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lkj_cholesky.LKJCholesky.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.laplace.Laplace|是|-|
|torch.distributions.laplace.Laplace.arg_constraints|是|-|
|torch.distributions.laplace.Laplace.cdf|是|-|
|torch.distributions.laplace.Laplace.entropy|是|-|
|torch.distributions.laplace.Laplace.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.laplace.Laplace.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.laplace.Laplace.icdf|是|可能回退至CPU执行|
|torch.distributions.laplace.Laplace.log_prob|是|-|
|torch.distributions.laplace.Laplace.mean|是|-|
|torch.distributions.laplace.Laplace.mode|是|-|
|torch.distributions.laplace.Laplace.rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.laplace.Laplace.stddev|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.laplace.Laplace.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.laplace.Laplace.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.log_normal.LogNormal|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.log_normal.LogNormal.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.log_normal.LogNormal.entropy|是|可能回退至CPU执行|
|torch.distributions.log_normal.LogNormal.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.log_normal.LogNormal.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.log_normal.LogNormal.loc|是|-|
|torch.distributions.log_normal.LogNormal.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.log_normal.LogNormal.mode|是|-|
|torch.distributions.log_normal.LogNormal.scale|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.log_normal.LogNormal.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.log_normal.LogNormal.variance|是|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.covariance_matrix|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.entropy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.precision_matrix|是|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.scale_tril|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.cdf|是|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.component_distribution|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.mixture_distribution|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.mixture_same_family.MixtureSameFamily.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.entropy|是|-|
|torch.distributions.multinomial.Multinomial.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.logits|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.param_shape|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.probs|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.total_count|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multinomial.Multinomial.variance|是|-|
|torch.distributions.multivariate_normal.MultivariateNormal|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.covariance_matrix|是<br>暂不支持<term>Ascend 950DT</term>|dim需小于等于8192|
|torch.distributions.multivariate_normal.MultivariateNormal.entropy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|可能回退至CPU执行|
|torch.distributions.multivariate_normal.MultivariateNormal.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.precision_matrix|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.rsample|是<br>暂不支持<term>Ascend 950DT</term>|可能回退至CPU执行|
|torch.distributions.multivariate_normal.MultivariateNormal.scale_tril|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.multivariate_normal.MultivariateNormal.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.negative_binomial.NegativeBinomial|是|可能回退至CPU执行|
|torch.distributions.negative_binomial.NegativeBinomial.arg_constraints|是|-|
|torch.distributions.negative_binomial.NegativeBinomial.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.negative_binomial.NegativeBinomial.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.negative_binomial.NegativeBinomial.logits|是|-|
|torch.distributions.negative_binomial.NegativeBinomial.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.negative_binomial.NegativeBinomial.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.negative_binomial.NegativeBinomial.param_shape|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.negative_binomial.NegativeBinomial.probs|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.negative_binomial.NegativeBinomial.sample|是|可能回退至CPU执行|
|torch.distributions.negative_binomial.NegativeBinomial.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.negative_binomial.NegativeBinomial.variance|是|-|
|torch.distributions.normal.Normal.arg_constraints|是|-|
|torch.distributions.normal.Normal.cdf|是|-|
|torch.distributions.normal.Normal.entropy|是|-|
|torch.distributions.normal.Normal.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.normal.Normal.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.normal.Normal.icdf|是|-|
|torch.distributions.normal.Normal.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.normal.Normal.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.normal.Normal.mode|是|-|
|torch.distributions.normal.Normal.rsample|是|-|
|torch.distributions.normal.Normal.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.normal.Normal.stddev|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.normal.Normal.support|是|-|
|torch.distributions.normal.Normal.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.one_hot_categorical.OneHotCategorical|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.arg_constraints|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.entropy|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.has_enumerate_support|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.log_prob|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.logits|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.mode|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.param_shape|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.probs|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.sample|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.support|是|-|
|torch.distributions.one_hot_categorical.OneHotCategorical.variance|是|-|
|torch.distributions.pareto.Pareto|是|-|
|torch.distributions.pareto.Pareto.arg_constraints|是|-|
|torch.distributions.pareto.Pareto.entropy|是|-|
|torch.distributions.pareto.Pareto.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.pareto.Pareto.mean|是|-|
|torch.distributions.pareto.Pareto.mode|是|-|
|torch.distributions.pareto.Pareto.support|是|-|
|torch.distributions.pareto.Pareto.variance|是|-|
|torch.distributions.poisson.Poisson|是|可能回退至CPU执行|
|torch.distributions.poisson.Poisson.arg_constraints|是|-|
|torch.distributions.poisson.Poisson.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.poisson.Poisson.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.poisson.Poisson.mean|是|-|
|torch.distributions.poisson.Poisson.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.poisson.Poisson.sample|是|-|
|torch.distributions.poisson.Poisson.support|是|-|
|torch.distributions.poisson.Poisson.variance|是|-|
|torch.distributions.relaxed_bernoulli.RelaxedBernoulli|是|-|
|torch.distributions.relaxed_bernoulli.RelaxedBernoulli.arg_constraints|是|-|
|torch.distributions.relaxed_bernoulli.RelaxedBernoulli.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.relaxed_bernoulli.RelaxedBernoulli.has_rsample|是|-|
|torch.distributions.relaxed_bernoulli.RelaxedBernoulli.logits|是|-|
|torch.distributions.relaxed_bernoulli.RelaxedBernoulli.probs|是|-|
|torch.distributions.relaxed_bernoulli.RelaxedBernoulli.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.relaxed_bernoulli.RelaxedBernoulli.temperature|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli|是|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.expand|是|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.logits|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.param_shape|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.probs|是|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.rsample|是|-|
|torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli.support|是|-|
|torch.distributions.relaxed_categorical.RelaxedOneHotCategorical|是|可能回退至CPU执行|
|torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.arg_constraints|是|-|
|torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.expand|是|-|
|torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.logits|是|-|
|torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.probs|是|-|
|torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.support|是|-|
|torch.distributions.relaxed_categorical.RelaxedOneHotCategorical.temperature|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.studentT.StudentT|是|可能回退至CPU执行|
|torch.distributions.studentT.StudentT.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.studentT.StudentT.entropy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.studentT.StudentT.expand|是|-|
|torch.distributions.studentT.StudentT.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.studentT.StudentT.log_prob|是|-|
|torch.distributions.studentT.StudentT.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.studentT.StudentT.mode|是|-|
|torch.distributions.studentT.StudentT.rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.studentT.StudentT.support|是|-|
|torch.distributions.studentT.StudentT.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transformed_distribution.TransformedDistribution|是|-|
|torch.distributions.transformed_distribution.TransformedDistribution.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transformed_distribution.TransformedDistribution.cdf|是|-|
|torch.distributions.transformed_distribution.TransformedDistribution.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transformed_distribution.TransformedDistribution.has_rsample|是|-|
|torch.distributions.transformed_distribution.TransformedDistribution.icdf|是|-|
|torch.distributions.transformed_distribution.TransformedDistribution.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transformed_distribution.TransformedDistribution.rsample|是|-|
|torch.distributions.transformed_distribution.TransformedDistribution.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transformed_distribution.TransformedDistribution.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.uniform.Uniform.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.uniform.Uniform.cdf|是|-|
|torch.distributions.uniform.Uniform.entropy|是|-|
|torch.distributions.uniform.Uniform.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.uniform.Uniform.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.uniform.Uniform.icdf|是|-|
|torch.distributions.uniform.Uniform.log_prob|是|-|
|torch.distributions.uniform.Uniform.mean|是|-|
|torch.distributions.uniform.Uniform.mode|是|-|
|torch.distributions.uniform.Uniform.rsample|是|-|
|torch.distributions.uniform.Uniform.stddev|是|-|
|torch.distributions.uniform.Uniform.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.uniform.Uniform.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.von_mises.VonMises|是|-|
|torch.distributions.von_mises.VonMises.arg_constraints|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.von_mises.VonMises.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.von_mises.VonMises.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.von_mises.VonMises.log_prob|是|-|
|torch.distributions.von_mises.VonMises.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.von_mises.VonMises.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.von_mises.VonMises.sample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.von_mises.VonMises.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.von_mises.VonMises.variance|是|-|
|torch.distributions.weibull.Weibull|是|-|
|torch.distributions.weibull.Weibull.arg_constraints|是|-|
|torch.distributions.weibull.Weibull.entropy|是|-|
|torch.distributions.weibull.Weibull.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.weibull.Weibull.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.weibull.Weibull.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.weibull.Weibull.support|是|-|
|torch.distributions.weibull.Weibull.variance|是|-|
|torch.distributions.wishart.Wishart|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.arg_constraints|是|-|
|torch.distributions.wishart.Wishart.covariance_matrix|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.entropy|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.expand|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.has_rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.log_prob|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.mean|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.mode|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.precision_matrix|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.rsample|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.scale_tril|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.support|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.wishart.Wishart.variance|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.kl.kl_divergence|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transforms.AbsTransform|是|<term>Ascend 950DT</term>：不支持complex64，complex128|
|torch.distributions.transforms.AffineTransform|是|-|
|torch.distributions.transforms.CatTransform|是|-|
|torch.distributions.transforms.CorrCholeskyTransform|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transforms.ExpTransform|是|-|
|torch.distributions.transforms.LowerCholeskyTransform|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transforms.PositiveDefiniteTransform|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transforms.PowerTransform|是|-|
|torch.distributions.transforms.ReshapeTransform|是|-|
|torch.distributions.transforms.SigmoidTransform|是|-|
|torch.distributions.transforms.SoftplusTransform|是|-|
|torch.distributions.transforms.TanhTransform|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transforms.SoftmaxTransform|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.transforms.StackTransform|是|-|
|torch.distributions.transforms.StickBreakingTransform|是|-|
|torch.distributions.transforms.Transform.inv|是|-|
|torch.distributions.transforms.Transform.sign|是|-|
|torch.distributions.transforms.Transform.log_abs_det_jacobian|是|-|
|torch.distributions.transforms.Transform.forward_shape|是|-|
|torch.distributions.transforms.Transform.inverse_shape|是|-|
|torch.distributions.constraints.cat|是|-|
|torch.distributions.constraints.dependent_property|是|-|
|torch.distributions.constraints.greater_than|是|-|
|torch.distributions.constraints.less_than|是|-|
|torch.distributions.constraints.multinomial|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.constraints.stack|是|-|
|torch.distributions.constraint_registry.ConstraintRegistry|是<br>暂不支持<term>Ascend 950DT</term>|-|
|torch.distributions.constraint_registry.ConstraintRegistry.register|是<br>暂不支持<term>Ascend 950DT</term>|-|
