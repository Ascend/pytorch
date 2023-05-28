#ifndef OP_API_ACLNN_OP_H_
#define OP_API_ACLNN_OP_H_

#include "acl/acl_op_api.h"

#ifdef __cplusplus
extern "C" {
#endif


aclnnStatus aclnnAbsGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnAbs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnAdaptiveAvgPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *output_size,
                                                   aclTensor *out, uint64_t *workspace_size, aclOpExecutor **executor);

aclnnStatus aclnnAdaptiveAvgPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream);

aclnnStatus aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnAdaptiveAvgPool2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnAddGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                     aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnAddsGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnAdds(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceAddGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                            uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnAddcdivGetWorkspaceSize(const aclTensor *self, const aclTensor *tensor1, const aclTensor *tensor2,
                                         const aclScalar *value, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnAddcdiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceAddcdivGetWorkspaceSize(const aclTensor *selfRef, const aclTensor *tensor1, const aclTensor *tensor2,
                                                const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceAddcdiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnAddcmulGetWorkspaceSize(const aclTensor *self, const aclTensor *tensor1, const aclTensor *tensor2,
                                         const aclScalar *value, aclTensor *out, uint64_t *workspaceSize,
                                         aclOpExecutor **executor);

aclnnStatus aclnnAddcmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceAddcmulGetWorkspaceSize(const aclTensor *self, const aclTensor *tensor1,
                                                const aclTensor *tensor2, const aclScalar *value,
                                                uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceAddcmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnAddmvGetWorkspaceSize(const aclTensor *self, const aclTensor *mat, const aclTensor *vec, const aclScalar *alpha,
                                       const aclScalar *beta, aclTensor *out, int8_t cubeMathType, uint64_t *workspace_size, aclOpExecutor **executor);

aclnnStatus aclnnAddmv(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceAddmvGetWorkspaceSize(const aclTensor *self, const aclTensor *mat, const aclTensor *vec, const aclScalar *alpha,
                                              const aclScalar *beta, int8_t cubeMathType, uint64_t *workspace_size, aclOpExecutor **executor);

aclnnStatus aclnnInplaceAddmv(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnAllGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepdim, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnArangeGetWorkspaceSize(const aclScalar *start, const aclScalar *end, const aclScalar *step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnArange(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnArgMaxGetWorkspaceSize(const aclTensor *self, const int64_t dim, const bool keepdim,
                                        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnArgMax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnAvgPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernelSize,
                                           const aclIntArray *strides, const aclIntArray *paddings,
                                           const bool ceilMode, const bool countIncludePad,
                                           const uint64_t divisorOverride, const int8_t cubeMathType,
                                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnAvgPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out,
                                             int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                           aclTensor *runningMean, aclTensor *runningVar, bool training,
                                           float momentum, float eps, aclTensor *output, aclTensor *saveMean,
                                           aclTensor *saveInvstd, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnBatchNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *input,
                                                   const aclTensor *weight, const aclTensor *runningMean,
                                                   const aclTensor *runningVar, const aclTensor *saveMean,
                                                   const aclTensor *saveInvstd, bool training, float eps,
                                                   const aclBoolArray *outputMask, aclTensor *gradInput,
                                                   aclTensor *gradWeight, aclTensor *gradBias,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnBatchNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream);

aclnnStatus aclnnBernoulliGetWorkspaceSize(const aclTensor *input, const aclScalar *prob, int64_t seed, int64_t offset,
                                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnBernoulli(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(const aclTensor *input, const aclTensor *prob, int64_t seed,
                                                 int64_t offset, aclTensor *out, uint64_t *workspaceSize,
                                                 aclOpExecutor **executor);

aclnnStatus aclnnBernoulliTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(const aclTensor *input, const aclScalar *prob, int64_t seed,
                                                  int64_t offset, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceBernoulli(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(const aclTensor *input, const aclTensor *prob, int64_t seed,
                                                        int64_t offset, uint64_t *workspaceSize,
                                                        aclOpExecutor **executor);

aclnnStatus aclnnInplaceBernoulliTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream);

aclnnStatus aclnnBinaryCrossEntropyGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight,
    aclTensor *out, int64_t reduction, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnBinaryCrossEntropy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnBitwiseAndTensorOutGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
                                                     aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnBitwiseAndTensorOut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceBitwiseAndTensorOut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor *self, const aclDataType dtype,
                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnCast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnCatGetWorkspaceSize(const aclTensorList *tensors, int64_t dim, aclTensor *out, uint64_t *workspaceSize,
                                     aclOpExecutor **executor);

aclnnStatus aclnnCat(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnCeilGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnCeil(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnClampGetWorkspaceSize(const aclTensor *self, const aclScalar *clipValueMin,
                                       const aclScalar *clipValueMax, aclTensor *out, uint64_t *workspaceSize,
                                       aclOpExecutor **executor);

aclnnStatus aclnnClamp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnClampMinGetWorkspaceSize(const aclTensor *self, const aclScalar *clipValueMin, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnClampMin(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnClampTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *clipValueMin,
                                             const aclTensor *clipValueMax, aclTensor *out, uint64_t *workspaceSize,
                                             aclOpExecutor **executor);

aclnnStatus aclnnClampTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                             const aclrtStream stream);

aclnnStatus aclnnConvolutionGetWorkspaceSize(
    const aclTensor *input, const aclTensor *weight,
    const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding,
    const aclIntArray *dilation, const bool transposed, const aclIntArray *outputPadding,
    const int64_t groups, aclTensor *output,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnConvolution(void *workspace, const uint64_t workspaceSize, aclOpExecutor *executor,
                             const aclrtStream stream);

aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight,
                                                     const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding,
                                                     const aclIntArray *dilation, const bool transposed, const aclIntArray *outputPadding,
                                                     const int groups, const aclBoolArray *outputMask, const int8_t cubeMathType,
                                                     aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnConvolutionBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnCumsumGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype,
                                        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnDivGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnDiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceDivGetWorkspaceSize(const aclTensor *self, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceDiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnDivModGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int roundingMode, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnDivMod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnDivsGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int roundingMode, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnDivs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnDivMods(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnDotGetWorkspaceSize(const aclTensor *self, const aclTensor *tensor, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnDot(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus  aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *indices,
                                                         uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq,
                                                         const aclTensor *out, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor);

aclnnStatus aclnnEmbeddingDenseBackward(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnEmbeddingRenormGetWorkspaceSize(aclTensor *selfRef,
                                                 const aclTensor *indices,
                                                 double maxNorm,
                                                 double normType,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor);

aclnnStatus aclnnEmbeddingRenorm(void *workspace, uint64_t workspace_size, aclOpExecutor *executor,
                                 const aclrtStream stream);

aclnnStatus aclnnEqualGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnEqual(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnEqScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnEqScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnEqTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnEqTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnExpGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                     aclOpExecutor **executor);

aclnnStatus aclnnExp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceExpGetWorkspaceSize(const aclTensor *selfRef, uint64_t *workspaceSize,
                                            aclOpExecutor **executor);

aclnnStatus aclnnInplaceExp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceFillScalarGetWorkspaceSize(aclTensor * selfRef, const aclScalar * value, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceFillScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream);

aclnnStatus aclnnInplaceFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *value, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor);

aclnnStatus aclnnInplaceFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream);

aclnnStatus aclnnFlipGetWorkspaceSize(const aclTensor *self, const aclIntArray *dims, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnFlip(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnFloorGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnFloor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnGeScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnGeScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnGeluGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnGelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnGtScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnGtScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnGtTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnGtTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnHardswishGetWorkspaceSize(const aclTensor *self, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnHardswish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceHardswishGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceHardswish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnHardswishBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *out,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnHardswishBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnHistcGetWorkspaceSize(const aclTensor *self, int32_t bins, const aclScalar *min, const aclScalar *max,
                                       aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnHistc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnIndex(void *workspace,
                       uint64_t workspaceSize,
                       aclOpExecutor *executor,
                       const aclrtStream stream);

aclnnStatus aclnnIndexGetWorkspaceSize(const aclTensor *self,
                                       const aclTensorList*indices,
                                       aclTensor *out,
                                       uint64_t *workspaceSize,
                                       aclOpExecutor **executor);

aclnnStatus aclnnIndexPutImpl(void *workspace,
                              uint64_t workspace_size,
                              aclOpExecutor *executor,
                              const aclrtStream stream);

aclnnStatus aclnnIndexPutImplGetWorkspaceSize(const aclTensor *selfRef,
                                              const aclTensorList *indices,
                                              const aclTensor *values,
                                              const bool accumulate,
                                              const bool unsafe,
                                              uint64_t *workspace_size,
                                              aclOpExecutor **executor);

aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out,
                                             uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnIndexSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnLayerNormGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape,
                                           const aclTensor *weight, const aclTensor *bias, double eps,
                                           aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut,
                                           uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnLayerNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *input,
                                                   const aclIntArray *normalizedShape, const aclTensor *mean,
                                                   const aclTensor *rstd, const aclTensor *weight,
                                                   const aclTensor *bias, const aclBoolArray *outputMask,
                                                   aclTensor *gradInputOut, aclTensor *gradWeightOut,
                                                   aclTensor *gradBiasOut, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor);

aclnnStatus aclnnLayerNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream);

aclnnStatus aclnnLogGetWorkspaceSize(const aclTensor *self, const aclTensor *out,
                                     uint64_t *workspace_size, aclOpExecutor **executor);

aclnnStatus aclnnLog(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnLogicalAndGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                            uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnLogicalAnd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnLogicalOrGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                           uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnLogicalOr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnLogSoftmaxGetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* out, uint64_t* workspaceSize,
                                            aclOpExecutor** executor);

aclnnStatus aclnnLogSoftmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

aclnnStatus aclnnLogSoftmaxBackwardGetWorkspaceSize(const aclTensor *grad_output, const aclTensor *output,
    int64_t dim, aclTensor *out, uint64_t *workspace_size, aclOpExecutor **executor);

aclnnStatus aclnnLogSoftmaxBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    const aclrtStream stream);

aclnnStatus aclnnLessTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                            uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnLessTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask,
    const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceMaskedFillScalar(void *workspace, uint64_t workspace_size,
    aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask,
    const aclTensor *value, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceMaskedFillTensor(void *workspace, uint64_t workspace_size,
    aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnMaskedSelectGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus  aclnnMaskedSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                        int8_t cubeMathType, uint64_t *workspace_size, aclOpExecutor **executor);

aclnnStatus aclnnMatmul(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnMaximumGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                         uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMaximum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnMeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dimTest, bool keepDim, aclDataType dtype,
                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMean(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnMulsGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMuls(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnNegGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                     aclOpExecutor **executor);

aclnnStatus aclnnNeg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnNeScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnNeScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnNeTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnNeTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnNLLLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight,
                                         int64_t reduction, int64_t ignoreIndex, aclTensor *out,
                                         aclTensor *totalWeightOut, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnNLLLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnNllLoss2dGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight,
                                           int64_t reduction, int64_t ignoreIndex, aclTensor *out, aclTensor *totalWeight,
                                           uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnNllLoss2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnNLLLoss2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                   const aclTensor *target, const aclTensor *weight, int64_t reduction,
                                                   int64_t ignoreIndex, aclTensor *totalWeight,  aclTensor *out,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnNLLLoss2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

aclnnStatus aclnnNLLLossBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                 const aclTensor *target, const aclTensor *weight, int64_t reduction,
                                                 int64_t ignoreIndex, const aclTensor *totalWeight, aclTensor *out,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnNLLLossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnNormalGetWorkspaceSize(const aclTensor *self, float mean, float std,
                                        int64_t seed, int64_t offset, aclTensor *out,
                                        uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceNormalGetWorkspaceSize(const aclTensor *selfRef, float mean, float std,
                                               int64_t seed, int64_t offset, uint64_t *workspaceSize,
                                               aclOpExecutor **executor);

aclnnStatus aclnnNormal(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(const aclTensor *self,
                                                 const aclScalar *exponent,
                                                 const aclTensor *out,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor);

aclnnStatus aclnnPowTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(const aclTensor *self,
                                                        const aclScalar *exponent,
                                                        uint64_t *workspaceSize,
                                                        aclOpExecutor **executor);

aclnnStatus aclnnInplacePowTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnPowTensorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* exponent, aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnPowTensorTensor(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnPreluGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnPrelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceRandomGetWorkspaceSize(const aclTensor *selfRef, int64_t from, int64_t to, int64_t seed, int64_t offset,
                                               uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnRandomGetWorkspaceSize(const aclTensor *self, int64_t from, int64_t to, int64_t seed, int64_t offset,
                                        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceRandom(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnRandom(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnRandpermGetWorkspaceSize(int64_t n, int64_t seed, int64_t offset, aclTensor* out,  uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnRandperm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnReduceSumGetWorkspaceSize(const aclTensor *self,
                                           const aclIntArray *dim,
                                           bool keep_dim,
                                           aclTensor *out,
                                           uint64_t *workspaceSize,
                                           aclOpExecutor **executor);

aclnnStatus aclnnReduceSum(void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnReflectionPad2dBackwardGetWorkspaceSize(const aclTensor *gradOutput,
    const aclTensor *input, const aclIntArray *padding, aclTensor *gradInput,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnReflectionPad2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream);

aclnnStatus aclnnReluGetWorkspaceSize(const aclTensor *self, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnRelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceReluGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceRelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnRollGetWorkspaceSize(const aclTensor *x, const aclIntArray *shifts, const aclIntArray *dims,
                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnRoll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnScatterAddGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
                                            const aclTensor *src, aclTensor *out, uint64_t *workspaceSize,
                                            aclOpExecutor **executor);

aclnnStatus aclnnScatterAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnSigmoidGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                         aclOpExecutor **executor);

aclnnStatus aclnnSigmoid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceSigmoidGetWorkspaceSize(const aclTensor *selfRef, uint64_t *workspace_size,
                                                aclOpExecutor **executor);

aclnnStatus aclnnInplaceSigmoid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                const aclrtStream stream);

aclnnStatus aclnnSigmoidBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output,
    aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnSigmoidBackward(void *workspace, uint64_t workspaceSize,
                                 aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnSoftmaxGetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnSoftmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnSortGetWorkspaceSize(const aclTensor *self, bool stable, int64_t dim, bool descending,
    aclTensor *valuesOut, aclTensor *indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnSort(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnSqrtGetWorkspaceSize(const aclTensor *self, const aclTensor *out, uint64_t *workspaceSize,aclOpExecutor **opExecutor);

aclnnStatus aclnnSqrt(void *workspace, uint64_t workspaceSize, aclOpExecutor *opExecutor, const aclrtStream stream);

aclnnStatus aclnnInplaceSqrtGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceSqrt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnStackGetWorkspaceSize(const aclTensorList *tensors, int64_t dim, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnStack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnSubGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                     aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnSub(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceSubGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                            uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceSub(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnTanhGetWorkspaceSize(const aclTensor *self, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnTanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceTanhGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceTanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *threshold,
                                                   aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnThresholdBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted, aclTensor *valuesOut,
                                      aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnTopk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnInplaceUniformGetWorkspaceSize(const aclTensor *selfRef, double from, double to, uint64_t seed,
                                                uint64_t offset, uint64_t *workspace_size, aclOpExecutor **executor);

aclnnStatus aclnnUniformGetWorkspaceSize(const aclTensor *self, double from, double to, uint64_t seed,
                                         uint64_t offset, aclTensor *out,
                                         uint64_t *workspace_size, aclOpExecutor **executor);

aclnnStatus aclnnInplaceUniform(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnUniform(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnUpsampleBilinear2D(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream);

aclnnStatus aclnnUpsampleBilinear2DGetWorkspaceSize(const aclTensor *self, const aclIntArray *output_size, const bool align_corners,
                                                    const double scales_h, const double scales_w, aclTensor *out, uint64_t *workspace_size,
                                                    aclOpExecutor **executor);

aclnnStatus aclnnZeroGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnZero(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnInplaceZeroGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnInplaceZero(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);


#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_OP_H_