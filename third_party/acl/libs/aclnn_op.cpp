#include "acl/op_api/aclnn_op.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAbsGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnAbs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnAdaptiveAvgPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *output_size,
                                                   aclTensor *out, uint64_t *workspace_size, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnAdaptiveAvgPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnAdaptiveAvgPool2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnAddGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                     aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnAddsGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnAdds(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceAddGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                            uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnAddcdivGetWorkspaceSize(const aclTensor *self, const aclTensor *tensor1, const aclTensor *tensor2,
                                         const aclScalar *value, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnAddcdiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceAddcdivGetWorkspaceSize(const aclTensor *selfRef, const aclTensor *tensor1, const aclTensor *tensor2,
                                                const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceAddcdiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnAddcmulGetWorkspaceSize(const aclTensor *self, const aclTensor *tensor1, const aclTensor *tensor2,
                                         const aclScalar *value, aclTensor *out, uint64_t *workspaceSize,
                                         aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnAddcmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceAddcmulGetWorkspaceSize(const aclTensor *self, const aclTensor *tensor1,
                                                const aclTensor *tensor2, const aclScalar *value,
                                                uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceAddcmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnAddmvGetWorkspaceSize(const aclTensor *self, const aclTensor *mat, const aclTensor *vec, const aclScalar *alpha,
                                       const aclScalar *beta, aclTensor *out, int8_t cubeMathType, uint64_t *workspace_size, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnAddmv(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceAddmvGetWorkspaceSize(const aclTensor *self, const aclTensor *mat, const aclTensor *vec, const aclScalar *alpha,
                                              const aclScalar *beta, int8_t cubeMathType, uint64_t *workspace_size, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceAddmv(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnAllGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepdim, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnArangeGetWorkspaceSize(const aclScalar *start, const aclScalar *end, const aclScalar *step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnArange(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnArgMaxGetWorkspaceSize(const aclTensor *self, const int64_t dim, const bool keepdim,
                                        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnArgMax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnAvgPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernelSize,
                                           const aclIntArray *strides, const aclIntArray *paddings,
                                           const bool ceilMode, const bool countIncludePad,
                                           const uint64_t divisorOverride, const int8_t cubeMathType,
                                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnAvgPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out,
                                             int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                           aclTensor *runningMean, aclTensor *runningVar, bool training,
                                           float momentum, float eps, aclTensor *output, aclTensor *saveMean,
                                           aclTensor *saveInvstd, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnBatchNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *input,
                                                   const aclTensor *weight, const aclTensor *runningMean,
                                                   const aclTensor *runningVar, const aclTensor *saveMean,
                                                   const aclTensor *saveInvstd, bool training, float eps,
                                                   const aclBoolArray *outputMask, aclTensor *gradInput,
                                                   aclTensor *gradWeight, aclTensor *gradBias,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnBatchNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnBernoulliGetWorkspaceSize(const aclTensor *input, const aclScalar *prob, int64_t seed, int64_t offset,
                                           aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnBernoulli(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(const aclTensor *input, const aclTensor *prob, int64_t seed,
                                                 int64_t offset, aclTensor *out, uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnBernoulliTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(const aclTensor *input, const aclScalar *prob, int64_t seed,
                                                  int64_t offset, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceBernoulli(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(const aclTensor *input, const aclTensor *prob, int64_t seed,
                                                        int64_t offset, uint64_t *workspaceSize,
                                                        aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceBernoulliTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream) {return 0;}; 

aclnnStatus aclnnBinaryCrossEntropyGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight,
    aclTensor *out, int64_t reduction, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnBinaryCrossEntropy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnBitwiseAndTensorOutGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
                                                     aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnBitwiseAndTensorOut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
    uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceBitwiseAndTensorOut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor *self, const aclDataType dtype,
                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnCast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnCatGetWorkspaceSize(const aclTensorList *tensors, int64_t dim, aclTensor *out, uint64_t *workspaceSize,
                                     aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnCat(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnCeilGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnCeil(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnClampGetWorkspaceSize(const aclTensor *self, const aclScalar *clipValueMin,
                                       const aclScalar *clipValueMax, aclTensor *out, uint64_t *workspaceSize,
                                       aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnClamp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnClampMinGetWorkspaceSize(const aclTensor *self, const aclScalar *clipValueMin, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnClampMin(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnClampTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *clipValueMin,
                                             const aclTensor *clipValueMax, aclTensor *out, uint64_t *workspaceSize,
                                             aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnClampTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                             const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnConvolutionGetWorkspaceSize(
    const aclTensor *input, const aclTensor *weight,
    const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding,
    const aclIntArray *dilation, const bool transposed, const aclIntArray *outputPadding,
    const int64_t groups, aclTensor *output,
    uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnConvolution(void *workspace, const uint64_t workspaceSize, aclOpExecutor *executor,
                             const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight,
                                                     const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding,
                                                     const aclIntArray *dilation, const bool transposed, const aclIntArray *outputPadding,
                                                     const int groups, const aclBoolArray *outputMask, const int8_t cubeMathType,
                                                     aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnConvolutionBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnCumsumGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype,
                                        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnDivGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnDiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceDivGetWorkspaceSize(const aclTensor *self, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnInplaceDiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnDivModGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int roundingMode, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnDivMod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnDivsGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int roundingMode, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnDivs(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnDivMods(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnDotGetWorkspaceSize(const aclTensor *self, const aclTensor *tensor, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnDot(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus  aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *indices,
                                                         uint64_t numWeights, uint64_t paddingIdx, bool scaleGradByFreq,
                                                         const aclTensor *out, uint64_t *workspaceSize,
                                                         aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnEmbeddingDenseBackward(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor, const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnEmbeddingRenormGetWorkspaceSize(aclTensor *selfRef,
                                                 const aclTensor *indices,
                                                 double maxNorm,
                                                 double normType,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnEmbeddingRenorm(void *workspace, uint64_t workspace_size, aclOpExecutor *executor,
                                 const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnEqualGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnEqual(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnEqScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnEqScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnEqTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnEqTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnExpGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                     aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnExp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceExpGetWorkspaceSize(const aclTensor *selfRef, uint64_t *workspaceSize,
                                            aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceExp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceFillScalarGetWorkspaceSize(aclTensor * selfRef, const aclScalar * value, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnInplaceFillScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnInplaceFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *value, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnFlipGetWorkspaceSize(const aclTensor *self, const aclIntArray *dims, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnFlip(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnFloorGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnFloor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnGeScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnGeScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnGeluGetWorkspaceSize(const aclTensor *self, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnGelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnGtScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnGtScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnGtTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnGtTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnHardswishGetWorkspaceSize(const aclTensor *self, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnHardswish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceHardswishGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnInplaceHardswish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnHardswishBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *out,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnHardswishBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnHistcGetWorkspaceSize(const aclTensor *self, int32_t bins, const aclScalar *min, const aclScalar *max,
                                       aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnHistc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnIndex(void *workspace,
                       uint64_t workspaceSize,
                       aclOpExecutor *executor,
                       const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnIndexGetWorkspaceSize(const aclTensor *self,
                                       const aclTensorList*indices,
                                       aclTensor *out,
                                       uint64_t *workspaceSize,
                                       aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnIndexPutImpl(void *workspace,
                              uint64_t workspace_size,
                              aclOpExecutor *executor,
                              const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnIndexPutImplGetWorkspaceSize(const aclTensor *selfRef,
                                              const aclTensorList *indices,
                                              const aclTensor *values,
                                              const bool accumulate,
                                              const bool unsafe,
                                              uint64_t *workspace_size,
                                              aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out,
                                             uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnIndexSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnLayerNormGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape,
                                           const aclTensor *weight, const aclTensor *bias, double eps,
                                           aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut,
                                           uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnLayerNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *input,
                                                   const aclIntArray *normalizedShape, const aclTensor *mean,
                                                   const aclTensor *rstd, const aclTensor *weight,
                                                   const aclTensor *bias, const aclBoolArray *outputMask,
                                                   aclTensor *gradInputOut, aclTensor *gradWeightOut,
                                                   aclTensor *gradBiasOut, uint64_t *workspaceSize,
                                                   aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnLayerNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnLogGetWorkspaceSize(const aclTensor *self, const aclTensor *out,
                                     uint64_t *workspace_size, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnLog(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnLogicalAndGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                            uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnLogicalAnd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnLogicalOrGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                           uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnLogicalOr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnLogSoftmaxGetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* out, uint64_t* workspaceSize,
                                            aclOpExecutor** executor) {return 0;}; 

aclnnStatus aclnnLogSoftmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnLogSoftmaxBackwardGetWorkspaceSize(const aclTensor *grad_output, const aclTensor *output,
    int64_t dim, aclTensor *out, uint64_t *workspace_size, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnLogSoftmaxBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnLessTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                            uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnLessTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask,
    const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceMaskedFillScalar(void *workspace, uint64_t workspace_size,
    aclOpExecutor *executor, const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask,
    const aclTensor *value, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceMaskedFillTensor(void *workspace, uint64_t workspace_size,
    aclOpExecutor *executor, const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnMaskedSelectGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, aclTensor *out, 
    uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus  aclnnMaskedSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                        int8_t cubeMathType, uint64_t *workspace_size, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnMatmul(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnMaximumGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                         uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnMaximum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnMeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dimTest, bool keepDim, aclDataType dtype,
                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnMean(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnMulsGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnMuls(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};                                   

aclnnStatus aclnnNegGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                     aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnNeg(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnNeScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnNeScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnNeTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnNeTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnNLLLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight,
                                         int64_t reduction, int64_t ignoreIndex, aclTensor *out,
                                         aclTensor *totalWeightOut, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnNLLLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnNllLoss2dGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight,
                                           int64_t reduction, int64_t ignoreIndex, aclTensor *out, aclTensor *totalWeight,
                                           uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnNllLoss2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnNLLLoss2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                   const aclTensor *target, const aclTensor *weight, int64_t reduction,
                                                   int64_t ignoreIndex, aclTensor *totalWeight,  aclTensor *out,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnNLLLoss2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream) {return 0;}; 

aclnnStatus aclnnNLLLossBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                                 const aclTensor *target, const aclTensor *weight, int64_t reduction,
                                                 int64_t ignoreIndex, const aclTensor *totalWeight, aclTensor *out,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnNLLLossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnNormalGetWorkspaceSize(const aclTensor *self, float mean, float std,
                                        int64_t seed, int64_t offset, aclTensor *out,
                                        uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceNormalGetWorkspaceSize(const aclTensor *selfRef, float mean, float std,
                                               int64_t seed, int64_t offset, uint64_t *workspaceSize,
                                               aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnNormal(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnPowTensorScalarGetWorkspaceSize(const aclTensor *self,
                                                 const aclScalar *exponent,
                                                 const aclTensor *out,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnPowTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplacePowTensorScalarGetWorkspaceSize(const aclTensor *self,
                                                        const aclScalar *exponent,
                                                        uint64_t *workspaceSize,
                                                        aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplacePowTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnPowTensorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* exponent, aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnPowTensorTensor(void *workspace, uint64_t workspaceSize,  aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnPreluGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnPrelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceRandomGetWorkspaceSize(const aclTensor *selfRef, int64_t from, int64_t to, int64_t seed, int64_t offset,
                                               uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnRandomGetWorkspaceSize(const aclTensor *self, int64_t from, int64_t to, int64_t seed, int64_t offset,
                                        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceRandom(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnRandom(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnRandpermGetWorkspaceSize(int64_t n, int64_t seed, int64_t offset, aclTensor* out,  uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnRandperm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnReduceSumGetWorkspaceSize(const aclTensor *self,
                                           const aclIntArray *dim,
                                           bool keep_dim,
                                           aclTensor *out,
                                           uint64_t *workspaceSize,
                                           aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnReduceSum(void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnReflectionPad2dBackwardGetWorkspaceSize(const aclTensor *gradOutput,
    const aclTensor *input, const aclIntArray *padding, aclTensor *gradInput,
    uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnReflectionPad2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnReluGetWorkspaceSize(const aclTensor *self, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnRelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceReluGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnInplaceRelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnRollGetWorkspaceSize(const aclTensor *x, const aclIntArray *shifts, const aclIntArray *dims,
                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnRoll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnScatterAddGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
                                            const aclTensor *src, aclTensor *out, uint64_t *workspaceSize,
                                            aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnScatterAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnSigmoidGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize,
                                         aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnSigmoid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceSigmoidGetWorkspaceSize(const aclTensor *selfRef, uint64_t *workspace_size,
                                                aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceSigmoid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnSigmoidBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output,
    aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnSigmoidBackward(void *workspace, uint64_t workspaceSize,
                                 aclOpExecutor *executor, const aclrtStream stream) {return 0;}; 

aclnnStatus aclnnSoftmaxGetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor) {return 0;}; 

aclnnStatus aclnnSoftmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnSortGetWorkspaceSize(const aclTensor *self, bool stable, int64_t dim, bool descending,
    aclTensor *valuesOut, aclTensor *indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor) {return 0;}; 

aclnnStatus aclnnSort(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnSqrtGetWorkspaceSize(const aclTensor *self, const aclTensor *out, uint64_t *workspaceSize,aclOpExecutor **opExecutor) {return 0;};

aclnnStatus aclnnSqrt(void *workspace, uint64_t workspaceSize, aclOpExecutor *opExecutor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceSqrtGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnInplaceSqrt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnStackGetWorkspaceSize(const aclTensorList *tensors, int64_t dim, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnStack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnSubGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                     aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnSub(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceSubGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,
                                            uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceSub(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnTanhGetWorkspaceSize(const aclTensor *self, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnTanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceTanhGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnInplaceTanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *threshold,
                                                   aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnThresholdBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted, aclTensor *valuesOut,
                                      aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnTopk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceUniformGetWorkspaceSize(const aclTensor *selfRef, double from, double to, uint64_t seed,
                                                uint64_t offset, uint64_t *workspace_size, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnUniformGetWorkspaceSize(const aclTensor *self, double from, double to, uint64_t seed,
                                         uint64_t offset, aclTensor *out,
                                         uint64_t *workspace_size, aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnInplaceUniform(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnUniform(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnUpsampleBilinear2D(void *workspace, uint64_t workspace_size, aclOpExecutor *executor, const aclrtStream stream) {return 0;};

aclnnStatus aclnnUpsampleBilinear2DGetWorkspaceSize(const aclTensor *self, const aclIntArray *output_size, const bool align_corners,
                                                    const double scales_h, const double scales_w, aclTensor *out, uint64_t *workspace_size,
                                                    aclOpExecutor **executor) {return 0;}; 

aclnnStatus aclnnZeroGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnZero(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};

aclnnStatus aclnnInplaceZeroGetWorkspaceSize(const aclTensor *self, uint64_t *workspaceSize, aclOpExecutor **executor) {return 0;};

aclnnStatus aclnnInplaceZero(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {return 0;};


#ifdef __cplusplus
}
#endif
