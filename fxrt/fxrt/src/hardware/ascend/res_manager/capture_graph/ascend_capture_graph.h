#ifndef FXRT_SRC_HARDWARE_ASCEND_ASCEND_CAPTURE_GRAPH_H_
#define FXRT_SRC_HARDWARE_ASCEND_ASCEND_CAPTURE_GRAPH_H_

#include "acl/acl_mdl.h"
#include "hardware/hardware_abstract/capture_graph.h"

namespace fxrt::device::ascend {

class AscendCaptureGraph : public CaptureGraph {
 public:
  AscendCaptureGraph() = default;
  ~AscendCaptureGraph() override;
  bool CaptureBegin(void* stream) override;
  void CaptureGetInfo(void* stream) override;
  void CaptureEnd(void* stream) override;
  void ExecuteCaptureGraph(void* stream) override;
  void CaptureTaskGrpBegin(void* stream) override;
  void CaptureTaskGrpEnd(void* stream, void** task_grp) override;
  void CaptureTaskUpdateBegin(void* updateStream, void* task_grp) override;
  void CaptureTaskUpdateEnd(void* updateStream) override;

 protected:
  aclrtStream capture_stream_{nullptr};
#if defined(__linux__)
  aclmdlRICaptureMode mode_{aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED};
  aclmdlRI model_ri_{nullptr};
#endif
  bool finish_capture_graph_{false};
};
} // namespace fxrt::device::ascend
#endif // FXRT_SRC_HARDWARE_ASCEND_ASCEND_CAPTURE_GRAPH_H_
