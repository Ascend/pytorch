#ifndef FXRT_SRC_HARDWARE_CAPTURE_GRAPH_H
#define FXRT_SRC_HARDWARE_CAPTURE_GRAPH_H

#include <memory>
#include <vector>

namespace fxrt {
class CaptureGraph {
 public:
  virtual ~CaptureGraph() = default;
  virtual bool CaptureBegin(void* stream) = 0;
  virtual void CaptureGetInfo(void* stream) = 0;
  virtual void CaptureEnd(void* stream) = 0;
  virtual void ExecuteCaptureGraph(void* stream) = 0;
  virtual void CaptureTaskGrpBegin(void* stream) = 0;
  virtual void CaptureTaskGrpEnd(void* stream, void** task_grp) = 0;
  virtual void CaptureTaskUpdateBegin(void* updateStream, void* task_grp) = 0;
  virtual void CaptureTaskUpdateEnd(void* updateStream) = 0;
};
using CaptureGraphPtr = std::shared_ptr<CaptureGraph>;
} // namespace fxrt
#endif // FXRT_SRC_HARDWARE_CAPTURE_GRAPH_H
