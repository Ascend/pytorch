#include "hardware/ascend/res_manager/capture_graph/ascend_capture_graph.h"

#include <cstdint>
#include <string>

#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_mdl_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/ascend_event.h"

#include "common/logger.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace fxrt::device::ascend {
AscendCaptureGraph::~AscendCaptureGraph() {
#if defined(__linux__)
  if (finish_capture_graph_ && model_ri_) {
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      RT_GLOG(ERROR) << "Sync All streams failed";
    }
    auto ret = CALL_ASCEND_API(aclmdlRIDestroy, model_ri_);
    if (ret != ACL_ERROR_NONE) {
      RT_VLOG(VL_HARDWARE) << "aclmdlRIDestroy failed, ret:" << ret;
    }
  }
#endif
}

bool AscendCaptureGraph::CaptureBegin(void* stream) {
  if (finish_capture_graph_) {
    RT_GLOG(ERROR) << "Already capture a graph.";
    return false;
  }

  capture_stream_ = stream;
#if defined(__linux__)
  auto ret = CALL_ASCEND_API(aclmdlRICaptureBegin, capture_stream_, mode_);
  if (ret != ACL_ERROR_NONE) {
    RT_GLOG(ERROR) << "aclmdlRICaptureBegin failed, ret:" << ret;
    return false;
  }
  CaptureGetInfo(stream);
#endif
  return true;
}

void AscendCaptureGraph::CaptureGetInfo(void* stream) {
  CHECK_IF_NULL(stream);
  CHECK_IF_NULL(capture_stream_);
  if (stream != capture_stream_) {
    RT_GLOG(EXCEPTION) << "The current stream is not in capture status.";
  }
#if defined(__linux__)
  aclmdlRICaptureStatus status;
  auto ret = CALL_ASCEND_API(aclmdlRICaptureGetInfo, capture_stream_, &status, &model_ri_);
  if (ret != ACL_ERROR_NONE) {
    RT_GLOG(EXCEPTION) << "aclmdlRICaptureGetInfo failed, ret:" << ret;
  }
  if (status != aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
    RT_GLOG(EXCEPTION) << "aclmdlRICaptureGetInfo got wrong status: " << status;
  }
#endif
}

void AscendCaptureGraph::CaptureEnd(void* stream) {
  CHECK_IF_NULL(stream);
  if (stream != capture_stream_) {
    RT_GLOG(EXCEPTION) << "The current stream is not in capture status.";
  }
#if defined(__linux__)
  auto ret = CALL_ASCEND_API(aclmdlRICaptureEnd, capture_stream_, &model_ri_);
  if (ret != ACL_ERROR_NONE) {
    RT_GLOG(EXCEPTION) << "aclmdlRICaptureEnd failed, ret:" << ret;
  }

  finish_capture_graph_ = true;
#endif
}

void AscendCaptureGraph::ExecuteCaptureGraph(void* stream) {
  CHECK_IF_NULL(stream);
#if defined(__linux__)
  CHECK_IF_NULL(model_ri_);

  auto ret = CALL_ASCEND_API(aclmdlRIExecuteAsync, model_ri_, stream);
  if (ret != ACL_ERROR_NONE) {
    RT_GLOG(EXCEPTION) << "aclmdlRIExecuteAsync failed, ret:" << ret;
  }
#endif
}

void AscendCaptureGraph::CaptureTaskGrpBegin(void* stream) {
  CHECK_IF_NULL(stream);
  if (stream != capture_stream_) {
    RT_GLOG(EXCEPTION) << "The current stream is not in capture status.";
  }
#if defined(__linux__)
  auto ret = CALL_ASCEND_API(aclmdlRICaptureTaskGrpBegin, stream);
  if (ret != ACL_ERROR_NONE) {
    RT_GLOG(EXCEPTION) << "aclmdlRICaptureTaskGrpBegin failed, ret:" << ret;
  }
#endif
}

void AscendCaptureGraph::CaptureTaskGrpEnd(void* stream, void** task_grp) {
  CHECK_IF_NULL(stream);
  if (stream != capture_stream_) {
    RT_GLOG(EXCEPTION) << "The current stream is not in capture status.";
  }
#if defined(__linux__)
  auto ret = CALL_ASCEND_API(aclmdlRICaptureTaskGrpEnd, stream, task_grp);
  if (task_grp == nullptr) {
    RT_GLOG(EXCEPTION) << "aclmdlRICaptureTaskGrpEnd failed, ret:" << ret;
  }
#endif
}

void AscendCaptureGraph::CaptureTaskUpdateBegin(void* updateStream, void* task_grp) {
  CHECK_IF_NULL(updateStream);
#if defined(__linux__)
  auto ret = CALL_ASCEND_API(aclmdlRICaptureTaskUpdateBegin, updateStream, task_grp);
  if (ret != ACL_ERROR_NONE) {
    RT_GLOG(EXCEPTION) << "aclmdlRICaptureTaskUpdateBegin failed, ret:" << ret;
  }
#endif
}

void AscendCaptureGraph::CaptureTaskUpdateEnd(void* updateStream) {
  CHECK_IF_NULL(updateStream);
#if defined(__linux__)
  auto ret = CALL_ASCEND_API(aclmdlRICaptureTaskUpdateEnd, updateStream);
  if (ret != ACL_ERROR_NONE) {
    RT_GLOG(EXCEPTION) << "aclmdlRICaptureTaskUpdateEnd failed, ret:" << ret;
  }
#endif
}
} // namespace fxrt::device::ascend
