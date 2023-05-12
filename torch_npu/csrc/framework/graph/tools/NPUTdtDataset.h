#pragma once

#include <string>
#include <memory>
#include <tuple>

#include "torch_npu/csrc/framework/interface/AclTdtInterface.h"
namespace c10_npu {
class TdtDataSet {
public:
  TdtDataSet() {
    dataset_ = std::shared_ptr<acltdtDataset>(acl_tdt::AcltdtCreateDataset(),
                                              [](acltdtDataset* item) {
                                                acl_tdt::AcltdtDestroyDataset(item);
                                              });
  }
  std::shared_ptr<acltdtDataset> GetPtr() const {
    return dataset_;
  }
private:
  std::shared_ptr<acltdtDataset> dataset_ = nullptr;
};
} // namespace c10_npu

