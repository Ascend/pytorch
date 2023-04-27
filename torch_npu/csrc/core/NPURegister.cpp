#include <c10/core/DeviceType.h>

namespace {

int register_npu() {
    c10::register_privateuse1_backend("npu");
    return 0;
}

int _ = register_npu();

}
