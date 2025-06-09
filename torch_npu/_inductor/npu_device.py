import torch
from torch_npu.npu import device_count
from torch_npu.utils._dynamo_device import NpuInterface, current_device, set_device
from torch_npu.utils._inductor import NPUDeviceOpOverrides
from . import config as npu_config


## Override original inductor device overrides in torch_npu
class NewNPUDeviceOpOverrides(NPUDeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch_npu._inductor import get_current_raw_stream as {name}"

    def set_device(self, device_idx):
        return f"torch.npu.set_device({device_idx})"

    def synchronize(self):
        return """
                stream = torch.npu.current_stream()
                stream.synchronize()
                """

    def device_guard(self, device_idx):
        return f"torch.npu._DeviceGuard({device_idx})"

    def cpp_aoti_device_guard(self):
        raise NotImplementedError

    def cpp_aoti_stream_guard(self):
        return "AOTICudaStreamGuard"

    def kernel_driver(self):
        source_code = """
            namespace {

            struct Grid {
                Grid(uint32_t x, uint32_t y, uint32_t z)
                  : grid_x(x), grid_y(y), grid_z(z) {}
                uint32_t grid_x;
                uint32_t grid_y;
                uint32_t grid_z;

                bool is_non_zero() {
                    return grid_x > 0 && grid_y > 0 && grid_z > 0;
                }
            };

            }  // anonymous namespace

            extern "C" {
                typedef int (* callback)(unsigned int type, void* data, unsigned int len);
                extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
                extern unsigned long int  MsprofSysCycleTime();
                extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
                static unsigned int __MsprofFlagL0  = 0;
                static unsigned int __MsprofFlagL1  = 0;

                int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
                    if ((CtrlData == nullptr) || (DataLen == 0U)) {
                        return 1;
                    }

                    if (CtrlType == 1) {
                        MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
                        if (handle->type >= 6)  // 6 is not used here
                            return 1;
                        if (handle->type == 1) {  // init - 0  , start - 1
                            __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
                            __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
                        }
                    }
                    return 0;
                }
            }
        """

        load_code = """
            static std::unordered_map<std::string, size_t> registered_names;
            static std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs;
            
            static inline void * loadKernel(
                    std::string filePath,
                    const std::string &&nameFuncMode,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }
                std::string funcName;
                std::string kernel_mode_str;
                size_t spacePos = nameFuncMode.find(' ');
                if (spacePos != std::string::npos) {
                    kernel_mode_str = nameFuncMode.substr(spacePos + 1);
                    funcName = nameFuncMode.substr(0, spacePos);
                } else {
                    throw std::runtime_error(std::string("Parse kernel name failed, expect "
                                                        "'kernel_name kernel_mode', bug got: ") + nameFuncMode);
                }

                std::ifstream file(std::string(filePath), std::ios::binary | std::ios::ate);
                if (!file.is_open()) {
                    throw std::runtime_error(std::string("open npubin failed"));
                }

                std::streamsize data_size = file.tellg();

                file.seekg(0, std::ios::beg);
                char* buffer = new char[data_size];
                if (!file.read(buffer, data_size)) {
                    throw std::runtime_error(std::string("read npubin failed"));
                }

                rtError_t rtRet;

                rtDevBinary_t devbin;
                devbin.data = buffer;
                devbin.length = data_size;
                const std::string kernel_mode{kernel_mode_str};
                if (kernel_mode == "aiv") {
                    devbin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
                } else {
                    devbin.magic = RT_DEV_BINARY_MAGIC_ELF;
                }
                devbin.version = 0;

                int device = 0;
                rtRet = rtSetDevice(device);
                if (rtRet != RT_ERROR_NONE) {
                    throw std::runtime_error(std::string("rtSetDevice failed, 0x") + std::to_string(rtRet));
                }

                void *devbinHandle = NULL;
                rtRet = rtDevBinaryRegister(&devbin, &devbinHandle);
                if (rtRet != RT_ERROR_NONE) {
                    throw std::runtime_error(std::string("rtDevBinaryRegister failed, 0x") + std::to_string(rtRet));
                }

                const char* name = funcName.c_str();

                std::string stubName(name);
                stubName += "_" + std::to_string(registered_names[name]);
                registered_names[name]++;
                auto registered = func_stubs.emplace(stubName, std::make_unique<size_t>(0));
                void *func_stub_handle = registered.first->second.get();
                rtRet = rtFunctionRegister(devbinHandle, func_stub_handle, stubName.c_str(),
                                            (void *)name, 0);
                if (rtRet != RT_ERROR_NONE) {
                    throw std::runtime_error(std::string("rtFunctionRegister failed, stubName = ") + stubName
                                + std::string(" , 0x") + std::to_string(rtRet));
                }

                return func_stub_handle;
            }
        """

        # Could not use OpCommand when debug_kernel, because we want to
        # use torch::save, which will cause dead lock in child thread.
        launch_code = """
            static inline void launchKernel(
                    std::function<int()> launch_call,
                    std::string&& kernel_name) {
                launch_call();
            }
        """ if npu_config.aot_inductor.debug_kernel else """
            static inline void launchKernel(
                    std::function<int()> launch_call,
                    std::string&& kernel_name) {
                at_npu::native::OpCommand cmd;
                cmd.Name(kernel_name.c_str())
                    .SetCustomHandler(launch_call)
                    .Run();
            }
        """
        extra_code = ""
        source_codes = source_code + load_code + launch_code + extra_code
        return source_codes

    def abi_compatible_header(self):
        return """
            #include <fstream>
            #include <vector>
            #include <iostream>
            #include <string>
            #include <tuple>
            #include <unordered_map>
            #include <memory>
            #include <filesystem>

            #include <assert.h>
            #include <stdbool.h>
            #include <sys/syscall.h>
            #include <torch_npu/csrc/framework/OpCommand.h>
            #include <torch_npu/csrc/core/npu/NPUStream.h>
            #include "experiment/runtime/runtime/rt.h"
        """

    def cpp_stream_type(self):
        return "aclrtStream"

    def aoti_get_stream(self):
        return "aoti_torch_get_current_cuda_stream"

    def cpp_kernel_type(self):
        return "void *"

    def cpp_device_ptr(self):
        return "void*"


## Override original dynamo device interface in torch_npu
class NewNpuInterface(NpuInterface):

    @staticmethod
    def is_available() -> bool:
        return device_count() > 0

    @staticmethod
    def get_compute_capability(device=None):
        # npu has no concept of cc. triton-npu compiler depends on subarch instead
        return torch.npu.get_device_name(device)

    @staticmethod
    def exchange_device(device: int) -> int:
        curr_device = current_device()
        set_device(device)
        return curr_device

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        return device

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False):
        return True