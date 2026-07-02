from torch._inductor.codegen.common import DeviceOpOverrides, register_device_op_overrides
import torch_npu


class NewNPUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        # Importing CATLASS loads the NPU config, which initializes NPU state.
        # Keep it lazy so forked compile workers can import device overrides.
        from torch_npu._inductor.codegen.catlass.catlass_utils import try_import_catlass

        enabled_catlass = try_import_catlass()
        if not enabled_catlass and hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
            return f"from torch_npu._C import _npu_getCurrentRawStreamNoWait as {name}"
        return f"from torch_npu._C import _npu_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch.npu.set_device({device_idx})"

    def synchronize(self):
        return """
                stream = torch.npu.current_stream()
                stream.synchronize()
                """

    def device_guard(self, device_idx):
        return f"torch.npu.utils.device({device_idx})"

    def cpp_aoti_device_guard(self):
        return "AOTINpuGuard"

    def cpp_aoti_stream_guard(self):
        return "AOTINpuStreamGuard"

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
            static std::unordered_map<std::string, std::pair<aclrtBinHandle, aclrtFuncHandle>> registered_kernels;

            static inline bool endsWith(const std::string &value, const std::string &suffix) {
                return value.size() >= suffix.size()
                    && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
            }

            static inline std::string inferKernelModeFromName(const std::string &nameFuncMode) {
                size_t spacePos = nameFuncMode.find(' ');
                if (spacePos != std::string::npos && spacePos + 1 < nameFuncMode.size()) {
                    return nameFuncMode.substr(spacePos + 1);
                }
                if (endsWith(nameFuncMode, "_aiv")) {
                    return "aiv";
                }
                if (endsWith(nameFuncMode, "_aic")) {
                    return "aic";
                }
                return "";
            }

            static inline std::string parseKernelFuncName(
                    const std::string &nameFuncMode,
                    const std::string &kernel_mode) {
                size_t spacePos = nameFuncMode.find(' ');
                if (spacePos != std::string::npos) {
                    return nameFuncMode.substr(0, spacePos);
                }
                if (!kernel_mode.empty()) {
                    const std::string mode_suffix = "_" + kernel_mode;
                    if (endsWith(nameFuncMode, mode_suffix)) {
                        return nameFuncMode.substr(0, nameFuncMode.size() - mode_suffix.size());
                    }
                }
                return nameFuncMode;
            }

            static inline void * loadKernel(
                    std::string filePath,
                    const std::string &&nameFunc,
                    const std::string &&kernel_mode_str,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                std::ifstream file(std::string(filePath), std::ios::binary | std::ios::ate);
                if (!file.is_open()) {
                    throw std::runtime_error(std::string("open npubin failed"));
                }

                std::streamsize data_size = file.tellg();

                file.seekg(0, std::ios::beg);
                auto buffer = std::make_unique<char[]>(data_size);
                if (!file.read(buffer.get(), data_size)) {
                    throw std::runtime_error(std::string("read npubin failed"));
                }

                aclError aclRet;

                uint32_t magic;
                std::string kernel_mode{kernel_mode_str};
                if (kernel_mode.empty()) {
                    kernel_mode = inferKernelModeFromName(nameFunc);
                }
                if (kernel_mode == "aiv") {
                    magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
                } else {
                    magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
                }

                aclrtBinaryLoadOption optArr[] = {
                    { .type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, .value = { .isLazyLoad = 0 } },
                    { .type = ACL_RT_BINARY_LOAD_OPT_MAGIC, .value = { .magic = magic } }
                };
                aclrtBinaryLoadOptions loadOptions = { .options = optArr, .numOpt = 2 };
                aclrtBinHandle binHandle = nullptr;
                aclRet = aclrtBinaryLoadFromData(buffer.get(), data_size, &loadOptions, &binHandle);
                if (aclRet != ACL_SUCCESS) {
                    throw std::runtime_error(std::string("aclrtBinaryLoadFromData failed, 0x") + std::to_string(aclRet));
                }

                std::string kernel_func_name = parseKernelFuncName(nameFunc, kernel_mode);
                const char* name = kernel_func_name.c_str();
                aclrtFuncHandle funcHandle = nullptr;
                aclRet = aclrtBinaryGetFunction(binHandle, name, &funcHandle);
                if (aclRet != ACL_SUCCESS) {
                    throw std::runtime_error(std::string("aclrtBinaryGetFunction failed(name = ") + name
                                + std::string("), 0x") + std::to_string(aclRet));
                }

                registered_kernels[nameFunc] = std::make_pair(binHandle, funcHandle);

                return reinterpret_cast<void *>(funcHandle);
            }

            static inline void * loadKernel(
                    std::string filePath,
                    const std::string &&nameFunc,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                return loadKernel(std::move(filePath), std::move(nameFunc), inferKernelModeFromName(nameFunc),
                                  sharedMemBytes, cubinDir);
            }
        """

        launch_code = """
            static inline void launchKernel(
                    std::function<int()> launch_call,
                    const char* kernel_name) {
                at_npu::native::OpCommand::RunOpApiV2(kernel_name, launch_call);
            }
        """
        extra_code = ""
        source_codes = source_code + load_code + launch_code + extra_code
        return source_codes

    def cpp_stream_type(self):
        return "aclrtStream"

    def aoti_get_stream(self):
        return "aoti_torch_get_current_npu_stream"

    def cpp_kernel_type(self):
        return "void *"

    def cpp_device_ptr(self):
        return "void*"

register_device_op_overrides('npu', NewNPUDeviceOpOverrides())
