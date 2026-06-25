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
            static std::unordered_map<std::string, size_t> registered_names;
            static std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs;

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
                char* buffer = new char[data_size];
                if (!file.read(buffer, data_size)) {
                    throw std::runtime_error(std::string("read npubin failed"));
                }

                rtError_t rtRet;

                rtDevBinary_t devbin;
                devbin.data = buffer;
                devbin.length = data_size;
                std::string kernel_mode{kernel_mode_str};
                if (kernel_mode.empty()) {
                    kernel_mode = inferKernelModeFromName(nameFunc);
                }
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

                std::string kernel_func_name = parseKernelFuncName(nameFunc, kernel_mode);
                const char* name = kernel_func_name.c_str();

                std::string stubName(name);
                stubName += "_" + std::to_string(registered_names[name]);
                registered_names[name]++;
                auto registered = func_stubs.emplace(stubName, std::make_unique<size_t>(0));
                void *func_stub_handle = registered.first->second.get();
                rtRet = rtFunctionRegister(devbinHandle, func_stub_handle, stubName.c_str(),
                                            (void *)name, 0);
                if (rtRet != RT_ERROR_NONE) {
                    throw std::runtime_error(std::string("rtFunctionRegister failed, stubName = ") + stubName
                                + std::string(", name = ") + name
                                + std::string(", original_name_func = ") + nameFunc
                                + std::string(", kernel_mode = ") + kernel_mode
                                + std::string(" , 0x") + std::to_string(rtRet));
                }

                return func_stub_handle;
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

        # Could not use OpCommand when debug_kernel, because we want to
        # use torch::save, which will cause dead lock in child thread.
        launch_code = """
            static inline void launchKernel(
                    std::function<int()> launch_call,
                    const char* kernel_name) {
                at_npu::native::OpCommand cmd;
                cmd.Name(kernel_name)
                    .SetCustomHandler(launch_call)
                    .Run();
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
