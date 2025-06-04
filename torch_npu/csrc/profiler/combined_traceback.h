#pragma once
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/combined_traceback.h>

#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "unwind/unwind.h"

using torch::SymbolizedTracebacks;

namespace torch_npu {

struct TORCH_NPU_API CapturedTraceback : public c10::GatheredContext {
    struct PyFrame {
        void* code; // PyCodeObject*, but python headers not present
        int lasti;
    };

    static std::shared_ptr<CapturedTraceback> gather(bool python, bool script, bool cpp);
    CapturedTraceback() = default;
    CapturedTraceback(const CapturedTraceback&) = delete;
    CapturedTraceback& operator=(const CapturedTraceback&) = delete;
    CapturedTraceback(CapturedTraceback&&) noexcept = default;
    CapturedTraceback& operator=(CapturedTraceback&&) noexcept = delete;
    ~CapturedTraceback() override;

    using visitproc = int (*)(void* self, void* arg);

    struct Python {
        virtual std::vector<PyFrame> gather() = 0;
        virtual void release(std::vector<PyFrame>& frames) = 0;
        virtual void appendSymbolized(const std::vector<PyFrame>& to_symbolize, SymbolizedTracebacks& st) = 0;
        // tp_traverse/tp_clear implementations
        virtual int traverse(std::vector<PyFrame>& frames, visitproc visit, void* arg) = 0;
        virtual int clear(std::vector<PyFrame>& frames) = 0;
        virtual ~Python() = default;
        Python* next_ = nullptr;
    };
    // called once by each python interpreter to
    // register python stack recording functionality
    // p cannot be deleted once added.
    static void addPythonUnwinder(Python* p);

    int traversePython(visitproc visit, void* arg);
    int clearPython();

private:
    std::vector<PyFrame> frames_;
    std::vector<void*> cpp_frames_;
    std::vector<torch::jit::StackEntry> script_frames_;
    friend TORCH_NPU_API SymbolizedTracebacks symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

    // non-owning reference to one of the immortal Python* objects
    // registered above.
    Python* python_ = nullptr;
};

TORCH_NPU_API SymbolizedTracebacks symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

} // namespace torch_npu
