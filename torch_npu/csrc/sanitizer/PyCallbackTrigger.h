#pragma once

#include <string>

// Actual implementation
namespace c10_npu {
namespace impl {

struct PyCallbackTrigger;

struct PyCallbackTriggerVTable {
    // Report the name of this interpreter
    virtual std::string name() const = 0;
    virtual ~PyCallbackTriggerVTable() = default;
    virtual void traceNpuAclExecution(std::string acl_name) const = 0;
};

struct PyCallbackTrigger {
    const PyCallbackTriggerVTable* vtable_;

    PyCallbackTrigger(const PyCallbackTriggerVTable* vtable) : vtable_(vtable){};

    const PyCallbackTriggerVTable& operator*() const noexcept
    {
        return *vtable_;
    }
    const PyCallbackTriggerVTable* operator->() const noexcept
    {
        return vtable_;
    }
};

PyCallbackTrigger* getPyCallbackTrigger();

} // namespace impl
} // namespace c10_npu
