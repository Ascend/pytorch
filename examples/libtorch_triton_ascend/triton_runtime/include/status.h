#pragma once
#include <string>

namespace triton_runtime {

class Status {
public:
    static Status OK() { return Status(true, ""); }
    static Status Error(const std::string& msg) { return Status(false, msg); }

    bool ok() const { return ok_; }
    const std::string& error_message() const { return msg_; }
    explicit operator bool() const { return ok_; }

private:
    Status(bool ok, std::string msg) : ok_(ok), msg_(std::move(msg)) {}
    bool ok_;
    std::string msg_;
};

} // namespace triton_runtime
