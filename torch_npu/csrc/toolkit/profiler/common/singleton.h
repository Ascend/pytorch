#pragma once

#include <memory>

namespace torch_npu {
namespace toolkit {
namespace profiler {
template<typename T>
class Singleton {
public:
    static T *GetInstance() noexcept(std::is_nothrow_constructible<T>::value)
    {
        static T instance;
        return &instance;
    }

    virtual ~Singleton() = default;

protected:
    explicit Singleton() = default;

private:
    explicit Singleton(const Singleton &obj) = delete;
    Singleton &operator = (const Singleton &obj) = delete;
    explicit Singleton(Singleton &&obj) = delete;
    Singleton &operator = (Singleton &&obj) = delete;
};
} // profiler
} // toolkit
} // torch_npu
