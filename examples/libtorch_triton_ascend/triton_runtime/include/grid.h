#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <variant>

namespace triton_runtime {

// Wraps ArgInfo::ScalarVariant with implicit conversion, so the lambda can do
//   int n = meta["n_elements"];   // operator int()
struct ScalarValue {
    using Var = std::variant<int32_t, float, bool>;
    Var v;

    ScalarValue() = default;
    ScalarValue(const Var& v) : v(v) {}
    ScalarValue(int32_t x) : v(x) {}
    ScalarValue(float x) : v(x) {}
    ScalarValue(bool x) : v(x) {}

    operator int()   const { return std::visit([](auto x) { return int(x); }, v); }
    operator float() const { return std::visit([](auto x) { return float(x); }, v); }
    operator bool()  const { return std::visit([](auto x) { return bool(x); }, v); }
};

using BoundArgs = std::unordered_map<std::string, ScalarValue>;

struct Grid;
using GridFn = std::function<Grid(BoundArgs&)>;

// Grid is either a fixed (x,y,z) or a callable that resolves (x,y,z)
// from bound_args. Both modes live in the same type — no extra wrapper.
struct Grid {
    int x = 1, y = 1, z = 1;
    GridFn fn;

    Grid() = default;
    Grid(int x, int y = 1, int z = 1) : x(x), y(y), z(z) {}
    Grid(GridFn f) : fn(std::move(f)) {}

    bool is_callable() const { return !!fn; }

    Grid resolve(BoundArgs& args) const {
        if (fn) return fn(args);
        return *this;
    }
};

} // namespace triton_runtime
