#ifndef FXRT_ACLNN_OP_API_H
#define FXRT_ACLNN_OP_API_H

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fxrt {
namespace ops {

// The caller supplies already-converted ACLNN arguments in interface order.
// Deduction happens at compile time; symbol lookup cannot validate these types.
// Each DSO instantiates the templates locally, without passing tuples across ABI.
template <typename Tuple, std::size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr, std::index_sequence<I...>) {
  using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

template <typename Function, typename Tuple, std::size_t... I>
auto CallOpApiFunc(Function f, [[maybe_unused]] Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto CallOpApiFunc(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return CallOpApiFunc(f, t, std::make_index_sequence<size>{});
}

} // namespace ops
} // namespace fxrt

#endif // FXRT_ACLNN_OP_API_H
