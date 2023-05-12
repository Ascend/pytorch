#ifndef INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_
#define INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_

namespace ge {
#if(defined(HOST_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#if(defined(DEV_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif
#ifdef __GNUC__
#define ATTRIBUTED_DEPRECATED(replacement) __attribute__((deprecated("Please use " #replacement " instead.")))
#else
#define ATTRIBUTED_DEPRECATED(replacement) __declspec(deprecated("Please use " #replacement " instead."))
#endif

using graphStatus = uint32_t;
const graphStatus GRAPH_FAILED = 0xFFFFFFFF;
const graphStatus GRAPH_SUCCESS = 0;
const graphStatus GRAPH_NOT_CHANGED = 1343242304;
const graphStatus GRAPH_PARAM_INVALID = 50331649;
const graphStatus GRAPH_NODE_WITHOUT_CONST_INPUT = 50331648;
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_
