FILE(GLOB _ATEN_SRCS
    *.cpp
    common/*.cpp
    mirror/*.cpp
    ops/*.cpp
    ops/op_api/*.cpp)

FILE(GLOB _EXCLUDE
    VariableTypeEverything.cpp
    ADInplaceOrViewTypeEverything.cpp
    python_functionsEverything.cpp
    RegisterFunctionalizationEverything.cpp)

FOREACH(ITEM ${_EXCLUDE})
    LIST(REMOVE_ITEM _ATEN_SRCS ${ITEM})
ENDFOREACH()

LIST(APPEND ATEN_SRCS ${_ATEN_SRCS})

# Pass to parent
set(ATEN_SRCS ${ATEN_SRCS} PARENT_SCOPE)
