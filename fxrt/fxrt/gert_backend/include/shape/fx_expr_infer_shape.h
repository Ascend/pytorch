#pragma once

#include "register/op_impl_registry.h"

// Generic InferShape: read FxShapeRule* from input slot (before func), bind + eval fxrt SymbolicExpr.
gert::OpImplRegisterV2::InferShapeKernelFunc GetFxExprInferShapeFunc();
