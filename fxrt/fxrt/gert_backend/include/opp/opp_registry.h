#pragma once

#include "base/registry/op_impl_space_registry_v2.h"
#include "register/op_impl_registry.h"

// aclInit + SetDevice (must run before loading OPP host SOs).
bool EnsureAclRuntimeReady(int32_t device_id = 0);

// Load built-in OPP host SOs into a process-wide SpaceRegistry (idempotent).
bool EnsureOppRegistryReady(gert::OpImplSpaceRegistryV2Ptr* out_registry = nullptr);

// Process-stable slot holding the SpaceRegistry raw pointer (for Const).
// Valid after EnsureOppRegistryReady succeeds.
gert::OpImplSpaceRegistryV2** OppSpaceRegistryAddrSlot();

// Resolve prepare/launch for an OPP op type via GetOpImpl(op_type).
bool ResolveOppDualStageFromRegistry(
    const char* opp_type,
    gert::OpImplRegisterV2::OpExecPrepareFunc* prepare,
    gert::OpImplRegisterV2::OpExecLaunchFunc* launch,
    gert::OpImplSpaceRegistryV2Ptr* out_registry = nullptr);
