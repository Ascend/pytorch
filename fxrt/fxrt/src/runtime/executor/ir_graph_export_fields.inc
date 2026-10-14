/**
 * @file ir_graph_export_fields.inc
 * @brief Authoritative key list for the IR export dict. Shared with out-of-tree consumers.
 *
 * This file is an X-macro include, in the same style as ops/op_def/ops.list. It is the
 * single source of truth for the KEY NAMES that GraphExecutor::ExportIrGraph puts into
 * each per-node dict, and it is included by both sides of the boundary:
 *
 *   - producer: ir_graph_export.h (key constants) and python/pybind/fxrt/pybind_ir.cpp
 *     (the dict writer, which is checked against this list);
 *   - consumer: the out-of-tree converter's reader, which includes this very file out of
 *     a torch_npu source tree.
 *
 * It deliberately lists key names and not struct fields. The two sides' DTOs are allowed
 * to differ in field type -- the producer carries `kind` as a string while a consumer may
 * decode it into an enum -- and each struct is private to its own translation unit. What
 * must never drift is the set and spelling of the keys, because a key the producer stops
 * emitting does not fail on the consumer: it reads back as "unset". That is exactly how
 * stride / storage_shape / storage_offset / sym_list came to be silently dropped before
 * schema version 2.
 *
 * IR_EXPORT_FIELD(key, always_emitted)
 *   always_emitted = 1  the key is present on every node dict
 *   always_emitted = 0  the key is omitted when it carries no information; a consumer
 *                       must treat absence as the documented default, never as an error
 *
 * Adding, removing or renaming a line here is a cross-repository contract change. Bump
 * kIrExportSchemaVersion in ir_graph_export.h in the same commit and update the golden
 * expectations in test/fxrt/.
 */

// Identity and topology.
IR_EXPORT_FIELD(name, 1)
IR_EXPORT_FIELD(kind, 1)
IR_EXPORT_FIELD(op, 1)
IR_EXPORT_FIELD(inputs, 1)

// Output layout. shape/sym_shape/dtype are always present; the three layout keys are
// emitted only for non-dense values, so their absence means "dense, offset 0".
IR_EXPORT_FIELD(shape, 1)
IR_EXPORT_FIELD(stride, 0)
IR_EXPORT_FIELD(storage_shape, 0)
IR_EXPORT_FIELD(storage_offset, 0)
IR_EXPORT_FIELD(sym_shape, 1)
IR_EXPORT_FIELD(dtype, 1)

// Scalar-ish operands lifted out of the tensor input list.
IR_EXPORT_FIELD(axis, 0)
IR_EXPORT_FIELD(scalar, 0)

// Int list plus its index-aligned symbol channel. sym_list is only emitted when at least
// one slot is symbolic; an empty entry in it means "that slot's int_list value is real".
IR_EXPORT_FIELD(int_list, 0)
IR_EXPORT_FIELD(sym_list, 0)
IR_EXPORT_FIELD(bool_list, 0)

// Multi-output ops: per-output shape and dtype.
IR_EXPORT_FIELD(output_shapes, 0)
IR_EXPORT_FIELD(output_dtypes, 0)

// Recursive compile-time metadata that the normalized tensor-only `inputs` cannot carry.
IR_EXPORT_FIELD(arguments, 0)
IR_EXPORT_FIELD(attrs, 0)
