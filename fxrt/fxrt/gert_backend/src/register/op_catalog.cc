#include "register/op_catalog.h"

#include <algorithm>
#include <string>
#include <vector>

#include "lowering/op_lowering_registry.h"
#include "register/op_catalog_register.h"
#include "ops/op_def/ops_name.h"

namespace fx_converter {

size_t CatalogCsvCount(const char *csv) {
  if (csv == nullptr || csv[0] == '\0') {
    return 0;
  }
  size_t n = 1;
  for (const char *p = csv; *p != '\0'; ++p) {
    if (*p == ',') {
      ++n;
    }
  }
  return n;
}

std::string CatalogCsvField(const char *csv, size_t index) {
  if (csv == nullptr) {
    return "";
  }
  size_t cur = 0;
  const char *begin = csv;
  for (const char *p = csv;; ++p) {
    if (*p != ',' && *p != '\0') {
      continue;
    }
    if (cur == index) {
      return std::string(begin, static_cast<size_t>(p - begin));
    }
    if (*p == '\0') {
      break;
    }
    ++cur;
    begin = p + 1;
  }
  return "";
}

AiCoreOptionalLayout DescribeAiCoreOptionalInputs(const OpCatalogEntry &entry) {
  AiCoreOptionalLayout layout;
  layout.pad_names = entry.launch_pad_inputs != nullptr ? entry.launch_pad_inputs : entry.optional_inputs;
  layout.required = CatalogCsvCount(entry.opp_input_names);
  layout.has_optional = layout.pad_names != nullptr;
  layout.optional_total = layout.has_optional ? CatalogCsvCount(layout.pad_names) : 0;
  return layout;
}

bool BindAiCoreOptionalConnected(AiCoreOptionalLayout *layout, size_t connected, const char *op_name,
                                 std::string *err) {
  if (layout == nullptr) {
    return false;
  }
  layout->pad = 0;
  if (!layout->has_optional) {
    return true;
  }
  const size_t upper = layout->required + layout->optional_total;
  if (connected < layout->required || connected > upper) {
    if (err != nullptr) {
      const char *name = (op_name != nullptr) ? op_name : "";
      *err = "aicore-binary lowering: optional input count " + std::to_string(connected) + " outside [" +
             std::to_string(layout->required) + "," + std::to_string(upper) + "] for '" + name + "'";
    }
    return false;
  }
  layout->pad = upper - connected;
  return true;
}

size_t CatalogOutputArity(const OpCatalogEntry &entry) {
  const size_t n = CatalogCsvCount(entry.opp_output_names);
  return n == 0 ? 1 : n;
}

int32_t CatalogRefInput(const OpCatalogEntry &entry, size_t output_index) {
  if (entry.output_refs == nullptr || entry.output_ref_count == 0) {
    return -1;
  }
  const OutputRef *begin = entry.output_refs;
  const OutputRef *end = begin + entry.output_ref_count;
  const OutputRef *found = std::find_if(begin, end, [output_index](const OutputRef &ref) {
    return ref.output >= 0 && static_cast<size_t>(ref.output) == output_index;
  });
  return found == end ? -1 : found->input;
}

std::string CatalogShapeInputName(const char *opp_input_names, int32_t shape_input_index) {
  if (shape_input_index < 0) {
    return "shape";
  }
  const std::string name = CatalogCsvField(opp_input_names, static_cast<size_t>(shape_input_index));
  return name.empty() ? "shape" : name;
}

std::string CatalogShapeInputName(const OpCatalogEntry &entry) {
  return CatalogShapeInputName(entry.opp_input_names, entry.shape_input_index);
}

const OpCatalogEntry *LookupOpCatalog(const fxrt::ir::NodePtr &node) {
  if (node == nullptr || node->op == fxrt::ops::Op_custom_call || node->op == fxrt::ops::Op_End ||
      node->op == fxrt::ops::Op_return) {
    return nullptr;
  }
  const std::string logical = fxrt::ops::ToStr(node->op);
  const OpCatalogEntry *entry = CatalogRegistry::Instance().Find(logical);
  if (entry == nullptr) {
    return nullptr;
  }
  // Path C: OPP arity is after Collect (e.g. getitem_slice IR has x+packed Tuple=2,
  // expands to x+offsets+size=3). Only require a non-empty IR input list here.
  if (entry->lowering == LoweringKind::kAiCoreBinary) {
    if (node->inputs.empty()) {
      return nullptr;
    }
    return entry;
  }
  const size_t need = (entry->arity == 0) ? 1 : entry->arity;
  if (node->inputs.size() < need) {
    return nullptr;
  }
  return entry;
}

std::vector<std::string> ListConverterSupportedLogicalOps() {
  std::vector<std::string> names;
  names.reserve(64U);
  // Overlay rows replace the same-named builtin row, so the registry pass covers both.
  // Dedicated lowerings add the non-catalog names they handle.
  CatalogRegistry::Instance().AppendLogicalNames(&names);
  OpLoweringRegistry::Instance().AppendSupportedLogicalOps(&names);
  return names;
}

}  // namespace fx_converter
