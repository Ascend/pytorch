#pragma once

#include <string>
#include <vector>

namespace fx_converter {

// Resolve Ascend toolkit roots from env + common install fallbacks (shared by OPP / binary selector).
std::vector<std::string> CollectAscendToolkitRoots();

} // namespace fx_converter
