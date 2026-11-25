#include "common/ascend_paths.h"

#include <cstdlib>

namespace fx_converter {

std::vector<std::string> CollectAscendToolkitRoots() {
  std::vector<std::string> roots;
  const char *env_keys[] = {"ASCEND_HOME_PATH", "ASCEND_HOME", "ASCEND_TOOLKIT_HOME"};
  for (const char *key : env_keys) {
    const char *v = std::getenv(key);
    if (v != nullptr && v[0] != '\0') {
      roots.emplace_back(v);
    }
  }
  // Toolkit "latest" symlink is portable; prefer it over versioned installs.
  roots.emplace_back("/usr/local/Ascend/ascend-toolkit/latest");
  // TODO(jiaorui): versioned CANN install paths are machine/env specific; remove once
  // ASCEND_HOME_PATH is required in all deploy/CI environments.
  roots.emplace_back("/usr/local/Ascend/cann-9.1.0-beta.3");
  roots.emplace_back("/usr/local/Ascend/cann");
  return roots;
}

}  // namespace fx_converter
