from ._constant import Constant
from ._singleton import Singleton

__all__ = []


class FeatureName:
    ATTR = "ATTR"


@Singleton
class FeatureManager:
    ALL = "all"
    PTA = "PTA"
    YES = "1"
    No = "0"

    def __init__(self) -> None:
        self.version = "2.1.0"
        self.fmkFeature = {}
        self.cannFeature = {}
        self._init_fmk_feature()

    def load_feature_info(self, feature_info: dict):
        self.cannFeature = feature_info

    def _init_fmk_feature(self):
        self.fmkFeature.setdefault(FeatureName.ATTR, {Constant.FeatureVersion: "0", Constant.Compatibility: self.YES})

    def is_supported_feature(self, feature_name: str) -> bool:
        """
        if component, component_version is different, return false
        if pta version > cann version, check pta compatibility
        if pta version == cann version, return true
        if pta version < cann version, return cann compatibility
        """
        feature = self.cannFeature.get(feature_name, {})
        component = feature.get(Constant.AffectedComponent, "")
        component_version = feature.get(Constant.AffectedComponentVersion, "")
        if component not in {self.ALL, self.PTA}:
            return False
        if component_version not in {self.ALL, self.version}:
            return False
        supported_feature = self.fmkFeature.get(feature_name, {})
        supported_version = supported_feature.get(Constant.FeatureVersion, "0")
        feature_version = feature.get(Constant.FeatureVersion, "0")
        if not supported_version.isdigit() or not feature_version.isdigit():
            return False
        if supported_version > feature_version:
            return supported_feature.get(Constant.Compatibility, self.No) == self.YES
        elif supported_version < feature_version:
            return feature.get(Constant.Compatibility, self.No) == self.YES
        return True

    def clear(self):
        self.cannFeature.clear()