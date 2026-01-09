from re import sub

from ..library import CastTypeTag, DataTypeTag, EpilogueOpTag, LayoutTag


class ImplBase:
    def __init__(self, node):
        self.node = node
        self.name = node.name
        self.element = node.metadata.element
        self.shape = node.metadata.shape
        self.layout = node.metadata.layout
        self._type_decl = None

    @property
    def type_name(self):
        return sub(r"(_|-)+", " ", self.name).title().replace(" ", "")

    @property
    def args_decl(self):
        return "{}"

    def make_layout(self):
        shape_str = ", ".join(str(x) for x in self.shape)
        t_name = self.type_name
        layout_name = f"layout{t_name}"
        layout_str = f"""
using LayoutTag{t_name} = {LayoutTag[self.layout]};
LayoutTag{t_name} tag{t_name}{{{shape_str}}};
auto {layout_name} = tla::MakeLayoutFromTag(tag{t_name});
"""
        return layout_name, layout_str


class ComputeImplBase(ImplBase):
    """
    Base class for compute node implementations
    """

    def __init__(self, node):
        super().__init__(node)
        self.fn = self.node.fn
        self.compute_element = node.metadata.element


class CastImplBase(ImplBase):
    """
    Base class for cast node implementations
    """

    def __init__(self, node):
        super().__init__(node)
        self.to_element = node.to_element
        self.from_element = node.from_element
        self.round_type = node.round_type


class ReductionImplBase(ImplBase):
    """
    Base class for reduction node implementations
    """

    def __init__(self, node):
        super().__init__(node)
        self.reduce_fn = self.node.reduce_fn


class NoOpImpl(ImplBase):
    """
    The NoOpImpl does nothing but forward its inputs to users.
    """
    def __init__(self, node):
        super().__init__(node)


class AccLoadImpl(ImplBase):
    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorAccLoad<{DataTypeTag[self.element]}>;
"""
        return self._type_decl


class AuxLoadImpl(ImplBase):
    def __init__(self, node):
        super().__init__(node)
        self.layout_name = None

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self.layout_name, layout_str = self.make_layout()
        self._type_decl = layout_str
        self._type_decl += f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorAuxLoad<
    {DataTypeTag[self.element]}, decltype({self.layout_name})
>;
"""
        return self._type_decl

    @property
    def args_decl(self):
        if not self.layout_name:
            self.layout_name, _ = self.make_layout()
        return f"{{{self.name}_ptr, {self.layout_name}}}"


class AuxStoreImpl(ImplBase):
    def __init__(self, node):
        super().__init__(node)
        self.layout_name = None

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self.layout_name, layout_str = self.make_layout()
        self._type_decl = layout_str
        self._type_decl += f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorAuxStore<
    {DataTypeTag[self.element]}, decltype({self.layout_name})
>;
"""
        return self._type_decl

    @property
    def args_decl(self):
        if not self.layout_name:
            self.layout_name, _ = self.make_layout()
        return f"{{deviceC, {self.layout_name}}}"


class CastImpl(CastImplBase):
    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorCast<
    {DataTypeTag[self.to_element]}, {DataTypeTag[self.from_element]},
    {CastTypeTag[self.round_type]}
>;
"""
        return self._type_decl


class ComputeImpl(ComputeImplBase):
    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorCompute<
    {EpilogueOpTag[self.fn]}, {DataTypeTag[self.compute_element]}
>;
"""
        return self._type_decl


class ScalarComputeImpl(ComputeImplBase):
    def __init__(self, node, values):
        super().__init__(node)
        self.scalar_values = values

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = ""
        for name, item in self.scalar_values.items():
            self._type_decl += f"""
{DataTypeTag[item[1]]} {name} = {item[0]};
"""

        self._type_decl += f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorCompute<
    {EpilogueOpTag[self.fn]}, {DataTypeTag[self.compute_element]},"""
        self._type_decl += ", ".join([DataTypeTag[item[1]] for _, item in self.scalar_values.items()])
        self._type_decl += """
>;
"""
        return self._type_decl

    @property
    def args_decl(self):
        return f"{{{{{', '.join([name for name in self.scalar_values])}}}}}"


class RowBroadcastImpl(ImplBase):
    def __init__(self, node):
        super().__init__(node)
        self.layout_name = None

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self.layout_name, layout_str = self.make_layout()
        self._type_decl = layout_str
        self._type_decl += f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorRowBroadcast<
    {DataTypeTag[self.element]}, decltype({self.layout_name})
>;
"""
        return self._type_decl

    @property
    def args_decl(self):
        if not self.layout_name:
            self.layout_name, _ = self.make_layout()
        return f"{{{self.name}_ptr, {self.layout_name}}}"


class TopoVisitorImpl(ImplBase):
    def __init__(self, node):
        super().__init__(node.output_node)
        self.name = node.name
        self.element = node.output_node.metadata.element
