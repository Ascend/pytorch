import functools
import itertools
import logging
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import patch

import torch

from torch._inductor.autotune_process import TensorMeta
from torch._inductor.ir import Buffer, IRNode, Layout
from torch._inductor.utils import IndentedBuffer, Placeholder, unique
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import KernelTemplate

from .catlass_kernel import (
    CATLASSTemplateCaller,
    CATLASSTemplateKernel,
    CATLASSTemplateBuffer,
)
from ...autotune_process import CATLASSBenchmarkRequest


log = logging.getLogger("torch._inductor")


class CATLASSTemplate(KernelTemplate):
    """
    CATLASSTemplate is a class that provides a template for generating CATLASS Templates. Used as a baseclass for the
    CATLASSGemmTemplate, providing functionality that might also be relevant for non-GEMM CATLASS Kernels.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: List[Buffer],
        layout: Layout,
        input_reorder: Optional[List[int]] = None,
    ):
        super().__init__(name)
        self.input_nodes = input_nodes
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)
        self.input_reorder = input_reorder
        self.layout = layout

    @staticmethod
    def is_mixed_template(op: "GemmKernelBase") -> bool:
        return False

    @staticmethod
    def epilogue_fusion_type(op: "GemmKernelBase") -> int:
        return 0

    def generate(  # type: ignore[override]
        self,
        description,
        **kwargs,
    ) -> CATLASSTemplateCaller:
        """
        Generates the CATLASS template caller object for the given GEMM template and operation. This CATLASSTemplateCaller
        may be used to call and benchmark the generated CATLASS kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A CATLASSTemplateCaller object representing the generated CATLASS template caller.
        """
        kernel_name = self.name
        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(self.output_node)
        ), CATLASSTemplateKernel(
            kernel_name=kernel_name,
        ) as kernel:
            code = self.render(kernel=kernel, **kwargs)
            _, call_args, _, _ = kernel.args.python_argdefs()
            log.debug("Generated Code:\n%s", code)
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )

        input_reorder = (
            self.input_reorder
            if self.input_reorder is not None
            else list(range(len(self.input_nodes)))
        )

        expected_args = list(
            unique(self.input_nodes[idx].get_name() for idx in input_reorder)
        )
        expected_args.extend([self.output_node.get_name()])
        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )
        size_args = V.graph.sizevars.size_hints(kernel.get_layout_args())

        kernel_hash_name = f"{self.name}_{next(self.index_counter)}"

        # kwargs has "op" argument in case of CATLASSGemmTemplate
        op = kwargs["op"]
        if not op:
            is_mix = False
            epilogue_fusion_type = 0
        else:
            is_mix = self.is_mixed_template(op)
            epilogue_fusion_type = self.epilogue_fusion_type(op)

        # create the BenchmarkRequest
        bmreq = CATLASSBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=size_args,
            source_code=code,
            is_mix=is_mix,
        )

        def make_kernel_render(
            template_node: CATLASSTemplateBuffer,
            epilogue_nodes: Optional[List[IRNode]] = None,
        ):
            assert epilogue_fusion_type or not epilogue_nodes, (
                "epilogue fusion is not supported for this kernel"
            )
            kernel = CATLASSTemplateKernel(
                kernel_name="KERNEL_NAME",
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CATLASSGemmTemplate
            )
            return kernel, render

        return CATLASSTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            epilogue_fusion_type,
            self,
            is_mix,
            kwargs,
            description,
        )

    def header(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                #include <exception>
                #include <iostream>
                #include <memory>
                #include <random>
                #include <vector>

                #include <acl/acl.h>
                #include <runtime/rt_ffts.h>
                #include <tiling/platform/platform_ascendc.h>
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = IndentedBuffer()
        res.splice(
            """
                // We compile all models with -fvisibility=hidden. Any symbols that need to be
                // exposed in the final shared library must be declared with PT_EXPORT to make
                // them visible.
                #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
                #define PT_EXPORT __attribute__((__visibility__("default")))
                #else
                #ifdef _WIN32
                #define PT_EXPORT __declspec(dllexport)
                #else
                #define PT_EXPORT
                #endif
                #endif

                #define ACL_CHECK(status)                                                                    \\
                    do {                                                                                     \\
                        aclError error = status;                                                             \\
                        if (error != ACL_ERROR_NONE) {                                                       \\
                            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \\
                        }                                                                                    \\
                    } while (0)

                // Macro function for unwinding rt errors.
                #define RT_CHECK(status)                                                                     \\
                    do {                                                                                     \\
                        rtError_t error = status;                                                            \\
                        if (error != RT_ERROR_NONE) {                                                        \\
                            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;   \\
                        }                                                                                    \\
                    } while (0)

                using namespace Catlass;
                using namespace tla;

                // Macro function for unwinding catlass errors.
                #define CATLASS_CHECK(status)                                                                \\
                    do {                                                                                     \\
                        Catlass::Status error = status;                                                      \\
                        if (error != Catlass::Status::kSuccess) {                                            \\
                            std::cerr << __FILE__ << ":" << __LINE__ << " raise catlassError" << std::endl;  \\
                        }                                                                                    \\
                    } while (0)

            """
        )
        return res


    def catlass_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"(uint8_t*)({ptr})"
    

    def render(self, **kwargs) -> str:
        raise NotImplementedError
