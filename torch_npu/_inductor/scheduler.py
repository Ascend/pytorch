from torch._inductor import dependencies
from torch._inductor import ir
from torch._inductor.ir import NoneLayout
from torch._inductor.scheduler import (
    Scheduler,
    GraphPartitionSignature,
    PartitionType,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


def patch_get_graph_partition_signature():
    """
    Backport upstream PR #165815 to PyTorch 2.9.

    Symptom on 2.9: with graph_partition enabled, the outer Runner.call
    emits `del buf0; del buf1` referencing buffers that only exist inside
    partition_0, raising UnboundLocalError. Failing test:
    test_graph_partition_view_fallback.

    Root cause: a partition's `last_usage` may include buffers allocated
    in *previous* partitions (e.g. via aliasing chains: outer code holds
    buf2 -> buf1 -> buf0). When the next partition (or the outer scope)
    runs, the scheduler tries to free those upstream buffers, but they
    were never returned, so they don't exist in the outer scope.

    Fix: when computing a partition's input signature, also include
    names from buffer_names_to_free that originate from a previous
    partition. This makes them inputs of the current partition (and
    outputs of the previous one), so the outer scope holds variables
    for them and `del` is valid.
    """

    def get_graph_partition_signature(
        self, partitions: list[PartitionType], skip_cudagraphs: list[bool]
    ) -> list[GraphPartitionSignature]:
        """
        Gets signature for each graph partition, including input nodes, output nodes, and
        whether deallocating an input within graph partition.
        """
        signatures = []

        unmet_output_names = OrderedSet(V.graph.get_output_names())
        name_to_node = self.get_name_to_nodes()

        def is_none_layout(buf_name: str) -> bool:
            """
            Checks if buf_name is NoneLayout. Buffers with NoneLayout is not allocated
            so graph partition should not take it as inputs or outputs.
            """
            buf = self.name_to_buf.get(buf_name, None)

            if buf is None:
                return False

            if isinstance(buf.node.layout, NoneLayout):
                if isinstance(buf.node, ir.MutationOutput) and (
                    real_name := self.mutation_real_name.get(buf_name, None)
                ):
                    return is_none_layout(real_name)

                return True

            return False

        for partition, skip_cudagraph in zip(
            reversed(partitions), reversed(skip_cudagraphs)
        ):
            output_names: OrderedSet[str] = OrderedSet()

            for node in partition:
                output_names.update(node.outputs_by_name.keys())

            returned_output_names = output_names.intersection(unmet_output_names)

            # all reads/writes are partition inputs except those generated
            # within the partition and tensor constants
            read_writes = dependencies.ReadWrites.merge_list(
                [node.read_writes for node in partition]
            )

            # WeakDep is fake dependency on unused buffer. It should not appear
            # in partition_input_names for inputs that are actually read or written.
            partition_input_names = (
                OrderedSet(
                    [
                        x.name
                        for x in read_writes.reads | read_writes.writes
                        if not is_none_layout(x.name)
                    ]
                )
                - output_names
            )

            partition_input_names = OrderedSet(
                self.mutation_real_name.get(name, name)
                for name in partition_input_names
            )

            buffer_names_to_free: OrderedSet[str] = OrderedSet()
            for node in partition:
                buffer_names_to_free.update(node.last_usage)

            # PR #165815: buffer_names_to_free may contain buffers
            # allocated in previous graph partitions. Make them inputs of
            # the current partition so they are returned-and-passed and
            # the outer wrapper has variables to del.
            extra_input_names = [
                name
                for name in (buffer_names_to_free - output_names)
                if name in name_to_node
            ]
            partition_input_names.update(extra_input_names)

            input_nodes = {
                name: name_to_node[name]
                for name in partition_input_names
                if name in name_to_node
            }
            input_deallocation = {
                name: True if name in buffer_names_to_free else False
                for name in partition_input_names
                if name in name_to_node
            }

            # if an input tensor is not freed in the partition function, it should
            # also be returned as an output. This brings benefits to cudagraph
            # since the returned output tensor is a cudagraph managed tensor with
            # a static tensor address.
            extra_output_names = [
                name
                for name in partition_input_names
                if name in name_to_node and name not in buffer_names_to_free
            ]

            returned_output_names.update(extra_output_names)

            returned_output_names = OrderedSet(
                self.mutation_real_name.get(name, name)
                for name in returned_output_names
            )

            output_nodes = [
                name_to_node[name]
                for name in returned_output_names
                if not is_none_layout(name)
            ]

            constant_names = [
                name for name in partition_input_names if name in V.graph.constants
            ]

            symbol_inputs = self.get_graph_partition_symbol_inputs(
                partition, input_nodes
            )

            partition_signature = GraphPartitionSignature(
                symbol_inputs,
                input_nodes,
                output_nodes,
                input_deallocation,
                skip_cudagraph,
                constant_names,
            )

            signatures.append(partition_signature)

            unmet_output_names = partition_input_names.union(
                unmet_output_names - returned_output_names
            )

        return signatures[::-1]

    Scheduler.get_graph_partition_signature = get_graph_partition_signature
