import os

os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"

from unittest.mock import patch, MagicMock, call, ANY
import weakref
import pytest
import torch
import torch_npu
from torch_npu.npu._graph_tree import (
    check_memory_pool,
    clear_cublass_cache,
    clear_cublas_manager,
    disable_conv_cache_emptying,
    enable_history_recording,
    format_tb,
    npugraphify,
    npugraphify_impl,
    TreeManagerContainer,
    StorageWeakRefWrapper,
    NPUWarmupNode,
    CompilationMode,
    get_container,
    get_block_addrs,
    get_manager,
    get_npugraph_segments,
    reset_npugraph_trees,
    local,
    OutputAliasInfo,
    UnaliasedStorage,
    AliasesPriorGraphOutput,
    AliasesNewOutput,
    NPUGraphNode,
    WrappedFunction,
    NPUGraphTreeManager,
    ExecutionState,
    FunctionID,
    GraphID,
)
from torch_npu.testing.testcase import TestCase, run_tests


device = "npu:0"
torch.npu.set_device(device)


class TestCublasCacheManagement(TestCase):
    @patch("torch_npu.npu._graph_tree.clear_cublass_cache")
    def test_clear_cublas_manager_context(self, mock_clear):
        with clear_cublas_manager():
            mock_clear.assert_called_once()
            mock_clear.reset_mock()
        mock_clear.assert_called_once()


class TestDisableConvCache(TestCase):
    def test_disable_conv_cache_emptying(self):
        with disable_conv_cache_emptying():
            pass  # No operation, just ensure no exceptions


class TestHistoryRecording(TestCase):
    @patch("torch.npu.memory._record_memory_history")
    def test_enable_history_recording(self, mock_record):
        original_state = torch_npu._C._npu_isHistoryEnabled()
        with enable_history_recording():
            if not original_state:
                mock_record.assert_called_once()
            else:
                mock_record.assert_not_called()
        mock_record.assert_any_call(None)


class TestNpuGraphFunctions(TestCase):
    def setUp(self):
        # Reset global state before each test
        reset_npugraph_trees()

    @patch("torch_npu.npu._graph_tree.TreeManagerContainer")
    def test_get_manager(self, mock_container):
        # Test manager creation
        mock_container.return_value.get_tree_manager.return_value = "mock_manager"
        manager = get_manager(0)
        self.assertEqual(manager, "mock_manager")

        # Test no-creation path
        manager = get_manager(0, create_if_none_exists=False)
        mock_container.return_value.get_tree_manager.assert_called_once()

    @patch("torch_npu.npu._graph_tree.npugraphify")
    @patch("torch._inductor.compile_fx.align_inputs_from_check_idxs")
    def test_npugraphify_impl(self, mock_align, mock_npugraphify):
        # Setup mock model and inputs
        mock_model = MagicMock()
        inputs = [1, torch.tensor([2]), 3]
        static_idxs = (1,)

        # Test caching behavior
        impl = npugraphify_impl(mock_model, inputs, static_idxs)

        # First call
        mock_npugraphify.return_value = (lambda x: "output1", "output1")
        result = impl(inputs)
        self.assertEqual(result, "output1")

        # Second call with same int keys
        result = impl(inputs)
        self.assertEqual(result, "output1")
        mock_npugraphify.assert_called_once()

    @patch("torch_npu.npu._graph_tree.get_container")
    def test_npugraphify(self, mock_container):
        # Setup mock manager
        mock_manager = MagicMock()
        mock_container.return_value.get_tree_manager.return_value = mock_manager

        # Test valid mode combinations
        model = MagicMock()
        inputs = [torch.tensor([1])]

        # Test forward mode
        npugraphify(
            model, inputs, (), device_index=0, is_backward=False, is_inference=False
        )
        mock_manager.add_function.assert_called_with(
            model, inputs, (), None, CompilationMode.FORWARD, (),
        )

        # Test backward mode
        mock_manager.reset_mock()
        npugraphify(
            model, inputs, (), device_index=0, is_backward=True, is_inference=False
        )
        mock_manager.add_function.assert_called_with(
            model, inputs, (), None, CompilationMode.BACKWARD, (),
        )

        # Test invalid mode combination
        with self.assertRaises(RuntimeError):
            npugraphify(
                model, inputs, (), device_index=0, is_backward=True, is_inference=True
            )


class TestTreeManagerContainer(TestCase):
    def setUp(self):
        self.container = TreeManagerContainer(0)

    def test_initial_state(self):
        self.assertIsNone(self.container.tree_manager)
        self.assertEqual(self.container.live_npugraphify_fns, 0)

    def test_add_strong_reference(self):
        self.container.add_strong_reference(lambda: None)
        # Simulate finalization of fn
        finalizer = weakref.finalize(
            lambda: None,
            self.container.finalize_npugraphify_fn,  # Object to monitor  # Callback
        )
        finalizer.atexit = False  # Prevent finalizer from running at exit

        # Simulate finalization
        finalizer()
        # If all references are gone, tree_manager should be None
        self.container._finalize_tree_manager = MagicMock()
        self.container._finalize_tree_manager()
        self.container._finalize_tree_manager.assert_called_once()

    def test_get_tree_manager(self):
        with patch("torch_npu.npu.graphs.NPUGraph.capture_begin"), patch(
            "torch_npu.npu.graphs.NPUGraph.capture_end"
        ):
            manager = self.container.get_tree_manager()
        self.assertIsNotNone(manager)
        self.assertIs(manager, self.container.get_tree_manager())  # Same instance


class TestStorageWeakRefWrapper(TestCase):
    def test_storage_ref(self):
        tensor = torch.tensor([1], device="npu")
        wrapper = StorageWeakRefWrapper(tensor)
        self.assertEqual(wrapper.data_ptr(), tensor.untyped_storage().data_ptr())
        del tensor
        # Storage might still be alive due to Python's ref counting; force GC
        import gc

        gc.collect()
        self.assertTrue(wrapper.expired())


class TestNPUWarmupNode(TestCase):
    @patch("torch_npu.npu._graph_tree.StorageWeakRefWrapper")
    @patch("torch_npu.npu._graph_tree.check_memory_pool")
    def test_run_captures_outputs(self, mock_check, mock_wrapper):
        mock_model = MagicMock(return_value=[torch.tensor([2], device="npu")])
        wrapped_fn = MagicMock(model=mock_model, constants=[])
        stream = torch.npu.Stream()
        node = NPUWarmupNode(
            wrapped_fn,
            parent=None,
            npu_graphs_pool=(0, 0),
            existing_npu_graph=None,
            device_index=0,
            stack_traces=None,
            stream=stream,
            already_warm=False,
        )
        outputs = node.run([])
        self.assertEqual(len(node.outputs_weakrefs), 1)


class TestTreeManagerIntegration(TestCase):
    def test_get_container_singleton_per_device(self):
        container1 = get_container(0)
        container2 = get_container(0)
        self.assertIs(container1, container2)
        container3 = get_container(1)
        self.assertIsNot(container1, container3)

    def test_reset_npugraph_trees(self):
        get_container(0)  # Initialize a container
        reset_npugraph_trees()
        container_dict = getattr(local, "tree_manager_containers", {})
        self.assertEqual(len(container_dict), 0)


@pytest.fixture
def mock_wrapped_function():
    def model_side_effect(inputs):
        # Clear inputs list while preserving reference
        inputs[:] = []
        return []

    return MagicMock(
        spec=WrappedFunction,
        static_input_idxs=[0],
        constants=[],
        model=MagicMock(side_effect=model_side_effect),
    )


@pytest.fixture
def mock_parent_node():
    parent = MagicMock(spec=NPUGraphNode)
    parent.outputs_weakrefs = []
    parent.path_weakrefs = []
    parent.parent = None
    parent.stack_traces = []
    parent.recorded_liveness_after_graph = []
    return parent


@pytest.fixture
def basic_npu_graph_node(mock_wrapped_function, mock_parent_node):
    with patch("torch_npu.npu._graph_tree._use_npu_memory_pool_manager"), patch(
        "torch_npu.npu._graph_tree.check_memory_pool"
    ), patch("torch_npu._C._npu_getCheckpointState"):
        return NPUGraphNode(
            wrapped_function=mock_wrapped_function,
            graph_id=1,
            parent=mock_parent_node,
            inputs=[torch.tensor([1.0], device="npu")],
            npu_graphs_pool=(0, 0),
            device_index=0,
            stack_traces=None,
            stream=torch.npu.Stream(),
        )


class TestOutputAliasInfo(TestCase):
    def test_aliases_prior_graph_output_validation(self):
        with pytest.raises(RuntimeError):
            AliasesPriorGraphOutput("invalid_index")

    def test_aliases_new_output_validation(self):
        with pytest.raises(RuntimeError):
            AliasesNewOutput("not_an_int")


class TestNPUGraphNode:
    def tearDown(self):
        torch_npu._C._npu_endAllocateCurrentStreamToPool(0, (0, 0))
        torch_npu._C._npu_releasePool(0, (0, 0))

    def test_initialization(self, mock_wrapped_function, mock_parent_node):
        inputs = [torch.tensor([1.0], device="npu")]
        with patch("torch_npu.npu._graph_tree._use_npu_memory_pool_manager"), patch(
            "torch_npu.npu._graph_tree.check_memory_pool"
        ), patch("torch_npu._C._npu_getCheckpointState"):
            node = NPUGraphNode(
                wrapped_function=mock_wrapped_function,
                graph_id=1,
                parent=mock_parent_node,
                inputs=inputs,
                npu_graphs_pool=(0, 0),
                device_index=0,
                stack_traces=None,
                stream=torch.npu.Stream(),
            )

        assert node.id == 1
        assert node.device == 0
        assert node.parent == mock_parent_node
        assert node.graph is not None

    def test_invalid_input_type(self, mock_wrapped_function):
        with pytest.raises(RuntimeError):
            NPUGraphNode(
                wrapped_function=mock_wrapped_function,
                graph_id=1,
                parent=None,
                inputs="not_a_list",
                npu_graphs_pool=(0, 0),
                device_index=0,
                stack_traces=None,
                stream=torch.npu.Stream(),
            )

    @patch("torch_npu.npu._graph_tree.check_memory_pool")
    def test_record_method(self, mock_check, basic_npu_graph_node):
        def model_side_effect(inputs):
            # Clear inputs list while preserving reference
            inputs[:] = []
            return []

        mock_model = MagicMock(side_effect=model_side_effect)
        mock_inputs = [torch.tensor([1.0], device="npu")]

        with patch("torch_npu.npu._graph_tree.clear_cublas_manager"), patch(
            "torch_npu.npu._graph_tree.get_history_recording"
        ), patch("torch_npu.npu.graphs.NPUGraph.capture_begin"), patch(
            "torch_npu.npu.graphs.NPUGraph.capture_end"
        ), patch(
            "torch_npu._C._npu_getCheckpointState"
        ), patch(
            "torch._dynamo.utils.preserve_rng_state"
        ):

            outputs = basic_npu_graph_node._record(mock_model, mock_inputs)

            mock_model.assert_called_once_with(mock_inputs)
            assert basic_npu_graph_node.recording_outputs == outputs

    def test_reconstruct_outputs(self, basic_npu_graph_node):
        # Setup mock metadata and storage info
        basic_npu_graph_node.outputs_metadata = [
            {
                "nbytes": 4,
                "data_ptr": 1234,
                "size": (1,),
                "stride": (1,),
                "dtype": torch.float32,
                "device": "npu",
                "storage_offset": 0,
            }
        ]
        basic_npu_graph_node.output_weakrefs = [MagicMock()]
        basic_npu_graph_node.output_storage_alias = [UnaliasedStorage]
        basic_npu_graph_node.cached_tensor_outputs = [MagicMock()]

        with patch(
            "torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata"
        ) as mock_construct:
            outputs = basic_npu_graph_node.reconstruct_outputs()
            assert len(outputs) == 1

    def test_aliased_output_reconstruction(self, basic_npu_graph_node):
        basic_npu_graph_node.outputs_metadata = [
            {
                "nbytes": 4,
                "data_ptr": 1234,
                "size": (1,),
                "stride": (1,),
                "dtype": torch.float32,
                "device": "npu",
                "storage_offset": 0,
            }
        ]
        basic_npu_graph_node.output_storage_alias = [AliasesPriorGraphOutput((0, 0))]
        basic_npu_graph_node.outputs_weakrefs = [MagicMock()]
        basic_npu_graph_node.cached_tensor_outputs = [MagicMock()]

        with patch("torch_npu.npu._graph_tree.maybe_deref") as mock_maybe_deref:
            mock_maybe_deref.return_value = (MagicMock(), 1234)
            outputs = basic_npu_graph_node.reconstruct_outputs()
            assert len(outputs) == 1

    def test_liveness_tracking(self, basic_npu_graph_node):
        mock_ref = MagicMock()
        basic_npu_graph_node.path_weakrefs = [[mock_ref]]

        with patch("torch_npu.npu._graph_tree.is_live") as mock_is_live:
            mock_is_live.return_value = True
            liveness = basic_npu_graph_node._get_liveness(
                basic_npu_graph_node.path_weakrefs
            )
            assert liveness == [[True]]

    def test_child_management(self, basic_npu_graph_node):
        mock_child = MagicMock()
        basic_npu_graph_node.add_child("test_func", mock_child)
        assert "test_func" in basic_npu_graph_node.children
        assert mock_child in basic_npu_graph_node.children["test_func"]

    def test_invalid_run_conditions(self, basic_npu_graph_node):
        basic_npu_graph_node.graph = None
        with pytest.raises(RuntimeError):
            basic_npu_graph_node.run_graph()

    def test_storage_metadata_handling(self, basic_npu_graph_node):
        tensor = torch.tensor([1.0], device="npu")
        metadata = basic_npu_graph_node._tensor_metadata(tensor)

        assert metadata["data_ptr"] == tensor.untyped_storage().data_ptr()
        assert metadata["size"] == tensor.shape

    @patch("torch.npu.synchronize")
    @patch("torch_npu.npu._graph_tree._use_npu_memory_pool_manager")
    def test_input_processing(self, mock_pool_manager, mock_sync, basic_npu_graph_node):
        inputs = [torch.tensor([1.0], device="npu")]
        processed = basic_npu_graph_node._allocate_and_copy_recording_inputs(inputs)
        assert len(processed) == 1
        assert isinstance(processed[0], torch.Tensor)

    def test_check_invariants(self, basic_npu_graph_node):
        mock_inputs = [torch.tensor([1.0], device="npu")]
        basic_npu_graph_node.static_input_data_ptrs = [mock_inputs[0].data_ptr()]
        basic_npu_graph_node.npugraph_managed_idxs = [0]

        assert basic_npu_graph_node.check_invariants(mock_inputs)

    def test_descendant_count(self, basic_npu_graph_node):
        mock_child = MagicMock(num_descendants=lambda: 0)
        basic_npu_graph_node.children["test"] = [mock_child]
        assert basic_npu_graph_node.num_descendants() == 1

    def test_prepare_alias_info_metadata_int(self, basic_npu_graph_node):
        result = basic_npu_graph_node.prepare_alias_info_for_tensor_construction(
            MagicMock(), 42
        )
        assert result is None

    def test_prepare_alias_info_unaliased_storage(self, basic_npu_graph_node):
        result = basic_npu_graph_node.prepare_alias_info_for_tensor_construction(
            UnaliasedStorage, {"meta": "data"}
        )
        assert result is None

    def test_prepare_alias_info_aliases_prior_graph_valid(self, basic_npu_graph_node):
        mock_ref = MagicMock()
        basic_npu_graph_node.path_weakrefs = [[mock_ref, mock_ref]]
        alias_info = AliasesPriorGraphOutput((0, 1))

        with patch("torch.UntypedStorage._new_with_weak_ptr") as mock_new:
            result = basic_npu_graph_node.prepare_alias_info_for_tensor_construction(
                alias_info, {"meta": "data"}
            )
            mock_new.assert_called_once_with(mock_ref())
            assert result == mock_new.return_value

    def test_prepare_alias_info_aliases_prior_graph_none_ref(
        self, basic_npu_graph_node
    ):
        basic_npu_graph_node.path_weakrefs = [[None, None]]
        alias_info = AliasesPriorGraphOutput((0, 1))

        with pytest.raises(RuntimeError):
            basic_npu_graph_node.prepare_alias_info_for_tensor_construction(
                alias_info, {"meta": "data"}
            )

    def test_prepare_alias_info_aliases_new_output(self, basic_npu_graph_node):
        alias_info = AliasesNewOutput(123)
        result = basic_npu_graph_node.prepare_alias_info_for_tensor_construction(
            alias_info, {"meta": "data"}
        )
        assert result == 123

    def test_prepare_alias_info_invalid_type(self, basic_npu_graph_node):
        with pytest.raises(RuntimeError):
            basic_npu_graph_node.prepare_alias_info_for_tensor_construction(
                "invalid_type", {"meta": "data"}
            )

    # Tests for prepare_storages_for_construction
    def test_prepare_storages_mixed_aliases(self, basic_npu_graph_node):
        basic_npu_graph_node.output_storage_alias = [
            UnaliasedStorage,
            AliasesNewOutput(123),
            AliasesPriorGraphOutput((0, 1)),
        ]
        basic_npu_graph_node.outputs_metadata = [None, {}, {}]
        basic_npu_graph_node.path_weakrefs = [[None, MagicMock(), MagicMock()]]

        with patch("torch.UntypedStorage._new_with_weak_ptr"):
            results = basic_npu_graph_node.prepare_storages_for_construction()

        assert len(results) == 3
        assert results[0] is None
        assert results[1] == 123

    # Tests for debug_assert_invariants
    def test_debug_assert_invariants_valid(self, basic_npu_graph_node):
        from torch._inductor import config

        config.triton.fast_path_cudagraph_asserts = True
        expected_liveness = [[], [True, False]]
        newly_dead = [(1, 1)]
        ref = MagicMock(return_value=None)
        basic_npu_graph_node.outputs_weakrefs = [None, ref]
        basic_npu_graph_node.parent.outputs_weakrefs = []
        basic_npu_graph_node.path_weakrefs = [
            basic_npu_graph_node.parent.outputs_weakrefs,
            basic_npu_graph_node.outputs_weakrefs,
        ]

        # Should not raise
        with patch("torch_npu.npu._graph_tree.get_block_addrs"):
            basic_npu_graph_node.debug_assert_invariants(expected_liveness, newly_dead)
        config.triton.fast_path_cudagraph_asserts = False

    def test_debug_assert_invariants_dead_ref_alive(self, basic_npu_graph_node):
        from torch._inductor import config

        config.triton.fast_path_cudagraph_asserts = True
        expected_liveness = [[False]]
        newly_dead = [(0, 0)]
        basic_npu_graph_node.path_weakrefs = [
            [MagicMock(return_value=("ptr", 123))]
        ]  # Live ref

        with pytest.raises(RuntimeError):
            basic_npu_graph_node.debug_assert_invariants(expected_liveness, newly_dead)
        config.triton.fast_path_cudagraph_asserts = False

    # Tests for _initialize_cached_tensors
    def test_initialize_cached_tensors_valid(self, basic_npu_graph_node):
        basic_npu_graph_node.output_storage_alias = [UnaliasedStorage, UnaliasedStorage]
        basic_npu_graph_node.outputs_metadata = [
            {"dtype": torch.float},
            {"dtype": torch.int},
        ]
        basic_npu_graph_node.unaliased_in_all_paths = [True, False]
        basic_npu_graph_node.outputs_weakrefs = [None, None]

        with patch.object(basic_npu_graph_node, "create_storage"), patch(
            "torch_npu._C._add_cached_tensor"
        ), patch.object(
            basic_npu_graph_node, "_reconstruct_from_tensor_metadata"
        ) as mock_reconstruct:

            mock_reconstruct.return_value = torch.tensor([1.0], device="npu:0")
            basic_npu_graph_node._initialize_cached_tensors()

            assert len(basic_npu_graph_node.cached_tensor_outputs) == 2
            assert basic_npu_graph_node.cached_tensor_outputs[0] is not None
            assert len(basic_npu_graph_node.outputs_weakrefs) == 2

    def test_initialize_cached_tensors_invalid_storage_info(self, basic_npu_graph_node):
        basic_npu_graph_node.output_storage_alias = ["invalid"]
        basic_npu_graph_node.unaliased_in_all_paths = [True]

        basic_npu_graph_node._initialize_cached_tensors()


@patch("torch_npu.npu.graphs.NPUGraph.replay")
@patch("torch_npu.npu._graph_tree.check_memory_pool")
@patch("torch_npu.npu._graph_tree._use_npu_memory_pool_manager")
class TestNPUGraphNodeRun(TestCase):
    def setUp(self):
        """Initialize common test components and configurations"""
        self.device = "npu:0"

        def model_side_effect(inputs):
            # Clear inputs list while preserving reference
            inputs[:] = []
            return []

        self.wrapped_function = MagicMock(
            spec=WrappedFunction,
            static_input_idxs=[0],
            constants=[],
            model=MagicMock(side_effect=model_side_effect),
        )
        self.graph_id = 1
        self.npu_graphs_pool = (0, 0)
        self.stream = torch.npu.Stream(device=self.device)

        # Create test tensors
        self.static_input = torch.randn(
            3, 3, device=self.device
        )  # Static input (parameter-like)
        self.dynamic_input = torch.randn(2, 2, device=self.device)  # Dynamic input

    def _create_node(self, inputs, parent=None):
        """Helper to create NPUGraphNode instance"""
        with patch("torch_npu._C._npu_getCheckpointState"), patch(
            "torch_npu.npu.graphs.NPUGraph.capture_begin"
        ), patch("torch_npu.npu.graphs.NPUGraph.capture_end"):
            return NPUGraphNode(
                wrapped_function=self.wrapped_function,
                graph_id=self.graph_id,
                parent=parent,
                inputs=inputs,
                npu_graphs_pool=self.npu_graphs_pool,
                device_index=0,
                stack_traces=None,
                stream=self.stream,
            )

    @patch.object(NPUGraphNode, "run_graph")
    def test_static_input_optimization(
        self, mock_run_graph, mock_pool, mock_check, mock_replay
    ):
        """Verify static inputs bypass copy operations"""
        # Mark all inputs as static
        self.wrapped_function.static_input_idxs = [0, 1]
        node = self._create_node([self.static_input, self.static_input.clone()])

        # Execute with cloned inputs
        node.run([self.static_input.clone(), self.static_input.clone()])

        # Validate no copy operations occurred
        self.assertEqual(mock_run_graph.call_count, 1)

    @patch.object(NPUGraphNode, "reconstruct_outputs")
    def test_output_reconstruction_flow(
        self, mock_reconstruct, mock_pool, mock_check, mock_replay
    ):
        """Test full output reconstruction pipeline"""
        # Configure mock reconstruction
        expected_output = torch.tensor([1.0], device=self.device)
        mock_reconstruct.return_value = [expected_output]

        node = self._create_node([self.static_input])
        outputs = node.run([self.static_input.clone()])

        # Validate outputs
        self.assertEqual(outputs, [expected_output])
        mock_reconstruct.assert_called_once()

    @patch("torch._foreach_copy_")
    def test_batched_copy_optimization(
        self, mock_batched_copy, mock_pool, mock_check, mock_replay
    ):
        """Verify batched copy operations for efficiency"""
        # Configure multiple dynamic inputs
        self.wrapped_function.static_input_idxs = []
        inputs = [torch.randn(2, 2, device=self.device) for _ in range(3)]
        new_inputs = [t.clone() for t in inputs]
        node = self._create_node(inputs)

        # Execute with new inputs
        node.run(new_inputs)

        # Validate single batched copy call
        args, _ = mock_batched_copy.call_args
        self.assertEqual(len(args[0]), 3)

    def test_memory_cleanup_after_execution(self, mock_pool, mock_check, mock_replay):
        """Validate input list cleanup post-execution"""
        initial_inputs = [self.static_input.clone(), self.dynamic_input.clone()]
        input_copy = [t.clone() for t in initial_inputs]
        node = self._create_node(initial_inputs)

        # Execute and verify cleanup
        node.run(input_copy)
        self.assertEqual(len(input_copy), 0)


class TestGetNpugraphSegments(TestCase):
    @patch('torch.npu.memory_snapshot')  
    def test_get_npugraph_segments(self, mock_snapshot):            
        mock_snapshot.return_value = [
                    {"segment_pool_id": (0, 1), "address": 1000, "blocks": []},
                    {"segment_pool_id": (0, 0), "address": 2000, "blocks": []},
                    {"segment_pool_id": (0, 1), "address": 3000, "blocks": []},
                ]                      
        result = get_npugraph_segments((0, 1))                      
        self.assertEqual(len(result), 2)      
        mock_snapshot.assert_called_once_with()


class TestGetBlockAddrs(TestCase):
    @patch('torch_npu.npu._graph_tree.get_npugraph_segments')
    def test_get_block_addrs_live_only(self, mock_segments):
        mock_segments.return_value = [
            {
                "segment_pool_id": (0, 0),
                "address": 1000,
                "blocks": [
                    {"state": "active_allocated", "size": 100},
                    {"state": "inactivate", "size": 200},
                    {"state": "active_allocated", "size": 300},
                ]
            },
            {
                "segment_pool_id": (0, 0),
                "address": 2000,
                "blocks": [
                    {"state": "active_allocated", "size": 50},
                    {"state": "inactivate", "size": 150},
                ]
            }
        ]
        result = get_block_addrs((0, 0), live_only=True)
        self.assertEqual(result, [1000, 1300, 2000])
        mock_segments.assert_called_once_with((0, 0))

    @patch('torch_npu.npu._graph_tree.get_npugraph_segments')
    def test_get_block_addrs_all_blocks(self, mock_segments):
        mock_segments.return_value = [
            {
                "segment_pool_id": (0, 0),
                "address": 1000,
                "blocks": [
                    {"state": "active_allocated", "size": 100},
                    {"state": "inactivate", "size": 200},
                ]
            }
        ]
        result = get_block_addrs((0, 0), live_only=False)
        self.assertEqual(result, [1000, 1100])
        mock_segments.assert_called_once_with((0, 0))


class TestFormatTb(TestCase):
    def test_format_tb(self):
        frames = [
            {"filename": "/path/to/file.py", "line": 42, "name": "test_function"},
            {"filename": "/path/to/module.py", "line": 100, "name": "helper_method"},
        ]
        result = format_tb(frames)
        self.assertIn("/path/to/file.py", result)
        self.assertIn("test_function", result)
        self.assertIn("/path/to/module.py", result)
        self.assertIn("helper_method", result)
        self.assertIn("line 100", result)


class TestCheckMemoryPool(TestCase):
    @patch('torch_npu._C._npu_checkPoolLiveAllocations')
    def test_check_memory_pool_fast_path_pass(self, mock_check):
        mock_check.return_value = True

        mock_storage1 = MagicMock(spec=StorageWeakRefWrapper)
        mock_storage1.data_ptr.return_value = 1001
        mock_storage1.return_value = True

        mock_storage2 = MagicMock(spec=StorageWeakRefWrapper)
        mock_storage2.data_ptr.return_value = 1002
        mock_storage2.return_value = True

        check_memory_pool("npu:0", (0, 0), [mock_storage1, mock_storage2])
        mock_check.assert_called_once_with(
            "npu:0", (0, 0), {1001, 1002}
        )

    @patch('torch_npu._C._npu_checkPoolLiveAllocations')
    @patch('torch_npu.npu._graph_tree.get_npugraph_segments')
    @patch('torch_npu.npu._graph_tree.format_tb')
    @patch('gc.collect')
    def test_check_memory_pool_slow_path_unallocated_storage(
        self, mock_gc, mock_format_tb, mock_segments, mock_check
    ):
        mock_check.return_value = False
        mock_segments.return_value = [
            {
                "segment_pool_id": (0, 0),
                "address": 2000,
                "blocks": [
                    {"state": "active_allocated", "size": 100, "frames": []},
                ]
            }
        ]
        mock_storage = MagicMock(spec=StorageWeakRefWrapper)
        mock_storage.data_ptr.return_value = 1000
        mock_storage.return_value = True
        with self.assertRaisesRegex(
            RuntimeError, r"These storage data ptrs are not allocated in pool \(0, 0\) but should be \{1000\}"
        ):
            check_memory_pool("npu:0", (0, 0), [mock_storage])

    @patch('torch_npu._C._npu_checkPoolLiveAllocations')
    @patch('torch_npu.npu._graph_tree.get_npugraph_segments')
    @patch('torch_npu.npu._graph_tree.format_tb')
    @patch('gc.collect')
    def test_check_memory_pool_slow_path_unaccounted_blocks(
        self, mock_gc, mock_format_tb, mock_segments, mock_check
    ):
        mock_check.return_value = False
        mock_segments.return_value = [
            {
                "segment_pool_id": (0, 0),
                "address": 1000,
                "blocks": [
                    {"state": "active_allocated", "size": 100, "frames": [
                        {"filename": "/path/to/file.py", "line": 42, "name": "allocate_func"}
                    ]},
                ]
            }
        ]
        live_storages = []
        mock_format_tb.return_value = "Formatted Traceback"
        with self.assertRaisesRegex(
            RuntimeError, "These live storage data ptrs are in the npugraph pool but not accounted for"
        ):
            check_memory_pool("npu:0", (0, 0), live_storages)

    def test_check_memory_pool_invalid_input(self):
        invalid_storages = [1, 2, 3]
        with self.assertRaisesRegex(
            RuntimeError, r"check all\(isinstance\(elem, StorageWeakRefWrapper\) for elem in live_storages_ptrs\) fail"
        ):
            check_memory_pool("npu:0", (0, 0), invalid_storages)


class TestNPUGraphTreeManager:
    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager._run')
    def test_run_forward_mode(self, mock_run):
        manager = NPUGraphTreeManager(0)
        manager.id_to_mode[FunctionID(1)] = CompilationMode.FORWARD
        result = manager.run([torch.tensor([1.0])], FunctionID(1))
        mock_run.assert_called_once_with([torch.tensor([1.0])], FunctionID(1))
        self.assertTrue(manager.running_forwards_with_pending_backwards)
        self.assertTrue(result == mock_run.return_value)

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager._run')
    def test_run_backward_mode(self, mock_run):
        manager = NPUGraphTreeManager(0)
        manager.id_to_mode[FunctionID(1)] = CompilationMode.BACKWARD
        result = manager.run([torch.tensor([1.0])], FunctionID(1))
        mock_run.assert_called_once_with([torch.tensor([1.0])], FunctionID(1))
        self.assertFalse(manager.running_forwards_with_pending_backwards)
        self.assertTrue(result == mock_run.return_value)

    def test_set_to_running_backward(self):
        manager = NPUGraphTreeManager(0)
        manager.running_forwards_with_pending_backwards = True
        manager.set_to_running_backward()
        self.assertFalse(manager.running_forwards_with_pending_backwards)

    def test_shutdown(self):
        manager = NPUGraphTreeManager(0)
        mock_node1 = MagicMock()
        mock_node2 = MagicMock()
        mock_node3 = MagicMock()
        manager.roots = {FunctionID(1): [mock_node1]}
        mock_node1.children = {FunctionID(2): [mock_node2]}
        mock_node2.children = {FunctionID(3): [mock_node3]}
        manager.shutdown()
        mock_node1.remove_node_cached_tensors.assert_called_once_with()
        mock_node2.remove_node_cached_tensors.assert_called_once_with()
        mock_node3.remove_node_cached_tensors.assert_called_once_with()
        assert mock_node1.graph is None
        assert mock_node2.graph is None
        assert mock_node3.graph is None
        assert manager.graph is None
        assert manager.roots is None
        assert manager.current_node is None

    @patch('torch.npu.synchronize')
    @patch('torch_npu.npu._graph_tree.NPUGraphNode')
    def test_record_function(self, mock_node, mock_synchronize):
        manager = NPUGraphTreeManager(0)
        manager.ids_to_funcs[FunctionID(1)] = MagicMock()
        manager.ids_to_stack_traces[FunctionID(1)] = "stack_trace"
        manager.npu_graphs_thread_pool = "pool_handle"
        manager.device_index = 0
        manager.stream = MagicMock()
        
        # 设置模拟返回值
        mock_node_instance = MagicMock()
        mock_node.return_value = mock_node_instance
        mock_node_instance.run_first_inputs.return_value = [torch.tensor([1.0])]
        
        # 执行测试
        result = manager.record_function([torch.tensor([1.0])], FunctionID(1))
        
        # 验证调用
        mock_synchronize.assert_any_call()
        mock_node.assert_called_once_with(
            manager.ids_to_funcs[FunctionID(1)],
            ANY,  # graph_id
            None,  # parent
            [torch.tensor([1.0])],
            "pool_handle",
            0,
            "stack_trace",
            manager.stream
        )
        assert isinstance(mock_node.call_args[0][1], GraphID)
        assert manager.current_node == mock_node_instance
        assert manager.path_state == ExecutionState.RECORDING
        assert result == [torch.tensor([1.0])]

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.update_generation')
    def test_execute_node(self, mock_update_gen):
        manager = NPUGraphTreeManager(0)
        mock_node = MagicMock()
        mock_node.run.return_value = [torch.tensor([1.0])]
        
        # 执行测试
        result = manager.execute_node(mock_node, [torch.tensor([1.0])])
        
        # 验证调用
        mock_update_gen.assert_called_once_with()
        assert manager.current_node == mock_node
        assert manager.path_state == ExecutionState.EXECUTION
        assert result == [torch.tensor([1.0])]

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.update_generation')
    @patch('torch_npu.npu._graph_tree.NPUWarmupNode')
    def test_run_eager(self, mock_warmup_node, mock_update_gen):
        manager = NPUGraphTreeManager(0)
        manager.ids_to_funcs[FunctionID(1)] = MagicMock()
        manager.ids_to_stack_traces[FunctionID(1)] = "stack_trace"
        manager.npu_graphs_thread_pool = "pool_handle"
        manager.graph = MagicMock()
        manager.device_index = 0
        manager.stream = MagicMock()
        
        # 设置模拟返回值
        mock_node_instance = MagicMock()
        mock_warmup_node.return_value = mock_node_instance
        mock_node_instance.run.return_value = [torch.tensor([1.0])]
        
        # 执行测试
        result = manager.run_eager([torch.tensor([1.0])], FunctionID(1))
        
        # 验证调用
        mock_update_gen.assert_called_once_with()
        mock_warmup_node.assert_called_once_with(
            manager.ids_to_funcs[FunctionID(1)],
            None,
            "pool_handle",
            manager.graph,
            0,
            "stack_trace",
            manager.stream,
            False,
        )
        assert manager.current_node == mock_node_instance
        assert manager.path_state == ExecutionState.WARMUP
        assert result == [torch.tensor([1.0])]

    def test_new_graph_id(self):
        manager = NPUGraphTreeManager(0)
        id1 = manager.new_graph_id()
        id2 = manager.new_graph_id()
        assert isinstance(id1, GraphID)
        assert isinstance(id2, GraphID)
        assert id1 != id2

    def test_new_func_id(self):
        manager = NPUGraphTreeManager(0)
        id1 = manager.new_func_id()
        id2 = manager.new_func_id()
        assert isinstance(id1, FunctionID)
        assert isinstance(id2, FunctionID)
        assert id1 != id2

    def test_in_recording_property(self):
        manager = NPUGraphTreeManager(0)
        manager.path_state = ExecutionState.NONE
        assert manager.in_recording is False
        manager.path_state = ExecutionState.RECORDING
        assert manager.in_recording is True

    def test_in_warmup_property(self):
        manager = NPUGraphTreeManager(0)
        manager.path_state = ExecutionState.NONE
        assert manager.in_warmup is False
        manager.path_state = ExecutionState.WARMUP
        assert manager.in_warmup is True

    def test_get_roots(self):
        manager = NPUGraphTreeManager(0)
        mock_node1 = MagicMock()
        mock_node2 = MagicMock()
        manager.roots = {
            FunctionID(1): [mock_node1],
            FunctionID(2): [mock_node2]
        }
        roots = list(manager.get_roots())
        assert roots == [mock_node1, mock_node2]

    def test_current_node_property_and_setter(self):
        manager = NPUGraphTreeManager(0)
        assert manager.current_node is None
        assert manager.path_state == ExecutionState.NONE
        mock_node = MagicMock()
        manager.current_node = mock_node
        assert manager.current_node == mock_node
        assert manager._current_node == mock_node
        manager.current_node = None
        assert manager.current_node is None
        assert manager.path_state == ExecutionState.NONE

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.get_curr_generation')
    def test_update_generation(self, mock_get_gen):
        manager = NPUGraphTreeManager(0)
        mock_get_gen.return_value = 5
        manager.update_generation()
        assert manager.current_gen == 5
        mock_get_gen.assert_called_once_with()

    @patch('torch_npu.npu._graph_tree.MarkStepBox.mark_step_counter', 3)
    def test_get_curr_generation_mark_step(self):
        result = NPUGraphTreeManager.get_curr_generation()
        assert result == 3

    @patch('torch_npu.npu._graph_tree.MarkStepBox.mark_step_counter', 0)
    @patch('torch_npu.npu._graph_tree.GenerationTracker.generation', 5)
    def test_get_curr_generation_generation_tracker(self):
        result = NPUGraphTreeManager.get_curr_generation()
        assert result == 5

    @patch('torch_npu.npu._graph_tree.MarkStepBox.mark_step_counter', 3)
    def test_user_invoked_mark_step_true(self):
        result = NPUGraphTreeManager.user_invoked_mark_step()
        assert result is True

    @patch('torch_npu.npu._graph_tree.MarkStepBox.mark_step_counter', 0)
    def test_user_invoked_mark_step_false(self):
        result = NPUGraphTreeManager.user_invoked_mark_step()
        assert result is False

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.in_new_torch_compile_invocation')
    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.user_invoked_mark_step')
    def test_can_start_new_generation_true_user_mark_step(
        self, mock_user_mark_step, mock_in_new_invocation
    ):
        manager = NPUGraphTreeManager(0)
        mock_in_new_invocation.return_value = True
        mock_user_mark_step.return_value = True
        result = manager.can_start_new_generation

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.in_new_torch_compile_invocation')
    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.user_invoked_mark_step')
    def test_can_start_new_generation_true_no_pending_backwards(
        self, mock_user_mark_step, mock_in_new_invocation
    ):
        manager = NPUGraphTreeManager(0)
        manager.running_forwards_with_pending_backwards = False
        mock_in_new_invocation.return_value = True
        mock_user_mark_step.return_value = False
        result = manager.can_start_new_generation()
        assert result is True

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.in_new_torch_compile_invocation')
    def test_can_start_new_generation_false_pending_backwards(
        self, mock_in_new_invocation
    ):
        manager = NPUGraphTreeManager(0)
        manager.running_forwards_with_pending_backwards = True
        mock_in_new_invocation.return_value = True
        result = manager.can_start_new_generation()
        assert result is False

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.in_new_torch_compile_invocation')
    def test_can_start_new_generation_false_not_new_invocation(
        self, mock_in_new_invocation
    ):
        manager = NPUGraphTreeManager(0)
        mock_in_new_invocation.return_value = False
        result = manager.can_start_new_generation()
        assert result is False

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.get_curr_generation')
    def test_in_new_torch_compile_invocation_true(self, mock_get_gen):
        manager = NPUGraphTreeManager(0)
        manager.current_gen = 1
        mock_get_gen.return_value = 2
        result = manager.in_new_torch_compile_invocation()
        assert result is True

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.get_curr_generation')
    def test_in_new_torch_compile_invocation_false(self, mock_get_gen):
        manager = NPUGraphTreeManager(0)
        manager.current_gen = 1
        mock_get_gen.return_value = 1
        result = manager.in_new_torch_compile_invocation()
        assert result is False

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.in_new_torch_compile_invocation')
    @patch('warnings.warn')
    def test_check_warn_on_unable_to_start_executing_no_warn(
        self, mock_warn, mock_in_new_invocation
    ):
        manager = NPUGraphTreeManager(0)
        mock_in_new_invocation.return_value = False
        manager.check_warn_on_unable_to_start_executing(FunctionID(1))
        mock_warn.assert_not_called()

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.in_new_torch_compile_invocation')
    @patch('warnings.warn')
    def test_check_warn_on_unable_to_start_executing_already_warned(
        self, mock_warn, mock_in_new_invocation
    ):
        manager = NPUGraphTreeManager(0)
        manager.warned_functions.add(FunctionID(1))
        mock_in_new_invocation.return_value = True
        manager.check_warn_on_unable_to_start_executing(FunctionID(1))
        mock_warn.assert_not_called()

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.in_new_torch_compile_invocation')
    @patch('warnings.warn')
    def test_check_warn_on_unable_to_start_executing_no_repeated_pattern(
        self, mock_warn, mock_in_new_invocation
    ):
        manager = NPUGraphTreeManager(0)
        mock_in_new_invocation.return_value = True
        
        mock_node = MagicMock()
        mock_node._path_from_root = [MagicMock()]
        mock_node._path_from_root[0].wrapped_function.id = FunctionID(2)
        mock_node.wrapped_function.id = FunctionID(1)
        manager.current_node = mock_node
        manager.check_warn_on_unable_to_start_executing(FunctionID(1))
        mock_warn.assert_not_called()

    @patch('torch_npu.npu._graph_tree.NPUGraphTreeManager.in_new_torch_compile_invocation')
    @patch('warnings.warn')
    def test_check_warn_on_unable_to_start_executing_warn(
        self, mock_warn, mock_in_new_invocation
    ):
        manager = NPUGraphTreeManager(0)
        mock_in_new_invocation.return_value = True
        
        mock_node1 = MagicMock()
        mock_node1.wrapped_function.id = FunctionID(1)
        mock_node1.parent = MagicMock()
        mock_node1.parent.wrapped_function.id = FunctionID(0)
        
        mock_node2 = MagicMock()
        mock_node2.wrapped_function.id = FunctionID(1)
        mock_node2.parent = MagicMock()
        mock_node2.parent.wrapped_function.id = FunctionID(0)
        
        mock_current_node = MagicMock()
        mock_current_node.wrapped_function.id = FunctionID(1)
        mock_current_node.parent = MagicMock()
        mock_current_node.parent.wrapped_function.id = FunctionID(0)
        
        mock_current_node._path_from_root = [mock_node1, mock_node2]
        manager.current_node = mock_current_node
        manager.check_warn_on_unable_to_start_executing(FunctionID(1))
        mock_warn.assert_called_once_with(
            "Unable to hit fast path of NPUGraphs because of pending, uninvoked backwards. "
            "Consider running with torch.no_grad() or using torch.compiler.npugraph_mark_step_begin() "
            "before each model invocation"
        )
        assert FunctionID(1) in manager.warned_functions


if __name__ == "__main__":
    run_tests()
