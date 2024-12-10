import os
import stat
import copy
import json
import time

import torch
from torch_npu.utils._path_manager import PathManager
from torch_npu.profiler.profiler import tensorboard_trace_handler, profile
from torch_npu.profiler.scheduler import Schedule as schedule
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.profiler._dynamic_profiler._dynamic_profiler_config_context import ConfigContext
from torch_npu.profiler._dynamic_profiler._dynamic_profiler_monitor_shm import DynamicProfilerShareMemory
import torch_npu.profiler.dynamic_profile as dp


class SmallModel(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=12):
        super(SmallModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, 3, padding=1)

    def forward(self, input_1):
        input_1 = self.conv1(input_1)
        input_1 = self.relu1(input_1)
        input_1 = self.conv2(input_1)
        return input_1.reshape(input_1.shape[0], -1)


class TrainModel:
    def __init__(self):
        self.input_shape = (4, 3, 24, 24)
        self.out_shape = (4, 12, 24, 24)
        self.device = torch.device("npu:0")
        self.model = SmallModel(self.input_shape[1], self.out_shape[1]).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)

    def train_one_step(self):
        inputs = torch.rand(self.input_shape).to(self.device)
        target = torch.rand(self.out_shape).reshape(self.out_shape[0], -1).to(self.device)
        output = self.model(inputs)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class TestDynamicProfiler(TestCase):
    TRACE_FILE_NAME = "trace_view.json"
    KERNEL_FILE_NAME = "kernel_details.csv"
    model_train = TrainModel()
    small_steps = 1
    large_steps = 5
    flags = os.O_WRONLY
    mode = stat.S_IRUSR | stat.S_IWUSR
    start_step = 0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.json_sample = DynamicProfilerShareMemory.JSON_DATA
        cls.results_path = f"./dynamic_profiler_results_{str(os.getpid())}"
        cls.default_prof_dir = os.path.join(cls.results_path, "default_prof_dir")
        cls.rank_prof_dir = os.path.join(cls.results_path, "rank_prof_dir")
        cls.invalid_rank_prof_dir = os.path.join(cls.results_path, "invalid_rank_prof_dir")
        cls.active_rank_prof_dir = os.path.join(cls.results_path, "active_rank_prof_dir")
        cls.cfg_prof_dir = os.path.join(cls.results_path, "cfg_prof_dir")
        cls.cfg_path = os.path.join(cls.results_path, "profiler_config.json")
        os.environ["RANK"] = "0"
        dp.init(cls.results_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.results_path):
            PathManager.remove_path_safety(cls.results_path)

    def test_modify_cfg_prof_dir_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['prof_dir'] = 1
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_analyse_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['analyse'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_record_shapes_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['record_shapes'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_profile_memory_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['profile_memory'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_with_stack_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['with_stack'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_with_flops_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['with_flops'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_with_modules_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['with_modules'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_rank_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['is_rank'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_rank_list_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['is_rank'] = True
        cfg_json['rank_list'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_profiler_level_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if 'experimental_config' not in cfg_json.keys():
            self.assertTrue('experimental_config' in cfg_json.keys())
        cfg_json['experimental_config']['profiler_level'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_aic_metrics_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if 'experimental_config' not in cfg_json.keys():
            self.assertTrue('experimental_config' in cfg_json.keys())
        cfg_json['experimental_config']['aic_metrics'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_l2_cache_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if 'experimental_config' not in cfg_json.keys():
            self.assertTrue('experimental_config' in cfg_json.keys())
        cfg_json['experimental_config']['l2_cache'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_data_simplification_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if 'experimental_config' not in cfg_json.keys():
            self.assertTrue('experimental_config' in cfg_json.keys())
        cfg_json['experimental_config']['data_simplification'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_record_op_args_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        if 'experimental_config' not in cfg_json.keys():
            self.assertTrue('experimental_config' in cfg_json.keys())
        cfg_json['experimental_config']['record_op_args'] = "1"
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_modify_cfg_export_type_invalid(self) -> None:
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['is_rank'] = False
        if 'experimental_config' not in cfg_json.keys():
            self.assertTrue('experimental_config' in cfg_json.keys())
        cfg_json['experimental_config']['export_type'] = 1
        cfg_ctx = ConfigContext(cfg_json)
        prof = profile(
            activities=cfg_ctx.activities(),
            schedule=schedule(wait=0, warmup=0, active=cfg_ctx.active(), repeat=1, skip_first=0),
            on_trace_ready=tensorboard_trace_handler(self.cfg_prof_dir, analyse_flag=cfg_ctx.analyse()),
            record_shapes=cfg_ctx.record_shapes,
            profile_memory=cfg_ctx.profile_memory,
            with_stack=cfg_ctx.with_stack,
            with_flops=cfg_ctx.with_flops,
            with_modules=cfg_ctx.with_modules,
            experimental_config=cfg_ctx.experimental_config
        )
        prof.start()
        prof.stop()
        has_prof = False
        if self.has_prof_dir(self.cfg_prof_dir):
            has_prof = True
        if os.path.exists(self.cfg_prof_dir):
            PathManager.remove_path_safety(self.cfg_prof_dir)
        self.assertTrue(has_prof)

    def test_dynamic_profiler_default(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['prof_dir'] = self.default_prof_dir
        cfg_json['start_step'] = TestDynamicProfiler.start_step + 1
        with os.fdopen(os.open(self.cfg_path, self.flags, self.mode), 'w') as f:
            time.sleep(1)
            json.dump(cfg_json, f, indent=4)
        time.sleep(3)
        dp.step()
        TestDynamicProfiler.start_step += 1
        self.model_train.train_one_step()
        dp.step()
        TestDynamicProfiler.start_step += 1
        has_prof = False
        if self.has_prof_dir(self.default_prof_dir):
            has_prof = True
        if os.path.exists(self.default_prof_dir):
            PathManager.remove_path_safety(self.default_prof_dir)
        self.assertTrue(has_prof)

    def test_dynamic_profiler_rank(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['prof_dir'] = self.rank_prof_dir
        cfg_json['is_rank'] = True
        cfg_json['rank_list'] = [0]
        cfg_json['start_step'] = TestDynamicProfiler.start_step + 1

        with os.fdopen(os.open(self.cfg_path, self.flags, self.mode), 'w') as f:
            time.sleep(1)
            json.dump(cfg_json, f, indent=4)
        time.sleep(3)
        dp.step()
        TestDynamicProfiler.start_step += 1
        self.model_train.train_one_step()
        dp.step()
        TestDynamicProfiler.start_step += 1
        has_prof = False
        if self.has_prof_dir(self.rank_prof_dir):
            has_prof = True
        if os.path.exists(self.rank_prof_dir):
            PathManager.remove_path_safety(self.rank_prof_dir)
        self.assertTrue(has_prof)

    def test_dynamic_profiler_rank_invalid(self):
        cfg_json = copy.deepcopy(self.json_sample)
        cfg_json['prof_dir'] = self.invalid_rank_prof_dir
        cfg_json['is_rank'] = True
        cfg_json['rank_list'] = [1]
        cfg_json['start_step'] = TestDynamicProfiler.start_step + 1

        with os.fdopen(os.open(self.cfg_path, self.flags, self.mode), 'w') as f:
            time.sleep(1)
            json.dump(cfg_json, f, indent=4)
        time.sleep(3)
        dp.step()
        TestDynamicProfiler.start_step += 1
        self.model_train.train_one_step()
        dp.step()
        TestDynamicProfiler.start_step += 1
        has_prof = False
        if self.has_prof_dir(self.invalid_rank_prof_dir):
            has_prof = True
        if os.path.exists(self.invalid_rank_prof_dir):
            PathManager.remove_path_safety(self.invalid_rank_prof_dir)
        self.assertFalse(has_prof)

    @staticmethod
    def has_prof_dir(path: str) -> bool:
        path = os.path.realpath(path)
        if not os.path.exists(path):
            return False
        for sub_dir in os.listdir(path):
            if sub_dir.endswith("_pt"):
                sub_dir = os.path.join(path, sub_dir)
                for p in os.listdir(sub_dir):
                    if p.startswith("PROF"):
                        return True
        return False

    @staticmethod
    def has_analyse_dir(path: str) -> bool:
        path = os.path.realpath(path)
        if not os.path.exists(path):
            return False
        for sub_dir in os.listdir(path):
            if sub_dir.endswith("_pt"):
                sub_dir = os.path.join(path, sub_dir)
                for p in os.listdir(sub_dir):
                    if p.startswith("ASCEND"):
                        return True
        return False


if __name__ == "__main__":
    run_tests()
