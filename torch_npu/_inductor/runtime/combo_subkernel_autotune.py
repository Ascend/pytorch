"""Measure isolated NPU subkernels, then benchmark promising whole-combo tiles."""
import copy
import json
import math
import time
from collections import defaultdict
from itertools import chain, zip_longest

from torch._inductor.runtime.triton_heuristics import Config, config_from_dict, config_to_dict
from .. import config
from ..codegen.triton_utils import NPUKernelType
from .tile_generator import TileGenerator, aligned_numel_32byte, ub_overflow_retry_configs
from .triton_heuristics import NPUCachingAutotuner, _ub_overflow_failures, build_candidate_plan


def _config_key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _block_keys(meta):
    names = meta['axis_names']
    return {
        *(names[a].upper() + 'BLOCK' for a in meta['split_axis']),
        *(names[a].upper() + 'BLOCK_SUB' for a in meta['tiling_axis']
          if a not in meta['no_loop_axis'] and not (
              names[a].startswith('r') and meta['persistent_reduction'])),
    }


def _options(cfg, block_keys):
    return ({k: v for k, v in cfg.kwargs.items() if k not in block_keys and 'BLOCK' not in k},
            cfg.num_warps, cfg.num_stages)


def _programs(meta, kwargs):
    return math.prod(
        (meta['size_hints'][meta['axis_names'][a]] + kwargs[meta['axis_names'][a].upper() + 'BLOCK'] - 1)
        // kwargs[meta['axis_names'][a].upper() + 'BLOCK'] for a in meta['split_axis'])


_COMBINATION_WIDTHS = (12, 4, 4)
_MAX_LOCAL_SUM_DEGRADATION = 0.50
_TARGET_WAVES = 8
_BUDGET_PATHS = ('equal', 'numel')
_NUMEL_COMBINATIONS = True


def _allocate_budgets(total, weights, minimum):
    """Meet each minimum, then apportion the remainder by integer largest remainder."""
    remaining = total - sum(minimum)
    if remaining < 0:
        return None
    denominator = sum(weights)
    numerators = [remaining * weight for weight in weights]
    budgets = [low + value // denominator for low, value in zip(minimum, numerators)]
    order = sorted(range(len(weights)), key=lambda i: (-(numerators[i] % denominator), i))
    for i in order[:total - sum(budgets)]:
        budgets[i] += 1
    return budgets


def _supplemental_targets(generator, index, numels):
    cores, members = config.num_vector_core, len(numels)
    targets = {}

    def add(target, origin):
        targets.setdefault(max(1, target), []).append(origin)

    add(cores, 'one_wave')
    # Keep the proven low-count candidates even when testing just one budget path.
    for target in {cores // members, (cores + members - 1) // members}:
        add(target, 'equal_w1')
    if generator.tiny_kernel:
        target = 1
        while target <= generator.program_threshold:
            add(target, f'small_{target}')
            target *= 2
    for wave in range(1, min(_TARGET_WAVES, config.combo_kernel_max_waves) + 1):
        total = wave * cores
        if 'equal' in _BUDGET_PATHS:
            for target in {total // members, (total + members - 1) // members}:
                add(target, f'equal_w{wave}')
        if 'numel' in _BUDGET_PATHS:
            budgets = _allocate_budgets(total, numels, [1] * members)
            if budgets is not None:
                add(budgets[index], f'numel_w{wave}')
    return {target: sorted(set(origins)) for target, origins in sorted(targets.items())}


def _supplemental_configs(generator, base, targets):
    """Add outer BLOCK choices to native seeds, keeping the native traversal intact."""
    alignment = 32 // generator.dtype_bytes
    for axis in generator.split_axis:
        name = generator.axis_name[axis].upper() + 'BLOCK'
        sub_name = name + '_SUB'
        extent = generator.numels[axis]
        for seed in base:
            others = math.prod(
                (generator.numels[a] + seed.kwargs[generator.axis_name[a].upper() + 'BLOCK'] - 1)
                // seed.kwargs[generator.axis_name[a].upper() + 'BLOCK']
                for a in generator.split_axis if a != axis)
            simt = seed.kwargs.get('compile_mode') == NPUKernelType.SIMT_ONLY.compile_mode()
            for target, origins in targets.items():
                budget = max(1, target // others)
                block = aligned_numel_32byte((extent + budget - 1) // budget, generator.dtype_bytes)
                for value in sorted({block, max(1, block - alignment)}):
                    trial = copy.deepcopy(seed)
                    trial.kwargs[name] = value
                    if sub_name in trial.kwargs and trial.kwargs[sub_name] > value:
                        sub = 1 << (value.bit_length() - 1) if simt else value
                        trial.kwargs[sub_name] = aligned_numel_32byte(sub, generator.dtype_bytes)
                    tiles = [trial]
                    if axis in generator.real_tiling_axis and sub_name in trial.kwargs:
                        full = copy.deepcopy(trial)
                        sub = 1 << (value.bit_length() - 1) if simt else value
                        full.kwargs[sub_name] = aligned_numel_32byte(sub, generator.dtype_bytes)
                        tiles.append(full)
                    for tile in tiles:
                        volume = generator.calculate_config_numel(tile.kwargs)
                        if generator.valid_tile_numel(volume) and (not simt or volume * generator.dtype_bytes <= 8192):
                            yield tile, origins


def make_subkernel_configs(inductor_meta, triton_meta):
    groups = []
    numels = [math.prod(meta['size_hints'][name] for name in meta['axis_names'])
              for meta in inductor_meta['combo_subkernel_meta']]
    ptrs = sum('ptr' in name for name in (triton_meta or {}).get('signature', {}))
    for index, meta in enumerate(inductor_meta['combo_subkernel_meta']):
        names = meta['axis_names']
        kind = NPUKernelType(meta['npu_kernel_type'])
        modes = (NPUKernelType.SIMT_ONLY, NPUKernelType.SIMT_TEMPLATE) if kind == NPUKernelType.SIMD_SIMT_MIX else (kind,)
        candidates, keys = {}, _block_keys(meta)
        # Use the standalone TileGenerator without changing its program budget,
        # axis rebuilding, split traversal or candidate domain. Keep its shared
        # generator/mode-switch order for SIMD_SIMT_MIX as well.
        generator = TileGenerator(
            [meta['size_hints'][name] for name in names], names,
            list(meta['tiling_axis']), list(meta['no_loop_axis']), list(meta['split_axis']),
            list(meta['low_dims']), meta['persistent_reduction'], meta['split_axis_dtype'],
            npu_kernel_type=kind, input_ptr_num=ptrs, dual_reduction=meta['dual_reduction'])
        generated = []
        for mode in modes:
            generator.set_kernel_type(mode)
            generated.extend(generator.descend_split_tiling())
        extra_seeds = []
        if kind == NPUKernelType.SIMD_SIMT_MIX and not any(
                cfg.kwargs.get('compile_mode') == NPUKernelType.SIMT_ONLY.compile_mode()
                for cfg in generated):
            # MIX initializes the stopping threshold from SIMD UB capacity.
            # With wide fixed axes it can stop before any SIMT tile fits 8 KiB.
            # Keep the native domain, and seed that missing mode with its own
            # native memory threshold; its program budget remains one wave.
            simt_generator = TileGenerator(
                [meta['size_hints'][name] for name in names], names,
                list(meta['tiling_axis']), list(meta['no_loop_axis']), list(meta['split_axis']),
                list(meta['low_dims']), meta['persistent_reduction'], meta['split_axis_dtype'],
                npu_kernel_type=NPUKernelType.SIMT_ONLY, input_ptr_num=ptrs,
                dual_reduction=meta['dual_reduction'])
            extra_seeds = simt_generator.descend_split_tiling()
        targets = _supplemental_targets(generator, index, numels)
        supplemented = _supplemental_configs(generator, generated + extra_seeds, targets)
        for cfg, origins in [*((cfg, ['tilegen']) for cfg in generated),
                             *((cfg, ['simt_mode_seed']) for cfg in extra_seeds), *supplemented]:
            kwargs = {k: v for k, v in cfg.kwargs.items() if k in keys or 'BLOCK' not in k}
            cfg = Config(kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
            programs = _programs(meta, kwargs)
            # The native generator still has its 56-program budget. This guard
            # only bounds the extra total-wave targets after BLOCK alignment.
            if origins != ['tilegen'] and programs > config.combo_kernel_max_waves * config.num_vector_core:
                continue
            key = _config_key(config_to_dict(cfg))
            item = candidates.setdefault(key, dict(config=config_to_dict(cfg), origins=[], programs=programs))
            item['origins'] = sorted(set(item['origins']).union(origins))
        if not candidates:
            raise RuntimeError(f'No legal tiles for combo subkernel {index}')
        groups.append(dict(index=index, block_keys=sorted(keys), candidates=list(candidates.values()),
                           numel=numels[index], supplementary_targets=targets,
                           native_max_programs=generator.max_programs))

    option_sets = []
    for group in groups:
        options = {}
        for item in group['candidates']:
            opt = _options(config_from_dict(item['config']), group['block_keys'])
            options.setdefault(_config_key(opt), opt)
        option_sets.append(options)
    common = [key for key in option_sets[0] if all(key in opts for opts in option_sets)]
    if not common:
        raise RuntimeError('No common NPU compile options for combo subkernels')
    seeds = []
    for key in common:
        shared, warps, stages = option_sets[0][key]
        initial = dict(shared, NPU_SUBKERNEL=-1)
        for group in groups:
            local = next(config_from_dict(item['config']) for item in group['candidates']
                         if _config_key(_options(config_from_dict(item['config']), group['block_keys'])) == key)
            initial.update({f'{field}_{group["index"]}': local.kwargs[field] for field in group['block_keys']})
        seeds.append(Config(initial, num_warps=warps, num_stages=stages))
    inductor_meta.update(
        npu_combo_strategy=(
            f'isolated_subkernel_prerun_ub_retry:{_COMBINATION_WIDTHS}:'
            f'{_MAX_LOCAL_SUM_DEGRADATION}:{_NUMEL_COMBINATIONS}:'
            f'{config.combo_kernel_max_waves}'),
        npu_combo_beam_width=_COMBINATION_WIDTHS,
        npu_combo_groups=groups,
        npu_combo_mode_seeds=[config_to_dict(seed) for seed in seeds],
        npu_combo_shared_options=[option_sets[0][key] for key in common],
        npu_combo_vector_cores=config.num_vector_core)
    # Keep a whole-combo default for cache lookup and single-config execution.
    return [seeds[0]]


def _combination_metrics(items, cores):
    programs = sum(item['programs'] for item in items)
    # Latency / occupied waves estimates block cost, not measured block time.
    # Actual whole-combo timing always decides the final winner.
    costs = [item['time_ms'] / math.ceil(item['programs'] / cores) for item in items]
    work = sum(item['programs'] * cost for item, cost in zip(items, costs)) / cores
    return dict(programs=programs, wave_slack=(-programs) % cores,
                independent_sum_ms=sum(item['time_ms'] for item in items),
                estimated_block_spread=max(costs) / min(costs) - 1,
                estimated_makespan_ms=max(work, max(costs)))


def _global_combinations(pools, cores, numels=None, origins=None):
    """Bounded joint search retaining fast, wave-aligned and balanced choices."""
    beam = [()]
    best = [min(item['time_ms'] for item in pool) for pool in pools]
    limit = (1 + _MAX_LOCAL_SUM_DEGRADATION) * sum(best)
    if origins is None:
        origins = {}
    for index, pool in enumerate(pools):
        ranked = sorted(pool, key=lambda item: item['time_ms'])
        choices = ranked[:3]
        counts = {x['programs'] for x in choices}
        for item in ranked:
            if item['programs'] not in counts:
                choices.append(item)
                counts.add(item['programs'])
            if len(choices) >= 12:
                break
        trials = [(*prefix, item) for prefix in beam for item in choices]
        metrics = {id(items): _combination_metrics(items, cores) for items in trials}
        scores = (
            lambda m: (m['independent_sum_ms'], m['estimated_makespan_ms']),
            lambda m: (m['wave_slack'], m['estimated_makespan_ms'], m['independent_sum_ms']),
            lambda m: (m['estimated_block_spread'], m['estimated_makespan_ms']),
        )
        selected = {}
        origins.clear()
        for source, (score, width) in enumerate(zip(scores, _COMBINATION_WIDTHS), 1):
            eligible = trials if source == 1 else [
                items for items in trials
                if metrics[id(items)]['independent_sum_ms'] + sum(best[index + 1:]) <= limit
            ]
            for items in sorted(eligible, key=lambda items: score(metrics[id(items)]))[:width]:
                key = tuple(_config_key(item['local_config']) for item in items)
                selected.setdefault(key, items)
                origins.setdefault(key, set()).add(str(source))
        beam = list(selected.values())
    selected = {tuple(_config_key(item['local_config']) for item in items): items for items in beam}
    # Standalone-fast tiles may all be too small, making their combined grid
    # cross a wave boundary. Also retain measured choices under shared program
    # budgets, without benchmarking any whole-kernel initialization seed.
    budgets = {max(1, int(rounding(waves * cores / len(pools))))
               for waves in range(1, config.combo_kernel_max_waves + 1)
               for rounding in (math.floor, math.ceil)}
    budget = 1
    while budget <= config.combo_kernel_max_waves * cores:
        budgets.add(budget)
        budget *= 2
    allocations = [(tuple([budget] * len(pools)), '4_equal') for budget in sorted(budgets)]
    if _NUMEL_COMBINATIONS and numels is not None:
        minimum = [min(item['programs'] for item in pool) for pool in pools]
        for wave in range(1, config.combo_kernel_max_waves + 1):
            allocated = _allocate_budgets(wave * cores, numels, minimum)
            if allocated is not None:
                allocations.append((tuple(allocated), '4_numel'))
    for allocated, source in allocations:
        eligible = [sorted((item for item in pool if item['programs'] <= budget),
                           key=lambda item: item['time_ms']) for pool, budget in zip(pools, allocated)]
        if not all(eligible):
            continue
        for rank in range(2):
            items = tuple(pool[min(rank, len(pool) - 1)] for pool in eligible)
            if source == '4_numel' and sum(item['time_ms'] for item in items) > limit:
                continue
            key = tuple(_config_key(item['local_config']) for item in items)
            selected.setdefault(key, items)
            origins.setdefault(key, set()).add(source)
    return list(selected.values())


class NPUComboSubkernelAutotuner(NPUCachingAutotuner):
    """Tune per-subkernel tiles in two stages.

    For each compile mode shared by all members, set NPU_SUBKERNEL to a
    member index and time that member's tile candidates in isolation. Use
    these timings and program counts to shortlist compatible combinations.
    Then set NPU_SUBKERNEL to -1 and benchmark each shortlisted combination
    as a whole kernel. Keep the fastest whole-kernel configuration for launch
    and save it when autotune caching is enabled.
    """

    def _benchmark_kernel_funcs_batch(self, kernel_funcs, benchmark_label):
        timings_us = super()._benchmark_kernel_funcs_batch(kernel_funcs, benchmark_label)
        if timings_us is None:
            return None
        return tuple(time_us / 1000 for time_us in timings_us)

    def _use_local_trials(self):
        return not (self.triton_interpret or self.inductor_meta.get('npu_combo_cache_hit')
                    or self._should_skip_autotune_for_determinism())

    def _modes(self):
        for seed, option in zip(self.inductor_meta['npu_combo_mode_seeds'],
                                self.inductor_meta['npu_combo_shared_options']):
            yield config_from_dict(seed), _config_key(option)

    def _local_trials(self, seed, option_key, group):
        idx = group['index']
        for item in chain(group['candidates'], self._combo_retry_items.get(idx, ())):
            local = config_from_dict(item['config'])
            if _config_key(_options(local, group['block_keys'])) == option_key:
                trial = copy.deepcopy(seed)
                trial.kwargs['NPU_SUBKERNEL'] = idx
                trial.kwargs.update({f'{key}_{idx}': local.kwargs[key] for key in group['block_keys']})
                yield trial, item

    def _prepare_configs_for_precompile(self):
        if not self._use_local_trials():
            return
        # The worker deduplicates runtime BLOCKs and compiles all branches
        # in one batch with the configured precompile_thread_num.
        self._combo_retry_items = {}
        self.configs = []
        for seed, option_key in self._modes():
            branches = [self._local_trials(seed, option_key, group)
                        for group in self.inductor_meta['npu_combo_groups']]
            for row in zip_longest(*branches):
                self.configs.extend(item[0] for item in row if item is not None)

    def _smaller_local_trials(self, batch, errors):
        by_branch = defaultdict(list)
        for cfg, error in zip(batch, errors):
            by_branch[cfg.kwargs['NPU_SUBKERNEL']].append((cfg, error))
        originals = defaultdict(list)
        for cfg in self.configs:
            originals[cfg.kwargs['NPU_SUBKERNEL']].append(cfg)

        constexprs = {self.fn.arg_names[i] for i in self.fn.constexprs}
        extra = []
        for group in self.inductor_meta['npu_combo_groups']:
            idx = group['index']
            pairs = by_branch[idx]
            if not pairs or any(error is None for _, error in pairs):
                continue
            failures = _ub_overflow_failures(*zip(*pairs))
            if not failures:
                continue
            meta = self.inductor_meta['combo_subkernel_meta'][idx]
            smaller = ub_overflow_retry_configs(
                originals[idx], failures, meta, constexprs, self.runtime_block_arg_names,
                persistent_reduction=meta['persistent_reduction'], block_suffix=f'_{idx}',
            )
            for trial in smaller:
                local = copy.deepcopy(trial)
                local.kwargs = {k: v for k, v in trial.kwargs.items()
                                if k != 'NPU_SUBKERNEL' and 'BLOCK' not in k}
                local.kwargs.update({k: trial.kwargs[f'{k}_{idx}'] for k in group['block_keys']})
                self._combo_retry_items.setdefault(idx, []).append(dict(
                    config=config_to_dict(local), origins=['ub_retry'], programs=_programs(meta, local.kwargs)))
            extra.extend(smaller)
        return extra

    def _precompile_configs_with(self, configs, compile_once):
        if not self._use_local_trials():
            return super()._precompile_configs_with(configs, compile_once)

        self.npu_combo_compile_failures = {}

        def compile_and_record(batch):
            results, errors, stacks = compile_once(batch)
            self.npu_combo_compile_failures.update({
                _config_key(config_to_dict(cfg)): f'{error}\n{stack}'
                for cfg, result, error, stack in zip(batch, results, errors, stacks)
                if result is None
            })
            return results, errors, stacks

        def compile_local(batch):
            results, errors, stacks = compile_and_record(batch)
            extra = self._smaller_local_trials(batch, errors)
            if extra:
                expanded = [*self.configs, *extra]
                # Appending preserves the variant IDs already compiled by the worker.
                new_variants = self._precompile_variant_configs(expanded)[len(batch):]
                retry_results, retry_errors, retry_stacks = compile_and_record(new_variants)
                results.extend(retry_results)
                errors.extend(retry_errors)
                stacks.extend(retry_stacks)
                self.configs = expanded
                self.candidate_plan = self._build_candidate_plan(expanded)
            return results, errors, stacks

        return super()._precompile_configs_with(configs, compile_local)

    def autotuner(self, *args, stream, benchmark_run=False, **kwargs):
        if self.best_launcher is not None:
            return
        if not self._use_local_trials():
            return super().autotuner(*args, stream=stream, benchmark_run=benchmark_run, **kwargs)
        if not self.launchers:
            self.precompile()
        self._tune_subkernels(*args, **kwargs)

    def _tune_subkernels(self, *args, **kwargs):
        self._ensure_kernel_loaded()
        start = time.time_ns()
        groups = self.inductor_meta['npu_combo_groups']
        variants = {_config_key(config_to_dict(x.config)): x for x in self.launchers}
        failures = self.npu_combo_compile_failures
        global_results, records = {}, []
        self.npu_combo_tuning_records = records

        def prepare_global_variants(configs):
            pending = {}
            for cfg in configs:
                plan = build_candidate_plan([cfg], self.runtime_block_arg_names)
                variant = plan['variants']['v0']['config']
                key = _config_key(variant)
                if key not in variants and key not in failures:
                    pending.setdefault(key, config_from_dict(variant))
            if not pending:
                return
            # Reuse the existing NPU compile pool and its configured parallelism.
            compile_fn = (self._precompile_configs_parallel_once
                          if config.precompile_thread_num > 1 else self._precompile_configs_once)
            results, errors, stacks = compile_fn(list(pending.values()))
            for key, result, error, stack in zip(pending, results, errors, stacks):
                if result is None:
                    failures[key] = f'{error}\n{stack}'
                else:
                    try:
                        variants[key] = result.make_launcher()
                        global_results[key] = result
                    except Exception as error:
                        failures[key] = str(error)

        def resolve(cfg):
            plan = build_candidate_plan([cfg], self.runtime_block_arg_names)
            candidate = plan['candidate_entries'][0]
            variant = plan['variants'][candidate['variant_id']]['config']
            vkey = _config_key(variant)
            launcher = variants.get(vkey)
            if launcher is None:
                failures.setdefault(vkey, 'No launcher for precompiled variant')
            return launcher, candidate, failures.get(vkey)

        def measure_local(trials, stage):
            plan = self._build_candidate_plan(trials)
            compiled = []
            local_records = {}
            for candidate in plan['candidate_entries']:
                key = _config_key(plan['variants'][candidate['variant_id']]['config'])
                record = dict(stage=stage, config=candidate['full_config'],
                              runtime_blocks=candidate['runtime_blocks'])
                if key in variants:
                    compiled.append(dict(candidate=candidate, launcher=variants[key]))
                else:
                    record['compile_error'] = failures.get(key, 'No launcher for precompiled variant')
                records.append(record)
                local_records[candidate['candidate_id']] = record
            timings = self._benchmark_candidate_plan(plan, compiled, args, kwargs, stage)
            for candidate_id, timing in timings.items():
                local_records[candidate_id]['time_ms'] = float(timing)
            return [record for record in local_records.values()
                    if 0 < record['time_ms'] < math.inf]

        def measure(configs, stage):
            configs = list({_config_key(config_to_dict(cfg)): cfg for cfg in configs}.values())
            device = self.get_device_interface()
            stream = device.get_raw_stream(device.current_device())
            functions, measured = [], []
            for cfg in configs:
                launcher, candidate, error = resolve(cfg)
                record = dict(stage=stage, config=config_to_dict(cfg),
                              runtime_blocks=candidate['runtime_blocks'], time_ms=float('inf'))
                records.append(record)
                if launcher is None:
                    record['compile_error'] = error
                    continue
                runtime = tuple(value for _, value in candidate['runtime_blocks'])
                launch_args = self._build_runtime_launch_args(args, runtime)

                def call(launcher=launcher, launch_args=launch_args):
                    if launcher.config.pre_hook is not None:
                        launcher.config.pre_hook({**dict(zip(self.arg_names, launch_args)), **launcher.config.kwargs})
                    cloned_args, cloned_kwargs = self.clone_args(*launch_args, **kwargs)
                    self.reset_to_zero_args(*args, **kwargs)
                    launcher(*cloned_args, **cloned_kwargs, stream=stream)

                functions.append(call)
                measured.append((record, launcher, runtime))
            timings = self._benchmark_kernel_funcs_batch(functions, stage)
            if timings is None:
                timings = [self.bench(launcher, *args, runtime_blocks=runtime, **kwargs)
                           for _, launcher, runtime in measured]
            for (record, _, _), timing in zip(measured, timings):
                record['time_ms'] = float(timing)
            return [record for record, _, _ in measured
                    if math.isfinite(record['time_ms']) and record['time_ms'] > 0]

        global_configs, combination_info = [], {}
        for seed, option_key in self._modes():
            pools = []
            for group in groups:
                idx, trials, local_items = group['index'], [], {}
                for trial, item in self._local_trials(seed, option_key, group):
                    trials.append(trial)
                    local_items[_config_key(config_to_dict(trial))] = item
                measured = measure_local(trials, f'isolated_subkernel_{idx}')
                pool = []
                for record in measured:
                    item = local_items[_config_key(record['config'])]
                    record.update(programs=item['programs'], local_config=item['config'],
                                  origins=item['origins'])
                    pool.append(record)
                if pool:
                    min(pool, key=lambda item: item['time_ms'])['local_winner'] = True
                pools.append(pool)
            if not all(pools):
                continue
            origins = {}
            best_sum = sum(min(item['time_ms'] for item in pool) for pool in pools)
            for items in _global_combinations(
                pools, config.num_vector_core, [group['numel'] for group in groups], origins
            ):
                trial = copy.deepcopy(seed)
                for group, item in zip(groups, items):
                    local = config_from_dict(item['local_config'])
                    trial.kwargs.update({f'{key}_{group["index"]}': local.kwargs[key] for key in group['block_keys']})
                global_configs.append(trial)
                metrics = _combination_metrics(items, config.num_vector_core)
                key = tuple(_config_key(item['local_config']) for item in items)
                metrics.update(sources=sorted(origins[key]), local_sum_slowdown=metrics['independent_sum_ms'] / best_sum)
                combination_info[_config_key(config_to_dict(trial))] = metrics
        if not global_configs:
            raise RuntimeError('No compatible measured subkernel configurations')
        # One global autotune after every isolated measurement. No coordinate
        # search or whole-kernel initialization feeds back into the local stage.
        prepare_global_variants(global_configs)
        measured = measure(global_configs, 'global_combination')
        for record in measured:
            record.update(combination_info[_config_key(record['config'])])
        if not measured:
            raise RuntimeError('No runnable global combo configurations')
        winner = min(measured, key=lambda item: item['time_ms'])
        winner['selected'] = True
        current = config_from_dict(winner['config'])
        best_launcher, best_candidate, _ = resolve(current)
        self._set_best_candidate(best_candidate, best_launcher)
        # Subsequent precompile/bundling must see the selected whole kernel.
        self.compile_results = [global_results[_config_key(config_to_dict(best_launcher.config))]]
        self.launchers = [best_launcher]
        self.configs = [current]
        self.candidate_plan = self._build_candidate_plan(self.configs)
        self._refresh_variant_launchers()
        self.autotune_time_taken_ns = time.time_ns() - start
        if self.save_cache_hook:
            self.save_cache_hook(current, self.autotune_time_taken_ns,
                                 triton_cache_hash=getattr(best_launcher, 'cache_hash', None))
