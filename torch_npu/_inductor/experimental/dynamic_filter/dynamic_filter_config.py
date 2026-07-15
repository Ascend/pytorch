from dataclasses import dataclass, field
import os

# Additional info on _fasta_dynamic_filter params and tuning strategies can be found at
# torch_npu/_inductor/docs/feature/autotuning_optimization/dynamic_filter_algo_params.md
@dataclass
class _fasta_dynamic_filter:
    r1_pct: float = field(default_factory=lambda: float(os.getenv("FASTA_R1_PCT", 0.3)))
    base_budget: float = field(default_factory=lambda: float(os.getenv("FASTA_BASE_BUDGET", 0.35)))
    high_budget: float = field(default_factory=lambda: float(os.getenv("FASTA_HIGH_BUDGET", 0.4)))
    low_budget: float = field(default_factory=lambda: float(os.getenv("FASTA_LOW_BUDGET", 0.25)))
    max_rounds: int = field(default_factory=lambda: int(os.getenv("FASTA_MAX_ROUNDS", 2)))

# Activate the dynamic filter algo - default=0
if os.getenv("FASTA_DYNAMIC_FILTER", "0") == "1":
    fasta_dynamic_filter = _fasta_dynamic_filter()
else:
    fasta_dynamic_filter = None

# Additional info on config_optimizer params and tuning strategies can be found at
# torch_npu/_inductor/docs/feature/autotuning_optimization/config_optimizer_params.md
# Activate the config optimizer - default=0
fasta_config_optimizer = os.getenv("FASTA_CONFIG_OPTIMIZER", "0") == "1"
# Activate the autotune statistics - default=0
fasta_autotune_stats = os.getenv("FASTA_AUTOTUNE_STATS", "0") == "1"
# Enable MSPTI profiler for autotuning benchmarking - default=0
fasta_mspti_en = os.getenv("FASTA_MSPTI_EN", "0") == "1"
