#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_pytorch_coverage.sh —— PyTorch 源码测试覆盖率采集脚本
#
# 仅供 .github/workflows/pytorch_coverage_nightly.yml 定时流水线调用，
# 参考 run_npu_tests.sh（torch_npu 测试覆盖率脚本）改写而来，核心差异：
#   1. 用例来源：从 pr_test_whitelist.yml（类别驱动的白名单配置，
#      core/tensor/distributed/graph/others 五类）解析出全部测试文件；
#   2. 测试主体：pytorch 源码仓库的 test/ 目录下的测试
#      （在 pytorch 源码树上执行，而不是 torch_npu 仓库的测试）；
#   3. 覆盖率目标：固定统计 torch（pytorch 安装包）的 Python 源码。
#
# 固化行为（不再提供开关）：
#   - 始终采集覆盖率：逐测试独立覆盖数据目录（互相不覆盖）+ 失败 FAILED 标记
#   - 结束时始终合并覆盖数据，生成 .coverage / coverage.xml
#   - 覆盖率模式下打包源码快照，保证报告生成时源码与数据一致
#   - 日志 / 汇总 / failed_cases.json
#   - 独立进程组 + 超时强杀 + 进程组清理（防 worker 进程残留占 NPU）
#
# 用法（流水线调用方式）：
#   ./run_pytorch_coverage.sh --test-dir <pytorch>/test [--category c1,c2]
#                             [--timeout <秒>]
#
#   选项：
#       --category <names>   只跑指定类别，逗号分隔（如 core,tensor），默认全部类别
#       --test-dir <path>    pytorch 源码的 test 目录，
#                            默认 ${GITHUB_WORKSPACE:-<仓库根>}/pytorch/test
#       --timeout <秒>       每个测试超时时间，默认 600（10 分钟），超时强杀并标 FAILED
#
#   环境变量：
#       OUT_ROOT     所有产物统一输出目录，默认 <仓库根>/outputs（流水线传入）
#       PYTHON_BIN   python 解释器，默认 python（流水线传 python3）
#
# 输出结构：
#   outputs/
#   ├── convstub/torch/              # 打包的源码（与运行环境一致）
#   │   ├── torch/                   #   安装包 torch/ 的 Python 源码
#   │   └── test/                    #   pytorch 的 test/ 目录
#   └── torch@YYYYMMDD/              # 覆盖率数据（每次运行独立时间戳目录）
#       ├── <类别>__<测试扁平名>/covdata/
#       │   ├── coverage             #   该测试独立的覆盖数据
#       │   └── FAILED               #   失败标记（仅失败时存在）
#       ├── logs/
#       │   ├── <序号>-<扁平名>.log
#       │   └── failed_cases.json
#       ├── .coverage                # 合并后的总覆盖数据
#       └── coverage.xml             # XML 报告（供覆盖率平台消费）
# ============================================================================

# ---- 运行时开关与输入 -----------------------------------------------------

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # 本脚本所在仓库根目录

config_file="${repo_root}/.github/config/pr_test_whitelist.yml"  # 白名单（固定，不接受覆盖）
category_filter=""                                               # 类别过滤（空=全部）
test_dir="${GITHUB_WORKSPACE:-${repo_root}}/pytorch/test"        # pytorch 测试目录
source_pkg="torch"                                               # 覆盖率统计目标包（固定）
device_env="privateuse1"                                         # 设备类型（NPU，固定）
timeout_seconds=600                                              # 每个测试超时秒数
python_bin="${PYTHON_BIN:-python}"                               # python 解释器（流水线可覆盖）

# 解析命令行参数：只识别流水线用到的选项，未知选项直接报错
while [ "$#" -gt 0 ]; do
  case "$1" in
    --category) category_filter="$2"; shift 2 ;;
    --test-dir) test_dir="$2";        shift 2 ;;
    --timeout)  timeout_seconds="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)  echo "Unexpected positional argument: $1 (targets come from ${config_file})" >&2; exit 2 ;;
  esac
done

# 基础校验：白名单、pytorch 测试目录、python 都必须就位
if [ ! -f "${config_file}" ]; then
  echo "ERROR: whitelist config not found: ${config_file}" >&2
  exit 1
fi
if [ ! -d "${test_dir}" ]; then
  echo "ERROR: pytorch test dir not found: ${test_dir} (use --test-dir)" >&2
  exit 1
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "ERROR: ${python_bin} not found in PATH" >&2
  exit 1
fi

# ---- 关键输出路径 ----------------------------------------------------------

out_root="${OUT_ROOT:-${repo_root}/outputs}"        # 所有产物统一输出根目录

# 源码快照目录：outputs/convstub/torch
src_snapshot_dir="${out_root}/convstub/${source_pkg}"

# 覆盖率数据目录：outputs/torch@YYYYMMDD（每次运行独立时间戳目录）
timestamp="$(date +%Y%m%d)"
covdata_root="${out_root}/${source_pkg}@${timestamp}"

log_dir="${covdata_root}/logs"                      # 日志/失败清单子目录
mkdir -p "${log_dir}"

# ---- 从白名单解析测试目标 --------------------------------------------------

# read_targets <config> <category_filter>
# 用 python3+yaml 解析类别驱动的白名单，输出 "类别<TAB>文件路径" 行。
# 兼容旧格式（顶层 whitelist:）：旧格式没有类别名，统一归入 regular。
read_targets() {
  python3 - "$1" "$2" <<'PYEOF'
import sys
try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")
data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
cat_filter = set(sys.argv[2].split(",")) if sys.argv[2] else None
cats = data.get("categories")
if cats:
    for name, cfg in cats.items():
        if cat_filter and name not in cat_filter:
            continue
        for f in (cfg or {}).get("files", []):
            print(f"{name}\t{f}")
elif "whitelist" in data:
    for f in data["whitelist"]:
        print(f"regular\t{f}")
PYEOF
}

# 解析目标到数组；每条形如 "类别|文件路径"
targets=()
while IFS=$'\t' read -r cat file; do
  [ -n "${cat:-}" ] && targets+=("${cat}|${file}")
done < <(read_targets "${config_file}" "${category_filter}")

if [ "${#targets[@]}" -eq 0 ]; then
  echo "ERROR: no test targets resolved from ${config_file} (category filter: '${category_filter}')" >&2
  exit 1
fi

echo "=== Targets: ${#targets[@]} file(s) from ${config_file} ==="
echo "=== test-dir: ${test_dir}  source: ${source_pkg}  device-env: ${device_env} ==="

# ---- 汇总状态（在脚本结束时打印/退出） -----------------------------------

test_results=()      # 每条： "<类别|文件>|PASSED|日志文件" 或 "...|FAILED|日志文件"
failed_logs=()       # 每条： "<类别|文件>|日志文件"（仅失败的）
test_index=0         # 已运行测试的序号（用于日志文件名）
overall_status=0     # 最终退出码（0=全部通过）

# ---- 源码打包 -------------------------------------------------------------

# dump_source_snapshot
# 把当前 Python 环境中实际使用的 pytorch（torch 包）源码以及 pytorch 的
# test/ 目录复制到 outputs/convstub/torch/，保证后续生成报告时源码与
# 覆盖率数据严格一致。
dump_source_snapshot() {
  mkdir -p "${src_snapshot_dir}"

  # 1) 打包当前环境里的 torch 安装目录（与统计目标 torch 对应）
  local torch_path
  torch_path="$("${python_bin}" -c 'import torch, os; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)"
  if [ -n "${torch_path}" ] && [ -d "${torch_path}" ]; then
    echo "=== Dumping torch source: ${torch_path} -> ${src_snapshot_dir}/torch ==="
    rm -rf "${src_snapshot_dir}/torch"
    cp -a "${torch_path}" "${src_snapshot_dir}/torch"
  else
    echo "WARNING: 无法定位 torch 安装路径，跳过源码打包" >&2
  fi

  # 2) 打包 pytorch 的 test/ 目录（测试脚本本身也可能被覆盖率统计）
  echo "=== Dumping test source: ${test_dir} -> ${src_snapshot_dir}/test ==="
  rm -rf "${src_snapshot_dir}/test"
  cp -a "${test_dir}" "${src_snapshot_dir}/test"
}

# ---- 工具函数 -------------------------------------------------------------

# flatten_name <category> <file>
# 把 "类别 + 测试文件" 拍平成合法的目录/文件名：去掉 .py、把 / 换成 __。
# 例：core|test/nn/test_dropout.py -> core__test__nn__test_dropout
flatten_name() {
  local n="$2"; n="${n%.py}"; n="${n//\//__}"
  printf '%s__%s' "$1" "$n"
}

# setup_coverage <flattened>
# 为某个测试初始化独立的覆盖数据目录，并把 COVERAGE_FILE 指到该目录。
# 每个测试一个目录，保证互相不覆盖；失败时该目录里会多一个 FAILED 标记。
setup_coverage() {
  local covdata_dir="${covdata_root}/$1/covdata"
  mkdir -p "${covdata_dir}"
  export COVERAGE_FILE="${covdata_dir}/coverage"
  echo "  COVERAGE_FILE: ${COVERAGE_FILE}"
}

# setup_env <file>
# 复现 CI 中为每个 pytorch 测试准备的环境：
#   - TORCH_DEVICE_BACKEND_AUTOLOAD=1  torch 导入时自动加载 torch_npu 后端（注册 privateuse1）
#   - PYTORCH_TESTING_DEVICE_ONLY_FOR  让设备参数化测试类只按指定设备实例化（NPU 侧用例）
#   - PYTHONPATH 加入 test/ 目录和测试所在子目录，让测试内的同级模块导入能解析
setup_env() {
  local td; td="$(dirname "$1")"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=1
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="${device_env}"
  # 同时加入 test/ 根目录与测试所在子目录
  export PYTHONPATH="${test_dir}:${test_dir}/${td}${PYTHONPATH:+:${PYTHONPATH}}"
}

# terminate_process_group <pgid>
# 清理某个进程组：先 TERM，10 次轮询后仍活着就 KILL。
# 目的：pytorch/distributed 的 spawn worker、inductor 编译子进程可能比主进程
# 活得久，不清理会占 NPU 显存和端口。
terminate_process_group() {
  local pg="$1"
  if ! kill -0 -- "-${pg}" 2>/dev/null; then return; fi
  kill -TERM -- "-${pg}" 2>/dev/null || true
  for _ in {1..10}; do
    if ! kill -0 -- "-${pg}" 2>/dev/null; then return; fi
    sleep 0.5
  done
  kill -KILL -- "-${pg}" 2>/dev/null || true
}

# run_logged_command <log_file> <cmd...>
# 执行命令并把输出写入日志文件，同时用 tail 实时刷到终端。
# 用 setsid 把命令放进独立进程组，便于后续整组清理；返回命令的退出码。
run_logged_command() {
  local log_file="$1"; shift
  local command_pid process_group="" tail_pid
  : > "${log_file}"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" > "${log_file}" 2>&1 &
    command_pid=$!
    process_group="${command_pid}"
  else
    "$@" > "${log_file}" 2>&1 &
    command_pid=$!
  fi
  tail --pid="${command_pid}" -n +1 -f "${log_file}" &
  tail_pid=$!
  wait "${command_pid}"
  local status=$?
  if [ -n "${process_group}" ]; then terminate_process_group "${process_group}"; fi
  wait "${tail_pid}" || true
  return "${status}"
}

# run_test_target <category> <file>
# 运行单个 pytorch 测试文件：用 coverage run 采集 torch 源码覆盖率。
# 测试在 pytorch 的 test/ 目录下执行（与上游 run_test.py 一致）；
# 失败时写 FAILED 标记并记录。
run_test_target() {
  local category="$1" file="$2"
  test_index=$((test_index + 1))
  local flat log_file
  flat="$(flatten_name "${category}" "${file}")"
  log_file="${log_dir}/${test_index}-${flat}.log"
  echo "=== [${category}] Running target: ${file} ==="

  # bash -c 内先 cd 到 test 目录，再 exec 真正命令；这样测试的工作目录正确
  # （与 pytorch 上游 run_test.py 的执行方式一致）；-u 关闭输出缓冲，日志实时可见
  local script='cd "$1"; shift; exec "$@"'
  local status=0
  set +e
  setup_coverage "${flat}"
  setup_env "${file}"
  run_logged_command "${log_file}" bash -c "${script}" _ "${test_dir}" \
    timeout --kill-after=30 "${timeout_seconds}" \
    "${python_bin}" -m coverage run --source="${source_pkg}" --branch -u "${file}"
  status=$?
  set -e

  # 失败时在对应 covdata 目录写 FAILED 标记，供下游识别并丢弃残缺覆盖
  if [ "${status}" -ne 0 ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
  fi

  # 汇总结果
  if [ "${status}" -eq 0 ]; then
    test_results+=( "${category}|${file}|PASSED|${log_file}" )
  else
    test_results+=( "${category}|${file}|FAILED|${log_file}" )
    failed_logs+=( "${category}|${file}|${log_file}" )
    if [ "${overall_status}" -eq 0 ]; then
      overall_status="${status}"
    fi
  fi
}

# combine_coverage
# 合并所有测试的独立覆盖数据为一份 .coverage，并输出 XML 报告。
# 带 FAILED 标记的目录数据仍会参与合并（部分数据好过没有），
# 下游如需剔除失败用例的覆盖，可按 FAILED 标记过滤。
combine_coverage() {
  local data_files=()
  mapfile -t data_files < <(find "${covdata_root}" -mindepth 3 -maxdepth 3 -name coverage -type f \
    | sort)
  if [ "${#data_files[@]}" -eq 0 ]; then
    echo "=== Combine: no coverage data files found, skip ==="
    return
  fi
  echo "=== Combining ${#data_files[@]} coverage data file(s) ==="
  # 合并结果写到 covdata_root/.coverage，避免与各测试目录里的同名文件混淆
  COVERAGE_FILE="${covdata_root}/.coverage" "${python_bin}" -m coverage combine "${data_files[@]}"
  # XML 报告（供覆盖率平台消费）；只统计目标包，源码根指向快照，保证路径一致
  COVERAGE_FILE="${covdata_root}/.coverage" "${python_bin}" -m coverage xml \
    --include="*/${source_pkg}/*" -o "${covdata_root}/coverage.xml"
  echo "=== Combined coverage: ${covdata_root}/.coverage, report: ${covdata_root}/coverage.xml ==="
  # 终端摘要：只打印末尾 TOTAL 行，避免刷屏
  COVERAGE_FILE="${covdata_root}/.coverage" "${python_bin}" -m coverage report --include="*/${source_pkg}/*" | tail -n 5 || true
}

# print_summary
# 打印每个测试的 PASSED/FAILED 结果及其日志路径（按类别缩进展示）。
print_summary() {
  local result category file status log_file
  echo "=== TEST SUMMARY ==="
  for result in "${test_results[@]}"; do
    IFS='|' read -r category file status log_file <<< "${result}"
    echo "  [${category}] ${status}: ${file}  (log: ${log_file})"
  done
}

# print_failed_summary
# 汇总所有失败用例：列出失败目标与日志路径，打印每个失败日志的末尾片段，
# 并把失败用例清单写入日志目录下的 failed_cases.json。
print_failed_summary() {
  if [ "${#failed_logs[@]}" -eq 0 ]; then
    echo "=== Failed cases: none ==="
    return
  fi

  echo "=== FAILED CASES (${#failed_logs[@]}) ==="
  local failed category file log_file
  local json="[" i=0
  for failed in "${failed_logs[@]}"; do
    IFS='|' read -r category file log_file <<< "${failed}"
    echo "  - [${category}] ${file}  (log: ${log_file})"
    [ "${i}" -gt 0 ] && json+=","
    json+="{\"category\":\"${category}\",\"name\":\"${file}\",\"log\":\"${log_file}\"}"
    i=$((i + 1))
  done
  json+="]"
  echo "${json}" > "${log_dir}/failed_cases.json"
  echo "  failed_cases.json: ${log_dir}/failed_cases.json"

  # 打印每个失败日志的末尾片段，便于快速定位报错
  echo ""
  for failed in "${failed_logs[@]}"; do
    IFS='|' read -r category file log_file <<< "${failed}"
    echo "----- tail of [${category}] ${file} -----"
    tail -n 30 "${log_file}" 2>/dev/null || true
    echo ""
  done
}

# ---- 主流程 ---------------------------------------------------------------

# 先固化当前运行环境的源码快照，避免后续生成报告时源码漂移
dump_source_snapshot

# 逐个运行测试（pytorch 测试涉及设备/编译子进程，串行执行最稳妥；
# distributed 类测试本身就要求串行，这里对全部类别统一串行处理）
for target in "${targets[@]}"; do
  IFS='|' read -r category file <<< "${target}"
  run_test_target "${category}" "${file}"
done

print_summary
print_failed_summary

# 合并覆盖数据并生成 XML 报告
combine_coverage

exit "${overall_status}"
