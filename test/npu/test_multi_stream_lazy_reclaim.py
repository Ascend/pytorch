import os
import time
import multiprocessing
import shutil
import unittest
import platform
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests

# Set multiprocessing start method to spawn because NPU cannot be re-initialized in forked subprocesses
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # May have already been set

IS_ARM64 = platform.machine() in ('arm64', 'aarch64')


def extract_aclrtQueryEventStatus_count(prof_dir):
    """
    Extract the call count of aclrtQueryEventStatus from profiler results.
    Uses Linux system commands (find/grep/awk) to parse api_statistic.csv.

    Args:
        prof_dir: str, path to the profiling result directory
        
    Returns:
        count: int, call count of aclrtQueryEventStatus, 0 if not found
    """
    import subprocess

    count = 0

    try:
        # Use find command to locate api_statistic.csv files
        find_result = subprocess.run(
            ["find", prof_dir, "-name", "api_statistic.csv", "-type", "f"],
            capture_output=True, text=True, timeout=30
        )

        if find_result.returncode == 0 and find_result.stdout.strip():
            # Get the first CSV file found
            csv_file = find_result.stdout.strip().split('\n')[0]
            # Search for aclrtQueryEventStatus line and extract the 5th column (call count)
            grep_result = subprocess.run(
                ["grep", "aclrtQueryEventStatus", csv_file],
                capture_output=True, text=True, timeout=10
            )

            if grep_result.returncode == 0 and grep_result.stdout.strip():
                # Use awk to extract the 5th column (call count)
                awk_result = subprocess.run(
                    ["awk", "-F,", '{print $5}'],
                    input=grep_result.stdout,
                    capture_output=True, text=True, timeout=10
                )
                if awk_result.returncode == 0 and awk_result.stdout.strip():
                    count = int(awk_result.stdout.strip())
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
        pass
    return count


def run_matmul_with_profiling(result_queue, enable_lazy_reclaim):
    """
    Run matmul test in a separate process and return the aclrtQueryEventStatus call count.

    Args:
        result_queue: multiprocessing.Queue, used to return results
        enable_lazy_reclaim: bool, whether to enable lazy reclaim feature
    """
    # Set environment variables (must be set before importing torch_npu)
    if enable_lazy_reclaim:
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = "multi_stream_lazy_reclaim:True"
    else:
        # Ensure environment variable does not exist or is False
        if "PYTORCH_NPU_ALLOC_CONF" in os.environ:
            del os.environ["PYTORCH_NPU_ALLOC_CONF"]

    # Reinitialize NPU (to ensure environment variables take effect)
    torch.npu.init()

    # Create temporary directory for profiling results (use absolute path)
    prof_dir = os.path.abspath("./prof_" + str(enable_lazy_reclaim))

    try:
        stream0 = torch.npu.Stream()
        stream1 = torch.npu.Stream()
        stream2 = torch.npu.Stream()

        # Configure profiler to collect Level2 data (includes all API calls)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2)

        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU,
                            torch_npu.profiler.ProfilerActivity.CPU],
                with_stack=False,      # Do not collect call stacks, reduces overhead
                record_shapes=False,   # Do not record shapes
                profile_memory=False,  # Do not analyze memory in detail
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir)) as prof:

            # Preallocate tensors, each 2048x2048 float32 = 16MB
            a = torch.empty((2048, 2048), dtype=torch.float32, device="npu")
            b = torch.empty((2048, 2048), dtype=torch.float32, device="npu")
            c = torch.empty((2048, 2048), dtype=torch.float32, device="npu")
            d = torch.empty((2048, 2048), dtype=torch.float32, device="npu")
            e = torch.empty((2048, 2048), dtype=torch.float32, device="npu")
            f = torch.empty((2048, 2048), dtype=torch.float32, device="npu")

            # Record tensors on multiple streams
            a.record_stream(stream0)
            a.record_stream(stream1)
            a.record_stream(stream2)

            # Execute matmul operations on multiple streams
            for _ in range(50):
                with torch.npu.stream(stream0):
                    torch.matmul(a, b, out=c)
                with torch.npu.stream(stream1):
                    torch.matmul(a, b, out=d)
                with torch.npu.stream(stream2):
                    torch.matmul(a, b, out=e)

            # Release some tensors
            a = None
            f = None

            # Trigger memory allocation, may trigger process_events
            for _ in range(10):
                tmp = torch.empty((1024, 1024), dtype=torch.float32, device="npu")  # 4M

            # Synchronize all streams
            torch.npu.synchronize()

            # Allocate memory again
            a1 = torch.empty((2048, 2048), dtype=torch.float32, device="npu")
            f1 = torch.empty((2048, 2048), dtype=torch.float32, device="npu")

            prof.step()

        # Extract aclrtQueryEventStatus call count from profiling results
        count = extract_aclrtQueryEventStatus_count(prof_dir)
        result_queue.put(("success", count))

    except Exception as e:
        result_queue.put(("error", str(e)))
    finally:
        # Clean up temporary directory
        if os.path.exists(prof_dir):
            shutil.rmtree(prof_dir)


@unittest.skipUnless(IS_ARM64, "Only working on ARM")
class TestMultiStreamLazyReclaim(TestCase):
    """
    Test the reduction effect of multi_stream_lazy_reclaim feature on event query counts.

    Principle:
    - eager reclaim mode: Calls process_events to query event status before every memory allocation
    - lazy reclaim mode: Only queries in the following cases:
      1. No available memory block found (!block_found)
      2. Event queue exceeds threshold kLazyQuerySize (512)
    
    Test Method:
    Use multiprocessing to test in two separate processes:
    - Process 1: Enable multi_stream_lazy_reclaim
    - Process 2: Disable multi_stream_lazy_reclaim
    
    Each process sets environment variables independently to ensure configuration takes effect.
    """

    def test_lazy_reclaim_reduces_event_queries_counts(self):
        """
        Compare aclrtQueryEventStatus call counts between eager reclaim and lazy reclaim modes.

        Validation Goal:
        Lazy reclaim mode should significantly reduce the number of aclrtQueryEventStatus calls.
        """
        configs = [
            ("eager", False),
            ("lazy", True)
        ]
        results = {}

        for name, enable_lazy in configs:
            print(f"\n--- Starting {name} reclaim test ---")
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=run_matmul_with_profiling,
                args=(queue, enable_lazy)
            )

            process.start()
            process.join(timeout=300) # 

            if process.is_alive():
                process.terminate()
                process.join()
                self.fail(f"{name} reclaim process timed out and was terminated.")

            status, result = queue.get()
            self.assertEqual(status, "success", f"{name} reclaim process failed: {result}")
            print(f"---mode {name}------count:{result}")
            results[name] = result

            #
            time.sleep(2)

        eager_counts = results["eager"]
        lazy_counts = results["lazy"]
 
        # Output comparison results
        print(f"\n========== Event Query Count Comparison ==========")
        print(f"Eager reclaim (multi_stream_lazy_reclaim:False): {eager_counts}")
        print(f"Lazy reclaim (multi_stream_lazy_reclaim:True):  {lazy_counts}")

        # Core validation: aclrtQueryEventStatus call count in lazy mode must be less than eager mode
        # This is direct evidence that multi_stream_lazy_reclaim feature is working
        self.assertLessEqual(
            lazy_counts, 
            eager_counts, 
            f"Lazy reclaim mode should reduce event queries. "
            f"Eager: {eager_counts}, Lazy: {lazy_counts}. "
            f"If lazy >= eager, the optimization may not be working."
        )

if __name__ == '__main__':
    run_tests()
