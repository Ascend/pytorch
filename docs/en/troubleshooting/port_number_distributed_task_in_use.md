# Port Number for Distributed Tasks Occupied

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-12T08:24:35.350Z pushedAt=2026-06-12T11:22:41.052Z -->

## Symptom

The keyword **Address already in use** appears in the output, similar to the following printout:

```ColdFusion
torch.distributed.run: [WARNING] *****************************************
[W socket.cpp:436] [c10d] The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use).
[W socket.cpp:436] [c10d] The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
[E socket.cpp:472] [c10d] The server socket has failed to listen on any local network address.
Traceback (most recent call last):
  File "/root/miniconda3/envs/pt2.1/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
    return f(*args, **kwargs)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/run.py", line 806, in main
    run(args)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/run.py", line 797, in run
    elastic_launch(
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 255, in launch_agent
    result = agent.run()
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/metrics/api.py", line 124, in wrapper
    result = f(*args, **kwargs)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 736, in run
    result = self._invoke_run(role)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 871, in _invoke_run
    self._initialize_workers(self._worker_group)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/metrics/api.py", line 124, in wrapper
    result = f(*args, **kwargs)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 705, in _initialize_workers
    self._rendezvous(worker_group)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/metrics/api.py", line 124, in wrapper
    result = f(*args, **kwargs)
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/agent/server/api.py", line 546, in _rendezvous
    store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
  File "/root/miniconda3/envs/pt2.1/lib/python3.8/site-packages/torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py", line 54, in next_rendezvous
    self._store = TCPStore(  # type: ignore[call-arg]
RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use). The server socket has failed to bind to ?UNKNOWN? (errno: 98 - Address already in use).
```

## Possible Cause

The error **Address already in use** is reported when the distributed task is just started.

After the running distributed task is interrupted, there are residual processes, or there are processes using the same port number in the environment, causing the port number to be occupied and normal communication to fail, which in turn causes the distributed task to fail.

## Solution

Check all port numbers in the code for distributed tasks and change them to other port numbers.

Check and modify the port number configured in `os.environ['MASTER_PORT']` in the script, for example:

Before modification

```python
os.environ['MASTER_PORT'] = '29500'
```

After modification

```python
os.environ['MASTER_PORT'] = '29580'
```
