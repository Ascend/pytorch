# 分布式任务端口号被占用

## 问题现象描述

回显信息中存在关键字“**Address already in use**”，类似如下打印信息：

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

## 原因分析

刚启动分布式任务时报错“Address already in use”。

运行的分布式任务中断后有残留进程，或者环境中存在同样端口号的进程，导致该端口号被占用，无法正常通信，进而使分布式任务失败。

## 解决措施

查看分布式代码所有的端口号，并修改为其他端口号。

查看并修改脚本中os.environ['MASTER_PORT']配置的端口号，例如：

修改前
```
os.environ['MASTER_PORT'] = '29500'
```
修改后
```
os.environ['MASTER_PORT'] = '29580'
```

