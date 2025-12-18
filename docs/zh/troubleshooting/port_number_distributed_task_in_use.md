# 分布式任务端口号被占用

## 问题现象

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

## 故障根因

关键过程：刚启动分布式任务时，就报该错误。

根本原因分析：运行的分布式任务中断后有残留进程，或者环境中存在同样端口号的进程，导致该端口号被占用。

## 处理方法


<table><tbody><tr id="row133331920165614"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p83339201562">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p19428111123714">无</p>
</td>
</tr>
<tr id="row58261416152019"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p78261916142014">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p24271511173717">端口号被占用</p>
</td>
</tr>
<tr id="row1082711617201"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1782741619205">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><p id="p15427111113718">环境中已有进程占用该端口号</p>
</td>
</tr>
<tr id="row1474663022115"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p774617303213">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p15426111163718">端口号被占用，导致无法正常通信，进而使分布式任务失败。</p>
</td>
</tr>
<tr id="row19915122652114"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p1791515262213">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><p id="p142311119376">查看分布式代码所有的端口号，并修改为其他端口号。</p>
<p id="p16491955122416">查看并修改脚本中os.environ['MASTER_PORT']配置的端口号，例如：</p>
<div class="p" id="p97898531692">修改前<pre class="screen" id="screen9561924153017">os.environ['MASTER_PORT'] =<em id="i18442204115219"> '</em><em id="i1794513189101"><strong id="b1662431622710">29500</strong></em><em id="i7442154152118">'</em></pre>
</div>
<div class="p" id="p7671021193018">修改后<pre class="screen" id="screen7154153663015">os.environ['MASTER_PORT'] =<em id="i1164634520212"> '<strong id="b7714182014276">29580</strong>'</em></pre>
</div>
</td>
</tr>
<tr id="row1356182417228"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p175662413229">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p164198119376">无需操作</p>
</td>
</tr>
</tbody>
</table>

