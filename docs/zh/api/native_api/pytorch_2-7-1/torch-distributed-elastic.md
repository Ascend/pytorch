# torch.distributed.elastic

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.7/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.7/distributed.elastic.html)。

<div style="border:1px solid #d1d5da;margin:10px 0;padding:16px 20px;background-color:#f3f4f5;border-radius:.25rem">
<div style="margin: 8px 0"><font size="5"><b>目录</b></font></div>

- [Elastic Agent](#elastic-agent)
- [Multiprocessing](#multiprocessing)
- [Error Propagation](#error-propagation)
- [Rendezvous](#rendezvous)
- [Expiration Timers](#expiration-timers)
- [Metrics](#metrics)
- [Events](#events)
- [Control Plane](#control-plane)

</div>

<div style="display:none;">

## &#8203;torch.distributed.elastic

</div>

## Elastic Agent

### <code><i>class</i></code> torch.distributed.elastic.agent.server.ElasticAgent

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.ElasticAgent](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.ElasticAgent)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get_worker_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.ElasticAgent.get_worker_group](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.ElasticAgent.get_worker_group)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">run()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.ElasticAgent.run](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.ElasticAgent.run)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.WorkerSpec

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerSpec](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerSpec)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get_entrypoint_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerSpec.get_entrypoint_name](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerSpec.get_entrypoint_name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.WorkerState

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerState](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerState)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">is_running()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerState.is_running](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerState.is_running)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.Worker

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.Worker](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.Worker)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.WorkerGroup

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerGroup](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerGroup)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.SimpleElasticAgent

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">_assign_worker_ranks()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._assign_worker_ranks](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._assign_worker_ranks)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">_exit_barrier()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._exit_barrier](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._exit_barrier)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">_initialize_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._initialize_workers](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._initialize_workers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">_monitor_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._monitor_workers](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._monitor_workers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">_rendezvous()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._rendezvous](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._rendezvous)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">_restart_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._restart_workers](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._restart_workers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">_shutdown()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._shutdown](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._shutdown)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">_start_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._start_workers](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._start_workers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">_stop_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._stop_workers](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._stop_workers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.api.RunResult

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.api.RunResult](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.api.RunResult)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">start()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.start](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.start)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">stop()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.stop](https://pytorch.org/docs/2.7/elastic/agent.html#torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.stop)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

## Multiprocessing

### torch.distributed.elastic.multiprocessing.start_processes

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.start_processes](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.start_processes)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.PContext

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.PContext](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.PContext)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.MultiprocessContext

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.MultiprocessContext](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.MultiprocessContext)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.SubprocessContext

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.SubprocessContext](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.SubprocessContext)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.RunProcsResult

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.RunProcsResult](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.RunProcsResult)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">reify()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs.reify](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs.reify)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.LogsDest

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.LogsDest](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.LogsDest)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.LogsSpecs

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.LogsSpecs](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.LogsSpecs)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">reify()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.LogsSpecs.reify](https://pytorch.org/docs/2.7/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.LogsSpecs.reify)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

## Error Propagation

### torch.distributed.elastic.multiprocessing.errors.record

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.errors.record](https://pytorch.org/docs/2.7/elastic/errors.html#torch.distributed.elastic.multiprocessing.errors.record)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.errors.ChildFailedError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.errors.ChildFailedError](https://pytorch.org/docs/2.7/elastic/errors.html#torch.distributed.elastic.multiprocessing.errors.ChildFailedError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.errors.ErrorHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.errors.ErrorHandler](https://pytorch.org/docs/2.7/elastic/errors.html#torch.distributed.elastic.multiprocessing.errors.ErrorHandler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.errors.ProcessFailure

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.errors.ProcessFailure](https://pytorch.org/docs/2.7/elastic/errors.html#torch.distributed.elastic.multiprocessing.errors.ProcessFailure)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## Rendezvous

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.RendezvousParameters

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousParameters](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousParameters)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousParameters.get](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousParameters.get)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_as_bool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_bool](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_bool)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_as_int()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_int](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_int)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousError](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousGracefulExitError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousGracefulExitError](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousGracefulExitError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.rendezvous.dynamic_rendezvous.create_handler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.create_handler](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.create_handler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.create_backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.create_backend](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.create_backend)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.create_backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.create_backend](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.create_backend)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvousHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvousHandler](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvousHandler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.etcd_store.EtcdStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.add](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.add)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">check()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.check](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.check)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.get](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.get)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.set](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.set)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.wait](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.wait)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.etcd_server.EtcdServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_server.EtcdServer](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_server.EtcdServer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：不支持
- <term>Atlas A3 训练系列产品</term>：不支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.RendezvousHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get_backend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.get_backend](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.get_backend)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_run_id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.get_run_id](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.get_run_id)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">is_closed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.is_closed](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.is_closed)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">next_rendezvous()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.next_rendezvous](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.next_rendezvous)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">num_nodes_waiting()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.num_nodes_waiting](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.num_nodes_waiting)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_closed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.set_closed](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.set_closed)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">shutdown()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.shutdown](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.shutdown)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">use_agent_store()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.use_agent_store](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.use_agent_store)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.RendezvousInfo

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousInfo](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousInfo)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousClosedError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousClosedError](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousClosedError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousTimeoutError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousTimeoutError](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousTimeoutError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousConnectionError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousConnectionError](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousConnectionError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousStateError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousStateError](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousStateError)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">build()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo.build](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo.build)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">from_backend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler.from_backend](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler.from_backend)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.get_state](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.get_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.name](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.set_state](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.set_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">close()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.close](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.close)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">heartbeat()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.heartbeat](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.heartbeat)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">join()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.join](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.join)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">last_call()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.last_call](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.last_call)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.get_state](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.get_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.name](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.set_state](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.set_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.get_state](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.get_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.name](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.name)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.set_state](https://pytorch.org/docs/2.7/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.set_state)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

## Expiration Timers

### torch.distributed.elastic.timer.configure

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.configure](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.configure)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.timer.expires

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.expires](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.expires)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.LocalTimerServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.LocalTimerServer](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.LocalTimerServer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.LocalTimerClient

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.LocalTimerClient](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.LocalTimerClient)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.FileTimerServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.FileTimerServer](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.FileTimerServer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.FileTimerClient

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.FileTimerClient](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.FileTimerClient)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.TimerRequest

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerRequest](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.TimerRequest)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.TimerServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerServer](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.TimerServer)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">clear_timers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerServer.clear_timers](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.TimerServer.clear_timers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">get_expired_timers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerServer.get_expired_timers](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.TimerServer.get_expired_timers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">register_timers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerServer.register_timers](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.TimerServer.register_timers)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.TimerClient

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerClient](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.TimerClient)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

> <font size="3">acquire()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerClient.acquire](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.TimerClient.acquire)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

> <font size="3">release()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerClient.release](https://pytorch.org/docs/2.7/elastic/timer.html#torch.distributed.elastic.timer.TimerClient.release)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

</div>

## Metrics

### <code><i>class</i></code> torch.distributed.elastic.metrics.api.MetricHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.api.MetricHandler](https://pytorch.org/docs/2.7/elastic/metrics.html#torch.distributed.elastic.metrics.api.MetricHandler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.metrics.api.ConsoleMetricHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.api.ConsoleMetricHandler](https://pytorch.org/docs/2.7/elastic/metrics.html#torch.distributed.elastic.metrics.api.ConsoleMetricHandler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.metrics.api.NullMetricHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.api.NullMetricHandler](https://pytorch.org/docs/2.7/elastic/metrics.html#torch.distributed.elastic.metrics.api.NullMetricHandler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.metrics.configure

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.configure](https://pytorch.org/docs/2.7/elastic/metrics.html#torch.distributed.elastic.metrics.configure)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.metrics.prof

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.prof](https://pytorch.org/docs/2.7/elastic/metrics.html#torch.distributed.elastic.metrics.prof)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.metrics.put_metric

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.put_metric](https://pytorch.org/docs/2.7/elastic/metrics.html#torch.distributed.elastic.metrics.put_metric)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## Events

### torch.distributed.elastic.events.record

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.record](https://pytorch.org/docs/2.7/elastic/events.html#torch.distributed.elastic.events.record)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.events.get_logging_handler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.get_logging_handler](https://pytorch.org/docs/2.7/elastic/events.html#torch.distributed.elastic.events.get_logging_handler)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.events.api.Event

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.api.Event](https://pytorch.org/docs/2.7/elastic/events.html#torch.distributed.elastic.events.api.Event)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.events.api.EventSource

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.api.EventSource](https://pytorch.org/docs/2.7/elastic/events.html#torch.distributed.elastic.events.api.EventSource)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### <code><i>class</i></code> torch.distributed.elastic.events.api.EventMetadataValue

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.api.EventMetadataValue](https://pytorch.org/docs/2.7/elastic/events.html#torch.distributed.elastic.events.api.EventMetadataValue)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

### torch.distributed.elastic.events.construct_and_record_rdzv_event

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.construct_and_record_rdzv_event](https://pytorch.org/docs/2.7/elastic/events.html#torch.distributed.elastic.events.construct_and_record_rdzv_event)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>

## Control Plane

### torch.distributed.elastic.control_plane.worker_main

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.control_plane.worker_main](https://pytorch.org/docs/2.7/elastic/control_plane.html#torch.distributed.elastic.control_plane.worker_main)

**产品支持情况**：

- <term>Atlas A2 训练系列产品</term>：支持
- <term>Atlas A3 训练系列产品</term>：支持
- <term>Ascend 950DT</term>：不支持

</div>
