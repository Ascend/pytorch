# torch.distributed.elastic

> [!NOTE]
>
> - 若API标注有“限制与说明”，表示该API在昇腾NPU上的支持度和原生版本存在差异，请务必查阅具体说明，以确保适配昇腾NPU平台。
> - 部分API虽在[PyTorch社区文档](https://pytorch.org/docs/2.14/)中存在，但未收录于本支持清单。此类API尚未验证，请谨慎使用。我们将持续进行验证工作，并在验证完成后更新文档。
> - 产品支持范围说明：文档中仅提供已验证的产品信息，未经过验证产品暂不纳入。
> - 目录下罗列的模块和原生文档一致，对于模块的相关说明请查看原生文档[LINK](https://docs.pytorch.org/docs/2.14/distributed.elastic.html)。

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

**原生文档**：[torch.distributed.elastic.agent.server.ElasticAgent](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.ElasticAgent)

**产品支持情况**：

<!-- npu="910b" id1 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id3 -->

> <font size="3">get_worker_group()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.ElasticAgent.get_worker_group](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.ElasticAgent.get_worker_group)

**产品支持情况**：

<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id5 -->
<!-- npu="950" id6 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id6 -->

</div>

> <font size="3">run()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.ElasticAgent.run](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.ElasticAgent.run)

**产品支持情况**：

<!-- npu="910b" id7 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id7 -->
<!-- npu="A3" id8 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id9 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.WorkerSpec

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerSpec](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerSpec)

**产品支持情况**：

<!-- npu="910b" id10 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id10 -->
<!-- npu="A3" id11 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id12 -->

> <font size="3">get_entrypoint_name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerSpec.get_entrypoint_name](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerSpec.get_entrypoint_name)

**产品支持情况**：

<!-- npu="910b" id13 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id15 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.WorkerState

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerState](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerState)

**产品支持情况**：

<!-- npu="910b" id16 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id16 -->
<!-- npu="A3" id17 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id17 -->
<!-- npu="950" id18 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id18 -->

> <font size="3">is_running()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerState.is_running](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerState.is_running)

**产品支持情况**：

<!-- npu="910b" id19 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id19 -->
<!-- npu="A3" id20 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id20 -->
<!-- npu="950" id21 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id21 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.Worker

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.Worker](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.Worker)

**产品支持情况**：

<!-- npu="910b" id22 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id22 -->
<!-- npu="A3" id23 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id23 -->
<!-- npu="950" id24 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id24 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.WorkerGroup

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.WorkerGroup](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.WorkerGroup)

**产品支持情况**：

<!-- npu="910b" id25 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id25 -->
<!-- npu="A3" id26 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id27 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.local_elastic_agent.LocalElasticAgent)

**产品支持情况**：

<!-- npu="910b" id28 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id28 -->
<!-- npu="A3" id29 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id29 -->
<!-- npu="950" id30 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id30 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.SimpleElasticAgent

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent)

**产品支持情况**：

<!-- npu="910b" id31 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id31 -->
<!-- npu="A3" id32 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id32 -->
<!-- npu="950" id33 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id33 -->

> <font size="3">_assign_worker_ranks()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._assign_worker_ranks](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._assign_worker_ranks)

**产品支持情况**：

<!-- npu="910b" id34 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id34 -->
<!-- npu="A3" id35 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id35 -->
<!-- npu="950" id36 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id36 -->

</div>

> <font size="3">_exit_barrier()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._exit_barrier](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._exit_barrier)

**产品支持情况**：

<!-- npu="910b" id37 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id37 -->
<!-- npu="A3" id38 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id38 -->
<!-- npu="950" id39 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id39 -->

</div>

> <font size="3">_initialize_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._initialize_workers](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._initialize_workers)

**产品支持情况**：

<!-- npu="910b" id40 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id40 -->
<!-- npu="A3" id41 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id41 -->
<!-- npu="950" id42 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id42 -->

</div>

> <font size="3">_monitor_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._monitor_workers](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._monitor_workers)

**产品支持情况**：

<!-- npu="910b" id43 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id44 -->
<!-- npu="950" id45 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id45 -->

</div>

> <font size="3">_rendezvous()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._rendezvous](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._rendezvous)

**产品支持情况**：

<!-- npu="910b" id46 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id46 -->
<!-- npu="A3" id47 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id47 -->
<!-- npu="950" id48 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id48 -->

</div>

> <font size="3">_restart_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._restart_workers](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._restart_workers)

**产品支持情况**：

<!-- npu="910b" id49 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id49 -->
<!-- npu="A3" id50 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id50 -->
<!-- npu="950" id51 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id51 -->

</div>

> <font size="3">_shutdown()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._shutdown](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._shutdown)

**产品支持情况**：

<!-- npu="910b" id52 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id52 -->
<!-- npu="A3" id53 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id53 -->
<!-- npu="950" id54 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id54 -->

</div>

> <font size="3">_start_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._start_workers](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._start_workers)

**产品支持情况**：

<!-- npu="910b" id55 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id55 -->
<!-- npu="A3" id56 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id56 -->
<!-- npu="950" id57 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id57 -->

</div>

> <font size="3">_stop_workers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.SimpleElasticAgent._stop_workers](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.SimpleElasticAgent._stop_workers)

**产品支持情况**：

<!-- npu="910b" id58 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id58 -->
<!-- npu="A3" id59 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id59 -->
<!-- npu="950" id60 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id60 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.api.RunResult

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.api.RunResult](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.api.RunResult)

**产品支持情况**：

<!-- npu="910b" id61 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id61 -->
<!-- npu="A3" id62 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id62 -->
<!-- npu="950" id63 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id63 -->

</div>

### torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.health_check_server.create_healthcheck_server)

**产品支持情况**：

<!-- npu="910b" id64 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id64 -->
<!-- npu="A3" id65 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id65 -->
<!-- npu="950" id66 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id66 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer)

**产品支持情况**：

<!-- npu="910b" id67 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id67 -->
<!-- npu="A3" id68 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id68 -->
<!-- npu="950" id69 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id69 -->

> <font size="3">start()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.start](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.start)

**产品支持情况**：

<!-- npu="910b" id70 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id70 -->
<!-- npu="A3" id71 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id71 -->
<!-- npu="950" id72 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id72 -->

</div>

> <font size="3">stop()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.stop](https://pytorch.org/docs/2.14/elastic/agent.html#torch.distributed.elastic.agent.server.health_check_server.HealthCheckServer.stop)

**产品支持情况**：

<!-- npu="910b" id73 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id73 -->
<!-- npu="A3" id74 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id74 -->
<!-- npu="950" id75 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id75 -->

</div>

</div>

## Multiprocessing

### torch.distributed.elastic.multiprocessing.start_processes

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.start_processes](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.start_processes)

**产品支持情况**：

<!-- npu="910b" id76 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id76 -->
<!-- npu="A3" id77 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id77 -->
<!-- npu="950" id78 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id78 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.PContext

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.PContext](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.PContext)

**产品支持情况**：

<!-- npu="910b" id79 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id79 -->
<!-- npu="A3" id80 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id80 -->
<!-- npu="950" id81 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id81 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.MultiprocessContext

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.MultiprocessContext](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.MultiprocessContext)

**产品支持情况**：

<!-- npu="910b" id82 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id82 -->
<!-- npu="A3" id83 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id83 -->
<!-- npu="950" id84 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id84 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.SubprocessContext

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.SubprocessContext](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.SubprocessContext)

**产品支持情况**：

<!-- npu="910b" id85 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id85 -->
<!-- npu="A3" id86 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id86 -->
<!-- npu="950" id87 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id87 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.RunProcsResult

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.RunProcsResult](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.RunProcsResult)

**产品支持情况**：

<!-- npu="910b" id88 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id88 -->
<!-- npu="A3" id89 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id89 -->
<!-- npu="950" id90 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id90 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs)

**产品支持情况**：

<!-- npu="910b" id91 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id91 -->
<!-- npu="A3" id92 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id92 -->
<!-- npu="950" id93 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id93 -->

> <font size="3">reify()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs.reify](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.DefaultLogsSpecs.reify)

**产品支持情况**：

<!-- npu="910b" id94 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id94 -->
<!-- npu="A3" id95 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id95 -->
<!-- npu="950" id96 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id96 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.LogsDest

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.LogsDest](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.LogsDest)

**产品支持情况**：

<!-- npu="910b" id97 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id97 -->
<!-- npu="A3" id98 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id98 -->
<!-- npu="950" id99 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id99 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.api.LogsSpecs

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.LogsSpecs](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.LogsSpecs)

**产品支持情况**：

<!-- npu="910b" id100 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id100 -->
<!-- npu="A3" id101 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id101 -->
<!-- npu="950" id102 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id102 -->

> <font size="3">reify()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.api.LogsSpecs.reify](https://pytorch.org/docs/2.14/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.LogsSpecs.reify)

**产品支持情况**：

<!-- npu="910b" id103 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id103 -->
<!-- npu="A3" id104 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id104 -->
<!-- npu="950" id105 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id105 -->

</div>

</div>

## Error Propagation

### torch.distributed.elastic.multiprocessing.errors.record

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.errors.record](https://pytorch.org/docs/2.14/elastic/errors.html#torch.distributed.elastic.multiprocessing.errors.record)

**产品支持情况**：

<!-- npu="910b" id106 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id106 -->
<!-- npu="A3" id107 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id107 -->
<!-- npu="950" id108 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id108 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.errors.ChildFailedError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.errors.ChildFailedError](https://pytorch.org/docs/2.14/elastic/errors.html#torch.distributed.elastic.multiprocessing.errors.ChildFailedError)

**产品支持情况**：

<!-- npu="910b" id109 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id109 -->
<!-- npu="A3" id110 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id110 -->
<!-- npu="950" id111 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id111 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.errors.ErrorHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.errors.ErrorHandler](https://pytorch.org/docs/2.14/elastic/errors.html#torch.distributed.elastic.multiprocessing.errors.ErrorHandler)

**产品支持情况**：

<!-- npu="910b" id112 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id112 -->
<!-- npu="A3" id113 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id113 -->
<!-- npu="950" id114 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id114 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.multiprocessing.errors.ProcessFailure

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.multiprocessing.errors.ProcessFailure](https://pytorch.org/docs/2.14/elastic/errors.html#torch.distributed.elastic.multiprocessing.errors.ProcessFailure)

**产品支持情况**：

<!-- npu="910b" id115 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id115 -->
<!-- npu="A3" id116 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id116 -->
<!-- npu="950" id117 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id117 -->

</div>

## Rendezvous

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.RendezvousParameters

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousParameters](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousParameters)

**产品支持情况**：

<!-- npu="910b" id118 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id118 -->
<!-- npu="A3" id119 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id119 -->
<!-- npu="950" id120 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id120 -->

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousParameters.get](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousParameters.get)

**产品支持情况**：

<!-- npu="910b" id121 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id121 -->
<!-- npu="A3" id122 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id122 -->
<!-- npu="950" id123 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id123 -->

</div>

> <font size="3">get_as_bool()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_bool](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_bool)

**产品支持情况**：

<!-- npu="910b" id124 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id124 -->
<!-- npu="A3" id125 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id125 -->
<!-- npu="950" id126 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id126 -->

</div>

> <font size="3">get_as_int()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_int](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousParameters.get_as_int)

**产品支持情况**：

<!-- npu="910b" id127 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id127 -->
<!-- npu="A3" id128 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id128 -->
<!-- npu="950" id129 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id129 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandlerRegistry)

**产品支持情况**：

<!-- npu="910b" id130 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id130 -->
<!-- npu="A3" id131 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id131 -->
<!-- npu="950" id132 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id132 -->

</div>

### torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.registry.get_rendezvous_handler)

**产品支持情况**：

<!-- npu="910b" id133 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id133 -->
<!-- npu="A3" id134 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id134 -->
<!-- npu="950" id135 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id135 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousError](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousError)

**产品支持情况**：

<!-- npu="910b" id136 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id136 -->
<!-- npu="A3" id137 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id137 -->
<!-- npu="950" id138 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id138 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousGracefulExitError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousGracefulExitError](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousGracefulExitError)

**产品支持情况**：

<!-- npu="910b" id139 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id139 -->
<!-- npu="A3" id140 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id140 -->
<!-- npu="950" id141 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id141 -->

</div>

### torch.distributed.elastic.rendezvous.dynamic_rendezvous.create_handler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.create_handler](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.create_handler)

**产品支持情况**：

<!-- npu="910b" id142 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id142 -->
<!-- npu="A3" id143 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id143 -->
<!-- npu="950" id144 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id144 -->

</div>

### torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.create_backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.create_backend](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.create_backend)

**产品支持情况**：

<!-- npu="910b" id145 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id145 -->
<!-- npu="A3" id146 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id146 -->
<!-- npu="950" id147 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id147 -->

</div>

### torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.create_backend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.create_backend](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.create_backend)

**产品支持情况**：

<!-- npu="910b" id148 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id148 -->
<!-- npu="A3" id149 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id149 -->
<!-- npu="950" id150 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id150 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvousHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvousHandler](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvousHandler)

**产品支持情况**：

<!-- npu="910b" id151 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id151 -->
<!-- npu="A3" id152 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id152 -->
<!-- npu="950" id153 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id153 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.etcd_store.EtcdStore

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore)

**产品支持情况**：

<!-- npu="910b" id154 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id154 -->
<!-- npu="A3" id155 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id155 -->
<!-- npu="950" id156 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id156 -->

> <font size="3">add()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.add](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.add)

**产品支持情况**：

<!-- npu="910b" id157 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id157 -->
<!-- npu="A3" id158 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id158 -->
<!-- npu="950" id159 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id159 -->

</div>

> <font size="3">check()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.check](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.check)

**产品支持情况**：

<!-- npu="910b" id160 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id160 -->
<!-- npu="A3" id161 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id161 -->
<!-- npu="950" id162 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id162 -->

</div>

> <font size="3">get()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.get](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.get)

**产品支持情况**：

<!-- npu="910b" id163 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id163 -->
<!-- npu="A3" id164 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id164 -->
<!-- npu="950" id165 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id165 -->

</div>

> <font size="3">set()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.set](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.set)

**产品支持情况**：

<!-- npu="910b" id166 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id166 -->
<!-- npu="A3" id167 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id167 -->
<!-- npu="950" id168 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id168 -->

</div>

> <font size="3">wait()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.wait](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_store.EtcdStore.wait)

**产品支持情况**：

<!-- npu="910b" id169 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id169 -->
<!-- npu="A3" id170 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id170 -->
<!-- npu="950" id171 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id171 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.etcd_server.EtcdServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_server.EtcdServer](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_server.EtcdServer)

**产品支持情况**：

<!-- npu="910b" id172 -->
- <term>Atlas A2训练系列产品</term>：不支持
<!-- end id172 -->
<!-- npu="A3" id173 -->
- <term>Atlas A3训练系列产品</term>：不支持
<!-- end id173 -->
<!-- npu="950" id174 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id174 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.RendezvousHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler)

**产品支持情况**：

<!-- npu="910b" id175 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id175 -->
<!-- npu="A3" id176 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id176 -->
<!-- npu="950" id177 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id177 -->

> <font size="3">get_backend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.get_backend](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.get_backend)

**产品支持情况**：

<!-- npu="910b" id178 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id178 -->
<!-- npu="A3" id179 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id179 -->
<!-- npu="950" id180 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id180 -->

</div>

> <font size="3">get_run_id()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.get_run_id](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.get_run_id)

**产品支持情况**：

<!-- npu="910b" id181 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id181 -->
<!-- npu="A3" id182 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id182 -->
<!-- npu="950" id183 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id183 -->

</div>

> <font size="3">is_closed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.is_closed](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.is_closed)

**产品支持情况**：

<!-- npu="910b" id184 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id184 -->
<!-- npu="A3" id185 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id185 -->
<!-- npu="950" id186 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id186 -->

</div>

> <font size="3">next_rendezvous()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.next_rendezvous](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.next_rendezvous)

**产品支持情况**：

<!-- npu="910b" id187 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id187 -->
<!-- npu="A3" id188 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id188 -->
<!-- npu="950" id189 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id189 -->

</div>

> <font size="3">num_nodes_waiting()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.num_nodes_waiting](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.num_nodes_waiting)

**产品支持情况**：

<!-- npu="910b" id190 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id190 -->
<!-- npu="A3" id191 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id191 -->
<!-- npu="950" id192 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id192 -->

</div>

> <font size="3">set_closed()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.set_closed](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.set_closed)

**产品支持情况**：

<!-- npu="910b" id193 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id193 -->
<!-- npu="A3" id194 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id194 -->
<!-- npu="950" id195 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id195 -->

</div>

> <font size="3">shutdown()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.shutdown](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.shutdown)

**产品支持情况**：

<!-- npu="910b" id196 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id196 -->
<!-- npu="A3" id197 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id197 -->
<!-- npu="950" id198 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id198 -->

</div>

> <font size="3">use_agent_store()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousHandler.use_agent_store](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousHandler.use_agent_store)

**产品支持情况**：

<!-- npu="910b" id199 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id199 -->
<!-- npu="A3" id200 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id200 -->
<!-- npu="950" id201 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id201 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.RendezvousInfo

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.RendezvousInfo](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.RendezvousInfo)

**产品支持情况**：

<!-- npu="910b" id202 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id202 -->
<!-- npu="A3" id203 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id203 -->
<!-- npu="950" id204 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id204 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousClosedError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousClosedError](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousClosedError)

**产品支持情况**：

<!-- npu="910b" id205 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id205 -->
<!-- npu="A3" id206 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id206 -->
<!-- npu="950" id207 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id207 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousTimeoutError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousTimeoutError](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousTimeoutError)

**产品支持情况**：

<!-- npu="910b" id208 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id208 -->
<!-- npu="A3" id209 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id209 -->
<!-- npu="950" id210 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id210 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousConnectionError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousConnectionError](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousConnectionError)

**产品支持情况**：

<!-- npu="910b" id211 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id211 -->
<!-- npu="A3" id212 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id212 -->
<!-- npu="950" id213 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id213 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousStateError

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousStateError](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousStateError)

**产品支持情况**：

<!-- npu="910b" id214 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id214 -->
<!-- npu="A3" id215 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id215 -->
<!-- npu="950" id216 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id216 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo)

**产品支持情况**：

<!-- npu="910b" id217 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id217 -->
<!-- npu="A3" id218 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id218 -->
<!-- npu="950" id219 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id219 -->

> <font size="3">build()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo.build](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.api.RendezvousStoreInfo.build)

**产品支持情况**：

<!-- npu="910b" id220 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id220 -->
<!-- npu="A3" id221 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id221 -->
<!-- npu="950" id222 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id222 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler)

**产品支持情况**：

<!-- npu="910b" id223 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id223 -->
<!-- npu="A3" id224 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id224 -->
<!-- npu="950" id225 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id225 -->

> <font size="3">from_backend()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler.from_backend](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.DynamicRendezvousHandler.from_backend)

**产品支持情况**：

<!-- npu="910b" id226 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id226 -->
<!-- npu="A3" id227 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id227 -->
<!-- npu="950" id228 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id228 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend)

**产品支持情况**：

<!-- npu="910b" id229 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id229 -->
<!-- npu="A3" id230 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id230 -->
<!-- npu="950" id231 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id231 -->

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.get_state](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.get_state)

**产品支持情况**：

<!-- npu="910b" id232 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id232 -->
<!-- npu="A3" id233 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id233 -->
<!-- npu="950" id234 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id234 -->

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.name](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.name)

**产品支持情况**：

<!-- npu="910b" id235 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id235 -->
<!-- npu="A3" id236 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id236 -->
<!-- npu="950" id237 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id237 -->

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.set_state](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousBackend.set_state)

**产品支持情况**：

<!-- npu="910b" id238 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id238 -->
<!-- npu="A3" id239 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id239 -->
<!-- npu="950" id240 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id240 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout)

**产品支持情况**：

<!-- npu="910b" id241 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id241 -->
<!-- npu="A3" id242 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id242 -->
<!-- npu="950" id243 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id243 -->

> <font size="3">close()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.close](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.close)

**产品支持情况**：

<!-- npu="910b" id244 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id244 -->
<!-- npu="A3" id245 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id245 -->
<!-- npu="950" id246 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id246 -->

</div>

> <font size="3">heartbeat()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.heartbeat](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.heartbeat)

**产品支持情况**：

<!-- npu="910b" id247 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id247 -->
<!-- npu="A3" id248 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id248 -->
<!-- npu="950" id249 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id249 -->

</div>

> <font size="3">join()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.join](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.join)

**产品支持情况**：

<!-- npu="910b" id250 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id250 -->
<!-- npu="A3" id251 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id251 -->
<!-- npu="950" id252 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id252 -->

</div>

> <font size="3">last_call()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.last_call](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.RendezvousTimeout.last_call)

**产品支持情况**：

<!-- npu="910b" id253 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id253 -->
<!-- npu="A3" id254 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id254 -->
<!-- npu="950" id255 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id255 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend)

**产品支持情况**：

<!-- npu="910b" id256 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id256 -->
<!-- npu="A3" id257 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id257 -->
<!-- npu="950" id258 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id258 -->

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.get_state](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.get_state)

**产品支持情况**：

<!-- npu="910b" id259 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id259 -->
<!-- npu="A3" id260 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id260 -->
<!-- npu="950" id261 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id261 -->

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.name](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.name)

**产品支持情况**：

<!-- npu="910b" id262 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id262 -->
<!-- npu="A3" id263 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id263 -->
<!-- npu="950" id264 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id264 -->

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.set_state](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.C10dRendezvousBackend.set_state)

**产品支持情况**：

<!-- npu="910b" id265 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id265 -->
<!-- npu="A3" id266 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id266 -->
<!-- npu="950" id267 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id267 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend)

**产品支持情况**：

<!-- npu="910b" id268 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id268 -->
<!-- npu="A3" id269 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id269 -->
<!-- npu="950" id270 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id270 -->

> <font size="3">get_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.get_state](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.get_state)

**产品支持情况**：

<!-- npu="910b" id271 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id271 -->
<!-- npu="A3" id272 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id272 -->
<!-- npu="950" id273 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id273 -->

</div>

> <font size="3">name()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.name](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.name)

**产品支持情况**：

<!-- npu="910b" id274 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id274 -->
<!-- npu="A3" id275 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id275 -->
<!-- npu="950" id276 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id276 -->

</div>

> <font size="3">set_state()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.set_state](https://pytorch.org/docs/2.14/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.etcd_rendezvous_backend.EtcdRendezvousBackend.set_state)

**产品支持情况**：

<!-- npu="910b" id277 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id277 -->
<!-- npu="A3" id278 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id278 -->
<!-- npu="950" id279 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id279 -->

</div>

</div>

## Expiration Timers

### torch.distributed.elastic.timer.configure

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.configure](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.configure)

**产品支持情况**：

<!-- npu="910b" id280 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id280 -->
<!-- npu="A3" id281 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id281 -->
<!-- npu="950" id282 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id282 -->

</div>

### torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.debug_info_logging.log_debug_info_for_expired_timers)

**产品支持情况**：

<!-- npu="910b" id283 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id283 -->
<!-- npu="A3" id284 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id284 -->
<!-- npu="950" id285 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id285 -->

</div>

### torch.distributed.elastic.timer.expires

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.expires](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.expires)

**产品支持情况**：

<!-- npu="910b" id286 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id286 -->
<!-- npu="A3" id287 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id287 -->
<!-- npu="950" id288 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id288 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.LocalTimerServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.LocalTimerServer](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.LocalTimerServer)

**产品支持情况**：

<!-- npu="910b" id289 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id289 -->
<!-- npu="A3" id290 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id290 -->
<!-- npu="950" id291 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id291 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.LocalTimerClient

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.LocalTimerClient](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.LocalTimerClient)

**产品支持情况**：

<!-- npu="910b" id292 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id292 -->
<!-- npu="A3" id293 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id293 -->
<!-- npu="950" id294 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id294 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.FileTimerServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.FileTimerServer](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.FileTimerServer)

**产品支持情况**：

<!-- npu="910b" id295 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id295 -->
<!-- npu="A3" id296 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id296 -->
<!-- npu="950" id297 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id297 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.FileTimerClient

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.FileTimerClient](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.FileTimerClient)

**产品支持情况**：

<!-- npu="910b" id298 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id298 -->
<!-- npu="A3" id299 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id299 -->
<!-- npu="950" id300 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id300 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.TimerRequest

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerRequest](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.TimerRequest)

**产品支持情况**：

<!-- npu="910b" id301 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id301 -->
<!-- npu="A3" id302 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id302 -->
<!-- npu="950" id303 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id303 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.TimerServer

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerServer](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.TimerServer)

**产品支持情况**：

<!-- npu="910b" id304 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id304 -->
<!-- npu="A3" id305 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id305 -->
<!-- npu="950" id306 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id306 -->

> <font size="3">clear_timers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerServer.clear_timers](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.TimerServer.clear_timers)

**产品支持情况**：

<!-- npu="910b" id307 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id307 -->
<!-- npu="A3" id308 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id308 -->
<!-- npu="950" id309 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id309 -->

</div>

> <font size="3">get_expired_timers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerServer.get_expired_timers](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.TimerServer.get_expired_timers)

**产品支持情况**：

<!-- npu="910b" id310 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id310 -->
<!-- npu="A3" id311 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id311 -->
<!-- npu="950" id312 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id312 -->

</div>

> <font size="3">register_timers()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerServer.register_timers](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.TimerServer.register_timers)

**产品支持情况**：

<!-- npu="910b" id313 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id313 -->
<!-- npu="A3" id314 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id314 -->
<!-- npu="950" id315 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id315 -->

</div>

</div>

### <code><i>class</i></code> torch.distributed.elastic.timer.TimerClient

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerClient](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.TimerClient)

**产品支持情况**：

<!-- npu="910b" id316 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id316 -->
<!-- npu="A3" id317 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id317 -->
<!-- npu="950" id318 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id318 -->

> <font size="3">acquire()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerClient.acquire](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.TimerClient.acquire)

**产品支持情况**：

<!-- npu="910b" id319 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id319 -->
<!-- npu="A3" id320 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id320 -->
<!-- npu="950" id321 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id321 -->

</div>

> <font size="3">release()</font>

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.timer.TimerClient.release](https://pytorch.org/docs/2.14/elastic/timer.html#torch.distributed.elastic.timer.TimerClient.release)

**产品支持情况**：

<!-- npu="910b" id322 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id322 -->
<!-- npu="A3" id323 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id323 -->
<!-- npu="950" id324 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id324 -->

</div>

</div>

## Metrics

### <code><i>class</i></code> torch.distributed.elastic.metrics.api.MetricHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.api.MetricHandler](https://pytorch.org/docs/2.14/elastic/metrics.html#torch.distributed.elastic.metrics.api.MetricHandler)

**产品支持情况**：

<!-- npu="910b" id325 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id325 -->
<!-- npu="A3" id326 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id326 -->
<!-- npu="950" id327 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id327 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.metrics.api.ConsoleMetricHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.api.ConsoleMetricHandler](https://pytorch.org/docs/2.14/elastic/metrics.html#torch.distributed.elastic.metrics.api.ConsoleMetricHandler)

**产品支持情况**：

<!-- npu="910b" id328 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id328 -->
<!-- npu="A3" id329 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id329 -->
<!-- npu="950" id330 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id330 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.metrics.api.NullMetricHandler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.api.NullMetricHandler](https://pytorch.org/docs/2.14/elastic/metrics.html#torch.distributed.elastic.metrics.api.NullMetricHandler)

**产品支持情况**：

<!-- npu="910b" id331 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id331 -->
<!-- npu="A3" id332 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id332 -->
<!-- npu="950" id333 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id333 -->

</div>

### torch.distributed.elastic.metrics.configure

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.configure](https://pytorch.org/docs/2.14/elastic/metrics.html#torch.distributed.elastic.metrics.configure)

**产品支持情况**：

<!-- npu="910b" id334 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id334 -->
<!-- npu="A3" id335 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id335 -->
<!-- npu="950" id336 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id336 -->

</div>

### torch.distributed.elastic.metrics.prof

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.prof](https://pytorch.org/docs/2.14/elastic/metrics.html#torch.distributed.elastic.metrics.prof)

**产品支持情况**：

<!-- npu="910b" id337 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id337 -->
<!-- npu="A3" id338 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id338 -->
<!-- npu="950" id339 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id339 -->

</div>

### torch.distributed.elastic.metrics.put_metric

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.metrics.put_metric](https://pytorch.org/docs/2.14/elastic/metrics.html#torch.distributed.elastic.metrics.put_metric)

**产品支持情况**：

<!-- npu="910b" id340 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id340 -->
<!-- npu="A3" id341 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id341 -->
<!-- npu="950" id342 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id342 -->

</div>

## Events

### torch.distributed.elastic.events.record

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.record](https://pytorch.org/docs/2.14/elastic/events.html#torch.distributed.elastic.events.record)

**产品支持情况**：

<!-- npu="910b" id343 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id343 -->
<!-- npu="A3" id344 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id344 -->
<!-- npu="950" id345 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id345 -->

</div>

### torch.distributed.elastic.events.get_logging_handler

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.get_logging_handler](https://pytorch.org/docs/2.14/elastic/events.html#torch.distributed.elastic.events.get_logging_handler)

**产品支持情况**：

<!-- npu="910b" id346 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id346 -->
<!-- npu="A3" id347 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id347 -->
<!-- npu="950" id348 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id348 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.events.api.Event

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.api.Event](https://pytorch.org/docs/2.14/elastic/events.html#torch.distributed.elastic.events.api.Event)

**产品支持情况**：

<!-- npu="910b" id349 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id349 -->
<!-- npu="A3" id350 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id350 -->
<!-- npu="950" id351 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id351 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.events.api.EventSource

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.api.EventSource](https://pytorch.org/docs/2.14/elastic/events.html#torch.distributed.elastic.events.api.EventSource)

**产品支持情况**：

<!-- npu="910b" id352 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id352 -->
<!-- npu="A3" id353 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id353 -->
<!-- npu="950" id354 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id354 -->

</div>

### <code><i>class</i></code> torch.distributed.elastic.events.api.EventMetadataValue

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.api.EventMetadataValue](https://pytorch.org/docs/2.14/elastic/events.html#torch.distributed.elastic.events.api.EventMetadataValue)

**产品支持情况**：

<!-- npu="910b" id355 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id355 -->
<!-- npu="A3" id356 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id356 -->
<!-- npu="950" id357 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id357 -->

</div>

### torch.distributed.elastic.events.construct_and_record_rdzv_event

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.events.construct_and_record_rdzv_event](https://pytorch.org/docs/2.14/elastic/events.html#torch.distributed.elastic.events.construct_and_record_rdzv_event)

**产品支持情况**：

<!-- npu="910b" id358 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id358 -->
<!-- npu="A3" id359 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id359 -->
<!-- npu="950" id360 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id360 -->

</div>

## Control Plane

### torch.distributed.elastic.control_plane.worker_main

<div style="margin-left: 2em">

**原生文档**：[torch.distributed.elastic.control_plane.worker_main](https://pytorch.org/docs/2.14/elastic/control_plane.html#torch.distributed.elastic.control_plane.worker_main)

**产品支持情况**：

<!-- npu="910b" id361 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id361 -->
<!-- npu="A3" id362 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id362 -->
<!-- npu="950" id363 -->
- <term>Ascend 950DT系列产品</term>：不支持
<!-- end id363 -->

</div>
