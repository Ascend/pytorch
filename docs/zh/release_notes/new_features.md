# 新增特性

<table>
  <tr>
    <th>组件</th>
    <th>描述</th>
    <th>目的</th>
  </tr>
  <tr>
    <th rowspan="6">Ascend Extension for PyTorch（即torch-npu）</th>
    <th>集合通信内存复用优化</th>
    <th>新增erase_record_stream的增强模式，内存复用率更高</th>
  </tr>
  <tr>
    <th>host allocator对齐社区</th>
    <th>复用社区已有的host allocator机制，增强host allocator能力</th>
  </tr>
  <tr>
    <th>新增支持PyTorch 2.9.0</th>
    <th>通用能力，与社区同步发布</th>
  </tr>
  <tr>
    <th>新增支持pytorch 3.12</th>
    <th>通用能力，支持3.12版本的python</th>
  </tr>
  <tr>
    <th>新增支持symmetric memory接入shmem</th>
    <th>对齐nvshmem，适配接入NPU的shmem能力</th>
  </tr>
  <tr>
    <th>通信域异常检测能力增强</th>
    <th>HCCL异常检测与watchdog解耦，支持未下发通信算子时也对hccl链路状态进行检测</th>
  </tr>
  <tr>
    <th rowspan="4">Driving SDK</th>
    <th>新增Pi0.5模型适配</th>
    <th>适配业界主流VLA模型，支持具身智能和自动驾驶场景</th>
  </tr>
  <tr>
    <th>新增GR00T-N1.5模型适配</th>
    <th>适配业界主流VLA模型，支持具身智能和自动驾驶场景</th>
  </tr>
  <tr>
    <th>新增VGGT模型适配</th>
    <th>适配业界主流世界模型，支持具身智能和自动驾驶场景</th>
  </tr>
  <tr>
    <th>新增自驾典型模型环境配置脚本</th>
    <th>提升Driving SDK易用性</th>
  </tr>
</table>
