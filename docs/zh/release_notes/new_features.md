# 新增特性<a name="ZH-CN_TOPIC_0000002503406503"></a>

<a name="table14945121216428"></a>
<table>
  <thead align="left">
    <tr>
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1">组件</th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2">描述</th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3">目的</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="cellrowborder" rowspan="6" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1"><span>Ascend Extension for PyTorch</span>（即torch-npu）</td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2">集合通信内存复用优化</td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3">新增erase_record_stream的增强模式，内存复用率更高</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">host allocator对齐社区</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">复用社区已有的host allocator机制，增强host allocator能力</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增支持PyTorch 2.9.0</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">通用能力，与社区同步发布</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增支持Python 3.12</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">通用能力，支持3.12版本的Python</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增支持symmetric memory接入shmem</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">对齐nvshmem，适配接入NPU的shmem能力</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">通信域异常检测能力增强</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">HCCL异常检测与watchdog解耦，支持未下发通信算子时也对hccl链路状态进行检测</td>
    </tr>
    <tr>
      <td class="cellrowborder" rowspan="4" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1">Driving SDK</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">新增Pi0.5模型适配</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.3">适配业界主流VLA模型，支持具身智能和自动驾驶场景</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增GR00T-N1.5模型适配</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">适配业界主流VLA模型，支持具身智能和自动驾驶场景</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增VGGT模型适配</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">适配业界主流世界模型，支持具身智能和自动驾驶场景</td>
    </tr>
    <tr>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1">新增自驾典型模型环境配置脚本</td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2">提升Driving SDK易用性</td>
    </tr>
  </tbody>
</table>
