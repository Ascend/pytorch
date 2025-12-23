# 新增特性<a name="ZH-CN_TOPIC_0000002503406503"></a>

<a name="table14945121216428"></a>
<table>
  <thead align="left">
    <tr id="row19946312124219">
      <th class="cellrowborder" valign="top" width="18.801880188018803%" id="mcps1.1.4.1.1"><p id="p1946131234216"><a name="p1946131234216"></a><a name="p1946131234216"></a>组件</p></th>
      <th class="cellrowborder" valign="top" width="32.603260326032604%" id="mcps1.1.4.1.2"><p id="p394611224214"><a name="p394611224214"></a><a name="p394611224214"></a>描述</p></th>
      <th class="cellrowborder" valign="top" width="48.5948594859486%" id="mcps1.1.4.1.3"><p id="p159461312154216"><a name="p159461312154216"></a><a name="p159461312154216"></a>目的</p></th>
    </tr>
  </thead>
  <tbody>
    <tr id="row18946612144217">
      <td class="cellrowborder" rowspan="6" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1"><p id="p18137656467"><a name="p18137656467"></a><a name="p18137656467"><span id="ph4778145519911"><a name="ph4778145519911"></a><a name="ph4778145519911"></a>Ascend Extension for PyTorch</span>（即torch-npu）</p></td>
      <td class="cellrowborder" valign="top" width="32.603260326032604%" headers="mcps1.1.4.1.2"><p id="p184761459182417"><a name="p184761459182417"></a><a name="p184761459182417"></a>集合通信内存复用优化</p></td>
      <td class="cellrowborder" valign="top" width="48.5948594859486%" headers="mcps1.1.4.1.3"><p id="p84765590243"><a name="p84765590243"></a><a name="p84765590243"></a>新增erase_record_stream的增强模式，内存复用率更高</p></td>
    </tr>
    <tr id="row156211934125710">
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1"><p id="p13476125912411"><a name="p13476125912411"></a><a name="p13476125912411"></a>host allocator对齐社区</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p1547513591240"><a name="p1547513591240"></a><a name="p1547513591240"></a>复用社区已有的host allocator机制，增强host allocator能力</p></td>
    </tr>
    <tr id="row196191324124519">
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1"><p id="p6464155972419"><a name="p6464155972419"></a><a name="p6464155972419"></a>新增支持PyTorch 2.9.0</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p1546325972413"><a name="p1546325972413"></a><a name="p1546325972413"></a>通用能力，与社区同步发布</p></td>
    </tr>
    <tr id="row59931050204815">
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1"><p id="p946355922419"><a name="p946355922419"></a><a name="p946355922419"></a>新增支持python 3.12</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p19462105942416"><a name="p19462105942416"></a><a name="p19462105942416"></a>通用能力，支持3.12版本的python</p></td>
    </tr>
    <tr id="row4100180125011">
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1"><p id="p6462259132418"><a name="p6462259132418"></a><a name="p6462259132418"></a>新增支持symmetric memory接入shmem</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p154623598243"><a name="p154623598243"></a><a name="p154623598243"></a>对齐nvshmem，适配接入NPU的shmem能力</p></td>
    </tr>
    <tr id="row76722381272">
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1"><p id="p76721381071"><a name="p76721381071"></a><a name="p76721381071"></a>通信域异常检测能力增强</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p867253819716"><a name="p867253819716"></a><a name="p867253819716"></a>HCCL异常检测与watchdog解耦，支持未下发通信算子时也对hccl链路状态进行检测</p></td>
    </tr>
    <tr id="row1737075241311">
      <td class="cellrowborder" rowspan="4" valign="top" width="18.801880188018803%" headers="mcps1.1.4.1.1"><p id="p144361913143214"><a name="p144361913143214"></a><a name="p144361913143214"></a>Driving SDK</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p64611659182414"><a name="p64611659182414"></a><a name="p64611659182414"></a>新增Pi0.5模型适配</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.3"><p id="p134615593241"><a name="p134615593241"></a><a name="p134615593241"></a>适配业界主流VLA模型，支持具身智能和自动驾驶场景</p></td>
    </tr>
    <tr id="row3683105419718">
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1"><p id="p668319546714"><a name="p668319546714"></a><a name="p668319546714"></a>新增GR00T-N1.5模型适配</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p10683165413718"><a name="p10683165413718"></a><a name="p10683165413718"></a>适配业界主流VLA模型，支持具身智能和自动驾驶场景</p></td>
    </tr>
    <tr id="row1446416501774">
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1"><p id="p2464165010710"><a name="p2464165010710"></a><a name="p2464165010710"></a>新增VGGT模型适配</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p2464135015712"><a name="p2464135015712"></a><a name="p2464135015712"></a>适配业界主流世界模型，支持具身智能和自动驾驶场景</p></td>
    </tr>
    <tr id="row124574451918">
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.1"><p id="p14575451395"><a name="p14575451395"></a><a name="p14575451395"></a>新增自驾典型模型环境配置脚本</p></td>
      <td class="cellrowborder" valign="top"  headers="mcps1.1.4.1.2"><p id="p1245714452912"><a name="p1245714452912"></a><a name="p1245714452912"></a>提升Driving SDK易用性</p></td>
    </tr>
  </tbody>
</table>
