# 通信域建链超时

## 问题现象

关键词"**Socket Timeout**"

![](figures/socket_timeout.png)

## 故障根因

关键过程：模型多卡训练过程中，中断报错。

根本原因分析：

1.  0卡与其他卡的网络存在异常，其他卡等待超时报错；
2.  0卡异常退出，其他卡等待超时报错；
3.  0卡比其他卡执行通信域建立慢，其他卡等待超时报错。

## 处理方法


<table><tbody><tr id="row133331920165614"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.1.1"><p id="p83339201562">Error Code</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.1.1 "><p id="p19428111123714">无</p>
</td>
</tr>
<tr id="row58261416152019"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.2.1"><p id="p78261916142014">故障事件名称</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.2.1 "><p id="p12416104333517">通信域建链超时</p>
</td>
</tr>
<tr id="row1082711617201"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.3.1"><p id="p1782741619205">故障解释/可能原因</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.3.1 "><ol id="ol240074517494"><li>0卡与其他卡的网络存在异常</li><li>0卡异常退出</li><li>0卡比其他卡晚执行通信域建立</li></ol>
</td>
</tr>
<tr id="row1474663022115"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.4.1"><p id="p774617303213">故障影响</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.4.1 "><p id="p35447463353">模型训练终止</p>
</td>
</tr>
<tr id="row19915122652114"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.5.1"><p id="p1791515262213">故障自处理模式</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.5.1 "><ol id="ol14419164125117"><li>检查0卡与其他卡网络情况</li><li>检查0卡有没有异常退出</li><li>检测0卡是否存在执行通信域建立操作比较慢的情况</li></ol>
</td>
</tr>
<tr id="row1356182417228"><th class="firstcol" valign="top" width="17.66%" id="mcps1.1.3.6.1"><p id="p175662413229">系统处理建议</p>
</th>
<td class="cellrowborder" valign="top" width="82.34%" headers="mcps1.1.3.6.1 "><p id="p164198119376">无需操作</p>
</td>
</tr>
</tbody>
</table>

