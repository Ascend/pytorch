# PyTorch适配算子清单
-   [PyTorch原生算子与昇腾算子对应表](#PyTorch原生算子与昇腾算子对应表.md)
-   [PyTorch昇腾自定义算子](#PyTorch昇腾自定义算子.md)
<h2 id="PyTorch原生算子与昇腾算子对应表.md">PyTorch原生算子与昇腾算子对应表</h2>

<a name="table16733238131217"></a>
<table><thead align="left"><tr id="row1869443913127"><th class="cellrowborder" valign="top" width="8.694379391100702%" id="mcps1.1.4.1.1"><p id="p354012516592"><a name="p354012516592"></a><a name="p354012516592"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="46.18462138953943%" id="mcps1.1.4.1.2"><p id="p1369433921218"><a name="p1369433921218"></a><a name="p1369433921218"></a>PyTorch 原生算子</p>
</th>
<th class="cellrowborder" valign="top" width="45.120999219359874%" id="mcps1.1.4.1.3"><p id="p369493911218"><a name="p369493911218"></a><a name="p369493911218"></a>昇腾适配算子</p>
</th>
</tr>
</thead>
<tbody><tr id="row1469411391123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164587169219"><a name="p164587169219"></a><a name="p164587169219"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6461936172717"><a name="p6461936172717"></a><a name="p6461936172717"></a>dropout</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p446117363273"><a name="p446117363273"></a><a name="p446117363273"></a>dropout_npu</p>
</td>
</tr>
<tr id="row469519391121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1458131614214"><a name="p1458131614214"></a><a name="p1458131614214"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34613369273"><a name="p34613369273"></a><a name="p34613369273"></a>dropout_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124611636162716"><a name="p124611636162716"></a><a name="p124611636162716"></a>dropout_npu_</p>
</td>
</tr>
<tr id="row156952394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1145861612218"><a name="p1145861612218"></a><a name="p1145861612218"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1646116361271"><a name="p1646116361271"></a><a name="p1646116361271"></a>abs</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34621536172711"><a name="p34621536172711"></a><a name="p34621536172711"></a>abs_npu</p>
</td>
</tr>
<tr id="row17695739101215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134581516024"><a name="p134581516024"></a><a name="p134581516024"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194621736152713"><a name="p194621736152713"></a><a name="p194621736152713"></a>abs_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9462936132718"><a name="p9462936132718"></a><a name="p9462936132718"></a>abs_npu_</p>
</td>
</tr>
<tr id="row569517398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1945819162213"><a name="p1945819162213"></a><a name="p1945819162213"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184621536192712"><a name="p184621536192712"></a><a name="p184621536192712"></a>abs.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144621361272"><a name="p144621361272"></a><a name="p144621361272"></a>abs_out_npu</p>
</td>
</tr>
<tr id="row6695123941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9458816327"><a name="p9458816327"></a><a name="p9458816327"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p646283619275"><a name="p646283619275"></a><a name="p646283619275"></a>acos</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p146233611275"><a name="p146233611275"></a><a name="p146233611275"></a>acos_npu</p>
</td>
</tr>
<tr id="row869593910122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124586161729"><a name="p124586161729"></a><a name="p124586161729"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2046223610276"><a name="p2046223610276"></a><a name="p2046223610276"></a>acos_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1546217361271"><a name="p1546217361271"></a><a name="p1546217361271"></a>acos_npu_</p>
</td>
</tr>
<tr id="row16695239121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p745881610211"><a name="p745881610211"></a><a name="p745881610211"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1546214363270"><a name="p1546214363270"></a><a name="p1546214363270"></a>acos.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p04621336132716"><a name="p04621336132716"></a><a name="p04621336132716"></a>acos_out_npu</p>
</td>
</tr>
<tr id="row18696133961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14458161613213"><a name="p14458161613213"></a><a name="p14458161613213"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124621036102718"><a name="p124621036102718"></a><a name="p124621036102718"></a>adaptive_avg_pool1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19462536142711"><a name="p19462536142711"></a><a name="p19462536142711"></a>adaptive_avg_pool1d_npu</p>
</td>
</tr>
<tr id="row1769693961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p345851614215"><a name="p345851614215"></a><a name="p345851614215"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1462736172718"><a name="p1462736172718"></a><a name="p1462736172718"></a>add.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10462536102716"><a name="p10462536102716"></a><a name="p10462536102716"></a>add_npu</p>
</td>
</tr>
<tr id="row1869623951214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1945814161127"><a name="p1945814161127"></a><a name="p1945814161127"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p64621636102714"><a name="p64621636102714"></a><a name="p64621636102714"></a>add_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3462163618271"><a name="p3462163618271"></a><a name="p3462163618271"></a>add_npu_</p>
</td>
</tr>
<tr id="row16961439181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13458916224"><a name="p13458916224"></a><a name="p13458916224"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p946253662717"><a name="p946253662717"></a><a name="p946253662717"></a>add.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114622360279"><a name="p114622360279"></a><a name="p114622360279"></a>add_out_npu</p>
</td>
</tr>
<tr id="row10696133931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6458116525"><a name="p6458116525"></a><a name="p6458116525"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p154621636112714"><a name="p154621636112714"></a><a name="p154621636112714"></a>add.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134625365273"><a name="p134625365273"></a><a name="p134625365273"></a>add_npu</p>
</td>
</tr>
<tr id="row6696143991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1345814161122"><a name="p1345814161122"></a><a name="p1345814161122"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24624369279"><a name="p24624369279"></a><a name="p24624369279"></a>add_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p64621136202711"><a name="p64621136202711"></a><a name="p64621136202711"></a>add_npu_</p>
</td>
</tr>
<tr id="row1969613901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04591116925"><a name="p04591116925"></a><a name="p04591116925"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11462143602711"><a name="p11462143602711"></a><a name="p11462143602711"></a>addmv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p446263662714"><a name="p446263662714"></a><a name="p446263662714"></a>addmv_npu</p>
</td>
</tr>
<tr id="row1169614395122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144592161423"><a name="p144592161423"></a><a name="p144592161423"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13462536162713"><a name="p13462536162713"></a><a name="p13462536162713"></a>addmv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1746263612718"><a name="p1746263612718"></a><a name="p1746263612718"></a>addmv_npu_</p>
</td>
</tr>
<tr id="row2696103911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54596161229"><a name="p54596161229"></a><a name="p54596161229"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1146283619277"><a name="p1146283619277"></a><a name="p1146283619277"></a>addmv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p446223614278"><a name="p446223614278"></a><a name="p446223614278"></a>addmv_out_npu</p>
</td>
</tr>
<tr id="row116976397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04591716922"><a name="p04591716922"></a><a name="p04591716922"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14462173612718"><a name="p14462173612718"></a><a name="p14462173612718"></a>addr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4463173616276"><a name="p4463173616276"></a><a name="p4463173616276"></a>addr_npu</p>
</td>
</tr>
<tr id="row1769718393121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p545914161122"><a name="p545914161122"></a><a name="p545914161122"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184631436142710"><a name="p184631436142710"></a><a name="p184631436142710"></a>addr_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p04637365271"><a name="p04637365271"></a><a name="p04637365271"></a>addr_npu_</p>
</td>
</tr>
<tr id="row1669716393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145901617213"><a name="p145901617213"></a><a name="p145901617213"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84639366272"><a name="p84639366272"></a><a name="p84639366272"></a>addr.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134633363275"><a name="p134633363275"></a><a name="p134633363275"></a>addr_out_npu</p>
</td>
</tr>
<tr id="row1469716399129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1145921614213"><a name="p1145921614213"></a><a name="p1145921614213"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p64631036112716"><a name="p64631036112716"></a><a name="p64631036112716"></a>affine_grid_generator</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16463113642715"><a name="p16463113642715"></a><a name="p16463113642715"></a>affine_grid_generator_npu</p>
</td>
</tr>
<tr id="row6697143981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194595161421"><a name="p194595161421"></a><a name="p194595161421"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84631236182712"><a name="p84631236182712"></a><a name="p84631236182712"></a>affine_grid_generator_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14463193619275"><a name="p14463193619275"></a><a name="p14463193619275"></a>affine_grid_generator_backward_npu</p>
</td>
</tr>
<tr id="row5697103931212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p545912161620"><a name="p545912161620"></a><a name="p545912161620"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p246343622714"><a name="p246343622714"></a><a name="p246343622714"></a>all.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p04631936102711"><a name="p04631936102711"></a><a name="p04631936102711"></a>all_npu</p>
</td>
</tr>
<tr id="row11697133961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174590161428"><a name="p174590161428"></a><a name="p174590161428"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15463123617274"><a name="p15463123617274"></a><a name="p15463123617274"></a>all.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5463203613271"><a name="p5463203613271"></a><a name="p5463203613271"></a>all_out_npu</p>
</td>
</tr>
<tr id="row13697239171218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p045914169210"><a name="p045914169210"></a><a name="p045914169210"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1146311368275"><a name="p1146311368275"></a><a name="p1146311368275"></a>any.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8463836152715"><a name="p8463836152715"></a><a name="p8463836152715"></a>any_npu</p>
</td>
</tr>
<tr id="row7698143951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3459916920"><a name="p3459916920"></a><a name="p3459916920"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p44638366276"><a name="p44638366276"></a><a name="p44638366276"></a>any.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6463136162717"><a name="p6463136162717"></a><a name="p6463136162717"></a>any_out_npu</p>
</td>
</tr>
<tr id="row3698133916123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04605168219"><a name="p04605168219"></a><a name="p04605168219"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1146343642710"><a name="p1146343642710"></a><a name="p1146343642710"></a>arange</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174631036192710"><a name="p174631036192710"></a><a name="p174631036192710"></a>arange_npu</p>
</td>
</tr>
<tr id="row86981439181215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1446091613214"><a name="p1446091613214"></a><a name="p1446091613214"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1146333618274"><a name="p1146333618274"></a><a name="p1146333618274"></a>arange.start</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p946303616275"><a name="p946303616275"></a><a name="p946303616275"></a>arange_npu</p>
</td>
</tr>
<tr id="row8698203971214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146015163211"><a name="p146015163211"></a><a name="p146015163211"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p446310362277"><a name="p446310362277"></a><a name="p446310362277"></a>arange.start_step</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18463136202710"><a name="p18463136202710"></a><a name="p18463136202710"></a>arange_npu</p>
</td>
</tr>
<tr id="row1698153941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194608160212"><a name="p194608160212"></a><a name="p194608160212"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1946353619279"><a name="p1946353619279"></a><a name="p1946353619279"></a>arange.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124631636202720"><a name="p124631636202720"></a><a name="p124631636202720"></a>arange_out_npu</p>
</td>
</tr>
<tr id="row4698143917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2460616021"><a name="p2460616021"></a><a name="p2460616021"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p44639369279"><a name="p44639369279"></a><a name="p44639369279"></a>arange.start_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10463153620277"><a name="p10463153620277"></a><a name="p10463153620277"></a>arange_out_npu</p>
</td>
</tr>
<tr id="row1469810393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19460716125"><a name="p19460716125"></a><a name="p19460716125"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p646323662710"><a name="p646323662710"></a><a name="p646323662710"></a>_dim_arange</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13463193662713"><a name="p13463193662713"></a><a name="p13463193662713"></a>_dim_arange_npu</p>
</td>
</tr>
<tr id="row17698153919124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174601316821"><a name="p174601316821"></a><a name="p174601316821"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8464636172714"><a name="p8464636172714"></a><a name="p8464636172714"></a>argmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p746411364271"><a name="p746411364271"></a><a name="p746411364271"></a>argmax_npu</p>
</td>
</tr>
<tr id="row46981739181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14460141616212"><a name="p14460141616212"></a><a name="p14460141616212"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2464183692719"><a name="p2464183692719"></a><a name="p2464183692719"></a>argmin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13464103612714"><a name="p13464103612714"></a><a name="p13464103612714"></a>argmin_npu</p>
</td>
</tr>
<tr id="row46981939141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114603161024"><a name="p114603161024"></a><a name="p114603161024"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1246453642718"><a name="p1246453642718"></a><a name="p1246453642718"></a>as_strided</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1146415361276"><a name="p1146415361276"></a><a name="p1146415361276"></a>as_strided_npu</p>
</td>
</tr>
<tr id="row2698339151220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11460816521"><a name="p11460816521"></a><a name="p11460816521"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19464173614275"><a name="p19464173614275"></a><a name="p19464173614275"></a>as_strided_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114641636102718"><a name="p114641636102718"></a><a name="p114641636102718"></a>as_strided_npu_</p>
</td>
</tr>
<tr id="row369911399124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4460916527"><a name="p4460916527"></a><a name="p4460916527"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12464153602719"><a name="p12464153602719"></a><a name="p12464153602719"></a>asin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34641936172715"><a name="p34641936172715"></a><a name="p34641936172715"></a>asin_npu</p>
</td>
</tr>
<tr id="row106992394125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446015161828"><a name="p446015161828"></a><a name="p446015161828"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p64641636172718"><a name="p64641636172718"></a><a name="p64641636172718"></a>asin_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24648367278"><a name="p24648367278"></a><a name="p24648367278"></a>asin_npu_</p>
</td>
</tr>
<tr id="row9699139121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104601161822"><a name="p104601161822"></a><a name="p104601161822"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p546493618272"><a name="p546493618272"></a><a name="p546493618272"></a>asin.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7464163614271"><a name="p7464163614271"></a><a name="p7464163614271"></a>asin_out_npu</p>
</td>
</tr>
<tr id="row166991339121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44608161825"><a name="p44608161825"></a><a name="p44608161825"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p646483612713"><a name="p646483612713"></a><a name="p646483612713"></a>atan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13464536162713"><a name="p13464536162713"></a><a name="p13464536162713"></a>atan_npu</p>
</td>
</tr>
<tr id="row3699139191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44604161725"><a name="p44604161725"></a><a name="p44604161725"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5464143610271"><a name="p5464143610271"></a><a name="p5464143610271"></a>atan_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p184644363273"><a name="p184644363273"></a><a name="p184644363273"></a>atan_npu_</p>
</td>
</tr>
<tr id="row269915391125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10460516425"><a name="p10460516425"></a><a name="p10460516425"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184641336112717"><a name="p184641336112717"></a><a name="p184641336112717"></a>atan.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p204647366272"><a name="p204647366272"></a><a name="p204647366272"></a>atan_out_npu</p>
</td>
</tr>
<tr id="row869983913127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10460716129"><a name="p10460716129"></a><a name="p10460716129"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p144646362278"><a name="p144646362278"></a><a name="p144646362278"></a>baddbmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p346410364279"><a name="p346410364279"></a><a name="p346410364279"></a>baddbmm_npu</p>
</td>
</tr>
<tr id="row46997391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84605163210"><a name="p84605163210"></a><a name="p84605163210"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1746412365273"><a name="p1746412365273"></a><a name="p1746412365273"></a>baddbmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p446453611279"><a name="p446453611279"></a><a name="p446453611279"></a>baddbmm_npu_</p>
</td>
</tr>
<tr id="row18699143911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846013161021"><a name="p846013161021"></a><a name="p846013161021"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1146414363278"><a name="p1146414363278"></a><a name="p1146414363278"></a>baddbmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11464183642719"><a name="p11464183642719"></a><a name="p11464183642719"></a>baddbmm_out_npu</p>
</td>
</tr>
<tr id="row9700163961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11461181616213"><a name="p11461181616213"></a><a name="p11461181616213"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12464436172717"><a name="p12464436172717"></a><a name="p12464436172717"></a>bartlett_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54641136152719"><a name="p54641136152719"></a><a name="p54641136152719"></a>bartlett_window_npu</p>
</td>
</tr>
<tr id="row57008394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p046141617218"><a name="p046141617218"></a><a name="p046141617218"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1546443692714"><a name="p1546443692714"></a><a name="p1546443692714"></a>bartlett_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8464636122718"><a name="p8464636122718"></a><a name="p8464636122718"></a>bartlett_window_npu</p>
</td>
</tr>
<tr id="row20700113951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74615161222"><a name="p74615161222"></a><a name="p74615161222"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p246483614271"><a name="p246483614271"></a><a name="p246483614271"></a>batch_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p546523610271"><a name="p546523610271"></a><a name="p546523610271"></a>batch_norm_npu_</p>
</td>
</tr>
<tr id="row1070043920122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194614162212"><a name="p194614162212"></a><a name="p194614162212"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p146512365276"><a name="p146512365276"></a><a name="p146512365276"></a>_batch_norm_impl_index</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1846520361276"><a name="p1846520361276"></a><a name="p1846520361276"></a>_batch_norm_impl_index_npu</p>
</td>
</tr>
<tr id="row1970093931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9461116320"><a name="p9461116320"></a><a name="p9461116320"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1146573682717"><a name="p1146573682717"></a><a name="p1146573682717"></a>_batch_norm_impl_index_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1465163616272"><a name="p1465163616272"></a><a name="p1465163616272"></a>_batch_norm_impl_index_backward_npu</p>
</td>
</tr>
<tr id="row270033916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16461151618219"><a name="p16461151618219"></a><a name="p16461151618219"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1465736152712"><a name="p1465736152712"></a><a name="p1465736152712"></a>bernoulli</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5465163622713"><a name="p5465163622713"></a><a name="p5465163622713"></a>bernoulli_npu</p>
</td>
</tr>
<tr id="row10700339151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4461191610214"><a name="p4461191610214"></a><a name="p4461191610214"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124651636192712"><a name="p124651636192712"></a><a name="p124651636192712"></a>bernoulli_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1465436142713"><a name="p1465436142713"></a><a name="p1465436142713"></a>bernoulli_npu_</p>
</td>
</tr>
<tr id="row12700539141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146118167215"><a name="p1146118167215"></a><a name="p1146118167215"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18465936202710"><a name="p18465936202710"></a><a name="p18465936202710"></a>bernoulli_.float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1746517365277"><a name="p1746517365277"></a><a name="p1746517365277"></a>bernoulli_npu_</p>
</td>
</tr>
<tr id="row8700203961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2461131611210"><a name="p2461131611210"></a><a name="p2461131611210"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184651236112718"><a name="p184651236112718"></a><a name="p184651236112718"></a>binary_cross_entropy</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11465143618277"><a name="p11465143618277"></a><a name="p11465143618277"></a>binary_cross_entropy_npu</p>
</td>
</tr>
<tr id="row1770043931218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746161614219"><a name="p1746161614219"></a><a name="p1746161614219"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15465103611271"><a name="p15465103611271"></a><a name="p15465103611271"></a>binary_cross_entropy.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194651836142711"><a name="p194651836142711"></a><a name="p194651836142711"></a>binary_cross_entropy_out_npu</p>
</td>
</tr>
<tr id="row5700139121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84611816526"><a name="p84611816526"></a><a name="p84611816526"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10465123613272"><a name="p10465123613272"></a><a name="p10465123613272"></a>binary_cross_entropy_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1546553622720"><a name="p1546553622720"></a><a name="p1546553622720"></a>binary_cross_entropy_backward_npu</p>
</td>
</tr>
<tr id="row137012039151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546191619213"><a name="p1546191619213"></a><a name="p1546191619213"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p946543615278"><a name="p946543615278"></a><a name="p946543615278"></a>binary_cross_entropy_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94659369270"><a name="p94659369270"></a><a name="p94659369270"></a>binary_cross_entropy_backward_out_npu</p>
</td>
</tr>
<tr id="row5701143914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746218164216"><a name="p1746218164216"></a><a name="p1746218164216"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1465183612714"><a name="p1465183612714"></a><a name="p1465183612714"></a>binary_cross_entropy_with_logits</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34651836122718"><a name="p34651836122718"></a><a name="p34651836122718"></a>binary_cross_entropy_with_logits_npu</p>
</td>
</tr>
<tr id="row18701439171211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17462616221"><a name="p17462616221"></a><a name="p17462616221"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17465163612712"><a name="p17465163612712"></a><a name="p17465163612712"></a>binary_cross_entropy_with_logits_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15465163612277"><a name="p15465163612277"></a><a name="p15465163612277"></a>binary_cross_entropy_with_logits_backward_npu</p>
</td>
</tr>
<tr id="row5701173912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246216169211"><a name="p1246216169211"></a><a name="p1246216169211"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1465236132717"><a name="p1465236132717"></a><a name="p1465236132717"></a>bitwise_not</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p246583682712"><a name="p246583682712"></a><a name="p246583682712"></a>bitwise_not_npu</p>
</td>
</tr>
<tr id="row270111390122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164621316724"><a name="p164621316724"></a><a name="p164621316724"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p046516369276"><a name="p046516369276"></a><a name="p046516369276"></a>bitwise_not_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8465123617276"><a name="p8465123617276"></a><a name="p8465123617276"></a>bitwise_not_npu_</p>
</td>
</tr>
<tr id="row27010399120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14627161629"><a name="p14627161629"></a><a name="p14627161629"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17465236152712"><a name="p17465236152712"></a><a name="p17465236152712"></a>bitwise_not.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16465153622719"><a name="p16465153622719"></a><a name="p16465153622719"></a>bitwise_not_out_npu</p>
</td>
</tr>
<tr id="row157011339201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15462316429"><a name="p15462316429"></a><a name="p15462316429"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7465113672720"><a name="p7465113672720"></a><a name="p7465113672720"></a>logical_not</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1346663632711"><a name="p1346663632711"></a><a name="p1346663632711"></a>logical_not_npu</p>
</td>
</tr>
<tr id="row187011339161218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346214163212"><a name="p346214163212"></a><a name="p346214163212"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20466183682710"><a name="p20466183682710"></a><a name="p20466183682710"></a>logical_not_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p164661936102715"><a name="p164661936102715"></a><a name="p164661936102715"></a>logical_not_npu_</p>
</td>
</tr>
<tr id="row20701183921218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146271617211"><a name="p1146271617211"></a><a name="p1146271617211"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p846683617278"><a name="p846683617278"></a><a name="p846683617278"></a>logical_not.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1846613672720"><a name="p1846613672720"></a><a name="p1846613672720"></a>logical_not_out_npu</p>
</td>
</tr>
<tr id="row177011539151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84621516924"><a name="p84621516924"></a><a name="p84621516924"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8466193692716"><a name="p8466193692716"></a><a name="p8466193692716"></a>logical_and</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p246653612716"><a name="p246653612716"></a><a name="p246653612716"></a>logical_and_npu</p>
</td>
</tr>
<tr id="row37015396121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24626162028"><a name="p24626162028"></a><a name="p24626162028"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184665363273"><a name="p184665363273"></a><a name="p184665363273"></a>logical_and_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p446617367274"><a name="p446617367274"></a><a name="p446617367274"></a>logical_and_npu_</p>
</td>
</tr>
<tr id="row1470243915128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11462101612215"><a name="p11462101612215"></a><a name="p11462101612215"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1946683672718"><a name="p1946683672718"></a><a name="p1946683672718"></a>logical_and.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24668362278"><a name="p24668362278"></a><a name="p24668362278"></a>logical_and_out_npu</p>
</td>
</tr>
<tr id="row870210392126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144622161322"><a name="p144622161322"></a><a name="p144622161322"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p846643642710"><a name="p846643642710"></a><a name="p846643642710"></a>logical_or</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p204666361272"><a name="p204666361272"></a><a name="p204666361272"></a>logical_or_npu</p>
</td>
</tr>
<tr id="row670210390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9462316321"><a name="p9462316321"></a><a name="p9462316321"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p946673662715"><a name="p946673662715"></a><a name="p946673662715"></a>logical_or_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1646616368278"><a name="p1646616368278"></a><a name="p1646616368278"></a>logical_or_npu_</p>
</td>
</tr>
<tr id="row1570215393125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546219161422"><a name="p546219161422"></a><a name="p546219161422"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3466143672713"><a name="p3466143672713"></a><a name="p3466143672713"></a>logical_or.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5466183611272"><a name="p5466183611272"></a><a name="p5466183611272"></a>logical_or_out_npu</p>
</td>
</tr>
<tr id="row18702203919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446291617213"><a name="p446291617213"></a><a name="p446291617213"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10466736102714"><a name="p10466736102714"></a><a name="p10466736102714"></a>blackman_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4466163612273"><a name="p4466163612273"></a><a name="p4466163612273"></a>blackman_window_npu</p>
</td>
</tr>
<tr id="row1870283916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14462151619216"><a name="p14462151619216"></a><a name="p14462151619216"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5466133612713"><a name="p5466133612713"></a><a name="p5466133612713"></a>blackman_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18466133617271"><a name="p18466133617271"></a><a name="p18466133617271"></a>blackman_window_npu</p>
</td>
</tr>
<tr id="row1870263914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64621016123"><a name="p64621016123"></a><a name="p64621016123"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12466336112713"><a name="p12466336112713"></a><a name="p12466336112713"></a>bmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1446613366274"><a name="p1446613366274"></a><a name="p1446613366274"></a>bmm_npu</p>
</td>
</tr>
<tr id="row12702103918126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146231616213"><a name="p146231616213"></a><a name="p146231616213"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20466336112710"><a name="p20466336112710"></a><a name="p20466336112710"></a>bmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p446643682710"><a name="p446643682710"></a><a name="p446643682710"></a>bmm_out_npu</p>
</td>
</tr>
<tr id="row97021739111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1046217164214"><a name="p1046217164214"></a><a name="p1046217164214"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1946693612271"><a name="p1946693612271"></a><a name="p1946693612271"></a>cat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11466736182711"><a name="p11466736182711"></a><a name="p11466736182711"></a>cat_npu</p>
</td>
</tr>
<tr id="row4702439171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14628160211"><a name="p14628160211"></a><a name="p14628160211"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1946719361273"><a name="p1946719361273"></a><a name="p1946719361273"></a>cat.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1467336112710"><a name="p1467336112710"></a><a name="p1467336112710"></a>cat_out_npu</p>
</td>
</tr>
<tr id="row12703153917121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54637161021"><a name="p54637161021"></a><a name="p54637161021"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184671636172712"><a name="p184671636172712"></a><a name="p184671636172712"></a>cat.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p746715364273"><a name="p746715364273"></a><a name="p746715364273"></a>cat_npu</p>
</td>
</tr>
<tr id="row1470363911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546371616214"><a name="p1546371616214"></a><a name="p1546371616214"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5467133613277"><a name="p5467133613277"></a><a name="p5467133613277"></a>cat.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144670365274"><a name="p144670365274"></a><a name="p144670365274"></a>cat_out_npu</p>
</td>
</tr>
<tr id="row170313398129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94631916625"><a name="p94631916625"></a><a name="p94631916625"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19467736142712"><a name="p19467736142712"></a><a name="p19467736142712"></a>ceil</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3467936122716"><a name="p3467936122716"></a><a name="p3467936122716"></a>ceil_npu</p>
</td>
</tr>
<tr id="row570333911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246314166216"><a name="p1246314166216"></a><a name="p1246314166216"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6467103662714"><a name="p6467103662714"></a><a name="p6467103662714"></a>ceil_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1946712368274"><a name="p1946712368274"></a><a name="p1946712368274"></a>ceil_npu_</p>
</td>
</tr>
<tr id="row127031039151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1463516628"><a name="p1463516628"></a><a name="p1463516628"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p746719367276"><a name="p746719367276"></a><a name="p746719367276"></a>ceil.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p546710363270"><a name="p546710363270"></a><a name="p546710363270"></a>ceil_out_npu</p>
</td>
</tr>
<tr id="row147031239181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94637162217"><a name="p94637162217"></a><a name="p94637162217"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1546711362271"><a name="p1546711362271"></a><a name="p1546711362271"></a>clamp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1746743612272"><a name="p1746743612272"></a><a name="p1746743612272"></a>clamp_npu</p>
</td>
</tr>
<tr id="row7703143911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846351612211"><a name="p846351612211"></a><a name="p846351612211"></a>84</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14467113632716"><a name="p14467113632716"></a><a name="p14467113632716"></a>clamp_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8467153611279"><a name="p8467153611279"></a><a name="p8467153611279"></a>clamp_npu_</p>
</td>
</tr>
<tr id="row137031396123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154635161025"><a name="p154635161025"></a><a name="p154635161025"></a>85</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1446713365275"><a name="p1446713365275"></a><a name="p1446713365275"></a>clamp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24671365276"><a name="p24671365276"></a><a name="p24671365276"></a>clamp_out_npu</p>
</td>
</tr>
<tr id="row12703133911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146311617218"><a name="p146311617218"></a><a name="p146311617218"></a>86</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1746773613276"><a name="p1746773613276"></a><a name="p1746773613276"></a>clamp_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44671236142716"><a name="p44671236142716"></a><a name="p44671236142716"></a>clamp_max_npu</p>
</td>
</tr>
<tr id="row37031139181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4463416326"><a name="p4463416326"></a><a name="p4463416326"></a>87</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17467123672713"><a name="p17467123672713"></a><a name="p17467123672713"></a>clamp_max_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1446713360278"><a name="p1446713360278"></a><a name="p1446713360278"></a>clamp_max_npu_</p>
</td>
</tr>
<tr id="row12703123961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7463816828"><a name="p7463816828"></a><a name="p7463816828"></a>88</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1546717361273"><a name="p1546717361273"></a><a name="p1546717361273"></a>clamp_max.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1446723662719"><a name="p1446723662719"></a><a name="p1446723662719"></a>clamp_max_out_npu</p>
</td>
</tr>
<tr id="row170473991217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12463516727"><a name="p12463516727"></a><a name="p12463516727"></a>89</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p546773620273"><a name="p546773620273"></a><a name="p546773620273"></a>clamp_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10467113652719"><a name="p10467113652719"></a><a name="p10467113652719"></a>clamp_min_npu</p>
</td>
</tr>
<tr id="row370416391125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17463181613212"><a name="p17463181613212"></a><a name="p17463181613212"></a>90</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p846703672716"><a name="p846703672716"></a><a name="p846703672716"></a>clamp_min_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6467173672716"><a name="p6467173672716"></a><a name="p6467173672716"></a>clamp_min_npu_</p>
</td>
</tr>
<tr id="row12704173941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246381613220"><a name="p1246381613220"></a><a name="p1246381613220"></a>91</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16468123612273"><a name="p16468123612273"></a><a name="p16468123612273"></a>clamp_min.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1846843619272"><a name="p1846843619272"></a><a name="p1846843619272"></a>clamp_min_out_npu</p>
</td>
</tr>
<tr id="row6704239131211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14464116423"><a name="p14464116423"></a><a name="p14464116423"></a>92</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15468163652720"><a name="p15468163652720"></a><a name="p15468163652720"></a>constant_pad_nd</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p046816362274"><a name="p046816362274"></a><a name="p046816362274"></a>constant_pad_nd_npu</p>
</td>
</tr>
<tr id="row1570493911129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15464141617210"><a name="p15464141617210"></a><a name="p15464141617210"></a>93</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1446819369277"><a name="p1446819369277"></a><a name="p1446819369277"></a>contiguous</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14689365273"><a name="p14689365273"></a><a name="p14689365273"></a>contiguous_npu</p>
</td>
</tr>
<tr id="row27048393125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104644164219"><a name="p104644164219"></a><a name="p104644164219"></a>94</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p346843692714"><a name="p346843692714"></a><a name="p346843692714"></a>convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3468103610274"><a name="p3468103610274"></a><a name="p3468103610274"></a>convolution_npu</p>
</td>
</tr>
<tr id="row6704173911219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p246411614210"><a name="p246411614210"></a><a name="p246411614210"></a>95</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p946893662716"><a name="p946893662716"></a><a name="p946893662716"></a>_convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1746873619271"><a name="p1746873619271"></a><a name="p1746873619271"></a>_convolution_npu</p>
</td>
</tr>
<tr id="row1070423914123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12464201619213"><a name="p12464201619213"></a><a name="p12464201619213"></a>96</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6468193602719"><a name="p6468193602719"></a><a name="p6468193602719"></a>_convolution_nogroup</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p846833611272"><a name="p846833611272"></a><a name="p846833611272"></a>_convolution_nogroup_npu</p>
</td>
</tr>
<tr id="row1704193951216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144646166213"><a name="p144646166213"></a><a name="p144646166213"></a>97</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1446833662710"><a name="p1446833662710"></a><a name="p1446833662710"></a>conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174681036182719"><a name="p174681036182719"></a><a name="p174681036182719"></a>conv2d_npu_</p>
</td>
</tr>
<tr id="row14704113914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84643166215"><a name="p84643166215"></a><a name="p84643166215"></a>98</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1246823672711"><a name="p1246823672711"></a><a name="p1246823672711"></a>conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34681736202717"><a name="p34681736202717"></a><a name="p34681736202717"></a>_conv3d_npu</p>
</td>
</tr>
<tr id="row207047394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154641616627"><a name="p154641616627"></a><a name="p154641616627"></a>99</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p04681036182715"><a name="p04681036182715"></a><a name="p04681036182715"></a>conv_tbc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p646853614272"><a name="p646853614272"></a><a name="p646853614272"></a>conv_tbc_npu</p>
</td>
</tr>
<tr id="row14705103916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19464181613216"><a name="p19464181613216"></a><a name="p19464181613216"></a>100</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5468113612270"><a name="p5468113612270"></a><a name="p5468113612270"></a>conv_tbc_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1546920366276"><a name="p1546920366276"></a><a name="p1546920366276"></a>conv_tbc_backward_npu</p>
</td>
</tr>
<tr id="row15705193961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6464131617218"><a name="p6464131617218"></a><a name="p6464131617218"></a>101</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1546914367276"><a name="p1546914367276"></a><a name="p1546914367276"></a>conv_transpose2d.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4469636202710"><a name="p4469636202710"></a><a name="p4469636202710"></a>conv_transpose2d_npu_</p>
</td>
</tr>
<tr id="row270513391126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p946419161529"><a name="p946419161529"></a><a name="p946419161529"></a>102</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17469936192719"><a name="p17469936192719"></a><a name="p17469936192719"></a>conv_transpose3d.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p446993622713"><a name="p446993622713"></a><a name="p446993622713"></a>conv_transpose3d_npu_</p>
</td>
</tr>
<tr id="row15705153951215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114647166218"><a name="p114647166218"></a><a name="p114647166218"></a>103</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10469103682711"><a name="p10469103682711"></a><a name="p10469103682711"></a>copy_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p446933652712"><a name="p446933652712"></a><a name="p446933652712"></a>copy_npu_</p>
</td>
</tr>
<tr id="row970573915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164642016822"><a name="p164642016822"></a><a name="p164642016822"></a>104</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p04691365272"><a name="p04691365272"></a><a name="p04691365272"></a>cos</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p154692362276"><a name="p154692362276"></a><a name="p154692362276"></a>cos_npu</p>
</td>
</tr>
<tr id="row107052039171216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246413168213"><a name="p1246413168213"></a><a name="p1246413168213"></a>105</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14691636102715"><a name="p14691636102715"></a><a name="p14691636102715"></a>cos_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54691436132710"><a name="p54691436132710"></a><a name="p54691436132710"></a>cos_npu_</p>
</td>
</tr>
<tr id="row17705203951218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04648168215"><a name="p04648168215"></a><a name="p04648168215"></a>106</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1446917364273"><a name="p1446917364273"></a><a name="p1446917364273"></a>cos.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44694364274"><a name="p44694364274"></a><a name="p44694364274"></a>cos_out_npu</p>
</td>
</tr>
<tr id="row1470543918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16465201612216"><a name="p16465201612216"></a><a name="p16465201612216"></a>107</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2469636162713"><a name="p2469636162713"></a><a name="p2469636162713"></a>cosh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4469143682712"><a name="p4469143682712"></a><a name="p4469143682712"></a>cosh_npu</p>
</td>
</tr>
<tr id="row12707133981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13465171617216"><a name="p13465171617216"></a><a name="p13465171617216"></a>108</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p546963612715"><a name="p546963612715"></a><a name="p546963612715"></a>cosh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p546923692713"><a name="p546923692713"></a><a name="p546923692713"></a>cosh_npu_</p>
</td>
</tr>
<tr id="row197089397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19465216628"><a name="p19465216628"></a><a name="p19465216628"></a>109</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10469103617276"><a name="p10469103617276"></a><a name="p10469103617276"></a>cosh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20469136172717"><a name="p20469136172717"></a><a name="p20469136172717"></a>cosh_out_npu</p>
</td>
</tr>
<tr id="row147081039121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104652165215"><a name="p104652165215"></a><a name="p104652165215"></a>110</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7469436182719"><a name="p7469436182719"></a><a name="p7469436182719"></a>_cummax_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7469153613279"><a name="p7469153613279"></a><a name="p7469153613279"></a>cummax_helper_npu</p>
</td>
</tr>
<tr id="row1470863918121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1046561620217"><a name="p1046561620217"></a><a name="p1046561620217"></a>111</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1646923612713"><a name="p1646923612713"></a><a name="p1646923612713"></a>_cummin_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3469193610274"><a name="p3469193610274"></a><a name="p3469193610274"></a>cummin_helper_npu</p>
</td>
</tr>
<tr id="row8708203951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14651162210"><a name="p14651162210"></a><a name="p14651162210"></a>112</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2469153622719"><a name="p2469153622719"></a><a name="p2469153622719"></a>cumprod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p746983614272"><a name="p746983614272"></a><a name="p746983614272"></a>cumprod_npu</p>
</td>
</tr>
<tr id="row8708103941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446541617213"><a name="p446541617213"></a><a name="p446541617213"></a>113</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19469173612714"><a name="p19469173612714"></a><a name="p19469173612714"></a>cumprod.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17469113662714"><a name="p17469113662714"></a><a name="p17469113662714"></a>cumprod_out_npu</p>
</td>
</tr>
<tr id="row17708143911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134651116426"><a name="p134651116426"></a><a name="p134651116426"></a>114</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4469163614278"><a name="p4469163614278"></a><a name="p4469163614278"></a>cumprod.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1469436112710"><a name="p1469436112710"></a><a name="p1469436112710"></a>cumprod_npu</p>
</td>
</tr>
<tr id="row11708839101210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12465161617219"><a name="p12465161617219"></a><a name="p12465161617219"></a>115</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17469536112714"><a name="p17469536112714"></a><a name="p17469536112714"></a>cumprod.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124705365277"><a name="p124705365277"></a><a name="p124705365277"></a>cumprod_out_npu</p>
</td>
</tr>
<tr id="row1870815396128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p204653163214"><a name="p204653163214"></a><a name="p204653163214"></a>116</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14470836102711"><a name="p14470836102711"></a><a name="p14470836102711"></a>ctc_loss.IntList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1747083632712"><a name="p1747083632712"></a><a name="p1747083632712"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row77081539121218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15465151619215"><a name="p15465151619215"></a><a name="p15465151619215"></a>117</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p134707365270"><a name="p134707365270"></a><a name="p134707365270"></a>ctc_loss.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p947043617278"><a name="p947043617278"></a><a name="p947043617278"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row18708123911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1346541613215"><a name="p1346541613215"></a><a name="p1346541613215"></a>118</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12470143616273"><a name="p12470143616273"></a><a name="p12470143616273"></a>_ctc_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114701936162717"><a name="p114701936162717"></a><a name="p114701936162717"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row15708153941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164652161822"><a name="p164652161822"></a><a name="p164652161822"></a>119</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p747063662711"><a name="p747063662711"></a><a name="p747063662711"></a>_ctc_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10470336152712"><a name="p10470336152712"></a><a name="p10470336152712"></a>ctc_loss_backward_npu</p>
</td>
</tr>
<tr id="row147081539111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34651516924"><a name="p34651516924"></a><a name="p34651516924"></a>120</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11470113611277"><a name="p11470113611277"></a><a name="p11470113611277"></a>fill_diagonal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p64701936162710"><a name="p64701936162710"></a><a name="p64701936162710"></a>fill_diagonal_npu_</p>
</td>
</tr>
<tr id="row47091839121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546510161423"><a name="p546510161423"></a><a name="p546510161423"></a>121</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7470136132716"><a name="p7470136132716"></a><a name="p7470136132716"></a>div.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147014364277"><a name="p1147014364277"></a><a name="p1147014364277"></a>div_npu</p>
</td>
</tr>
<tr id="row18709183971211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846571616211"><a name="p1846571616211"></a><a name="p1846571616211"></a>122</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3470236112711"><a name="p3470236112711"></a><a name="p3470236112711"></a>div_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3470236202711"><a name="p3470236202711"></a><a name="p3470236202711"></a>div_npu_</p>
</td>
</tr>
<tr id="row07096390129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1465121614218"><a name="p1465121614218"></a><a name="p1465121614218"></a>123</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13470133612277"><a name="p13470133612277"></a><a name="p13470133612277"></a>div.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147010363275"><a name="p1147010363275"></a><a name="p1147010363275"></a>div_out_npu</p>
</td>
</tr>
<tr id="row1370903971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446518161627"><a name="p446518161627"></a><a name="p446518161627"></a>124</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p104702369278"><a name="p104702369278"></a><a name="p104702369278"></a>div.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p247073622714"><a name="p247073622714"></a><a name="p247073622714"></a>div_npu</p>
</td>
</tr>
<tr id="row070993961220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84653161124"><a name="p84653161124"></a><a name="p84653161124"></a>125</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1047083612273"><a name="p1047083612273"></a><a name="p1047083612273"></a>div_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12470133632714"><a name="p12470133632714"></a><a name="p12470133632714"></a>div_npu_</p>
</td>
</tr>
<tr id="row0709143941212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114665161227"><a name="p114665161227"></a><a name="p114665161227"></a>126</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9470203632711"><a name="p9470203632711"></a><a name="p9470203632711"></a>dot</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p747053611270"><a name="p747053611270"></a><a name="p747053611270"></a>dot_npu</p>
</td>
</tr>
<tr id="row1570919393128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546613161522"><a name="p1546613161522"></a><a name="p1546613161522"></a>127</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1747043602713"><a name="p1747043602713"></a><a name="p1747043602713"></a>dot.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9470123612712"><a name="p9470123612712"></a><a name="p9470123612712"></a>dot_out_npu</p>
</td>
</tr>
<tr id="row670933917126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154661516327"><a name="p154661516327"></a><a name="p154661516327"></a>128</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84700368276"><a name="p84700368276"></a><a name="p84700368276"></a>embedding</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19470133622710"><a name="p19470133622710"></a><a name="p19470133622710"></a>embedding_npu</p>
</td>
</tr>
<tr id="row770983915126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546612161524"><a name="p546612161524"></a><a name="p546612161524"></a>129</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18470203682718"><a name="p18470203682718"></a><a name="p18470203682718"></a>embedding_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134701536182711"><a name="p134701536182711"></a><a name="p134701536182711"></a>embedding_backward_npu</p>
</td>
</tr>
<tr id="row57091739171214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746617161425"><a name="p1746617161425"></a><a name="p1746617161425"></a>130</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1447012362275"><a name="p1447012362275"></a><a name="p1447012362275"></a>embedding_dense_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174711736152717"><a name="p174711736152717"></a><a name="p174711736152717"></a>embedding_dense_backward_npu</p>
</td>
</tr>
<tr id="row1710123916127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p946671618212"><a name="p946671618212"></a><a name="p946671618212"></a>131</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p947113620274"><a name="p947113620274"></a><a name="p947113620274"></a>embedding_renorm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15471136132718"><a name="p15471136132718"></a><a name="p15471136132718"></a>embedding_renorm_npu_</p>
</td>
</tr>
<tr id="row1871033917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7466181613217"><a name="p7466181613217"></a><a name="p7466181613217"></a>132</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1747115365275"><a name="p1747115365275"></a><a name="p1747115365275"></a>_embedding_bag</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p447110363279"><a name="p447110363279"></a><a name="p447110363279"></a>_embedding_bag_npu</p>
</td>
</tr>
<tr id="row471017396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846691612218"><a name="p1846691612218"></a><a name="p1846691612218"></a>133</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0471113672715"><a name="p0471113672715"></a><a name="p0471113672715"></a>empty.memory_format</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p347193613279"><a name="p347193613279"></a><a name="p347193613279"></a>empty_npu</p>
</td>
</tr>
<tr id="row87101939181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184668167214"><a name="p184668167214"></a><a name="p184668167214"></a>134</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15471133613279"><a name="p15471133613279"></a><a name="p15471133613279"></a>resize_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8471143652717"><a name="p8471143652717"></a><a name="p8471143652717"></a>resize_npu_</p>
</td>
</tr>
<tr id="row7710193911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346618161427"><a name="p346618161427"></a><a name="p346618161427"></a>135</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1147143612712"><a name="p1147143612712"></a><a name="p1147143612712"></a>empty_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p247163617276"><a name="p247163617276"></a><a name="p247163617276"></a>empty_like_npu</p>
</td>
</tr>
<tr id="row1871053961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174663161022"><a name="p174663161022"></a><a name="p174663161022"></a>136</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p647163652719"><a name="p647163652719"></a><a name="p647163652719"></a>empty_strided</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p147153682715"><a name="p147153682715"></a><a name="p147153682715"></a>empty_strided_npu</p>
</td>
</tr>
<tr id="row87101439151213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546691616213"><a name="p1546691616213"></a><a name="p1546691616213"></a>137</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14714367271"><a name="p14714367271"></a><a name="p14714367271"></a>erf</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8471143617276"><a name="p8471143617276"></a><a name="p8471143617276"></a>erf_npu</p>
</td>
</tr>
<tr id="row9710113951217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846610161326"><a name="p1846610161326"></a><a name="p1846610161326"></a>138</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1471103618274"><a name="p1471103618274"></a><a name="p1471103618274"></a>erf_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2471163642713"><a name="p2471163642713"></a><a name="p2471163642713"></a>erf_npu_</p>
</td>
</tr>
<tr id="row4710143961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846614162214"><a name="p846614162214"></a><a name="p846614162214"></a>139</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9471103612277"><a name="p9471103612277"></a><a name="p9471103612277"></a>erf.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1847117367278"><a name="p1847117367278"></a><a name="p1847117367278"></a>erf_out_npu</p>
</td>
</tr>
<tr id="row107101539181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184671416828"><a name="p184671416828"></a><a name="p184671416828"></a>140</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p174710361274"><a name="p174710361274"></a><a name="p174710361274"></a>erfc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18471736172719"><a name="p18471736172719"></a><a name="p18471736172719"></a>erfc_npu</p>
</td>
</tr>
<tr id="row12710739111212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24678168216"><a name="p24678168216"></a><a name="p24678168216"></a>141</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11471163614279"><a name="p11471163614279"></a><a name="p11471163614279"></a>erfc_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11471183615277"><a name="p11471183615277"></a><a name="p11471183615277"></a>erfc_npu_</p>
</td>
</tr>
<tr id="row1771193971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24672161213"><a name="p24672161213"></a><a name="p24672161213"></a>142</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1647117367273"><a name="p1647117367273"></a><a name="p1647117367273"></a>erfc.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15471193612271"><a name="p15471193612271"></a><a name="p15471193612271"></a>erfc_out_npu</p>
</td>
</tr>
<tr id="row12711193917124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146716168216"><a name="p1146716168216"></a><a name="p1146716168216"></a>143</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1047110361270"><a name="p1047110361270"></a><a name="p1047110361270"></a>exp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134715364274"><a name="p134715364274"></a><a name="p134715364274"></a>exp_npu</p>
</td>
</tr>
<tr id="row5711439191211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246791614218"><a name="p1246791614218"></a><a name="p1246791614218"></a>144</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p947113615273"><a name="p947113615273"></a><a name="p947113615273"></a>exp_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p547113602715"><a name="p547113602715"></a><a name="p547113602715"></a>exp_npu_</p>
</td>
</tr>
<tr id="row8711113910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646741611219"><a name="p1646741611219"></a><a name="p1646741611219"></a>145</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1347110362277"><a name="p1347110362277"></a><a name="p1347110362277"></a>exp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4471203619278"><a name="p4471203619278"></a><a name="p4471203619278"></a>exp_out_npu</p>
</td>
</tr>
<tr id="row107111639131216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846717161126"><a name="p846717161126"></a><a name="p846717161126"></a>146</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1147243612716"><a name="p1147243612716"></a><a name="p1147243612716"></a>expm1</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134721936152711"><a name="p134721936152711"></a><a name="p134721936152711"></a>expm1_npu</p>
</td>
</tr>
<tr id="row18711103921219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12467416324"><a name="p12467416324"></a><a name="p12467416324"></a>147</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p747211369278"><a name="p747211369278"></a><a name="p747211369278"></a>expm1_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94721636122717"><a name="p94721636122717"></a><a name="p94721636122717"></a>expm1_npu_</p>
</td>
</tr>
<tr id="row14711839151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446713161229"><a name="p446713161229"></a><a name="p446713161229"></a>148</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18472436162716"><a name="p18472436162716"></a><a name="p18472436162716"></a>expm1.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34721236142711"><a name="p34721236142711"></a><a name="p34721236142711"></a>expm1_out_npu</p>
</td>
</tr>
<tr id="row67121739171219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34682016926"><a name="p34682016926"></a><a name="p34682016926"></a>149</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p44721336192714"><a name="p44721336192714"></a><a name="p44721336192714"></a>eye</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8472123618274"><a name="p8472123618274"></a><a name="p8472123618274"></a>eye_npu</p>
</td>
</tr>
<tr id="row167127398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44689161523"><a name="p44689161523"></a><a name="p44689161523"></a>150</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12472103619271"><a name="p12472103619271"></a><a name="p12472103619271"></a>eye.m</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18472236162711"><a name="p18472236162711"></a><a name="p18472236162711"></a>eye_npu</p>
</td>
</tr>
<tr id="row2712123912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154681916524"><a name="p154681916524"></a><a name="p154681916524"></a>151</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p164721136192714"><a name="p164721136192714"></a><a name="p164721136192714"></a>eye.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10472103614279"><a name="p10472103614279"></a><a name="p10472103614279"></a>eye_out_npu</p>
</td>
</tr>
<tr id="row157121739161214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24685161225"><a name="p24685161225"></a><a name="p24685161225"></a>152</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p04725365276"><a name="p04725365276"></a><a name="p04725365276"></a>eye.m_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13472163611272"><a name="p13472163611272"></a><a name="p13472163611272"></a>eye_out_npu</p>
</td>
</tr>
<tr id="row1171283971211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146881616216"><a name="p1146881616216"></a><a name="p1146881616216"></a>153</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p847243615271"><a name="p847243615271"></a><a name="p847243615271"></a>fill_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p347211362271"><a name="p347211362271"></a><a name="p347211362271"></a>fill_npu_</p>
</td>
</tr>
<tr id="row15712439111216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146815164212"><a name="p146815164212"></a><a name="p146815164212"></a>154</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16472143619277"><a name="p16472143619277"></a><a name="p16472143619277"></a>fill_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147253692714"><a name="p1147253692714"></a><a name="p1147253692714"></a>fill_npu_</p>
</td>
</tr>
<tr id="row18712133915128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p646821618213"><a name="p646821618213"></a><a name="p646821618213"></a>155</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10472136132719"><a name="p10472136132719"></a><a name="p10472136132719"></a>floor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144722366278"><a name="p144722366278"></a><a name="p144722366278"></a>floor_npu</p>
</td>
</tr>
<tr id="row171243912124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184689161121"><a name="p184689161121"></a><a name="p184689161121"></a>156</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34721836102713"><a name="p34721836102713"></a><a name="p34721836102713"></a>floor_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1747213364273"><a name="p1747213364273"></a><a name="p1747213364273"></a>floor_npu_</p>
</td>
</tr>
<tr id="row07121539141212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16468181619218"><a name="p16468181619218"></a><a name="p16468181619218"></a>157</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10472123622716"><a name="p10472123622716"></a><a name="p10472123622716"></a>floor.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44721436192719"><a name="p44721436192719"></a><a name="p44721436192719"></a>floor_out_npu</p>
</td>
</tr>
<tr id="row071373901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146814162215"><a name="p1146814162215"></a><a name="p1146814162215"></a>158</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9472103612719"><a name="p9472103612719"></a><a name="p9472103612719"></a>floor_divide</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5472736102718"><a name="p5472736102718"></a><a name="p5472736102718"></a>floor_divide_npu</p>
</td>
</tr>
<tr id="row107131393122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546810161821"><a name="p1546810161821"></a><a name="p1546810161821"></a>159</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7472113615276"><a name="p7472113615276"></a><a name="p7472113615276"></a>floor_divide_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19472936202712"><a name="p19472936202712"></a><a name="p19472936202712"></a>floor_divide_npu_</p>
</td>
</tr>
<tr id="row671383921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134681616022"><a name="p134681616022"></a><a name="p134681616022"></a>160</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11472123672719"><a name="p11472123672719"></a><a name="p11472123672719"></a>floor_divide.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p204721136162719"><a name="p204721136162719"></a><a name="p204721136162719"></a>floor_divide_out_npu</p>
</td>
</tr>
<tr id="row1171303931210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7468616423"><a name="p7468616423"></a><a name="p7468616423"></a>161</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9473113622713"><a name="p9473113622713"></a><a name="p9473113622713"></a>floor_divide.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1547373612710"><a name="p1547373612710"></a><a name="p1547373612710"></a>floor_divide_npu</p>
</td>
</tr>
<tr id="row117131339161213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646841618215"><a name="p1646841618215"></a><a name="p1646841618215"></a>162</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194731136172712"><a name="p194731136172712"></a><a name="p194731136172712"></a>floor_divide_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147393619271"><a name="p1147393619271"></a><a name="p1147393619271"></a>floor_divide_npu_</p>
</td>
</tr>
<tr id="row771333941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24691816223"><a name="p24691816223"></a><a name="p24691816223"></a>163</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1847313622710"><a name="p1847313622710"></a><a name="p1847313622710"></a>frac</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p847363619279"><a name="p847363619279"></a><a name="p847363619279"></a>frac_npu</p>
</td>
</tr>
<tr id="row371317396123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346916163210"><a name="p346916163210"></a><a name="p346916163210"></a>164</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1947314363272"><a name="p1947314363272"></a><a name="p1947314363272"></a>frac_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0473236162716"><a name="p0473236162716"></a><a name="p0473236162716"></a>frac_npu_</p>
</td>
</tr>
<tr id="row1871317392121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154692016822"><a name="p154692016822"></a><a name="p154692016822"></a>165</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3473203611278"><a name="p3473203611278"></a><a name="p3473203611278"></a>frac.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54731736132716"><a name="p54731736132716"></a><a name="p54731736132716"></a>frac_out_npu</p>
</td>
</tr>
<tr id="row971313918123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546913161320"><a name="p1546913161320"></a><a name="p1546913161320"></a>166</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1747343692716"><a name="p1747343692716"></a><a name="p1747343692716"></a>full.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15473636182711"><a name="p15473636182711"></a><a name="p15473636182711"></a>full_npu</p>
</td>
</tr>
<tr id="row2713939191216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18469416720"><a name="p18469416720"></a><a name="p18469416720"></a>167</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13473153612273"><a name="p13473153612273"></a><a name="p13473153612273"></a>full</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1847303620271"><a name="p1847303620271"></a><a name="p1847303620271"></a>full_npu</p>
</td>
</tr>
<tr id="row107131039161211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1446915162212"><a name="p1446915162212"></a><a name="p1446915162212"></a>168</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16473163611271"><a name="p16473163611271"></a><a name="p16473163611271"></a>full.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p447319368279"><a name="p447319368279"></a><a name="p447319368279"></a>full_out_npu</p>
</td>
</tr>
<tr id="row15714103901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44691016722"><a name="p44691016722"></a><a name="p44691016722"></a>169</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7473936182713"><a name="p7473936182713"></a><a name="p7473936182713"></a>grid_sampler</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14732036122715"><a name="p14732036122715"></a><a name="p14732036122715"></a>grid_sampler_npu</p>
</td>
</tr>
<tr id="row37144394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146921618216"><a name="p1146921618216"></a><a name="p1146921618216"></a>170</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18473103632712"><a name="p18473103632712"></a><a name="p18473103632712"></a>grid_sampler_3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1473836172712"><a name="p1473836172712"></a><a name="p1473836172712"></a>grid_sampler_3d_npu</p>
</td>
</tr>
<tr id="row107141639111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4469121610218"><a name="p4469121610218"></a><a name="p4469121610218"></a>171</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124733363276"><a name="p124733363276"></a><a name="p124733363276"></a>grid_sampler_3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4473163613275"><a name="p4473163613275"></a><a name="p4473163613275"></a>grid_sampler_3d_backward_npu</p>
</td>
</tr>
<tr id="row207141396120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546911161924"><a name="p1546911161924"></a><a name="p1546911161924"></a>172</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1347393612716"><a name="p1347393612716"></a><a name="p1347393612716"></a>hann_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0473173672715"><a name="p0473173672715"></a><a name="p0473173672715"></a>hann_window_npu</p>
</td>
</tr>
<tr id="row2714143971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19469131611216"><a name="p19469131611216"></a><a name="p19469131611216"></a>173</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17473143602710"><a name="p17473143602710"></a><a name="p17473143602710"></a>hann_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17473736142714"><a name="p17473736142714"></a><a name="p17473736142714"></a>hann_window_npu</p>
</td>
</tr>
<tr id="row871433991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146910166212"><a name="p146910166212"></a><a name="p146910166212"></a>174</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13473193642710"><a name="p13473193642710"></a><a name="p13473193642710"></a>hamming_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19473133652720"><a name="p19473133652720"></a><a name="p19473133652720"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row371493914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646918161724"><a name="p1646918161724"></a><a name="p1646918161724"></a>175</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34749369278"><a name="p34749369278"></a><a name="p34749369278"></a>hamming_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1847413672712"><a name="p1847413672712"></a><a name="p1847413672712"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row471433931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34691316923"><a name="p34691316923"></a><a name="p34691316923"></a>176</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p147423662713"><a name="p147423662713"></a><a name="p147423662713"></a>hamming_window.periodic_alpha</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1047453610275"><a name="p1047453610275"></a><a name="p1047453610275"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row9714173971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74692161215"><a name="p74692161215"></a><a name="p74692161215"></a>177</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16474183610272"><a name="p16474183610272"></a><a name="p16474183610272"></a>hamming_window.periodic_alpha_beta</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1647473662714"><a name="p1647473662714"></a><a name="p1647473662714"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row187141539171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54692016828"><a name="p54692016828"></a><a name="p54692016828"></a>178</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p74741536102716"><a name="p74741536102716"></a><a name="p74741536102716"></a>ger</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44747366275"><a name="p44747366275"></a><a name="p44747366275"></a>ger_npu</p>
</td>
</tr>
<tr id="row1714183941217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144693161523"><a name="p144693161523"></a><a name="p144693161523"></a>179</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p204748362272"><a name="p204748362272"></a><a name="p204748362272"></a>ger.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84749362274"><a name="p84749362274"></a><a name="p84749362274"></a>ger_out_npu</p>
</td>
</tr>
<tr id="row4715193981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547061610211"><a name="p547061610211"></a><a name="p547061610211"></a>180</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12474183602713"><a name="p12474183602713"></a><a name="p12474183602713"></a>index.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1047414369274"><a name="p1047414369274"></a><a name="p1047414369274"></a>index_npu</p>
</td>
</tr>
<tr id="row1715193921210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047012161524"><a name="p1047012161524"></a><a name="p1047012161524"></a>181</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1474183617271"><a name="p1474183617271"></a><a name="p1474183617271"></a>index_put_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19474123617272"><a name="p19474123617272"></a><a name="p19474123617272"></a>index_put_npu_</p>
</td>
</tr>
<tr id="row1671583917123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154708168213"><a name="p154708168213"></a><a name="p154708168213"></a>182</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84743368272"><a name="p84743368272"></a><a name="p84743368272"></a>index_put</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p154744363271"><a name="p154744363271"></a><a name="p154744363271"></a>index_put_npu</p>
</td>
</tr>
<tr id="row5715339141220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347041613213"><a name="p1347041613213"></a><a name="p1347041613213"></a>183</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9474103602713"><a name="p9474103602713"></a><a name="p9474103602713"></a>_index_put_impl_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3474536122716"><a name="p3474536122716"></a><a name="p3474536122716"></a>_index_put_impl_npu_</p>
</td>
</tr>
<tr id="row771512390120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19470111613218"><a name="p19470111613218"></a><a name="p19470111613218"></a>184</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84741361276"><a name="p84741361276"></a><a name="p84741361276"></a>inverse</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1947419368271"><a name="p1947419368271"></a><a name="p1947419368271"></a>inverse_npu</p>
</td>
</tr>
<tr id="row14715439151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74701416926"><a name="p74701416926"></a><a name="p74701416926"></a>185</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10474103612273"><a name="p10474103612273"></a><a name="p10474103612273"></a>inverse.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2047423632717"><a name="p2047423632717"></a><a name="p2047423632717"></a>inverse_out_npu</p>
</td>
</tr>
<tr id="row127151139161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14701166211"><a name="p14701166211"></a><a name="p14701166211"></a>186</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19474193615272"><a name="p19474193615272"></a><a name="p19474193615272"></a>isclose</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3474536142719"><a name="p3474536142719"></a><a name="p3474536142719"></a>isclose_npu</p>
</td>
</tr>
<tr id="row137154396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547011161120"><a name="p1547011161120"></a><a name="p1547011161120"></a>187</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34745366276"><a name="p34745366276"></a><a name="p34745366276"></a>isnan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p947483692718"><a name="p947483692718"></a><a name="p947483692718"></a>isnan_npu</p>
</td>
</tr>
<tr id="row1071553981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7470161619217"><a name="p7470161619217"></a><a name="p7470161619217"></a>188</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3474183611274"><a name="p3474183611274"></a><a name="p3474183611274"></a>is_nonzero</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17474736172710"><a name="p17474736172710"></a><a name="p17474736172710"></a>is_nonzero_npu</p>
</td>
</tr>
<tr id="row1871533901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747014161825"><a name="p747014161825"></a><a name="p747014161825"></a>189</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p147410362273"><a name="p147410362273"></a><a name="p147410362273"></a>kl_div</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134741836112717"><a name="p134741836112717"></a><a name="p134741836112717"></a>kl_div_npu</p>
</td>
</tr>
<tr id="row14715173914125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174701160210"><a name="p174701160210"></a><a name="p174701160210"></a>190</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1147573618276"><a name="p1147573618276"></a><a name="p1147573618276"></a>kl_div_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7475113616277"><a name="p7475113616277"></a><a name="p7475113616277"></a>kl_div_backward_npu</p>
</td>
</tr>
<tr id="row07162395125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3470121616216"><a name="p3470121616216"></a><a name="p3470121616216"></a>191</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114758364276"><a name="p114758364276"></a><a name="p114758364276"></a>kthvalue</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5475123612716"><a name="p5475123612716"></a><a name="p5475123612716"></a>kthvalue_npu</p>
</td>
</tr>
<tr id="row15716639181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114709169219"><a name="p114709169219"></a><a name="p114709169219"></a>192</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6475123622710"><a name="p6475123622710"></a><a name="p6475123622710"></a>kthvalue.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p247523632716"><a name="p247523632716"></a><a name="p247523632716"></a>kthvalue_out_npu</p>
</td>
</tr>
<tr id="row1671618395124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14706161026"><a name="p14706161026"></a><a name="p14706161026"></a>193</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p547543610273"><a name="p547543610273"></a><a name="p547543610273"></a>kthvalue.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9475103617279"><a name="p9475103617279"></a><a name="p9475103617279"></a>kthvalue_npu</p>
</td>
</tr>
<tr id="row12716203961220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447161611210"><a name="p1447161611210"></a><a name="p1447161611210"></a>194</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5475193613275"><a name="p5475193613275"></a><a name="p5475193613275"></a>kthvalue.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174751036182712"><a name="p174751036182712"></a><a name="p174751036182712"></a>kthvalue_out_npu</p>
</td>
</tr>
<tr id="row15716183918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347131616211"><a name="p1347131616211"></a><a name="p1347131616211"></a>195</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10475183692713"><a name="p10475183692713"></a><a name="p10475183692713"></a>native_layer_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147510363279"><a name="p1147510363279"></a><a name="p1147510363279"></a>layer_norm_npu</p>
</td>
</tr>
<tr id="row11716143981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74711616929"><a name="p74711616929"></a><a name="p74711616929"></a>196</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18475136112719"><a name="p18475136112719"></a><a name="p18475136112719"></a>native_layer_norm_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p64755364272"><a name="p64755364272"></a><a name="p64755364272"></a>layer_norm_backward_npu</p>
</td>
</tr>
<tr id="row3716193911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14471116426"><a name="p14471116426"></a><a name="p14471116426"></a>197</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24751636202710"><a name="p24751636202710"></a><a name="p24751636202710"></a>linspace</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0475113613272"><a name="p0475113613272"></a><a name="p0475113613272"></a>linspace_npu</p>
</td>
</tr>
<tr id="row207171039131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447113161821"><a name="p447113161821"></a><a name="p447113161821"></a>198</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1347513616278"><a name="p1347513616278"></a><a name="p1347513616278"></a>linspace.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2475153619277"><a name="p2475153619277"></a><a name="p2475153619277"></a>linspace_out_npu</p>
</td>
</tr>
<tr id="row2717113914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647111161126"><a name="p1647111161126"></a><a name="p1647111161126"></a>199</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p204751636142718"><a name="p204751636142718"></a><a name="p204751636142718"></a>log</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p347503612273"><a name="p347503612273"></a><a name="p347503612273"></a>log_npu</p>
</td>
</tr>
<tr id="row771710399126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134716161920"><a name="p134716161920"></a><a name="p134716161920"></a>200</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1647533642713"><a name="p1647533642713"></a><a name="p1647533642713"></a>log_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1647593662717"><a name="p1647593662717"></a><a name="p1647593662717"></a>log_npu_</p>
</td>
</tr>
<tr id="row77174392127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4471816729"><a name="p4471816729"></a><a name="p4471816729"></a>201</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14475143652719"><a name="p14475143652719"></a><a name="p14475143652719"></a>log.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20475136162712"><a name="p20475136162712"></a><a name="p20475136162712"></a>log_out_npu</p>
</td>
</tr>
<tr id="row1971733971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44711816725"><a name="p44711816725"></a><a name="p44711816725"></a>202</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p144751736112716"><a name="p144751736112716"></a><a name="p144751736112716"></a>log10</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124751368272"><a name="p124751368272"></a><a name="p124751368272"></a>log10_npu</p>
</td>
</tr>
<tr id="row7717939191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447171612212"><a name="p1447171612212"></a><a name="p1447171612212"></a>203</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p647533614274"><a name="p647533614274"></a><a name="p647533614274"></a>log10_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8475236132716"><a name="p8475236132716"></a><a name="p8475236132716"></a>log10_npu_</p>
</td>
</tr>
<tr id="row7717103981220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17471101610212"><a name="p17471101610212"></a><a name="p17471101610212"></a>204</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16475183612276"><a name="p16475183612276"></a><a name="p16475183612276"></a>log10.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14761436182719"><a name="p14761436182719"></a><a name="p14761436182719"></a>log10_out_npu</p>
</td>
</tr>
<tr id="row187181439131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647161610218"><a name="p647161610218"></a><a name="p647161610218"></a>205</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84762368273"><a name="p84762368273"></a><a name="p84762368273"></a>log1p</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14476153692716"><a name="p14476153692716"></a><a name="p14476153692716"></a>log1p_npu</p>
</td>
</tr>
<tr id="row0718139141218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15471116526"><a name="p15471116526"></a><a name="p15471116526"></a>206</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p134767364273"><a name="p134767364273"></a><a name="p134767364273"></a>log1p_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1047610364271"><a name="p1047610364271"></a><a name="p1047610364271"></a>log1p_npu_</p>
</td>
</tr>
<tr id="row571815397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1847118164218"><a name="p1847118164218"></a><a name="p1847118164218"></a>207</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1247663692713"><a name="p1247663692713"></a><a name="p1247663692713"></a>log1p.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34761736202716"><a name="p34761736202716"></a><a name="p34761736202716"></a>log1p_out_npu</p>
</td>
</tr>
<tr id="row187181639141214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14717161323"><a name="p14717161323"></a><a name="p14717161323"></a>208</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p347633611272"><a name="p347633611272"></a><a name="p347633611272"></a>log2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p747623622713"><a name="p747623622713"></a><a name="p747623622713"></a>log2_npu</p>
</td>
</tr>
<tr id="row117186395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847114161426"><a name="p847114161426"></a><a name="p847114161426"></a>209</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1347663622717"><a name="p1347663622717"></a><a name="p1347663622717"></a>log2_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34761636142719"><a name="p34761636142719"></a><a name="p34761636142719"></a>log2_npu_</p>
</td>
</tr>
<tr id="row1071819393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547114160216"><a name="p547114160216"></a><a name="p547114160216"></a>210</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p64761036182714"><a name="p64761036182714"></a><a name="p64761036182714"></a>log2.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13476183613278"><a name="p13476183613278"></a><a name="p13476183613278"></a>log2_out_npu</p>
</td>
</tr>
<tr id="row37188399121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24713161429"><a name="p24713161429"></a><a name="p24713161429"></a>211</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12476173613279"><a name="p12476173613279"></a><a name="p12476173613279"></a>logspace</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p247663652714"><a name="p247663652714"></a><a name="p247663652714"></a>logspace_npu</p>
</td>
</tr>
<tr id="row137187391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194711916925"><a name="p194711916925"></a><a name="p194711916925"></a>212</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p74761036112710"><a name="p74761036112710"></a><a name="p74761036112710"></a>logspace.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2047615366270"><a name="p2047615366270"></a><a name="p2047615366270"></a>logspace_out_npu</p>
</td>
</tr>
<tr id="row16718143912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34722016821"><a name="p34722016821"></a><a name="p34722016821"></a>213</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1476203682712"><a name="p1476203682712"></a><a name="p1476203682712"></a>log_softmax.int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p74760366276"><a name="p74760366276"></a><a name="p74760366276"></a>log_softmax_npu</p>
</td>
</tr>
<tr id="row4718103991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647210161321"><a name="p1647210161321"></a><a name="p1647210161321"></a>214</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p847693610276"><a name="p847693610276"></a><a name="p847693610276"></a>log_softmax.Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1447612363274"><a name="p1447612363274"></a><a name="p1447612363274"></a>log_softmax_npu</p>
</td>
</tr>
<tr id="row271833941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74726164211"><a name="p74726164211"></a><a name="p74726164211"></a>215</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p547620361276"><a name="p547620361276"></a><a name="p547620361276"></a>_log_softmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3476036132716"><a name="p3476036132716"></a><a name="p3476036132716"></a>_log_softmax_npu</p>
</td>
</tr>
<tr id="row197181539111220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9472516521"><a name="p9472516521"></a><a name="p9472516521"></a>216</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1447673652718"><a name="p1447673652718"></a><a name="p1447673652718"></a>_log_softmax_backward_data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16476736112714"><a name="p16476736112714"></a><a name="p16476736112714"></a>_log_softmax_backward_npu</p>
</td>
</tr>
<tr id="row8719239121220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10472151610219"><a name="p10472151610219"></a><a name="p10472151610219"></a>217</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17477736172713"><a name="p17477736172713"></a><a name="p17477736172713"></a>logsumexp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7477836112719"><a name="p7477836112719"></a><a name="p7477836112719"></a>logsumexp_npu</p>
</td>
</tr>
<tr id="row18719173916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047217161822"><a name="p2047217161822"></a><a name="p2047217161822"></a>218</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1147733618272"><a name="p1147733618272"></a><a name="p1147733618272"></a>logsumexp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p447720364279"><a name="p447720364279"></a><a name="p447720364279"></a>logsumexp_out_npu</p>
</td>
</tr>
<tr id="row2719103912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184729161128"><a name="p184729161128"></a><a name="p184729161128"></a>219</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1947713632717"><a name="p1947713632717"></a><a name="p1947713632717"></a>logsumexp.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1747743632710"><a name="p1747743632710"></a><a name="p1747743632710"></a>logsumexp_npu</p>
</td>
</tr>
<tr id="row371910396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347271610213"><a name="p1347271610213"></a><a name="p1347271610213"></a>220</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184771436112715"><a name="p184771436112715"></a><a name="p184771436112715"></a>logsumexp.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p447743618278"><a name="p447743618278"></a><a name="p447743618278"></a>logsumexp_out_npu</p>
</td>
</tr>
<tr id="row371919398126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547211618210"><a name="p547211618210"></a><a name="p547211618210"></a>221</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5477193617277"><a name="p5477193617277"></a><a name="p5477193617277"></a>matmul</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p447753632716"><a name="p447753632716"></a><a name="p447753632716"></a>matmul_npu</p>
</td>
</tr>
<tr id="row9719439151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947219161422"><a name="p1947219161422"></a><a name="p1947219161422"></a>222</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24771436162716"><a name="p24771436162716"></a><a name="p24771436162716"></a>matmul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p247773615271"><a name="p247773615271"></a><a name="p247773615271"></a>matmul_out_npu</p>
</td>
</tr>
<tr id="row2719193921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047213161929"><a name="p1047213161929"></a><a name="p1047213161929"></a>223</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p847720368275"><a name="p847720368275"></a><a name="p847720368275"></a>max.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6477336132710"><a name="p6477336132710"></a><a name="p6477336132710"></a>max_npu</p>
</td>
</tr>
<tr id="row1471913910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24727161023"><a name="p24727161023"></a><a name="p24727161023"></a>224</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114775365279"><a name="p114775365279"></a><a name="p114775365279"></a>max.dim_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1947713692716"><a name="p1947713692716"></a><a name="p1947713692716"></a>max_out_npu</p>
</td>
</tr>
<tr id="row197191739101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18472151613210"><a name="p18472151613210"></a><a name="p18472151613210"></a>225</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p547743672717"><a name="p547743672717"></a><a name="p547743672717"></a>max_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1647713362278"><a name="p1647713362278"></a><a name="p1647713362278"></a>max_npu</p>
</td>
</tr>
<tr id="row137191939101212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164723162023"><a name="p164723162023"></a><a name="p164723162023"></a>226</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p447723618275"><a name="p447723618275"></a><a name="p447723618275"></a>max.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1847793618273"><a name="p1847793618273"></a><a name="p1847793618273"></a>max_npu</p>
</td>
</tr>
<tr id="row471917394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34727161326"><a name="p34727161326"></a><a name="p34727161326"></a>227</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p547753652715"><a name="p547753652715"></a><a name="p547753652715"></a>max.names_dim_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1847753613274"><a name="p1847753613274"></a><a name="p1847753613274"></a>max_out_npu</p>
</td>
</tr>
<tr id="row5720239191220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5473181619210"><a name="p5473181619210"></a><a name="p5473181619210"></a>228</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p847773612714"><a name="p847773612714"></a><a name="p847773612714"></a>max_values.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p847793672713"><a name="p847793672713"></a><a name="p847793672713"></a>max_npu</p>
</td>
</tr>
<tr id="row172093913121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047391614217"><a name="p2047391614217"></a><a name="p2047391614217"></a>229</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1847710368272"><a name="p1847710368272"></a><a name="p1847710368272"></a>max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7477103642719"><a name="p7477103642719"></a><a name="p7477103642719"></a>max_pool2d_npu</p>
</td>
</tr>
<tr id="row1772073901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847391612212"><a name="p847391612212"></a><a name="p847391612212"></a>230</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17477436142714"><a name="p17477436142714"></a><a name="p17477436142714"></a>mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p164771436172720"><a name="p164771436172720"></a><a name="p164771436172720"></a>mean_npu</p>
</td>
</tr>
<tr id="row17720163931217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847314161525"><a name="p847314161525"></a><a name="p847314161525"></a>231</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19477163614278"><a name="p19477163614278"></a><a name="p19477163614278"></a>mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p247773619277"><a name="p247773619277"></a><a name="p247773619277"></a>mean_npu</p>
</td>
</tr>
<tr id="row9720143916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174734164213"><a name="p174734164213"></a><a name="p174734164213"></a>232</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p134771036192715"><a name="p134771036192715"></a><a name="p134771036192715"></a>mean.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1047823662710"><a name="p1047823662710"></a><a name="p1047823662710"></a>mean_out_npu</p>
</td>
</tr>
<tr id="row17201339131212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7473116621"><a name="p7473116621"></a><a name="p7473116621"></a>233</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5478836142711"><a name="p5478836142711"></a><a name="p5478836142711"></a>mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19478163652719"><a name="p19478163652719"></a><a name="p19478163652719"></a>mean_npu</p>
</td>
</tr>
<tr id="row57201039191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114738167211"><a name="p114738167211"></a><a name="p114738167211"></a>234</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18478236152713"><a name="p18478236152713"></a><a name="p18478236152713"></a>mean.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1047814366274"><a name="p1047814366274"></a><a name="p1047814366274"></a>mean_out_npu</p>
</td>
</tr>
<tr id="row1372013918126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p247310161322"><a name="p247310161322"></a><a name="p247310161322"></a>235</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3478123672719"><a name="p3478123672719"></a><a name="p3478123672719"></a>median.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1547812364279"><a name="p1547812364279"></a><a name="p1547812364279"></a>median_npu</p>
</td>
</tr>
<tr id="row3720639201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047313163215"><a name="p1047313163215"></a><a name="p1047313163215"></a>236</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p94781636172710"><a name="p94781636172710"></a><a name="p94781636172710"></a>median.dim_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134788361275"><a name="p134788361275"></a><a name="p134788361275"></a>median_out_npu</p>
</td>
</tr>
<tr id="row107201839161213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2473131618212"><a name="p2473131618212"></a><a name="p2473131618212"></a>237</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18478936142714"><a name="p18478936142714"></a><a name="p18478936142714"></a>median.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p347818361279"><a name="p347818361279"></a><a name="p347818361279"></a>median_npu</p>
</td>
</tr>
<tr id="row1872083991217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164736161622"><a name="p164736161622"></a><a name="p164736161622"></a>238</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1447810367275"><a name="p1447810367275"></a><a name="p1447810367275"></a>median.names_dim_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94782368273"><a name="p94782368273"></a><a name="p94782368273"></a>median_out_npu</p>
</td>
</tr>
<tr id="row1172183941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347314161213"><a name="p1347314161213"></a><a name="p1347314161213"></a>239</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p164784368274"><a name="p164784368274"></a><a name="p164784368274"></a>min.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14781036112715"><a name="p14781036112715"></a><a name="p14781036112715"></a>min_npu</p>
</td>
</tr>
<tr id="row172116399124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747311161124"><a name="p747311161124"></a><a name="p747311161124"></a>240</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1247816366276"><a name="p1247816366276"></a><a name="p1247816366276"></a>min.dim_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p347817368275"><a name="p347817368275"></a><a name="p347817368275"></a>min_out_npu</p>
</td>
</tr>
<tr id="row67218393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4473161612212"><a name="p4473161612212"></a><a name="p4473161612212"></a>241</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p174781736122711"><a name="p174781736122711"></a><a name="p174781736122711"></a>min_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114781736182715"><a name="p114781736182715"></a><a name="p114781736182715"></a>min_npu</p>
</td>
</tr>
<tr id="row1672117397122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164741216821"><a name="p164741216821"></a><a name="p164741216821"></a>242</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1847893622711"><a name="p1847893622711"></a><a name="p1847893622711"></a>min.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11478736152711"><a name="p11478736152711"></a><a name="p11478736152711"></a>min_npu</p>
</td>
</tr>
<tr id="row1572114394124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174741016525"><a name="p174741016525"></a><a name="p174741016525"></a>243</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1047893682718"><a name="p1047893682718"></a><a name="p1047893682718"></a>min.names_dim_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147853619273"><a name="p1147853619273"></a><a name="p1147853619273"></a>min_out_npu</p>
</td>
</tr>
<tr id="row8721139131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447415165218"><a name="p1447415165218"></a><a name="p1447415165218"></a>244</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11478143632719"><a name="p11478143632719"></a><a name="p11478143632719"></a>min_values.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1047816364272"><a name="p1047816364272"></a><a name="p1047816364272"></a>min_npu</p>
</td>
</tr>
<tr id="row1072153917123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174744163217"><a name="p174744163217"></a><a name="p174744163217"></a>245</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p94781236142719"><a name="p94781236142719"></a><a name="p94781236142719"></a>mm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3478153614276"><a name="p3478153614276"></a><a name="p3478153614276"></a>mm_npu</p>
</td>
</tr>
<tr id="row37219396128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54741916527"><a name="p54741916527"></a><a name="p54741916527"></a>246</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84781936122712"><a name="p84781936122712"></a><a name="p84781936122712"></a>mm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p74781436102718"><a name="p74781436102718"></a><a name="p74781436102718"></a>mm_out_npu</p>
</td>
</tr>
<tr id="row1572183912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447412161427"><a name="p447412161427"></a><a name="p447412161427"></a>247</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1947803692714"><a name="p1947803692714"></a><a name="p1947803692714"></a>mul.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3479123616278"><a name="p3479123616278"></a><a name="p3479123616278"></a>mul_npu</p>
</td>
</tr>
<tr id="row167216395126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54741416527"><a name="p54741416527"></a><a name="p54741416527"></a>248</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6479193618274"><a name="p6479193618274"></a><a name="p6479193618274"></a>mul_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2479436132714"><a name="p2479436132714"></a><a name="p2479436132714"></a>mul_npu_</p>
</td>
</tr>
<tr id="row137211039151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64749163218"><a name="p64749163218"></a><a name="p64749163218"></a>249</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1147933672713"><a name="p1147933672713"></a><a name="p1147933672713"></a>mul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1747910369273"><a name="p1747910369273"></a><a name="p1747910369273"></a>mul_out_npu</p>
</td>
</tr>
<tr id="row11722153918121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10474161613215"><a name="p10474161613215"></a><a name="p10474161613215"></a>250</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p147916366272"><a name="p147916366272"></a><a name="p147916366272"></a>mul.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1847903652717"><a name="p1847903652717"></a><a name="p1847903652717"></a>mul_npu</p>
</td>
</tr>
<tr id="row1472273981219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547414161121"><a name="p547414161121"></a><a name="p547414161121"></a>251</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1447993692716"><a name="p1447993692716"></a><a name="p1447993692716"></a>mul_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p347920368275"><a name="p347920368275"></a><a name="p347920368275"></a>mul_npu_</p>
</td>
</tr>
<tr id="row19722103916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047401616212"><a name="p2047401616212"></a><a name="p2047401616212"></a>252</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19479123619277"><a name="p19479123619277"></a><a name="p19479123619277"></a>mv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p447993652713"><a name="p447993652713"></a><a name="p447993652713"></a>mv_npu</p>
</td>
</tr>
<tr id="row16722143916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p347411161023"><a name="p347411161023"></a><a name="p347411161023"></a>253</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13479153682715"><a name="p13479153682715"></a><a name="p13479153682715"></a>mv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12479113642711"><a name="p12479113642711"></a><a name="p12479113642711"></a>mv_out_npu</p>
</td>
</tr>
<tr id="row197221239151219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144745161624"><a name="p144745161624"></a><a name="p144745161624"></a>254</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194791436132715"><a name="p194791436132715"></a><a name="p194791436132715"></a>narrow_copy</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147993616273"><a name="p1147993616273"></a><a name="p1147993616273"></a>narrow_copy_npu</p>
</td>
</tr>
<tr id="row672219394123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547431616212"><a name="p1547431616212"></a><a name="p1547431616212"></a>255</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24791236192715"><a name="p24791236192715"></a><a name="p24791236192715"></a>native_batch_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8479173612711"><a name="p8479173612711"></a><a name="p8479173612711"></a>batch_norm_npu</p>
</td>
</tr>
<tr id="row1872263971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1747416161923"><a name="p1747416161923"></a><a name="p1747416161923"></a>256</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19479163620270"><a name="p19479163620270"></a><a name="p19479163620270"></a>native_batch_norm_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194795369279"><a name="p194795369279"></a><a name="p194795369279"></a>batch_norm_backward_npu</p>
</td>
</tr>
<tr id="row12722123915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147418168214"><a name="p1147418168214"></a><a name="p1147418168214"></a>257</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3479336182713"><a name="p3479336182713"></a><a name="p3479336182713"></a>_nnpack_spatial_convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18479133616276"><a name="p18479133616276"></a><a name="p18479133616276"></a>_nnpack_spatial_convolution_npu</p>
</td>
</tr>
<tr id="row187221739191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5475171617220"><a name="p5475171617220"></a><a name="p5475171617220"></a>258</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p164792364273"><a name="p164792364273"></a><a name="p164792364273"></a>ones.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4479936142716"><a name="p4479936142716"></a><a name="p4479936142716"></a>ones_npu</p>
</td>
</tr>
<tr id="row87224394120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647561618215"><a name="p647561618215"></a><a name="p647561618215"></a>259</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1647953662717"><a name="p1647953662717"></a><a name="p1647953662717"></a>ones</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1047923617271"><a name="p1047923617271"></a><a name="p1047923617271"></a>ones_npu</p>
</td>
</tr>
<tr id="row107221239181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174754169211"><a name="p174754169211"></a><a name="p174754169211"></a>260</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p647923652717"><a name="p647923652717"></a><a name="p647923652717"></a>ones.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8479143615277"><a name="p8479143615277"></a><a name="p8479143615277"></a>ones_out_npu</p>
</td>
</tr>
<tr id="row7722153916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54755160212"><a name="p54755160212"></a><a name="p54755160212"></a>261</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p104798365274"><a name="p104798365274"></a><a name="p104798365274"></a>ones_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6479123615273"><a name="p6479123615273"></a><a name="p6479123615273"></a>ones_like_npu</p>
</td>
</tr>
<tr id="row15723163901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047571615214"><a name="p1047571615214"></a><a name="p1047571615214"></a>262</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2480153672710"><a name="p2480153672710"></a><a name="p2480153672710"></a>cdist</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19480336142713"><a name="p19480336142713"></a><a name="p19480336142713"></a>cdist_npu</p>
</td>
</tr>
<tr id="row11723139141218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647513161229"><a name="p647513161229"></a><a name="p647513161229"></a>263</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p174801936112716"><a name="p174801936112716"></a><a name="p174801936112716"></a>_cdist_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1748018360274"><a name="p1748018360274"></a><a name="p1748018360274"></a>_cdist_forward_npu</p>
</td>
</tr>
<tr id="row18723183917129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9475111610212"><a name="p9475111610212"></a><a name="p9475111610212"></a>264</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1548073692712"><a name="p1548073692712"></a><a name="p1548073692712"></a>_cdist_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1748015367278"><a name="p1748015367278"></a><a name="p1748015367278"></a>_cdist_backward_npu</p>
</td>
</tr>
<tr id="row572323917129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9475616020"><a name="p9475616020"></a><a name="p9475616020"></a>265</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p048043642710"><a name="p048043642710"></a><a name="p048043642710"></a>pdist</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20480133617271"><a name="p20480133617271"></a><a name="p20480133617271"></a>pdist_npu</p>
</td>
</tr>
<tr id="row157236395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147517167217"><a name="p147517167217"></a><a name="p147517167217"></a>266</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8480103613277"><a name="p8480103613277"></a><a name="p8480103613277"></a>_pdist_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p248053614272"><a name="p248053614272"></a><a name="p248053614272"></a>_pdist_forward_npu</p>
</td>
</tr>
<tr id="row12723539181215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347531612211"><a name="p1347531612211"></a><a name="p1347531612211"></a>267</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1148011368278"><a name="p1148011368278"></a><a name="p1148011368278"></a>randperm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148083632718"><a name="p1148083632718"></a><a name="p1148083632718"></a>randperm_npu</p>
</td>
</tr>
<tr id="row972373971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194751160213"><a name="p194751160213"></a><a name="p194751160213"></a>268</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9480536152717"><a name="p9480536152717"></a><a name="p9480536152717"></a>randperm.generator</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0480183652720"><a name="p0480183652720"></a><a name="p0480183652720"></a>randperm_npu</p>
</td>
</tr>
<tr id="row87231839101216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10475016423"><a name="p10475016423"></a><a name="p10475016423"></a>269</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16480103672717"><a name="p16480103672717"></a><a name="p16480103672717"></a>randperm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7480123610275"><a name="p7480123610275"></a><a name="p7480123610275"></a>randperm_out_npu</p>
</td>
</tr>
<tr id="row157231339181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144755161829"><a name="p144755161829"></a><a name="p144755161829"></a>270</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19480133619278"><a name="p19480133619278"></a><a name="p19480133619278"></a>randperm.generator_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8480136122713"><a name="p8480136122713"></a><a name="p8480136122713"></a>randperm_out_npu</p>
</td>
</tr>
<tr id="row472310397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847514161323"><a name="p847514161323"></a><a name="p847514161323"></a>271</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1748053612276"><a name="p1748053612276"></a><a name="p1748053612276"></a>range.step</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1480133622717"><a name="p1480133622717"></a><a name="p1480133622717"></a>range_npu</p>
</td>
</tr>
<tr id="row372323991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647518161920"><a name="p1647518161920"></a><a name="p1647518161920"></a>272</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19480173612714"><a name="p19480173612714"></a><a name="p19480173612714"></a>range</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5480193616274"><a name="p5480193616274"></a><a name="p5480193616274"></a>range_npu</p>
</td>
</tr>
<tr id="row18724183911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04768167214"><a name="p04768167214"></a><a name="p04768167214"></a>273</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7480123632719"><a name="p7480123632719"></a><a name="p7480123632719"></a>range.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p104807368274"><a name="p104807368274"></a><a name="p104807368274"></a>range_out_npu</p>
</td>
</tr>
<tr id="row16725639181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p247614161022"><a name="p247614161022"></a><a name="p247614161022"></a>274</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p748063612271"><a name="p748063612271"></a><a name="p748063612271"></a>reciprocal</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1648033652714"><a name="p1648033652714"></a><a name="p1648033652714"></a>reciprocal_npu</p>
</td>
</tr>
<tr id="row87254399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p947613161127"><a name="p947613161127"></a><a name="p947613161127"></a>275</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24801036152714"><a name="p24801036152714"></a><a name="p24801036152714"></a>reciprocal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9480103672715"><a name="p9480103672715"></a><a name="p9480103672715"></a>reciprocal_npu_</p>
</td>
</tr>
<tr id="row3726173971212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147611161727"><a name="p147611161727"></a><a name="p147611161727"></a>276</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7480153632713"><a name="p7480153632713"></a><a name="p7480153632713"></a>reciprocal.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p204811236162716"><a name="p204811236162716"></a><a name="p204811236162716"></a>reciprocal_out_npu</p>
</td>
</tr>
<tr id="row5726103918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3476916627"><a name="p3476916627"></a><a name="p3476916627"></a>277</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p54811436112713"><a name="p54811436112713"></a><a name="p54811436112713"></a>neg</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1648193611272"><a name="p1648193611272"></a><a name="p1648193611272"></a>neg_npu</p>
</td>
</tr>
<tr id="row972617399127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24769161524"><a name="p24769161524"></a><a name="p24769161524"></a>278</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7481193611276"><a name="p7481193611276"></a><a name="p7481193611276"></a>neg_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2048173672719"><a name="p2048173672719"></a><a name="p2048173672719"></a>neg_npu_</p>
</td>
</tr>
<tr id="row1972693961213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16476151615220"><a name="p16476151615220"></a><a name="p16476151615220"></a>279</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5481183615275"><a name="p5481183615275"></a><a name="p5481183615275"></a>neg.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6481636152711"><a name="p6481636152711"></a><a name="p6481636152711"></a>neg_out_npu</p>
</td>
</tr>
<tr id="row8726133920129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447618168218"><a name="p447618168218"></a><a name="p447618168218"></a>280</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5481336132712"><a name="p5481336132712"></a><a name="p5481336132712"></a>repeat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5481123672720"><a name="p5481123672720"></a><a name="p5481123672720"></a>repeat_npu</p>
</td>
</tr>
<tr id="row12726193914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34765162029"><a name="p34765162029"></a><a name="p34765162029"></a>281</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6481193662716"><a name="p6481193662716"></a><a name="p6481193662716"></a>repeat_interleave.self_int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2481133642710"><a name="p2481133642710"></a><a name="p2481133642710"></a>repeat_interleave_npu</p>
</td>
</tr>
<tr id="row8726339101213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547614161929"><a name="p1547614161929"></a><a name="p1547614161929"></a>282</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7481136112714"><a name="p7481136112714"></a><a name="p7481136112714"></a>round</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148115364273"><a name="p1148115364273"></a><a name="p1148115364273"></a>round_npu</p>
</td>
</tr>
<tr id="row15726123913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144763161127"><a name="p144763161127"></a><a name="p144763161127"></a>283</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p748163618278"><a name="p748163618278"></a><a name="p748163618278"></a>round_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1448114369273"><a name="p1448114369273"></a><a name="p1448114369273"></a>round_npu_</p>
</td>
</tr>
<tr id="row1472614394127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147631614219"><a name="p147631614219"></a><a name="p147631614219"></a>284</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p848143619279"><a name="p848143619279"></a><a name="p848143619279"></a>round.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p248118364276"><a name="p248118364276"></a><a name="p248118364276"></a>round_out_npu</p>
</td>
</tr>
<tr id="row1072663911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847610167218"><a name="p847610167218"></a><a name="p847610167218"></a>285</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p648112369276"><a name="p648112369276"></a><a name="p648112369276"></a>relu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5481133614279"><a name="p5481133614279"></a><a name="p5481133614279"></a>relu_npu</p>
</td>
</tr>
<tr id="row16727239141217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3476191615213"><a name="p3476191615213"></a><a name="p3476191615213"></a>286</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1348119363273"><a name="p1348119363273"></a><a name="p1348119363273"></a>relu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9481136102715"><a name="p9481136102715"></a><a name="p9481136102715"></a>relu_npu_</p>
</td>
</tr>
<tr id="row1572710397121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1476516527"><a name="p1476516527"></a><a name="p1476516527"></a>287</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p54811536162711"><a name="p54811536162711"></a><a name="p54811536162711"></a>prelu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16481203614279"><a name="p16481203614279"></a><a name="p16481203614279"></a>prelu_npu</p>
</td>
</tr>
<tr id="row572713392127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04761516824"><a name="p04761516824"></a><a name="p04761516824"></a>288</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p748103613279"><a name="p748103613279"></a><a name="p748103613279"></a>prelu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1648110362278"><a name="p1648110362278"></a><a name="p1648110362278"></a>prelu_backward_npu</p>
</td>
</tr>
<tr id="row17727639181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184761916926"><a name="p184761916926"></a><a name="p184761916926"></a>289</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p748113617276"><a name="p748113617276"></a><a name="p748113617276"></a>gelu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p048173616271"><a name="p048173616271"></a><a name="p048173616271"></a>gelu_npu</p>
</td>
</tr>
<tr id="row1872783910125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114769161927"><a name="p114769161927"></a><a name="p114769161927"></a>290</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18481193642712"><a name="p18481193642712"></a><a name="p18481193642712"></a>gelu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1448112367270"><a name="p1448112367270"></a><a name="p1448112367270"></a>gelu_backward_npu</p>
</td>
</tr>
<tr id="row272718396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947611620210"><a name="p1947611620210"></a><a name="p1947611620210"></a>291</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18481436122717"><a name="p18481436122717"></a><a name="p18481436122717"></a>hardshrink</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44815369275"><a name="p44815369275"></a><a name="p44815369275"></a>hardshrink_npu</p>
</td>
</tr>
<tr id="row1727163912122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84771716627"><a name="p84771716627"></a><a name="p84771716627"></a>292</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13482836132718"><a name="p13482836132718"></a><a name="p13482836132718"></a>hardshrink_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1648233662717"><a name="p1648233662717"></a><a name="p1648233662717"></a>hardshrink_backward_npu</p>
</td>
</tr>
<tr id="row9727133901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104774161721"><a name="p104774161721"></a><a name="p104774161721"></a>293</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10482236122718"><a name="p10482236122718"></a><a name="p10482236122718"></a>rsqrt</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24821036192716"><a name="p24821036192716"></a><a name="p24821036192716"></a>rsqrt_npu</p>
</td>
</tr>
<tr id="row17727103918124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124775161217"><a name="p124775161217"></a><a name="p124775161217"></a>294</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p048283615279"><a name="p048283615279"></a><a name="p048283615279"></a>rsqrt_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14482436152710"><a name="p14482436152710"></a><a name="p14482436152710"></a>rsqrt_npu_</p>
</td>
</tr>
<tr id="row572773917122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84773161021"><a name="p84773161021"></a><a name="p84773161021"></a>295</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9482336172719"><a name="p9482336172719"></a><a name="p9482336172719"></a>rsqrt.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94824360275"><a name="p94824360275"></a><a name="p94824360275"></a>rsqrt_out_npu</p>
</td>
</tr>
<tr id="row1172723911216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147711161526"><a name="p147711161526"></a><a name="p147711161526"></a>296</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p448210364274"><a name="p448210364274"></a><a name="p448210364274"></a>selu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p748219365271"><a name="p748219365271"></a><a name="p748219365271"></a>selu_npu</p>
</td>
</tr>
<tr id="row1672763961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14772161226"><a name="p14772161226"></a><a name="p14772161226"></a>297</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1348253610271"><a name="p1348253610271"></a><a name="p1348253610271"></a>selu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24821365279"><a name="p24821365279"></a><a name="p24821365279"></a>selu_npu_</p>
</td>
</tr>
<tr id="row87284399129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947711161521"><a name="p1947711161521"></a><a name="p1947711161521"></a>298</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6482143619270"><a name="p6482143619270"></a><a name="p6482143619270"></a>celu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p184829366271"><a name="p184829366271"></a><a name="p184829366271"></a>celu_npu</p>
</td>
</tr>
<tr id="row20728839161216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647771615217"><a name="p647771615217"></a><a name="p647771615217"></a>299</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9483936202717"><a name="p9483936202717"></a><a name="p9483936202717"></a>celu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5483143611275"><a name="p5483143611275"></a><a name="p5483143611275"></a>celu_npu_</p>
</td>
</tr>
<tr id="row1072893910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047716161929"><a name="p2047716161929"></a><a name="p2047716161929"></a>300</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1848313622720"><a name="p1848313622720"></a><a name="p1848313622720"></a>sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1648317369274"><a name="p1648317369274"></a><a name="p1648317369274"></a>sigmoid_npu</p>
</td>
</tr>
<tr id="row1672863915127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184772161020"><a name="p184772161020"></a><a name="p184772161020"></a>301</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1748333614277"><a name="p1748333614277"></a><a name="p1748333614277"></a>sigmoid_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17483136162719"><a name="p17483136162719"></a><a name="p17483136162719"></a>sigmoid_npu_</p>
</td>
</tr>
<tr id="row5728739101210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54773168211"><a name="p54773168211"></a><a name="p54773168211"></a>302</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14483536102710"><a name="p14483536102710"></a><a name="p14483536102710"></a>sigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124831236102718"><a name="p124831236102718"></a><a name="p124831236102718"></a>sigmoid_out_npu</p>
</td>
</tr>
<tr id="row15728153941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3477316220"><a name="p3477316220"></a><a name="p3477316220"></a>303</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1548393615273"><a name="p1548393615273"></a><a name="p1548393615273"></a>sin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1448313682714"><a name="p1448313682714"></a><a name="p1448313682714"></a>sin_npu</p>
</td>
</tr>
<tr id="row15728239111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19477181614214"><a name="p19477181614214"></a><a name="p19477181614214"></a>304</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16483163622719"><a name="p16483163622719"></a><a name="p16483163622719"></a>sin_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p04835362273"><a name="p04835362273"></a><a name="p04835362273"></a>sin_npu_</p>
</td>
</tr>
<tr id="row67282039151218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15477151614219"><a name="p15477151614219"></a><a name="p15477151614219"></a>305</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14483836182716"><a name="p14483836182716"></a><a name="p14483836182716"></a>sin.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1848314361279"><a name="p1848314361279"></a><a name="p1848314361279"></a>sin_out_npu</p>
</td>
</tr>
<tr id="row172818396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347814161129"><a name="p1347814161129"></a><a name="p1347814161129"></a>306</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p134831636112711"><a name="p134831636112711"></a><a name="p134831636112711"></a>sinh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24831636102716"><a name="p24831636102716"></a><a name="p24831636102716"></a>sinh_npu</p>
</td>
</tr>
<tr id="row2728039181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134785167218"><a name="p134785167218"></a><a name="p134785167218"></a>307</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11483153612273"><a name="p11483153612273"></a><a name="p11483153612273"></a>sinh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6483173672720"><a name="p6483173672720"></a><a name="p6483173672720"></a>sinh_npu_</p>
</td>
</tr>
<tr id="row77287391121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647851613215"><a name="p647851613215"></a><a name="p647851613215"></a>308</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p848393613277"><a name="p848393613277"></a><a name="p848393613277"></a>sinh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p348363672715"><a name="p348363672715"></a><a name="p348363672715"></a>sinh_out_npu</p>
</td>
</tr>
<tr id="row772923912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184787161725"><a name="p184787161725"></a><a name="p184787161725"></a>309</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17483183692716"><a name="p17483183692716"></a><a name="p17483183692716"></a>slogdet</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1348319369278"><a name="p1348319369278"></a><a name="p1348319369278"></a>slogdet_npu</p>
</td>
</tr>
<tr id="row1872912397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p047819161626"><a name="p047819161626"></a><a name="p047819161626"></a>310</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1348323616273"><a name="p1348323616273"></a><a name="p1348323616273"></a>softmax.int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10483193632713"><a name="p10483193632713"></a><a name="p10483193632713"></a>softmax_npu</p>
</td>
</tr>
<tr id="row772923941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747820160210"><a name="p747820160210"></a><a name="p747820160210"></a>311</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18484936112715"><a name="p18484936112715"></a><a name="p18484936112715"></a>softmax.Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p748415361273"><a name="p748415361273"></a><a name="p748415361273"></a>softmax_npu</p>
</td>
</tr>
<tr id="row7729143919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174781416028"><a name="p174781416028"></a><a name="p174781416028"></a>312</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p148415366276"><a name="p148415366276"></a><a name="p148415366276"></a>_softmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1648414360275"><a name="p1648414360275"></a><a name="p1648414360275"></a>_softmax_npu</p>
</td>
</tr>
<tr id="row972917397121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047811169220"><a name="p1047811169220"></a><a name="p1047811169220"></a>313</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5484636192710"><a name="p5484636192710"></a><a name="p5484636192710"></a>_softmax_backward_data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2048413652714"><a name="p2048413652714"></a><a name="p2048413652714"></a>_softmax_backward_npu</p>
</td>
</tr>
<tr id="row1172903920127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447819161020"><a name="p1447819161020"></a><a name="p1447819161020"></a>314</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11484133613279"><a name="p11484133613279"></a><a name="p11484133613279"></a>stack</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84845369278"><a name="p84845369278"></a><a name="p84845369278"></a>stack_npu</p>
</td>
</tr>
<tr id="row12729133920124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2478191614211"><a name="p2478191614211"></a><a name="p2478191614211"></a>315</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11484183610279"><a name="p11484183610279"></a><a name="p11484183610279"></a>stack.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144841636162719"><a name="p144841636162719"></a><a name="p144841636162719"></a>stack_out_npu</p>
</td>
</tr>
<tr id="row6729203951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847811620215"><a name="p847811620215"></a><a name="p847811620215"></a>316</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p94841362274"><a name="p94841362274"></a><a name="p94841362274"></a>sum</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p104841136152710"><a name="p104841136152710"></a><a name="p104841136152710"></a>sum_npu</p>
</td>
</tr>
<tr id="row18729153951219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154781165218"><a name="p154781165218"></a><a name="p154781165218"></a>317</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p248473611276"><a name="p248473611276"></a><a name="p248473611276"></a>sum.dim_IntList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5484153612710"><a name="p5484153612710"></a><a name="p5484153612710"></a>sum_npu</p>
</td>
</tr>
<tr id="row137291539131215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7478121619218"><a name="p7478121619218"></a><a name="p7478121619218"></a>318</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1548443632714"><a name="p1548443632714"></a><a name="p1548443632714"></a>sum.dim_DimnameList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p748473662719"><a name="p748473662719"></a><a name="p748473662719"></a>sum_npu</p>
</td>
</tr>
<tr id="row207291839121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447871616213"><a name="p1447871616213"></a><a name="p1447871616213"></a>319</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13484536182711"><a name="p13484536182711"></a><a name="p13484536182711"></a>sum.IntList_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p154846364271"><a name="p154846364271"></a><a name="p154846364271"></a>sum_out_npu</p>
</td>
</tr>
<tr id="row77291139121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147818165213"><a name="p1147818165213"></a><a name="p1147818165213"></a>320</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1848413662711"><a name="p1848413662711"></a><a name="p1848413662711"></a>sum.DimnameList_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194845367270"><a name="p194845367270"></a><a name="p194845367270"></a>sum_out_npu</p>
</td>
</tr>
<tr id="row137301239111219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447821618211"><a name="p447821618211"></a><a name="p447821618211"></a>321</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p148413616275"><a name="p148413616275"></a><a name="p148413616275"></a>sqrt</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148453618278"><a name="p1148453618278"></a><a name="p1148453618278"></a>sqrt_npu</p>
</td>
</tr>
<tr id="row187301139101214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194786167215"><a name="p194786167215"></a><a name="p194786167215"></a>322</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9484143610278"><a name="p9484143610278"></a><a name="p9484143610278"></a>sqrt_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1248483652715"><a name="p1248483652715"></a><a name="p1248483652715"></a>sqrt_npu_</p>
</td>
</tr>
<tr id="row1173053981217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1847931611220"><a name="p1847931611220"></a><a name="p1847931611220"></a>323</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19484103613273"><a name="p19484103613273"></a><a name="p19484103613273"></a>sqrt.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84851936122714"><a name="p84851936122714"></a><a name="p84851936122714"></a>sqrt_out_npu</p>
</td>
</tr>
<tr id="row20730123981213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947910165210"><a name="p1947910165210"></a><a name="p1947910165210"></a>324</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14485183615279"><a name="p14485183615279"></a><a name="p14485183615279"></a>std</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7485163642711"><a name="p7485163642711"></a><a name="p7485163642711"></a>std_npu</p>
</td>
</tr>
<tr id="row773013920126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34791616524"><a name="p34791616524"></a><a name="p34791616524"></a>325</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1148543672716"><a name="p1148543672716"></a><a name="p1148543672716"></a>std.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84858361273"><a name="p84858361273"></a><a name="p84858361273"></a>std_dim_npu</p>
</td>
</tr>
<tr id="row1973018398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647910167219"><a name="p647910167219"></a><a name="p647910167219"></a>326</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p548593642719"><a name="p548593642719"></a><a name="p548593642719"></a>std_mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p448593619271"><a name="p448593619271"></a><a name="p448593619271"></a>std_mean_npu</p>
</td>
</tr>
<tr id="row12730153913122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4479161613216"><a name="p4479161613216"></a><a name="p4479161613216"></a>327</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1448573662712"><a name="p1448573662712"></a><a name="p1448573662712"></a>std_mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148613369275"><a name="p1148613369275"></a><a name="p1148613369275"></a>std_mean_dim_npu</p>
</td>
</tr>
<tr id="row47301639101218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047911613210"><a name="p2047911613210"></a><a name="p2047911613210"></a>328</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p648603622715"><a name="p648603622715"></a><a name="p648603622715"></a>std_mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1048633619275"><a name="p1048633619275"></a><a name="p1048633619275"></a>std_mean_names_npu</p>
</td>
</tr>
<tr id="row1873012396124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134791816721"><a name="p134791816721"></a><a name="p134791816721"></a>329</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0486936192719"><a name="p0486936192719"></a><a name="p0486936192719"></a>std.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174861636192720"><a name="p174861636192720"></a><a name="p174861636192720"></a>std_out_npu</p>
</td>
</tr>
<tr id="row773043919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10479141613216"><a name="p10479141613216"></a><a name="p10479141613216"></a>330</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2486736152710"><a name="p2486736152710"></a><a name="p2486736152710"></a>std.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174866362277"><a name="p174866362277"></a><a name="p174866362277"></a>std_names_npu</p>
</td>
</tr>
<tr id="row1473013911211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164795165212"><a name="p164795165212"></a><a name="p164795165212"></a>331</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1748663611278"><a name="p1748663611278"></a><a name="p1748663611278"></a>std.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34861036142715"><a name="p34861036142715"></a><a name="p34861036142715"></a>std_out_npu</p>
</td>
</tr>
<tr id="row1173173951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124795161215"><a name="p124795161215"></a><a name="p124795161215"></a>332</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3486436142712"><a name="p3486436142712"></a><a name="p3486436142712"></a>prod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3486123612275"><a name="p3486123612275"></a><a name="p3486123612275"></a>prod_npu</p>
</td>
</tr>
<tr id="row47311839111211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194792161028"><a name="p194792161028"></a><a name="p194792161028"></a>333</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p164868362279"><a name="p164868362279"></a><a name="p164868362279"></a>prod.dim_int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4486163616274"><a name="p4486163616274"></a><a name="p4486163616274"></a>prod_npu</p>
</td>
</tr>
<tr id="row7731439171216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447941618212"><a name="p447941618212"></a><a name="p447941618212"></a>334</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1748643614275"><a name="p1748643614275"></a><a name="p1748643614275"></a>prod.int_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114867368272"><a name="p114867368272"></a><a name="p114867368272"></a>prod_out_npu</p>
</td>
</tr>
<tr id="row47311439191213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8479101619218"><a name="p8479101619218"></a><a name="p8479101619218"></a>335</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p548693692713"><a name="p548693692713"></a><a name="p548693692713"></a>prod.dim_Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1448723617279"><a name="p1448723617279"></a><a name="p1448723617279"></a>prod_npu</p>
</td>
</tr>
<tr id="row1373193911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447917167215"><a name="p447917167215"></a><a name="p447917167215"></a>336</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1487143682714"><a name="p1487143682714"></a><a name="p1487143682714"></a>prod.Dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1948733662716"><a name="p1948733662716"></a><a name="p1948733662716"></a>prod_out_npu</p>
</td>
</tr>
<tr id="row47315396122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19479171620216"><a name="p19479171620216"></a><a name="p19479171620216"></a>337</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p448711364271"><a name="p448711364271"></a><a name="p448711364271"></a>tan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134878361277"><a name="p134878361277"></a><a name="p134878361277"></a>tan_npu</p>
</td>
</tr>
<tr id="row27311139111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124798161629"><a name="p124798161629"></a><a name="p124798161629"></a>338</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84872363278"><a name="p84872363278"></a><a name="p84872363278"></a>tan_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94871436162716"><a name="p94871436162716"></a><a name="p94871436162716"></a>tan_npu_</p>
</td>
</tr>
<tr id="row27312391123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147919169217"><a name="p1147919169217"></a><a name="p1147919169217"></a>339</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p148753682714"><a name="p148753682714"></a><a name="p148753682714"></a>tan.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1748711365274"><a name="p1748711365274"></a><a name="p1748711365274"></a>tan_out_npu</p>
</td>
</tr>
<tr id="row187311539101213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548014165210"><a name="p1548014165210"></a><a name="p1548014165210"></a>340</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p748763632719"><a name="p748763632719"></a><a name="p748763632719"></a>tanh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24871836182720"><a name="p24871836182720"></a><a name="p24871836182720"></a>tanh_npu</p>
</td>
</tr>
<tr id="row187313392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64802161029"><a name="p64802161029"></a><a name="p64802161029"></a>341</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8487203618274"><a name="p8487203618274"></a><a name="p8487203618274"></a>tanh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p148763602714"><a name="p148763602714"></a><a name="p148763602714"></a>tanh_npu_</p>
</td>
</tr>
<tr id="row1873183981220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548081614214"><a name="p548081614214"></a><a name="p548081614214"></a>342</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p748733672713"><a name="p748733672713"></a><a name="p748733672713"></a>tanh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p948783652717"><a name="p948783652717"></a><a name="p948783652717"></a>tanh_out_npu</p>
</td>
</tr>
<tr id="row12731103951212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948081611215"><a name="p1948081611215"></a><a name="p1948081611215"></a>343</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7487113615272"><a name="p7487113615272"></a><a name="p7487113615272"></a>threshold</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7487636142712"><a name="p7487636142712"></a><a name="p7487636142712"></a>threshold_npu</p>
</td>
</tr>
<tr id="row16732139111215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648091614213"><a name="p1648091614213"></a><a name="p1648091614213"></a>344</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p748711369278"><a name="p748711369278"></a><a name="p748711369278"></a>threshold_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1248793672711"><a name="p1248793672711"></a><a name="p1248793672711"></a>threshold_npu_</p>
</td>
</tr>
<tr id="row6732113961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20480151612213"><a name="p20480151612213"></a><a name="p20480151612213"></a>345</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1148713613279"><a name="p1148713613279"></a><a name="p1148713613279"></a>threshold.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1748723682716"><a name="p1748723682716"></a><a name="p1748723682716"></a>threshold_out_npu</p>
</td>
</tr>
<tr id="row1473273916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04802161927"><a name="p04802161927"></a><a name="p04802161927"></a>346</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1748714369277"><a name="p1748714369277"></a><a name="p1748714369277"></a>threshold_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p148713612716"><a name="p148713612716"></a><a name="p148713612716"></a>threshold_backward_npu</p>
</td>
</tr>
<tr id="row673213910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8480151619212"><a name="p8480151619212"></a><a name="p8480151619212"></a>347</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20487103682714"><a name="p20487103682714"></a><a name="p20487103682714"></a>one_hot</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p748713367275"><a name="p748713367275"></a><a name="p748713367275"></a>one_hot_npu1</p>
</td>
</tr>
<tr id="row873263916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648015161520"><a name="p648015161520"></a><a name="p648015161520"></a>348</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p44874368273"><a name="p44874368273"></a><a name="p44874368273"></a>flip</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1748716364277"><a name="p1748716364277"></a><a name="p1748716364277"></a>flip_npu</p>
</td>
</tr>
<tr id="row07321739181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148016167216"><a name="p1148016167216"></a><a name="p1148016167216"></a>349</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34871436162719"><a name="p34871436162719"></a><a name="p34871436162719"></a>roll</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24881236192719"><a name="p24881236192719"></a><a name="p24881236192719"></a>roll_npu</p>
</td>
</tr>
<tr id="row12732163911214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8480516328"><a name="p8480516328"></a><a name="p8480516328"></a>350</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p848819363270"><a name="p848819363270"></a><a name="p848819363270"></a>true_divide.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34886363274"><a name="p34886363274"></a><a name="p34886363274"></a>true_divide_npu</p>
</td>
</tr>
<tr id="row117321397120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144801316221"><a name="p144801316221"></a><a name="p144801316221"></a>351</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1048815363278"><a name="p1048815363278"></a><a name="p1048815363278"></a>true_divide_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194881736162715"><a name="p194881736162715"></a><a name="p194881736162715"></a>true_divide_npu_</p>
</td>
</tr>
<tr id="row873218398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194801316925"><a name="p194801316925"></a><a name="p194801316925"></a>352</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p848818367273"><a name="p848818367273"></a><a name="p848818367273"></a>true_divide.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1048817362278"><a name="p1048817362278"></a><a name="p1048817362278"></a>true_divide_out_npu</p>
</td>
</tr>
<tr id="row1173243919126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548041618219"><a name="p548041618219"></a><a name="p548041618219"></a>353</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12488133642719"><a name="p12488133642719"></a><a name="p12488133642719"></a>true_divide.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p184881836112710"><a name="p184881836112710"></a><a name="p184881836112710"></a>true_divide_npu</p>
</td>
</tr>
<tr id="row1673219391128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24817161215"><a name="p24817161215"></a><a name="p24817161215"></a>354</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1048817363274"><a name="p1048817363274"></a><a name="p1048817363274"></a>true_divide_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5488203616275"><a name="p5488203616275"></a><a name="p5488203616275"></a>true_divide_npu_</p>
</td>
</tr>
<tr id="row117331397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5481111615216"><a name="p5481111615216"></a><a name="p5481111615216"></a>355</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84881536132720"><a name="p84881536132720"></a><a name="p84881536132720"></a>trunc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p248833617273"><a name="p248833617273"></a><a name="p248833617273"></a>trunc_npu</p>
</td>
</tr>
<tr id="row6733239131216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18481516224"><a name="p18481516224"></a><a name="p18481516224"></a>356</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p748818363275"><a name="p748818363275"></a><a name="p748818363275"></a>trunc_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1548863632714"><a name="p1548863632714"></a><a name="p1548863632714"></a>trunc_npu_</p>
</td>
</tr>
<tr id="row8733113981214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10481716720"><a name="p10481716720"></a><a name="p10481716720"></a>357</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1048883619279"><a name="p1048883619279"></a><a name="p1048883619279"></a>trunc.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p348873618271"><a name="p348873618271"></a><a name="p348873618271"></a>trunc_out_npu</p>
</td>
</tr>
<tr id="row107332039151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64817161211"><a name="p64817161211"></a><a name="p64817161211"></a>358</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184881236162711"><a name="p184881236162711"></a><a name="p184881236162711"></a>_unique2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16488173652711"><a name="p16488173652711"></a><a name="p16488173652711"></a>_unique2_npu</p>
</td>
</tr>
<tr id="row13733339161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64811616724"><a name="p64811616724"></a><a name="p64811616724"></a>359</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1448810369276"><a name="p1448810369276"></a><a name="p1448810369276"></a>var</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144880362276"><a name="p144880362276"></a><a name="p144880362276"></a>var_npu</p>
</td>
</tr>
<tr id="row3733153931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3481101611218"><a name="p3481101611218"></a><a name="p3481101611218"></a>360</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p548823682711"><a name="p548823682711"></a><a name="p548823682711"></a>var.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12488113613274"><a name="p12488113613274"></a><a name="p12488113613274"></a>var_npu</p>
</td>
</tr>
<tr id="row10733139111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5481111613213"><a name="p5481111613213"></a><a name="p5481111613213"></a>361</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1348813363271"><a name="p1348813363271"></a><a name="p1348813363271"></a>var.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p64887367278"><a name="p64887367278"></a><a name="p64887367278"></a>var_out_npu</p>
</td>
</tr>
<tr id="row7733739101220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174811716224"><a name="p174811716224"></a><a name="p174811716224"></a>362</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16488103615272"><a name="p16488103615272"></a><a name="p16488103615272"></a>var.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2488136162710"><a name="p2488136162710"></a><a name="p2488136162710"></a>var_npu</p>
</td>
</tr>
<tr id="row8733183911217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44811516922"><a name="p44811516922"></a><a name="p44811516922"></a>363</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10488136182715"><a name="p10488136182715"></a><a name="p10488136182715"></a>var.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p74881436122720"><a name="p74881436122720"></a><a name="p74881436122720"></a>var_out_npu</p>
</td>
</tr>
<tr id="row1733193961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1248141617211"><a name="p1248141617211"></a><a name="p1248141617211"></a>364</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14489163682719"><a name="p14489163682719"></a><a name="p14489163682719"></a>var_mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12489336162712"><a name="p12489336162712"></a><a name="p12489336162712"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row157331039111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104817163214"><a name="p104817163214"></a><a name="p104817163214"></a>365</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1648911369277"><a name="p1648911369277"></a><a name="p1648911369277"></a>var_mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p748943662714"><a name="p748943662714"></a><a name="p748943662714"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row1573414397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144812161020"><a name="p144812161020"></a><a name="p144812161020"></a>366</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p548973642716"><a name="p548973642716"></a><a name="p548973642716"></a>var_mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p848963602712"><a name="p848963602712"></a><a name="p848963602712"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row11734153931213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1481131615216"><a name="p1481131615216"></a><a name="p1481131615216"></a>367</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p148914368275"><a name="p148914368275"></a><a name="p148914368275"></a>where.self</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13489836112717"><a name="p13489836112717"></a><a name="p13489836112717"></a>where_npu</p>
</td>
</tr>
<tr id="row173473941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174814160210"><a name="p174814160210"></a><a name="p174814160210"></a>368</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2489173615277"><a name="p2489173615277"></a><a name="p2489173615277"></a>where</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94893367276"><a name="p94893367276"></a><a name="p94893367276"></a>where_npu</p>
</td>
</tr>
<tr id="row137341239201216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13481116221"><a name="p13481116221"></a><a name="p13481116221"></a>369</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184891436152710"><a name="p184891436152710"></a><a name="p184891436152710"></a>_s_where</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11489436172711"><a name="p11489436172711"></a><a name="p11489436172711"></a>_s_where_npu</p>
</td>
</tr>
<tr id="row19734113931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848161618210"><a name="p848161618210"></a><a name="p848161618210"></a>370</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4489133611277"><a name="p4489133611277"></a><a name="p4489133611277"></a>zeros.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5489183619276"><a name="p5489183619276"></a><a name="p5489183619276"></a>zeros_npu</p>
</td>
</tr>
<tr id="row77341439121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16481516222"><a name="p16481516222"></a><a name="p16481516222"></a>371</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11489836162719"><a name="p11489836162719"></a><a name="p11489836162719"></a>zeros</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p448918369277"><a name="p448918369277"></a><a name="p448918369277"></a>zeros_npu</p>
</td>
</tr>
<tr id="row147341939121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1448118161522"><a name="p1448118161522"></a><a name="p1448118161522"></a>372</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14489113642713"><a name="p14489113642713"></a><a name="p14489113642713"></a>zeros.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p848914363273"><a name="p848914363273"></a><a name="p848914363273"></a>zeros_out_npu</p>
</td>
</tr>
<tr id="row4734439181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34811016329"><a name="p34811016329"></a><a name="p34811016329"></a>373</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8489133672715"><a name="p8489133672715"></a><a name="p8489133672715"></a>zeros_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p164899361272"><a name="p164899361272"></a><a name="p164899361272"></a>zeros_like_npu</p>
</td>
</tr>
<tr id="row47341839121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164821516622"><a name="p164821516622"></a><a name="p164821516622"></a>374</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p348943620274"><a name="p348943620274"></a><a name="p348943620274"></a>norm.ScalarOpt_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9489103662717"><a name="p9489103662717"></a><a name="p9489103662717"></a>norm_npu</p>
</td>
</tr>
<tr id="row2734139131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448291619216"><a name="p448291619216"></a><a name="p448291619216"></a>375</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10489113642715"><a name="p10489113642715"></a><a name="p10489113642715"></a>norm.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24891536192718"><a name="p24891536192718"></a><a name="p24891536192718"></a>norm_npu</p>
</td>
</tr>
<tr id="row10734153910128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164824161121"><a name="p164824161121"></a><a name="p164824161121"></a>376</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114896365272"><a name="p114896365272"></a><a name="p114896365272"></a>norm.ScalarOpt_dim_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148993612719"><a name="p1148993612719"></a><a name="p1148993612719"></a>norm_npu</p>
</td>
</tr>
<tr id="row47351039151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12482201610215"><a name="p12482201610215"></a><a name="p12482201610215"></a>377</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1648943611276"><a name="p1648943611276"></a><a name="p1648943611276"></a>norm.ScalarOpt_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124895369278"><a name="p124895369278"></a><a name="p124895369278"></a>norm_npu</p>
</td>
</tr>
<tr id="row13735339131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6482116629"><a name="p6482116629"></a><a name="p6482116629"></a>378</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6489203619275"><a name="p6489203619275"></a><a name="p6489203619275"></a>norm.dtype_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p148963642717"><a name="p148963642717"></a><a name="p148963642717"></a>norm_out_npu</p>
</td>
</tr>
<tr id="row167351939191219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948217161428"><a name="p1948217161428"></a><a name="p1948217161428"></a>379</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18489193616272"><a name="p18489193616272"></a><a name="p18489193616272"></a>norm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1648963682717"><a name="p1648963682717"></a><a name="p1648963682717"></a>norm_out_npu</p>
</td>
</tr>
<tr id="row7735113961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548217161024"><a name="p548217161024"></a><a name="p548217161024"></a>380</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6490133682718"><a name="p6490133682718"></a><a name="p6490133682718"></a>clone</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0490236172714"><a name="p0490236172714"></a><a name="p0490236172714"></a>clone_npu</p>
</td>
</tr>
<tr id="row373503911216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194825167217"><a name="p194825167217"></a><a name="p194825167217"></a>381</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16490103617275"><a name="p16490103617275"></a><a name="p16490103617275"></a>resize_as_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14901336202711"><a name="p14901336202711"></a><a name="p14901336202711"></a>resize_as_npu_</p>
</td>
</tr>
<tr id="row16735639151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154827161425"><a name="p154827161425"></a><a name="p154827161425"></a>382</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194906369278"><a name="p194906369278"></a><a name="p194906369278"></a>pow.Tensor_Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4490153672717"><a name="p4490153672717"></a><a name="p4490153672717"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row173518397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948291615214"><a name="p948291615214"></a><a name="p948291615214"></a>383</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1949093662716"><a name="p1949093662716"></a><a name="p1949093662716"></a>pow.Tensor_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3490123619279"><a name="p3490123619279"></a><a name="p3490123619279"></a>pow_npu</p>
</td>
</tr>
<tr id="row173512397127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194821016020"><a name="p194821016020"></a><a name="p194821016020"></a>384</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p949063692713"><a name="p949063692713"></a><a name="p949063692713"></a>zero_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1449011367276"><a name="p1449011367276"></a><a name="p1449011367276"></a>zero_npu_</p>
</td>
</tr>
<tr id="row117358394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348216161828"><a name="p1348216161828"></a><a name="p1348216161828"></a>385</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1249011361278"><a name="p1249011361278"></a><a name="p1249011361278"></a>sub.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18490143612270"><a name="p18490143612270"></a><a name="p18490143612270"></a>sub_out_npu</p>
</td>
</tr>
<tr id="row7735113910125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748214161326"><a name="p1748214161326"></a><a name="p1748214161326"></a>386</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p549053672718"><a name="p549053672718"></a><a name="p549053672718"></a>sub.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9490143619279"><a name="p9490143619279"></a><a name="p9490143619279"></a>sub_npu</p>
</td>
</tr>
<tr id="row20735123919121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948251617211"><a name="p1948251617211"></a><a name="p1948251617211"></a>387</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8490436112713"><a name="p8490436112713"></a><a name="p8490436112713"></a>sub_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p149016363273"><a name="p149016363273"></a><a name="p149016363273"></a>sub_npu_</p>
</td>
</tr>
<tr id="row4735103911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p748315169219"><a name="p748315169219"></a><a name="p748315169219"></a>388</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p449043612719"><a name="p449043612719"></a><a name="p449043612719"></a>sub.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p249083610277"><a name="p249083610277"></a><a name="p249083610277"></a>sub_npu</p>
</td>
</tr>
<tr id="row137367398126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24831016128"><a name="p24831016128"></a><a name="p24831016128"></a>389</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10490193612710"><a name="p10490193612710"></a><a name="p10490193612710"></a>sub_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1049003652711"><a name="p1049003652711"></a><a name="p1049003652711"></a>sub_npu_</p>
</td>
</tr>
<tr id="row873693961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15483141619216"><a name="p15483141619216"></a><a name="p15483141619216"></a>390</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16490173620274"><a name="p16490173620274"></a><a name="p16490173620274"></a>rsub.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2049013369275"><a name="p2049013369275"></a><a name="p2049013369275"></a>rsub_npu</p>
</td>
</tr>
<tr id="row1573610394128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2048301616219"><a name="p2048301616219"></a><a name="p2048301616219"></a>391</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15490123632718"><a name="p15490123632718"></a><a name="p15490123632718"></a>rsub.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2490163612719"><a name="p2490163612719"></a><a name="p2490163612719"></a>rsub_npu</p>
</td>
</tr>
<tr id="row17736103910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24836162020"><a name="p24836162020"></a><a name="p24836162020"></a>392</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p849073616277"><a name="p849073616277"></a><a name="p849073616277"></a>addmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10490103632712"><a name="p10490103632712"></a><a name="p10490103632712"></a>addmm_out_npu</p>
</td>
</tr>
<tr id="row137361039171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74832162212"><a name="p74832162212"></a><a name="p74832162212"></a>393</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p549019368275"><a name="p549019368275"></a><a name="p549019368275"></a>addmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84907365275"><a name="p84907365275"></a><a name="p84907365275"></a>addmm_npu</p>
</td>
</tr>
<tr id="row973673921220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174831161928"><a name="p174831161928"></a><a name="p174831161928"></a>394</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5490133682720"><a name="p5490133682720"></a><a name="p5490133682720"></a>addmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13490153614274"><a name="p13490153614274"></a><a name="p13490153614274"></a>addmm_npu_</p>
</td>
</tr>
<tr id="row2736539161210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124833164215"><a name="p124833164215"></a><a name="p124833164215"></a>395</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p949016366274"><a name="p949016366274"></a><a name="p949016366274"></a>quantize_per_tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p949012368272"><a name="p949012368272"></a><a name="p949012368272"></a>quantize_per_tensor_npu</p>
</td>
</tr>
<tr id="row1736139121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54837161825"><a name="p54837161825"></a><a name="p54837161825"></a>396</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9490183642713"><a name="p9490183642713"></a><a name="p9490183642713"></a>quantize_per_channel</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p849163617276"><a name="p849163617276"></a><a name="p849163617276"></a>quantize_per_channel_npu</p>
</td>
</tr>
<tr id="row373683917121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948311161028"><a name="p948311161028"></a><a name="p948311161028"></a>397</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1749133611271"><a name="p1749133611271"></a><a name="p1749133611271"></a>to.dtype_layout</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1849116363272"><a name="p1849116363272"></a><a name="p1849116363272"></a>to_npu</p>
</td>
</tr>
<tr id="row1773611392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048319161218"><a name="p1048319161218"></a><a name="p1048319161218"></a>398</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1949118362277"><a name="p1949118362277"></a><a name="p1949118362277"></a>to.device</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p549193613278"><a name="p549193613278"></a><a name="p549193613278"></a>to_device_npu</p>
</td>
</tr>
<tr id="row2736539161217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124831416728"><a name="p124831416728"></a><a name="p124831416728"></a>399</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p134911736152712"><a name="p134911736152712"></a><a name="p134911736152712"></a>to.dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p104911536202712"><a name="p104911536202712"></a><a name="p104911536202712"></a>to_dtype_npu</p>
</td>
</tr>
<tr id="row11737239181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54834161021"><a name="p54834161021"></a><a name="p54834161021"></a>400</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1849153617278"><a name="p1849153617278"></a><a name="p1849153617278"></a>to.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16491163662711"><a name="p16491163662711"></a><a name="p16491163662711"></a>to_other_npu</p>
</td>
</tr>
<tr id="row17737439171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048312163211"><a name="p1048312163211"></a><a name="p1048312163211"></a>401</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1749123692717"><a name="p1749123692717"></a><a name="p1749123692717"></a>_local_scalar_dense</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144911636112717"><a name="p144911636112717"></a><a name="p144911636112717"></a>_local_scalar_dense_npu</p>
</td>
</tr>
<tr id="row37372039201219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748381616213"><a name="p1748381616213"></a><a name="p1748381616213"></a>402</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1349117361278"><a name="p1349117361278"></a><a name="p1349117361278"></a>lstm.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p349110360276"><a name="p349110360276"></a><a name="p349110360276"></a>lstm_npu</p>
</td>
</tr>
<tr id="row167378397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2048321613210"><a name="p2048321613210"></a><a name="p2048321613210"></a>403</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1849153642719"><a name="p1849153642719"></a><a name="p1849153642719"></a>lstm.data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18491193617273"><a name="p18491193617273"></a><a name="p18491193617273"></a>lstm_npu</p>
</td>
</tr>
<tr id="row10737113912126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14842161621"><a name="p14842161621"></a><a name="p14842161621"></a>404</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1449183632717"><a name="p1449183632717"></a><a name="p1449183632717"></a>gru.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16491436132714"><a name="p16491436132714"></a><a name="p16491436132714"></a>gru_npu_</p>
</td>
</tr>
<tr id="row1873733981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548414164215"><a name="p1548414164215"></a><a name="p1548414164215"></a>405</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p849115369278"><a name="p849115369278"></a><a name="p849115369278"></a>_pack_padded_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p549113616279"><a name="p549113616279"></a><a name="p549113616279"></a>_pack_padded_sequence_npu</p>
</td>
</tr>
<tr id="row673773971216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84847161826"><a name="p84847161826"></a><a name="p84847161826"></a>406</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1649243652717"><a name="p1649243652717"></a><a name="p1649243652717"></a>_pad_packed_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p104926361273"><a name="p104926361273"></a><a name="p104926361273"></a>_pad_packed_sequence_npu</p>
</td>
</tr>
<tr id="row4737173910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104846161627"><a name="p104846161627"></a><a name="p104846161627"></a>407</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p449212364279"><a name="p449212364279"></a><a name="p449212364279"></a>set_.source_Storage</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5492123615278"><a name="p5492123615278"></a><a name="p5492123615278"></a>set_npu_</p>
</td>
</tr>
<tr id="row137371539151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048413161524"><a name="p048413161524"></a><a name="p048413161524"></a>408</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p154921936122715"><a name="p154921936122715"></a><a name="p154921936122715"></a>set_.source_Storage_storage_offset</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2492163602719"><a name="p2492163602719"></a><a name="p2492163602719"></a>set_npu_</p>
</td>
</tr>
<tr id="row27371039181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14484121619211"><a name="p14484121619211"></a><a name="p14484121619211"></a>409</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34921336182718"><a name="p34921336182718"></a><a name="p34921336182718"></a>set_.source_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p149283619278"><a name="p149283619278"></a><a name="p149283619278"></a>set_npu_</p>
</td>
</tr>
<tr id="row1173783981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84844163212"><a name="p84844163212"></a><a name="p84844163212"></a>410</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p134925368274"><a name="p134925368274"></a><a name="p134925368274"></a>set_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p149263614273"><a name="p149263614273"></a><a name="p149263614273"></a>set_npu_</p>
</td>
</tr>
<tr id="row673883991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548418163214"><a name="p1548418163214"></a><a name="p1548418163214"></a>411</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124921636132718"><a name="p124921636132718"></a><a name="p124921636132718"></a>masked_fill_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54921136152714"><a name="p54921136152714"></a><a name="p54921136152714"></a>masked_fill_npu_</p>
</td>
</tr>
<tr id="row19738173951218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2484131617213"><a name="p2484131617213"></a><a name="p2484131617213"></a>412</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p949214360270"><a name="p949214360270"></a><a name="p949214360270"></a>masked_fill_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2492173616275"><a name="p2492173616275"></a><a name="p2492173616275"></a>masked_fill_npu_</p>
</td>
</tr>
<tr id="row27381039101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154841416629"><a name="p154841416629"></a><a name="p154841416629"></a>413</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p549253616274"><a name="p549253616274"></a><a name="p549253616274"></a>masked_scatter_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16492143615275"><a name="p16492143615275"></a><a name="p16492143615275"></a>masked_scatter_npu_</p>
</td>
</tr>
<tr id="row16738939101214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948421616218"><a name="p948421616218"></a><a name="p948421616218"></a>414</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p549233662716"><a name="p549233662716"></a><a name="p549233662716"></a>view</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17492336182719"><a name="p17492336182719"></a><a name="p17492336182719"></a>view_npu</p>
</td>
</tr>
<tr id="row273863917124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p348451610215"><a name="p348451610215"></a><a name="p348451610215"></a>415</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3492163618278"><a name="p3492163618278"></a><a name="p3492163618278"></a>put_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10493936112718"><a name="p10493936112718"></a><a name="p10493936112718"></a>put_npu_</p>
</td>
</tr>
<tr id="row11738183918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84841016325"><a name="p84841016325"></a><a name="p84841016325"></a>416</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1493123617278"><a name="p1493123617278"></a><a name="p1493123617278"></a>index_add_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p749383652716"><a name="p749383652716"></a><a name="p749383652716"></a>index_add_npu_</p>
</td>
</tr>
<tr id="row5738739161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104841316522"><a name="p104841316522"></a><a name="p104841316522"></a>417</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p94937366276"><a name="p94937366276"></a><a name="p94937366276"></a>index_add</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17493936122713"><a name="p17493936122713"></a><a name="p17493936122713"></a>index_add_npu</p>
</td>
</tr>
<tr id="row273811393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24840161020"><a name="p24840161020"></a><a name="p24840161020"></a>418</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p74931636152714"><a name="p74931636152714"></a><a name="p74931636152714"></a>index_add.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16493143614272"><a name="p16493143614272"></a><a name="p16493143614272"></a>index_add_npu</p>
</td>
</tr>
<tr id="row13738739111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3484101617211"><a name="p3484101617211"></a><a name="p3484101617211"></a>419</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p44936360277"><a name="p44936360277"></a><a name="p44936360277"></a>index_fill_.int_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18493133612277"><a name="p18493133612277"></a><a name="p18493133612277"></a>index_fill_npu_</p>
</td>
</tr>
<tr id="row773812399125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5484151613217"><a name="p5484151613217"></a><a name="p5484151613217"></a>420</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10493153617277"><a name="p10493153617277"></a><a name="p10493153617277"></a>index_fill.int_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18493436132713"><a name="p18493436132713"></a><a name="p18493436132713"></a>index_fill_npu</p>
</td>
</tr>
<tr id="row2738123901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64842162216"><a name="p64842162216"></a><a name="p64842162216"></a>421</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17493836112720"><a name="p17493836112720"></a><a name="p17493836112720"></a>index_fill_.int_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13493436172716"><a name="p13493436172716"></a><a name="p13493436172716"></a>index_fill_npu_</p>
</td>
</tr>
<tr id="row27381739161215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124858161217"><a name="p124858161217"></a><a name="p124858161217"></a>422</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11493173619275"><a name="p11493173619275"></a><a name="p11493173619275"></a>index_fill.int_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14493153610275"><a name="p14493153610275"></a><a name="p14493153610275"></a>index_fill_npu</p>
</td>
</tr>
<tr id="row197395394127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2485516229"><a name="p2485516229"></a><a name="p2485516229"></a>423</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18493103672711"><a name="p18493103672711"></a><a name="p18493103672711"></a>scatter_.src</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p549333619275"><a name="p549333619275"></a><a name="p549333619275"></a>scatter_npu_</p>
</td>
</tr>
<tr id="row97393398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54852161229"><a name="p54852161229"></a><a name="p54852161229"></a>424</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p749393602713"><a name="p749393602713"></a><a name="p749393602713"></a>scatter_.value</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18494143617277"><a name="p18494143617277"></a><a name="p18494143617277"></a>scatter_npu_</p>
</td>
</tr>
<tr id="row18739193916123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174851116725"><a name="p174851116725"></a><a name="p174851116725"></a>425</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17494136122710"><a name="p17494136122710"></a><a name="p17494136122710"></a>scatter_add_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2494163615275"><a name="p2494163615275"></a><a name="p2494163615275"></a>scatter_add_npu_</p>
</td>
</tr>
<tr id="row19739103918129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5485416227"><a name="p5485416227"></a><a name="p5485416227"></a>426</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1549412366275"><a name="p1549412366275"></a><a name="p1549412366275"></a>scatter_add</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15494103611279"><a name="p15494103611279"></a><a name="p15494103611279"></a>scatter_add_npu</p>
</td>
</tr>
<tr id="row157396393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24851316624"><a name="p24851316624"></a><a name="p24851316624"></a>427</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1494436122714"><a name="p1494436122714"></a><a name="p1494436122714"></a>scatter_add.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144946365273"><a name="p144946365273"></a><a name="p144946365273"></a>scatter_add_npu</p>
</td>
</tr>
<tr id="row187391139131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648501620215"><a name="p1648501620215"></a><a name="p1648501620215"></a>428</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5494163632717"><a name="p5494163632717"></a><a name="p5494163632717"></a>lt_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144941336102710"><a name="p144941336102710"></a><a name="p144941336102710"></a>lt_npu_</p>
</td>
</tr>
<tr id="row1273973941211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19485516328"><a name="p19485516328"></a><a name="p19485516328"></a>429</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1494103614274"><a name="p1494103614274"></a><a name="p1494103614274"></a>lt_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84941136132717"><a name="p84941136132717"></a><a name="p84941136132717"></a>lt_npu_</p>
</td>
</tr>
<tr id="row77391439201210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648511163215"><a name="p648511163215"></a><a name="p648511163215"></a>430</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1549463642719"><a name="p1549463642719"></a><a name="p1549463642719"></a>gt_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p849418366272"><a name="p849418366272"></a><a name="p849418366272"></a>gt_npu_</p>
</td>
</tr>
<tr id="row3739163911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9485416425"><a name="p9485416425"></a><a name="p9485416425"></a>431</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4494203612720"><a name="p4494203612720"></a><a name="p4494203612720"></a>gt_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44941936162710"><a name="p44941936162710"></a><a name="p44941936162710"></a>gt_npu_</p>
</td>
</tr>
<tr id="row37391539141215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12485141616210"><a name="p12485141616210"></a><a name="p12485141616210"></a>432</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24941736142712"><a name="p24941736142712"></a><a name="p24941736142712"></a>le_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p649419367277"><a name="p649419367277"></a><a name="p649419367277"></a>le_npu_</p>
</td>
</tr>
<tr id="row573993941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44851016222"><a name="p44851016222"></a><a name="p44851016222"></a>433</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2494123618279"><a name="p2494123618279"></a><a name="p2494123618279"></a>le_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p249411365273"><a name="p249411365273"></a><a name="p249411365273"></a>le_npu_</p>
</td>
</tr>
<tr id="row3740239171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15485141617211"><a name="p15485141617211"></a><a name="p15485141617211"></a>434</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9495103616275"><a name="p9495103616275"></a><a name="p9495103616275"></a>ge_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124951361272"><a name="p124951361272"></a><a name="p124951361272"></a>ge_npu_</p>
</td>
</tr>
<tr id="row874013971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94851716128"><a name="p94851716128"></a><a name="p94851716128"></a>435</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p849523617279"><a name="p849523617279"></a><a name="p849523617279"></a>ge_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11495183616275"><a name="p11495183616275"></a><a name="p11495183616275"></a>ge_npu_</p>
</td>
</tr>
<tr id="row67401395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3486116926"><a name="p3486116926"></a><a name="p3486116926"></a>436</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1149513612278"><a name="p1149513612278"></a><a name="p1149513612278"></a>eq_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15495123642712"><a name="p15495123642712"></a><a name="p15495123642712"></a>eq_npu_</p>
</td>
</tr>
<tr id="row13740439201212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948613161926"><a name="p948613161926"></a><a name="p948613161926"></a>437</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0495123616273"><a name="p0495123616273"></a><a name="p0495123616273"></a>eq_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4495636132714"><a name="p4495636132714"></a><a name="p4495636132714"></a>eq_npu_</p>
</td>
</tr>
<tr id="row1174013916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448661614217"><a name="p448661614217"></a><a name="p448661614217"></a>438</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19495103672720"><a name="p19495103672720"></a><a name="p19495103672720"></a>ne_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1649543682717"><a name="p1649543682717"></a><a name="p1649543682717"></a>ne_npu_</p>
</td>
</tr>
<tr id="row774016390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1248616161221"><a name="p1248616161221"></a><a name="p1248616161221"></a>439</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p134955364273"><a name="p134955364273"></a><a name="p134955364273"></a>ne_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p849515365275"><a name="p849515365275"></a><a name="p849515365275"></a>ne_npu_</p>
</td>
</tr>
<tr id="row7740193917126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848620161423"><a name="p1848620161423"></a><a name="p1848620161423"></a>440</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p049516367272"><a name="p049516367272"></a><a name="p049516367272"></a>bitwise_and.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24951236142718"><a name="p24951236142718"></a><a name="p24951236142718"></a>bitwise_and_out_npu</p>
</td>
</tr>
<tr id="row67401439171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648691611212"><a name="p1648691611212"></a><a name="p1648691611212"></a>441</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1495143617278"><a name="p1495143617278"></a><a name="p1495143617278"></a>bitwise_and.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11495183618271"><a name="p11495183618271"></a><a name="p11495183618271"></a>bitwise_and_out_npu</p>
</td>
</tr>
<tr id="row1174003910120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194861216228"><a name="p194861216228"></a><a name="p194861216228"></a>442</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p74951836152711"><a name="p74951836152711"></a><a name="p74951836152711"></a>bitwise_and.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1149518369276"><a name="p1149518369276"></a><a name="p1149518369276"></a>bitwise_and_npu</p>
</td>
</tr>
<tr id="row12740739111211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44861016721"><a name="p44861016721"></a><a name="p44861016721"></a>443</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p749683613279"><a name="p749683613279"></a><a name="p749683613279"></a>bitwise_and.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18496113682710"><a name="p18496113682710"></a><a name="p18496113682710"></a>bitwise_and_npu</p>
</td>
</tr>
<tr id="row18740163913121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148616161926"><a name="p1148616161926"></a><a name="p1148616161926"></a>444</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p249623618278"><a name="p249623618278"></a><a name="p249623618278"></a>bitwise_and_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6496173614272"><a name="p6496173614272"></a><a name="p6496173614272"></a>bitwise_and_npu_</p>
</td>
</tr>
<tr id="row574163931217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1486816423"><a name="p1486816423"></a><a name="p1486816423"></a>445</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p649683616275"><a name="p649683616275"></a><a name="p649683616275"></a>bitwise_and_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p04961836182717"><a name="p04961836182717"></a><a name="p04961836182717"></a>bitwise_and_npu_</p>
</td>
</tr>
<tr id="row1774114393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6486111614216"><a name="p6486111614216"></a><a name="p6486111614216"></a>446</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19496173612276"><a name="p19496173612276"></a><a name="p19496173612276"></a>__and__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12496173617279"><a name="p12496173617279"></a><a name="p12496173617279"></a>__and___npu</p>
</td>
</tr>
<tr id="row14741639161211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14869169213"><a name="p14869169213"></a><a name="p14869169213"></a>447</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9496936192711"><a name="p9496936192711"></a><a name="p9496936192711"></a>__and__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14496133612716"><a name="p14496133612716"></a><a name="p14496133612716"></a>__and___npu</p>
</td>
</tr>
<tr id="row0741193911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048613162213"><a name="p048613162213"></a><a name="p048613162213"></a>448</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p449612361272"><a name="p449612361272"></a><a name="p449612361272"></a>bitwise_or.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8496203610274"><a name="p8496203610274"></a><a name="p8496203610274"></a>bitwise_or_out_npu</p>
</td>
</tr>
<tr id="row6741839161215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548619162217"><a name="p548619162217"></a><a name="p548619162217"></a>449</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p54962368279"><a name="p54962368279"></a><a name="p54962368279"></a>bitwise_or.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2049613632720"><a name="p2049613632720"></a><a name="p2049613632720"></a>bitwise_or_out_npu</p>
</td>
</tr>
<tr id="row9741193991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16486716328"><a name="p16486716328"></a><a name="p16486716328"></a>450</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18496193632718"><a name="p18496193632718"></a><a name="p18496193632718"></a>bitwise_or.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p549610363278"><a name="p549610363278"></a><a name="p549610363278"></a>bitwise_or_npu</p>
</td>
</tr>
<tr id="row974103910121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048613168218"><a name="p048613168218"></a><a name="p048613168218"></a>451</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1749613612719"><a name="p1749613612719"></a><a name="p1749613612719"></a>bitwise_or.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p649711366270"><a name="p649711366270"></a><a name="p649711366270"></a>bitwise_or_npu</p>
</td>
</tr>
<tr id="row1741103911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948615169212"><a name="p948615169212"></a><a name="p948615169212"></a>452</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1949718364279"><a name="p1949718364279"></a><a name="p1949718364279"></a>bitwise_or_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134971736132713"><a name="p134971736132713"></a><a name="p134971736132713"></a>bitwise_or_npu_</p>
</td>
</tr>
<tr id="row1674113914126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184861161224"><a name="p184861161224"></a><a name="p184861161224"></a>453</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1049710361276"><a name="p1049710361276"></a><a name="p1049710361276"></a>bitwise_or_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p949743662716"><a name="p949743662716"></a><a name="p949743662716"></a>bitwise_or_npu_</p>
</td>
</tr>
<tr id="row4741839101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174861816527"><a name="p174861816527"></a><a name="p174861816527"></a>454</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11497436102719"><a name="p11497436102719"></a><a name="p11497436102719"></a>__or__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14975369270"><a name="p14975369270"></a><a name="p14975369270"></a>__or___npu</p>
</td>
</tr>
<tr id="row137421539161220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948717168215"><a name="p948717168215"></a><a name="p948717168215"></a>455</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5497336132710"><a name="p5497336132710"></a><a name="p5497336132710"></a>__or__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14971936132716"><a name="p14971936132716"></a><a name="p14971936132716"></a>__or___npu</p>
</td>
</tr>
<tr id="row8742143911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44871016526"><a name="p44871016526"></a><a name="p44871016526"></a>456</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20497133618275"><a name="p20497133618275"></a><a name="p20497133618275"></a>__ior__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14971536162717"><a name="p14971536162717"></a><a name="p14971536162717"></a>__ior___npu</p>
</td>
</tr>
<tr id="row1274263912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15487716629"><a name="p15487716629"></a><a name="p15487716629"></a>457</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p249718365272"><a name="p249718365272"></a><a name="p249718365272"></a>__ior__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1849733672718"><a name="p1849733672718"></a><a name="p1849733672718"></a>__ior___npu</p>
</td>
</tr>
<tr id="row1374210390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348719168215"><a name="p1348719168215"></a><a name="p1348719168215"></a>458</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p04971336202715"><a name="p04971336202715"></a><a name="p04971336202715"></a>bitwise_xor.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17497133612710"><a name="p17497133612710"></a><a name="p17497133612710"></a>bitwise_xor_out_npu</p>
</td>
</tr>
<tr id="row15742123901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4487316323"><a name="p4487316323"></a><a name="p4487316323"></a>459</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17497936102717"><a name="p17497936102717"></a><a name="p17497936102717"></a>bitwise_xor.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14971236162717"><a name="p14971236162717"></a><a name="p14971236162717"></a>bitwise_xor_out_npu</p>
</td>
</tr>
<tr id="row2742133918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848781610212"><a name="p1848781610212"></a><a name="p1848781610212"></a>460</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124971836122711"><a name="p124971836122711"></a><a name="p124971836122711"></a>bitwise_xor.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20498153602712"><a name="p20498153602712"></a><a name="p20498153602712"></a>bitwise_xor_npu</p>
</td>
</tr>
<tr id="row1974273913128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74878161021"><a name="p74878161021"></a><a name="p74878161021"></a>461</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19498173620277"><a name="p19498173620277"></a><a name="p19498173620277"></a>bitwise_xor.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1849893622718"><a name="p1849893622718"></a><a name="p1849893622718"></a>bitwise_xor_npu</p>
</td>
</tr>
<tr id="row274223916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p148717162219"><a name="p148717162219"></a><a name="p148717162219"></a>462</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10498436122718"><a name="p10498436122718"></a><a name="p10498436122718"></a>bitwise_xor_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1049853617278"><a name="p1049853617278"></a><a name="p1049853617278"></a>bitwise_xor_npu_</p>
</td>
</tr>
<tr id="row13742739201212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16487216924"><a name="p16487216924"></a><a name="p16487216924"></a>463</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194983369279"><a name="p194983369279"></a><a name="p194983369279"></a>bitwise_xor_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p04981636192713"><a name="p04981636192713"></a><a name="p04981636192713"></a>bitwise_xor_npu_</p>
</td>
</tr>
<tr id="row3742143941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748719165213"><a name="p1748719165213"></a><a name="p1748719165213"></a>464</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20498193619272"><a name="p20498193619272"></a><a name="p20498193619272"></a>__xor__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p749811365275"><a name="p749811365275"></a><a name="p749811365275"></a>__xor___npu</p>
</td>
</tr>
<tr id="row57420390126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348713163210"><a name="p1348713163210"></a><a name="p1348713163210"></a>465</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p949843662716"><a name="p949843662716"></a><a name="p949843662716"></a>__xor__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p349823632718"><a name="p349823632718"></a><a name="p349823632718"></a>__xor___npu</p>
</td>
</tr>
<tr id="row197431539141210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18487316622"><a name="p18487316622"></a><a name="p18487316622"></a>466</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p549833622711"><a name="p549833622711"></a><a name="p549833622711"></a>__lshift__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5498336182713"><a name="p5498336182713"></a><a name="p5498336182713"></a>__lshift___npu</p>
</td>
</tr>
<tr id="row18743173911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18487116624"><a name="p18487116624"></a><a name="p18487116624"></a>467</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16498636182720"><a name="p16498636182720"></a><a name="p16498636182720"></a>__lshift__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194987364273"><a name="p194987364273"></a><a name="p194987364273"></a>__lshift___npu</p>
</td>
</tr>
<tr id="row15743103916125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14871161324"><a name="p14871161324"></a><a name="p14871161324"></a>468</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p449811361274"><a name="p449811361274"></a><a name="p449811361274"></a>__ilshift__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174981369270"><a name="p174981369270"></a><a name="p174981369270"></a>__iLshift___npu</p>
</td>
</tr>
<tr id="row774363951216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048711162023"><a name="p1048711162023"></a><a name="p1048711162023"></a>469</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13498636202714"><a name="p13498636202714"></a><a name="p13498636202714"></a>__ilshift__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10498123622719"><a name="p10498123622719"></a><a name="p10498123622719"></a>__iLshift___npu</p>
</td>
</tr>
<tr id="row47431839151217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13488201612217"><a name="p13488201612217"></a><a name="p13488201612217"></a>470</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1149916368278"><a name="p1149916368278"></a><a name="p1149916368278"></a>__rshift__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14991936132710"><a name="p14991936132710"></a><a name="p14991936132710"></a>__rshift___npu</p>
</td>
</tr>
<tr id="row117431739171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44885167216"><a name="p44885167216"></a><a name="p44885167216"></a>471</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1649923672713"><a name="p1649923672713"></a><a name="p1649923672713"></a>__rshift__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54991936192710"><a name="p54991936192710"></a><a name="p54991936192710"></a>__rshift___npu</p>
</td>
</tr>
<tr id="row14743639201214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448815161827"><a name="p448815161827"></a><a name="p448815161827"></a>472</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124994362278"><a name="p124994362278"></a><a name="p124994362278"></a>__irshift__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1649933682717"><a name="p1649933682717"></a><a name="p1649933682717"></a>__iRshift___npu</p>
</td>
</tr>
<tr id="row16743183921211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8488131612215"><a name="p8488131612215"></a><a name="p8488131612215"></a>473</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p449913611279"><a name="p449913611279"></a><a name="p449913611279"></a>__irshift__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8499836152720"><a name="p8499836152720"></a><a name="p8499836152720"></a>__iRshift___npu</p>
</td>
</tr>
<tr id="row4743103913128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648841617216"><a name="p1648841617216"></a><a name="p1648841617216"></a>474</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4499203602714"><a name="p4499203602714"></a><a name="p4499203602714"></a>atan2_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44991236192714"><a name="p44991236192714"></a><a name="p44991236192714"></a>atan2_npu_</p>
</td>
</tr>
<tr id="row137431039181217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124884161627"><a name="p124884161627"></a><a name="p124884161627"></a>475</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p104990369275"><a name="p104990369275"></a><a name="p104990369275"></a>tril_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54991436142712"><a name="p54991436142712"></a><a name="p54991436142712"></a>tril_npu_</p>
</td>
</tr>
<tr id="row674333911217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12488111614211"><a name="p12488111614211"></a><a name="p12488111614211"></a>476</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0499113632711"><a name="p0499113632711"></a><a name="p0499113632711"></a>triu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p164991936172715"><a name="p164991936172715"></a><a name="p164991936172715"></a>triu_npu_</p>
</td>
</tr>
<tr id="row1374313971215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9488191612213"><a name="p9488191612213"></a><a name="p9488191612213"></a>477</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124991736182720"><a name="p124991736182720"></a><a name="p124991736182720"></a>renorm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124992369275"><a name="p124992369275"></a><a name="p124992369275"></a>renorm_npu_</p>
</td>
</tr>
<tr id="row9744133919120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144881516723"><a name="p144881516723"></a><a name="p144881516723"></a>478</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12499103672714"><a name="p12499103672714"></a><a name="p12499103672714"></a>pow_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144991836102711"><a name="p144991836102711"></a><a name="p144991836102711"></a>pow_npu_</p>
</td>
</tr>
<tr id="row37441439121211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348831619216"><a name="p1348831619216"></a><a name="p1348831619216"></a>479</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p249973611272"><a name="p249973611272"></a><a name="p249973611272"></a>pow_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65001368275"><a name="p65001368275"></a><a name="p65001368275"></a>pow_npu_</p>
</td>
</tr>
<tr id="row1774413961216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15488716822"><a name="p15488716822"></a><a name="p15488716822"></a>480</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5500153672720"><a name="p5500153672720"></a><a name="p5500153672720"></a>lerp_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p205001236152710"><a name="p205001236152710"></a><a name="p205001236152710"></a>lerp_npu_</p>
</td>
</tr>
<tr id="row12744153911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848811161924"><a name="p1848811161924"></a><a name="p1848811161924"></a>481</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17500736202714"><a name="p17500736202714"></a><a name="p17500736202714"></a>lerp_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4500203614277"><a name="p4500203614277"></a><a name="p4500203614277"></a>lerp_npu_</p>
</td>
</tr>
<tr id="row197441439141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748814161327"><a name="p1748814161327"></a><a name="p1748814161327"></a>482</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p150003610277"><a name="p150003610277"></a><a name="p150003610277"></a>fmod_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9500436132711"><a name="p9500436132711"></a><a name="p9500436132711"></a>fmod_npu_</p>
</td>
</tr>
<tr id="row77441939151220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194883161223"><a name="p194883161223"></a><a name="p194883161223"></a>483</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17500183692710"><a name="p17500183692710"></a><a name="p17500183692710"></a>fmod_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18500836152717"><a name="p18500836152717"></a><a name="p18500836152717"></a>fmod_npu_</p>
</td>
</tr>
<tr id="row10744193918120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1488111619216"><a name="p1488111619216"></a><a name="p1488111619216"></a>484</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p750073622711"><a name="p750073622711"></a><a name="p750073622711"></a>remainder_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175001936152719"><a name="p175001936152719"></a><a name="p175001936152719"></a>remainder_npu_</p>
</td>
</tr>
<tr id="row574493911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4489316329"><a name="p4489316329"></a><a name="p4489316329"></a>485</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10500193612270"><a name="p10500193612270"></a><a name="p10500193612270"></a>remainder_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195001636162710"><a name="p195001636162710"></a><a name="p195001636162710"></a>remainder_npu_</p>
</td>
</tr>
<tr id="row4744123901217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148981611212"><a name="p1148981611212"></a><a name="p1148981611212"></a>486</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15500193652711"><a name="p15500193652711"></a><a name="p15500193652711"></a>addbmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1350019361274"><a name="p1350019361274"></a><a name="p1350019361274"></a>addbmm_npu_</p>
</td>
</tr>
<tr id="row874423915126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848915161423"><a name="p848915161423"></a><a name="p848915161423"></a>487</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450023642712"><a name="p450023642712"></a><a name="p450023642712"></a>addbmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p205006368278"><a name="p205006368278"></a><a name="p205006368278"></a>addbmm_out_npu</p>
</td>
</tr>
<tr id="row1774411397122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114898164213"><a name="p114898164213"></a><a name="p114898164213"></a>488</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75001836132714"><a name="p75001836132714"></a><a name="p75001836132714"></a>addbmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11500936192713"><a name="p11500936192713"></a><a name="p11500936192713"></a>addbmm_npu</p>
</td>
</tr>
<tr id="row1074423913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04891916228"><a name="p04891916228"></a><a name="p04891916228"></a>489</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p145011336192712"><a name="p145011336192712"></a><a name="p145011336192712"></a>addcdiv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125011936192710"><a name="p125011936192710"></a><a name="p125011936192710"></a>addcdiv_npu_</p>
</td>
</tr>
<tr id="row14745239101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184895164211"><a name="p184895164211"></a><a name="p184895164211"></a>490</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1250118367272"><a name="p1250118367272"></a><a name="p1250118367272"></a>random_.from</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1950173615276"><a name="p1950173615276"></a><a name="p1950173615276"></a>random_npu_</p>
</td>
</tr>
<tr id="row67451339151211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348901615217"><a name="p1348901615217"></a><a name="p1348901615217"></a>491</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7501173614274"><a name="p7501173614274"></a><a name="p7501173614274"></a>random_.to</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25011136192716"><a name="p25011136192716"></a><a name="p25011136192716"></a>random_npu_</p>
</td>
</tr>
<tr id="row5745239191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5489316928"><a name="p5489316928"></a><a name="p5489316928"></a>492</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35014363271"><a name="p35014363271"></a><a name="p35014363271"></a>random_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75011136202714"><a name="p75011136202714"></a><a name="p75011136202714"></a>random_npu_</p>
</td>
</tr>
<tr id="row174516395120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94891716826"><a name="p94891716826"></a><a name="p94891716826"></a>493</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450163619271"><a name="p450163619271"></a><a name="p450163619271"></a>uniform_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3501123616278"><a name="p3501123616278"></a><a name="p3501123616278"></a>uniform_npu_</p>
</td>
</tr>
<tr id="row1174518399127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748912166214"><a name="p1748912166214"></a><a name="p1748912166214"></a>494</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p55011736152718"><a name="p55011736152718"></a><a name="p55011736152718"></a>diag.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25011236152718"><a name="p25011236152718"></a><a name="p25011236152718"></a>diag_out_npu</p>
</td>
</tr>
<tr id="row16745123913122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848911612218"><a name="p848911612218"></a><a name="p848911612218"></a>495</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35011836182710"><a name="p35011836182710"></a><a name="p35011836182710"></a>diag</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18501836172710"><a name="p18501836172710"></a><a name="p18501836172710"></a>diag_npu</p>
</td>
</tr>
<tr id="row197450393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p748915161220"><a name="p748915161220"></a><a name="p748915161220"></a>496</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175011361275"><a name="p175011361275"></a><a name="p175011361275"></a>cross.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25011336172712"><a name="p25011336172712"></a><a name="p25011336172712"></a>cross_out_npu</p>
</td>
</tr>
<tr id="row11745739111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648914163218"><a name="p648914163218"></a><a name="p648914163218"></a>497</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1150113610273"><a name="p1150113610273"></a><a name="p1150113610273"></a>cross</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8501193652711"><a name="p8501193652711"></a><a name="p8501193652711"></a>cross_npu</p>
</td>
</tr>
<tr id="row177451439181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134901116021"><a name="p134901116021"></a><a name="p134901116021"></a>498</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1050203662717"><a name="p1050203662717"></a><a name="p1050203662717"></a>triu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65022360271"><a name="p65022360271"></a><a name="p65022360271"></a>triu_out_npu</p>
</td>
</tr>
<tr id="row13745143981210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174903162215"><a name="p174903162215"></a><a name="p174903162215"></a>499</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p750273611277"><a name="p750273611277"></a><a name="p750273611277"></a>triu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p450210364277"><a name="p450210364277"></a><a name="p450210364277"></a>triu_npu</p>
</td>
</tr>
<tr id="row9745103914127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2490216726"><a name="p2490216726"></a><a name="p2490216726"></a>500</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1502536142713"><a name="p1502536142713"></a><a name="p1502536142713"></a>tril.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7502336182714"><a name="p7502336182714"></a><a name="p7502336182714"></a>tril_out_npu</p>
</td>
</tr>
<tr id="row1074693914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194901716527"><a name="p194901716527"></a><a name="p194901716527"></a>501</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8502113612715"><a name="p8502113612715"></a><a name="p8502113612715"></a>tril</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4502636102712"><a name="p4502636102712"></a><a name="p4502636102712"></a>tril_npu</p>
</td>
</tr>
<tr id="row12746103917120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24908161421"><a name="p24908161421"></a><a name="p24908161421"></a>502</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3502136102717"><a name="p3502136102717"></a><a name="p3502136102717"></a>tril_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17502123617273"><a name="p17502123617273"></a><a name="p17502123617273"></a>tril_indices_npu</p>
</td>
</tr>
<tr id="row1474623981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64901216329"><a name="p64901216329"></a><a name="p64901216329"></a>503</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1450203682716"><a name="p1450203682716"></a><a name="p1450203682716"></a>triu_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p250211368271"><a name="p250211368271"></a><a name="p250211368271"></a>triu_indices_npu</p>
</td>
</tr>
<tr id="row2074613920121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2490116325"><a name="p2490116325"></a><a name="p2490116325"></a>504</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p205021736142712"><a name="p205021736142712"></a><a name="p205021736142712"></a>ne.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1150218365275"><a name="p1150218365275"></a><a name="p1150218365275"></a>ne_out_npu</p>
</td>
</tr>
<tr id="row16746839101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049021616210"><a name="p1049021616210"></a><a name="p1049021616210"></a>505</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155021236102716"><a name="p155021236102716"></a><a name="p155021236102716"></a>ne.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1150263642711"><a name="p1150263642711"></a><a name="p1150263642711"></a>ne_npu</p>
</td>
</tr>
<tr id="row1674643912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74914162213"><a name="p74914162213"></a><a name="p74914162213"></a>506</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45027368272"><a name="p45027368272"></a><a name="p45027368272"></a>ne.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1150213610275"><a name="p1150213610275"></a><a name="p1150213610275"></a>ne_out_npu</p>
</td>
</tr>
<tr id="row774653921210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049119161224"><a name="p1049119161224"></a><a name="p1049119161224"></a>507</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45021936152715"><a name="p45021936152715"></a><a name="p45021936152715"></a>ne.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7503163614279"><a name="p7503163614279"></a><a name="p7503163614279"></a>ne_npu</p>
</td>
</tr>
<tr id="row0746339141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2491916325"><a name="p2491916325"></a><a name="p2491916325"></a>508</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1550311368277"><a name="p1550311368277"></a><a name="p1550311368277"></a>eq.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p35038365278"><a name="p35038365278"></a><a name="p35038365278"></a>eq_out_npu</p>
</td>
</tr>
<tr id="row6748143914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6491101613213"><a name="p6491101613213"></a><a name="p6491101613213"></a>509</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11503736192710"><a name="p11503736192710"></a><a name="p11503736192710"></a>eq.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14503936172711"><a name="p14503936172711"></a><a name="p14503936172711"></a>eq_npu</p>
</td>
</tr>
<tr id="row67489392120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84911416126"><a name="p84911416126"></a><a name="p84911416126"></a>510</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1850316363271"><a name="p1850316363271"></a><a name="p1850316363271"></a>eq.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10503143614276"><a name="p10503143614276"></a><a name="p10503143614276"></a>eq_out_npu</p>
</td>
</tr>
<tr id="row5748203971215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6491171620217"><a name="p6491171620217"></a><a name="p6491171620217"></a>511</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1503153612713"><a name="p1503153612713"></a><a name="p1503153612713"></a>eq.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1250316362279"><a name="p1250316362279"></a><a name="p1250316362279"></a>eq_npu</p>
</td>
</tr>
<tr id="row774883921211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649171619216"><a name="p1649171619216"></a><a name="p1649171619216"></a>512</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175031236152715"><a name="p175031236152715"></a><a name="p175031236152715"></a>ge.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1650333617278"><a name="p1650333617278"></a><a name="p1650333617278"></a>ge_out_npu</p>
</td>
</tr>
<tr id="row17748203901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184913162217"><a name="p184913162217"></a><a name="p184913162217"></a>513</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11503183652714"><a name="p11503183652714"></a><a name="p11503183652714"></a>ge.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20503123652714"><a name="p20503123652714"></a><a name="p20503123652714"></a>ge_npu</p>
</td>
</tr>
<tr id="row147481539151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94912161728"><a name="p94912161728"></a><a name="p94912161728"></a>514</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85037364272"><a name="p85037364272"></a><a name="p85037364272"></a>ge.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1050310363279"><a name="p1050310363279"></a><a name="p1050310363279"></a>ge_out_npu</p>
</td>
</tr>
<tr id="row177481139191214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349117161621"><a name="p349117161621"></a><a name="p349117161621"></a>515</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19503036102720"><a name="p19503036102720"></a><a name="p19503036102720"></a>ge.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p650312364279"><a name="p650312364279"></a><a name="p650312364279"></a>ge_npu</p>
</td>
</tr>
<tr id="row87480397120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184911316522"><a name="p184911316522"></a><a name="p184911316522"></a>516</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5503136152711"><a name="p5503136152711"></a><a name="p5503136152711"></a>le.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19504163610274"><a name="p19504163610274"></a><a name="p19504163610274"></a>le_out_npu</p>
</td>
</tr>
<tr id="row7748163971216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10491141612217"><a name="p10491141612217"></a><a name="p10491141612217"></a>517</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95048361270"><a name="p95048361270"></a><a name="p95048361270"></a>le.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13504183682712"><a name="p13504183682712"></a><a name="p13504183682712"></a>le_npu</p>
</td>
</tr>
<tr id="row0748239151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1491111611219"><a name="p1491111611219"></a><a name="p1491111611219"></a>518</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p950453692717"><a name="p950453692717"></a><a name="p950453692717"></a>le.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9504436122711"><a name="p9504436122711"></a><a name="p9504436122711"></a>le_out_npu</p>
</td>
</tr>
<tr id="row12748133913129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149112161722"><a name="p1149112161722"></a><a name="p1149112161722"></a>519</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16504183611275"><a name="p16504183611275"></a><a name="p16504183611275"></a>le.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8504336132720"><a name="p8504336132720"></a><a name="p8504336132720"></a>le_npu</p>
</td>
</tr>
<tr id="row1474915397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74921816428"><a name="p74921816428"></a><a name="p74921816428"></a>520</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p145041936102714"><a name="p145041936102714"></a><a name="p145041936102714"></a>gt.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p450423619276"><a name="p450423619276"></a><a name="p450423619276"></a>gt_out_npu</p>
</td>
</tr>
<tr id="row18749153921216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649217161928"><a name="p649217161928"></a><a name="p649217161928"></a>521</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85041536132712"><a name="p85041536132712"></a><a name="p85041536132712"></a>gt.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p45041136122710"><a name="p45041136122710"></a><a name="p45041136122710"></a>gt_npu</p>
</td>
</tr>
<tr id="row1674923991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54922161211"><a name="p54922161211"></a><a name="p54922161211"></a>522</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2504153692716"><a name="p2504153692716"></a><a name="p2504153692716"></a>gt.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p450443632711"><a name="p450443632711"></a><a name="p450443632711"></a>gt_out_npu</p>
</td>
</tr>
<tr id="row12749153919120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84922161524"><a name="p84922161524"></a><a name="p84922161524"></a>523</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p750411361272"><a name="p750411361272"></a><a name="p750411361272"></a>gt.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14504153617277"><a name="p14504153617277"></a><a name="p14504153617277"></a>gt_npu</p>
</td>
</tr>
<tr id="row12749339171214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14492916124"><a name="p14492916124"></a><a name="p14492916124"></a>524</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1750453613275"><a name="p1750453613275"></a><a name="p1750453613275"></a>lt.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p250453682713"><a name="p250453682713"></a><a name="p250453682713"></a>lt_out_npu</p>
</td>
</tr>
<tr id="row1749193913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8492171612220"><a name="p8492171612220"></a><a name="p8492171612220"></a>525</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19504536122719"><a name="p19504536122719"></a><a name="p19504536122719"></a>lt.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p650453672717"><a name="p650453672717"></a><a name="p650453672717"></a>lt_npu</p>
</td>
</tr>
<tr id="row1474913393124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1449251616216"><a name="p1449251616216"></a><a name="p1449251616216"></a>526</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p050553662718"><a name="p050553662718"></a><a name="p050553662718"></a>lt.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p450511361276"><a name="p450511361276"></a><a name="p450511361276"></a>lt_out_npu</p>
</td>
</tr>
<tr id="row87491639201213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10492181619219"><a name="p10492181619219"></a><a name="p10492181619219"></a>527</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185051036172711"><a name="p185051036172711"></a><a name="p185051036172711"></a>lt.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p850553614272"><a name="p850553614272"></a><a name="p850553614272"></a>lt_npu</p>
</td>
</tr>
<tr id="row0749113919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24922161127"><a name="p24922161127"></a><a name="p24922161127"></a>528</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12505173662717"><a name="p12505173662717"></a><a name="p12505173662717"></a>take.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19505123618277"><a name="p19505123618277"></a><a name="p19505123618277"></a>take_out_npu</p>
</td>
</tr>
<tr id="row374933913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2492816122"><a name="p2492816122"></a><a name="p2492816122"></a>529</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9505536152714"><a name="p9505536152714"></a><a name="p9505536152714"></a>take</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1450517360279"><a name="p1450517360279"></a><a name="p1450517360279"></a>take_npu</p>
</td>
</tr>
<tr id="row3749339111220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94924165210"><a name="p94924165210"></a><a name="p94924165210"></a>530</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p750583612278"><a name="p750583612278"></a><a name="p750583612278"></a>index_select.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1350523622719"><a name="p1350523622719"></a><a name="p1350523622719"></a>index_select_out_npu</p>
</td>
</tr>
<tr id="row117501939131219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194924161022"><a name="p194924161022"></a><a name="p194924161022"></a>531</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1850510368272"><a name="p1850510368272"></a><a name="p1850510368272"></a>index_select</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1250563622718"><a name="p1250563622718"></a><a name="p1250563622718"></a>index_select_npu</p>
</td>
</tr>
<tr id="row13750123914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649217161124"><a name="p1649217161124"></a><a name="p1649217161124"></a>532</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1950573612276"><a name="p1950573612276"></a><a name="p1950573612276"></a>index_select.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1750573612272"><a name="p1750573612272"></a><a name="p1750573612272"></a>index_select_out_npu</p>
</td>
</tr>
<tr id="row47504399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114923161120"><a name="p114923161120"></a><a name="p114923161120"></a>533</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p050503620271"><a name="p050503620271"></a><a name="p050503620271"></a>index_select.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145057360279"><a name="p145057360279"></a><a name="p145057360279"></a>index_select_npu</p>
</td>
</tr>
<tr id="row0750163971213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24921016328"><a name="p24921016328"></a><a name="p24921016328"></a>534</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p050553672720"><a name="p050553672720"></a><a name="p050553672720"></a>masked_select.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1250533692712"><a name="p1250533692712"></a><a name="p1250533692712"></a>masked_select_out_npu</p>
</td>
</tr>
<tr id="row07509395124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949213161621"><a name="p949213161621"></a><a name="p949213161621"></a>535</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1550543618278"><a name="p1550543618278"></a><a name="p1550543618278"></a>masked_select</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p650613360274"><a name="p650613360274"></a><a name="p650613360274"></a>masked_select_npu</p>
</td>
</tr>
<tr id="row197501839181212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154921016224"><a name="p154921016224"></a><a name="p154921016224"></a>536</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3506203632713"><a name="p3506203632713"></a><a name="p3506203632713"></a>nonzero.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7506103682711"><a name="p7506103682711"></a><a name="p7506103682711"></a>nonzero_out_npu</p>
</td>
</tr>
<tr id="row1075017392123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18492181618220"><a name="p18492181618220"></a><a name="p18492181618220"></a>537</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p850617361273"><a name="p850617361273"></a><a name="p850617361273"></a>nonzero</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1950693617275"><a name="p1950693617275"></a><a name="p1950693617275"></a>nonzero_npu</p>
</td>
</tr>
<tr id="row1375017398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184927163211"><a name="p184927163211"></a><a name="p184927163211"></a>538</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p950613682715"><a name="p950613682715"></a><a name="p950613682715"></a>gather.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8506123612718"><a name="p8506123612718"></a><a name="p8506123612718"></a>gather_out_npu</p>
</td>
</tr>
<tr id="row0750739111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249316167213"><a name="p249316167213"></a><a name="p249316167213"></a>539</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13506113610274"><a name="p13506113610274"></a><a name="p13506113610274"></a>gather</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8506173611276"><a name="p8506173611276"></a><a name="p8506173611276"></a>gather_npu</p>
</td>
</tr>
<tr id="row13750939151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949317161626"><a name="p949317161626"></a><a name="p949317161626"></a>540</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12506536172718"><a name="p12506536172718"></a><a name="p12506536172718"></a>gather.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165060368272"><a name="p165060368272"></a><a name="p165060368272"></a>gather_out_npu</p>
</td>
</tr>
<tr id="row1175014398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20493171615215"><a name="p20493171615215"></a><a name="p20493171615215"></a>541</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p950620369274"><a name="p950620369274"></a><a name="p950620369274"></a>gather.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19506163672714"><a name="p19506163672714"></a><a name="p19506163672714"></a>gather_npu</p>
</td>
</tr>
<tr id="row2751163911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8493111611212"><a name="p8493111611212"></a><a name="p8493111611212"></a>542</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p250614368278"><a name="p250614368278"></a><a name="p250614368278"></a>addcmul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p350613613279"><a name="p350613613279"></a><a name="p350613613279"></a>addcmul_out_npu</p>
</td>
</tr>
<tr id="row4751113917122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849312167218"><a name="p849312167218"></a><a name="p849312167218"></a>543</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85061036112719"><a name="p85061036112719"></a><a name="p85061036112719"></a>addcmul</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p650763610279"><a name="p650763610279"></a><a name="p650763610279"></a>addcmul_npu</p>
</td>
</tr>
<tr id="row1875113398121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049315161020"><a name="p1049315161020"></a><a name="p1049315161020"></a>544</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10507173622718"><a name="p10507173622718"></a><a name="p10507173622718"></a>addcmul_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6507113617271"><a name="p6507113617271"></a><a name="p6507113617271"></a>addcmul_npu_</p>
</td>
</tr>
<tr id="row117511339141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20493141612212"><a name="p20493141612212"></a><a name="p20493141612212"></a>545</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1050793618275"><a name="p1050793618275"></a><a name="p1050793618275"></a>addcdiv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7507636112713"><a name="p7507636112713"></a><a name="p7507636112713"></a>addcdiv_out_npu</p>
</td>
</tr>
<tr id="row47513398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049321619219"><a name="p1049321619219"></a><a name="p1049321619219"></a>546</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p750715365276"><a name="p750715365276"></a><a name="p750715365276"></a>addcdiv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14507336192716"><a name="p14507336192716"></a><a name="p14507336192716"></a>addcdiv_npu</p>
</td>
</tr>
<tr id="row177517395121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649311161213"><a name="p649311161213"></a><a name="p649311161213"></a>547</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195075362273"><a name="p195075362273"></a><a name="p195075362273"></a>_triangular_solve_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55078366271"><a name="p55078366271"></a><a name="p55078366271"></a>_triangular_solve_helper_npu</p>
</td>
</tr>
<tr id="row2751193991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15493016624"><a name="p15493016624"></a><a name="p15493016624"></a>548</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105071236102719"><a name="p105071236102719"></a><a name="p105071236102719"></a>_symeig_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175071536142716"><a name="p175071536142716"></a><a name="p175071536142716"></a>_symeig_helper_npu</p>
</td>
</tr>
<tr id="row18751203991216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5493191618213"><a name="p5493191618213"></a><a name="p5493191618213"></a>549</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3507103632710"><a name="p3507103632710"></a><a name="p3507103632710"></a>_svd_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1150715368274"><a name="p1150715368274"></a><a name="p1150715368274"></a>_svd_helper_npu</p>
</td>
</tr>
<tr id="row875123941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249316161223"><a name="p1249316161223"></a><a name="p1249316161223"></a>550</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15071136182716"><a name="p15071136182716"></a><a name="p15071136182716"></a>qr.Q</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1650773620278"><a name="p1650773620278"></a><a name="p1650773620278"></a>qr_out_npu</p>
</td>
</tr>
<tr id="row275114391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049314161527"><a name="p2049314161527"></a><a name="p2049314161527"></a>551</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1550763615272"><a name="p1550763615272"></a><a name="p1550763615272"></a>qr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19507113614271"><a name="p19507113614271"></a><a name="p19507113614271"></a>qr_npu</p>
</td>
</tr>
<tr id="row17751123961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17493916625"><a name="p17493916625"></a><a name="p17493916625"></a>552</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95081136112720"><a name="p95081136112720"></a><a name="p95081136112720"></a>multinomial.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145089364270"><a name="p145089364270"></a><a name="p145089364270"></a>multinomial_out_npu</p>
</td>
</tr>
<tr id="row1475113393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10493111614214"><a name="p10493111614214"></a><a name="p10493111614214"></a>553</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13508193602714"><a name="p13508193602714"></a><a name="p13508193602714"></a>multinomial</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1950833611279"><a name="p1950833611279"></a><a name="p1950833611279"></a>multinomial_npu</p>
</td>
</tr>
<tr id="row1752839141213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104941016626"><a name="p104941016626"></a><a name="p104941016626"></a>554</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p205081936152719"><a name="p205081936152719"></a><a name="p205081936152719"></a>erfinv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0508636132712"><a name="p0508636132712"></a><a name="p0508636132712"></a>erfinv_npu</p>
</td>
</tr>
<tr id="row775233991216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1494191618216"><a name="p1494191618216"></a><a name="p1494191618216"></a>555</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1850883652717"><a name="p1850883652717"></a><a name="p1850883652717"></a>erfinv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18508236132713"><a name="p18508236132713"></a><a name="p18508236132713"></a>erfinv_npu_</p>
</td>
</tr>
<tr id="row3752183961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134941616122"><a name="p134941616122"></a><a name="p134941616122"></a>556</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6508113652713"><a name="p6508113652713"></a><a name="p6508113652713"></a>erfinv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75081636172719"><a name="p75081636172719"></a><a name="p75081636172719"></a>erfinv_out_npu</p>
</td>
</tr>
<tr id="row1275293918124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1549441619220"><a name="p1549441619220"></a><a name="p1549441619220"></a>557</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75084361271"><a name="p75084361271"></a><a name="p75084361271"></a>sign</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p650803662719"><a name="p650803662719"></a><a name="p650803662719"></a>sign_npu</p>
</td>
</tr>
<tr id="row1275214396124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194941161125"><a name="p194941161125"></a><a name="p194941161125"></a>558</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p650817369279"><a name="p650817369279"></a><a name="p650817369279"></a>sign_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3508193692710"><a name="p3508193692710"></a><a name="p3508193692710"></a>sign_npu_</p>
</td>
</tr>
<tr id="row1275223951213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13494151614215"><a name="p13494151614215"></a><a name="p13494151614215"></a>559</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p050883612712"><a name="p050883612712"></a><a name="p050883612712"></a>sign.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2508183614272"><a name="p2508183614272"></a><a name="p2508183614272"></a>sign_out_npu</p>
</td>
</tr>
<tr id="row1475273915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114944161621"><a name="p114944161621"></a><a name="p114944161621"></a>560</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15508736152717"><a name="p15508736152717"></a><a name="p15508736152717"></a>atan2.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1950843615278"><a name="p1950843615278"></a><a name="p1950843615278"></a>atan2_out_npu</p>
</td>
</tr>
<tr id="row1575273961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449481619218"><a name="p449481619218"></a><a name="p449481619218"></a>561</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1950893672720"><a name="p1950893672720"></a><a name="p1950893672720"></a>atan2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3508173620277"><a name="p3508173620277"></a><a name="p3508173620277"></a>atan2_npu</p>
</td>
</tr>
<tr id="row275283919128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74941016925"><a name="p74941016925"></a><a name="p74941016925"></a>562</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5509163682718"><a name="p5509163682718"></a><a name="p5509163682718"></a>lerp.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7509193622715"><a name="p7509193622715"></a><a name="p7509193622715"></a>lerp_out_npu</p>
</td>
</tr>
<tr id="row15752163931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249419161427"><a name="p249419161427"></a><a name="p249419161427"></a>563</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13509133619271"><a name="p13509133619271"></a><a name="p13509133619271"></a>lerp.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1250983652714"><a name="p1250983652714"></a><a name="p1250983652714"></a>lerp_out_npu</p>
</td>
</tr>
<tr id="row15752113921216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74941916727"><a name="p74941916727"></a><a name="p74941916727"></a>564</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8509123613274"><a name="p8509123613274"></a><a name="p8509123613274"></a>lerp.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7509193652719"><a name="p7509193652719"></a><a name="p7509193652719"></a>lerp_npu</p>
</td>
</tr>
<tr id="row375343912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184947161222"><a name="p184947161222"></a><a name="p184947161222"></a>565</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p850917364274"><a name="p850917364274"></a><a name="p850917364274"></a>lerp.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16509336172713"><a name="p16509336172713"></a><a name="p16509336172713"></a>lerp_npu</p>
</td>
</tr>
<tr id="row9753203991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149414160218"><a name="p149414160218"></a><a name="p149414160218"></a>566</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195091536172713"><a name="p195091536172713"></a><a name="p195091536172713"></a>fmod.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16509153642718"><a name="p16509153642718"></a><a name="p16509153642718"></a>fmod_out_npu</p>
</td>
</tr>
<tr id="row575313910120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249551610212"><a name="p1249551610212"></a><a name="p1249551610212"></a>567</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14509163642712"><a name="p14509163642712"></a><a name="p14509163642712"></a>fmod.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1550943615279"><a name="p1550943615279"></a><a name="p1550943615279"></a>fmod_npu</p>
</td>
</tr>
<tr id="row1675311393122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149510169212"><a name="p1149510169212"></a><a name="p1149510169212"></a>568</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p145096369277"><a name="p145096369277"></a><a name="p145096369277"></a>fmod.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7509143612270"><a name="p7509143612270"></a><a name="p7509143612270"></a>fmod_out_npu</p>
</td>
</tr>
<tr id="row12753193981217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149514161324"><a name="p1149514161324"></a><a name="p1149514161324"></a>569</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p135095366278"><a name="p135095366278"></a><a name="p135095366278"></a>fmod.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1450933622715"><a name="p1450933622715"></a><a name="p1450933622715"></a>fmod_npu</p>
</td>
</tr>
<tr id="row27537391124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849591611212"><a name="p849591611212"></a><a name="p849591611212"></a>570</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p050916369271"><a name="p050916369271"></a><a name="p050916369271"></a>remainder.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55091636152713"><a name="p55091636152713"></a><a name="p55091636152713"></a>remainder_out_npu</p>
</td>
</tr>
<tr id="row1753153911214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194950161529"><a name="p194950161529"></a><a name="p194950161529"></a>571</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1510133619271"><a name="p1510133619271"></a><a name="p1510133619271"></a>remainder.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85101336152718"><a name="p85101336152718"></a><a name="p85101336152718"></a>remainder_npu</p>
</td>
</tr>
<tr id="row8753163971217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949518161428"><a name="p949518161428"></a><a name="p949518161428"></a>572</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1651019368277"><a name="p1651019368277"></a><a name="p1651019368277"></a>remainder.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1851013615274"><a name="p1851013615274"></a><a name="p1851013615274"></a>remainder_out_npu</p>
</td>
</tr>
<tr id="row1875323910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184951516925"><a name="p184951516925"></a><a name="p184951516925"></a>573</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4510143616273"><a name="p4510143616273"></a><a name="p4510143616273"></a>remainder.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9510236192711"><a name="p9510236192711"></a><a name="p9510236192711"></a>remainder_npu</p>
</td>
</tr>
<tr id="row1775333911120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349512168210"><a name="p349512168210"></a><a name="p349512168210"></a>574</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12510136162711"><a name="p12510136162711"></a><a name="p12510136162711"></a>min.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1751019362272"><a name="p1751019362272"></a><a name="p1751019362272"></a>min_out_npu</p>
</td>
</tr>
<tr id="row10754139131211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4495141617213"><a name="p4495141617213"></a><a name="p4495141617213"></a>575</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18510203616279"><a name="p18510203616279"></a><a name="p18510203616279"></a>min.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15510183620273"><a name="p15510183620273"></a><a name="p15510183620273"></a>min_npu</p>
</td>
</tr>
<tr id="row127541139151210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149581615211"><a name="p149581615211"></a><a name="p149581615211"></a>576</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p751043652717"><a name="p751043652717"></a><a name="p751043652717"></a>min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12510193615278"><a name="p12510193615278"></a><a name="p12510193615278"></a>min_npu</p>
</td>
</tr>
<tr id="row17754113913120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144956165214"><a name="p144956165214"></a><a name="p144956165214"></a>577</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1951073616278"><a name="p1951073616278"></a><a name="p1951073616278"></a>max.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p155101236202712"><a name="p155101236202712"></a><a name="p155101236202712"></a>max_out_npu</p>
</td>
</tr>
<tr id="row13754639171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04954161425"><a name="p04954161425"></a><a name="p04954161425"></a>578</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7510103632713"><a name="p7510103632713"></a><a name="p7510103632713"></a>max.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13510336172716"><a name="p13510336172716"></a><a name="p13510336172716"></a>max_npu</p>
</td>
</tr>
<tr id="row18754143914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44951416822"><a name="p44951416822"></a><a name="p44951416822"></a>579</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1851023622718"><a name="p1851023622718"></a><a name="p1851023622718"></a>max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55106363271"><a name="p55106363271"></a><a name="p55106363271"></a>max_npu</p>
</td>
</tr>
<tr id="row1675463991214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104953161219"><a name="p104953161219"></a><a name="p104953161219"></a>580</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p051033610271"><a name="p051033610271"></a><a name="p051033610271"></a>median</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14510636192712"><a name="p14510636192712"></a><a name="p14510636192712"></a>median_npu</p>
</td>
</tr>
<tr id="row575410392129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13495131616220"><a name="p13495131616220"></a><a name="p13495131616220"></a>581</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19511036162717"><a name="p19511036162717"></a><a name="p19511036162717"></a>sort.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6511163692713"><a name="p6511163692713"></a><a name="p6511163692713"></a>sort_out_npu</p>
</td>
</tr>
<tr id="row157541139121219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164951160219"><a name="p164951160219"></a><a name="p164951160219"></a>582</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p851111360275"><a name="p851111360275"></a><a name="p851111360275"></a>sort</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p95111836152712"><a name="p95111836152712"></a><a name="p95111836152712"></a>sort_npu</p>
</td>
</tr>
<tr id="row1275423991213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1749516161921"><a name="p1749516161921"></a><a name="p1749516161921"></a>583</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6511133613273"><a name="p6511133613273"></a><a name="p6511133613273"></a>sort.dimname_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145114364277"><a name="p145114364277"></a><a name="p145114364277"></a>sort_out_npu</p>
</td>
</tr>
<tr id="row5754193914121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2495161616215"><a name="p2495161616215"></a><a name="p2495161616215"></a>584</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p205111836152719"><a name="p205111836152719"></a><a name="p205111836152719"></a>sort.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p951183610271"><a name="p951183610271"></a><a name="p951183610271"></a>sort_npu</p>
</td>
</tr>
<tr id="row18754113941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1749514161728"><a name="p1749514161728"></a><a name="p1749514161728"></a>585</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451113662716"><a name="p1451113662716"></a><a name="p1451113662716"></a>argsort</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0511153613273"><a name="p0511153613273"></a><a name="p0511153613273"></a>argsort_npu</p>
</td>
</tr>
<tr id="row1175517394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049571610214"><a name="p2049571610214"></a><a name="p2049571610214"></a>586</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p051112367277"><a name="p051112367277"></a><a name="p051112367277"></a>argsort.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18511136102713"><a name="p18511136102713"></a><a name="p18511136102713"></a>argsort_npu</p>
</td>
</tr>
<tr id="row47551239161216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15496816329"><a name="p15496816329"></a><a name="p15496816329"></a>587</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15511183613279"><a name="p15511183613279"></a><a name="p15511183613279"></a>topk.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75111636172715"><a name="p75111636172715"></a><a name="p75111636172715"></a>topk_out_npu</p>
</td>
</tr>
<tr id="row20755739121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144967161429"><a name="p144967161429"></a><a name="p144967161429"></a>588</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p651110361275"><a name="p651110361275"></a><a name="p651110361275"></a>topk</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5511113616272"><a name="p5511113616272"></a><a name="p5511113616272"></a>topk_npu</p>
</td>
</tr>
<tr id="row1675517394128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p549621620217"><a name="p549621620217"></a><a name="p549621620217"></a>589</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175111436142713"><a name="p175111436142713"></a><a name="p175111436142713"></a>all</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15119364279"><a name="p15119364279"></a><a name="p15119364279"></a>all_npu</p>
</td>
</tr>
<tr id="row17755163920126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149681614215"><a name="p149681614215"></a><a name="p149681614215"></a>590</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p851203612712"><a name="p851203612712"></a><a name="p851203612712"></a>any</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1051283611271"><a name="p1051283611271"></a><a name="p1051283611271"></a>any_npu</p>
</td>
</tr>
<tr id="row167551839111212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144963169210"><a name="p144963169210"></a><a name="p144963169210"></a>591</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p351217362273"><a name="p351217362273"></a><a name="p351217362273"></a>renorm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1151263692719"><a name="p1151263692719"></a><a name="p1151263692719"></a>renorm_out_npu</p>
</td>
</tr>
<tr id="row177559399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164968169216"><a name="p164968169216"></a><a name="p164968169216"></a>592</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16512103613273"><a name="p16512103613273"></a><a name="p16512103613273"></a>renorm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75121536192715"><a name="p75121536192715"></a><a name="p75121536192715"></a>renorm_npu</p>
</td>
</tr>
<tr id="row9755539121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349613161228"><a name="p349613161228"></a><a name="p349613161228"></a>593</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195125362274"><a name="p195125362274"></a><a name="p195125362274"></a>unfold</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17512103622713"><a name="p17512103622713"></a><a name="p17512103622713"></a>unfold</p>
</td>
</tr>
<tr id="row16755203919122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849611614213"><a name="p849611614213"></a><a name="p849611614213"></a>594</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1551293662717"><a name="p1551293662717"></a><a name="p1551293662717"></a>equal</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115121369274"><a name="p115121369274"></a><a name="p115121369274"></a>equal_npu</p>
</td>
</tr>
<tr id="row8755103913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349681613211"><a name="p349681613211"></a><a name="p349681613211"></a>595</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1751223662711"><a name="p1751223662711"></a><a name="p1751223662711"></a>pow.Tensor_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25121136142711"><a name="p25121136142711"></a><a name="p25121136142711"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row16755103912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349611161928"><a name="p349611161928"></a><a name="p349611161928"></a>596</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p751223619276"><a name="p751223619276"></a><a name="p751223619276"></a>pow.Tensor_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p155121636102717"><a name="p155121636102717"></a><a name="p155121636102717"></a>pow_npu</p>
</td>
</tr>
<tr id="row3755183901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8496151611215"><a name="p8496151611215"></a><a name="p8496151611215"></a>597</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175123361274"><a name="p175123361274"></a><a name="p175123361274"></a>pow.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175121636132712"><a name="p175121636132712"></a><a name="p175121636132712"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row1375613961210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13496111615215"><a name="p13496111615215"></a><a name="p13496111615215"></a>598</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11512143692720"><a name="p11512143692720"></a><a name="p11512143692720"></a>pow.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1451223620273"><a name="p1451223620273"></a><a name="p1451223620273"></a>pow_npu</p>
</td>
</tr>
<tr id="row1175623951217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1849615161424"><a name="p1849615161424"></a><a name="p1849615161424"></a>599</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p55121436112716"><a name="p55121436112716"></a><a name="p55121436112716"></a>normal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9513133672714"><a name="p9513133672714"></a><a name="p9513133672714"></a>normal_npu_</p>
</td>
</tr>
<tr id="row275613921217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44964161029"><a name="p44964161029"></a><a name="p44964161029"></a>600</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14513636192719"><a name="p14513636192719"></a><a name="p14513636192719"></a>normal.Tensor_float_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65135363274"><a name="p65135363274"></a><a name="p65135363274"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row275623991214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124966163217"><a name="p124966163217"></a><a name="p124966163217"></a>601</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16513173642713"><a name="p16513173642713"></a><a name="p16513173642713"></a>normal.Tensor_float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3513183652716"><a name="p3513183652716"></a><a name="p3513183652716"></a>normal_npu</p>
</td>
</tr>
<tr id="row17756123941211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349715161628"><a name="p349715161628"></a><a name="p349715161628"></a>602</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1251383617276"><a name="p1251383617276"></a><a name="p1251383617276"></a>normal.float_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p05131936202717"><a name="p05131936202717"></a><a name="p05131936202717"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row775611393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15497111618214"><a name="p15497111618214"></a><a name="p15497111618214"></a>603</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185134365274"><a name="p185134365274"></a><a name="p185134365274"></a>normal.float_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15513163618271"><a name="p15513163618271"></a><a name="p15513163618271"></a>normal_npu</p>
</td>
</tr>
<tr id="row17756183951214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17497141612216"><a name="p17497141612216"></a><a name="p17497141612216"></a>604</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11513123616278"><a name="p11513123616278"></a><a name="p11513123616278"></a>normal.Tensor_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3513436142710"><a name="p3513436142710"></a><a name="p3513436142710"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row197561439121218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1349712166211"><a name="p1349712166211"></a><a name="p1349712166211"></a>605</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1351312363278"><a name="p1351312363278"></a><a name="p1351312363278"></a>normal.Tensor_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1751303642716"><a name="p1751303642716"></a><a name="p1751303642716"></a>normal_npu</p>
</td>
</tr>
<tr id="row37563398122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p749720161529"><a name="p749720161529"></a><a name="p749720161529"></a>606</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p205131236172720"><a name="p205131236172720"></a><a name="p205131236172720"></a>normal.float_float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115132364274"><a name="p115132364274"></a><a name="p115132364274"></a>normal_npu</p>
</td>
</tr>
<tr id="row1075683921219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13497141617213"><a name="p13497141617213"></a><a name="p13497141617213"></a>607</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10513736162711"><a name="p10513736162711"></a><a name="p10513736162711"></a>normal.float_float_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1051333632713"><a name="p1051333632713"></a><a name="p1051333632713"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row975733971217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849761616210"><a name="p849761616210"></a><a name="p849761616210"></a>608</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175131336152712"><a name="p175131336152712"></a><a name="p175131336152712"></a>_addr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1851313612273"><a name="p1851313612273"></a><a name="p1851313612273"></a>_addr_npu</p>
</td>
</tr>
<tr id="row157571439151211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1949741610213"><a name="p1949741610213"></a><a name="p1949741610213"></a>609</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16514113692716"><a name="p16514113692716"></a><a name="p16514113692716"></a>_addr_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1651416363271"><a name="p1651416363271"></a><a name="p1651416363271"></a>_addr_npu_</p>
</td>
</tr>
<tr id="row075718391126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54977161324"><a name="p54977161324"></a><a name="p54977161324"></a>610</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p751483617275"><a name="p751483617275"></a><a name="p751483617275"></a>_addr.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1951415369274"><a name="p1951415369274"></a><a name="p1951415369274"></a>_addr_out_npu</p>
</td>
</tr>
<tr id="row197572391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3497116121"><a name="p3497116121"></a><a name="p3497116121"></a>611</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1651443616272"><a name="p1651443616272"></a><a name="p1651443616272"></a>_index_copy_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145145362273"><a name="p145145362273"></a><a name="p145145362273"></a>index_copy_npu_</p>
</td>
</tr>
<tr id="row15757173917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18497816524"><a name="p18497816524"></a><a name="p18497816524"></a>612</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185141636122716"><a name="p185141636122716"></a><a name="p185141636122716"></a>_cumsum</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7514113618279"><a name="p7514113618279"></a><a name="p7514113618279"></a>_cumsum_npu</p>
</td>
</tr>
<tr id="row9757039131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849781614219"><a name="p849781614219"></a><a name="p849781614219"></a>613</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125143363276"><a name="p125143363276"></a><a name="p125143363276"></a>_cumsum.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19514173642714"><a name="p19514173642714"></a><a name="p19514173642714"></a>_cumsum_out_npu</p>
</td>
</tr>
<tr id="row1757139191214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049716161224"><a name="p2049716161224"></a><a name="p2049716161224"></a>614</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15141436202712"><a name="p15141436202712"></a><a name="p15141436202712"></a>_cumprod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p851410360270"><a name="p851410360270"></a><a name="p851410360270"></a>_cumprod_npu</p>
</td>
</tr>
<tr id="row275716390126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649751617211"><a name="p1649751617211"></a><a name="p1649751617211"></a>615</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1851413366277"><a name="p1851413366277"></a><a name="p1851413366277"></a>_cumprod.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17514193616273"><a name="p17514193616273"></a><a name="p17514193616273"></a>_cumprod_out_npu</p>
</td>
</tr>
<tr id="row575717398122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5497316921"><a name="p5497316921"></a><a name="p5497316921"></a>616</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10514133692718"><a name="p10514133692718"></a><a name="p10514133692718"></a>_var</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18514103618277"><a name="p18514103618277"></a><a name="p18514103618277"></a>_var_npu</p>
</td>
</tr>
<tr id="row1275733901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174972161124"><a name="p174972161124"></a><a name="p174972161124"></a>617</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65141536172715"><a name="p65141536172715"></a><a name="p65141536172715"></a>_amp_non_finite_check_and_unscale_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20514133662716"><a name="p20514133662716"></a><a name="p20514133662716"></a>_amp_non_finite_check_and_unscale_npu_</p>
</td>
</tr>
<tr id="row475703981210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449771612210"><a name="p449771612210"></a><a name="p449771612210"></a>618</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45141136152716"><a name="p45141136152716"></a><a name="p45141136152716"></a>_cat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8515163614273"><a name="p8515163614273"></a><a name="p8515163614273"></a>_cat_npu</p>
</td>
</tr>
<tr id="row137581539161214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949761610212"><a name="p949761610212"></a><a name="p949761610212"></a>619</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p951516369277"><a name="p951516369277"></a><a name="p951516369277"></a>_cat.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18515173612275"><a name="p18515173612275"></a><a name="p18515173612275"></a>_cat_out_npu</p>
</td>
</tr>
<tr id="row12758113941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84971316729"><a name="p84971316729"></a><a name="p84971316729"></a>620</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p115151367272"><a name="p115151367272"></a><a name="p115151367272"></a>_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1551593652711"><a name="p1551593652711"></a><a name="p1551593652711"></a>_max_npu</p>
</td>
</tr>
<tr id="row167581739101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11497171619218"><a name="p11497171619218"></a><a name="p11497171619218"></a>621</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p135151436172712"><a name="p135151436172712"></a><a name="p135151436172712"></a>_max.max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p851513361273"><a name="p851513361273"></a><a name="p851513361273"></a>_max_out_npu</p>
</td>
</tr>
<tr id="row18758143931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649813161220"><a name="p649813161220"></a><a name="p649813161220"></a>622</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1051519365274"><a name="p1051519365274"></a><a name="p1051519365274"></a>_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p185151936112710"><a name="p185151936112710"></a><a name="p185151936112710"></a>_min_npu</p>
</td>
</tr>
<tr id="row77581939111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84985160212"><a name="p84985160212"></a><a name="p84985160212"></a>623</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p751514368279"><a name="p751514368279"></a><a name="p751514368279"></a>_min.min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16515336162715"><a name="p16515336162715"></a><a name="p16515336162715"></a>_min_out_npu</p>
</td>
</tr>
<tr id="row1575893910121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44980167211"><a name="p44980167211"></a><a name="p44980167211"></a>624</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8515113652716"><a name="p8515113652716"></a><a name="p8515113652716"></a>mse_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115151360274"><a name="p115151360274"></a><a name="p115151360274"></a>mse_loss_out_npu</p>
</td>
</tr>
<tr id="row675812396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174986164213"><a name="p174986164213"></a><a name="p174986164213"></a>625</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p351593611275"><a name="p351593611275"></a><a name="p351593611275"></a>mse_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151513692712"><a name="p151513692712"></a><a name="p151513692712"></a>mse_loss_npu</p>
</td>
</tr>
<tr id="row1375812391128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194985164213"><a name="p194985164213"></a><a name="p194985164213"></a>626</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0515103612717"><a name="p0515103612717"></a><a name="p0515103612717"></a>mse_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16515103619273"><a name="p16515103619273"></a><a name="p16515103619273"></a>mse_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row5758173920123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14498181611220"><a name="p14498181611220"></a><a name="p14498181611220"></a>627</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p651563620277"><a name="p651563620277"></a><a name="p651563620277"></a>mse_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9515113616272"><a name="p9515113616272"></a><a name="p9515113616272"></a>mse_loss_backward_npu</p>
</td>
</tr>
<tr id="row2758113911211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p049861611210"><a name="p049861611210"></a><a name="p049861611210"></a>628</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p751533614277"><a name="p751533614277"></a><a name="p751533614277"></a>l1_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1515133617275"><a name="p1515133617275"></a><a name="p1515133617275"></a>l1_loss_out_npu</p>
</td>
</tr>
<tr id="row975863921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34981116524"><a name="p34981116524"></a><a name="p34981116524"></a>629</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15515136122711"><a name="p15515136122711"></a><a name="p15515136122711"></a>l1_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2515103692715"><a name="p2515103692715"></a><a name="p2515103692715"></a>l1_loss_npu</p>
</td>
</tr>
<tr id="row3758339121211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p549818161224"><a name="p549818161224"></a><a name="p549818161224"></a>630</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11515153622718"><a name="p11515153622718"></a><a name="p11515153622718"></a>l1_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1851515369271"><a name="p1851515369271"></a><a name="p1851515369271"></a>l1_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row18759193912122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8498181611210"><a name="p8498181611210"></a><a name="p8498181611210"></a>631</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17515103612715"><a name="p17515103612715"></a><a name="p17515103612715"></a>l1_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451543682717"><a name="p451543682717"></a><a name="p451543682717"></a>l1_loss_backward_npu</p>
</td>
</tr>
<tr id="row5759103991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149881614210"><a name="p1149881614210"></a><a name="p1149881614210"></a>632</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p55152036152713"><a name="p55152036152713"></a><a name="p55152036152713"></a>multilabel_margin_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1951613362273"><a name="p1951613362273"></a><a name="p1951613362273"></a>multilabel_margin_loss_out_npu</p>
</td>
</tr>
<tr id="row11759163919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849819161024"><a name="p849819161024"></a><a name="p849819161024"></a>633</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2051653602719"><a name="p2051653602719"></a><a name="p2051653602719"></a>multilabel_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1951643617279"><a name="p1951643617279"></a><a name="p1951643617279"></a>multilabel_margin_loss_npu</p>
</td>
</tr>
<tr id="row13759133910129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54981716923"><a name="p54981716923"></a><a name="p54981716923"></a>634</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19516936102713"><a name="p19516936102713"></a><a name="p19516936102713"></a>multilabel_margin_loss_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1516193620271"><a name="p1516193620271"></a><a name="p1516193620271"></a>multilabel_margin_loss_forward_out_npu</p>
</td>
</tr>
<tr id="row57591039171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114983162219"><a name="p114983162219"></a><a name="p114983162219"></a>635</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105161036152710"><a name="p105161036152710"></a><a name="p105161036152710"></a>multilabel_margin_loss_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165167363272"><a name="p165167363272"></a><a name="p165167363272"></a>multilabel_margin_loss_forward_npu</p>
</td>
</tr>
<tr id="row20759193931213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10498111612210"><a name="p10498111612210"></a><a name="p10498111612210"></a>636</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1951623682713"><a name="p1951623682713"></a><a name="p1951623682713"></a>nll_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p851673642715"><a name="p851673642715"></a><a name="p851673642715"></a>nll_loss_out_npu</p>
</td>
</tr>
<tr id="row97594394129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9499111619217"><a name="p9499111619217"></a><a name="p9499111619217"></a>637</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2516536122719"><a name="p2516536122719"></a><a name="p2516536122719"></a>nll_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151618368279"><a name="p151618368279"></a><a name="p151618368279"></a>nll_loss_npu</p>
</td>
</tr>
<tr id="row8759339171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049912165210"><a name="p1049912165210"></a><a name="p1049912165210"></a>638</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p651612361277"><a name="p651612361277"></a><a name="p651612361277"></a>nll_loss_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15516536142714"><a name="p15516536142714"></a><a name="p15516536142714"></a>nll_loss_forward_out_npu</p>
</td>
</tr>
<tr id="row6759193961213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p049915161629"><a name="p049915161629"></a><a name="p049915161629"></a>639</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7516736132716"><a name="p7516736132716"></a><a name="p7516736132716"></a>nll_loss_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1516736192711"><a name="p1516736192711"></a><a name="p1516736192711"></a>nll_loss_forward_npu</p>
</td>
</tr>
<tr id="row9759339191211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449951618214"><a name="p449951618214"></a><a name="p449951618214"></a>640</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p135162036112712"><a name="p135162036112712"></a><a name="p135162036112712"></a>nll_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19516133612714"><a name="p19516133612714"></a><a name="p19516133612714"></a>nll_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row2759183916127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54999161522"><a name="p54999161522"></a><a name="p54999161522"></a>641</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1516136182712"><a name="p1516136182712"></a><a name="p1516136182712"></a>nll_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2516183612271"><a name="p2516183612271"></a><a name="p2516183612271"></a>nll_loss_backward_npu</p>
</td>
</tr>
<tr id="row18760123951219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14999161024"><a name="p14999161024"></a><a name="p14999161024"></a>642</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p205160362275"><a name="p205160362275"></a><a name="p205160362275"></a>nll_loss2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p851663662717"><a name="p851663662717"></a><a name="p851663662717"></a>nll_loss2d_out_npu</p>
</td>
</tr>
<tr id="row18760153913129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144991316028"><a name="p144991316028"></a><a name="p144991316028"></a>643</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1351611367279"><a name="p1351611367279"></a><a name="p1351611367279"></a>nll_loss2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75167367275"><a name="p75167367275"></a><a name="p75167367275"></a>nll_loss2d_npu</p>
</td>
</tr>
<tr id="row1776043931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249917161724"><a name="p249917161724"></a><a name="p249917161724"></a>644</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175161236172711"><a name="p175161236172711"></a><a name="p175161236172711"></a>nll_loss2d_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p651614368279"><a name="p651614368279"></a><a name="p651614368279"></a>nll_loss2d_forward_out_npu</p>
</td>
</tr>
<tr id="row9760113961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249917162213"><a name="p1249917162213"></a><a name="p1249917162213"></a>645</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p351614365279"><a name="p351614365279"></a><a name="p351614365279"></a>nll_loss2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1516113612273"><a name="p1516113612273"></a><a name="p1516113612273"></a>nll_loss2d_forward_npu</p>
</td>
</tr>
<tr id="row6760103981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9499141618217"><a name="p9499141618217"></a><a name="p9499141618217"></a>646</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11516113618271"><a name="p11516113618271"></a><a name="p11516113618271"></a>nll_loss2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16516936152712"><a name="p16516936152712"></a><a name="p16516936152712"></a>nll_loss2d_backward_out_npu</p>
</td>
</tr>
<tr id="row2760143971212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144990165211"><a name="p144990165211"></a><a name="p144990165211"></a>647</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p951619362277"><a name="p951619362277"></a><a name="p951619362277"></a>nll_loss2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p751603615272"><a name="p751603615272"></a><a name="p751603615272"></a>nll_loss2d_backward_npu</p>
</td>
</tr>
<tr id="row9760133981218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1499161610211"><a name="p1499161610211"></a><a name="p1499161610211"></a>648</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p251683619272"><a name="p251683619272"></a><a name="p251683619272"></a>smooth_l1_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3516153620278"><a name="p3516153620278"></a><a name="p3516153620278"></a>smooth_l1_loss_out_npu</p>
</td>
</tr>
<tr id="row1276053916122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p749918161328"><a name="p749918161328"></a><a name="p749918161328"></a>649</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1951714368276"><a name="p1951714368276"></a><a name="p1951714368276"></a>smooth_l1_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5517183612717"><a name="p5517183612717"></a><a name="p5517183612717"></a>smooth_l1_loss_npu</p>
</td>
</tr>
<tr id="row1076053919126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114990166220"><a name="p114990166220"></a><a name="p114990166220"></a>650</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45171536172714"><a name="p45171536172714"></a><a name="p45171536172714"></a>smooth_l1_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75171936102716"><a name="p75171936102716"></a><a name="p75171936102716"></a>smooth_l1_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row07605390120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44991816520"><a name="p44991816520"></a><a name="p44991816520"></a>651</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11517436132718"><a name="p11517436132718"></a><a name="p11517436132718"></a>smooth_l1_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p351723610276"><a name="p351723610276"></a><a name="p351723610276"></a>smooth_l1_loss_backward_npu</p>
</td>
</tr>
<tr id="row176093951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125001916829"><a name="p125001916829"></a><a name="p125001916829"></a>652</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6517183612274"><a name="p6517183612274"></a><a name="p6517183612274"></a>soft_margin_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1517236132717"><a name="p1517236132717"></a><a name="p1517236132717"></a>soft_margin_loss_out_npu</p>
</td>
</tr>
<tr id="row147611239111213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p165001616523"><a name="p165001616523"></a><a name="p165001616523"></a>653</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6517173602712"><a name="p6517173602712"></a><a name="p6517173602712"></a>soft_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p95176368273"><a name="p95176368273"></a><a name="p95176368273"></a>soft_margin_loss_npu</p>
</td>
</tr>
<tr id="row0761839171218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1950014161729"><a name="p1950014161729"></a><a name="p1950014161729"></a>654</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85171436142710"><a name="p85171436142710"></a><a name="p85171436142710"></a>soft_margin_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1451717361270"><a name="p1451717361270"></a><a name="p1451717361270"></a>soft_margin_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row147617390125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p85005168219"><a name="p85005168219"></a><a name="p85005168219"></a>655</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p251753632719"><a name="p251753632719"></a><a name="p251753632719"></a>soft_margin_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13517636112716"><a name="p13517636112716"></a><a name="p13517636112716"></a>soft_margin_loss_backward_npu</p>
</td>
</tr>
<tr id="row18761113941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p950031613212"><a name="p950031613212"></a><a name="p950031613212"></a>656</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16517143618279"><a name="p16517143618279"></a><a name="p16517143618279"></a>elu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14517133672710"><a name="p14517133672710"></a><a name="p14517133672710"></a>elu_out_npu</p>
</td>
</tr>
<tr id="row97611639151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1750012161220"><a name="p1750012161220"></a><a name="p1750012161220"></a>657</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12517203602711"><a name="p12517203602711"></a><a name="p12517203602711"></a>elu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2517163622712"><a name="p2517163622712"></a><a name="p2517163622712"></a>elu_npu</p>
</td>
</tr>
<tr id="row4761173931218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05001816822"><a name="p05001816822"></a><a name="p05001816822"></a>658</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1551773617272"><a name="p1551773617272"></a><a name="p1551773617272"></a>elu_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6517436112719"><a name="p6517436112719"></a><a name="p6517436112719"></a>elu_backward_out_npu</p>
</td>
</tr>
<tr id="row9761173917125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2500916321"><a name="p2500916321"></a><a name="p2500916321"></a>659</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12517136102719"><a name="p12517136102719"></a><a name="p12517136102719"></a>elu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165171936152715"><a name="p165171936152715"></a><a name="p165171936152715"></a>elu_backward_npu</p>
</td>
</tr>
<tr id="row137611839121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p550041617212"><a name="p550041617212"></a><a name="p550041617212"></a>660</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195171536102718"><a name="p195171536102718"></a><a name="p195171536102718"></a>elu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4517936102716"><a name="p4517936102716"></a><a name="p4517936102716"></a>elu_npu_</p>
</td>
</tr>
<tr id="row176153961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750071612214"><a name="p750071612214"></a><a name="p750071612214"></a>661</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12517183610270"><a name="p12517183610270"></a><a name="p12517183610270"></a>glu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17517173616279"><a name="p17517173616279"></a><a name="p17517173616279"></a>glu_out_npu</p>
</td>
</tr>
<tr id="row77611239121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7500101620218"><a name="p7500101620218"></a><a name="p7500101620218"></a>662</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p55176369277"><a name="p55176369277"></a><a name="p55176369277"></a>glu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p851703616273"><a name="p851703616273"></a><a name="p851703616273"></a>glu_npu</p>
</td>
</tr>
<tr id="row376123941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15005161023"><a name="p15005161023"></a><a name="p15005161023"></a>663</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19517103618279"><a name="p19517103618279"></a><a name="p19517103618279"></a>glu_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p851733652718"><a name="p851733652718"></a><a name="p851733652718"></a>glu_backward_out_npu</p>
</td>
</tr>
<tr id="row1976173916120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p350020161425"><a name="p350020161425"></a><a name="p350020161425"></a>664</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185176362274"><a name="p185176362274"></a><a name="p185176362274"></a>glu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p135176369277"><a name="p135176369277"></a><a name="p135176369277"></a>glu_backward_npu</p>
</td>
</tr>
<tr id="row11762339181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250017161027"><a name="p1250017161027"></a><a name="p1250017161027"></a>665</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8518143611275"><a name="p8518143611275"></a><a name="p8518143611275"></a>hardsigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65181436172720"><a name="p65181436172720"></a><a name="p65181436172720"></a>hardsigmoid_out_npu</p>
</td>
</tr>
<tr id="row117623394125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135000161828"><a name="p135000161828"></a><a name="p135000161828"></a>666</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1951883610278"><a name="p1951883610278"></a><a name="p1951883610278"></a>hardsigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11518113692714"><a name="p11518113692714"></a><a name="p11518113692714"></a>hardsigmoid_npu</p>
</td>
</tr>
<tr id="row57620396121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p55006166220"><a name="p55006166220"></a><a name="p55006166220"></a>667</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195181136102711"><a name="p195181136102711"></a><a name="p195181136102711"></a>hardsigmoid_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451810364274"><a name="p451810364274"></a><a name="p451810364274"></a>hardsigmoid_npu_</p>
</td>
</tr>
<tr id="row15762153919129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p850018161820"><a name="p850018161820"></a><a name="p850018161820"></a>668</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p25181136162715"><a name="p25181136162715"></a><a name="p25181136162715"></a>hardsigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p551819365276"><a name="p551819365276"></a><a name="p551819365276"></a>hardsigmoid_backward_npu</p>
</td>
</tr>
<tr id="row0762639141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19500916222"><a name="p19500916222"></a><a name="p19500916222"></a>669</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85181036112711"><a name="p85181036112711"></a><a name="p85181036112711"></a>hardtanh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2051883617279"><a name="p2051883617279"></a><a name="p2051883617279"></a>hardtanh_out_npu</p>
</td>
</tr>
<tr id="row7762239161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15001161211"><a name="p15001161211"></a><a name="p15001161211"></a>670</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17518636162716"><a name="p17518636162716"></a><a name="p17518636162716"></a>hardtanh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8518163622714"><a name="p8518163622714"></a><a name="p8518163622714"></a>hardtanh_npu</p>
</td>
</tr>
<tr id="row1776243931220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35003162219"><a name="p35003162219"></a><a name="p35003162219"></a>671</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1551819368279"><a name="p1551819368279"></a><a name="p1551819368279"></a>hardtanh_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1851819363274"><a name="p1851819363274"></a><a name="p1851819363274"></a>hardtanh_backward_out_npu</p>
</td>
</tr>
<tr id="row17621539171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450071618216"><a name="p1450071618216"></a><a name="p1450071618216"></a>672</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14518103632714"><a name="p14518103632714"></a><a name="p14518103632714"></a>hardtanh_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5518736102715"><a name="p5518736102715"></a><a name="p5518736102715"></a>hardtanh_backward_npu</p>
</td>
</tr>
<tr id="row3762123931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17501121613220"><a name="p17501121613220"></a><a name="p17501121613220"></a>673</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451833622712"><a name="p1451833622712"></a><a name="p1451833622712"></a>hardtanh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17518036162715"><a name="p17518036162715"></a><a name="p17518036162715"></a>hardtanh_npu_</p>
</td>
</tr>
<tr id="row1376293918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p205011116925"><a name="p205011116925"></a><a name="p205011116925"></a>674</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15181336182719"><a name="p15181336182719"></a><a name="p15181336182719"></a>leaky_relu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1551818366273"><a name="p1551818366273"></a><a name="p1551818366273"></a>leaky_relu_out_npu</p>
</td>
</tr>
<tr id="row117621239181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35018161521"><a name="p35018161521"></a><a name="p35018161521"></a>675</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451843682713"><a name="p1451843682713"></a><a name="p1451843682713"></a>leaky_relu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145181367274"><a name="p145181367274"></a><a name="p145181367274"></a>leaky_relu_npu</p>
</td>
</tr>
<tr id="row77631939131212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p185015162219"><a name="p185015162219"></a><a name="p185015162219"></a>676</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9518336122715"><a name="p9518336122715"></a><a name="p9518336122715"></a>leaky_relu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451893692716"><a name="p451893692716"></a><a name="p451893692716"></a>leaky_relu_backward_npu</p>
</td>
</tr>
<tr id="row1763183919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45011216427"><a name="p45011216427"></a><a name="p45011216427"></a>677</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3518173615272"><a name="p3518173615272"></a><a name="p3518173615272"></a>leaky_relu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16518536132715"><a name="p16518536132715"></a><a name="p16518536132715"></a>leaky_relu_npu_</p>
</td>
</tr>
<tr id="row11763113911215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7501216828"><a name="p7501216828"></a><a name="p7501216828"></a>678</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2518436182716"><a name="p2518436182716"></a><a name="p2518436182716"></a>log_sigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451810361278"><a name="p451810361278"></a><a name="p451810361278"></a>log_sigmoid_out_npu</p>
</td>
</tr>
<tr id="row177631039111214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750112163214"><a name="p750112163214"></a><a name="p750112163214"></a>679</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7518836172715"><a name="p7518836172715"></a><a name="p7518836172715"></a>log_sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p135182036192714"><a name="p135182036192714"></a><a name="p135182036192714"></a>log_sigmoid_npu</p>
</td>
</tr>
<tr id="row16763539181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4501016221"><a name="p4501016221"></a><a name="p4501016221"></a>680</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13518123692718"><a name="p13518123692718"></a><a name="p13518123692718"></a>log_sigmoid_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p155187367275"><a name="p155187367275"></a><a name="p155187367275"></a>log_sigmoid_forward_out_npu</p>
</td>
</tr>
<tr id="row17763939191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05011116527"><a name="p05011116527"></a><a name="p05011116527"></a>681</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451812363270"><a name="p1451812363270"></a><a name="p1451812363270"></a>log_sigmoid_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9518173615274"><a name="p9518173615274"></a><a name="p9518173615274"></a>log_sigmoid_forward_npu</p>
</td>
</tr>
<tr id="row27631039121220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8501171613213"><a name="p8501171613213"></a><a name="p8501171613213"></a>682</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13519536172712"><a name="p13519536172712"></a><a name="p13519536172712"></a>log_sigmoid_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14519183610272"><a name="p14519183610272"></a><a name="p14519183610272"></a>log_sigmoid_backward_out_npu</p>
</td>
</tr>
<tr id="row117631439111219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11501171616211"><a name="p11501171616211"></a><a name="p11501171616211"></a>683</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105190361273"><a name="p105190361273"></a><a name="p105190361273"></a>log_sigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p155195367273"><a name="p155195367273"></a><a name="p155195367273"></a>log_sigmoid_backward_npu</p>
</td>
</tr>
<tr id="row167636392122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p105011716824"><a name="p105011716824"></a><a name="p105011716824"></a>684</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1151923612279"><a name="p1151923612279"></a><a name="p1151923612279"></a>rrelu_with_noise.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1351973652719"><a name="p1351973652719"></a><a name="p1351973652719"></a>rrelu_with_noise_out_npu</p>
</td>
</tr>
<tr id="row77634398120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135018161020"><a name="p135018161020"></a><a name="p135018161020"></a>685</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195191136202720"><a name="p195191136202720"></a><a name="p195191136202720"></a>rrelu_with_noise</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8519163602717"><a name="p8519163602717"></a><a name="p8519163602717"></a>rrelu_with_noise_npu</p>
</td>
</tr>
<tr id="row876373921213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p165015161219"><a name="p165015161219"></a><a name="p165015161219"></a>686</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p165196369274"><a name="p165196369274"></a><a name="p165196369274"></a>rrelu_with_noise_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13519136162716"><a name="p13519136162716"></a><a name="p13519136162716"></a>rrelu_with_noise_backward_npu</p>
</td>
</tr>
<tr id="row157641939171215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16501816824"><a name="p16501816824"></a><a name="p16501816824"></a>687</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0519123652717"><a name="p0519123652717"></a><a name="p0519123652717"></a>rrelu_with_noise_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p751943615278"><a name="p751943615278"></a><a name="p751943615278"></a>rrelu_with_noise_npu_</p>
</td>
</tr>
<tr id="row676463913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25013163216"><a name="p25013163216"></a><a name="p25013163216"></a>688</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9519133620273"><a name="p9519133620273"></a><a name="p9519133620273"></a>softplus.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5519193613273"><a name="p5519193613273"></a><a name="p5519193613273"></a>softplus_out_npu</p>
</td>
</tr>
<tr id="row576412392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p55021161522"><a name="p55021161522"></a><a name="p55021161522"></a>689</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19519103617275"><a name="p19519103617275"></a><a name="p19519103617275"></a>softplus</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4519143692711"><a name="p4519143692711"></a><a name="p4519143692711"></a>softplus_npu</p>
</td>
</tr>
<tr id="row0764539201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250219163212"><a name="p1250219163212"></a><a name="p1250219163212"></a>690</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p851993611275"><a name="p851993611275"></a><a name="p851993611275"></a>softplus_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165192367272"><a name="p165192367272"></a><a name="p165192367272"></a>softplus_backward_out_npu</p>
</td>
</tr>
<tr id="row6764153914123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35027161327"><a name="p35027161327"></a><a name="p35027161327"></a>691</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11519193618274"><a name="p11519193618274"></a><a name="p11519193618274"></a>softplus_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p95198361279"><a name="p95198361279"></a><a name="p95198361279"></a>softplus_backward_npu</p>
</td>
</tr>
<tr id="row876453916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15021316426"><a name="p15021316426"></a><a name="p15021316426"></a>692</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p951903619271"><a name="p951903619271"></a><a name="p951903619271"></a>softshrink.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p185191036102719"><a name="p185191036102719"></a><a name="p185191036102719"></a>softshrink_out_npu</p>
</td>
</tr>
<tr id="row47641139131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12502216929"><a name="p12502216929"></a><a name="p12502216929"></a>693</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1751953662718"><a name="p1751953662718"></a><a name="p1751953662718"></a>softshrink</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p651983618273"><a name="p651983618273"></a><a name="p651983618273"></a>softshrink_npu</p>
</td>
</tr>
<tr id="row16764039191218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17502316825"><a name="p17502316825"></a><a name="p17502316825"></a>694</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p451973617272"><a name="p451973617272"></a><a name="p451973617272"></a>softshrink_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165191736152720"><a name="p165191736152720"></a><a name="p165191736152720"></a>softshrink_backward_out_npu</p>
</td>
</tr>
<tr id="row776420399123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4502171611217"><a name="p4502171611217"></a><a name="p4502171611217"></a>695</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9519536162715"><a name="p9519536162715"></a><a name="p9519536162715"></a>softshrink_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1751923632713"><a name="p1751923632713"></a><a name="p1751923632713"></a>softshrink_backward_npu</p>
</td>
</tr>
<tr id="row167641939141216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12502716729"><a name="p12502716729"></a><a name="p12502716729"></a>696</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155191436132713"><a name="p155191436132713"></a><a name="p155191436132713"></a>adaptive_avg_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15519113618275"><a name="p15519113618275"></a><a name="p15519113618275"></a>adaptive_avg_pool2d_out_npu</p>
</td>
</tr>
<tr id="row77645392124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250213166210"><a name="p1250213166210"></a><a name="p1250213166210"></a>697</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125191736192714"><a name="p125191736192714"></a><a name="p125191736192714"></a>adaptive_avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1451933632710"><a name="p1451933632710"></a><a name="p1451933632710"></a>adaptive_avg_pool2d_npu</p>
</td>
</tr>
<tr id="row1576483991213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35021316429"><a name="p35021316429"></a><a name="p35021316429"></a>698</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p451973610276"><a name="p451973610276"></a><a name="p451973610276"></a>_adaptive_avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13520436192714"><a name="p13520436192714"></a><a name="p13520436192714"></a>_adaptive_avg_pool2d_npu</p>
</td>
</tr>
<tr id="row1576515398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350214169210"><a name="p1350214169210"></a><a name="p1350214169210"></a>699</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75201236172717"><a name="p75201236172717"></a><a name="p75201236172717"></a>_adaptive_avg_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2520203612271"><a name="p2520203612271"></a><a name="p2520203612271"></a>adaptive_avg_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row47651139101216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19502161614214"><a name="p19502161614214"></a><a name="p19502161614214"></a>700</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1520636192715"><a name="p1520636192715"></a><a name="p1520636192715"></a>adaptive_avg_pool3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p352023632712"><a name="p352023632712"></a><a name="p352023632712"></a>adaptive_avg_pool3d_out_npu</p>
</td>
</tr>
<tr id="row13765939101211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16502171620211"><a name="p16502171620211"></a><a name="p16502171620211"></a>701</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p452043692719"><a name="p452043692719"></a><a name="p452043692719"></a>adaptive_avg_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14520836122716"><a name="p14520836122716"></a><a name="p14520836122716"></a>adaptive_avg_pool3d_npu</p>
</td>
</tr>
<tr id="row1976518395127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450217169212"><a name="p1450217169212"></a><a name="p1450217169212"></a>702</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p352016360278"><a name="p352016360278"></a><a name="p352016360278"></a>adaptive_avg_pool3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1552073692719"><a name="p1552073692719"></a><a name="p1552073692719"></a>adaptive_avg_pool3d_backward_out_npu</p>
</td>
</tr>
<tr id="row1476511392129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15025163218"><a name="p15025163218"></a><a name="p15025163218"></a>703</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155202368275"><a name="p155202368275"></a><a name="p155202368275"></a>adaptive_avg_pool3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8520153613270"><a name="p8520153613270"></a><a name="p8520153613270"></a>adaptive_avg_pool3d_backward_npu</p>
</td>
</tr>
<tr id="row0765173961215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145029161423"><a name="p145029161423"></a><a name="p145029161423"></a>704</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0520183692715"><a name="p0520183692715"></a><a name="p0520183692715"></a>adaptive_max_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1152093617271"><a name="p1152093617271"></a><a name="p1152093617271"></a>adaptive_max_pool2d_out_npu</p>
</td>
</tr>
<tr id="row0765163915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p65022161522"><a name="p65022161522"></a><a name="p65022161522"></a>705</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35201236192714"><a name="p35201236192714"></a><a name="p35201236192714"></a>adaptive_max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18520143617271"><a name="p18520143617271"></a><a name="p18520143617271"></a>adaptive_max_pool2d_npu</p>
</td>
</tr>
<tr id="row12765639121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125031016820"><a name="p125031016820"></a><a name="p125031016820"></a>706</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75204367277"><a name="p75204367277"></a><a name="p75204367277"></a>adaptive_max_pool2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1852011365279"><a name="p1852011365279"></a><a name="p1852011365279"></a>adaptive_max_pool2d_backward_out_npu</p>
</td>
</tr>
<tr id="row1476513931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45036166220"><a name="p45036166220"></a><a name="p45036166220"></a>707</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p145205361279"><a name="p145205361279"></a><a name="p145205361279"></a>adaptive_max_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1452043672713"><a name="p1452043672713"></a><a name="p1452043672713"></a>adaptive_max_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row976583971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125032016329"><a name="p125032016329"></a><a name="p125032016329"></a>708</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12520183614270"><a name="p12520183614270"></a><a name="p12520183614270"></a>avg_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p252023615275"><a name="p252023615275"></a><a name="p252023615275"></a>avg_pool2d_out_npu</p>
</td>
</tr>
<tr id="row2076512399126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p105031516926"><a name="p105031516926"></a><a name="p105031516926"></a>709</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p25201836152719"><a name="p25201836152719"></a><a name="p25201836152719"></a>avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1552013615279"><a name="p1552013615279"></a><a name="p1552013615279"></a>avg_pool2d_npu</p>
</td>
</tr>
<tr id="row87660397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155038161329"><a name="p155038161329"></a><a name="p155038161329"></a>710</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p452020360278"><a name="p452020360278"></a><a name="p452020360278"></a>avg_pool2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1552073622716"><a name="p1552073622716"></a><a name="p1552073622716"></a>avg_pool2d_backward_out_npu</p>
</td>
</tr>
<tr id="row876610396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18503216729"><a name="p18503216729"></a><a name="p18503216729"></a>711</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17520136202718"><a name="p17520136202718"></a><a name="p17520136202718"></a>avg_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1520103617272"><a name="p1520103617272"></a><a name="p1520103617272"></a>avg_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row12766339201217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135035167217"><a name="p135035167217"></a><a name="p135035167217"></a>712</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p752017368279"><a name="p752017368279"></a><a name="p752017368279"></a>avg_pool3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p155204369277"><a name="p155204369277"></a><a name="p155204369277"></a>avg_pool3d_out_npu</p>
</td>
</tr>
<tr id="row976693981213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19503201613220"><a name="p19503201613220"></a><a name="p19503201613220"></a>713</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2052083662714"><a name="p2052083662714"></a><a name="p2052083662714"></a>avg_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p155205369279"><a name="p155205369279"></a><a name="p155205369279"></a>avg_pool3d_npu</p>
</td>
</tr>
<tr id="row157661339191218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115032016329"><a name="p115032016329"></a><a name="p115032016329"></a>714</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p135208363278"><a name="p135208363278"></a><a name="p135208363278"></a>avg_pool3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14520153642716"><a name="p14520153642716"></a><a name="p14520153642716"></a>avg_pool3d_backward_out_npu</p>
</td>
</tr>
<tr id="row076693919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16503816826"><a name="p16503816826"></a><a name="p16503816826"></a>715</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1952183662711"><a name="p1952183662711"></a><a name="p1952183662711"></a>avg_pool3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1952183662716"><a name="p1952183662716"></a><a name="p1952183662716"></a>avg_pool3d_backward_npu</p>
</td>
</tr>
<tr id="row1876633961215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145031316127"><a name="p145031316127"></a><a name="p145031316127"></a>716</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p552193618272"><a name="p552193618272"></a><a name="p552193618272"></a>max_pool2d_with_indices.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165216360271"><a name="p165216360271"></a><a name="p165216360271"></a>max_pool2d_with_indices_out_npu</p>
</td>
</tr>
<tr id="row9767103941216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25033165211"><a name="p25033165211"></a><a name="p25033165211"></a>717</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75211236172718"><a name="p75211236172718"></a><a name="p75211236172718"></a>max_pool2d_with_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p852193616279"><a name="p852193616279"></a><a name="p852193616279"></a>max_pool2d_with_indices_npu</p>
</td>
</tr>
<tr id="row167671939131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2503116426"><a name="p2503116426"></a><a name="p2503116426"></a>718</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5521636112710"><a name="p5521636112710"></a><a name="p5521636112710"></a>max_pool2d_with_indices_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p252153617276"><a name="p252153617276"></a><a name="p252153617276"></a>max_pool2d_with_indices_backward_out_npu</p>
</td>
</tr>
<tr id="row1676714395121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6503316326"><a name="p6503316326"></a><a name="p6503316326"></a>719</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85219367273"><a name="p85219367273"></a><a name="p85219367273"></a>max_pool2d_with_indices_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17521173616278"><a name="p17521173616278"></a><a name="p17521173616278"></a>max_pool2d_with_indices_backward_npu</p>
</td>
</tr>
<tr id="row18149195017234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p175032169219"><a name="p175032169219"></a><a name="p175032169219"></a>720</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p352183613270"><a name="p352183613270"></a><a name="p352183613270"></a>max_pool3d_with_indices.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75211936102715"><a name="p75211936102715"></a><a name="p75211936102715"></a>max_pool3d_with_indices_out_npu</p>
</td>
</tr>
<tr id="row1614985042313"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7503816528"><a name="p7503816528"></a><a name="p7503816528"></a>721</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75211836152720"><a name="p75211836152720"></a><a name="p75211836152720"></a>max_pool3d_with_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115211536102716"><a name="p115211536102716"></a><a name="p115211536102716"></a>max_pool3d_with_indices_npu</p>
</td>
</tr>
<tr id="row17149115012238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1550401614213"><a name="p1550401614213"></a><a name="p1550401614213"></a>722</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35211636142715"><a name="p35211636142715"></a><a name="p35211636142715"></a>max_pool3d_with_indices_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p35217367275"><a name="p35217367275"></a><a name="p35217367275"></a>max_pool3d_with_indices_backward_out_npu</p>
</td>
</tr>
<tr id="row614965016234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1750491617211"><a name="p1750491617211"></a><a name="p1750491617211"></a>723</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7521203662715"><a name="p7521203662715"></a><a name="p7521203662715"></a>max_pool3d_with_indices_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p852173616271"><a name="p852173616271"></a><a name="p852173616271"></a>max_pool3d_with_indices_backward_npu</p>
</td>
</tr>
<tr id="row8149155011235"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5504121617216"><a name="p5504121617216"></a><a name="p5504121617216"></a>724</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14521173642713"><a name="p14521173642713"></a><a name="p14521173642713"></a>reflection_pad2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1152117366271"><a name="p1152117366271"></a><a name="p1152117366271"></a>reflection_pad2d_out_npu</p>
</td>
</tr>
<tr id="row914945052310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15048168211"><a name="p15048168211"></a><a name="p15048168211"></a>725</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18521203618270"><a name="p18521203618270"></a><a name="p18521203618270"></a>reflection_pad2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125215365273"><a name="p125215365273"></a><a name="p125215365273"></a>reflection_pad2d_npu</p>
</td>
</tr>
<tr id="row414875013236"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450413161926"><a name="p1450413161926"></a><a name="p1450413161926"></a>726</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75212368275"><a name="p75212368275"></a><a name="p75212368275"></a>replication_pad2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p152103615278"><a name="p152103615278"></a><a name="p152103615278"></a>replication_pad2d_out_npu</p>
</td>
</tr>
<tr id="row181481650152310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250416161226"><a name="p250416161226"></a><a name="p250416161226"></a>727</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35219367279"><a name="p35219367279"></a><a name="p35219367279"></a>replication_pad2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p552153615272"><a name="p552153615272"></a><a name="p552153615272"></a>replication_pad2d_npu</p>
</td>
</tr>
<tr id="row914810500234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p175042161925"><a name="p175042161925"></a><a name="p175042161925"></a>728</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13521153616276"><a name="p13521153616276"></a><a name="p13521153616276"></a>upsample_linear1d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75212363278"><a name="p75212363278"></a><a name="p75212363278"></a>upsample_linear1d_out_npu</p>
</td>
</tr>
<tr id="row51481550142317"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11504816824"><a name="p11504816824"></a><a name="p11504816824"></a>729</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19521153611276"><a name="p19521153611276"></a><a name="p19521153611276"></a>upsample_linear1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p252213642711"><a name="p252213642711"></a><a name="p252213642711"></a>upsample_linear1d_npu</p>
</td>
</tr>
<tr id="row101481250142314"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250471613220"><a name="p250471613220"></a><a name="p250471613220"></a>730</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12522123618273"><a name="p12522123618273"></a><a name="p12522123618273"></a>upsample_linear1d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p552263662718"><a name="p552263662718"></a><a name="p552263662718"></a>upsample_linear1d_backward_npu</p>
</td>
</tr>
<tr id="row91484505232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1950420161621"><a name="p1950420161621"></a><a name="p1950420161621"></a>731</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13522183611275"><a name="p13522183611275"></a><a name="p13522183611275"></a>upsample_bilinear2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1522536132720"><a name="p1522536132720"></a><a name="p1522536132720"></a>upsample_bilinear2d_out_npu</p>
</td>
</tr>
<tr id="row2148155019234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p95049161126"><a name="p95049161126"></a><a name="p95049161126"></a>732</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195225363279"><a name="p195225363279"></a><a name="p195225363279"></a>upsample_bilinear2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p952273617276"><a name="p952273617276"></a><a name="p952273617276"></a>upsample_bilinear2d_npu</p>
</td>
</tr>
<tr id="row151481250172312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45043161823"><a name="p45043161823"></a><a name="p45043161823"></a>733</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4522163614272"><a name="p4522163614272"></a><a name="p4522163614272"></a>upsample_bilinear2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0522143618279"><a name="p0522143618279"></a><a name="p0522143618279"></a>upsample_bilinear2d_backward_out_npu</p>
</td>
</tr>
<tr id="row214811500239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155048163216"><a name="p155048163216"></a><a name="p155048163216"></a>734</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175221362275"><a name="p175221362275"></a><a name="p175221362275"></a>upsample_bilinear2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1252283618276"><a name="p1252283618276"></a><a name="p1252283618276"></a>upsample_bilinear2d_backward_npu</p>
</td>
</tr>
<tr id="row9148450142310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p450431613211"><a name="p450431613211"></a><a name="p450431613211"></a>735</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1952213620275"><a name="p1952213620275"></a><a name="p1952213620275"></a>upsample_bicubic2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2052210364277"><a name="p2052210364277"></a><a name="p2052210364277"></a>upsample_bicubic2d_out_npu</p>
</td>
</tr>
<tr id="row914819503239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115041816427"><a name="p115041816427"></a><a name="p115041816427"></a>736</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p115221936132712"><a name="p115221936132712"></a><a name="p115221936132712"></a>upsample_bicubic2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8522153620276"><a name="p8522153620276"></a><a name="p8522153620276"></a>upsample_bicubic2d_npu</p>
</td>
</tr>
<tr id="row1514765042314"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p450510161726"><a name="p450510161726"></a><a name="p450510161726"></a>737</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0522143618271"><a name="p0522143618271"></a><a name="p0522143618271"></a>upsample_bicubic2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1522936132713"><a name="p1522936132713"></a><a name="p1522936132713"></a>upsample_bicubic2d_backward_out_npu</p>
</td>
</tr>
<tr id="row10147145019234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1850515161326"><a name="p1850515161326"></a><a name="p1850515161326"></a>738</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15522113642715"><a name="p15522113642715"></a><a name="p15522113642715"></a>upsample_bicubic2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14522036142710"><a name="p14522036142710"></a><a name="p14522036142710"></a>upsample_bicubic2d_backward_npu</p>
</td>
</tr>
<tr id="row131471350162312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14505916423"><a name="p14505916423"></a><a name="p14505916423"></a>739</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p165221336102717"><a name="p165221336102717"></a><a name="p165221336102717"></a>upsample_trilinear3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11522203672715"><a name="p11522203672715"></a><a name="p11522203672715"></a>upsample_trilinear3d_out_npu</p>
</td>
</tr>
<tr id="row514745013232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p950512161228"><a name="p950512161228"></a><a name="p950512161228"></a>740</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1852219361279"><a name="p1852219361279"></a><a name="p1852219361279"></a>upsample_trilinear3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125225360274"><a name="p125225360274"></a><a name="p125225360274"></a>upsample_trilinear3d_npu</p>
</td>
</tr>
<tr id="row16147115062311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25054161225"><a name="p25054161225"></a><a name="p25054161225"></a>741</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p652220365275"><a name="p652220365275"></a><a name="p652220365275"></a>upsample_trilinear3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175221036122712"><a name="p175221036122712"></a><a name="p175221036122712"></a>upsample_trilinear3d_backward_out_npu</p>
</td>
</tr>
<tr id="row1514755018239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p150501616218"><a name="p150501616218"></a><a name="p150501616218"></a>742</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10522133632712"><a name="p10522133632712"></a><a name="p10522133632712"></a>upsample_trilinear3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p152243620276"><a name="p152243620276"></a><a name="p152243620276"></a>upsample_trilinear3d_backward_npu</p>
</td>
</tr>
<tr id="row914712503239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p185052161625"><a name="p185052161625"></a><a name="p185052161625"></a>743</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18522236192719"><a name="p18522236192719"></a><a name="p18522236192719"></a>upsample_nearest1d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7522163611275"><a name="p7522163611275"></a><a name="p7522163611275"></a>upsample_nearest1d_out_npu</p>
</td>
</tr>
<tr id="row151473505232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16505716022"><a name="p16505716022"></a><a name="p16505716022"></a>744</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85221636112718"><a name="p85221636112718"></a><a name="p85221636112718"></a>upsample_nearest1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3522123613275"><a name="p3522123613275"></a><a name="p3522123613275"></a>upsample_nearest1d_npu</p>
</td>
</tr>
<tr id="row171471350182312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250518161825"><a name="p250518161825"></a><a name="p250518161825"></a>745</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18522103619275"><a name="p18522103619275"></a><a name="p18522103619275"></a>upsample_nearest1d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4522163611273"><a name="p4522163611273"></a><a name="p4522163611273"></a>upsample_nearest1d_backward_out_npu</p>
</td>
</tr>
<tr id="row12147150142310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3505151619219"><a name="p3505151619219"></a><a name="p3505151619219"></a>746</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p25221636172710"><a name="p25221636172710"></a><a name="p25221636172710"></a>upsample_nearest1d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13523036172711"><a name="p13523036172711"></a><a name="p13523036172711"></a>upsample_nearest1d_backward_npu</p>
</td>
</tr>
<tr id="row101472050152318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350511163213"><a name="p1350511163213"></a><a name="p1350511163213"></a>747</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6523036182714"><a name="p6523036182714"></a><a name="p6523036182714"></a>upsample_nearest2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15523173618272"><a name="p15523173618272"></a><a name="p15523173618272"></a>upsample_nearest2d_out_npu</p>
</td>
</tr>
<tr id="row31463506231"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350517161214"><a name="p1350517161214"></a><a name="p1350517161214"></a>748</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95237369276"><a name="p95237369276"></a><a name="p95237369276"></a>upsample_nearest2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16523173610273"><a name="p16523173610273"></a><a name="p16523173610273"></a>upsample_nearest2d_npu</p>
</td>
</tr>
<tr id="row1814612508238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5505416525"><a name="p5505416525"></a><a name="p5505416525"></a>749</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2523193652717"><a name="p2523193652717"></a><a name="p2523193652717"></a>upsample_nearest2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195231436162718"><a name="p195231436162718"></a><a name="p195231436162718"></a>upsample_nearest2d_backward_out_npu</p>
</td>
</tr>
<tr id="row714614509238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9505201610211"><a name="p9505201610211"></a><a name="p9505201610211"></a>750</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p252343612277"><a name="p252343612277"></a><a name="p252343612277"></a>upsample_nearest2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15523236162711"><a name="p15523236162711"></a><a name="p15523236162711"></a>upsample_nearest2d_backward_npu</p>
</td>
</tr>
<tr id="row1714605042318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9505171614218"><a name="p9505171614218"></a><a name="p9505171614218"></a>751</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45238365273"><a name="p45238365273"></a><a name="p45238365273"></a>upsample_nearest3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6523103619271"><a name="p6523103619271"></a><a name="p6523103619271"></a>upsample_nearest3d_out_npu</p>
</td>
</tr>
<tr id="row111461750132318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125051161024"><a name="p125051161024"></a><a name="p125051161024"></a>752</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1052317367279"><a name="p1052317367279"></a><a name="p1052317367279"></a>upsample_nearest3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p852319361276"><a name="p852319361276"></a><a name="p852319361276"></a>upsample_nearest3d_npu</p>
</td>
</tr>
<tr id="row7146185018238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250518161620"><a name="p250518161620"></a><a name="p250518161620"></a>753</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35233366277"><a name="p35233366277"></a><a name="p35233366277"></a>upsample_nearest3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p952312360278"><a name="p952312360278"></a><a name="p952312360278"></a>upsample_nearest3d_backward_out_npu</p>
</td>
</tr>
<tr id="row1514675082318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11505191612218"><a name="p11505191612218"></a><a name="p11505191612218"></a>754</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11523133682710"><a name="p11523133682710"></a><a name="p11523133682710"></a>upsample_nearest3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1152303612279"><a name="p1152303612279"></a><a name="p1152303612279"></a>upsample_nearest3d_backward_npu</p>
</td>
</tr>
<tr id="row1814645062311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19505131610215"><a name="p19505131610215"></a><a name="p19505131610215"></a>755</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p552315366273"><a name="p552315366273"></a><a name="p552315366273"></a>sigmoid_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18523123652710"><a name="p18523123652710"></a><a name="p18523123652710"></a>sigmoid_backward_out_npu</p>
</td>
</tr>
<tr id="row12146135072311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19505216728"><a name="p19505216728"></a><a name="p19505216728"></a>756</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1252318369275"><a name="p1252318369275"></a><a name="p1252318369275"></a>sigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19523203672711"><a name="p19523203672711"></a><a name="p19523203672711"></a>sigmoid_backward_npu</p>
</td>
</tr>
<tr id="row1214625011237"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750511614212"><a name="p750511614212"></a><a name="p750511614212"></a>757</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85231536182719"><a name="p85231536182719"></a><a name="p85231536182719"></a>tanh_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5523103611271"><a name="p5523103611271"></a><a name="p5523103611271"></a>tanh_backward_out_npu</p>
</td>
</tr>
<tr id="row14146155022318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7505101616219"><a name="p7505101616219"></a><a name="p7505101616219"></a>758</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45236362276"><a name="p45236362276"></a><a name="p45236362276"></a>tanh_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1052313610276"><a name="p1052313610276"></a><a name="p1052313610276"></a>tanh_backward_npu</p>
</td>
</tr>
<tr id="row12145250202315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5506141611210"><a name="p5506141611210"></a><a name="p5506141611210"></a>759</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105231436182710"><a name="p105231436182710"></a><a name="p105231436182710"></a>slow_conv_transpose2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125231836102715"><a name="p125231836102715"></a><a name="p125231836102715"></a>slow_conv_transpose2d_out_npu</p>
</td>
</tr>
<tr id="row19145125011236"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p85061716126"><a name="p85061716126"></a><a name="p85061716126"></a>760</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2523123612272"><a name="p2523123612272"></a><a name="p2523123612272"></a>slow_conv_transpose2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p152319366271"><a name="p152319366271"></a><a name="p152319366271"></a>slow_conv_transpose2d_npu</p>
</td>
</tr>
<tr id="row16145205013238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20506316325"><a name="p20506316325"></a><a name="p20506316325"></a>761</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p152383616272"><a name="p152383616272"></a><a name="p152383616272"></a>slow_conv_transpose2d_backward.grad_output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6524436132710"><a name="p6524436132710"></a><a name="p6524436132710"></a>slow_conv_transpose2d_backward_out_npu</p>
</td>
</tr>
<tr id="row1914555052319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155062016921"><a name="p155062016921"></a><a name="p155062016921"></a>762</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p952493618277"><a name="p952493618277"></a><a name="p952493618277"></a>slow_conv_transpose2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195241136112720"><a name="p195241136112720"></a><a name="p195241136112720"></a>slow_conv_transpose2d_backward_npu</p>
</td>
</tr>
<tr id="row114511508237"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12506316228"><a name="p12506316228"></a><a name="p12506316228"></a>763</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3524123618277"><a name="p3524123618277"></a><a name="p3524123618277"></a>thnn_conv2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4524143672716"><a name="p4524143672716"></a><a name="p4524143672716"></a>thnn_conv2d_out_npu</p>
</td>
</tr>
<tr id="row71456502232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05061161223"><a name="p05061161223"></a><a name="p05061161223"></a>764</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12524143692719"><a name="p12524143692719"></a><a name="p12524143692719"></a>thnn_conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19524636132714"><a name="p19524636132714"></a><a name="p19524636132714"></a>thnn_conv2d_npu</p>
</td>
</tr>
<tr id="row11145115062319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1150651618213"><a name="p1150651618213"></a><a name="p1150651618213"></a>765</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95241936162710"><a name="p95241936162710"></a><a name="p95241936162710"></a>thnn_conv2d_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1452416362275"><a name="p1452416362275"></a><a name="p1452416362275"></a>thnn_conv2d_forward_out_npu</p>
</td>
</tr>
<tr id="row61451350172318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10506151611212"><a name="p10506151611212"></a><a name="p10506151611212"></a>766</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20524113619278"><a name="p20524113619278"></a><a name="p20524113619278"></a>thnn_conv2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10524183613271"><a name="p10524183613271"></a><a name="p10524183613271"></a>thnn_conv2d_forward_npu</p>
</td>
</tr>
<tr id="row10145115042317"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p150612162028"><a name="p150612162028"></a><a name="p150612162028"></a>767</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9524153617272"><a name="p9524153617272"></a><a name="p9524153617272"></a>thnn_conv2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p252473613273"><a name="p252473613273"></a><a name="p252473613273"></a>thnn_conv2d_backward_npu</p>
</td>
</tr>
<tr id="row19145165022315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125067162029"><a name="p125067162029"></a><a name="p125067162029"></a>768</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75241136152716"><a name="p75241136152716"></a><a name="p75241136152716"></a>thnn_conv_depthwise2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16524736152715"><a name="p16524736152715"></a><a name="p16524736152715"></a>thnn_conv_depthwise2d_out_npu</p>
</td>
</tr>
<tr id="row1314495014234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19506816928"><a name="p19506816928"></a><a name="p19506816928"></a>769</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1252417367279"><a name="p1252417367279"></a><a name="p1252417367279"></a>thnn_conv_depthwise2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16524153632720"><a name="p16524153632720"></a><a name="p16524153632720"></a>thnn_conv_depthwise2d_npu</p>
</td>
</tr>
<tr id="row4144750152311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2506616229"><a name="p2506616229"></a><a name="p2506616229"></a>770</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5524173610271"><a name="p5524173610271"></a><a name="p5524173610271"></a>thnn_conv_depthwise2d_forward.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3524133632719"><a name="p3524133632719"></a><a name="p3524133632719"></a>thnn_conv_depthwise2d_forward_out_npu</p>
</td>
</tr>
<tr id="row71441550172312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2506141618216"><a name="p2506141618216"></a><a name="p2506141618216"></a>771</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7524153622711"><a name="p7524153622711"></a><a name="p7524153622711"></a>thnn_conv_depthwise2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1352483632718"><a name="p1352483632718"></a><a name="p1352483632718"></a>thnn_conv_depthwise2d_forward_npu</p>
</td>
</tr>
<tr id="row1714475010231"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1550616163213"><a name="p1550616163213"></a><a name="p1550616163213"></a>772</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18524736192718"><a name="p18524736192718"></a><a name="p18524736192718"></a>thnn_conv_depthwise2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17524123610276"><a name="p17524123610276"></a><a name="p17524123610276"></a>thnn_conv_depthwise2d_backward_out_npu</p>
</td>
</tr>
<tr id="row514435052316"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p95064161821"><a name="p95064161821"></a><a name="p95064161821"></a>773</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1552483617272"><a name="p1552483617272"></a><a name="p1552483617272"></a>thnn_conv_depthwise2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195241036202710"><a name="p195241036202710"></a><a name="p195241036202710"></a>thnn_conv_depthwise2d_backward_npu</p>
</td>
</tr>
<tr id="row1514410509239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p185073162220"><a name="p185073162220"></a><a name="p185073162220"></a>774</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7524153617278"><a name="p7524153617278"></a><a name="p7524153617278"></a>slow_conv3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1352416368273"><a name="p1352416368273"></a><a name="p1352416368273"></a>slow_conv3d_out_npu</p>
</td>
</tr>
<tr id="row161441950202312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7507181619213"><a name="p7507181619213"></a><a name="p7507181619213"></a>775</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1052417363278"><a name="p1052417363278"></a><a name="p1052417363278"></a>slow_conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19524103692715"><a name="p19524103692715"></a><a name="p19524103692715"></a>slow_conv3d_npu</p>
</td>
</tr>
<tr id="row201444501230"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p650712164215"><a name="p650712164215"></a><a name="p650712164215"></a>776</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45242036202713"><a name="p45242036202713"></a><a name="p45242036202713"></a>slow_conv3d_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p352423612711"><a name="p352423612711"></a><a name="p352423612711"></a>slow_conv3d_forward_out_npu</p>
</td>
</tr>
<tr id="row111441350162315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115073162027"><a name="p115073162027"></a><a name="p115073162027"></a>777</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1652453619271"><a name="p1652453619271"></a><a name="p1652453619271"></a>slow_conv3d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p552413682710"><a name="p552413682710"></a><a name="p552413682710"></a>slow_conv3d_forward_npu</p>
</td>
</tr>
<tr id="row1914495012313"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p75072167211"><a name="p75072167211"></a><a name="p75072167211"></a>778</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18525836112712"><a name="p18525836112712"></a><a name="p18525836112712"></a>slow_conv_dilated2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p452593652715"><a name="p452593652715"></a><a name="p452593652715"></a>slow_conv_dilated2d_npu</p>
</td>
</tr>
<tr id="row91431350182319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115077161821"><a name="p115077161821"></a><a name="p115077161821"></a>779</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7525193652719"><a name="p7525193652719"></a><a name="p7525193652719"></a>slow_conv_dilated2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3525143619277"><a name="p3525143619277"></a><a name="p3525143619277"></a>slow_conv_dilated2d_backward_npu</p>
</td>
</tr>
<tr id="row16143150132315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p55075164215"><a name="p55075164215"></a><a name="p55075164215"></a>780</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p152518367274"><a name="p152518367274"></a><a name="p152518367274"></a>col2im.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175251136102717"><a name="p175251136102717"></a><a name="p175251136102717"></a>im2col_backward_out_npu</p>
</td>
</tr>
<tr id="row840154544916"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1341154518498"><a name="p1341154518498"></a><a name="p1341154518498"></a>781</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105256366270"><a name="p105256366270"></a><a name="p105256366270"></a>col2im</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p135257362272"><a name="p135257362272"></a><a name="p135257362272"></a>im2col_backward_npu</p>
</td>
</tr>
<tr id="row1279874816495"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p979824820499"><a name="p979824820499"></a><a name="p979824820499"></a>782</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45258360279"><a name="p45258360279"></a><a name="p45258360279"></a>col2im_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1152553611278"><a name="p1152553611278"></a><a name="p1152553611278"></a>im2col_out_npu</p>
</td>
</tr>
<tr id="row78031655104915"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18031855144911"><a name="p18031855144911"></a><a name="p18031855144911"></a>783</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19525636122719"><a name="p19525636122719"></a><a name="p19525636122719"></a>col2im_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16525183642716"><a name="p16525183642716"></a><a name="p16525183642716"></a>im2col_npu</p>
</td>
</tr>
<tr id="row19157115292316"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p161574523236"><a name="p161574523236"></a><a name="p161574523236"></a>784</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16525536152713"><a name="p16525536152713"></a><a name="p16525536152713"></a>im2col.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165251036132714"><a name="p165251036132714"></a><a name="p165251036132714"></a>im2col_out_npu</p>
</td>
</tr>
<tr id="row72331655162312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p142331855182311"><a name="p142331855182311"></a><a name="p142331855182311"></a>785</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10525836142718"><a name="p10525836142718"></a><a name="p10525836142718"></a>im2col</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2525143642713"><a name="p2525143642713"></a><a name="p2525143642713"></a>im2col_npu</p>
</td>
</tr>
<tr id="row644295816233"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p644245813239"><a name="p644245813239"></a><a name="p644245813239"></a>786</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p452515362278"><a name="p452515362278"></a><a name="p452515362278"></a>im2col_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17525173632714"><a name="p17525173632714"></a><a name="p17525173632714"></a>im2col_backward_out_npu</p>
</td>
</tr>
<tr id="row1226016262418"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p02603212245"><a name="p02603212245"></a><a name="p02603212245"></a>787</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13525113672719"><a name="p13525113672719"></a><a name="p13525113672719"></a>im2col_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p185251436202716"><a name="p185251436202716"></a><a name="p185251436202716"></a>im2col_backward_npu</p>
</td>
</tr>
<tr id="row78092782413"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1280910711240"><a name="p1280910711240"></a><a name="p1280910711240"></a>788</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p25251369274"><a name="p25251369274"></a><a name="p25251369274"></a>isfinite</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1952518362277"><a name="p1952518362277"></a><a name="p1952518362277"></a>isfinite_npu</p>
</td>
</tr>
</tbody>
</table>

<h2 id="PyTorch昇腾自定义算子.md">PyTorch昇腾自定义算子</h2>

<a name="table336910472136"></a>
<table><thead align="left"><tr id="row145917478132"><th class="cellrowborder" valign="top" width="8.334944884935215%" id="mcps1.1.4.1.1"><p id="p1055148824"><a name="p1055148824"></a><a name="p1055148824"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="46.954167472442464%" id="mcps1.1.4.1.2"><p id="p0459204781319"><a name="p0459204781319"></a><a name="p0459204781319"></a>PyTorch 算子（由昇腾开发）</p>
</th>
<th class="cellrowborder" valign="top" width="44.71088764262232%" id="mcps1.1.4.1.3"><p id="p1145994714134"><a name="p1145994714134"></a><a name="p1145994714134"></a>昇腾适配算子</p>
</th>
</tr>
</thead>
<tbody><tr id="row1459447181319"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p33711491045"><a name="p33711491045"></a><a name="p33711491045"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p17912038162814"><a name="p17912038162814"></a><a name="p17912038162814"></a>npu_convolution_transpose</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p199121538112819"><a name="p199121538112819"></a><a name="p199121538112819"></a>npu_convolution_transpose</p>
</td>
</tr>
<tr id="row345954751320"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1937159446"><a name="p1937159446"></a><a name="p1937159446"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1291212384284"><a name="p1291212384284"></a><a name="p1291212384284"></a>npu_conv_transpose2d</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p17912123882817"><a name="p17912123882817"></a><a name="p17912123882817"></a>conv_transpose2d_npu</p>
</td>
</tr>
<tr id="row645954711320"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p103711291640"><a name="p103711291640"></a><a name="p103711291640"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p12913193811283"><a name="p12913193811283"></a><a name="p12913193811283"></a>npu_convolution_transpose_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p10913183813283"><a name="p10913183813283"></a><a name="p10913183813283"></a>npu_convolution_transpose_backward</p>
</td>
</tr>
<tr id="row104591947131315"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p15371791346"><a name="p15371791346"></a><a name="p15371791346"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p39131138162813"><a name="p39131138162813"></a><a name="p39131138162813"></a>npu_conv_transpose2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p15913183892818"><a name="p15913183892818"></a><a name="p15913183892818"></a>conv_transpose2d_backward_npu</p>
</td>
</tr>
<tr id="row17459124711134"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p12371109442"><a name="p12371109442"></a><a name="p12371109442"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p3913193812282"><a name="p3913193812282"></a><a name="p3913193812282"></a>npu_conv_transpose3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p991303812285"><a name="p991303812285"></a><a name="p991303812285"></a>conv_transpose3d_backward_npu</p>
</td>
</tr>
<tr id="row16459847161315"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p537119916419"><a name="p537119916419"></a><a name="p537119916419"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p99131385288"><a name="p99131385288"></a><a name="p99131385288"></a>npu_convolution</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p17913123882820"><a name="p17913123882820"></a><a name="p17913123882820"></a>npu_convolution</p>
</td>
</tr>
<tr id="row1145915478138"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p103711991749"><a name="p103711991749"></a><a name="p103711991749"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p17913153814281"><a name="p17913153814281"></a><a name="p17913153814281"></a>npu_convolution_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p139134387285"><a name="p139134387285"></a><a name="p139134387285"></a>npu_convolution_backward</p>
</td>
</tr>
<tr id="row14606476135"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p163721691444"><a name="p163721691444"></a><a name="p163721691444"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p9913738102816"><a name="p9913738102816"></a><a name="p9913738102816"></a>npu_convolution_double_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p17913203802813"><a name="p17913203802813"></a><a name="p17913203802813"></a>npu_convolution_double_backward</p>
</td>
</tr>
<tr id="row446084710139"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p18372159142"><a name="p18372159142"></a><a name="p18372159142"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p9913113810287"><a name="p9913113810287"></a><a name="p9913113810287"></a>npu_conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p491317386285"><a name="p491317386285"></a><a name="p491317386285"></a>conv2d_npu</p>
</td>
</tr>
<tr id="row04607478133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p113729919417"><a name="p113729919417"></a><a name="p113729919417"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1591311385281"><a name="p1591311385281"></a><a name="p1591311385281"></a>npu_conv2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1991393862812"><a name="p1991393862812"></a><a name="p1991393862812"></a>conv2d_out_npu</p>
</td>
</tr>
<tr id="row9460347191318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5372499412"><a name="p5372499412"></a><a name="p5372499412"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p17913038142818"><a name="p17913038142818"></a><a name="p17913038142818"></a>npu_conv2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p10913338152814"><a name="p10913338152814"></a><a name="p10913338152814"></a>conv2d_backward_npu</p>
</td>
</tr>
<tr id="row2460174710139"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p937219918416"><a name="p937219918416"></a><a name="p937219918416"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p691343832812"><a name="p691343832812"></a><a name="p691343832812"></a>npu_conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p10913338172816"><a name="p10913338172816"></a><a name="p10913338172816"></a>conv3d_npu</p>
</td>
</tr>
<tr id="row2046034712131"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p19372794415"><a name="p19372794415"></a><a name="p19372794415"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p9913238152814"><a name="p9913238152814"></a><a name="p9913238152814"></a>npu_conv3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1191323818286"><a name="p1191323818286"></a><a name="p1191323818286"></a>conv3d_out_npu</p>
</td>
</tr>
<tr id="row246010470133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p437229345"><a name="p437229345"></a><a name="p437229345"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p79131538152810"><a name="p79131538152810"></a><a name="p79131538152810"></a>npu_conv3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p17913123872812"><a name="p17913123872812"></a><a name="p17913123872812"></a>conv3d_backward_npu</p>
</td>
</tr>
<tr id="row1246074751311"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p113721095410"><a name="p113721095410"></a><a name="p113721095410"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p5913123852810"><a name="p5913123852810"></a><a name="p5913123852810"></a>one_</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p179131738122810"><a name="p179131738122810"></a><a name="p179131738122810"></a>one_npu_</p>
</td>
</tr>
<tr id="row546074711139"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p8372994413"><a name="p8372994413"></a><a name="p8372994413"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1591353813285"><a name="p1591353813285"></a><a name="p1591353813285"></a>npu_sort_v2.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1491383816288"><a name="p1491383816288"></a><a name="p1491383816288"></a>sort_without_indices_out_npu</p>
</td>
</tr>
<tr id="row204603471134"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p183721491943"><a name="p183721491943"></a><a name="p183721491943"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p129133382283"><a name="p129133382283"></a><a name="p129133382283"></a>npu_sort_v2</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p391315387289"><a name="p391315387289"></a><a name="p391315387289"></a>sort_without_indices_npu</p>
</td>
</tr>
<tr id="row18460144714136"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p193721891744"><a name="p193721891744"></a><a name="p193721891744"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p12913038202813"><a name="p12913038202813"></a><a name="p12913038202813"></a>npu_format_cast</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p199131388287"><a name="p199131388287"></a><a name="p199131388287"></a>format_cast_npu</p>
</td>
</tr>
<tr id="row1460247101318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p537239842"><a name="p537239842"></a><a name="p537239842"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p19914163810284"><a name="p19914163810284"></a><a name="p19914163810284"></a>npu_format_cast_.acl_format</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p991403852813"><a name="p991403852813"></a><a name="p991403852813"></a>format_cast_npu_</p>
</td>
</tr>
<tr id="row846074712133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1537209944"><a name="p1537209944"></a><a name="p1537209944"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p139141438132811"><a name="p139141438132811"></a><a name="p139141438132811"></a>npu_format_cast_.src</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1291413852817"><a name="p1291413852817"></a><a name="p1291413852817"></a>format_cast_npu_</p>
</td>
</tr>
<tr id="row17461164716137"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p9372491449"><a name="p9372491449"></a><a name="p9372491449"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p891433852814"><a name="p891433852814"></a><a name="p891433852814"></a>npu_transpose_to_contiguous</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p29141138122810"><a name="p29141138122810"></a><a name="p29141138122810"></a>transpose_to_contiguous_npu</p>
</td>
</tr>
<tr id="row17461204715132"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p13372191246"><a name="p13372191246"></a><a name="p13372191246"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p69141338132819"><a name="p69141338132819"></a><a name="p69141338132819"></a>npu_transpose</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p9914143832811"><a name="p9914143832811"></a><a name="p9914143832811"></a>transpose_npu</p>
</td>
</tr>
<tr id="row9461104712136"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p183726915412"><a name="p183726915412"></a><a name="p183726915412"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1291433832814"><a name="p1291433832814"></a><a name="p1291433832814"></a>npu_transpose.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p991413381285"><a name="p991413381285"></a><a name="p991413381285"></a>transpose_out_npu</p>
</td>
</tr>
<tr id="row8461114771315"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p63721791410"><a name="p63721791410"></a><a name="p63721791410"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p291416382284"><a name="p291416382284"></a><a name="p291416382284"></a>npu_broadcast</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p991453816281"><a name="p991453816281"></a><a name="p991453816281"></a>broadcast_npu</p>
</td>
</tr>
<tr id="row3461104717133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5372199748"><a name="p5372199748"></a><a name="p5372199748"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p14914338182817"><a name="p14914338182817"></a><a name="p14914338182817"></a>npu_broadcast.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p8914143852814"><a name="p8914143852814"></a><a name="p8914143852814"></a>broadcast_out_npu</p>
</td>
</tr>
<tr id="row946154714132"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5372395413"><a name="p5372395413"></a><a name="p5372395413"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p8914638172816"><a name="p8914638172816"></a><a name="p8914638172816"></a>npu_dtype_cast</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1691413892819"><a name="p1691413892819"></a><a name="p1691413892819"></a>dtype_cast_npu</p>
</td>
</tr>
<tr id="row1146114713137"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p93721916412"><a name="p93721916412"></a><a name="p93721916412"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p8914153822811"><a name="p8914153822811"></a><a name="p8914153822811"></a>npu_dtype_cast_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p29141838182816"><a name="p29141838182816"></a><a name="p29141838182816"></a>dtype_cast_npu_</p>
</td>
</tr>
<tr id="row146112473133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p2037316914417"><a name="p2037316914417"></a><a name="p2037316914417"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p11914113819285"><a name="p11914113819285"></a><a name="p11914113819285"></a>npu_roi_alignbk</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p291410385286"><a name="p291410385286"></a><a name="p291410385286"></a>roi_align_backward_npu</p>
</td>
</tr>
<tr id="row1461447161310"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p637339248"><a name="p637339248"></a><a name="p637339248"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p4914338132815"><a name="p4914338132815"></a><a name="p4914338132815"></a>empty_with_format</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1491419389286"><a name="p1491419389286"></a><a name="p1491419389286"></a>empty_with_format_npu</p>
</td>
</tr>
<tr id="row7461134721317"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p7373591343"><a name="p7373591343"></a><a name="p7373591343"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p5914193815283"><a name="p5914193815283"></a><a name="p5914193815283"></a>empty_with_format.names</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p11914143817288"><a name="p11914143817288"></a><a name="p11914143817288"></a>empty_with_format_npu</p>
</td>
</tr>
<tr id="row15461147191318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p103731098418"><a name="p103731098418"></a><a name="p103731098418"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p79141638102811"><a name="p79141638102811"></a><a name="p79141638102811"></a>copy_memory_</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p491483815288"><a name="p491483815288"></a><a name="p491483815288"></a>copy_memory_npu_</p>
</td>
</tr>
<tr id="row1046164717135"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p14373391641"><a name="p14373391641"></a><a name="p14373391641"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p19141638122819"><a name="p19141638122819"></a><a name="p19141638122819"></a>npu_one_hot</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1291453812281"><a name="p1291453812281"></a><a name="p1291453812281"></a>one_hot_npu</p>
</td>
</tr>
<tr id="row184627475137"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5373591748"><a name="p5373591748"></a><a name="p5373591748"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p17914438112817"><a name="p17914438112817"></a><a name="p17914438112817"></a>npu_stride_add</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1891417384285"><a name="p1891417384285"></a><a name="p1891417384285"></a>stride_add_npu</p>
</td>
</tr>
<tr id="row24621147121312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p123731099411"><a name="p123731099411"></a><a name="p123731099411"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1591423812813"><a name="p1591423812813"></a><a name="p1591423812813"></a>npu_softmax_cross_entropy_with_logits</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p11914138172811"><a name="p11914138172811"></a><a name="p11914138172811"></a>softmax_cross_entropy_with_logits_npu</p>
</td>
</tr>
<tr id="row204621747161315"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p3373199645"><a name="p3373199645"></a><a name="p3373199645"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1291653852813"><a name="p1291653852813"></a><a name="p1291653852813"></a>npu_softmax_cross_entropy_with_logits_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p16916173813287"><a name="p16916173813287"></a><a name="p16916173813287"></a>softmax_cross_entropy_with_logits_backward_npu</p>
</td>
</tr>
<tr id="row246294719133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p133731594419"><a name="p133731594419"></a><a name="p133731594419"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p159161138152820"><a name="p159161138152820"></a><a name="p159161138152820"></a>npu_ps_roi_pooling</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p091653818285"><a name="p091653818285"></a><a name="p091653818285"></a>ps_roi_pooling_npu</p>
</td>
</tr>
<tr id="row84621047181311"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p63731699416"><a name="p63731699416"></a><a name="p63731699416"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p5916538202813"><a name="p5916538202813"></a><a name="p5916538202813"></a>npu_ps_roi_pooling_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p13916103882816"><a name="p13916103882816"></a><a name="p13916103882816"></a>ps_roi_pooling_backward_npu</p>
</td>
</tr>
<tr id="row16462174731312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p53731791349"><a name="p53731791349"></a><a name="p53731791349"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p8916163815289"><a name="p8916163815289"></a><a name="p8916163815289"></a>npu_roi_align</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1291610385281"><a name="p1291610385281"></a><a name="p1291610385281"></a>roi_align_npu</p>
</td>
</tr>
<tr id="row3462124716130"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1037310917412"><a name="p1037310917412"></a><a name="p1037310917412"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p15916193822814"><a name="p15916193822814"></a><a name="p15916193822814"></a>npu_nms_v4</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1391663822814"><a name="p1391663822814"></a><a name="p1391663822814"></a>nms_v4_npu</p>
</td>
</tr>
<tr id="row1046244751316"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p113738910412"><a name="p113738910412"></a><a name="p113738910412"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p391673816284"><a name="p391673816284"></a><a name="p391673816284"></a>npu_lstm</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p39161838102819"><a name="p39161838102819"></a><a name="p39161838102819"></a>lstm_npu</p>
</td>
</tr>
<tr id="row1546218475131"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p23731791641"><a name="p23731791641"></a><a name="p23731791641"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p169161238102816"><a name="p169161238102816"></a><a name="p169161238102816"></a>npu_lstm_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p2916123812282"><a name="p2916123812282"></a><a name="p2916123812282"></a>lstm_backward_npu</p>
</td>
</tr>
<tr id="row11462847141320"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p63731991745"><a name="p63731991745"></a><a name="p63731991745"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p4916938182812"><a name="p4916938182812"></a><a name="p4916938182812"></a>npu_iou</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p491613822820"><a name="p491613822820"></a><a name="p491613822820"></a>iou_npu</p>
</td>
</tr>
<tr id="row114621347101318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p103736912411"><a name="p103736912411"></a><a name="p103736912411"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p149161738162812"><a name="p149161738162812"></a><a name="p149161738162812"></a>npu_ptiou</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p3916438192814"><a name="p3916438192814"></a><a name="p3916438192814"></a>ptiou_npu</p>
</td>
</tr>
<tr id="row585972817535"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p785911289531"><a name="p785911289531"></a><a name="p785911289531"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p69160381286"><a name="p69160381286"></a><a name="p69160381286"></a>npu_nms_with_mask</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p179168386288"><a name="p179168386288"></a><a name="p179168386288"></a>nms_with_mask_npu</p>
</td>
</tr>
<tr id="row146351227195319"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p563572745318"><a name="p563572745318"></a><a name="p563572745318"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p59166384282"><a name="p59166384282"></a><a name="p59166384282"></a>npu_pad</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p139168380281"><a name="p139168380281"></a><a name="p139168380281"></a>pad_npu</p>
</td>
</tr>
<tr id="row10604926145316"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1760582611533"><a name="p1760582611533"></a><a name="p1760582611533"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p179166387289"><a name="p179166387289"></a><a name="p179166387289"></a>npu_bounding_box_encode</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p149161038182811"><a name="p149161038182811"></a><a name="p149161038182811"></a>bounding_box_encode_npu</p>
</td>
</tr>
<tr id="row1216382525314"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1216319254534"><a name="p1216319254534"></a><a name="p1216319254534"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p109161138102819"><a name="p109161138102819"></a><a name="p109161138102819"></a>npu_bounding_box_decode</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p7916338122819"><a name="p7916338122819"></a><a name="p7916338122819"></a>bounding_box_decode_npu</p>
</td>
</tr>
<tr id="row1369152495317"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p369124185315"><a name="p369124185315"></a><a name="p369124185315"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p291613389289"><a name="p291613389289"></a><a name="p291613389289"></a>npu_gru</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p12916103872811"><a name="p12916103872811"></a><a name="p12916103872811"></a>gru_npu</p>
</td>
</tr>
<tr id="row17636162114539"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p18636152175314"><a name="p18636152175314"></a><a name="p18636152175314"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p169179385287"><a name="p169179385287"></a><a name="p169179385287"></a>npu_gru_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p89178389280"><a name="p89178389280"></a><a name="p89178389280"></a>gru_backward_npu</p>
</td>
</tr>
<tr id="row12499620185311"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p45001320135317"><a name="p45001320135317"></a><a name="p45001320135317"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p891763892810"><a name="p891763892810"></a><a name="p891763892810"></a>npu_set_.source_Storage_storage_offset_format</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p991743815288"><a name="p991743815288"></a><a name="p991743815288"></a>set_npu_</p>
</td>
</tr>
<tr id="row173888191535"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1338815194530"><a name="p1338815194530"></a><a name="p1338815194530"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p79171738102816"><a name="p79171738102816"></a><a name="p79171738102816"></a>npu_random_choice_with_mask</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p291793872811"><a name="p291793872811"></a><a name="p291793872811"></a>random_choice_with_mask_npu</p>
</td>
</tr>
<tr id="row192551918125314"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p16255171818533"><a name="p16255171818533"></a><a name="p16255171818533"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1491719384286"><a name="p1491719384286"></a><a name="p1491719384286"></a>npu_batch_nms</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p20917123813281"><a name="p20917123813281"></a><a name="p20917123813281"></a>batch_nms_npu</p>
</td>
</tr>
<tr id="row20198181745319"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p12198161765312"><a name="p12198161765312"></a><a name="p12198161765312"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p891773816280"><a name="p891773816280"></a><a name="p891773816280"></a>npu_slice</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p49172038152818"><a name="p49172038152818"></a><a name="p49172038152818"></a>slice_npu</p>
</td>
</tr>
<tr id="row1717121610536"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p91711216165319"><a name="p91711216165319"></a><a name="p91711216165319"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p189171138142815"><a name="p189171138142815"></a><a name="p189171138142815"></a>npu_slice.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1891714384282"><a name="p1891714384282"></a><a name="p1891714384282"></a>slice_out_npu</p>
</td>
</tr>
<tr id="row6772114195312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p19772101418536"><a name="p19772101418536"></a><a name="p19772101418536"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p13917838202815"><a name="p13917838202815"></a><a name="p13917838202815"></a>npu_dropoutV2</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1691773818288"><a name="p1691773818288"></a><a name="p1691773818288"></a>dropout_v2_npu</p>
</td>
</tr>
<tr id="row1372431312535"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p672491310532"><a name="p672491310532"></a><a name="p672491310532"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p8917103882819"><a name="p8917103882819"></a><a name="p8917103882819"></a>npu_dropoutV2_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p11917123812284"><a name="p11917123812284"></a><a name="p11917123812284"></a>dropout_v2_backward_npu</p>
</td>
</tr>
<tr id="row34271912175319"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1842761225311"><a name="p1842761225311"></a><a name="p1842761225311"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p99175385286"><a name="p99175385286"></a><a name="p99175385286"></a>_npu_dropout</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p69171038132813"><a name="p69171038132813"></a><a name="p69171038132813"></a>_dropout_npu</p>
</td>
</tr>
<tr id="row6462134711313"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p143731791411"><a name="p143731791411"></a><a name="p143731791411"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p99171938172814"><a name="p99171938172814"></a><a name="p99171938172814"></a>_npu_dropout_inplace</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p199171438102818"><a name="p199171438102818"></a><a name="p199171438102818"></a>_dropout_npu_inplace</p>
</td>
</tr>
<tr id="row97791110115313"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p13779310195315"><a name="p13779310195315"></a><a name="p13779310195315"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p59171938102815"><a name="p59171938102815"></a><a name="p59171938102815"></a>npu_dropout_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p491714385287"><a name="p491714385287"></a><a name="p491714385287"></a>dropout_backward_npu</p>
</td>
</tr>
<tr id="row184631247171312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p2373189647"><a name="p2373189647"></a><a name="p2373189647"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1591753872810"><a name="p1591753872810"></a><a name="p1591753872810"></a>npu_indexing</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p20917143818285"><a name="p20917143818285"></a><a name="p20917143818285"></a>indexing_npu</p>
</td>
</tr>
<tr id="row1346364731320"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p337311910420"><a name="p337311910420"></a><a name="p337311910420"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p209171938172819"><a name="p209171938172819"></a><a name="p209171938172819"></a>npu_indexing.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p2917143817288"><a name="p2917143817288"></a><a name="p2917143817288"></a>indexing_out_npu</p>
</td>
</tr>
<tr id="row1463124714138"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p8374392416"><a name="p8374392416"></a><a name="p8374392416"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p11917138192818"><a name="p11917138192818"></a><a name="p11917138192818"></a>npu_ifmr</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p8917173815281"><a name="p8917173815281"></a><a name="p8917173815281"></a>ifmr_npu</p>
</td>
</tr>
<tr id="row104631747161314"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p23741894415"><a name="p23741894415"></a><a name="p23741894415"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p7917153813286"><a name="p7917153813286"></a><a name="p7917153813286"></a>npu_max.dim</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1917123812814"><a name="p1917123812814"></a><a name="p1917123812814"></a>max_v1_npu</p>
</td>
</tr>
<tr id="row739518135312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p3395208125315"><a name="p3395208125315"></a><a name="p3395208125315"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p10917123882817"><a name="p10917123882817"></a><a name="p10917123882817"></a>npu_max.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p99171238112820"><a name="p99171238112820"></a><a name="p99171238112820"></a>max_v1_npu</p>
</td>
</tr>
<tr id="row11492352531"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p13492856530"><a name="p13492856530"></a><a name="p13492856530"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p5917113832815"><a name="p5917113832815"></a><a name="p5917113832815"></a>npu_scatter</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1918173819289"><a name="p1918173819289"></a><a name="p1918173819289"></a>scatter_npu</p>
</td>
</tr>
<tr id="row4579195725211"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p957975714522"><a name="p957975714522"></a><a name="p957975714522"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p109184382282"><a name="p109184382282"></a><a name="p109184382282"></a>npu_max_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p491813386285"><a name="p491813386285"></a><a name="p491813386285"></a>max_backward_npu</p>
</td>
</tr>
<tr id="row1316510579407"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p15165125717406"><a name="p15165125717406"></a><a name="p15165125717406"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1591819384284"><a name="p1591819384284"></a><a name="p1591819384284"></a>npu_apply_adam</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p5918203822818"><a name="p5918203822818"></a><a name="p5918203822818"></a>apply_adam_npu</p>
</td>
</tr>
<tr id="row238075912407"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p338035912408"><a name="p338035912408"></a><a name="p338035912408"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p179181938132811"><a name="p179181938132811"></a><a name="p179181938132811"></a>npu_layer_norm_eval</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p8918123817286"><a name="p8918123817286"></a><a name="p8918123817286"></a>layer_norm_eval_npu</p>
</td>
</tr>
<tr id="row1862019531825"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p9620953028"><a name="p9620953028"></a><a name="p9620953028"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p6918538142818"><a name="p6918538142818"></a><a name="p6918538142818"></a>npu_alloc_float_status</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p11918143810282"><a name="p11918143810282"></a><a name="p11918143810282"></a>alloc_float_status_npu</p>
</td>
</tr>
<tr id="row1728911511238"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p7289125119316"><a name="p7289125119316"></a><a name="p7289125119316"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p12918113862810"><a name="p12918113862810"></a><a name="p12918113862810"></a>npu_get_float_status</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1891893818281"><a name="p1891893818281"></a><a name="p1891893818281"></a>get_float_status_npu</p>
</td>
</tr>
<tr id="row360924815311"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5609648036"><a name="p5609648036"></a><a name="p5609648036"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p189181438152817"><a name="p189181438152817"></a><a name="p189181438152817"></a>npu_clear_float_status</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p79181238152817"><a name="p79181238152817"></a><a name="p79181238152817"></a>clear_float_status_npu</p>
</td>
</tr>
<tr id="row15706114616318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p147061846733"><a name="p147061846733"></a><a name="p147061846733"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p13918193862812"><a name="p13918193862812"></a><a name="p13918193862812"></a>npu_confusion_transpose</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p2918738182812"><a name="p2918738182812"></a><a name="p2918738182812"></a>confusion_transpose_npu</p>
</td>
</tr>
<tr id="row13860184418316"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p486084412314"><a name="p486084412314"></a><a name="p486084412314"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p191813812283"><a name="p191813812283"></a><a name="p191813812283"></a>npu_confusion_transpose_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p109181438132818"><a name="p109181438132818"></a><a name="p109181438132818"></a>confusion_transpose_backward_npu</p>
</td>
</tr>
<tr id="row155784016310"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p15571640231"><a name="p15571640231"></a><a name="p15571640231"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1918193819285"><a name="p1918193819285"></a><a name="p1918193819285"></a>npu_bmmV2</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p10918163842815"><a name="p10918163842815"></a><a name="p10918163842815"></a>bmm_v2_npu</p>
</td>
</tr>
<tr id="row38693421313"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p178692423316"><a name="p178692423316"></a><a name="p178692423316"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p139186383286"><a name="p139186383286"></a><a name="p139186383286"></a>fast_gelu</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p18918338152816"><a name="p18918338152816"></a><a name="p18918338152816"></a>fast_gelu_npu</p>
</td>
</tr>
<tr id="row8404138936"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5404193820319"><a name="p5404193820319"></a><a name="p5404193820319"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1191817383283"><a name="p1191817383283"></a><a name="p1191817383283"></a>fast_gelu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p591863852815"><a name="p591863852815"></a><a name="p591863852815"></a>fast_gelu_backward_npu</p>
</td>
</tr>
<tr id="row48728361436"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1287253614310"><a name="p1287253614310"></a><a name="p1287253614310"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p5918133814288"><a name="p5918133814288"></a><a name="p5918133814288"></a>npu_sub_sample</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p15918113852813"><a name="p15918113852813"></a><a name="p15918113852813"></a>sub_sample_npu</p>
</td>
</tr>
<tr id="row193801734737"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p163818348318"><a name="p163818348318"></a><a name="p163818348318"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p4918738132818"><a name="p4918738132818"></a><a name="p4918738132818"></a>npu_deformable_conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p89183388282"><a name="p89183388282"></a><a name="p89183388282"></a>deformable_conv2d_npu</p>
</td>
</tr>
<tr id="row887819321539"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p38781332934"><a name="p38781332934"></a><a name="p38781332934"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p391863852817"><a name="p391863852817"></a><a name="p391863852817"></a>npu_deformable_conv2dbk</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1191813813287"><a name="p1191813813287"></a><a name="p1191813813287"></a>deformable_conv2d_backward_npu</p>
</td>
</tr>
<tr id="row1936802914316"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p43691329937"><a name="p43691329937"></a><a name="p43691329937"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p29181238142811"><a name="p29181238142811"></a><a name="p29181238142811"></a>npu_mish</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1091818382288"><a name="p1091818382288"></a><a name="p1091818382288"></a>mish_npu</p>
</td>
</tr>
<tr id="row161409311933"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p2014003110319"><a name="p2014003110319"></a><a name="p2014003110319"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p691813802815"><a name="p691813802815"></a><a name="p691813802815"></a>npu_anchor_response_flags</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1918183814285"><a name="p1918183814285"></a><a name="p1918183814285"></a>anchor_response_flags_npu</p>
</td>
</tr>
<tr id="row124539561211"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1845314561229"><a name="p1845314561229"></a><a name="p1845314561229"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p7919103852816"><a name="p7919103852816"></a><a name="p7919103852816"></a>npu_yolo_boxes_encode</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p6919153892812"><a name="p6919153892812"></a><a name="p6919153892812"></a>yolo_boxes_encode_npu</p>
</td>
</tr>
<tr id="row47113234310"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p14711182310319"><a name="p14711182310319"></a><a name="p14711182310319"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1991923815289"><a name="p1991923815289"></a><a name="p1991923815289"></a>npu_grid_assign_positive</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1491913842815"><a name="p1491913842815"></a><a name="p1491913842815"></a>grid_assign_positive_npu</p>
</td>
</tr>
<tr id="row44811071817"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p124812011810"><a name="p124812011810"></a><a name="p124812011810"></a>84</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p8919123842811"><a name="p8919123842811"></a><a name="p8919123842811"></a>npu_mish_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p18919183882818"><a name="p18919183882818"></a><a name="p18919183882818"></a>mish_backward_npu</p>
</td>
</tr>
<tr id="row16374103182"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p46381410161810"><a name="p46381410161810"></a><a name="p46381410161810"></a>85</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1691983819287"><a name="p1691983819287"></a><a name="p1691983819287"></a>npu_normalize_batch</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p9919638122811"><a name="p9919638122811"></a><a name="p9919638122811"></a>normalize_batch_npu</p>
</td>
</tr>
<tr id="row174712815182"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p167471888185"><a name="p167471888185"></a><a name="p167471888185"></a>86</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p091913819287"><a name="p091913819287"></a><a name="p091913819287"></a>npu_masked_fill_range</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p16919538192817"><a name="p16919538192817"></a><a name="p16919538192817"></a>masked_fill_range_npu</p>
</td>
</tr>
<tr id="row127494641817"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p147498614182"><a name="p147498614182"></a><a name="p147498614182"></a>87</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1691913822813"><a name="p1691913822813"></a><a name="p1691913822813"></a>npu_linear</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p691993813282"><a name="p691993813282"></a><a name="p691993813282"></a>linear_npu</p>
</td>
</tr>
<tr id="row7428158141712"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p34281258131711"><a name="p34281258131711"></a><a name="p34281258131711"></a>88</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p991911388289"><a name="p991911388289"></a><a name="p991911388289"></a>npu_linear_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p691903817282"><a name="p691903817282"></a><a name="p691903817282"></a>linear_backward_npu</p>
</td>
</tr>
<tr id="row5792255101720"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p279220558174"><a name="p279220558174"></a><a name="p279220558174"></a>89</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p3919123832817"><a name="p3919123832817"></a><a name="p3919123832817"></a>npu_bert_apply_adam</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p189191338192816"><a name="p189191338192816"></a><a name="p189191338192816"></a>bert_apply_adam_npu</p>
</td>
</tr>
<tr id="row189851647122810"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p16985847112818"><a name="p16985847112818"></a><a name="p16985847112818"></a>90</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p69199386282"><a name="p69199386282"></a><a name="p69199386282"></a>npu_giou</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1398516477286"><a name="p1398516477286"></a><a name="p1398516477286"></a>giou_npu</p>
</td>
</tr>
<tr id="row185177501285"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p6517105012288"><a name="p6517105012288"></a><a name="p6517105012288"></a>91</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p17517155092819"><a name="p17517155092819"></a><a name="p17517155092819"></a>npu_giou_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p16517115017288"><a name="p16517115017288"></a><a name="p16517115017288"></a>giou_backward_npu</p>
</td>
</tr>
</tbody>
</table>

