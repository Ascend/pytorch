# PyTorch适配算子清单
-   [PyTorch原生算子与昇腾算子对应表](#PyTorch原生算子与昇腾算子对应表md)
-   [PyTorch昇腾自定义算子](#PyTorch昇腾自定义算子md)
<h2 id="PyTorch原生算子与昇腾算子对应表md">PyTorch原生算子与昇腾算子对应表</h2>

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
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p174776236345"><a name="p174776236345"></a><a name="p174776236345"></a>dropout</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15477923103420"><a name="p15477923103420"></a><a name="p15477923103420"></a>dropout_npu</p>
</td>
</tr>
<tr id="row469519391121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1458131614214"><a name="p1458131614214"></a><a name="p1458131614214"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184771823123419"><a name="p184771823123419"></a><a name="p184771823123419"></a>dropout_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10477112323410"><a name="p10477112323410"></a><a name="p10477112323410"></a>dropout_npu_</p>
</td>
</tr>
<tr id="row156952394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1145861612218"><a name="p1145861612218"></a><a name="p1145861612218"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p174771823103414"><a name="p174771823103414"></a><a name="p174771823103414"></a>abs</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13477162318344"><a name="p13477162318344"></a><a name="p13477162318344"></a>abs_npu</p>
</td>
</tr>
<tr id="row17695739101215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134581516024"><a name="p134581516024"></a><a name="p134581516024"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3477223133420"><a name="p3477223133420"></a><a name="p3477223133420"></a>abs_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1477122323418"><a name="p1477122323418"></a><a name="p1477122323418"></a>abs_npu_</p>
</td>
</tr>
<tr id="row569517398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1945819162213"><a name="p1945819162213"></a><a name="p1945819162213"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184771423163414"><a name="p184771423163414"></a><a name="p184771423163414"></a>abs.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8477162343417"><a name="p8477162343417"></a><a name="p8477162343417"></a>abs_out_npu</p>
</td>
</tr>
<tr id="row6695123941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9458816327"><a name="p9458816327"></a><a name="p9458816327"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114777231348"><a name="p114777231348"></a><a name="p114777231348"></a>acos</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1477122373416"><a name="p1477122373416"></a><a name="p1477122373416"></a>acos_npu</p>
</td>
</tr>
<tr id="row869593910122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124586161729"><a name="p124586161729"></a><a name="p124586161729"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13477102373410"><a name="p13477102373410"></a><a name="p13477102373410"></a>acos_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144771823133419"><a name="p144771823133419"></a><a name="p144771823133419"></a>acos_npu_</p>
</td>
</tr>
<tr id="row16695239121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p745881610211"><a name="p745881610211"></a><a name="p745881610211"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17477523193411"><a name="p17477523193411"></a><a name="p17477523193411"></a>acos.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147712235341"><a name="p1147712235341"></a><a name="p1147712235341"></a>acos_out_npu</p>
</td>
</tr>
<tr id="row18696133961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14458161613213"><a name="p14458161613213"></a><a name="p14458161613213"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24771823183417"><a name="p24771823183417"></a><a name="p24771823183417"></a>adaptive_avg_pool1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5477723173413"><a name="p5477723173413"></a><a name="p5477723173413"></a>adaptive_avg_pool1d_npu</p>
</td>
</tr>
<tr id="row1769693961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p345851614215"><a name="p345851614215"></a><a name="p345851614215"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20478112314344"><a name="p20478112314344"></a><a name="p20478112314344"></a>add.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174781237349"><a name="p174781237349"></a><a name="p174781237349"></a>add_npu</p>
</td>
</tr>
<tr id="row1869623951214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1945814161127"><a name="p1945814161127"></a><a name="p1945814161127"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p174781023113413"><a name="p174781023113413"></a><a name="p174781023113413"></a>add_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p247814230342"><a name="p247814230342"></a><a name="p247814230342"></a>add_npu_</p>
</td>
</tr>
<tr id="row16961439181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13458916224"><a name="p13458916224"></a><a name="p13458916224"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1247832313417"><a name="p1247832313417"></a><a name="p1247832313417"></a>add.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16478923193419"><a name="p16478923193419"></a><a name="p16478923193419"></a>add_out_npu</p>
</td>
</tr>
<tr id="row10696133931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6458116525"><a name="p6458116525"></a><a name="p6458116525"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13478112317346"><a name="p13478112317346"></a><a name="p13478112317346"></a>add.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1147802343418"><a name="p1147802343418"></a><a name="p1147802343418"></a>add_npu</p>
</td>
</tr>
<tr id="row6696143991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1345814161122"><a name="p1345814161122"></a><a name="p1345814161122"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1947842320346"><a name="p1947842320346"></a><a name="p1947842320346"></a>add_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11478923113418"><a name="p11478923113418"></a><a name="p11478923113418"></a>add_npu_</p>
</td>
</tr>
<tr id="row1969613901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04591116925"><a name="p04591116925"></a><a name="p04591116925"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p144781823203410"><a name="p144781823203410"></a><a name="p144781823203410"></a>addmv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10478102393417"><a name="p10478102393417"></a><a name="p10478102393417"></a>addmv_npu</p>
</td>
</tr>
<tr id="row1169614395122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144592161423"><a name="p144592161423"></a><a name="p144592161423"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0478102316347"><a name="p0478102316347"></a><a name="p0478102316347"></a>addmv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p104787239342"><a name="p104787239342"></a><a name="p104787239342"></a>addmv_npu_</p>
</td>
</tr>
<tr id="row2696103911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54596161229"><a name="p54596161229"></a><a name="p54596161229"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6478202310348"><a name="p6478202310348"></a><a name="p6478202310348"></a>addmv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p154781823183420"><a name="p154781823183420"></a><a name="p154781823183420"></a>addmv_out_npu</p>
</td>
</tr>
<tr id="row116976397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04591716922"><a name="p04591716922"></a><a name="p04591716922"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14478152311348"><a name="p14478152311348"></a><a name="p14478152311348"></a>addr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p947812373411"><a name="p947812373411"></a><a name="p947812373411"></a>addr_npu</p>
</td>
</tr>
<tr id="row1769718393121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p545914161122"><a name="p545914161122"></a><a name="p545914161122"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7478162318348"><a name="p7478162318348"></a><a name="p7478162318348"></a>addr_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0478223183415"><a name="p0478223183415"></a><a name="p0478223183415"></a>addr_npu_</p>
</td>
</tr>
<tr id="row1669716393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145901617213"><a name="p145901617213"></a><a name="p145901617213"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1047814237343"><a name="p1047814237343"></a><a name="p1047814237343"></a>addr.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p247817236340"><a name="p247817236340"></a><a name="p247817236340"></a>addr_out_npu</p>
</td>
</tr>
<tr id="row1469716399129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1145921614213"><a name="p1145921614213"></a><a name="p1145921614213"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1478423193417"><a name="p1478423193417"></a><a name="p1478423193417"></a>affine_grid_generator</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1847812313418"><a name="p1847812313418"></a><a name="p1847812313418"></a>affine_grid_generator_npu</p>
</td>
</tr>
<tr id="row6697143981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194595161421"><a name="p194595161421"></a><a name="p194595161421"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p44785232348"><a name="p44785232348"></a><a name="p44785232348"></a>affine_grid_generator_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p184784231348"><a name="p184784231348"></a><a name="p184784231348"></a>affine_grid_generator_backward_npu</p>
</td>
</tr>
<tr id="row5697103931212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p545912161620"><a name="p545912161620"></a><a name="p545912161620"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p247815233347"><a name="p247815233347"></a><a name="p247815233347"></a>all.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13478132316340"><a name="p13478132316340"></a><a name="p13478132316340"></a>all_npu</p>
</td>
</tr>
<tr id="row11697133961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174590161428"><a name="p174590161428"></a><a name="p174590161428"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1547822313415"><a name="p1547822313415"></a><a name="p1547822313415"></a>all.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p547862303413"><a name="p547862303413"></a><a name="p547862303413"></a>all_out_npu</p>
</td>
</tr>
<tr id="row13697239171218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p045914169210"><a name="p045914169210"></a><a name="p045914169210"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p547892323413"><a name="p547892323413"></a><a name="p547892323413"></a>any.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17479523133418"><a name="p17479523133418"></a><a name="p17479523133418"></a>any_npu</p>
</td>
</tr>
<tr id="row7698143951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3459916920"><a name="p3459916920"></a><a name="p3459916920"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124799233349"><a name="p124799233349"></a><a name="p124799233349"></a>any.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2479172393419"><a name="p2479172393419"></a><a name="p2479172393419"></a>any_out_npu</p>
</td>
</tr>
<tr id="row3698133916123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04605168219"><a name="p04605168219"></a><a name="p04605168219"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p747932383413"><a name="p747932383413"></a><a name="p747932383413"></a>arange</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9479162333417"><a name="p9479162333417"></a><a name="p9479162333417"></a>arange_npu</p>
</td>
</tr>
<tr id="row86981439181215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1446091613214"><a name="p1446091613214"></a><a name="p1446091613214"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p247913237347"><a name="p247913237347"></a><a name="p247913237347"></a>arange.start</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13479923123410"><a name="p13479923123410"></a><a name="p13479923123410"></a>arange_npu</p>
</td>
</tr>
<tr id="row8698203971214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146015163211"><a name="p146015163211"></a><a name="p146015163211"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1647912232349"><a name="p1647912232349"></a><a name="p1647912232349"></a>arange.start_step</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p184797231347"><a name="p184797231347"></a><a name="p184797231347"></a>arange_npu</p>
</td>
</tr>
<tr id="row1698153941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194608160212"><a name="p194608160212"></a><a name="p194608160212"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5479112315341"><a name="p5479112315341"></a><a name="p5479112315341"></a>arange.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11479123103415"><a name="p11479123103415"></a><a name="p11479123103415"></a>arange_out_npu</p>
</td>
</tr>
<tr id="row4698143917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2460616021"><a name="p2460616021"></a><a name="p2460616021"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p347902343417"><a name="p347902343417"></a><a name="p347902343417"></a>arange.start_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11479122343416"><a name="p11479122343416"></a><a name="p11479122343416"></a>arange_out_npu</p>
</td>
</tr>
<tr id="row1469810393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19460716125"><a name="p19460716125"></a><a name="p19460716125"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17479132333413"><a name="p17479132333413"></a><a name="p17479132333413"></a>_dim_arange</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1479182343412"><a name="p1479182343412"></a><a name="p1479182343412"></a>_dim_arange_npu</p>
</td>
</tr>
<tr id="row17698153919124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174601316821"><a name="p174601316821"></a><a name="p174601316821"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1747910236341"><a name="p1747910236341"></a><a name="p1747910236341"></a>argmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p447914233349"><a name="p447914233349"></a><a name="p447914233349"></a>argmax_npu</p>
</td>
</tr>
<tr id="row46981739181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14460141616212"><a name="p14460141616212"></a><a name="p14460141616212"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8479112311344"><a name="p8479112311344"></a><a name="p8479112311344"></a>argmin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14791123123418"><a name="p14791123123418"></a><a name="p14791123123418"></a>argmin_npu</p>
</td>
</tr>
<tr id="row46981939141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114603161024"><a name="p114603161024"></a><a name="p114603161024"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1247918238345"><a name="p1247918238345"></a><a name="p1247918238345"></a>as_strided</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2047992318345"><a name="p2047992318345"></a><a name="p2047992318345"></a>as_strided_npu</p>
</td>
</tr>
<tr id="row2698339151220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11460816521"><a name="p11460816521"></a><a name="p11460816521"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18479723103413"><a name="p18479723103413"></a><a name="p18479723103413"></a>as_strided_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14791423173414"><a name="p14791423173414"></a><a name="p14791423173414"></a>as_strided_npu_</p>
</td>
</tr>
<tr id="row369911399124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4460916527"><a name="p4460916527"></a><a name="p4460916527"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p64798231342"><a name="p64798231342"></a><a name="p64798231342"></a>asin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1647952383416"><a name="p1647952383416"></a><a name="p1647952383416"></a>asin_npu</p>
</td>
</tr>
<tr id="row106992394125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446015161828"><a name="p446015161828"></a><a name="p446015161828"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5479023183413"><a name="p5479023183413"></a><a name="p5479023183413"></a>asin_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3479162313345"><a name="p3479162313345"></a><a name="p3479162313345"></a>asin_npu_</p>
</td>
</tr>
<tr id="row9699139121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104601161822"><a name="p104601161822"></a><a name="p104601161822"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14479523143419"><a name="p14479523143419"></a><a name="p14479523143419"></a>asin.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1047992373410"><a name="p1047992373410"></a><a name="p1047992373410"></a>asin_out_npu</p>
</td>
</tr>
<tr id="row166991339121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44608161825"><a name="p44608161825"></a><a name="p44608161825"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11479223143415"><a name="p11479223143415"></a><a name="p11479223143415"></a>atan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p204791423203416"><a name="p204791423203416"></a><a name="p204791423203416"></a>atan_npu</p>
</td>
</tr>
<tr id="row3699139191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44604161725"><a name="p44604161725"></a><a name="p44604161725"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p348092312348"><a name="p348092312348"></a><a name="p348092312348"></a>atan_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148052393419"><a name="p1148052393419"></a><a name="p1148052393419"></a>atan_npu_</p>
</td>
</tr>
<tr id="row269915391125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10460516425"><a name="p10460516425"></a><a name="p10460516425"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p44801323113412"><a name="p44801323113412"></a><a name="p44801323113412"></a>atan.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18480123103418"><a name="p18480123103418"></a><a name="p18480123103418"></a>atan_out_npu</p>
</td>
</tr>
<tr id="row869983913127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10460716129"><a name="p10460716129"></a><a name="p10460716129"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4480023123419"><a name="p4480023123419"></a><a name="p4480023123419"></a>baddbmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p74804232343"><a name="p74804232343"></a><a name="p74804232343"></a>baddbmm_npu</p>
</td>
</tr>
<tr id="row46997391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84605163210"><a name="p84605163210"></a><a name="p84605163210"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1748015231341"><a name="p1748015231341"></a><a name="p1748015231341"></a>baddbmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194801723113418"><a name="p194801723113418"></a><a name="p194801723113418"></a>baddbmm_npu_</p>
</td>
</tr>
<tr id="row18699143911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846013161021"><a name="p846013161021"></a><a name="p846013161021"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13480323203412"><a name="p13480323203412"></a><a name="p13480323203412"></a>baddbmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1480123173416"><a name="p1480123173416"></a><a name="p1480123173416"></a>baddbmm_out_npu</p>
</td>
</tr>
<tr id="row9700163961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11461181616213"><a name="p11461181616213"></a><a name="p11461181616213"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11480323113411"><a name="p11480323113411"></a><a name="p11480323113411"></a>bartlett_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20480152319349"><a name="p20480152319349"></a><a name="p20480152319349"></a>bartlett_window_npu</p>
</td>
</tr>
<tr id="row57008394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p046141617218"><a name="p046141617218"></a><a name="p046141617218"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p164801023163418"><a name="p164801023163418"></a><a name="p164801023163418"></a>bartlett_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12480112333411"><a name="p12480112333411"></a><a name="p12480112333411"></a>bartlett_window_npu</p>
</td>
</tr>
<tr id="row20700113951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74615161222"><a name="p74615161222"></a><a name="p74615161222"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7480112323412"><a name="p7480112323412"></a><a name="p7480112323412"></a>batch_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1248015233349"><a name="p1248015233349"></a><a name="p1248015233349"></a>batch_norm_npu_</p>
</td>
</tr>
<tr id="row1070043920122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194614162212"><a name="p194614162212"></a><a name="p194614162212"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p648042313419"><a name="p648042313419"></a><a name="p648042313419"></a>_batch_norm_impl_index</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174801623123414"><a name="p174801623123414"></a><a name="p174801623123414"></a>_batch_norm_impl_index_npu</p>
</td>
</tr>
<tr id="row1970093931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9461116320"><a name="p9461116320"></a><a name="p9461116320"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1848012315341"><a name="p1848012315341"></a><a name="p1848012315341"></a>_batch_norm_impl_index_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p74806238347"><a name="p74806238347"></a><a name="p74806238347"></a>_batch_norm_impl_index_backward_npu</p>
</td>
</tr>
<tr id="row270033916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16461151618219"><a name="p16461151618219"></a><a name="p16461151618219"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1248012318347"><a name="p1248012318347"></a><a name="p1248012318347"></a>bernoulli</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5480122320347"><a name="p5480122320347"></a><a name="p5480122320347"></a>bernoulli_npu</p>
</td>
</tr>
<tr id="row10700339151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4461191610214"><a name="p4461191610214"></a><a name="p4461191610214"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p848022314343"><a name="p848022314343"></a><a name="p848022314343"></a>bernoulli_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148012319348"><a name="p1148012319348"></a><a name="p1148012319348"></a>bernoulli_npu_</p>
</td>
</tr>
<tr id="row12700539141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146118167215"><a name="p1146118167215"></a><a name="p1146118167215"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9480223153415"><a name="p9480223153415"></a><a name="p9480223153415"></a>bernoulli_.float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p148119239344"><a name="p148119239344"></a><a name="p148119239344"></a>bernoulli_npu_</p>
</td>
</tr>
<tr id="row8700203961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2461131611210"><a name="p2461131611210"></a><a name="p2461131611210"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5481023123410"><a name="p5481023123410"></a><a name="p5481023123410"></a>binary_cross_entropy</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p948142303419"><a name="p948142303419"></a><a name="p948142303419"></a>binary_cross_entropy_npu</p>
</td>
</tr>
<tr id="row1770043931218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746161614219"><a name="p1746161614219"></a><a name="p1746161614219"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14811723123415"><a name="p14811723123415"></a><a name="p14811723123415"></a>binary_cross_entropy.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p148117236340"><a name="p148117236340"></a><a name="p148117236340"></a>binary_cross_entropy_out_npu</p>
</td>
</tr>
<tr id="row5700139121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84611816526"><a name="p84611816526"></a><a name="p84611816526"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24811723203419"><a name="p24811723203419"></a><a name="p24811723203419"></a>binary_cross_entropy_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14481142363413"><a name="p14481142363413"></a><a name="p14481142363413"></a>binary_cross_entropy_backward_npu</p>
</td>
</tr>
<tr id="row137012039151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546191619213"><a name="p1546191619213"></a><a name="p1546191619213"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1481132373417"><a name="p1481132373417"></a><a name="p1481132373417"></a>binary_cross_entropy_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84817237346"><a name="p84817237346"></a><a name="p84817237346"></a>binary_cross_entropy_backward_out_npu</p>
</td>
</tr>
<tr id="row5701143914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746218164216"><a name="p1746218164216"></a><a name="p1746218164216"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2481132333410"><a name="p2481132333410"></a><a name="p2481132333410"></a>binary_cross_entropy_with_logits</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4481122343413"><a name="p4481122343413"></a><a name="p4481122343413"></a>binary_cross_entropy_with_logits_npu</p>
</td>
</tr>
<tr id="row18701439171211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17462616221"><a name="p17462616221"></a><a name="p17462616221"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15481623153410"><a name="p15481623153410"></a><a name="p15481623153410"></a>binary_cross_entropy_with_logits_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84812023193412"><a name="p84812023193412"></a><a name="p84812023193412"></a>binary_cross_entropy_with_logits_backward_npu</p>
</td>
</tr>
<tr id="row5701173912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246216169211"><a name="p1246216169211"></a><a name="p1246216169211"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p048114239349"><a name="p048114239349"></a><a name="p048114239349"></a>bitwise_not</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1248152313343"><a name="p1248152313343"></a><a name="p1248152313343"></a>bitwise_not_npu</p>
</td>
</tr>
<tr id="row270111390122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164621316724"><a name="p164621316724"></a><a name="p164621316724"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8481162303418"><a name="p8481162303418"></a><a name="p8481162303418"></a>bitwise_not_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1748162317344"><a name="p1748162317344"></a><a name="p1748162317344"></a>bitwise_not_npu_</p>
</td>
</tr>
<tr id="row27010399120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14627161629"><a name="p14627161629"></a><a name="p14627161629"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p154811423113412"><a name="p154811423113412"></a><a name="p154811423113412"></a>bitwise_not.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1548152311346"><a name="p1548152311346"></a><a name="p1548152311346"></a>bitwise_not_out_npu</p>
</td>
</tr>
<tr id="row157011339201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15462316429"><a name="p15462316429"></a><a name="p15462316429"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p184812023193413"><a name="p184812023193413"></a><a name="p184812023193413"></a>logical_not</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10481423193410"><a name="p10481423193410"></a><a name="p10481423193410"></a>logical_not_npu</p>
</td>
</tr>
<tr id="row187011339161218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346214163212"><a name="p346214163212"></a><a name="p346214163212"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1548192353418"><a name="p1548192353418"></a><a name="p1548192353418"></a>logical_not_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114819234346"><a name="p114819234346"></a><a name="p114819234346"></a>logical_not_npu_</p>
</td>
</tr>
<tr id="row20701183921218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146271617211"><a name="p1146271617211"></a><a name="p1146271617211"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p204811723183416"><a name="p204811723183416"></a><a name="p204811723183416"></a>logical_not.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1548112313349"><a name="p1548112313349"></a><a name="p1548112313349"></a>logical_not_out_npu</p>
</td>
</tr>
<tr id="row177011539151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84621516924"><a name="p84621516924"></a><a name="p84621516924"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p94811233347"><a name="p94811233347"></a><a name="p94811233347"></a>logical_and</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2481023103415"><a name="p2481023103415"></a><a name="p2481023103415"></a>logical_and_npu</p>
</td>
</tr>
<tr id="row37015396121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24626162028"><a name="p24626162028"></a><a name="p24626162028"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10481623183417"><a name="p10481623183417"></a><a name="p10481623183417"></a>logical_and_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9481112343415"><a name="p9481112343415"></a><a name="p9481112343415"></a>logical_and_npu_</p>
</td>
</tr>
<tr id="row1470243915128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11462101612215"><a name="p11462101612215"></a><a name="p11462101612215"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p74821223123412"><a name="p74821223123412"></a><a name="p74821223123412"></a>logical_and.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94821523103418"><a name="p94821523103418"></a><a name="p94821523103418"></a>logical_and_out_npu</p>
</td>
</tr>
<tr id="row870210392126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144622161322"><a name="p144622161322"></a><a name="p144622161322"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1548222320341"><a name="p1548222320341"></a><a name="p1548222320341"></a>logical_or</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194827238342"><a name="p194827238342"></a><a name="p194827238342"></a>logical_or_npu</p>
</td>
</tr>
<tr id="row670210390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9462316321"><a name="p9462316321"></a><a name="p9462316321"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24829230345"><a name="p24829230345"></a><a name="p24829230345"></a>logical_or_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10482523163414"><a name="p10482523163414"></a><a name="p10482523163414"></a>logical_or_npu_</p>
</td>
</tr>
<tr id="row1570215393125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546219161422"><a name="p546219161422"></a><a name="p546219161422"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1048216232346"><a name="p1048216232346"></a><a name="p1048216232346"></a>logical_or.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p64827234346"><a name="p64827234346"></a><a name="p64827234346"></a>logical_or_out_npu</p>
</td>
</tr>
<tr id="row18702203919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446291617213"><a name="p446291617213"></a><a name="p446291617213"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p548217239342"><a name="p548217239342"></a><a name="p548217239342"></a>blackman_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1448262333415"><a name="p1448262333415"></a><a name="p1448262333415"></a>blackman_window_npu</p>
</td>
</tr>
<tr id="row1870283916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14462151619216"><a name="p14462151619216"></a><a name="p14462151619216"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24825231342"><a name="p24825231342"></a><a name="p24825231342"></a>blackman_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144822239340"><a name="p144822239340"></a><a name="p144822239340"></a>blackman_window_npu</p>
</td>
</tr>
<tr id="row1870263914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64621016123"><a name="p64621016123"></a><a name="p64621016123"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2482823103411"><a name="p2482823103411"></a><a name="p2482823103411"></a>bmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1948282333411"><a name="p1948282333411"></a><a name="p1948282333411"></a>bmm_npu</p>
</td>
</tr>
<tr id="row12702103918126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146231616213"><a name="p146231616213"></a><a name="p146231616213"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84821823123415"><a name="p84821823123415"></a><a name="p84821823123415"></a>bmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7482923143412"><a name="p7482923143412"></a><a name="p7482923143412"></a>bmm_out_npu</p>
</td>
</tr>
<tr id="row97021739111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1046217164214"><a name="p1046217164214"></a><a name="p1046217164214"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17482182343419"><a name="p17482182343419"></a><a name="p17482182343419"></a>cat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1448272315345"><a name="p1448272315345"></a><a name="p1448272315345"></a>cat_npu</p>
</td>
</tr>
<tr id="row4702439171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14628160211"><a name="p14628160211"></a><a name="p14628160211"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2048272318349"><a name="p2048272318349"></a><a name="p2048272318349"></a>cat.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14482112312343"><a name="p14482112312343"></a><a name="p14482112312343"></a>cat_out_npu</p>
</td>
</tr>
<tr id="row12703153917121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54637161021"><a name="p54637161021"></a><a name="p54637161021"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24821523153417"><a name="p24821523153417"></a><a name="p24821523153417"></a>cat.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1482323183411"><a name="p1482323183411"></a><a name="p1482323183411"></a>cat_npu</p>
</td>
</tr>
<tr id="row1470363911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546371616214"><a name="p1546371616214"></a><a name="p1546371616214"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7482112319341"><a name="p7482112319341"></a><a name="p7482112319341"></a>cat.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14482423183413"><a name="p14482423183413"></a><a name="p14482423183413"></a>cat_out_npu</p>
</td>
</tr>
<tr id="row170313398129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94631916625"><a name="p94631916625"></a><a name="p94631916625"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1248212333414"><a name="p1248212333414"></a><a name="p1248212333414"></a>ceil</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p548218233345"><a name="p548218233345"></a><a name="p548218233345"></a>ceil_npu</p>
</td>
</tr>
<tr id="row570333911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246314166216"><a name="p1246314166216"></a><a name="p1246314166216"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p348222314341"><a name="p348222314341"></a><a name="p348222314341"></a>ceil_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1048262319349"><a name="p1048262319349"></a><a name="p1048262319349"></a>ceil_npu_</p>
</td>
</tr>
<tr id="row127031039151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1463516628"><a name="p1463516628"></a><a name="p1463516628"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14482122318349"><a name="p14482122318349"></a><a name="p14482122318349"></a>ceil.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2482192363414"><a name="p2482192363414"></a><a name="p2482192363414"></a>ceil_out_npu</p>
</td>
</tr>
<tr id="row147031239181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94637162217"><a name="p94637162217"></a><a name="p94637162217"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1348372363411"><a name="p1348372363411"></a><a name="p1348372363411"></a>clamp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44834239341"><a name="p44834239341"></a><a name="p44834239341"></a>clamp_npu</p>
</td>
</tr>
<tr id="row7703143911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846351612211"><a name="p846351612211"></a><a name="p846351612211"></a>84</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11483123143412"><a name="p11483123143412"></a><a name="p11483123143412"></a>clamp_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148382316343"><a name="p1148382316343"></a><a name="p1148382316343"></a>clamp_npu_</p>
</td>
</tr>
<tr id="row137031396123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154635161025"><a name="p154635161025"></a><a name="p154635161025"></a>85</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p144838239346"><a name="p144838239346"></a><a name="p144838239346"></a>clamp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134831423143414"><a name="p134831423143414"></a><a name="p134831423143414"></a>clamp_out_npu</p>
</td>
</tr>
<tr id="row12703133911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146311617218"><a name="p146311617218"></a><a name="p146311617218"></a>86</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114831523173416"><a name="p114831523173416"></a><a name="p114831523173416"></a>clamp_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94831323163419"><a name="p94831323163419"></a><a name="p94831323163419"></a>clamp_max_npu</p>
</td>
</tr>
<tr id="row37031139181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4463416326"><a name="p4463416326"></a><a name="p4463416326"></a>87</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19483162373417"><a name="p19483162373417"></a><a name="p19483162373417"></a>clamp_max_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p648315235347"><a name="p648315235347"></a><a name="p648315235347"></a>clamp_max_npu_</p>
</td>
</tr>
<tr id="row12703123961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7463816828"><a name="p7463816828"></a><a name="p7463816828"></a>88</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5483122353418"><a name="p5483122353418"></a><a name="p5483122353418"></a>clamp_max.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p74831523153412"><a name="p74831523153412"></a><a name="p74831523153412"></a>clamp_max_out_npu</p>
</td>
</tr>
<tr id="row170473991217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12463516727"><a name="p12463516727"></a><a name="p12463516727"></a>89</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1848313231346"><a name="p1848313231346"></a><a name="p1848313231346"></a>clamp_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1648362373418"><a name="p1648362373418"></a><a name="p1648362373418"></a>clamp_min_npu</p>
</td>
</tr>
<tr id="row370416391125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17463181613212"><a name="p17463181613212"></a><a name="p17463181613212"></a>90</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17483162311349"><a name="p17483162311349"></a><a name="p17483162311349"></a>clamp_min_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8483102363415"><a name="p8483102363415"></a><a name="p8483102363415"></a>clamp_min_npu_</p>
</td>
</tr>
<tr id="row12704173941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246381613220"><a name="p1246381613220"></a><a name="p1246381613220"></a>91</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10483102333415"><a name="p10483102333415"></a><a name="p10483102333415"></a>clamp_min.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p248382393414"><a name="p248382393414"></a><a name="p248382393414"></a>clamp_min_out_npu</p>
</td>
</tr>
<tr id="row6704239131211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14464116423"><a name="p14464116423"></a><a name="p14464116423"></a>92</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18483162393417"><a name="p18483162393417"></a><a name="p18483162393417"></a>constant_pad_nd</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1948316231347"><a name="p1948316231347"></a><a name="p1948316231347"></a>constant_pad_nd_npu</p>
</td>
</tr>
<tr id="row1570493911129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15464141617210"><a name="p15464141617210"></a><a name="p15464141617210"></a>93</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1248314237347"><a name="p1248314237347"></a><a name="p1248314237347"></a>contiguous</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194834238343"><a name="p194834238343"></a><a name="p194834238343"></a>contiguous_npu</p>
</td>
</tr>
<tr id="row27048393125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104644164219"><a name="p104644164219"></a><a name="p104644164219"></a>94</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p144831823103416"><a name="p144831823103416"></a><a name="p144831823103416"></a>convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p648319235345"><a name="p648319235345"></a><a name="p648319235345"></a>convolution_npu</p>
</td>
</tr>
<tr id="row6704173911219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p246411614210"><a name="p246411614210"></a><a name="p246411614210"></a>95</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194831823133418"><a name="p194831823133418"></a><a name="p194831823133418"></a>_convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6483023193418"><a name="p6483023193418"></a><a name="p6483023193418"></a>_convolution_npu</p>
</td>
</tr>
<tr id="row1070423914123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12464201619213"><a name="p12464201619213"></a><a name="p12464201619213"></a>96</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7483172383413"><a name="p7483172383413"></a><a name="p7483172383413"></a>_convolution_nogroup</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p848372383414"><a name="p848372383414"></a><a name="p848372383414"></a>_convolution_nogroup_npu</p>
</td>
</tr>
<tr id="row1704193951216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144646166213"><a name="p144646166213"></a><a name="p144646166213"></a>97</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1448312235345"><a name="p1448312235345"></a><a name="p1448312235345"></a>conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134831823173417"><a name="p134831823173417"></a><a name="p134831823173417"></a>conv2d_npu_</p>
</td>
</tr>
<tr id="row14704113914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84643166215"><a name="p84643166215"></a><a name="p84643166215"></a>98</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p148362313349"><a name="p148362313349"></a><a name="p148362313349"></a>conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p448442353410"><a name="p448442353410"></a><a name="p448442353410"></a>_conv3d_npu</p>
</td>
</tr>
<tr id="row207047394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154641616627"><a name="p154641616627"></a><a name="p154641616627"></a>99</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19484123193417"><a name="p19484123193417"></a><a name="p19484123193417"></a>conv_tbc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11484142317343"><a name="p11484142317343"></a><a name="p11484142317343"></a>conv_tbc_npu</p>
</td>
</tr>
<tr id="row14705103916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19464181613216"><a name="p19464181613216"></a><a name="p19464181613216"></a>100</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p948414235341"><a name="p948414235341"></a><a name="p948414235341"></a>conv_tbc_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1448412239346"><a name="p1448412239346"></a><a name="p1448412239346"></a>conv_tbc_backward_npu</p>
</td>
</tr>
<tr id="row15705193961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6464131617218"><a name="p6464131617218"></a><a name="p6464131617218"></a>101</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34841923153412"><a name="p34841923153412"></a><a name="p34841923153412"></a>conv_transpose2d.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4484172313344"><a name="p4484172313344"></a><a name="p4484172313344"></a>conv_transpose2d_npu_</p>
</td>
</tr>
<tr id="row270513391126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p946419161529"><a name="p946419161529"></a><a name="p946419161529"></a>102</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18484023183419"><a name="p18484023183419"></a><a name="p18484023183419"></a>conv_transpose3d.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p648482316345"><a name="p648482316345"></a><a name="p648482316345"></a>conv_transpose3d_npu_</p>
</td>
</tr>
<tr id="row15705153951215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114647166218"><a name="p114647166218"></a><a name="p114647166218"></a>103</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8484142333413"><a name="p8484142333413"></a><a name="p8484142333413"></a>copy_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1348419236341"><a name="p1348419236341"></a><a name="p1348419236341"></a>copy_npu_</p>
</td>
</tr>
<tr id="row970573915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164642016822"><a name="p164642016822"></a><a name="p164642016822"></a>104</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1348410237348"><a name="p1348410237348"></a><a name="p1348410237348"></a>cos</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1548402319346"><a name="p1548402319346"></a><a name="p1548402319346"></a>cos_npu</p>
</td>
</tr>
<tr id="row107052039171216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246413168213"><a name="p1246413168213"></a><a name="p1246413168213"></a>105</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p54845232347"><a name="p54845232347"></a><a name="p54845232347"></a>cos_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18484142383419"><a name="p18484142383419"></a><a name="p18484142383419"></a>cos_npu_</p>
</td>
</tr>
<tr id="row17705203951218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04648168215"><a name="p04648168215"></a><a name="p04648168215"></a>106</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14484192319344"><a name="p14484192319344"></a><a name="p14484192319344"></a>cos.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54841231347"><a name="p54841231347"></a><a name="p54841231347"></a>cos_out_npu</p>
</td>
</tr>
<tr id="row1470543918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16465201612216"><a name="p16465201612216"></a><a name="p16465201612216"></a>107</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1648452333413"><a name="p1648452333413"></a><a name="p1648452333413"></a>cosh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5484423103412"><a name="p5484423103412"></a><a name="p5484423103412"></a>cosh_npu</p>
</td>
</tr>
<tr id="row12707133981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13465171617216"><a name="p13465171617216"></a><a name="p13465171617216"></a>108</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16484112310349"><a name="p16484112310349"></a><a name="p16484112310349"></a>cosh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24845231340"><a name="p24845231340"></a><a name="p24845231340"></a>cosh_npu_</p>
</td>
</tr>
<tr id="row197089397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19465216628"><a name="p19465216628"></a><a name="p19465216628"></a>109</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p148413238341"><a name="p148413238341"></a><a name="p148413238341"></a>cosh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54841123163417"><a name="p54841123163417"></a><a name="p54841123163417"></a>cosh_out_npu</p>
</td>
</tr>
<tr id="row147081039121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104652165215"><a name="p104652165215"></a><a name="p104652165215"></a>110</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p74849231343"><a name="p74849231343"></a><a name="p74849231343"></a>_cummax_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14484162310349"><a name="p14484162310349"></a><a name="p14484162310349"></a>cummax_helper_npu</p>
</td>
</tr>
<tr id="row1470863918121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1046561620217"><a name="p1046561620217"></a><a name="p1046561620217"></a>111</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p94841223143412"><a name="p94841223143412"></a><a name="p94841223143412"></a>_cummin_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20484723123419"><a name="p20484723123419"></a><a name="p20484723123419"></a>cummin_helper_npu</p>
</td>
</tr>
<tr id="row8708203951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14651162210"><a name="p14651162210"></a><a name="p14651162210"></a>112</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p248413232347"><a name="p248413232347"></a><a name="p248413232347"></a>cumprod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124841623163410"><a name="p124841623163410"></a><a name="p124841623163410"></a>cumprod_npu</p>
</td>
</tr>
<tr id="row8708103941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446541617213"><a name="p446541617213"></a><a name="p446541617213"></a>113</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p04848237342"><a name="p04848237342"></a><a name="p04848237342"></a>cumprod.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144841223193418"><a name="p144841223193418"></a><a name="p144841223193418"></a>cumprod_out_npu</p>
</td>
</tr>
<tr id="row17708143911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134651116426"><a name="p134651116426"></a><a name="p134651116426"></a>114</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1485323173415"><a name="p1485323173415"></a><a name="p1485323173415"></a>cumprod.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p154851723143414"><a name="p154851723143414"></a><a name="p154851723143414"></a>cumprod_npu</p>
</td>
</tr>
<tr id="row11708839101210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12465161617219"><a name="p12465161617219"></a><a name="p12465161617219"></a>115</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1748502316342"><a name="p1748502316342"></a><a name="p1748502316342"></a>cumprod.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14485162316344"><a name="p14485162316344"></a><a name="p14485162316344"></a>cumprod_out_npu</p>
</td>
</tr>
<tr id="row1870815396128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p204653163214"><a name="p204653163214"></a><a name="p204653163214"></a>116</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16485152311340"><a name="p16485152311340"></a><a name="p16485152311340"></a>ctc_loss.IntList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8485152313419"><a name="p8485152313419"></a><a name="p8485152313419"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row77081539121218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15465151619215"><a name="p15465151619215"></a><a name="p15465151619215"></a>117</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p848552383415"><a name="p848552383415"></a><a name="p848552383415"></a>ctc_loss.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p948518238349"><a name="p948518238349"></a><a name="p948518238349"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row18708123911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1346541613215"><a name="p1346541613215"></a><a name="p1346541613215"></a>118</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6485182323411"><a name="p6485182323411"></a><a name="p6485182323411"></a>_ctc_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5485192311346"><a name="p5485192311346"></a><a name="p5485192311346"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row15708153941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164652161822"><a name="p164652161822"></a><a name="p164652161822"></a>119</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16485123153415"><a name="p16485123153415"></a><a name="p16485123153415"></a>_ctc_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19485202317348"><a name="p19485202317348"></a><a name="p19485202317348"></a>ctc_loss_backward_npu</p>
</td>
</tr>
<tr id="row147081539111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34651516924"><a name="p34651516924"></a><a name="p34651516924"></a>120</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12485152319349"><a name="p12485152319349"></a><a name="p12485152319349"></a>fill_diagonal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34855239341"><a name="p34855239341"></a><a name="p34855239341"></a>fill_diagonal_npu_</p>
</td>
</tr>
<tr id="row47091839121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546510161423"><a name="p546510161423"></a><a name="p546510161423"></a>121</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p94851723203415"><a name="p94851723203415"></a><a name="p94851723203415"></a>div.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11485172333419"><a name="p11485172333419"></a><a name="p11485172333419"></a>div_npu</p>
</td>
</tr>
<tr id="row18709183971211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846571616211"><a name="p1846571616211"></a><a name="p1846571616211"></a>122</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14485132311343"><a name="p14485132311343"></a><a name="p14485132311343"></a>div_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5485823153414"><a name="p5485823153414"></a><a name="p5485823153414"></a>div_npu_</p>
</td>
</tr>
<tr id="row07096390129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1465121614218"><a name="p1465121614218"></a><a name="p1465121614218"></a>123</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9485132353410"><a name="p9485132353410"></a><a name="p9485132353410"></a>div.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1848513233341"><a name="p1848513233341"></a><a name="p1848513233341"></a>div_out_npu</p>
</td>
</tr>
<tr id="row1370903971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446518161627"><a name="p446518161627"></a><a name="p446518161627"></a>124</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p204851423173416"><a name="p204851423173416"></a><a name="p204851423173416"></a>div.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144856231348"><a name="p144856231348"></a><a name="p144856231348"></a>div_npu</p>
</td>
</tr>
<tr id="row070993961220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84653161124"><a name="p84653161124"></a><a name="p84653161124"></a>125</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p04851723113411"><a name="p04851723113411"></a><a name="p04851723113411"></a>div_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3485142311344"><a name="p3485142311344"></a><a name="p3485142311344"></a>div_npu_</p>
</td>
</tr>
<tr id="row0709143941212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114665161227"><a name="p114665161227"></a><a name="p114665161227"></a>126</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1548572323410"><a name="p1548572323410"></a><a name="p1548572323410"></a>dot</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144851723103411"><a name="p144851723103411"></a><a name="p144851723103411"></a>dot_npu</p>
</td>
</tr>
<tr id="row1570919393128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546613161522"><a name="p1546613161522"></a><a name="p1546613161522"></a>127</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1748519238347"><a name="p1748519238347"></a><a name="p1748519238347"></a>dot.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34857234344"><a name="p34857234344"></a><a name="p34857234344"></a>dot_out_npu</p>
</td>
</tr>
<tr id="row670933917126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154661516327"><a name="p154661516327"></a><a name="p154661516327"></a>128</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24854230341"><a name="p24854230341"></a><a name="p24854230341"></a>embedding</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20485172319341"><a name="p20485172319341"></a><a name="p20485172319341"></a>embedding_npu</p>
</td>
</tr>
<tr id="row770983915126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546612161524"><a name="p546612161524"></a><a name="p546612161524"></a>129</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9486223103419"><a name="p9486223103419"></a><a name="p9486223103419"></a>embedding_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3486323113414"><a name="p3486323113414"></a><a name="p3486323113414"></a>embedding_backward_npu</p>
</td>
</tr>
<tr id="row57091739171214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746617161425"><a name="p1746617161425"></a><a name="p1746617161425"></a>130</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15486182312341"><a name="p15486182312341"></a><a name="p15486182312341"></a>embedding_dense_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p64861923163415"><a name="p64861923163415"></a><a name="p64861923163415"></a>embedding_dense_backward_npu</p>
</td>
</tr>
<tr id="row1710123916127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p946671618212"><a name="p946671618212"></a><a name="p946671618212"></a>131</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p74861123163418"><a name="p74861123163418"></a><a name="p74861123163418"></a>embedding_renorm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114861023203415"><a name="p114861023203415"></a><a name="p114861023203415"></a>embedding_renorm_npu_</p>
</td>
</tr>
<tr id="row1871033917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7466181613217"><a name="p7466181613217"></a><a name="p7466181613217"></a>132</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12486122319344"><a name="p12486122319344"></a><a name="p12486122319344"></a>_embedding_bag</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16486123143412"><a name="p16486123143412"></a><a name="p16486123143412"></a>_embedding_bag_npu</p>
</td>
</tr>
<tr id="row471017396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846691612218"><a name="p1846691612218"></a><a name="p1846691612218"></a>133</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p248682383418"><a name="p248682383418"></a><a name="p248682383418"></a>empty.memory_format</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134861923133419"><a name="p134861923133419"></a><a name="p134861923133419"></a>empty_npu</p>
</td>
</tr>
<tr id="row87101939181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184668167214"><a name="p184668167214"></a><a name="p184668167214"></a>134</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24861223193418"><a name="p24861223193418"></a><a name="p24861223193418"></a>resize_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1748614234340"><a name="p1748614234340"></a><a name="p1748614234340"></a>resize_npu_</p>
</td>
</tr>
<tr id="row7710193911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346618161427"><a name="p346618161427"></a><a name="p346618161427"></a>135</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14486182310348"><a name="p14486182310348"></a><a name="p14486182310348"></a>empty_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17486723193417"><a name="p17486723193417"></a><a name="p17486723193417"></a>empty_like_npu</p>
</td>
</tr>
<tr id="row1871053961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174663161022"><a name="p174663161022"></a><a name="p174663161022"></a>136</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9486142311344"><a name="p9486142311344"></a><a name="p9486142311344"></a>empty_strided</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24869233341"><a name="p24869233341"></a><a name="p24869233341"></a>empty_strided_npu</p>
</td>
</tr>
<tr id="row87101439151213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546691616213"><a name="p1546691616213"></a><a name="p1546691616213"></a>137</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17486192313410"><a name="p17486192313410"></a><a name="p17486192313410"></a>erf</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p184861523153415"><a name="p184861523153415"></a><a name="p184861523153415"></a>erf_npu</p>
</td>
</tr>
<tr id="row9710113951217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846610161326"><a name="p1846610161326"></a><a name="p1846610161326"></a>138</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1248616233349"><a name="p1248616233349"></a><a name="p1248616233349"></a>erf_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p948611233349"><a name="p948611233349"></a><a name="p948611233349"></a>erf_npu_</p>
</td>
</tr>
<tr id="row4710143961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846614162214"><a name="p846614162214"></a><a name="p846614162214"></a>139</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p948692319341"><a name="p948692319341"></a><a name="p948692319341"></a>erf.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15486192310342"><a name="p15486192310342"></a><a name="p15486192310342"></a>erf_out_npu</p>
</td>
</tr>
<tr id="row107101539181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184671416828"><a name="p184671416828"></a><a name="p184671416828"></a>140</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p648617237342"><a name="p648617237342"></a><a name="p648617237342"></a>erfc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24861323183413"><a name="p24861323183413"></a><a name="p24861323183413"></a>erfc_npu</p>
</td>
</tr>
<tr id="row12710739111212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24678168216"><a name="p24678168216"></a><a name="p24678168216"></a>141</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1486112314345"><a name="p1486112314345"></a><a name="p1486112314345"></a>erfc_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3486823103412"><a name="p3486823103412"></a><a name="p3486823103412"></a>erfc_npu_</p>
</td>
</tr>
<tr id="row1771193971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24672161213"><a name="p24672161213"></a><a name="p24672161213"></a>142</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1548611233347"><a name="p1548611233347"></a><a name="p1548611233347"></a>erfc.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19486192315349"><a name="p19486192315349"></a><a name="p19486192315349"></a>erfc_out_npu</p>
</td>
</tr>
<tr id="row12711193917124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146716168216"><a name="p1146716168216"></a><a name="p1146716168216"></a>143</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p54861523133416"><a name="p54861523133416"></a><a name="p54861523133416"></a>exp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2486182317349"><a name="p2486182317349"></a><a name="p2486182317349"></a>exp_npu</p>
</td>
</tr>
<tr id="row5711439191211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246791614218"><a name="p1246791614218"></a><a name="p1246791614218"></a>144</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17486142312346"><a name="p17486142312346"></a><a name="p17486142312346"></a>exp_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8487162316341"><a name="p8487162316341"></a><a name="p8487162316341"></a>exp_npu_</p>
</td>
</tr>
<tr id="row8711113910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646741611219"><a name="p1646741611219"></a><a name="p1646741611219"></a>145</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1248792373416"><a name="p1248792373416"></a><a name="p1248792373416"></a>exp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5487323183414"><a name="p5487323183414"></a><a name="p5487323183414"></a>exp_out_npu</p>
</td>
</tr>
<tr id="row107111639131216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846717161126"><a name="p846717161126"></a><a name="p846717161126"></a>146</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7487723123413"><a name="p7487723123413"></a><a name="p7487723123413"></a>expm1</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p184871023173411"><a name="p184871023173411"></a><a name="p184871023173411"></a>expm1_npu</p>
</td>
</tr>
<tr id="row18711103921219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12467416324"><a name="p12467416324"></a><a name="p12467416324"></a>147</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4487142320341"><a name="p4487142320341"></a><a name="p4487142320341"></a>expm1_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1248762323418"><a name="p1248762323418"></a><a name="p1248762323418"></a>expm1_npu_</p>
</td>
</tr>
<tr id="row14711839151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446713161229"><a name="p446713161229"></a><a name="p446713161229"></a>148</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114874234343"><a name="p114874234343"></a><a name="p114874234343"></a>expm1.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44871723143412"><a name="p44871723143412"></a><a name="p44871723143412"></a>expm1_out_npu</p>
</td>
</tr>
<tr id="row67121739171219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34682016926"><a name="p34682016926"></a><a name="p34682016926"></a>149</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1048719238346"><a name="p1048719238346"></a><a name="p1048719238346"></a>eye</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1148720239348"><a name="p1148720239348"></a><a name="p1148720239348"></a>eye_npu</p>
</td>
</tr>
<tr id="row167127398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44689161523"><a name="p44689161523"></a><a name="p44689161523"></a>150</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1448719232344"><a name="p1448719232344"></a><a name="p1448719232344"></a>eye.m</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10487142319341"><a name="p10487142319341"></a><a name="p10487142319341"></a>eye_npu</p>
</td>
</tr>
<tr id="row2712123912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154681916524"><a name="p154681916524"></a><a name="p154681916524"></a>151</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1848782312341"><a name="p1848782312341"></a><a name="p1848782312341"></a>eye.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44877231343"><a name="p44877231343"></a><a name="p44877231343"></a>eye_out_npu</p>
</td>
</tr>
<tr id="row157121739161214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24685161225"><a name="p24685161225"></a><a name="p24685161225"></a>152</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8487152333419"><a name="p8487152333419"></a><a name="p8487152333419"></a>eye.m_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7487523183415"><a name="p7487523183415"></a><a name="p7487523183415"></a>eye_out_npu</p>
</td>
</tr>
<tr id="row1171283971211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146881616216"><a name="p1146881616216"></a><a name="p1146881616216"></a>153</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84871232347"><a name="p84871232347"></a><a name="p84871232347"></a>fill_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19487523173411"><a name="p19487523173411"></a><a name="p19487523173411"></a>fill_npu_</p>
</td>
</tr>
<tr id="row15712439111216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146815164212"><a name="p146815164212"></a><a name="p146815164212"></a>154</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1487122312344"><a name="p1487122312344"></a><a name="p1487122312344"></a>fill_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p748772393418"><a name="p748772393418"></a><a name="p748772393418"></a>fill_npu_</p>
</td>
</tr>
<tr id="row18712133915128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p646821618213"><a name="p646821618213"></a><a name="p646821618213"></a>155</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p144871623143420"><a name="p144871623143420"></a><a name="p144871623143420"></a>floor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1448718236344"><a name="p1448718236344"></a><a name="p1448718236344"></a>floor_npu</p>
</td>
</tr>
<tr id="row171243912124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184689161121"><a name="p184689161121"></a><a name="p184689161121"></a>156</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124875236349"><a name="p124875236349"></a><a name="p124875236349"></a>floor_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1848702318345"><a name="p1848702318345"></a><a name="p1848702318345"></a>floor_npu_</p>
</td>
</tr>
<tr id="row07121539141212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16468181619218"><a name="p16468181619218"></a><a name="p16468181619218"></a>157</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3487152310344"><a name="p3487152310344"></a><a name="p3487152310344"></a>floor.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14487423133417"><a name="p14487423133417"></a><a name="p14487423133417"></a>floor_out_npu</p>
</td>
</tr>
<tr id="row071373901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146814162215"><a name="p1146814162215"></a><a name="p1146814162215"></a>158</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7487123203415"><a name="p7487123203415"></a><a name="p7487123203415"></a>floor_divide</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20487112314340"><a name="p20487112314340"></a><a name="p20487112314340"></a>floor_divide_npu</p>
</td>
</tr>
<tr id="row107131393122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546810161821"><a name="p1546810161821"></a><a name="p1546810161821"></a>159</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14871123203415"><a name="p14871123203415"></a><a name="p14871123203415"></a>floor_divide_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p94871423153418"><a name="p94871423153418"></a><a name="p94871423153418"></a>floor_divide_npu_</p>
</td>
</tr>
<tr id="row671383921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134681616022"><a name="p134681616022"></a><a name="p134681616022"></a>160</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14488123183416"><a name="p14488123183416"></a><a name="p14488123183416"></a>floor_divide.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19488223183413"><a name="p19488223183413"></a><a name="p19488223183413"></a>floor_divide_out_npu</p>
</td>
</tr>
<tr id="row1171303931210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7468616423"><a name="p7468616423"></a><a name="p7468616423"></a>161</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15488112317347"><a name="p15488112317347"></a><a name="p15488112317347"></a>floor_divide.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1488623103415"><a name="p1488623103415"></a><a name="p1488623103415"></a>floor_divide_npu</p>
</td>
</tr>
<tr id="row117131339161213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646841618215"><a name="p1646841618215"></a><a name="p1646841618215"></a>162</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17488122319341"><a name="p17488122319341"></a><a name="p17488122319341"></a>floor_divide_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134883231346"><a name="p134883231346"></a><a name="p134883231346"></a>floor_divide_npu_</p>
</td>
</tr>
<tr id="row771333941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24691816223"><a name="p24691816223"></a><a name="p24691816223"></a>163</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p54887234347"><a name="p54887234347"></a><a name="p54887234347"></a>frac</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4488623133414"><a name="p4488623133414"></a><a name="p4488623133414"></a>frac_npu</p>
</td>
</tr>
<tr id="row371317396123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346916163210"><a name="p346916163210"></a><a name="p346916163210"></a>164</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p948812311347"><a name="p948812311347"></a><a name="p948812311347"></a>frac_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6488172320345"><a name="p6488172320345"></a><a name="p6488172320345"></a>frac_npu_</p>
</td>
</tr>
<tr id="row1871317392121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154692016822"><a name="p154692016822"></a><a name="p154692016822"></a>165</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13488723113412"><a name="p13488723113412"></a><a name="p13488723113412"></a>frac.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4488123193416"><a name="p4488123193416"></a><a name="p4488123193416"></a>frac_out_npu</p>
</td>
</tr>
<tr id="row971313918123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546913161320"><a name="p1546913161320"></a><a name="p1546913161320"></a>166</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1488122343412"><a name="p1488122343412"></a><a name="p1488122343412"></a>full.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1348815232344"><a name="p1348815232344"></a><a name="p1348815232344"></a>full_npu</p>
</td>
</tr>
<tr id="row2713939191216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18469416720"><a name="p18469416720"></a><a name="p18469416720"></a>167</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114881223103412"><a name="p114881223103412"></a><a name="p114881223103412"></a>full</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1948819231343"><a name="p1948819231343"></a><a name="p1948819231343"></a>full_npu</p>
</td>
</tr>
<tr id="row107131039161211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1446915162212"><a name="p1446915162212"></a><a name="p1446915162212"></a>168</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114881523193410"><a name="p114881523193410"></a><a name="p114881523193410"></a>full.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18488112313344"><a name="p18488112313344"></a><a name="p18488112313344"></a>full_out_npu</p>
</td>
</tr>
<tr id="row15714103901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44691016722"><a name="p44691016722"></a><a name="p44691016722"></a>169</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4488122310348"><a name="p4488122310348"></a><a name="p4488122310348"></a>grid_sampler</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7488162323419"><a name="p7488162323419"></a><a name="p7488162323419"></a>grid_sampler_npu</p>
</td>
</tr>
<tr id="row37144394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146921618216"><a name="p1146921618216"></a><a name="p1146921618216"></a>170</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13488142393410"><a name="p13488142393410"></a><a name="p13488142393410"></a>grid_sampler_3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2488182343420"><a name="p2488182343420"></a><a name="p2488182343420"></a>grid_sampler_3d_npu</p>
</td>
</tr>
<tr id="row107141639111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4469121610218"><a name="p4469121610218"></a><a name="p4469121610218"></a>171</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9488102353413"><a name="p9488102353413"></a><a name="p9488102353413"></a>grid_sampler_3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4488102312343"><a name="p4488102312343"></a><a name="p4488102312343"></a>grid_sampler_3d_backward_npu</p>
</td>
</tr>
<tr id="row207141396120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546911161924"><a name="p1546911161924"></a><a name="p1546911161924"></a>172</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1148882316344"><a name="p1148882316344"></a><a name="p1148882316344"></a>hann_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p848872343419"><a name="p848872343419"></a><a name="p848872343419"></a>hann_window_npu</p>
</td>
</tr>
<tr id="row2714143971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19469131611216"><a name="p19469131611216"></a><a name="p19469131611216"></a>173</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10488102311342"><a name="p10488102311342"></a><a name="p10488102311342"></a>hann_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14883232348"><a name="p14883232348"></a><a name="p14883232348"></a>hann_window_npu</p>
</td>
</tr>
<tr id="row871433991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146910166212"><a name="p146910166212"></a><a name="p146910166212"></a>174</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p648811236349"><a name="p648811236349"></a><a name="p648811236349"></a>hamming_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44881223113410"><a name="p44881223113410"></a><a name="p44881223113410"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row371493914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646918161724"><a name="p1646918161724"></a><a name="p1646918161724"></a>175</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p948815231346"><a name="p948815231346"></a><a name="p948815231346"></a>hamming_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1248952373417"><a name="p1248952373417"></a><a name="p1248952373417"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row471433931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34691316923"><a name="p34691316923"></a><a name="p34691316923"></a>176</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7489122311343"><a name="p7489122311343"></a><a name="p7489122311343"></a>hamming_window.periodic_alpha</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p248919233344"><a name="p248919233344"></a><a name="p248919233344"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row9714173971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74692161215"><a name="p74692161215"></a><a name="p74692161215"></a>177</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3489152373419"><a name="p3489152373419"></a><a name="p3489152373419"></a>hamming_window.periodic_alpha_beta</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p54891323133416"><a name="p54891323133416"></a><a name="p54891323133416"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row187141539171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54692016828"><a name="p54692016828"></a><a name="p54692016828"></a>178</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1448918238346"><a name="p1448918238346"></a><a name="p1448918238346"></a>ger</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2048920237346"><a name="p2048920237346"></a><a name="p2048920237346"></a>ger_npu</p>
</td>
</tr>
<tr id="row1714183941217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144693161523"><a name="p144693161523"></a><a name="p144693161523"></a>179</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p948952310349"><a name="p948952310349"></a><a name="p948952310349"></a>ger.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13489162353418"><a name="p13489162353418"></a><a name="p13489162353418"></a>ger_out_npu</p>
</td>
</tr>
<tr id="row4715193981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547061610211"><a name="p547061610211"></a><a name="p547061610211"></a>180</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p54891923193418"><a name="p54891923193418"></a><a name="p54891923193418"></a>index.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p248911233344"><a name="p248911233344"></a><a name="p248911233344"></a>index_npu</p>
</td>
</tr>
<tr id="row1715193921210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047012161524"><a name="p1047012161524"></a><a name="p1047012161524"></a>181</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p204891223153411"><a name="p204891223153411"></a><a name="p204891223153411"></a>index_put_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p84891323193419"><a name="p84891323193419"></a><a name="p84891323193419"></a>index_put_npu_</p>
</td>
</tr>
<tr id="row1671583917123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154708168213"><a name="p154708168213"></a><a name="p154708168213"></a>182</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3489152383414"><a name="p3489152383414"></a><a name="p3489152383414"></a>index_put</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124899238348"><a name="p124899238348"></a><a name="p124899238348"></a>index_put_npu</p>
</td>
</tr>
<tr id="row5715339141220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347041613213"><a name="p1347041613213"></a><a name="p1347041613213"></a>183</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2048913239346"><a name="p2048913239346"></a><a name="p2048913239346"></a>_index_put_impl_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12489192315345"><a name="p12489192315345"></a><a name="p12489192315345"></a>_index_put_impl_npu_</p>
</td>
</tr>
<tr id="row771512390120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19470111613218"><a name="p19470111613218"></a><a name="p19470111613218"></a>184</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1489182343414"><a name="p1489182343414"></a><a name="p1489182343414"></a>inverse</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6489172318343"><a name="p6489172318343"></a><a name="p6489172318343"></a>inverse_npu</p>
</td>
</tr>
<tr id="row14715439151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74701416926"><a name="p74701416926"></a><a name="p74701416926"></a>185</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1448982323410"><a name="p1448982323410"></a><a name="p1448982323410"></a>inverse.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34891237342"><a name="p34891237342"></a><a name="p34891237342"></a>inverse_out_npu</p>
</td>
</tr>
<tr id="row127151139161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14701166211"><a name="p14701166211"></a><a name="p14701166211"></a>186</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5489172383419"><a name="p5489172383419"></a><a name="p5489172383419"></a>isclose</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194891523113412"><a name="p194891523113412"></a><a name="p194891523113412"></a>isclose_npu</p>
</td>
</tr>
<tr id="row137154396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547011161120"><a name="p1547011161120"></a><a name="p1547011161120"></a>187</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1489112318346"><a name="p1489112318346"></a><a name="p1489112318346"></a>isnan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1848972320346"><a name="p1848972320346"></a><a name="p1848972320346"></a>isnan_npu</p>
</td>
</tr>
<tr id="row1071553981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7470161619217"><a name="p7470161619217"></a><a name="p7470161619217"></a>188</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p204895232343"><a name="p204895232343"></a><a name="p204895232343"></a>is_nonzero</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1848982353420"><a name="p1848982353420"></a><a name="p1848982353420"></a>is_nonzero_npu</p>
</td>
</tr>
<tr id="row1871533901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747014161825"><a name="p747014161825"></a><a name="p747014161825"></a>189</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1348913237341"><a name="p1348913237341"></a><a name="p1348913237341"></a>kl_div</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p848922315349"><a name="p848922315349"></a><a name="p848922315349"></a>kl_div_npu</p>
</td>
</tr>
<tr id="row14715173914125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174701160210"><a name="p174701160210"></a><a name="p174701160210"></a>190</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5489623103412"><a name="p5489623103412"></a><a name="p5489623103412"></a>kl_div_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24900235347"><a name="p24900235347"></a><a name="p24900235347"></a>kl_div_backward_npu</p>
</td>
</tr>
<tr id="row07162395125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3470121616216"><a name="p3470121616216"></a><a name="p3470121616216"></a>191</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18490192315346"><a name="p18490192315346"></a><a name="p18490192315346"></a>kthvalue</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4490112311346"><a name="p4490112311346"></a><a name="p4490112311346"></a>kthvalue_npu</p>
</td>
</tr>
<tr id="row15716639181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114709169219"><a name="p114709169219"></a><a name="p114709169219"></a>192</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34904236345"><a name="p34904236345"></a><a name="p34904236345"></a>kthvalue.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p249017232347"><a name="p249017232347"></a><a name="p249017232347"></a>kthvalue_out_npu</p>
</td>
</tr>
<tr id="row1671618395124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14706161026"><a name="p14706161026"></a><a name="p14706161026"></a>193</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13490423103417"><a name="p13490423103417"></a><a name="p13490423103417"></a>kthvalue.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5490172343419"><a name="p5490172343419"></a><a name="p5490172343419"></a>kthvalue_npu</p>
</td>
</tr>
<tr id="row12716203961220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447161611210"><a name="p1447161611210"></a><a name="p1447161611210"></a>194</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p449092323412"><a name="p449092323412"></a><a name="p449092323412"></a>kthvalue.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11491162353418"><a name="p11491162353418"></a><a name="p11491162353418"></a>kthvalue_out_npu</p>
</td>
</tr>
<tr id="row15716183918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347131616211"><a name="p1347131616211"></a><a name="p1347131616211"></a>195</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15491152318346"><a name="p15491152318346"></a><a name="p15491152318346"></a>native_layer_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18491192313414"><a name="p18491192313414"></a><a name="p18491192313414"></a>layer_norm_npu</p>
</td>
</tr>
<tr id="row11716143981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74711616929"><a name="p74711616929"></a><a name="p74711616929"></a>196</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8491192319342"><a name="p8491192319342"></a><a name="p8491192319342"></a>native_layer_norm_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1849112353416"><a name="p1849112353416"></a><a name="p1849112353416"></a>layer_norm_backward_npu</p>
</td>
</tr>
<tr id="row3716193911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14471116426"><a name="p14471116426"></a><a name="p14471116426"></a>197</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12491202393418"><a name="p12491202393418"></a><a name="p12491202393418"></a>linspace</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5491123133414"><a name="p5491123133414"></a><a name="p5491123133414"></a>linspace_npu</p>
</td>
</tr>
<tr id="row207171039131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447113161821"><a name="p447113161821"></a><a name="p447113161821"></a>198</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p649162373418"><a name="p649162373418"></a><a name="p649162373418"></a>linspace.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194911823153415"><a name="p194911823153415"></a><a name="p194911823153415"></a>linspace_out_npu</p>
</td>
</tr>
<tr id="row2717113914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647111161126"><a name="p1647111161126"></a><a name="p1647111161126"></a>199</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1049112318346"><a name="p1049112318346"></a><a name="p1049112318346"></a>log</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5491132363418"><a name="p5491132363418"></a><a name="p5491132363418"></a>log_npu</p>
</td>
</tr>
<tr id="row771710399126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134716161920"><a name="p134716161920"></a><a name="p134716161920"></a>200</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13491162313342"><a name="p13491162313342"></a><a name="p13491162313342"></a>log_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p164911623143416"><a name="p164911623143416"></a><a name="p164911623143416"></a>log_npu_</p>
</td>
</tr>
<tr id="row77174392127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4471816729"><a name="p4471816729"></a><a name="p4471816729"></a>201</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3491123143415"><a name="p3491123143415"></a><a name="p3491123143415"></a>log.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1149182315345"><a name="p1149182315345"></a><a name="p1149182315345"></a>log_out_npu</p>
</td>
</tr>
<tr id="row1971733971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44711816725"><a name="p44711816725"></a><a name="p44711816725"></a>202</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18491172310343"><a name="p18491172310343"></a><a name="p18491172310343"></a>log10</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18491182313343"><a name="p18491182313343"></a><a name="p18491182313343"></a>log10_npu</p>
</td>
</tr>
<tr id="row7717939191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447171612212"><a name="p1447171612212"></a><a name="p1447171612212"></a>203</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17491102319342"><a name="p17491102319342"></a><a name="p17491102319342"></a>log10_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7491923143412"><a name="p7491923143412"></a><a name="p7491923143412"></a>log10_npu_</p>
</td>
</tr>
<tr id="row7717103981220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17471101610212"><a name="p17471101610212"></a><a name="p17471101610212"></a>204</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p049114234344"><a name="p049114234344"></a><a name="p049114234344"></a>log10.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8491112383415"><a name="p8491112383415"></a><a name="p8491112383415"></a>log10_out_npu</p>
</td>
</tr>
<tr id="row187181439131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647161610218"><a name="p647161610218"></a><a name="p647161610218"></a>205</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12491112393418"><a name="p12491112393418"></a><a name="p12491112393418"></a>log1p</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1149112313341"><a name="p1149112313341"></a><a name="p1149112313341"></a>log1p_npu</p>
</td>
</tr>
<tr id="row0718139141218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15471116526"><a name="p15471116526"></a><a name="p15471116526"></a>206</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4491423123413"><a name="p4491423123413"></a><a name="p4491423123413"></a>log1p_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p164911523153410"><a name="p164911523153410"></a><a name="p164911523153410"></a>log1p_npu_</p>
</td>
</tr>
<tr id="row571815397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1847118164218"><a name="p1847118164218"></a><a name="p1847118164218"></a>207</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17491323143418"><a name="p17491323143418"></a><a name="p17491323143418"></a>log1p.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8492112318347"><a name="p8492112318347"></a><a name="p8492112318347"></a>log1p_out_npu</p>
</td>
</tr>
<tr id="row187181639141214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14717161323"><a name="p14717161323"></a><a name="p14717161323"></a>208</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0492172312346"><a name="p0492172312346"></a><a name="p0492172312346"></a>log2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1492623203418"><a name="p1492623203418"></a><a name="p1492623203418"></a>log2_npu</p>
</td>
</tr>
<tr id="row117186395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847114161426"><a name="p847114161426"></a><a name="p847114161426"></a>209</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19492102320347"><a name="p19492102320347"></a><a name="p19492102320347"></a>log2_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9492192316347"><a name="p9492192316347"></a><a name="p9492192316347"></a>log2_npu_</p>
</td>
</tr>
<tr id="row1071819393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547114160216"><a name="p547114160216"></a><a name="p547114160216"></a>210</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p174927233347"><a name="p174927233347"></a><a name="p174927233347"></a>log2.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14492723143418"><a name="p14492723143418"></a><a name="p14492723143418"></a>log2_out_npu</p>
</td>
</tr>
<tr id="row37188399121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24713161429"><a name="p24713161429"></a><a name="p24713161429"></a>211</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10492823123420"><a name="p10492823123420"></a><a name="p10492823123420"></a>logspace</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7492923173418"><a name="p7492923173418"></a><a name="p7492923173418"></a>logspace_npu</p>
</td>
</tr>
<tr id="row137187391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194711916925"><a name="p194711916925"></a><a name="p194711916925"></a>212</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1149212311349"><a name="p1149212311349"></a><a name="p1149212311349"></a>logspace.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11492182316348"><a name="p11492182316348"></a><a name="p11492182316348"></a>logspace_out_npu</p>
</td>
</tr>
<tr id="row16718143912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34722016821"><a name="p34722016821"></a><a name="p34722016821"></a>213</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1049219239344"><a name="p1049219239344"></a><a name="p1049219239344"></a>log_softmax.int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p74924236349"><a name="p74924236349"></a><a name="p74924236349"></a>log_softmax_npu</p>
</td>
</tr>
<tr id="row4718103991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647210161321"><a name="p1647210161321"></a><a name="p1647210161321"></a>214</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p174921923153419"><a name="p174921923153419"></a><a name="p174921923153419"></a>log_softmax.Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p44921233340"><a name="p44921233340"></a><a name="p44921233340"></a>log_softmax_npu</p>
</td>
</tr>
<tr id="row271833941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74726164211"><a name="p74726164211"></a><a name="p74726164211"></a>215</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13492102312346"><a name="p13492102312346"></a><a name="p13492102312346"></a>_log_softmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1492223133413"><a name="p1492223133413"></a><a name="p1492223133413"></a>_log_softmax_npu</p>
</td>
</tr>
<tr id="row197181539111220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9472516521"><a name="p9472516521"></a><a name="p9472516521"></a>216</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1549262373417"><a name="p1549262373417"></a><a name="p1549262373417"></a>_log_softmax_backward_data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12492142363410"><a name="p12492142363410"></a><a name="p12492142363410"></a>_log_softmax_backward_npu</p>
</td>
</tr>
<tr id="row8719239121220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10472151610219"><a name="p10472151610219"></a><a name="p10472151610219"></a>217</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6492142313344"><a name="p6492142313344"></a><a name="p6492142313344"></a>logsumexp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p349216237341"><a name="p349216237341"></a><a name="p349216237341"></a>logsumexp_npu</p>
</td>
</tr>
<tr id="row18719173916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047217161822"><a name="p2047217161822"></a><a name="p2047217161822"></a>218</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1149282303417"><a name="p1149282303417"></a><a name="p1149282303417"></a>logsumexp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20492192314349"><a name="p20492192314349"></a><a name="p20492192314349"></a>logsumexp_out_npu</p>
</td>
</tr>
<tr id="row2719103912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184729161128"><a name="p184729161128"></a><a name="p184729161128"></a>219</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7492823193415"><a name="p7492823193415"></a><a name="p7492823193415"></a>logsumexp.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p74921223163412"><a name="p74921223163412"></a><a name="p74921223163412"></a>logsumexp_npu</p>
</td>
</tr>
<tr id="row371910396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347271610213"><a name="p1347271610213"></a><a name="p1347271610213"></a>220</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18492202333416"><a name="p18492202333416"></a><a name="p18492202333416"></a>logsumexp.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19492223183413"><a name="p19492223183413"></a><a name="p19492223183413"></a>logsumexp_out_npu</p>
</td>
</tr>
<tr id="row371919398126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547211618210"><a name="p547211618210"></a><a name="p547211618210"></a>221</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13492723173412"><a name="p13492723173412"></a><a name="p13492723173412"></a>matmul</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3492623123420"><a name="p3492623123420"></a><a name="p3492623123420"></a>matmul_npu</p>
</td>
</tr>
<tr id="row9719439151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947219161422"><a name="p1947219161422"></a><a name="p1947219161422"></a>222</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1249262315345"><a name="p1249262315345"></a><a name="p1249262315345"></a>matmul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1549215230342"><a name="p1549215230342"></a><a name="p1549215230342"></a>matmul_out_npu</p>
</td>
</tr>
<tr id="row2719193921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047213161929"><a name="p1047213161929"></a><a name="p1047213161929"></a>223</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12493112333419"><a name="p12493112333419"></a><a name="p12493112333419"></a>max.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124931723113414"><a name="p124931723113414"></a><a name="p124931723113414"></a>max_npu</p>
</td>
</tr>
<tr id="row1471913910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24727161023"><a name="p24727161023"></a><a name="p24727161023"></a>224</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10493202323420"><a name="p10493202323420"></a><a name="p10493202323420"></a>max.dim_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12493172343420"><a name="p12493172343420"></a><a name="p12493172343420"></a>max_out_npu</p>
</td>
</tr>
<tr id="row197191739101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18472151613210"><a name="p18472151613210"></a><a name="p18472151613210"></a>225</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7493123173414"><a name="p7493123173414"></a><a name="p7493123173414"></a>max_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1493122363410"><a name="p1493122363410"></a><a name="p1493122363410"></a>max_npu</p>
</td>
</tr>
<tr id="row137191939101212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164723162023"><a name="p164723162023"></a><a name="p164723162023"></a>226</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13493142313342"><a name="p13493142313342"></a><a name="p13493142313342"></a>max.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p124932023203410"><a name="p124932023203410"></a><a name="p124932023203410"></a>max_npu</p>
</td>
</tr>
<tr id="row471917394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34727161326"><a name="p34727161326"></a><a name="p34727161326"></a>227</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34931823163412"><a name="p34931823163412"></a><a name="p34931823163412"></a>max.names_dim_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11493523173412"><a name="p11493523173412"></a><a name="p11493523173412"></a>max_out_npu</p>
</td>
</tr>
<tr id="row5720239191220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5473181619210"><a name="p5473181619210"></a><a name="p5473181619210"></a>228</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16493152314345"><a name="p16493152314345"></a><a name="p16493152314345"></a>max_values.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p849316239345"><a name="p849316239345"></a><a name="p849316239345"></a>max_npu</p>
</td>
</tr>
<tr id="row172093913121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047391614217"><a name="p2047391614217"></a><a name="p2047391614217"></a>229</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14493132313345"><a name="p14493132313345"></a><a name="p14493132313345"></a>max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19493523103417"><a name="p19493523103417"></a><a name="p19493523103417"></a>max_pool2d_npu</p>
</td>
</tr>
<tr id="row1772073901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847391612212"><a name="p847391612212"></a><a name="p847391612212"></a>230</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18493172363417"><a name="p18493172363417"></a><a name="p18493172363417"></a>mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1849312393414"><a name="p1849312393414"></a><a name="p1849312393414"></a>mean_npu</p>
</td>
</tr>
<tr id="row17720163931217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847314161525"><a name="p847314161525"></a><a name="p847314161525"></a>231</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2049392310344"><a name="p2049392310344"></a><a name="p2049392310344"></a>mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10493923113420"><a name="p10493923113420"></a><a name="p10493923113420"></a>mean_npu</p>
</td>
</tr>
<tr id="row9720143916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174734164213"><a name="p174734164213"></a><a name="p174734164213"></a>232</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194936238340"><a name="p194936238340"></a><a name="p194936238340"></a>mean.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4493142383420"><a name="p4493142383420"></a><a name="p4493142383420"></a>mean_out_npu</p>
</td>
</tr>
<tr id="row17201339131212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7473116621"><a name="p7473116621"></a><a name="p7473116621"></a>233</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1949372363412"><a name="p1949372363412"></a><a name="p1949372363412"></a>mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14493142314341"><a name="p14493142314341"></a><a name="p14493142314341"></a>mean_npu</p>
</td>
</tr>
<tr id="row57201039191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114738167211"><a name="p114738167211"></a><a name="p114738167211"></a>234</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p04931923183412"><a name="p04931923183412"></a><a name="p04931923183412"></a>mean.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1449318233341"><a name="p1449318233341"></a><a name="p1449318233341"></a>mean_out_npu</p>
</td>
</tr>
<tr id="row1372013918126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p247310161322"><a name="p247310161322"></a><a name="p247310161322"></a>235</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p144931823183414"><a name="p144931823183414"></a><a name="p144931823183414"></a>median.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p449332353413"><a name="p449332353413"></a><a name="p449332353413"></a>median_npu</p>
</td>
</tr>
<tr id="row3720639201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047313163215"><a name="p1047313163215"></a><a name="p1047313163215"></a>236</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4493172383417"><a name="p4493172383417"></a><a name="p4493172383417"></a>median.dim_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p174931623183413"><a name="p174931623183413"></a><a name="p174931623183413"></a>median_out_npu</p>
</td>
</tr>
<tr id="row107201839161213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2473131618212"><a name="p2473131618212"></a><a name="p2473131618212"></a>237</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1493102315344"><a name="p1493102315344"></a><a name="p1493102315344"></a>median.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1149316239346"><a name="p1149316239346"></a><a name="p1149316239346"></a>median_npu</p>
</td>
</tr>
<tr id="row1872083991217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164736161622"><a name="p164736161622"></a><a name="p164736161622"></a>238</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p04930238346"><a name="p04930238346"></a><a name="p04930238346"></a>median.names_dim_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p249482315347"><a name="p249482315347"></a><a name="p249482315347"></a>median_out_npu</p>
</td>
</tr>
<tr id="row1172183941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347314161213"><a name="p1347314161213"></a><a name="p1347314161213"></a>239</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p144945235348"><a name="p144945235348"></a><a name="p144945235348"></a>min.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15494162319340"><a name="p15494162319340"></a><a name="p15494162319340"></a>min_npu</p>
</td>
</tr>
<tr id="row172116399124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747311161124"><a name="p747311161124"></a><a name="p747311161124"></a>240</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1149412393420"><a name="p1149412393420"></a><a name="p1149412393420"></a>min.dim_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11494102353415"><a name="p11494102353415"></a><a name="p11494102353415"></a>min_out_npu</p>
</td>
</tr>
<tr id="row67218393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4473161612212"><a name="p4473161612212"></a><a name="p4473161612212"></a>241</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p134941823103412"><a name="p134941823103412"></a><a name="p134941823103412"></a>min_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19494152318349"><a name="p19494152318349"></a><a name="p19494152318349"></a>min_npu</p>
</td>
</tr>
<tr id="row1672117397122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164741216821"><a name="p164741216821"></a><a name="p164741216821"></a>242</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p449412333412"><a name="p449412333412"></a><a name="p449412333412"></a>min.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17494523203416"><a name="p17494523203416"></a><a name="p17494523203416"></a>min_npu</p>
</td>
</tr>
<tr id="row1572114394124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174741016525"><a name="p174741016525"></a><a name="p174741016525"></a>243</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1549420233348"><a name="p1549420233348"></a><a name="p1549420233348"></a>min.names_dim_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2494192383417"><a name="p2494192383417"></a><a name="p2494192383417"></a>min_out_npu</p>
</td>
</tr>
<tr id="row8721139131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447415165218"><a name="p1447415165218"></a><a name="p1447415165218"></a>244</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2494142353419"><a name="p2494142353419"></a><a name="p2494142353419"></a>min_values.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p249415238349"><a name="p249415238349"></a><a name="p249415238349"></a>min_npu</p>
</td>
</tr>
<tr id="row1072153917123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174744163217"><a name="p174744163217"></a><a name="p174744163217"></a>245</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1249462343418"><a name="p1249462343418"></a><a name="p1249462343418"></a>mm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1494192313344"><a name="p1494192313344"></a><a name="p1494192313344"></a>mm_npu</p>
</td>
</tr>
<tr id="row37219396128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54741916527"><a name="p54741916527"></a><a name="p54741916527"></a>246</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15494182343413"><a name="p15494182343413"></a><a name="p15494182343413"></a>mm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3494112343418"><a name="p3494112343418"></a><a name="p3494112343418"></a>mm_out_npu</p>
</td>
</tr>
<tr id="row1572183912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447412161427"><a name="p447412161427"></a><a name="p447412161427"></a>247</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1949412311341"><a name="p1949412311341"></a><a name="p1949412311341"></a>mul.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1649419231342"><a name="p1649419231342"></a><a name="p1649419231342"></a>mul_npu</p>
</td>
</tr>
<tr id="row167216395126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54741416527"><a name="p54741416527"></a><a name="p54741416527"></a>248</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p104943230345"><a name="p104943230345"></a><a name="p104943230345"></a>mul_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p449462319342"><a name="p449462319342"></a><a name="p449462319342"></a>mul_npu_</p>
</td>
</tr>
<tr id="row137211039151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64749163218"><a name="p64749163218"></a><a name="p64749163218"></a>249</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11494172393415"><a name="p11494172393415"></a><a name="p11494172393415"></a>mul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1749412343411"><a name="p1749412343411"></a><a name="p1749412343411"></a>mul_out_npu</p>
</td>
</tr>
<tr id="row11722153918121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10474161613215"><a name="p10474161613215"></a><a name="p10474161613215"></a>250</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1149432311345"><a name="p1149432311345"></a><a name="p1149432311345"></a>mul.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1649413238349"><a name="p1649413238349"></a><a name="p1649413238349"></a>mul_npu</p>
</td>
</tr>
<tr id="row1472273981219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547414161121"><a name="p547414161121"></a><a name="p547414161121"></a>251</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114941923143418"><a name="p114941923143418"></a><a name="p114941923143418"></a>mul_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p649412313343"><a name="p649412313343"></a><a name="p649412313343"></a>mul_npu_</p>
</td>
</tr>
<tr id="row19722103916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047401616212"><a name="p2047401616212"></a><a name="p2047401616212"></a>252</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1649442316343"><a name="p1649442316343"></a><a name="p1649442316343"></a>mv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p24941123103419"><a name="p24941123103419"></a><a name="p24941123103419"></a>mv_npu</p>
</td>
</tr>
<tr id="row16722143916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p347411161023"><a name="p347411161023"></a><a name="p347411161023"></a>253</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3494102323418"><a name="p3494102323418"></a><a name="p3494102323418"></a>mv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p049452343419"><a name="p049452343419"></a><a name="p049452343419"></a>mv_out_npu</p>
</td>
</tr>
<tr id="row197221239151219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144745161624"><a name="p144745161624"></a><a name="p144745161624"></a>254</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12495162323416"><a name="p12495162323416"></a><a name="p12495162323416"></a>narrow_copy</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6495172303414"><a name="p6495172303414"></a><a name="p6495172303414"></a>narrow_copy_npu</p>
</td>
</tr>
<tr id="row672219394123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547431616212"><a name="p1547431616212"></a><a name="p1547431616212"></a>255</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18495162318342"><a name="p18495162318342"></a><a name="p18495162318342"></a>native_batch_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1495423143412"><a name="p1495423143412"></a><a name="p1495423143412"></a>batch_norm_npu</p>
</td>
</tr>
<tr id="row1872263971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1747416161923"><a name="p1747416161923"></a><a name="p1747416161923"></a>256</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15495323113418"><a name="p15495323113418"></a><a name="p15495323113418"></a>batch_norm_stats</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7495723133414"><a name="p7495723133414"></a><a name="p7495723133414"></a>batch_norm_stats_npu</p>
</td>
</tr>
<tr id="row12722123915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147418168214"><a name="p1147418168214"></a><a name="p1147418168214"></a>257</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24951723103420"><a name="p24951723103420"></a><a name="p24951723103420"></a>batch_norm_elemt</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19495142383419"><a name="p19495142383419"></a><a name="p19495142383419"></a>batch_norm_elemt_npu</p>
</td>
</tr>
<tr id="row187221739191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5475171617220"><a name="p5475171617220"></a><a name="p5475171617220"></a>258</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12495182313344"><a name="p12495182313344"></a><a name="p12495182313344"></a>batch_norm_elemt.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1249582323415"><a name="p1249582323415"></a><a name="p1249582323415"></a>batch_norm_elemt_out_npu</p>
</td>
</tr>
<tr id="row87224394120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647561618215"><a name="p647561618215"></a><a name="p647561618215"></a>259</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p104952236340"><a name="p104952236340"></a><a name="p104952236340"></a>native_batch_norm_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16495172353416"><a name="p16495172353416"></a><a name="p16495172353416"></a>batch_norm_backward_npu</p>
</td>
</tr>
<tr id="row107221239181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174754169211"><a name="p174754169211"></a><a name="p174754169211"></a>260</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6495523123411"><a name="p6495523123411"></a><a name="p6495523123411"></a>batch_norm_backward_reduce</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p149552373413"><a name="p149552373413"></a><a name="p149552373413"></a>batch_norm_backward_reduce_npu</p>
</td>
</tr>
<tr id="row7722153916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54755160212"><a name="p54755160212"></a><a name="p54755160212"></a>261</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p649512312349"><a name="p649512312349"></a><a name="p649512312349"></a>_nnpack_spatial_convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9495192313343"><a name="p9495192313343"></a><a name="p9495192313343"></a>_nnpack_spatial_convolution_npu</p>
</td>
</tr>
<tr id="row15723163901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047571615214"><a name="p1047571615214"></a><a name="p1047571615214"></a>262</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p104951923153417"><a name="p104951923153417"></a><a name="p104951923153417"></a>ones.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16495823183419"><a name="p16495823183419"></a><a name="p16495823183419"></a>ones_npu</p>
</td>
</tr>
<tr id="row11723139141218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647513161229"><a name="p647513161229"></a><a name="p647513161229"></a>263</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24951723173412"><a name="p24951723173412"></a><a name="p24951723173412"></a>ones</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p949662393412"><a name="p949662393412"></a><a name="p949662393412"></a>ones_npu</p>
</td>
</tr>
<tr id="row18723183917129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9475111610212"><a name="p9475111610212"></a><a name="p9475111610212"></a>264</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124961123163417"><a name="p124961123163417"></a><a name="p124961123163417"></a>ones.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p149662315349"><a name="p149662315349"></a><a name="p149662315349"></a>ones_out_npu</p>
</td>
</tr>
<tr id="row572323917129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9475616020"><a name="p9475616020"></a><a name="p9475616020"></a>265</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p249612320347"><a name="p249612320347"></a><a name="p249612320347"></a>ones_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12496152303419"><a name="p12496152303419"></a><a name="p12496152303419"></a>ones_like_npu</p>
</td>
</tr>
<tr id="row157236395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147517167217"><a name="p147517167217"></a><a name="p147517167217"></a>266</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0496182316341"><a name="p0496182316341"></a><a name="p0496182316341"></a>cdist</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1496723183414"><a name="p1496723183414"></a><a name="p1496723183414"></a>cdist_npu</p>
</td>
</tr>
<tr id="row12723539181215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347531612211"><a name="p1347531612211"></a><a name="p1347531612211"></a>267</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14965236342"><a name="p14965236342"></a><a name="p14965236342"></a>_cdist_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4496152353411"><a name="p4496152353411"></a><a name="p4496152353411"></a>_cdist_forward_npu</p>
</td>
</tr>
<tr id="row972373971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194751160213"><a name="p194751160213"></a><a name="p194751160213"></a>268</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p949615232347"><a name="p949615232347"></a><a name="p949615232347"></a>_cdist_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10496122323420"><a name="p10496122323420"></a><a name="p10496122323420"></a>_cdist_backward_npu</p>
</td>
</tr>
<tr id="row87231839101216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10475016423"><a name="p10475016423"></a><a name="p10475016423"></a>269</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p44961523123411"><a name="p44961523123411"></a><a name="p44961523123411"></a>pdist</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7496172313343"><a name="p7496172313343"></a><a name="p7496172313343"></a>pdist_npu</p>
</td>
</tr>
<tr id="row157231339181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144755161829"><a name="p144755161829"></a><a name="p144755161829"></a>270</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9496192315345"><a name="p9496192315345"></a><a name="p9496192315345"></a>_pdist_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15496323193415"><a name="p15496323193415"></a><a name="p15496323193415"></a>_pdist_forward_npu</p>
</td>
</tr>
<tr id="row472310397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847514161323"><a name="p847514161323"></a><a name="p847514161323"></a>271</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14496152363412"><a name="p14496152363412"></a><a name="p14496152363412"></a>randperm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p049632313340"><a name="p049632313340"></a><a name="p049632313340"></a>randperm_npu</p>
</td>
</tr>
<tr id="row372323991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647518161920"><a name="p1647518161920"></a><a name="p1647518161920"></a>272</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1749618238345"><a name="p1749618238345"></a><a name="p1749618238345"></a>randperm.generator</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7496152313342"><a name="p7496152313342"></a><a name="p7496152313342"></a>randperm_npu</p>
</td>
</tr>
<tr id="row18724183911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04768167214"><a name="p04768167214"></a><a name="p04768167214"></a>273</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16496102311340"><a name="p16496102311340"></a><a name="p16496102311340"></a>randperm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p849602353417"><a name="p849602353417"></a><a name="p849602353417"></a>randperm_out_npu</p>
</td>
</tr>
<tr id="row16725639181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p247614161022"><a name="p247614161022"></a><a name="p247614161022"></a>274</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1449612323416"><a name="p1449612323416"></a><a name="p1449612323416"></a>randperm.generator_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p649619237348"><a name="p649619237348"></a><a name="p649619237348"></a>randperm_out_npu</p>
</td>
</tr>
<tr id="row87254399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p947613161127"><a name="p947613161127"></a><a name="p947613161127"></a>275</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1949610233346"><a name="p1949610233346"></a><a name="p1949610233346"></a>range.step</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p144964233340"><a name="p144964233340"></a><a name="p144964233340"></a>range_npu</p>
</td>
</tr>
<tr id="row3726173971212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147611161727"><a name="p147611161727"></a><a name="p147611161727"></a>276</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p549622318342"><a name="p549622318342"></a><a name="p549622318342"></a>range</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8496523143415"><a name="p8496523143415"></a><a name="p8496523143415"></a>range_npu</p>
</td>
</tr>
<tr id="row5726103918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3476916627"><a name="p3476916627"></a><a name="p3476916627"></a>277</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13496823183416"><a name="p13496823183416"></a><a name="p13496823183416"></a>range.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1549620232348"><a name="p1549620232348"></a><a name="p1549620232348"></a>range_out_npu</p>
</td>
</tr>
<tr id="row972617399127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24769161524"><a name="p24769161524"></a><a name="p24769161524"></a>278</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194971123183418"><a name="p194971123183418"></a><a name="p194971123183418"></a>reciprocal</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13497123133416"><a name="p13497123133416"></a><a name="p13497123133416"></a>reciprocal_npu</p>
</td>
</tr>
<tr id="row1972693961213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16476151615220"><a name="p16476151615220"></a><a name="p16476151615220"></a>279</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p349702318340"><a name="p349702318340"></a><a name="p349702318340"></a>reciprocal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1249702383414"><a name="p1249702383414"></a><a name="p1249702383414"></a>reciprocal_npu_</p>
</td>
</tr>
<tr id="row8726133920129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447618168218"><a name="p447618168218"></a><a name="p447618168218"></a>280</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16497172319341"><a name="p16497172319341"></a><a name="p16497172319341"></a>reciprocal.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17497172313413"><a name="p17497172313413"></a><a name="p17497172313413"></a>reciprocal_out_npu</p>
</td>
</tr>
<tr id="row12726193914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34765162029"><a name="p34765162029"></a><a name="p34765162029"></a>281</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p124975239348"><a name="p124975239348"></a><a name="p124975239348"></a>neg</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2049713235343"><a name="p2049713235343"></a><a name="p2049713235343"></a>neg_npu</p>
</td>
</tr>
<tr id="row8726339101213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547614161929"><a name="p1547614161929"></a><a name="p1547614161929"></a>282</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p94971623103411"><a name="p94971623103411"></a><a name="p94971623103411"></a>neg_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114971234347"><a name="p114971234347"></a><a name="p114971234347"></a>neg_npu_</p>
</td>
</tr>
<tr id="row15726123913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144763161127"><a name="p144763161127"></a><a name="p144763161127"></a>283</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1149712231349"><a name="p1149712231349"></a><a name="p1149712231349"></a>neg.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15497102311345"><a name="p15497102311345"></a><a name="p15497102311345"></a>neg_out_npu</p>
</td>
</tr>
<tr id="row1472614394127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147631614219"><a name="p147631614219"></a><a name="p147631614219"></a>284</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114971523183410"><a name="p114971523183410"></a><a name="p114971523183410"></a>repeat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114971236348"><a name="p114971236348"></a><a name="p114971236348"></a>repeat_npu</p>
</td>
</tr>
<tr id="row1072663911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847610167218"><a name="p847610167218"></a><a name="p847610167218"></a>285</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p249762315348"><a name="p249762315348"></a><a name="p249762315348"></a>repeat_interleave.self_int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8497152323417"><a name="p8497152323417"></a><a name="p8497152323417"></a>repeat_interleave_npu</p>
</td>
</tr>
<tr id="row16727239141217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3476191615213"><a name="p3476191615213"></a><a name="p3476191615213"></a>286</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8497112383419"><a name="p8497112383419"></a><a name="p8497112383419"></a>round</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34974235349"><a name="p34974235349"></a><a name="p34974235349"></a>round_npu</p>
</td>
</tr>
<tr id="row1572710397121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1476516527"><a name="p1476516527"></a><a name="p1476516527"></a>287</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p24971323163415"><a name="p24971323163415"></a><a name="p24971323163415"></a>round_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18497623103417"><a name="p18497623103417"></a><a name="p18497623103417"></a>round_npu_</p>
</td>
</tr>
<tr id="row572713392127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04761516824"><a name="p04761516824"></a><a name="p04761516824"></a>288</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14497142317342"><a name="p14497142317342"></a><a name="p14497142317342"></a>round.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p34971923143412"><a name="p34971923143412"></a><a name="p34971923143412"></a>round_out_npu</p>
</td>
</tr>
<tr id="row17727639181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184761916926"><a name="p184761916926"></a><a name="p184761916926"></a>289</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p34971423103417"><a name="p34971423103417"></a><a name="p34971423103417"></a>relu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2049762373420"><a name="p2049762373420"></a><a name="p2049762373420"></a>relu_npu</p>
</td>
</tr>
<tr id="row1872783910125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114769161927"><a name="p114769161927"></a><a name="p114769161927"></a>290</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1249718233342"><a name="p1249718233342"></a><a name="p1249718233342"></a>relu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9497623153414"><a name="p9497623153414"></a><a name="p9497623153414"></a>relu_npu_</p>
</td>
</tr>
<tr id="row272718396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947611620210"><a name="p1947611620210"></a><a name="p1947611620210"></a>291</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194971323103412"><a name="p194971323103412"></a><a name="p194971323103412"></a>prelu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p549722373418"><a name="p549722373418"></a><a name="p549722373418"></a>prelu_npu</p>
</td>
</tr>
<tr id="row1727163912122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84771716627"><a name="p84771716627"></a><a name="p84771716627"></a>292</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p114971523143416"><a name="p114971523143416"></a><a name="p114971523143416"></a>prelu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8497102333416"><a name="p8497102333416"></a><a name="p8497102333416"></a>prelu_backward_npu</p>
</td>
</tr>
<tr id="row9727133901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104774161721"><a name="p104774161721"></a><a name="p104774161721"></a>293</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p549702314343"><a name="p549702314343"></a><a name="p549702314343"></a>gelu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7498152303420"><a name="p7498152303420"></a><a name="p7498152303420"></a>gelu_npu</p>
</td>
</tr>
<tr id="row17727103918124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124775161217"><a name="p124775161217"></a><a name="p124775161217"></a>294</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p64981423193412"><a name="p64981423193412"></a><a name="p64981423193412"></a>gelu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p349872310347"><a name="p349872310347"></a><a name="p349872310347"></a>gelu_backward_npu</p>
</td>
</tr>
<tr id="row572773917122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84773161021"><a name="p84773161021"></a><a name="p84773161021"></a>295</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1449892316345"><a name="p1449892316345"></a><a name="p1449892316345"></a>hardshrink</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p164981523113411"><a name="p164981523113411"></a><a name="p164981523113411"></a>hardshrink_npu</p>
</td>
</tr>
<tr id="row1172723911216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147711161526"><a name="p147711161526"></a><a name="p147711161526"></a>296</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p849872317341"><a name="p849872317341"></a><a name="p849872317341"></a>hardshrink_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p134981823113414"><a name="p134981823113414"></a><a name="p134981823113414"></a>hardshrink_backward_npu</p>
</td>
</tr>
<tr id="row1672763961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14772161226"><a name="p14772161226"></a><a name="p14772161226"></a>297</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1749812383416"><a name="p1749812383416"></a><a name="p1749812383416"></a>rsqrt</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1449842343410"><a name="p1449842343410"></a><a name="p1449842343410"></a>rsqrt_npu</p>
</td>
</tr>
<tr id="row87284399129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947711161521"><a name="p1947711161521"></a><a name="p1947711161521"></a>298</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p949812234344"><a name="p949812234344"></a><a name="p949812234344"></a>rsqrt_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7498202311344"><a name="p7498202311344"></a><a name="p7498202311344"></a>rsqrt_npu_</p>
</td>
</tr>
<tr id="row20728839161216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647771615217"><a name="p647771615217"></a><a name="p647771615217"></a>299</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12498223133410"><a name="p12498223133410"></a><a name="p12498223133410"></a>rsqrt.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p154981723103413"><a name="p154981723103413"></a><a name="p154981723103413"></a>rsqrt_out_npu</p>
</td>
</tr>
<tr id="row1072893910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047716161929"><a name="p2047716161929"></a><a name="p2047716161929"></a>300</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p84981323113413"><a name="p84981323113413"></a><a name="p84981323113413"></a>selu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1949882310344"><a name="p1949882310344"></a><a name="p1949882310344"></a>selu_npu</p>
</td>
</tr>
<tr id="row1672863915127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184772161020"><a name="p184772161020"></a><a name="p184772161020"></a>301</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1449942353410"><a name="p1449942353410"></a><a name="p1449942353410"></a>selu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p114991923103420"><a name="p114991923103420"></a><a name="p114991923103420"></a>selu_npu_</p>
</td>
</tr>
<tr id="row5728739101210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54773168211"><a name="p54773168211"></a><a name="p54773168211"></a>302</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p949962319349"><a name="p949962319349"></a><a name="p949962319349"></a>celu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7499122316341"><a name="p7499122316341"></a><a name="p7499122316341"></a>celu_npu</p>
</td>
</tr>
<tr id="row15728153941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3477316220"><a name="p3477316220"></a><a name="p3477316220"></a>303</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p949932311347"><a name="p949932311347"></a><a name="p949932311347"></a>celu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p849915237347"><a name="p849915237347"></a><a name="p849915237347"></a>celu_npu_</p>
</td>
</tr>
<tr id="row15728239111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19477181614214"><a name="p19477181614214"></a><a name="p19477181614214"></a>304</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17499102312341"><a name="p17499102312341"></a><a name="p17499102312341"></a>sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17499192373412"><a name="p17499192373412"></a><a name="p17499192373412"></a>sigmoid_npu</p>
</td>
</tr>
<tr id="row67282039151218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15477151614219"><a name="p15477151614219"></a><a name="p15477151614219"></a>305</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p64991423163416"><a name="p64991423163416"></a><a name="p64991423163416"></a>sigmoid_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19499122353417"><a name="p19499122353417"></a><a name="p19499122353417"></a>sigmoid_npu_</p>
</td>
</tr>
<tr id="row172818396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347814161129"><a name="p1347814161129"></a><a name="p1347814161129"></a>306</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1149962323417"><a name="p1149962323417"></a><a name="p1149962323417"></a>sigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p249942343415"><a name="p249942343415"></a><a name="p249942343415"></a>sigmoid_out_npu</p>
</td>
</tr>
<tr id="row2728039181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134785167218"><a name="p134785167218"></a><a name="p134785167218"></a>307</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6499142312343"><a name="p6499142312343"></a><a name="p6499142312343"></a>sin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p64993236346"><a name="p64993236346"></a><a name="p64993236346"></a>sin_npu</p>
</td>
</tr>
<tr id="row77287391121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647851613215"><a name="p647851613215"></a><a name="p647851613215"></a>308</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p194991323143420"><a name="p194991323143420"></a><a name="p194991323143420"></a>sin_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p164991423133416"><a name="p164991423133416"></a><a name="p164991423133416"></a>sin_npu_</p>
</td>
</tr>
<tr id="row772923912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184787161725"><a name="p184787161725"></a><a name="p184787161725"></a>309</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6499202363414"><a name="p6499202363414"></a><a name="p6499202363414"></a>sin.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3499112313420"><a name="p3499112313420"></a><a name="p3499112313420"></a>sin_out_npu</p>
</td>
</tr>
<tr id="row1872912397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p047819161626"><a name="p047819161626"></a><a name="p047819161626"></a>310</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4499122383414"><a name="p4499122383414"></a><a name="p4499122383414"></a>sinh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8499152333414"><a name="p8499152333414"></a><a name="p8499152333414"></a>sinh_npu</p>
</td>
</tr>
<tr id="row772923941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747820160210"><a name="p747820160210"></a><a name="p747820160210"></a>311</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p749922312344"><a name="p749922312344"></a><a name="p749922312344"></a>sinh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p049932383410"><a name="p049932383410"></a><a name="p049932383410"></a>sinh_npu_</p>
</td>
</tr>
<tr id="row7729143919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174781416028"><a name="p174781416028"></a><a name="p174781416028"></a>312</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7499172313349"><a name="p7499172313349"></a><a name="p7499172313349"></a>sinh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1150092353411"><a name="p1150092353411"></a><a name="p1150092353411"></a>sinh_out_npu</p>
</td>
</tr>
<tr id="row972917397121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047811169220"><a name="p1047811169220"></a><a name="p1047811169220"></a>313</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1350017236345"><a name="p1350017236345"></a><a name="p1350017236345"></a>slogdet</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p050012393411"><a name="p050012393411"></a><a name="p050012393411"></a>slogdet_npu</p>
</td>
</tr>
<tr id="row1172903920127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447819161020"><a name="p1447819161020"></a><a name="p1447819161020"></a>314</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3500162315346"><a name="p3500162315346"></a><a name="p3500162315346"></a>softmax.int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1550015235347"><a name="p1550015235347"></a><a name="p1550015235347"></a>softmax_npu</p>
</td>
</tr>
<tr id="row12729133920124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2478191614211"><a name="p2478191614211"></a><a name="p2478191614211"></a>315</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10500162353414"><a name="p10500162353414"></a><a name="p10500162353414"></a>softmax.Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6500112343418"><a name="p6500112343418"></a><a name="p6500112343418"></a>softmax_npu</p>
</td>
</tr>
<tr id="row6729203951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847811620215"><a name="p847811620215"></a><a name="p847811620215"></a>316</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185009238348"><a name="p185009238348"></a><a name="p185009238348"></a>_softmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145001823153415"><a name="p145001823153415"></a><a name="p145001823153415"></a>_softmax_npu</p>
</td>
</tr>
<tr id="row18729153951219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154781165218"><a name="p154781165218"></a><a name="p154781165218"></a>317</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12500152313343"><a name="p12500152313343"></a><a name="p12500152313343"></a>_softmax_backward_data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1650018235348"><a name="p1650018235348"></a><a name="p1650018235348"></a>_softmax_backward_npu</p>
</td>
</tr>
<tr id="row137291539131215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7478121619218"><a name="p7478121619218"></a><a name="p7478121619218"></a>318</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105006231348"><a name="p105006231348"></a><a name="p105006231348"></a>stack</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1500723183419"><a name="p1500723183419"></a><a name="p1500723183419"></a>stack_npu</p>
</td>
</tr>
<tr id="row207291839121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447871616213"><a name="p1447871616213"></a><a name="p1447871616213"></a>319</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1150012238342"><a name="p1150012238342"></a><a name="p1150012238342"></a>stack.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195004234346"><a name="p195004234346"></a><a name="p195004234346"></a>stack_out_npu</p>
</td>
</tr>
<tr id="row77291139121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147818165213"><a name="p1147818165213"></a><a name="p1147818165213"></a>320</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p750092303413"><a name="p750092303413"></a><a name="p750092303413"></a>sum</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p050018230341"><a name="p050018230341"></a><a name="p050018230341"></a>sum_npu</p>
</td>
</tr>
<tr id="row137301239111219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447821618211"><a name="p447821618211"></a><a name="p447821618211"></a>321</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8500152313411"><a name="p8500152313411"></a><a name="p8500152313411"></a>sum.dim_IntList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1950012318346"><a name="p1950012318346"></a><a name="p1950012318346"></a>sum_npu</p>
</td>
</tr>
<tr id="row187301139101214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194786167215"><a name="p194786167215"></a><a name="p194786167215"></a>322</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19500142311347"><a name="p19500142311347"></a><a name="p19500142311347"></a>sum.dim_DimnameList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4500523143412"><a name="p4500523143412"></a><a name="p4500523143412"></a>sum_npu</p>
</td>
</tr>
<tr id="row1173053981217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1847931611220"><a name="p1847931611220"></a><a name="p1847931611220"></a>323</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175000231347"><a name="p175000231347"></a><a name="p175000231347"></a>sum.IntList_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8500152373414"><a name="p8500152373414"></a><a name="p8500152373414"></a>sum_out_npu</p>
</td>
</tr>
<tr id="row20730123981213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947910165210"><a name="p1947910165210"></a><a name="p1947910165210"></a>324</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9500132333417"><a name="p9500132333417"></a><a name="p9500132333417"></a>sum.DimnameList_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p250062373419"><a name="p250062373419"></a><a name="p250062373419"></a>sum_out_npu</p>
</td>
</tr>
<tr id="row773013920126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34791616524"><a name="p34791616524"></a><a name="p34791616524"></a>325</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p650042373415"><a name="p650042373415"></a><a name="p650042373415"></a>sqrt</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1350010234346"><a name="p1350010234346"></a><a name="p1350010234346"></a>sqrt_npu</p>
</td>
</tr>
<tr id="row1973018398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647910167219"><a name="p647910167219"></a><a name="p647910167219"></a>326</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9501142363417"><a name="p9501142363417"></a><a name="p9501142363417"></a>sqrt_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0501723203419"><a name="p0501723203419"></a><a name="p0501723203419"></a>sqrt_npu_</p>
</td>
</tr>
<tr id="row12730153913122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4479161613216"><a name="p4479161613216"></a><a name="p4479161613216"></a>327</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95011223193417"><a name="p95011223193417"></a><a name="p95011223193417"></a>sqrt.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85010235346"><a name="p85010235346"></a><a name="p85010235346"></a>sqrt_out_npu</p>
</td>
</tr>
<tr id="row47301639101218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047911613210"><a name="p2047911613210"></a><a name="p2047911613210"></a>328</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155011123153411"><a name="p155011123153411"></a><a name="p155011123153411"></a>std</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p105011623173412"><a name="p105011623173412"></a><a name="p105011623173412"></a>std_npu</p>
</td>
</tr>
<tr id="row1873012396124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134791816721"><a name="p134791816721"></a><a name="p134791816721"></a>329</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p650192353412"><a name="p650192353412"></a><a name="p650192353412"></a>std.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p450152314345"><a name="p450152314345"></a><a name="p450152314345"></a>std_dim_npu</p>
</td>
</tr>
<tr id="row773043919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10479141613216"><a name="p10479141613216"></a><a name="p10479141613216"></a>330</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p650132353415"><a name="p650132353415"></a><a name="p650132353415"></a>std_mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6501923143413"><a name="p6501923143413"></a><a name="p6501923143413"></a>std_mean_npu</p>
</td>
</tr>
<tr id="row1473013911211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164795165212"><a name="p164795165212"></a><a name="p164795165212"></a>331</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105011923153419"><a name="p105011923153419"></a><a name="p105011923153419"></a>std_mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85010233342"><a name="p85010233342"></a><a name="p85010233342"></a>std_mean_dim_npu</p>
</td>
</tr>
<tr id="row1173173951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124795161215"><a name="p124795161215"></a><a name="p124795161215"></a>332</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p150132311345"><a name="p150132311345"></a><a name="p150132311345"></a>std_mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175011323113417"><a name="p175011323113417"></a><a name="p175011323113417"></a>std_mean_names_npu</p>
</td>
</tr>
<tr id="row47311839111211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194792161028"><a name="p194792161028"></a><a name="p194792161028"></a>333</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450117234349"><a name="p450117234349"></a><a name="p450117234349"></a>std.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1850182373411"><a name="p1850182373411"></a><a name="p1850182373411"></a>std_out_npu</p>
</td>
</tr>
<tr id="row7731439171216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447941618212"><a name="p447941618212"></a><a name="p447941618212"></a>334</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5501182315344"><a name="p5501182315344"></a><a name="p5501182315344"></a>std.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1850117237342"><a name="p1850117237342"></a><a name="p1850117237342"></a>std_names_npu</p>
</td>
</tr>
<tr id="row47311439191213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8479101619218"><a name="p8479101619218"></a><a name="p8479101619218"></a>335</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2050142383416"><a name="p2050142383416"></a><a name="p2050142383416"></a>std.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p950142314343"><a name="p950142314343"></a><a name="p950142314343"></a>std_out_npu</p>
</td>
</tr>
<tr id="row1373193911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447917167215"><a name="p447917167215"></a><a name="p447917167215"></a>336</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p550132383413"><a name="p550132383413"></a><a name="p550132383413"></a>prod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8501112316341"><a name="p8501112316341"></a><a name="p8501112316341"></a>prod_npu</p>
</td>
</tr>
<tr id="row47315396122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19479171620216"><a name="p19479171620216"></a><a name="p19479171620216"></a>337</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1450118239348"><a name="p1450118239348"></a><a name="p1450118239348"></a>prod.dim_int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9501112363412"><a name="p9501112363412"></a><a name="p9501112363412"></a>prod_npu</p>
</td>
</tr>
<tr id="row27311139111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124798161629"><a name="p124798161629"></a><a name="p124798161629"></a>338</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p850152314343"><a name="p850152314343"></a><a name="p850152314343"></a>prod.int_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1550110236343"><a name="p1550110236343"></a><a name="p1550110236343"></a>prod_out_npu</p>
</td>
</tr>
<tr id="row27312391123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147919169217"><a name="p1147919169217"></a><a name="p1147919169217"></a>339</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p145024230347"><a name="p145024230347"></a><a name="p145024230347"></a>prod.dim_Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p150232363412"><a name="p150232363412"></a><a name="p150232363412"></a>prod_npu</p>
</td>
</tr>
<tr id="row187311539101213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548014165210"><a name="p1548014165210"></a><a name="p1548014165210"></a>340</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105021423123410"><a name="p105021423123410"></a><a name="p105021423123410"></a>prod.Dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p250272317346"><a name="p250272317346"></a><a name="p250272317346"></a>prod_out_npu</p>
</td>
</tr>
<tr id="row187313392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64802161029"><a name="p64802161029"></a><a name="p64802161029"></a>341</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105023238340"><a name="p105023238340"></a><a name="p105023238340"></a>tan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p650212311344"><a name="p650212311344"></a><a name="p650212311344"></a>tan_npu</p>
</td>
</tr>
<tr id="row1873183981220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548081614214"><a name="p548081614214"></a><a name="p548081614214"></a>342</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450222313411"><a name="p450222313411"></a><a name="p450222313411"></a>tan_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1850212343418"><a name="p1850212343418"></a><a name="p1850212343418"></a>tan_npu_</p>
</td>
</tr>
<tr id="row12731103951212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948081611215"><a name="p1948081611215"></a><a name="p1948081611215"></a>343</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0502132323411"><a name="p0502132323411"></a><a name="p0502132323411"></a>tan.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p950220230341"><a name="p950220230341"></a><a name="p950220230341"></a>tan_out_npu</p>
</td>
</tr>
<tr id="row16732139111215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648091614213"><a name="p1648091614213"></a><a name="p1648091614213"></a>344</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0502723163413"><a name="p0502723163413"></a><a name="p0502723163413"></a>tanh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65021223163418"><a name="p65021223163418"></a><a name="p65021223163418"></a>tanh_npu</p>
</td>
</tr>
<tr id="row6732113961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20480151612213"><a name="p20480151612213"></a><a name="p20480151612213"></a>345</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p250220230343"><a name="p250220230343"></a><a name="p250220230343"></a>tanh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1750282314340"><a name="p1750282314340"></a><a name="p1750282314340"></a>tanh_npu_</p>
</td>
</tr>
<tr id="row1473273916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04802161927"><a name="p04802161927"></a><a name="p04802161927"></a>346</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65025236341"><a name="p65025236341"></a><a name="p65025236341"></a>tanh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10502132393419"><a name="p10502132393419"></a><a name="p10502132393419"></a>tanh_out_npu</p>
</td>
</tr>
<tr id="row673213910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8480151619212"><a name="p8480151619212"></a><a name="p8480151619212"></a>347</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1850222313345"><a name="p1850222313345"></a><a name="p1850222313345"></a>threshold</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20502122373414"><a name="p20502122373414"></a><a name="p20502122373414"></a>threshold_npu</p>
</td>
</tr>
<tr id="row873263916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648015161520"><a name="p648015161520"></a><a name="p648015161520"></a>348</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p145021323193420"><a name="p145021323193420"></a><a name="p145021323193420"></a>threshold_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p150222323411"><a name="p150222323411"></a><a name="p150222323411"></a>threshold_npu_</p>
</td>
</tr>
<tr id="row07321739181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148016167216"><a name="p1148016167216"></a><a name="p1148016167216"></a>349</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1150218231343"><a name="p1150218231343"></a><a name="p1150218231343"></a>threshold.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1550392333419"><a name="p1550392333419"></a><a name="p1550392333419"></a>threshold_out_npu</p>
</td>
</tr>
<tr id="row12732163911214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8480516328"><a name="p8480516328"></a><a name="p8480516328"></a>350</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450316233342"><a name="p450316233342"></a><a name="p450316233342"></a>threshold_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75039231346"><a name="p75039231346"></a><a name="p75039231346"></a>threshold_backward_npu</p>
</td>
</tr>
<tr id="row117321397120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144801316221"><a name="p144801316221"></a><a name="p144801316221"></a>351</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p250317233347"><a name="p250317233347"></a><a name="p250317233347"></a>one_hot</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125031923133415"><a name="p125031923133415"></a><a name="p125031923133415"></a>one_hot_npu1</p>
</td>
</tr>
<tr id="row873218398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194801316925"><a name="p194801316925"></a><a name="p194801316925"></a>352</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17503923183416"><a name="p17503923183416"></a><a name="p17503923183416"></a>flip</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125031623113417"><a name="p125031623113417"></a><a name="p125031623113417"></a>flip_npu</p>
</td>
</tr>
<tr id="row1173243919126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548041618219"><a name="p548041618219"></a><a name="p548041618219"></a>353</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19503102313349"><a name="p19503102313349"></a><a name="p19503102313349"></a>roll</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4503132393416"><a name="p4503132393416"></a><a name="p4503132393416"></a>roll_npu</p>
</td>
</tr>
<tr id="row1673219391128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24817161215"><a name="p24817161215"></a><a name="p24817161215"></a>354</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450320230345"><a name="p450320230345"></a><a name="p450320230345"></a>true_divide.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15031423103414"><a name="p15031423103414"></a><a name="p15031423103414"></a>true_divide_npu</p>
</td>
</tr>
<tr id="row117331397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5481111615216"><a name="p5481111615216"></a><a name="p5481111615216"></a>355</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17503023103416"><a name="p17503023103416"></a><a name="p17503023103416"></a>true_divide_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65037230341"><a name="p65037230341"></a><a name="p65037230341"></a>true_divide_npu_</p>
</td>
</tr>
<tr id="row6733239131216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18481516224"><a name="p18481516224"></a><a name="p18481516224"></a>356</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75033234342"><a name="p75033234342"></a><a name="p75033234342"></a>true_divide.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115032023163418"><a name="p115032023163418"></a><a name="p115032023163418"></a>true_divide_out_npu</p>
</td>
</tr>
<tr id="row8733113981214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10481716720"><a name="p10481716720"></a><a name="p10481716720"></a>357</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17503112320345"><a name="p17503112320345"></a><a name="p17503112320345"></a>true_divide.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p150312237349"><a name="p150312237349"></a><a name="p150312237349"></a>true_divide_npu</p>
</td>
</tr>
<tr id="row107332039151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64817161211"><a name="p64817161211"></a><a name="p64817161211"></a>358</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20503423163419"><a name="p20503423163419"></a><a name="p20503423163419"></a>true_divide_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145032234342"><a name="p145032234342"></a><a name="p145032234342"></a>true_divide_npu_</p>
</td>
</tr>
<tr id="row13733339161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64811616724"><a name="p64811616724"></a><a name="p64811616724"></a>359</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10503152313346"><a name="p10503152313346"></a><a name="p10503152313346"></a>trunc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p205035236340"><a name="p205035236340"></a><a name="p205035236340"></a>trunc_npu</p>
</td>
</tr>
<tr id="row3733153931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3481101611218"><a name="p3481101611218"></a><a name="p3481101611218"></a>360</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5503202310346"><a name="p5503202310346"></a><a name="p5503202310346"></a>trunc_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8503152383418"><a name="p8503152383418"></a><a name="p8503152383418"></a>trunc_npu_</p>
</td>
</tr>
<tr id="row10733139111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5481111613213"><a name="p5481111613213"></a><a name="p5481111613213"></a>361</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175031023193415"><a name="p175031023193415"></a><a name="p175031023193415"></a>trunc.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2503122318342"><a name="p2503122318342"></a><a name="p2503122318342"></a>trunc_out_npu</p>
</td>
</tr>
<tr id="row7733739101220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174811716224"><a name="p174811716224"></a><a name="p174811716224"></a>362</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95031423123413"><a name="p95031423123413"></a><a name="p95031423123413"></a>_unique2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1750302393417"><a name="p1750302393417"></a><a name="p1750302393417"></a>_unique2_npu</p>
</td>
</tr>
<tr id="row8733183911217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44811516922"><a name="p44811516922"></a><a name="p44811516922"></a>363</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16504102319346"><a name="p16504102319346"></a><a name="p16504102319346"></a>var</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1950412232348"><a name="p1950412232348"></a><a name="p1950412232348"></a>var_npu</p>
</td>
</tr>
<tr id="row1733193961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1248141617211"><a name="p1248141617211"></a><a name="p1248141617211"></a>364</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1950492383417"><a name="p1950492383417"></a><a name="p1950492383417"></a>var.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p750442363411"><a name="p750442363411"></a><a name="p750442363411"></a>var_npu</p>
</td>
</tr>
<tr id="row157331039111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104817163214"><a name="p104817163214"></a><a name="p104817163214"></a>365</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p350442363417"><a name="p350442363417"></a><a name="p350442363417"></a>var.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5504112333410"><a name="p5504112333410"></a><a name="p5504112333410"></a>var_out_npu</p>
</td>
</tr>
<tr id="row1573414397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144812161020"><a name="p144812161020"></a><a name="p144812161020"></a>366</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p750492317347"><a name="p750492317347"></a><a name="p750492317347"></a>var.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p350432312341"><a name="p350432312341"></a><a name="p350432312341"></a>var_npu</p>
</td>
</tr>
<tr id="row11734153931213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1481131615216"><a name="p1481131615216"></a><a name="p1481131615216"></a>367</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p250419238346"><a name="p250419238346"></a><a name="p250419238346"></a>var.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1350432317344"><a name="p1350432317344"></a><a name="p1350432317344"></a>var_out_npu</p>
</td>
</tr>
<tr id="row173473941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174814160210"><a name="p174814160210"></a><a name="p174814160210"></a>368</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p950442310346"><a name="p950442310346"></a><a name="p950442310346"></a>var_mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16504142323412"><a name="p16504142323412"></a><a name="p16504142323412"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row137341239201216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13481116221"><a name="p13481116221"></a><a name="p13481116221"></a>369</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2050432383416"><a name="p2050432383416"></a><a name="p2050432383416"></a>var_mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p450472383420"><a name="p450472383420"></a><a name="p450472383420"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row19734113931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848161618210"><a name="p848161618210"></a><a name="p848161618210"></a>370</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17504112393411"><a name="p17504112393411"></a><a name="p17504112393411"></a>var_mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p135041823183415"><a name="p135041823183415"></a><a name="p135041823183415"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row77341439121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16481516222"><a name="p16481516222"></a><a name="p16481516222"></a>371</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16504192323417"><a name="p16504192323417"></a><a name="p16504192323417"></a>where.self</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0504112363418"><a name="p0504112363418"></a><a name="p0504112363418"></a>where_npu</p>
</td>
</tr>
<tr id="row147341939121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1448118161522"><a name="p1448118161522"></a><a name="p1448118161522"></a>372</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45041423163411"><a name="p45041423163411"></a><a name="p45041423163411"></a>where</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12504623163413"><a name="p12504623163413"></a><a name="p12504623163413"></a>where_npu</p>
</td>
</tr>
<tr id="row4734439181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34811016329"><a name="p34811016329"></a><a name="p34811016329"></a>373</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0504122315341"><a name="p0504122315341"></a><a name="p0504122315341"></a>_s_where</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1350411233344"><a name="p1350411233344"></a><a name="p1350411233344"></a>_s_where_npu</p>
</td>
</tr>
<tr id="row47341839121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164821516622"><a name="p164821516622"></a><a name="p164821516622"></a>374</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1750442323415"><a name="p1750442323415"></a><a name="p1750442323415"></a>zeros.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195041423103418"><a name="p195041423103418"></a><a name="p195041423103418"></a>zeros_npu</p>
</td>
</tr>
<tr id="row2734139131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448291619216"><a name="p448291619216"></a><a name="p448291619216"></a>375</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125041238345"><a name="p125041238345"></a><a name="p125041238345"></a>zeros</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1850432343411"><a name="p1850432343411"></a><a name="p1850432343411"></a>zeros_npu</p>
</td>
</tr>
<tr id="row10734153910128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164824161121"><a name="p164824161121"></a><a name="p164824161121"></a>376</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p150412323417"><a name="p150412323417"></a><a name="p150412323417"></a>zeros.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14504823203417"><a name="p14504823203417"></a><a name="p14504823203417"></a>zeros_out_npu</p>
</td>
</tr>
<tr id="row47351039151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12482201610215"><a name="p12482201610215"></a><a name="p12482201610215"></a>377</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1250520234341"><a name="p1250520234341"></a><a name="p1250520234341"></a>zeros_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10505142311345"><a name="p10505142311345"></a><a name="p10505142311345"></a>zeros_like_npu</p>
</td>
</tr>
<tr id="row13735339131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6482116629"><a name="p6482116629"></a><a name="p6482116629"></a>378</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450582316344"><a name="p450582316344"></a><a name="p450582316344"></a>norm.ScalarOpt_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p850502383412"><a name="p850502383412"></a><a name="p850502383412"></a>norm_npu</p>
</td>
</tr>
<tr id="row167351939191219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948217161428"><a name="p1948217161428"></a><a name="p1948217161428"></a>379</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p250532318347"><a name="p250532318347"></a><a name="p250532318347"></a>norm.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1950520237342"><a name="p1950520237342"></a><a name="p1950520237342"></a>norm_npu</p>
</td>
</tr>
<tr id="row7735113961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548217161024"><a name="p548217161024"></a><a name="p548217161024"></a>380</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0505723113412"><a name="p0505723113412"></a><a name="p0505723113412"></a>norm.ScalarOpt_dim_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1850582319343"><a name="p1850582319343"></a><a name="p1850582319343"></a>norm_npu</p>
</td>
</tr>
<tr id="row373503911216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194825167217"><a name="p194825167217"></a><a name="p194825167217"></a>381</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18505112311344"><a name="p18505112311344"></a><a name="p18505112311344"></a>norm.ScalarOpt_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3505152316342"><a name="p3505152316342"></a><a name="p3505152316342"></a>norm_npu</p>
</td>
</tr>
<tr id="row16735639151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154827161425"><a name="p154827161425"></a><a name="p154827161425"></a>382</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450542318342"><a name="p450542318342"></a><a name="p450542318342"></a>norm.dtype_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p950519231341"><a name="p950519231341"></a><a name="p950519231341"></a>norm_out_npu</p>
</td>
</tr>
<tr id="row173518397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948291615214"><a name="p948291615214"></a><a name="p948291615214"></a>383</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p350514231341"><a name="p350514231341"></a><a name="p350514231341"></a>norm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1450512316347"><a name="p1450512316347"></a><a name="p1450512316347"></a>norm_out_npu</p>
</td>
</tr>
<tr id="row173512397127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194821016020"><a name="p194821016020"></a><a name="p194821016020"></a>384</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1150542343417"><a name="p1150542343417"></a><a name="p1150542343417"></a>clone</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165051523163419"><a name="p165051523163419"></a><a name="p165051523163419"></a>clone_npu</p>
</td>
</tr>
<tr id="row117358394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348216161828"><a name="p1348216161828"></a><a name="p1348216161828"></a>385</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p150532311347"><a name="p150532311347"></a><a name="p150532311347"></a>resize_as_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55056237345"><a name="p55056237345"></a><a name="p55056237345"></a>resize_as_npu_</p>
</td>
</tr>
<tr id="row7735113910125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748214161326"><a name="p1748214161326"></a><a name="p1748214161326"></a>386</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450512314342"><a name="p450512314342"></a><a name="p450512314342"></a>pow.Tensor_Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1550513235342"><a name="p1550513235342"></a><a name="p1550513235342"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row20735123919121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948251617211"><a name="p1948251617211"></a><a name="p1948251617211"></a>387</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p650519235343"><a name="p650519235343"></a><a name="p650519235343"></a>pow.Tensor_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1250592383418"><a name="p1250592383418"></a><a name="p1250592383418"></a>pow_npu</p>
</td>
</tr>
<tr id="row4735103911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p748315169219"><a name="p748315169219"></a><a name="p748315169219"></a>388</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1050512233342"><a name="p1050512233342"></a><a name="p1050512233342"></a>zero_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145051123183414"><a name="p145051123183414"></a><a name="p145051123183414"></a>zero_npu_</p>
</td>
</tr>
<tr id="row137367398126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24831016128"><a name="p24831016128"></a><a name="p24831016128"></a>389</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65051123203419"><a name="p65051123203419"></a><a name="p65051123203419"></a>sub.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p650572312346"><a name="p650572312346"></a><a name="p650572312346"></a>sub_out_npu</p>
</td>
</tr>
<tr id="row873693961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15483141619216"><a name="p15483141619216"></a><a name="p15483141619216"></a>390</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15054230342"><a name="p15054230342"></a><a name="p15054230342"></a>sub.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1450542316341"><a name="p1450542316341"></a><a name="p1450542316341"></a>sub_npu</p>
</td>
</tr>
<tr id="row1573610394128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2048301616219"><a name="p2048301616219"></a><a name="p2048301616219"></a>391</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11505323153413"><a name="p11505323153413"></a><a name="p11505323153413"></a>sub_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1650502315340"><a name="p1650502315340"></a><a name="p1650502315340"></a>sub_npu_</p>
</td>
</tr>
<tr id="row17736103910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24836162020"><a name="p24836162020"></a><a name="p24836162020"></a>392</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1650514234343"><a name="p1650514234343"></a><a name="p1650514234343"></a>sub.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p350522313347"><a name="p350522313347"></a><a name="p350522313347"></a>sub_npu</p>
</td>
</tr>
<tr id="row137361039171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74832162212"><a name="p74832162212"></a><a name="p74832162212"></a>393</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16505132314346"><a name="p16505132314346"></a><a name="p16505132314346"></a>sub_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17506162320340"><a name="p17506162320340"></a><a name="p17506162320340"></a>sub_npu_</p>
</td>
</tr>
<tr id="row973673921220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174831161928"><a name="p174831161928"></a><a name="p174831161928"></a>394</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1250682333414"><a name="p1250682333414"></a><a name="p1250682333414"></a>rsub.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125063238349"><a name="p125063238349"></a><a name="p125063238349"></a>rsub_npu</p>
</td>
</tr>
<tr id="row2736539161210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124833164215"><a name="p124833164215"></a><a name="p124833164215"></a>395</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125061123143411"><a name="p125061123143411"></a><a name="p125061123143411"></a>rsub.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p155062232345"><a name="p155062232345"></a><a name="p155062232345"></a>rsub_npu</p>
</td>
</tr>
<tr id="row1736139121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54837161825"><a name="p54837161825"></a><a name="p54837161825"></a>396</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450616236347"><a name="p450616236347"></a><a name="p450616236347"></a>addmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p150619237341"><a name="p150619237341"></a><a name="p150619237341"></a>addmm_out_npu</p>
</td>
</tr>
<tr id="row373683917121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948311161028"><a name="p948311161028"></a><a name="p948311161028"></a>397</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p050616232343"><a name="p050616232343"></a><a name="p050616232343"></a>addmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1350613235348"><a name="p1350613235348"></a><a name="p1350613235348"></a>addmm_npu</p>
</td>
</tr>
<tr id="row1773611392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048319161218"><a name="p1048319161218"></a><a name="p1048319161218"></a>398</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10506142319345"><a name="p10506142319345"></a><a name="p10506142319345"></a>addmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1450682393414"><a name="p1450682393414"></a><a name="p1450682393414"></a>addmm_npu_</p>
</td>
</tr>
<tr id="row2736539161217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124831416728"><a name="p124831416728"></a><a name="p124831416728"></a>399</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10506423163415"><a name="p10506423163415"></a><a name="p10506423163415"></a>quantize_per_tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4506172343413"><a name="p4506172343413"></a><a name="p4506172343413"></a>quantize_per_tensor_npu</p>
</td>
</tr>
<tr id="row11737239181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54834161021"><a name="p54834161021"></a><a name="p54834161021"></a>400</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p950662343411"><a name="p950662343411"></a><a name="p950662343411"></a>quantize_per_channel</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65061423133417"><a name="p65061423133417"></a><a name="p65061423133417"></a>quantize_per_channel_npu</p>
</td>
</tr>
<tr id="row17737439171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048312163211"><a name="p1048312163211"></a><a name="p1048312163211"></a>401</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18506132315343"><a name="p18506132315343"></a><a name="p18506132315343"></a>to.dtype_layout</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165063235341"><a name="p165063235341"></a><a name="p165063235341"></a>to_npu</p>
</td>
</tr>
<tr id="row37372039201219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748381616213"><a name="p1748381616213"></a><a name="p1748381616213"></a>402</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450614232347"><a name="p450614232347"></a><a name="p450614232347"></a>to.device</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13506172353414"><a name="p13506172353414"></a><a name="p13506172353414"></a>to_device_npu</p>
</td>
</tr>
<tr id="row167378397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2048321613210"><a name="p2048321613210"></a><a name="p2048321613210"></a>403</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65061423183412"><a name="p65061423183412"></a><a name="p65061423183412"></a>to.dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p550682393416"><a name="p550682393416"></a><a name="p550682393416"></a>to_dtype_npu</p>
</td>
</tr>
<tr id="row10737113912126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14842161621"><a name="p14842161621"></a><a name="p14842161621"></a>404</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2050692373415"><a name="p2050692373415"></a><a name="p2050692373415"></a>to.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1550662315345"><a name="p1550662315345"></a><a name="p1550662315345"></a>to_other_npu</p>
</td>
</tr>
<tr id="row1873733981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548414164215"><a name="p1548414164215"></a><a name="p1548414164215"></a>405</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2050652393418"><a name="p2050652393418"></a><a name="p2050652393418"></a>_local_scalar_dense</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p105068235344"><a name="p105068235344"></a><a name="p105068235344"></a>_local_scalar_dense_npu</p>
</td>
</tr>
<tr id="row673773971216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84847161826"><a name="p84847161826"></a><a name="p84847161826"></a>406</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35061023123419"><a name="p35061023123419"></a><a name="p35061023123419"></a>lstm.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1050602319343"><a name="p1050602319343"></a><a name="p1050602319343"></a>lstm_npu</p>
</td>
</tr>
<tr id="row4737173910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104846161627"><a name="p104846161627"></a><a name="p104846161627"></a>407</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9506142343414"><a name="p9506142343414"></a><a name="p9506142343414"></a>lstm.data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0506523143414"><a name="p0506523143414"></a><a name="p0506523143414"></a>lstm_npu</p>
</td>
</tr>
<tr id="row137371539151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048413161524"><a name="p048413161524"></a><a name="p048413161524"></a>408</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7506423203414"><a name="p7506423203414"></a><a name="p7506423203414"></a>gru.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p750672315342"><a name="p750672315342"></a><a name="p750672315342"></a>gru_npu_</p>
</td>
</tr>
<tr id="row27371039181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14484121619211"><a name="p14484121619211"></a><a name="p14484121619211"></a>409</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155068231345"><a name="p155068231345"></a><a name="p155068231345"></a>_pack_padded_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125063238346"><a name="p125063238346"></a><a name="p125063238346"></a>_pack_padded_sequence_npu</p>
</td>
</tr>
<tr id="row1173783981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84844163212"><a name="p84844163212"></a><a name="p84844163212"></a>410</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1950752310341"><a name="p1950752310341"></a><a name="p1950752310341"></a>_pad_packed_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3507172311349"><a name="p3507172311349"></a><a name="p3507172311349"></a>_pad_packed_sequence_npu</p>
</td>
</tr>
<tr id="row673883991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548418163214"><a name="p1548418163214"></a><a name="p1548418163214"></a>411</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p115071623193417"><a name="p115071623193417"></a><a name="p115071623193417"></a>set_.source_Storage</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7507182353419"><a name="p7507182353419"></a><a name="p7507182353419"></a>set_npu_</p>
</td>
</tr>
<tr id="row19738173951218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2484131617213"><a name="p2484131617213"></a><a name="p2484131617213"></a>412</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5507223113417"><a name="p5507223113417"></a><a name="p5507223113417"></a>set_.source_Storage_storage_offset</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1750717238341"><a name="p1750717238341"></a><a name="p1750717238341"></a>set_npu_</p>
</td>
</tr>
<tr id="row27381039101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154841416629"><a name="p154841416629"></a><a name="p154841416629"></a>413</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p150792393415"><a name="p150792393415"></a><a name="p150792393415"></a>set_.source_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7507623103410"><a name="p7507623103410"></a><a name="p7507623103410"></a>set_npu_</p>
</td>
</tr>
<tr id="row16738939101214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948421616218"><a name="p948421616218"></a><a name="p948421616218"></a>414</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1250762383419"><a name="p1250762383419"></a><a name="p1250762383419"></a>set_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1550712230343"><a name="p1550712230343"></a><a name="p1550712230343"></a>set_npu_</p>
</td>
</tr>
<tr id="row273863917124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p348451610215"><a name="p348451610215"></a><a name="p348451610215"></a>415</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17507023153417"><a name="p17507023153417"></a><a name="p17507023153417"></a>masked_fill_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5507192363417"><a name="p5507192363417"></a><a name="p5507192363417"></a>masked_fill_npu_</p>
</td>
</tr>
<tr id="row11738183918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84841016325"><a name="p84841016325"></a><a name="p84841016325"></a>416</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65071523163413"><a name="p65071523163413"></a><a name="p65071523163413"></a>masked_fill_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12507123103410"><a name="p12507123103410"></a><a name="p12507123103410"></a>masked_fill_npu_</p>
</td>
</tr>
<tr id="row5738739161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104841316522"><a name="p104841316522"></a><a name="p104841316522"></a>417</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10507102316343"><a name="p10507102316343"></a><a name="p10507102316343"></a>masked_scatter_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1350752315344"><a name="p1350752315344"></a><a name="p1350752315344"></a>masked_scatter_npu_</p>
</td>
</tr>
<tr id="row273811393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24840161020"><a name="p24840161020"></a><a name="p24840161020"></a>418</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1350720237346"><a name="p1350720237346"></a><a name="p1350720237346"></a>view</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9507132310342"><a name="p9507132310342"></a><a name="p9507132310342"></a>view_npu</p>
</td>
</tr>
<tr id="row13738739111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3484101617211"><a name="p3484101617211"></a><a name="p3484101617211"></a>419</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1750782343413"><a name="p1750782343413"></a><a name="p1750782343413"></a>put_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15081323173416"><a name="p15081323173416"></a><a name="p15081323173416"></a>put_npu_</p>
</td>
</tr>
<tr id="row773812399125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5484151613217"><a name="p5484151613217"></a><a name="p5484151613217"></a>420</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10508202317344"><a name="p10508202317344"></a><a name="p10508202317344"></a>index_add_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11508323173413"><a name="p11508323173413"></a><a name="p11508323173413"></a>index_add_npu_</p>
</td>
</tr>
<tr id="row2738123901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64842162216"><a name="p64842162216"></a><a name="p64842162216"></a>421</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15508823113411"><a name="p15508823113411"></a><a name="p15508823113411"></a>index_add</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2508523123417"><a name="p2508523123417"></a><a name="p2508523123417"></a>index_add_npu</p>
</td>
</tr>
<tr id="row27381739161215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124858161217"><a name="p124858161217"></a><a name="p124858161217"></a>422</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2050882383412"><a name="p2050882383412"></a><a name="p2050882383412"></a>index_add.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p550832363411"><a name="p550832363411"></a><a name="p550832363411"></a>index_add_npu</p>
</td>
</tr>
<tr id="row197395394127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2485516229"><a name="p2485516229"></a><a name="p2485516229"></a>423</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16508122319344"><a name="p16508122319344"></a><a name="p16508122319344"></a>index_fill_.int_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125081123103417"><a name="p125081123103417"></a><a name="p125081123103417"></a>index_fill_npu_</p>
</td>
</tr>
<tr id="row97393398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54852161229"><a name="p54852161229"></a><a name="p54852161229"></a>424</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p850852323414"><a name="p850852323414"></a><a name="p850852323414"></a>index_fill.int_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p950811239348"><a name="p950811239348"></a><a name="p950811239348"></a>index_fill_npu</p>
</td>
</tr>
<tr id="row18739193916123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174851116725"><a name="p174851116725"></a><a name="p174851116725"></a>425</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p55081923203415"><a name="p55081923203415"></a><a name="p55081923203415"></a>index_fill_.int_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175081723103411"><a name="p175081723103411"></a><a name="p175081723103411"></a>index_fill_npu_</p>
</td>
</tr>
<tr id="row19739103918129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5485416227"><a name="p5485416227"></a><a name="p5485416227"></a>426</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65081623133419"><a name="p65081623133419"></a><a name="p65081623133419"></a>index_fill.int_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p450862343416"><a name="p450862343416"></a><a name="p450862343416"></a>index_fill_npu</p>
</td>
</tr>
<tr id="row157396393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24851316624"><a name="p24851316624"></a><a name="p24851316624"></a>427</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65081923103416"><a name="p65081923103416"></a><a name="p65081923103416"></a>scatter_.src</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2508223143413"><a name="p2508223143413"></a><a name="p2508223143413"></a>scatter_npu_</p>
</td>
</tr>
<tr id="row187391139131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648501620215"><a name="p1648501620215"></a><a name="p1648501620215"></a>428</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1050892323412"><a name="p1050892323412"></a><a name="p1050892323412"></a>scatter_.value</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p45089233345"><a name="p45089233345"></a><a name="p45089233345"></a>scatter_npu_</p>
</td>
</tr>
<tr id="row1273973941211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19485516328"><a name="p19485516328"></a><a name="p19485516328"></a>429</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p950872319345"><a name="p950872319345"></a><a name="p950872319345"></a>scatter_add_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1508142310341"><a name="p1508142310341"></a><a name="p1508142310341"></a>scatter_add_npu_</p>
</td>
</tr>
<tr id="row77391439201210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648511163215"><a name="p648511163215"></a><a name="p648511163215"></a>430</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12508223103417"><a name="p12508223103417"></a><a name="p12508223103417"></a>scatter_add</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p105081223153419"><a name="p105081223153419"></a><a name="p105081223153419"></a>scatter_add_npu</p>
</td>
</tr>
<tr id="row3739163911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9485416425"><a name="p9485416425"></a><a name="p9485416425"></a>431</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1450815236341"><a name="p1450815236341"></a><a name="p1450815236341"></a>scatter_add.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1650822303413"><a name="p1650822303413"></a><a name="p1650822303413"></a>scatter_add_npu</p>
</td>
</tr>
<tr id="row37391539141215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12485141616210"><a name="p12485141616210"></a><a name="p12485141616210"></a>432</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0508192320345"><a name="p0508192320345"></a><a name="p0508192320345"></a>lt_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1850832323419"><a name="p1850832323419"></a><a name="p1850832323419"></a>lt_npu_</p>
</td>
</tr>
<tr id="row573993941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44851016222"><a name="p44851016222"></a><a name="p44851016222"></a>433</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p950817230341"><a name="p950817230341"></a><a name="p950817230341"></a>lt_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3508102353415"><a name="p3508102353415"></a><a name="p3508102353415"></a>lt_npu_</p>
</td>
</tr>
<tr id="row3740239171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15485141617211"><a name="p15485141617211"></a><a name="p15485141617211"></a>434</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p450972312345"><a name="p450972312345"></a><a name="p450972312345"></a>gt_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14509223103412"><a name="p14509223103412"></a><a name="p14509223103412"></a>gt_npu_</p>
</td>
</tr>
<tr id="row874013971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94851716128"><a name="p94851716128"></a><a name="p94851716128"></a>435</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85091323153419"><a name="p85091323153419"></a><a name="p85091323153419"></a>gt_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17509122317348"><a name="p17509122317348"></a><a name="p17509122317348"></a>gt_npu_</p>
</td>
</tr>
<tr id="row67401395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3486116926"><a name="p3486116926"></a><a name="p3486116926"></a>436</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3509202315346"><a name="p3509202315346"></a><a name="p3509202315346"></a>le_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2509112317344"><a name="p2509112317344"></a><a name="p2509112317344"></a>le_npu_</p>
</td>
</tr>
<tr id="row13740439201212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948613161926"><a name="p948613161926"></a><a name="p948613161926"></a>437</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1350922323412"><a name="p1350922323412"></a><a name="p1350922323412"></a>le_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p450972323412"><a name="p450972323412"></a><a name="p450972323412"></a>le_npu_</p>
</td>
</tr>
<tr id="row1174013916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448661614217"><a name="p448661614217"></a><a name="p448661614217"></a>438</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p350992311343"><a name="p350992311343"></a><a name="p350992311343"></a>ge_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16509923143411"><a name="p16509923143411"></a><a name="p16509923143411"></a>ge_npu_</p>
</td>
</tr>
<tr id="row774016390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1248616161221"><a name="p1248616161221"></a><a name="p1248616161221"></a>439</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1350942383416"><a name="p1350942383416"></a><a name="p1350942383416"></a>ge_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p105091423173411"><a name="p105091423173411"></a><a name="p105091423173411"></a>ge_npu_</p>
</td>
</tr>
<tr id="row7740193917126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848620161423"><a name="p1848620161423"></a><a name="p1848620161423"></a>440</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65091123193415"><a name="p65091123193415"></a><a name="p65091123193415"></a>eq_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115096234342"><a name="p115096234342"></a><a name="p115096234342"></a>eq_npu_</p>
</td>
</tr>
<tr id="row67401439171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648691611212"><a name="p1648691611212"></a><a name="p1648691611212"></a>441</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16509122313417"><a name="p16509122313417"></a><a name="p16509122313417"></a>eq_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p205099233342"><a name="p205099233342"></a><a name="p205099233342"></a>eq_npu_</p>
</td>
</tr>
<tr id="row1174003910120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194861216228"><a name="p194861216228"></a><a name="p194861216228"></a>442</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0509162363419"><a name="p0509162363419"></a><a name="p0509162363419"></a>ne_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1650982312347"><a name="p1650982312347"></a><a name="p1650982312347"></a>ne_npu_</p>
</td>
</tr>
<tr id="row12740739111211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44861016721"><a name="p44861016721"></a><a name="p44861016721"></a>443</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2050972393415"><a name="p2050972393415"></a><a name="p2050972393415"></a>ne_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p105091323103418"><a name="p105091323103418"></a><a name="p105091323103418"></a>ne_npu_</p>
</td>
</tr>
<tr id="row18740163913121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148616161926"><a name="p1148616161926"></a><a name="p1148616161926"></a>444</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1650962316345"><a name="p1650962316345"></a><a name="p1650962316345"></a>bitwise_and.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1250902363411"><a name="p1250902363411"></a><a name="p1250902363411"></a>bitwise_and_out_npu</p>
</td>
</tr>
<tr id="row574163931217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1486816423"><a name="p1486816423"></a><a name="p1486816423"></a>445</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10509623103414"><a name="p10509623103414"></a><a name="p10509623103414"></a>bitwise_and.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5509162333415"><a name="p5509162333415"></a><a name="p5509162333415"></a>bitwise_and_out_npu</p>
</td>
</tr>
<tr id="row1774114393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6486111614216"><a name="p6486111614216"></a><a name="p6486111614216"></a>446</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14509112393410"><a name="p14509112393410"></a><a name="p14509112393410"></a>bitwise_and.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55094239341"><a name="p55094239341"></a><a name="p55094239341"></a>bitwise_and_npu</p>
</td>
</tr>
<tr id="row14741639161211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14869169213"><a name="p14869169213"></a><a name="p14869169213"></a>447</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45091823133419"><a name="p45091823133419"></a><a name="p45091823133419"></a>bitwise_and.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1350918230341"><a name="p1350918230341"></a><a name="p1350918230341"></a>bitwise_and_npu</p>
</td>
</tr>
<tr id="row0741193911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048613162213"><a name="p048613162213"></a><a name="p048613162213"></a>448</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15098231345"><a name="p15098231345"></a><a name="p15098231345"></a>bitwise_and_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85092023133420"><a name="p85092023133420"></a><a name="p85092023133420"></a>bitwise_and_npu_</p>
</td>
</tr>
<tr id="row6741839161215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548619162217"><a name="p548619162217"></a><a name="p548619162217"></a>449</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1650982317345"><a name="p1650982317345"></a><a name="p1650982317345"></a>bitwise_and_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8509132373410"><a name="p8509132373410"></a><a name="p8509132373410"></a>bitwise_and_npu_</p>
</td>
</tr>
<tr id="row9741193991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16486716328"><a name="p16486716328"></a><a name="p16486716328"></a>450</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p350911232340"><a name="p350911232340"></a><a name="p350911232340"></a>__and__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1851052313347"><a name="p1851052313347"></a><a name="p1851052313347"></a>__and___npu</p>
</td>
</tr>
<tr id="row974103910121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048613168218"><a name="p048613168218"></a><a name="p048613168218"></a>451</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11510142314341"><a name="p11510142314341"></a><a name="p11510142314341"></a>__and__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12510102314341"><a name="p12510102314341"></a><a name="p12510102314341"></a>__and___npu</p>
</td>
</tr>
<tr id="row1741103911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948615169212"><a name="p948615169212"></a><a name="p948615169212"></a>452</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1151082313347"><a name="p1151082313347"></a><a name="p1151082313347"></a>bitwise_or.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19510192312345"><a name="p19510192312345"></a><a name="p19510192312345"></a>bitwise_or_out_npu</p>
</td>
</tr>
<tr id="row1674113914126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184861161224"><a name="p184861161224"></a><a name="p184861161224"></a>453</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35101923173410"><a name="p35101923173410"></a><a name="p35101923173410"></a>bitwise_or.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19510132333414"><a name="p19510132333414"></a><a name="p19510132333414"></a>bitwise_or_out_npu</p>
</td>
</tr>
<tr id="row4741839101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174861816527"><a name="p174861816527"></a><a name="p174861816527"></a>454</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p651011232348"><a name="p651011232348"></a><a name="p651011232348"></a>bitwise_or.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55108236346"><a name="p55108236346"></a><a name="p55108236346"></a>bitwise_or_npu</p>
</td>
</tr>
<tr id="row137421539161220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948717168215"><a name="p948717168215"></a><a name="p948717168215"></a>455</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5510172383417"><a name="p5510172383417"></a><a name="p5510172383417"></a>bitwise_or.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1251052303413"><a name="p1251052303413"></a><a name="p1251052303413"></a>bitwise_or_npu</p>
</td>
</tr>
<tr id="row8742143911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44871016526"><a name="p44871016526"></a><a name="p44871016526"></a>456</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1851015235343"><a name="p1851015235343"></a><a name="p1851015235343"></a>bitwise_or_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151022311344"><a name="p151022311344"></a><a name="p151022311344"></a>bitwise_or_npu_</p>
</td>
</tr>
<tr id="row1274263912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15487716629"><a name="p15487716629"></a><a name="p15487716629"></a>457</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13510132353415"><a name="p13510132353415"></a><a name="p13510132353415"></a>bitwise_or_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175101723113416"><a name="p175101723113416"></a><a name="p175101723113416"></a>bitwise_or_npu_</p>
</td>
</tr>
<tr id="row1374210390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348719168215"><a name="p1348719168215"></a><a name="p1348719168215"></a>458</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1851019239348"><a name="p1851019239348"></a><a name="p1851019239348"></a>__or__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151014235347"><a name="p151014235347"></a><a name="p151014235347"></a>__or___npu</p>
</td>
</tr>
<tr id="row15742123901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4487316323"><a name="p4487316323"></a><a name="p4487316323"></a>459</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151092315344"><a name="p151092315344"></a><a name="p151092315344"></a>__or__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6510923183418"><a name="p6510923183418"></a><a name="p6510923183418"></a>__or___npu</p>
</td>
</tr>
<tr id="row2742133918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848781610212"><a name="p1848781610212"></a><a name="p1848781610212"></a>460</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1551016230346"><a name="p1551016230346"></a><a name="p1551016230346"></a>__ior__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13510182383419"><a name="p13510182383419"></a><a name="p13510182383419"></a>__ior___npu</p>
</td>
</tr>
<tr id="row1974273913128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74878161021"><a name="p74878161021"></a><a name="p74878161021"></a>461</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p551032319341"><a name="p551032319341"></a><a name="p551032319341"></a>__ior__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1051072315343"><a name="p1051072315343"></a><a name="p1051072315343"></a>__ior___npu</p>
</td>
</tr>
<tr id="row274223916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p148717162219"><a name="p148717162219"></a><a name="p148717162219"></a>462</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11510172333413"><a name="p11510172333413"></a><a name="p11510172333413"></a>bitwise_xor.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5511123153419"><a name="p5511123153419"></a><a name="p5511123153419"></a>bitwise_xor_out_npu</p>
</td>
</tr>
<tr id="row13742739201212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16487216924"><a name="p16487216924"></a><a name="p16487216924"></a>463</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155111123113419"><a name="p155111123113419"></a><a name="p155111123113419"></a>bitwise_xor.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p251116232346"><a name="p251116232346"></a><a name="p251116232346"></a>bitwise_xor_out_npu</p>
</td>
</tr>
<tr id="row3742143941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748719165213"><a name="p1748719165213"></a><a name="p1748719165213"></a>464</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175112023173413"><a name="p175112023173413"></a><a name="p175112023173413"></a>bitwise_xor.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195111723173412"><a name="p195111723173412"></a><a name="p195111723173412"></a>bitwise_xor_npu</p>
</td>
</tr>
<tr id="row57420390126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348713163210"><a name="p1348713163210"></a><a name="p1348713163210"></a>465</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p751116234346"><a name="p751116234346"></a><a name="p751116234346"></a>bitwise_xor.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p105111723193418"><a name="p105111723193418"></a><a name="p105111723193418"></a>bitwise_xor_npu</p>
</td>
</tr>
<tr id="row197431539141210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18487316622"><a name="p18487316622"></a><a name="p18487316622"></a>466</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185114236345"><a name="p185114236345"></a><a name="p185114236345"></a>bitwise_xor_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451112343420"><a name="p451112343420"></a><a name="p451112343420"></a>bitwise_xor_npu_</p>
</td>
</tr>
<tr id="row18743173911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18487116624"><a name="p18487116624"></a><a name="p18487116624"></a>467</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p165111923173412"><a name="p165111923173412"></a><a name="p165111923173412"></a>bitwise_xor_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75110232344"><a name="p75110232344"></a><a name="p75110232344"></a>bitwise_xor_npu_</p>
</td>
</tr>
<tr id="row15743103916125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14871161324"><a name="p14871161324"></a><a name="p14871161324"></a>468</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12511323163414"><a name="p12511323163414"></a><a name="p12511323163414"></a>__xor__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17511162313413"><a name="p17511162313413"></a><a name="p17511162313413"></a>__xor___npu</p>
</td>
</tr>
<tr id="row774363951216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048711162023"><a name="p1048711162023"></a><a name="p1048711162023"></a>469</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15511102313341"><a name="p15511102313341"></a><a name="p15511102313341"></a>__xor__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15511523153412"><a name="p15511523153412"></a><a name="p15511523153412"></a>__xor___npu</p>
</td>
</tr>
<tr id="row47431839151217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13488201612217"><a name="p13488201612217"></a><a name="p13488201612217"></a>470</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17511182320343"><a name="p17511182320343"></a><a name="p17511182320343"></a>__lshift__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55111823183416"><a name="p55111823183416"></a><a name="p55111823183416"></a>__lshift___npu</p>
</td>
</tr>
<tr id="row117431739171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44885167216"><a name="p44885167216"></a><a name="p44885167216"></a>471</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15111323163413"><a name="p15111323163413"></a><a name="p15111323163413"></a>__lshift__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145117236349"><a name="p145117236349"></a><a name="p145117236349"></a>__lshift___npu</p>
</td>
</tr>
<tr id="row14743639201214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448815161827"><a name="p448815161827"></a><a name="p448815161827"></a>472</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5511192318343"><a name="p5511192318343"></a><a name="p5511192318343"></a>__ilshift__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p551182393412"><a name="p551182393412"></a><a name="p551182393412"></a>__iLshift___npu</p>
</td>
</tr>
<tr id="row16743183921211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8488131612215"><a name="p8488131612215"></a><a name="p8488131612215"></a>473</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p751142315343"><a name="p751142315343"></a><a name="p751142315343"></a>__ilshift__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25111723193411"><a name="p25111723193411"></a><a name="p25111723193411"></a>__iLshift___npu</p>
</td>
</tr>
<tr id="row4743103913128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648841617216"><a name="p1648841617216"></a><a name="p1648841617216"></a>474</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1751102312342"><a name="p1751102312342"></a><a name="p1751102312342"></a>__rshift__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1651102393416"><a name="p1651102393416"></a><a name="p1651102393416"></a>__rshift___npu</p>
</td>
</tr>
<tr id="row137431039181217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124884161627"><a name="p124884161627"></a><a name="p124884161627"></a>475</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p135111123113412"><a name="p135111123113412"></a><a name="p135111123113412"></a>__rshift__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451113239342"><a name="p451113239342"></a><a name="p451113239342"></a>__rshift___npu</p>
</td>
</tr>
<tr id="row674333911217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12488111614211"><a name="p12488111614211"></a><a name="p12488111614211"></a>476</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18511112316344"><a name="p18511112316344"></a><a name="p18511112316344"></a>__irshift__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14511102323410"><a name="p14511102323410"></a><a name="p14511102323410"></a>__iRshift___npu</p>
</td>
</tr>
<tr id="row1374313971215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9488191612213"><a name="p9488191612213"></a><a name="p9488191612213"></a>477</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1511823133412"><a name="p1511823133412"></a><a name="p1511823133412"></a>__irshift__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p851232333412"><a name="p851232333412"></a><a name="p851232333412"></a>__iRshift___npu</p>
</td>
</tr>
<tr id="row9744133919120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144881516723"><a name="p144881516723"></a><a name="p144881516723"></a>478</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1051292383418"><a name="p1051292383418"></a><a name="p1051292383418"></a>atan2_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6512152319346"><a name="p6512152319346"></a><a name="p6512152319346"></a>atan2_npu_</p>
</td>
</tr>
<tr id="row37441439121211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348831619216"><a name="p1348831619216"></a><a name="p1348831619216"></a>479</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1512192393410"><a name="p1512192393410"></a><a name="p1512192393410"></a>tril_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p651217231344"><a name="p651217231344"></a><a name="p651217231344"></a>tril_npu_</p>
</td>
</tr>
<tr id="row1774413961216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15488716822"><a name="p15488716822"></a><a name="p15488716822"></a>480</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5512182353419"><a name="p5512182353419"></a><a name="p5512182353419"></a>triu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125123236344"><a name="p125123236344"></a><a name="p125123236344"></a>triu_npu_</p>
</td>
</tr>
<tr id="row12744153911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848811161924"><a name="p1848811161924"></a><a name="p1848811161924"></a>481</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12512112353410"><a name="p12512112353410"></a><a name="p12512112353410"></a>renorm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151212232346"><a name="p151212232346"></a><a name="p151212232346"></a>renorm_npu_</p>
</td>
</tr>
<tr id="row197441439141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748814161327"><a name="p1748814161327"></a><a name="p1748814161327"></a>482</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15122023163415"><a name="p15122023163415"></a><a name="p15122023163415"></a>pow_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1851202311341"><a name="p1851202311341"></a><a name="p1851202311341"></a>pow_npu_</p>
</td>
</tr>
<tr id="row77441939151220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194883161223"><a name="p194883161223"></a><a name="p194883161223"></a>483</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p051212312349"><a name="p051212312349"></a><a name="p051212312349"></a>pow_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p155121923133411"><a name="p155121923133411"></a><a name="p155121923133411"></a>pow_npu_</p>
</td>
</tr>
<tr id="row10744193918120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1488111619216"><a name="p1488111619216"></a><a name="p1488111619216"></a>484</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451262303410"><a name="p1451262303410"></a><a name="p1451262303410"></a>lerp_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13512102383413"><a name="p13512102383413"></a><a name="p13512102383413"></a>lerp_npu_</p>
</td>
</tr>
<tr id="row574493911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4489316329"><a name="p4489316329"></a><a name="p4489316329"></a>485</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125129238344"><a name="p125129238344"></a><a name="p125129238344"></a>lerp_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11512142319345"><a name="p11512142319345"></a><a name="p11512142319345"></a>lerp_npu_</p>
</td>
</tr>
<tr id="row4744123901217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148981611212"><a name="p1148981611212"></a><a name="p1148981611212"></a>486</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65121123103411"><a name="p65121123103411"></a><a name="p65121123103411"></a>fmod_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18512192323410"><a name="p18512192323410"></a><a name="p18512192323410"></a>fmod_npu_</p>
</td>
</tr>
<tr id="row874423915126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848915161423"><a name="p848915161423"></a><a name="p848915161423"></a>487</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95122023123414"><a name="p95122023123414"></a><a name="p95122023123414"></a>fmod_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15128231342"><a name="p15128231342"></a><a name="p15128231342"></a>fmod_npu_</p>
</td>
</tr>
<tr id="row1774411397122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114898164213"><a name="p114898164213"></a><a name="p114898164213"></a>488</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75123237348"><a name="p75123237348"></a><a name="p75123237348"></a>remainder_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15512723173411"><a name="p15512723173411"></a><a name="p15512723173411"></a>remainder_npu_</p>
</td>
</tr>
<tr id="row1074423913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04891916228"><a name="p04891916228"></a><a name="p04891916228"></a>489</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15512152315348"><a name="p15512152315348"></a><a name="p15512152315348"></a>remainder_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1251282314344"><a name="p1251282314344"></a><a name="p1251282314344"></a>remainder_npu_</p>
</td>
</tr>
<tr id="row14745239101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184895164211"><a name="p184895164211"></a><a name="p184895164211"></a>490</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175121123193416"><a name="p175121123193416"></a><a name="p175121123193416"></a>addbmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18512323193417"><a name="p18512323193417"></a><a name="p18512323193417"></a>addbmm_npu_</p>
</td>
</tr>
<tr id="row67451339151211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348901615217"><a name="p1348901615217"></a><a name="p1348901615217"></a>491</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19512223113419"><a name="p19512223113419"></a><a name="p19512223113419"></a>addbmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8512132315346"><a name="p8512132315346"></a><a name="p8512132315346"></a>addbmm_out_npu</p>
</td>
</tr>
<tr id="row5745239191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5489316928"><a name="p5489316928"></a><a name="p5489316928"></a>492</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13512223173410"><a name="p13512223173410"></a><a name="p13512223173410"></a>addbmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3512423103415"><a name="p3512423103415"></a><a name="p3512423103415"></a>addbmm_npu</p>
</td>
</tr>
<tr id="row174516395120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94891716826"><a name="p94891716826"></a><a name="p94891716826"></a>493</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15138232341"><a name="p15138232341"></a><a name="p15138232341"></a>addcdiv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165131023193413"><a name="p165131023193413"></a><a name="p165131023193413"></a>addcdiv_npu_</p>
</td>
</tr>
<tr id="row1174518399127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748912166214"><a name="p1748912166214"></a><a name="p1748912166214"></a>494</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p351352343412"><a name="p351352343412"></a><a name="p351352343412"></a>random_.from</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25136231348"><a name="p25136231348"></a><a name="p25136231348"></a>random_npu_</p>
</td>
</tr>
<tr id="row16745123913122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848911612218"><a name="p848911612218"></a><a name="p848911612218"></a>495</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185132023183415"><a name="p185132023183415"></a><a name="p185132023183415"></a>random_.to</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p185135235340"><a name="p185135235340"></a><a name="p185135235340"></a>random_npu_</p>
</td>
</tr>
<tr id="row197450393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p748915161220"><a name="p748915161220"></a><a name="p748915161220"></a>496</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19513112316347"><a name="p19513112316347"></a><a name="p19513112316347"></a>random_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19513152363410"><a name="p19513152363410"></a><a name="p19513152363410"></a>random_npu_</p>
</td>
</tr>
<tr id="row11745739111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648914163218"><a name="p648914163218"></a><a name="p648914163218"></a>497</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5513923193413"><a name="p5513923193413"></a><a name="p5513923193413"></a>uniform_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75134231343"><a name="p75134231343"></a><a name="p75134231343"></a>uniform_npu_</p>
</td>
</tr>
<tr id="row177451439181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134901116021"><a name="p134901116021"></a><a name="p134901116021"></a>498</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1051302316348"><a name="p1051302316348"></a><a name="p1051302316348"></a>diag.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5513132363411"><a name="p5513132363411"></a><a name="p5513132363411"></a>diag_out_npu</p>
</td>
</tr>
<tr id="row13745143981210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174903162215"><a name="p174903162215"></a><a name="p174903162215"></a>499</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8513023113413"><a name="p8513023113413"></a><a name="p8513023113413"></a>diag</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175137236343"><a name="p175137236343"></a><a name="p175137236343"></a>diag_npu</p>
</td>
</tr>
<tr id="row9745103914127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2490216726"><a name="p2490216726"></a><a name="p2490216726"></a>500</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2513132303418"><a name="p2513132303418"></a><a name="p2513132303418"></a>cross.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p105131823103416"><a name="p105131823103416"></a><a name="p105131823103416"></a>cross_out_npu</p>
</td>
</tr>
<tr id="row1074693914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194901716527"><a name="p194901716527"></a><a name="p194901716527"></a>501</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8513112315341"><a name="p8513112315341"></a><a name="p8513112315341"></a>cross</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151317239341"><a name="p151317239341"></a><a name="p151317239341"></a>cross_npu</p>
</td>
</tr>
<tr id="row12746103917120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24908161421"><a name="p24908161421"></a><a name="p24908161421"></a>502</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1051314237346"><a name="p1051314237346"></a><a name="p1051314237346"></a>triu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p051318236340"><a name="p051318236340"></a><a name="p051318236340"></a>triu_out_npu</p>
</td>
</tr>
<tr id="row1474623981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64901216329"><a name="p64901216329"></a><a name="p64901216329"></a>503</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p55131023173410"><a name="p55131023173410"></a><a name="p55131023173410"></a>triu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2513132363410"><a name="p2513132363410"></a><a name="p2513132363410"></a>triu_npu</p>
</td>
</tr>
<tr id="row2074613920121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2490116325"><a name="p2490116325"></a><a name="p2490116325"></a>504</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175131423203412"><a name="p175131423203412"></a><a name="p175131423203412"></a>tril.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12513182314346"><a name="p12513182314346"></a><a name="p12513182314346"></a>tril_out_npu</p>
</td>
</tr>
<tr id="row16746839101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049021616210"><a name="p1049021616210"></a><a name="p1049021616210"></a>505</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p551322383417"><a name="p551322383417"></a><a name="p551322383417"></a>tril</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5513182313419"><a name="p5513182313419"></a><a name="p5513182313419"></a>tril_npu</p>
</td>
</tr>
<tr id="row1674643912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74914162213"><a name="p74914162213"></a><a name="p74914162213"></a>506</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155132236349"><a name="p155132236349"></a><a name="p155132236349"></a>tril_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p185131323103412"><a name="p185131323103412"></a><a name="p185131323103412"></a>tril_indices_npu</p>
</td>
</tr>
<tr id="row774653921210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049119161224"><a name="p1049119161224"></a><a name="p1049119161224"></a>507</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p05131223143419"><a name="p05131223143419"></a><a name="p05131223143419"></a>triu_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65132023153415"><a name="p65132023153415"></a><a name="p65132023153415"></a>triu_indices_npu</p>
</td>
</tr>
<tr id="row0746339141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2491916325"><a name="p2491916325"></a><a name="p2491916325"></a>508</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5513723193418"><a name="p5513723193418"></a><a name="p5513723193418"></a>ne.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p351362313416"><a name="p351362313416"></a><a name="p351362313416"></a>ne_out_npu</p>
</td>
</tr>
<tr id="row6748143914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6491101613213"><a name="p6491101613213"></a><a name="p6491101613213"></a>509</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451482316348"><a name="p1451482316348"></a><a name="p1451482316348"></a>ne.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12514122373419"><a name="p12514122373419"></a><a name="p12514122373419"></a>ne_npu</p>
</td>
</tr>
<tr id="row67489392120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84911416126"><a name="p84911416126"></a><a name="p84911416126"></a>510</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11514102333416"><a name="p11514102333416"></a><a name="p11514102333416"></a>ne.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1851413231344"><a name="p1851413231344"></a><a name="p1851413231344"></a>ne_out_npu</p>
</td>
</tr>
<tr id="row5748203971215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6491171620217"><a name="p6491171620217"></a><a name="p6491171620217"></a>511</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p951418239349"><a name="p951418239349"></a><a name="p951418239349"></a>ne.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3514023203414"><a name="p3514023203414"></a><a name="p3514023203414"></a>ne_npu</p>
</td>
</tr>
<tr id="row774883921211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649171619216"><a name="p1649171619216"></a><a name="p1649171619216"></a>512</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1751492316346"><a name="p1751492316346"></a><a name="p1751492316346"></a>eq.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65148238349"><a name="p65148238349"></a><a name="p65148238349"></a>eq_out_npu</p>
</td>
</tr>
<tr id="row17748203901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184913162217"><a name="p184913162217"></a><a name="p184913162217"></a>513</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1051452319340"><a name="p1051452319340"></a><a name="p1051452319340"></a>eq.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1051412363412"><a name="p1051412363412"></a><a name="p1051412363412"></a>eq_npu</p>
</td>
</tr>
<tr id="row147481539151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94912161728"><a name="p94912161728"></a><a name="p94912161728"></a>514</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1151417238348"><a name="p1151417238348"></a><a name="p1151417238348"></a>eq.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1151412236346"><a name="p1151412236346"></a><a name="p1151412236346"></a>eq_out_npu</p>
</td>
</tr>
<tr id="row177481139191214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349117161621"><a name="p349117161621"></a><a name="p349117161621"></a>515</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105141423133412"><a name="p105141423133412"></a><a name="p105141423133412"></a>eq.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8514162303417"><a name="p8514162303417"></a><a name="p8514162303417"></a>eq_npu</p>
</td>
</tr>
<tr id="row87480397120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184911316522"><a name="p184911316522"></a><a name="p184911316522"></a>516</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7514142303413"><a name="p7514142303413"></a><a name="p7514142303413"></a>ge.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3514102313410"><a name="p3514102313410"></a><a name="p3514102313410"></a>ge_out_npu</p>
</td>
</tr>
<tr id="row7748163971216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10491141612217"><a name="p10491141612217"></a><a name="p10491141612217"></a>517</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20514172353411"><a name="p20514172353411"></a><a name="p20514172353411"></a>ge.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19514132383415"><a name="p19514132383415"></a><a name="p19514132383415"></a>ge_npu</p>
</td>
</tr>
<tr id="row0748239151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1491111611219"><a name="p1491111611219"></a><a name="p1491111611219"></a>518</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451472333412"><a name="p1451472333412"></a><a name="p1451472333412"></a>ge.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1351492314343"><a name="p1351492314343"></a><a name="p1351492314343"></a>ge_out_npu</p>
</td>
</tr>
<tr id="row12748133913129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149112161722"><a name="p1149112161722"></a><a name="p1149112161722"></a>519</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10514192383414"><a name="p10514192383414"></a><a name="p10514192383414"></a>ge.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1151462303413"><a name="p1151462303413"></a><a name="p1151462303413"></a>ge_npu</p>
</td>
</tr>
<tr id="row1474915397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74921816428"><a name="p74921816428"></a><a name="p74921816428"></a>520</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95141623193418"><a name="p95141623193418"></a><a name="p95141623193418"></a>le.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11514923123412"><a name="p11514923123412"></a><a name="p11514923123412"></a>le_out_npu</p>
</td>
</tr>
<tr id="row18749153921216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649217161928"><a name="p649217161928"></a><a name="p649217161928"></a>521</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8514202315342"><a name="p8514202315342"></a><a name="p8514202315342"></a>le.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1951415235345"><a name="p1951415235345"></a><a name="p1951415235345"></a>le_npu</p>
</td>
</tr>
<tr id="row1674923991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54922161211"><a name="p54922161211"></a><a name="p54922161211"></a>522</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1051442315347"><a name="p1051442315347"></a><a name="p1051442315347"></a>le.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18514112313411"><a name="p18514112313411"></a><a name="p18514112313411"></a>le_out_npu</p>
</td>
</tr>
<tr id="row12749153919120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84922161524"><a name="p84922161524"></a><a name="p84922161524"></a>523</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p751452373419"><a name="p751452373419"></a><a name="p751452373419"></a>le.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13514123123416"><a name="p13514123123416"></a><a name="p13514123123416"></a>le_npu</p>
</td>
</tr>
<tr id="row12749339171214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14492916124"><a name="p14492916124"></a><a name="p14492916124"></a>524</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125147237344"><a name="p125147237344"></a><a name="p125147237344"></a>gt.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10514142393414"><a name="p10514142393414"></a><a name="p10514142393414"></a>gt_out_npu</p>
</td>
</tr>
<tr id="row1749193913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8492171612220"><a name="p8492171612220"></a><a name="p8492171612220"></a>525</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p251511234347"><a name="p251511234347"></a><a name="p251511234347"></a>gt.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1251592373416"><a name="p1251592373416"></a><a name="p1251592373416"></a>gt_npu</p>
</td>
</tr>
<tr id="row1474913393124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1449251616216"><a name="p1449251616216"></a><a name="p1449251616216"></a>526</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1251572318344"><a name="p1251572318344"></a><a name="p1251572318344"></a>gt.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7515192363414"><a name="p7515192363414"></a><a name="p7515192363414"></a>gt_out_npu</p>
</td>
</tr>
<tr id="row87491639201213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10492181619219"><a name="p10492181619219"></a><a name="p10492181619219"></a>527</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95151523103412"><a name="p95151523103412"></a><a name="p95151523103412"></a>gt.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1751518234348"><a name="p1751518234348"></a><a name="p1751518234348"></a>gt_npu</p>
</td>
</tr>
<tr id="row0749113919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24922161127"><a name="p24922161127"></a><a name="p24922161127"></a>528</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19515223163418"><a name="p19515223163418"></a><a name="p19515223163418"></a>lt.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p951542383410"><a name="p951542383410"></a><a name="p951542383410"></a>lt_out_npu</p>
</td>
</tr>
<tr id="row374933913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2492816122"><a name="p2492816122"></a><a name="p2492816122"></a>529</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4515152383410"><a name="p4515152383410"></a><a name="p4515152383410"></a>lt.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15150232347"><a name="p15150232347"></a><a name="p15150232347"></a>lt_npu</p>
</td>
</tr>
<tr id="row3749339111220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94924165210"><a name="p94924165210"></a><a name="p94924165210"></a>530</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3515162353412"><a name="p3515162353412"></a><a name="p3515162353412"></a>lt.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7515423143419"><a name="p7515423143419"></a><a name="p7515423143419"></a>lt_out_npu</p>
</td>
</tr>
<tr id="row117501939131219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194924161022"><a name="p194924161022"></a><a name="p194924161022"></a>531</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11515132313346"><a name="p11515132313346"></a><a name="p11515132313346"></a>lt.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5515202303419"><a name="p5515202303419"></a><a name="p5515202303419"></a>lt_npu</p>
</td>
</tr>
<tr id="row13750123914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649217161124"><a name="p1649217161124"></a><a name="p1649217161124"></a>532</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p351510238347"><a name="p351510238347"></a><a name="p351510238347"></a>take.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8515923113418"><a name="p8515923113418"></a><a name="p8515923113418"></a>take_out_npu</p>
</td>
</tr>
<tr id="row47504399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114923161120"><a name="p114923161120"></a><a name="p114923161120"></a>533</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65154234346"><a name="p65154234346"></a><a name="p65154234346"></a>take</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12515192353414"><a name="p12515192353414"></a><a name="p12515192353414"></a>take_npu</p>
</td>
</tr>
<tr id="row0750163971213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24921016328"><a name="p24921016328"></a><a name="p24921016328"></a>534</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451510237341"><a name="p1451510237341"></a><a name="p1451510237341"></a>index_select.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p95151523153413"><a name="p95151523153413"></a><a name="p95151523153413"></a>index_select_out_npu</p>
</td>
</tr>
<tr id="row07509395124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949213161621"><a name="p949213161621"></a><a name="p949213161621"></a>535</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4515112316344"><a name="p4515112316344"></a><a name="p4515112316344"></a>index_select</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6515723143419"><a name="p6515723143419"></a><a name="p6515723143419"></a>index_select_npu</p>
</td>
</tr>
<tr id="row197501839181212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154921016224"><a name="p154921016224"></a><a name="p154921016224"></a>536</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p55151231345"><a name="p55151231345"></a><a name="p55151231345"></a>index_select.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10515123113417"><a name="p10515123113417"></a><a name="p10515123113417"></a>index_select_out_npu</p>
</td>
</tr>
<tr id="row1075017392123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18492181618220"><a name="p18492181618220"></a><a name="p18492181618220"></a>537</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18515142343413"><a name="p18515142343413"></a><a name="p18515142343413"></a>index_select.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p251582317347"><a name="p251582317347"></a><a name="p251582317347"></a>index_select_npu</p>
</td>
</tr>
<tr id="row1375017398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184927163211"><a name="p184927163211"></a><a name="p184927163211"></a>538</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p951552323414"><a name="p951552323414"></a><a name="p951552323414"></a>masked_select.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p951532313413"><a name="p951532313413"></a><a name="p951532313413"></a>masked_select_out_npu</p>
</td>
</tr>
<tr id="row0750739111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249316167213"><a name="p249316167213"></a><a name="p249316167213"></a>539</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1251516238343"><a name="p1251516238343"></a><a name="p1251516238343"></a>masked_select</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2515152310342"><a name="p2515152310342"></a><a name="p2515152310342"></a>masked_select_npu</p>
</td>
</tr>
<tr id="row13750939151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949317161626"><a name="p949317161626"></a><a name="p949317161626"></a>540</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8515122333411"><a name="p8515122333411"></a><a name="p8515122333411"></a>nonzero.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p135151239347"><a name="p135151239347"></a><a name="p135151239347"></a>nonzero_out_npu</p>
</td>
</tr>
<tr id="row1175014398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20493171615215"><a name="p20493171615215"></a><a name="p20493171615215"></a>541</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16516132373411"><a name="p16516132373411"></a><a name="p16516132373411"></a>nonzero</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6516723143411"><a name="p6516723143411"></a><a name="p6516723143411"></a>nonzero_npu</p>
</td>
</tr>
<tr id="row2751163911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8493111611212"><a name="p8493111611212"></a><a name="p8493111611212"></a>542</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7516182313342"><a name="p7516182313342"></a><a name="p7516182313342"></a>gather.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4516152353412"><a name="p4516152353412"></a><a name="p4516152353412"></a>gather_out_npu</p>
</td>
</tr>
<tr id="row4751113917122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849312167218"><a name="p849312167218"></a><a name="p849312167218"></a>543</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17516162313420"><a name="p17516162313420"></a><a name="p17516162313420"></a>gather</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18516192312348"><a name="p18516192312348"></a><a name="p18516192312348"></a>gather_npu</p>
</td>
</tr>
<tr id="row1875113398121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049315161020"><a name="p1049315161020"></a><a name="p1049315161020"></a>544</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75161323173419"><a name="p75161323173419"></a><a name="p75161323173419"></a>gather.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p851642383413"><a name="p851642383413"></a><a name="p851642383413"></a>gather_out_npu</p>
</td>
</tr>
<tr id="row117511339141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20493141612212"><a name="p20493141612212"></a><a name="p20493141612212"></a>545</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1516132343411"><a name="p1516132343411"></a><a name="p1516132343411"></a>gather.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p05166236343"><a name="p05166236343"></a><a name="p05166236343"></a>gather_npu</p>
</td>
</tr>
<tr id="row47513398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049321619219"><a name="p1049321619219"></a><a name="p1049321619219"></a>546</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451652323410"><a name="p1451652323410"></a><a name="p1451652323410"></a>addcmul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16516723163417"><a name="p16516723163417"></a><a name="p16516723163417"></a>addcmul_out_npu</p>
</td>
</tr>
<tr id="row177517395121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649311161213"><a name="p649311161213"></a><a name="p649311161213"></a>547</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8516172314341"><a name="p8516172314341"></a><a name="p8516172314341"></a>addcmul</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7516122343418"><a name="p7516122343418"></a><a name="p7516122343418"></a>addcmul_npu</p>
</td>
</tr>
<tr id="row2751193991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15493016624"><a name="p15493016624"></a><a name="p15493016624"></a>548</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15516323193420"><a name="p15516323193420"></a><a name="p15516323193420"></a>addcmul_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75161523173412"><a name="p75161523173412"></a><a name="p75161523173412"></a>addcmul_npu_</p>
</td>
</tr>
<tr id="row18751203991216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5493191618213"><a name="p5493191618213"></a><a name="p5493191618213"></a>549</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20516152314340"><a name="p20516152314340"></a><a name="p20516152314340"></a>addcdiv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p751622363411"><a name="p751622363411"></a><a name="p751622363411"></a>addcdiv_out_npu</p>
</td>
</tr>
<tr id="row875123941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249316161223"><a name="p1249316161223"></a><a name="p1249316161223"></a>550</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p551611231349"><a name="p551611231349"></a><a name="p551611231349"></a>addcdiv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p551662333414"><a name="p551662333414"></a><a name="p551662333414"></a>addcdiv_npu</p>
</td>
</tr>
<tr id="row275114391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049314161527"><a name="p2049314161527"></a><a name="p2049314161527"></a>551</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85161123113413"><a name="p85161123113413"></a><a name="p85161123113413"></a>_triangular_solve_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p351652353414"><a name="p351652353414"></a><a name="p351652353414"></a>_triangular_solve_helper_npu</p>
</td>
</tr>
<tr id="row17751123961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17493916625"><a name="p17493916625"></a><a name="p17493916625"></a>552</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6516423163418"><a name="p6516423163418"></a><a name="p6516423163418"></a>_symeig_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1451662315346"><a name="p1451662315346"></a><a name="p1451662315346"></a>_symeig_helper_npu</p>
</td>
</tr>
<tr id="row1475113393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10493111614214"><a name="p10493111614214"></a><a name="p10493111614214"></a>553</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155164239344"><a name="p155164239344"></a><a name="p155164239344"></a>_svd_helper</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451662373418"><a name="p451662373418"></a><a name="p451662373418"></a>_svd_helper_npu</p>
</td>
</tr>
<tr id="row1752839141213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104941016626"><a name="p104941016626"></a><a name="p104941016626"></a>554</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p951618238344"><a name="p951618238344"></a><a name="p951618238344"></a>qr.Q</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1516132373419"><a name="p1516132373419"></a><a name="p1516132373419"></a>qr_out_npu</p>
</td>
</tr>
<tr id="row775233991216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1494191618216"><a name="p1494191618216"></a><a name="p1494191618216"></a>555</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195161023113416"><a name="p195161023113416"></a><a name="p195161023113416"></a>qr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5516112318348"><a name="p5516112318348"></a><a name="p5516112318348"></a>qr_npu</p>
</td>
</tr>
<tr id="row3752183961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134941616122"><a name="p134941616122"></a><a name="p134941616122"></a>556</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1516122373420"><a name="p1516122373420"></a><a name="p1516122373420"></a>multinomial.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p35160232349"><a name="p35160232349"></a><a name="p35160232349"></a>multinomial_out_npu</p>
</td>
</tr>
<tr id="row1275293918124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1549441619220"><a name="p1549441619220"></a><a name="p1549441619220"></a>557</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16517152323414"><a name="p16517152323414"></a><a name="p16517152323414"></a>multinomial</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p95171623163418"><a name="p95171623163418"></a><a name="p95171623163418"></a>multinomial_npu</p>
</td>
</tr>
<tr id="row1275214396124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194941161125"><a name="p194941161125"></a><a name="p194941161125"></a>558</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11517192311343"><a name="p11517192311343"></a><a name="p11517192311343"></a>erfinv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p551762317341"><a name="p551762317341"></a><a name="p551762317341"></a>erfinv_npu</p>
</td>
</tr>
<tr id="row1275223951213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13494151614215"><a name="p13494151614215"></a><a name="p13494151614215"></a>559</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155176230348"><a name="p155176230348"></a><a name="p155176230348"></a>erfinv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20517112313346"><a name="p20517112313346"></a><a name="p20517112313346"></a>erfinv_npu_</p>
</td>
</tr>
<tr id="row1475273915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114944161621"><a name="p114944161621"></a><a name="p114944161621"></a>560</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1151762313418"><a name="p1151762313418"></a><a name="p1151762313418"></a>erfinv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p251732316345"><a name="p251732316345"></a><a name="p251732316345"></a>erfinv_out_npu</p>
</td>
</tr>
<tr id="row1575273961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449481619218"><a name="p449481619218"></a><a name="p449481619218"></a>561</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p25171239342"><a name="p25171239342"></a><a name="p25171239342"></a>sign</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1651772311349"><a name="p1651772311349"></a><a name="p1651772311349"></a>sign_npu</p>
</td>
</tr>
<tr id="row275283919128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74941016925"><a name="p74941016925"></a><a name="p74941016925"></a>562</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175171723123418"><a name="p175171723123418"></a><a name="p175171723123418"></a>sign_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1351762383413"><a name="p1351762383413"></a><a name="p1351762383413"></a>sign_npu_</p>
</td>
</tr>
<tr id="row15752163931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249419161427"><a name="p249419161427"></a><a name="p249419161427"></a>563</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17517323173420"><a name="p17517323173420"></a><a name="p17517323173420"></a>sign.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2517102310349"><a name="p2517102310349"></a><a name="p2517102310349"></a>sign_out_npu</p>
</td>
</tr>
<tr id="row15752113921216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74941916727"><a name="p74941916727"></a><a name="p74941916727"></a>564</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17517112313342"><a name="p17517112313342"></a><a name="p17517112313342"></a>atan2.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2517102313411"><a name="p2517102313411"></a><a name="p2517102313411"></a>atan2_out_npu</p>
</td>
</tr>
<tr id="row375343912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184947161222"><a name="p184947161222"></a><a name="p184947161222"></a>565</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15171723133415"><a name="p15171723133415"></a><a name="p15171723133415"></a>atan2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1051719235349"><a name="p1051719235349"></a><a name="p1051719235349"></a>atan2_npu</p>
</td>
</tr>
<tr id="row9753203991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149414160218"><a name="p149414160218"></a><a name="p149414160218"></a>566</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1751752323420"><a name="p1751752323420"></a><a name="p1751752323420"></a>lerp.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1951712311342"><a name="p1951712311342"></a><a name="p1951712311342"></a>lerp_out_npu</p>
</td>
</tr>
<tr id="row575313910120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249551610212"><a name="p1249551610212"></a><a name="p1249551610212"></a>567</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1651732393411"><a name="p1651732393411"></a><a name="p1651732393411"></a>lerp.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18517172303412"><a name="p18517172303412"></a><a name="p18517172303412"></a>lerp_out_npu</p>
</td>
</tr>
<tr id="row1675311393122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149510169212"><a name="p1149510169212"></a><a name="p1149510169212"></a>568</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1751742320345"><a name="p1751742320345"></a><a name="p1751742320345"></a>lerp.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451752373412"><a name="p451752373412"></a><a name="p451752373412"></a>lerp_npu</p>
</td>
</tr>
<tr id="row12753193981217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149514161324"><a name="p1149514161324"></a><a name="p1149514161324"></a>569</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p251742323420"><a name="p251742323420"></a><a name="p251742323420"></a>lerp.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p251772310345"><a name="p251772310345"></a><a name="p251772310345"></a>lerp_npu</p>
</td>
</tr>
<tr id="row27537391124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849591611212"><a name="p849591611212"></a><a name="p849591611212"></a>570</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0517122373413"><a name="p0517122373413"></a><a name="p0517122373413"></a>fmod.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p051742316346"><a name="p051742316346"></a><a name="p051742316346"></a>fmod_out_npu</p>
</td>
</tr>
<tr id="row1753153911214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194950161529"><a name="p194950161529"></a><a name="p194950161529"></a>571</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6517323163414"><a name="p6517323163414"></a><a name="p6517323163414"></a>fmod.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1851772312347"><a name="p1851772312347"></a><a name="p1851772312347"></a>fmod_npu</p>
</td>
</tr>
<tr id="row8753163971217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949518161428"><a name="p949518161428"></a><a name="p949518161428"></a>572</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451812235341"><a name="p1451812235341"></a><a name="p1451812235341"></a>fmod.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1951852353413"><a name="p1951852353413"></a><a name="p1951852353413"></a>fmod_out_npu</p>
</td>
</tr>
<tr id="row1875323910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184951516925"><a name="p184951516925"></a><a name="p184951516925"></a>573</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65181523193416"><a name="p65181523193416"></a><a name="p65181523193416"></a>fmod.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145184235342"><a name="p145184235342"></a><a name="p145184235342"></a>fmod_npu</p>
</td>
</tr>
<tr id="row1775333911120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349512168210"><a name="p349512168210"></a><a name="p349512168210"></a>574</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0518423113414"><a name="p0518423113414"></a><a name="p0518423113414"></a>remainder.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151832393413"><a name="p151832393413"></a><a name="p151832393413"></a>remainder_out_npu</p>
</td>
</tr>
<tr id="row10754139131211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4495141617213"><a name="p4495141617213"></a><a name="p4495141617213"></a>575</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1451822320347"><a name="p1451822320347"></a><a name="p1451822320347"></a>remainder.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25184233346"><a name="p25184233346"></a><a name="p25184233346"></a>remainder_npu</p>
</td>
</tr>
<tr id="row127541139151210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149581615211"><a name="p149581615211"></a><a name="p149581615211"></a>576</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5518142313410"><a name="p5518142313410"></a><a name="p5518142313410"></a>remainder.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6518223133418"><a name="p6518223133418"></a><a name="p6518223133418"></a>remainder_out_npu</p>
</td>
</tr>
<tr id="row17754113913120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144956165214"><a name="p144956165214"></a><a name="p144956165214"></a>577</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p651822323415"><a name="p651822323415"></a><a name="p651822323415"></a>remainder.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1518923123416"><a name="p1518923123416"></a><a name="p1518923123416"></a>remainder_npu</p>
</td>
</tr>
<tr id="row13754639171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04954161425"><a name="p04954161425"></a><a name="p04954161425"></a>578</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1151852343410"><a name="p1151852343410"></a><a name="p1151852343410"></a>min.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1351820231349"><a name="p1351820231349"></a><a name="p1351820231349"></a>min_out_npu</p>
</td>
</tr>
<tr id="row18754143914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44951416822"><a name="p44951416822"></a><a name="p44951416822"></a>579</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151862333418"><a name="p151862333418"></a><a name="p151862333418"></a>min.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9518162383418"><a name="p9518162383418"></a><a name="p9518162383418"></a>min_npu</p>
</td>
</tr>
<tr id="row1675463991214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104953161219"><a name="p104953161219"></a><a name="p104953161219"></a>580</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1751819235346"><a name="p1751819235346"></a><a name="p1751819235346"></a>min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1451842310345"><a name="p1451842310345"></a><a name="p1451842310345"></a>min_npu</p>
</td>
</tr>
<tr id="row575410392129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13495131616220"><a name="p13495131616220"></a><a name="p13495131616220"></a>581</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1951812383415"><a name="p1951812383415"></a><a name="p1951812383415"></a>max.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85181523183411"><a name="p85181523183411"></a><a name="p85181523183411"></a>max_out_npu</p>
</td>
</tr>
<tr id="row157541139121219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164951160219"><a name="p164951160219"></a><a name="p164951160219"></a>582</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p205181523203420"><a name="p205181523203420"></a><a name="p205181523203420"></a>max.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10518142311342"><a name="p10518142311342"></a><a name="p10518142311342"></a>max_npu</p>
</td>
</tr>
<tr id="row1275423991213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1749516161921"><a name="p1749516161921"></a><a name="p1749516161921"></a>583</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1751817232344"><a name="p1751817232344"></a><a name="p1751817232344"></a>max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75183232345"><a name="p75183232345"></a><a name="p75183232345"></a>max_npu</p>
</td>
</tr>
<tr id="row5754193914121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2495161616215"><a name="p2495161616215"></a><a name="p2495161616215"></a>584</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9518423143411"><a name="p9518423143411"></a><a name="p9518423143411"></a>median</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p185185234347"><a name="p185185234347"></a><a name="p185185234347"></a>median_npu</p>
</td>
</tr>
<tr id="row18754113941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1749514161728"><a name="p1749514161728"></a><a name="p1749514161728"></a>585</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2518152316345"><a name="p2518152316345"></a><a name="p2518152316345"></a>sort.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4518823133412"><a name="p4518823133412"></a><a name="p4518823133412"></a>sort_out_npu</p>
</td>
</tr>
<tr id="row1175517394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049571610214"><a name="p2049571610214"></a><a name="p2049571610214"></a>586</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2518142310342"><a name="p2518142310342"></a><a name="p2518142310342"></a>sort</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10518182313414"><a name="p10518182313414"></a><a name="p10518182313414"></a>sort_npu</p>
</td>
</tr>
<tr id="row47551239161216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15496816329"><a name="p15496816329"></a><a name="p15496816329"></a>587</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15518323203411"><a name="p15518323203411"></a><a name="p15518323203411"></a>sort.dimname_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15181523173416"><a name="p15181523173416"></a><a name="p15181523173416"></a>sort_out_npu</p>
</td>
</tr>
<tr id="row20755739121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144967161429"><a name="p144967161429"></a><a name="p144967161429"></a>588</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1351952373411"><a name="p1351952373411"></a><a name="p1351952373411"></a>sort.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14519192318345"><a name="p14519192318345"></a><a name="p14519192318345"></a>sort_npu</p>
</td>
</tr>
<tr id="row1675517394128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p549621620217"><a name="p549621620217"></a><a name="p549621620217"></a>589</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15519142320342"><a name="p15519142320342"></a><a name="p15519142320342"></a>argsort</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p251916236349"><a name="p251916236349"></a><a name="p251916236349"></a>argsort_npu</p>
</td>
</tr>
<tr id="row17755163920126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149681614215"><a name="p149681614215"></a><a name="p149681614215"></a>590</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p05191423103414"><a name="p05191423103414"></a><a name="p05191423103414"></a>argsort.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15519112343416"><a name="p15519112343416"></a><a name="p15519112343416"></a>argsort_npu</p>
</td>
</tr>
<tr id="row167551839111212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144963169210"><a name="p144963169210"></a><a name="p144963169210"></a>591</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p951922313348"><a name="p951922313348"></a><a name="p951922313348"></a>topk.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10519192383419"><a name="p10519192383419"></a><a name="p10519192383419"></a>topk_out_npu</p>
</td>
</tr>
<tr id="row177559399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164968169216"><a name="p164968169216"></a><a name="p164968169216"></a>592</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p55198238345"><a name="p55198238345"></a><a name="p55198238345"></a>topk</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1519323193416"><a name="p1519323193416"></a><a name="p1519323193416"></a>topk_npu</p>
</td>
</tr>
<tr id="row9755539121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349613161228"><a name="p349613161228"></a><a name="p349613161228"></a>593</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1851918238346"><a name="p1851918238346"></a><a name="p1851918238346"></a>all</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p351962323417"><a name="p351962323417"></a><a name="p351962323417"></a>all_npu</p>
</td>
</tr>
<tr id="row16755203919122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849611614213"><a name="p849611614213"></a><a name="p849611614213"></a>594</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125191023203412"><a name="p125191023203412"></a><a name="p125191023203412"></a>any</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75191823133412"><a name="p75191823133412"></a><a name="p75191823133412"></a>any_npu</p>
</td>
</tr>
<tr id="row8755103913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349681613211"><a name="p349681613211"></a><a name="p349681613211"></a>595</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2519823133411"><a name="p2519823133411"></a><a name="p2519823133411"></a>renorm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1051952353410"><a name="p1051952353410"></a><a name="p1051952353410"></a>renorm_out_npu</p>
</td>
</tr>
<tr id="row16755103912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349611161928"><a name="p349611161928"></a><a name="p349611161928"></a>596</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7519182312344"><a name="p7519182312344"></a><a name="p7519182312344"></a>renorm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10519172373411"><a name="p10519172373411"></a><a name="p10519172373411"></a>renorm_npu</p>
</td>
</tr>
<tr id="row3755183901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8496151611215"><a name="p8496151611215"></a><a name="p8496151611215"></a>597</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45191223133417"><a name="p45191223133417"></a><a name="p45191223133417"></a>unfold</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p451915233346"><a name="p451915233346"></a><a name="p451915233346"></a>unfold</p>
</td>
</tr>
<tr id="row1375613961210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13496111615215"><a name="p13496111615215"></a><a name="p13496111615215"></a>598</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35209234344"><a name="p35209234344"></a><a name="p35209234344"></a>equal</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16520152319342"><a name="p16520152319342"></a><a name="p16520152319342"></a>equal_npu</p>
</td>
</tr>
<tr id="row1175623951217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1849615161424"><a name="p1849615161424"></a><a name="p1849615161424"></a>599</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1252042316343"><a name="p1252042316343"></a><a name="p1252042316343"></a>pow.Tensor_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p152018235348"><a name="p152018235348"></a><a name="p152018235348"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row275613921217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44964161029"><a name="p44964161029"></a><a name="p44964161029"></a>600</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p65202233344"><a name="p65202233344"></a><a name="p65202233344"></a>pow.Tensor_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p452010238343"><a name="p452010238343"></a><a name="p452010238343"></a>pow_npu</p>
</td>
</tr>
<tr id="row275623991214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124966163217"><a name="p124966163217"></a><a name="p124966163217"></a>601</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1852082313414"><a name="p1852082313414"></a><a name="p1852082313414"></a>pow.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14520142343410"><a name="p14520142343410"></a><a name="p14520142343410"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row17756123941211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349715161628"><a name="p349715161628"></a><a name="p349715161628"></a>602</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1552032313416"><a name="p1552032313416"></a><a name="p1552032313416"></a>pow.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25201723113419"><a name="p25201723113419"></a><a name="p25201723113419"></a>pow_npu</p>
</td>
</tr>
<tr id="row775611393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15497111618214"><a name="p15497111618214"></a><a name="p15497111618214"></a>603</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p552010234341"><a name="p552010234341"></a><a name="p552010234341"></a>normal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1852092316345"><a name="p1852092316345"></a><a name="p1852092316345"></a>normal_npu_</p>
</td>
</tr>
<tr id="row17756183951214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17497141612216"><a name="p17497141612216"></a><a name="p17497141612216"></a>604</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p252052316347"><a name="p252052316347"></a><a name="p252052316347"></a>normal.Tensor_float_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17520142303412"><a name="p17520142303412"></a><a name="p17520142303412"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row197561439121218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1349712166211"><a name="p1349712166211"></a><a name="p1349712166211"></a>605</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15204238345"><a name="p15204238345"></a><a name="p15204238345"></a>normal.Tensor_float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5520423183416"><a name="p5520423183416"></a><a name="p5520423183416"></a>normal_npu</p>
</td>
</tr>
<tr id="row37563398122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p749720161529"><a name="p749720161529"></a><a name="p749720161529"></a>606</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p252032317344"><a name="p252032317344"></a><a name="p252032317344"></a>normal.float_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11520182315346"><a name="p11520182315346"></a><a name="p11520182315346"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row1075683921219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13497141617213"><a name="p13497141617213"></a><a name="p13497141617213"></a>607</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1652072310349"><a name="p1652072310349"></a><a name="p1652072310349"></a>normal.float_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p852092316345"><a name="p852092316345"></a><a name="p852092316345"></a>normal_npu</p>
</td>
</tr>
<tr id="row975733971217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849761616210"><a name="p849761616210"></a><a name="p849761616210"></a>608</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1252092343416"><a name="p1252092343416"></a><a name="p1252092343416"></a>normal.Tensor_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1152072393414"><a name="p1152072393414"></a><a name="p1152072393414"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row157571439151211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1949741610213"><a name="p1949741610213"></a><a name="p1949741610213"></a>609</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195201723103416"><a name="p195201723103416"></a><a name="p195201723103416"></a>normal.Tensor_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13520423193418"><a name="p13520423193418"></a><a name="p13520423193418"></a>normal_npu</p>
</td>
</tr>
<tr id="row075718391126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54977161324"><a name="p54977161324"></a><a name="p54977161324"></a>610</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8520202317348"><a name="p8520202317348"></a><a name="p8520202317348"></a>normal.float_float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p752010233349"><a name="p752010233349"></a><a name="p752010233349"></a>normal_npu</p>
</td>
</tr>
<tr id="row197572391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3497116121"><a name="p3497116121"></a><a name="p3497116121"></a>611</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p652042353417"><a name="p652042353417"></a><a name="p652042353417"></a>normal.float_float_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1352092363414"><a name="p1352092363414"></a><a name="p1352092363414"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row15757173917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18497816524"><a name="p18497816524"></a><a name="p18497816524"></a>612</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p052012232349"><a name="p052012232349"></a><a name="p052012232349"></a>_addr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1152016238342"><a name="p1152016238342"></a><a name="p1152016238342"></a>_addr_npu</p>
</td>
</tr>
<tr id="row9757039131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849781614219"><a name="p849781614219"></a><a name="p849781614219"></a>613</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1452172313343"><a name="p1452172313343"></a><a name="p1452172313343"></a>_addr_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3521423163416"><a name="p3521423163416"></a><a name="p3521423163416"></a>_addr_npu_</p>
</td>
</tr>
<tr id="row1757139191214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049716161224"><a name="p2049716161224"></a><a name="p2049716161224"></a>614</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1652112238343"><a name="p1652112238343"></a><a name="p1652112238343"></a>_addr.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1552111230341"><a name="p1552111230341"></a><a name="p1552111230341"></a>_addr_out_npu</p>
</td>
</tr>
<tr id="row275716390126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649751617211"><a name="p1649751617211"></a><a name="p1649751617211"></a>615</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19521172323411"><a name="p19521172323411"></a><a name="p19521172323411"></a>_index_copy_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1952117232341"><a name="p1952117232341"></a><a name="p1952117232341"></a>index_copy_npu_</p>
</td>
</tr>
<tr id="row575717398122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5497316921"><a name="p5497316921"></a><a name="p5497316921"></a>616</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15521192317347"><a name="p15521192317347"></a><a name="p15521192317347"></a>_cumsum</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1952112233348"><a name="p1952112233348"></a><a name="p1952112233348"></a>_cumsum_npu</p>
</td>
</tr>
<tr id="row1275733901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174972161124"><a name="p174972161124"></a><a name="p174972161124"></a>617</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16521162318348"><a name="p16521162318348"></a><a name="p16521162318348"></a>_cumsum.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1652172343418"><a name="p1652172343418"></a><a name="p1652172343418"></a>_cumsum_out_npu</p>
</td>
</tr>
<tr id="row475703981210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449771612210"><a name="p449771612210"></a><a name="p449771612210"></a>618</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1252114238345"><a name="p1252114238345"></a><a name="p1252114238345"></a>_cumprod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85211823153417"><a name="p85211823153417"></a><a name="p85211823153417"></a>_cumprod_npu</p>
</td>
</tr>
<tr id="row137581539161214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949761610212"><a name="p949761610212"></a><a name="p949761610212"></a>619</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1521122373416"><a name="p1521122373416"></a><a name="p1521122373416"></a>_cumprod.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12521162315348"><a name="p12521162315348"></a><a name="p12521162315348"></a>_cumprod_out_npu</p>
</td>
</tr>
<tr id="row12758113941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84971316729"><a name="p84971316729"></a><a name="p84971316729"></a>620</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p752162363410"><a name="p752162363410"></a><a name="p752162363410"></a>_var</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10521623203416"><a name="p10521623203416"></a><a name="p10521623203416"></a>_var_npu</p>
</td>
</tr>
<tr id="row167581739101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11497171619218"><a name="p11497171619218"></a><a name="p11497171619218"></a>621</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125214239347"><a name="p125214239347"></a><a name="p125214239347"></a>_amp_non_finite_check_and_unscale_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1452113233349"><a name="p1452113233349"></a><a name="p1452113233349"></a>_amp_non_finite_check_and_unscale_npu_</p>
</td>
</tr>
<tr id="row18758143931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649813161220"><a name="p649813161220"></a><a name="p649813161220"></a>622</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1552119235348"><a name="p1552119235348"></a><a name="p1552119235348"></a>_cat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12521623163413"><a name="p12521623163413"></a><a name="p12521623163413"></a>_cat_npu</p>
</td>
</tr>
<tr id="row77581939111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84985160212"><a name="p84985160212"></a><a name="p84985160212"></a>623</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10521142343414"><a name="p10521142343414"></a><a name="p10521142343414"></a>_cat.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13521152314342"><a name="p13521152314342"></a><a name="p13521152314342"></a>_cat_out_npu</p>
</td>
</tr>
<tr id="row1575893910121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44980167211"><a name="p44980167211"></a><a name="p44980167211"></a>624</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1852132393416"><a name="p1852132393416"></a><a name="p1852132393416"></a>_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7521192353417"><a name="p7521192353417"></a><a name="p7521192353417"></a>_max_npu</p>
</td>
</tr>
<tr id="row675812396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174986164213"><a name="p174986164213"></a><a name="p174986164213"></a>625</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45211423203415"><a name="p45211423203415"></a><a name="p45211423203415"></a>_max.max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115211323183414"><a name="p115211323183414"></a><a name="p115211323183414"></a>_max_out_npu</p>
</td>
</tr>
<tr id="row1375812391128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194985164213"><a name="p194985164213"></a><a name="p194985164213"></a>626</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175216239342"><a name="p175216239342"></a><a name="p175216239342"></a>_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12521223193419"><a name="p12521223193419"></a><a name="p12521223193419"></a>_min_npu</p>
</td>
</tr>
<tr id="row5758173920123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14498181611220"><a name="p14498181611220"></a><a name="p14498181611220"></a>627</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35212234340"><a name="p35212234340"></a><a name="p35212234340"></a>_min.min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1252118235347"><a name="p1252118235347"></a><a name="p1252118235347"></a>_min_out_npu</p>
</td>
</tr>
<tr id="row2758113911211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p049861611210"><a name="p049861611210"></a><a name="p049861611210"></a>628</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5522172313417"><a name="p5522172313417"></a><a name="p5522172313417"></a>mse_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p185221423153413"><a name="p185221423153413"></a><a name="p185221423153413"></a>mse_loss_out_npu</p>
</td>
</tr>
<tr id="row975863921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34981116524"><a name="p34981116524"></a><a name="p34981116524"></a>629</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p852211237349"><a name="p852211237349"></a><a name="p852211237349"></a>mse_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1952215231344"><a name="p1952215231344"></a><a name="p1952215231344"></a>mse_loss_npu</p>
</td>
</tr>
<tr id="row3758339121211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p549818161224"><a name="p549818161224"></a><a name="p549818161224"></a>630</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p752242313412"><a name="p752242313412"></a><a name="p752242313412"></a>mse_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1452282303414"><a name="p1452282303414"></a><a name="p1452282303414"></a>mse_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row18759193912122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8498181611210"><a name="p8498181611210"></a><a name="p8498181611210"></a>631</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p252262393417"><a name="p252262393417"></a><a name="p252262393417"></a>mse_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85220230342"><a name="p85220230342"></a><a name="p85220230342"></a>mse_loss_backward_npu</p>
</td>
</tr>
<tr id="row5759103991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149881614210"><a name="p1149881614210"></a><a name="p1149881614210"></a>632</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p115221523103410"><a name="p115221523103410"></a><a name="p115221523103410"></a>l1_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115222023103413"><a name="p115222023103413"></a><a name="p115222023103413"></a>l1_loss_out_npu</p>
</td>
</tr>
<tr id="row11759163919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849819161024"><a name="p849819161024"></a><a name="p849819161024"></a>633</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1522142314347"><a name="p1522142314347"></a><a name="p1522142314347"></a>l1_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85221323123411"><a name="p85221323123411"></a><a name="p85221323123411"></a>l1_loss_npu</p>
</td>
</tr>
<tr id="row13759133910129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54981716923"><a name="p54981716923"></a><a name="p54981716923"></a>634</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p252292319346"><a name="p252292319346"></a><a name="p252292319346"></a>l1_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1522102353420"><a name="p1522102353420"></a><a name="p1522102353420"></a>l1_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row57591039171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114983162219"><a name="p114983162219"></a><a name="p114983162219"></a>635</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16522423143417"><a name="p16522423143417"></a><a name="p16522423143417"></a>l1_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1752220235342"><a name="p1752220235342"></a><a name="p1752220235342"></a>l1_loss_backward_npu</p>
</td>
</tr>
<tr id="row20759193931213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10498111612210"><a name="p10498111612210"></a><a name="p10498111612210"></a>636</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11522152317344"><a name="p11522152317344"></a><a name="p11522152317344"></a>multilabel_margin_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7522132343419"><a name="p7522132343419"></a><a name="p7522132343419"></a>multilabel_margin_loss_out_npu</p>
</td>
</tr>
<tr id="row97594394129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9499111619217"><a name="p9499111619217"></a><a name="p9499111619217"></a>637</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75222023103412"><a name="p75222023103412"></a><a name="p75222023103412"></a>multilabel_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p752242312349"><a name="p752242312349"></a><a name="p752242312349"></a>multilabel_margin_loss_npu</p>
</td>
</tr>
<tr id="row8759339171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049912165210"><a name="p1049912165210"></a><a name="p1049912165210"></a>638</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14522142318345"><a name="p14522142318345"></a><a name="p14522142318345"></a>multilabel_margin_loss_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1522112316348"><a name="p1522112316348"></a><a name="p1522112316348"></a>multilabel_margin_loss_forward_out_npu</p>
</td>
</tr>
<tr id="row6759193961213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p049915161629"><a name="p049915161629"></a><a name="p049915161629"></a>639</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4522152318344"><a name="p4522152318344"></a><a name="p4522152318344"></a>multilabel_margin_loss_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15522723123411"><a name="p15522723123411"></a><a name="p15522723123411"></a>multilabel_margin_loss_forward_npu</p>
</td>
</tr>
<tr id="row9759339191211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449951618214"><a name="p449951618214"></a><a name="p449951618214"></a>640</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6522623133411"><a name="p6522623133411"></a><a name="p6522623133411"></a>nll_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7522132393419"><a name="p7522132393419"></a><a name="p7522132393419"></a>nll_loss_out_npu</p>
</td>
</tr>
<tr id="row2759183916127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54999161522"><a name="p54999161522"></a><a name="p54999161522"></a>641</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4522122319348"><a name="p4522122319348"></a><a name="p4522122319348"></a>nll_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2522132393420"><a name="p2522132393420"></a><a name="p2522132393420"></a>nll_loss_npu</p>
</td>
</tr>
<tr id="row18760123951219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14999161024"><a name="p14999161024"></a><a name="p14999161024"></a>642</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10522182393415"><a name="p10522182393415"></a><a name="p10522182393415"></a>nll_loss_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1522123163418"><a name="p1522123163418"></a><a name="p1522123163418"></a>nll_loss_forward_out_npu</p>
</td>
</tr>
<tr id="row18760153913129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144991316028"><a name="p144991316028"></a><a name="p144991316028"></a>643</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95221623183410"><a name="p95221623183410"></a><a name="p95221623183410"></a>nll_loss_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10522182303420"><a name="p10522182303420"></a><a name="p10522182303420"></a>nll_loss_forward_npu</p>
</td>
</tr>
<tr id="row1776043931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249917161724"><a name="p249917161724"></a><a name="p249917161724"></a>644</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1052382363411"><a name="p1052382363411"></a><a name="p1052382363411"></a>nll_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13523723153413"><a name="p13523723153413"></a><a name="p13523723153413"></a>nll_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row9760113961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249917162213"><a name="p1249917162213"></a><a name="p1249917162213"></a>645</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1552392313346"><a name="p1552392313346"></a><a name="p1552392313346"></a>nll_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1752382310343"><a name="p1752382310343"></a><a name="p1752382310343"></a>nll_loss_backward_npu</p>
</td>
</tr>
<tr id="row6760103981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9499141618217"><a name="p9499141618217"></a><a name="p9499141618217"></a>646</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4523623133418"><a name="p4523623133418"></a><a name="p4523623133418"></a>nll_loss2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125231823133415"><a name="p125231823133415"></a><a name="p125231823133415"></a>nll_loss2d_out_npu</p>
</td>
</tr>
<tr id="row2760143971212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144990165211"><a name="p144990165211"></a><a name="p144990165211"></a>647</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7523123183419"><a name="p7523123183419"></a><a name="p7523123183419"></a>nll_loss2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p452318237347"><a name="p452318237347"></a><a name="p452318237347"></a>nll_loss2d_npu</p>
</td>
</tr>
<tr id="row9760133981218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1499161610211"><a name="p1499161610211"></a><a name="p1499161610211"></a>648</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p952311235341"><a name="p952311235341"></a><a name="p952311235341"></a>nll_loss2d_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165238235348"><a name="p165238235348"></a><a name="p165238235348"></a>nll_loss2d_forward_out_npu</p>
</td>
</tr>
<tr id="row1276053916122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p749918161328"><a name="p749918161328"></a><a name="p749918161328"></a>649</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185231623143419"><a name="p185231623143419"></a><a name="p185231623143419"></a>nll_loss2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19523123173420"><a name="p19523123173420"></a><a name="p19523123173420"></a>nll_loss2d_forward_npu</p>
</td>
</tr>
<tr id="row1076053919126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114990166220"><a name="p114990166220"></a><a name="p114990166220"></a>650</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p352362311341"><a name="p352362311341"></a><a name="p352362311341"></a>nll_loss2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p205231523143418"><a name="p205231523143418"></a><a name="p205231523143418"></a>nll_loss2d_backward_out_npu</p>
</td>
</tr>
<tr id="row07605390120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44991816520"><a name="p44991816520"></a><a name="p44991816520"></a>651</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1523152310346"><a name="p1523152310346"></a><a name="p1523152310346"></a>nll_loss2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9523112314344"><a name="p9523112314344"></a><a name="p9523112314344"></a>nll_loss2d_backward_npu</p>
</td>
</tr>
<tr id="row176093951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125001916829"><a name="p125001916829"></a><a name="p125001916829"></a>652</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p652315236345"><a name="p652315236345"></a><a name="p652315236345"></a>smooth_l1_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1752314238347"><a name="p1752314238347"></a><a name="p1752314238347"></a>smooth_l1_loss_out_npu</p>
</td>
</tr>
<tr id="row147611239111213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p165001616523"><a name="p165001616523"></a><a name="p165001616523"></a>653</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1952322353410"><a name="p1952322353410"></a><a name="p1952322353410"></a>smooth_l1_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p135231237349"><a name="p135231237349"></a><a name="p135231237349"></a>smooth_l1_loss_npu</p>
</td>
</tr>
<tr id="row0761839171218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1950014161729"><a name="p1950014161729"></a><a name="p1950014161729"></a>654</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1352302383413"><a name="p1352302383413"></a><a name="p1352302383413"></a>smooth_l1_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1952332315341"><a name="p1952332315341"></a><a name="p1952332315341"></a>smooth_l1_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row147617390125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p85005168219"><a name="p85005168219"></a><a name="p85005168219"></a>655</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95231623123417"><a name="p95231623123417"></a><a name="p95231623123417"></a>smooth_l1_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1952313233347"><a name="p1952313233347"></a><a name="p1952313233347"></a>smooth_l1_loss_backward_npu</p>
</td>
</tr>
<tr id="row18761113941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p950031613212"><a name="p950031613212"></a><a name="p950031613212"></a>656</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16523102310346"><a name="p16523102310346"></a><a name="p16523102310346"></a>soft_margin_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175231230348"><a name="p175231230348"></a><a name="p175231230348"></a>soft_margin_loss_out_npu</p>
</td>
</tr>
<tr id="row97611639151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1750012161220"><a name="p1750012161220"></a><a name="p1750012161220"></a>657</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195231823173417"><a name="p195231823173417"></a><a name="p195231823173417"></a>soft_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165231523203414"><a name="p165231523203414"></a><a name="p165231523203414"></a>soft_margin_loss_npu</p>
</td>
</tr>
<tr id="row4761173931218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05001816822"><a name="p05001816822"></a><a name="p05001816822"></a>658</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3523923103411"><a name="p3523923103411"></a><a name="p3523923103411"></a>soft_margin_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p65236233345"><a name="p65236233345"></a><a name="p65236233345"></a>soft_margin_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row9761173917125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2500916321"><a name="p2500916321"></a><a name="p2500916321"></a>659</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p652372312344"><a name="p652372312344"></a><a name="p652372312344"></a>soft_margin_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p352313231349"><a name="p352313231349"></a><a name="p352313231349"></a>soft_margin_loss_backward_npu</p>
</td>
</tr>
<tr id="row137611839121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p550041617212"><a name="p550041617212"></a><a name="p550041617212"></a>660</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1452422311349"><a name="p1452422311349"></a><a name="p1452422311349"></a>elu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1552415232348"><a name="p1552415232348"></a><a name="p1552415232348"></a>elu_out_npu</p>
</td>
</tr>
<tr id="row176153961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750071612214"><a name="p750071612214"></a><a name="p750071612214"></a>661</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14524323123419"><a name="p14524323123419"></a><a name="p14524323123419"></a>elu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15524202373419"><a name="p15524202373419"></a><a name="p15524202373419"></a>elu_npu</p>
</td>
</tr>
<tr id="row77611239121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7500101620218"><a name="p7500101620218"></a><a name="p7500101620218"></a>662</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1852462353412"><a name="p1852462353412"></a><a name="p1852462353412"></a>elu_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175241123103414"><a name="p175241123103414"></a><a name="p175241123103414"></a>elu_backward_out_npu</p>
</td>
</tr>
<tr id="row376123941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15005161023"><a name="p15005161023"></a><a name="p15005161023"></a>663</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16524423133416"><a name="p16524423133416"></a><a name="p16524423133416"></a>elu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17524112363414"><a name="p17524112363414"></a><a name="p17524112363414"></a>elu_backward_npu</p>
</td>
</tr>
<tr id="row1976173916120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p350020161425"><a name="p350020161425"></a><a name="p350020161425"></a>664</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2524162318341"><a name="p2524162318341"></a><a name="p2524162318341"></a>elu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17524122353418"><a name="p17524122353418"></a><a name="p17524122353418"></a>elu_npu_</p>
</td>
</tr>
<tr id="row11762339181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250017161027"><a name="p1250017161027"></a><a name="p1250017161027"></a>665</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155241923163416"><a name="p155241923163416"></a><a name="p155241923163416"></a>glu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4524112363410"><a name="p4524112363410"></a><a name="p4524112363410"></a>glu_out_npu</p>
</td>
</tr>
<tr id="row117623394125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135000161828"><a name="p135000161828"></a><a name="p135000161828"></a>666</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p152411238340"><a name="p152411238340"></a><a name="p152411238340"></a>glu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7524112323415"><a name="p7524112323415"></a><a name="p7524112323415"></a>glu_npu</p>
</td>
</tr>
<tr id="row57620396121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p55006166220"><a name="p55006166220"></a><a name="p55006166220"></a>667</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16524823153419"><a name="p16524823153419"></a><a name="p16524823153419"></a>glu_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85241232348"><a name="p85241232348"></a><a name="p85241232348"></a>glu_backward_out_npu</p>
</td>
</tr>
<tr id="row15762153919129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p850018161820"><a name="p850018161820"></a><a name="p850018161820"></a>668</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155241923123414"><a name="p155241923123414"></a><a name="p155241923123414"></a>glu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75255238343"><a name="p75255238343"></a><a name="p75255238343"></a>glu_backward_npu</p>
</td>
</tr>
<tr id="row0762639141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19500916222"><a name="p19500916222"></a><a name="p19500916222"></a>669</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8525132363414"><a name="p8525132363414"></a><a name="p8525132363414"></a>hardsigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15525023133411"><a name="p15525023133411"></a><a name="p15525023133411"></a>hardsigmoid_out_npu</p>
</td>
</tr>
<tr id="row7762239161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15001161211"><a name="p15001161211"></a><a name="p15001161211"></a>670</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p145251023123416"><a name="p145251023123416"></a><a name="p145251023123416"></a>hardsigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2525623103418"><a name="p2525623103418"></a><a name="p2525623103418"></a>hardsigmoid_npu</p>
</td>
</tr>
<tr id="row1776243931220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35003162219"><a name="p35003162219"></a><a name="p35003162219"></a>671</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1052582313341"><a name="p1052582313341"></a><a name="p1052582313341"></a>hardsigmoid_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p452562319349"><a name="p452562319349"></a><a name="p452562319349"></a>hardsigmoid_npu_</p>
</td>
</tr>
<tr id="row17621539171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450071618216"><a name="p1450071618216"></a><a name="p1450071618216"></a>672</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p652514234343"><a name="p652514234343"></a><a name="p652514234343"></a>hardsigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p652512393416"><a name="p652512393416"></a><a name="p652512393416"></a>hardsigmoid_backward_npu</p>
</td>
</tr>
<tr id="row3762123931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17501121613220"><a name="p17501121613220"></a><a name="p17501121613220"></a>673</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1752542313413"><a name="p1752542313413"></a><a name="p1752542313413"></a>hardtanh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5525172373419"><a name="p5525172373419"></a><a name="p5525172373419"></a>hardtanh_out_npu</p>
</td>
</tr>
<tr id="row1376293918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p205011116925"><a name="p205011116925"></a><a name="p205011116925"></a>674</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p152572313412"><a name="p152572313412"></a><a name="p152572313412"></a>hardtanh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25253232340"><a name="p25253232340"></a><a name="p25253232340"></a>hardtanh_npu</p>
</td>
</tr>
<tr id="row117621239181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35018161521"><a name="p35018161521"></a><a name="p35018161521"></a>675</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p452552312343"><a name="p452552312343"></a><a name="p452552312343"></a>hardtanh_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p852519239345"><a name="p852519239345"></a><a name="p852519239345"></a>hardtanh_backward_out_npu</p>
</td>
</tr>
<tr id="row77631939131212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p185015162219"><a name="p185015162219"></a><a name="p185015162219"></a>676</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p95253233343"><a name="p95253233343"></a><a name="p95253233343"></a>hardtanh_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19525182353417"><a name="p19525182353417"></a><a name="p19525182353417"></a>hardtanh_backward_npu</p>
</td>
</tr>
<tr id="row1763183919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45011216427"><a name="p45011216427"></a><a name="p45011216427"></a>677</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p152582319344"><a name="p152582319344"></a><a name="p152582319344"></a>hardtanh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85267230343"><a name="p85267230343"></a><a name="p85267230343"></a>hardtanh_npu_</p>
</td>
</tr>
<tr id="row11763113911215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7501216828"><a name="p7501216828"></a><a name="p7501216828"></a>678</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p125262237347"><a name="p125262237347"></a><a name="p125262237347"></a>leaky_relu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p252612313340"><a name="p252612313340"></a><a name="p252612313340"></a>leaky_relu_out_npu</p>
</td>
</tr>
<tr id="row177631039111214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750112163214"><a name="p750112163214"></a><a name="p750112163214"></a>679</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7526132314345"><a name="p7526132314345"></a><a name="p7526132314345"></a>leaky_relu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8526523143414"><a name="p8526523143414"></a><a name="p8526523143414"></a>leaky_relu_npu</p>
</td>
</tr>
<tr id="row16763539181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4501016221"><a name="p4501016221"></a><a name="p4501016221"></a>680</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175261523203410"><a name="p175261523203410"></a><a name="p175261523203410"></a>leaky_relu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p152615231340"><a name="p152615231340"></a><a name="p152615231340"></a>leaky_relu_backward_npu</p>
</td>
</tr>
<tr id="row17763939191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05011116527"><a name="p05011116527"></a><a name="p05011116527"></a>681</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8526323133416"><a name="p8526323133416"></a><a name="p8526323133416"></a>leaky_relu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p175267231341"><a name="p175267231341"></a><a name="p175267231341"></a>leaky_relu_npu_</p>
</td>
</tr>
<tr id="row27631039121220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8501171613213"><a name="p8501171613213"></a><a name="p8501171613213"></a>682</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11526123183416"><a name="p11526123183416"></a><a name="p11526123183416"></a>log_sigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165265237341"><a name="p165265237341"></a><a name="p165265237341"></a>log_sigmoid_out_npu</p>
</td>
</tr>
<tr id="row117631439111219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11501171616211"><a name="p11501171616211"></a><a name="p11501171616211"></a>683</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p152602313342"><a name="p152602313342"></a><a name="p152602313342"></a>log_sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2526172315349"><a name="p2526172315349"></a><a name="p2526172315349"></a>log_sigmoid_npu</p>
</td>
</tr>
<tr id="row167636392122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p105011716824"><a name="p105011716824"></a><a name="p105011716824"></a>684</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14526182310343"><a name="p14526182310343"></a><a name="p14526182310343"></a>log_sigmoid_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14526152319348"><a name="p14526152319348"></a><a name="p14526152319348"></a>log_sigmoid_forward_out_npu</p>
</td>
</tr>
<tr id="row77634398120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135018161020"><a name="p135018161020"></a><a name="p135018161020"></a>685</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9526202323418"><a name="p9526202323418"></a><a name="p9526202323418"></a>log_sigmoid_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p652612313349"><a name="p652612313349"></a><a name="p652612313349"></a>log_sigmoid_forward_npu</p>
</td>
</tr>
<tr id="row876373921213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p165015161219"><a name="p165015161219"></a><a name="p165015161219"></a>686</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p952618235345"><a name="p952618235345"></a><a name="p952618235345"></a>log_sigmoid_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1352672373418"><a name="p1352672373418"></a><a name="p1352672373418"></a>log_sigmoid_backward_out_npu</p>
</td>
</tr>
<tr id="row157641939171215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16501816824"><a name="p16501816824"></a><a name="p16501816824"></a>687</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1852692316342"><a name="p1852692316342"></a><a name="p1852692316342"></a>log_sigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p95261523163418"><a name="p95261523163418"></a><a name="p95261523163418"></a>log_sigmoid_backward_npu</p>
</td>
</tr>
<tr id="row676463913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25013163216"><a name="p25013163216"></a><a name="p25013163216"></a>688</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1052613239347"><a name="p1052613239347"></a><a name="p1052613239347"></a>rrelu_with_noise.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125268232348"><a name="p125268232348"></a><a name="p125268232348"></a>rrelu_with_noise_out_npu</p>
</td>
</tr>
<tr id="row576412392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p55021161522"><a name="p55021161522"></a><a name="p55021161522"></a>689</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1052612311349"><a name="p1052612311349"></a><a name="p1052612311349"></a>rrelu_with_noise</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1752642373412"><a name="p1752642373412"></a><a name="p1752642373412"></a>rrelu_with_noise_npu</p>
</td>
</tr>
<tr id="row0764539201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250219163212"><a name="p1250219163212"></a><a name="p1250219163212"></a>690</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2526323193414"><a name="p2526323193414"></a><a name="p2526323193414"></a>rrelu_with_noise_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3526122316340"><a name="p3526122316340"></a><a name="p3526122316340"></a>rrelu_with_noise_backward_npu</p>
</td>
</tr>
<tr id="row6764153914123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35027161327"><a name="p35027161327"></a><a name="p35027161327"></a>691</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19526142343415"><a name="p19526142343415"></a><a name="p19526142343415"></a>rrelu_with_noise_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3526122310344"><a name="p3526122310344"></a><a name="p3526122310344"></a>rrelu_with_noise_npu_</p>
</td>
</tr>
<tr id="row876453916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15021316426"><a name="p15021316426"></a><a name="p15021316426"></a>692</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p155268238345"><a name="p155268238345"></a><a name="p155268238345"></a>softplus.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16527132353419"><a name="p16527132353419"></a><a name="p16527132353419"></a>softplus_out_npu</p>
</td>
</tr>
<tr id="row47641139131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12502216929"><a name="p12502216929"></a><a name="p12502216929"></a>693</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9527172313418"><a name="p9527172313418"></a><a name="p9527172313418"></a>softplus</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6527423183418"><a name="p6527423183418"></a><a name="p6527423183418"></a>softplus_npu</p>
</td>
</tr>
<tr id="row16764039191218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17502316825"><a name="p17502316825"></a><a name="p17502316825"></a>694</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p152772343410"><a name="p152772343410"></a><a name="p152772343410"></a>softplus_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p152772312348"><a name="p152772312348"></a><a name="p152772312348"></a>softplus_backward_out_npu</p>
</td>
</tr>
<tr id="row776420399123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4502171611217"><a name="p4502171611217"></a><a name="p4502171611217"></a>695</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1452722383413"><a name="p1452722383413"></a><a name="p1452722383413"></a>softplus_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19527152383415"><a name="p19527152383415"></a><a name="p19527152383415"></a>softplus_backward_npu</p>
</td>
</tr>
<tr id="row167641939141216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12502716729"><a name="p12502716729"></a><a name="p12502716729"></a>696</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1552717233345"><a name="p1552717233345"></a><a name="p1552717233345"></a>softshrink.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4527182313349"><a name="p4527182313349"></a><a name="p4527182313349"></a>softshrink_out_npu</p>
</td>
</tr>
<tr id="row77645392124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250213166210"><a name="p1250213166210"></a><a name="p1250213166210"></a>697</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p752715234345"><a name="p752715234345"></a><a name="p752715234345"></a>softshrink</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20527122313346"><a name="p20527122313346"></a><a name="p20527122313346"></a>softshrink_npu</p>
</td>
</tr>
<tr id="row1576483991213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35021316429"><a name="p35021316429"></a><a name="p35021316429"></a>698</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1852712323419"><a name="p1852712323419"></a><a name="p1852712323419"></a>softshrink_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2527112353415"><a name="p2527112353415"></a><a name="p2527112353415"></a>softshrink_backward_out_npu</p>
</td>
</tr>
<tr id="row1576515398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350214169210"><a name="p1350214169210"></a><a name="p1350214169210"></a>699</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1252719230343"><a name="p1252719230343"></a><a name="p1252719230343"></a>softshrink_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15278231348"><a name="p15278231348"></a><a name="p15278231348"></a>softshrink_backward_npu</p>
</td>
</tr>
<tr id="row47651139101216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19502161614214"><a name="p19502161614214"></a><a name="p19502161614214"></a>700</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175271823123415"><a name="p175271823123415"></a><a name="p175271823123415"></a>adaptive_avg_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p135271223143420"><a name="p135271223143420"></a><a name="p135271223143420"></a>adaptive_avg_pool2d_out_npu</p>
</td>
</tr>
<tr id="row13765939101211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16502171620211"><a name="p16502171620211"></a><a name="p16502171620211"></a>701</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4527182343412"><a name="p4527182343412"></a><a name="p4527182343412"></a>adaptive_avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11527152311340"><a name="p11527152311340"></a><a name="p11527152311340"></a>adaptive_avg_pool2d_npu</p>
</td>
</tr>
<tr id="row1976518395127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450217169212"><a name="p1450217169212"></a><a name="p1450217169212"></a>702</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p452722315347"><a name="p452722315347"></a><a name="p452722315347"></a>_adaptive_avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6527523203412"><a name="p6527523203412"></a><a name="p6527523203412"></a>_adaptive_avg_pool2d_npu</p>
</td>
</tr>
<tr id="row1476511392129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15025163218"><a name="p15025163218"></a><a name="p15025163218"></a>703</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p052718232340"><a name="p052718232340"></a><a name="p052718232340"></a>_adaptive_avg_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1452722323413"><a name="p1452722323413"></a><a name="p1452722323413"></a>adaptive_avg_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row0765173961215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145029161423"><a name="p145029161423"></a><a name="p145029161423"></a>704</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18527723113420"><a name="p18527723113420"></a><a name="p18527723113420"></a>adaptive_avg_pool3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19527152315348"><a name="p19527152315348"></a><a name="p19527152315348"></a>adaptive_avg_pool3d_out_npu</p>
</td>
</tr>
<tr id="row0765163915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p65022161522"><a name="p65022161522"></a><a name="p65022161522"></a>705</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16527122333416"><a name="p16527122333416"></a><a name="p16527122333416"></a>adaptive_avg_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55270233344"><a name="p55270233344"></a><a name="p55270233344"></a>adaptive_avg_pool3d_npu</p>
</td>
</tr>
<tr id="row12765639121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125031016820"><a name="p125031016820"></a><a name="p125031016820"></a>706</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1352742323418"><a name="p1352742323418"></a><a name="p1352742323418"></a>adaptive_avg_pool3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17527152363410"><a name="p17527152363410"></a><a name="p17527152363410"></a>adaptive_avg_pool3d_backward_out_npu</p>
</td>
</tr>
<tr id="row1476513931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45036166220"><a name="p45036166220"></a><a name="p45036166220"></a>707</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2527202312343"><a name="p2527202312343"></a><a name="p2527202312343"></a>adaptive_avg_pool3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25271523183415"><a name="p25271523183415"></a><a name="p25271523183415"></a>adaptive_avg_pool3d_backward_npu</p>
</td>
</tr>
<tr id="row976583971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125032016329"><a name="p125032016329"></a><a name="p125032016329"></a>708</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5527123123417"><a name="p5527123123417"></a><a name="p5527123123417"></a>adaptive_max_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p752742373417"><a name="p752742373417"></a><a name="p752742373417"></a>adaptive_max_pool2d_out_npu</p>
</td>
</tr>
<tr id="row2076512399126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p105031516926"><a name="p105031516926"></a><a name="p105031516926"></a>709</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1652811231343"><a name="p1652811231343"></a><a name="p1652811231343"></a>adaptive_max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17528202311346"><a name="p17528202311346"></a><a name="p17528202311346"></a>adaptive_max_pool2d_npu</p>
</td>
</tr>
<tr id="row87660397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155038161329"><a name="p155038161329"></a><a name="p155038161329"></a>710</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1052802310342"><a name="p1052802310342"></a><a name="p1052802310342"></a>adaptive_max_pool2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1552882393413"><a name="p1552882393413"></a><a name="p1552882393413"></a>adaptive_max_pool2d_backward_out_npu</p>
</td>
</tr>
<tr id="row876610396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18503216729"><a name="p18503216729"></a><a name="p18503216729"></a>711</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2528142319348"><a name="p2528142319348"></a><a name="p2528142319348"></a>adaptive_max_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10528172363415"><a name="p10528172363415"></a><a name="p10528172363415"></a>adaptive_max_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row12766339201217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135035167217"><a name="p135035167217"></a><a name="p135035167217"></a>712</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1552819233344"><a name="p1552819233344"></a><a name="p1552819233344"></a>avg_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6528172312343"><a name="p6528172312343"></a><a name="p6528172312343"></a>avg_pool2d_out_npu</p>
</td>
</tr>
<tr id="row976693981213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19503201613220"><a name="p19503201613220"></a><a name="p19503201613220"></a>713</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12528182333413"><a name="p12528182333413"></a><a name="p12528182333413"></a>avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1952892314344"><a name="p1952892314344"></a><a name="p1952892314344"></a>avg_pool2d_npu</p>
</td>
</tr>
<tr id="row157661339191218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115032016329"><a name="p115032016329"></a><a name="p115032016329"></a>714</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1752802310348"><a name="p1752802310348"></a><a name="p1752802310348"></a>avg_pool2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p205282023133419"><a name="p205282023133419"></a><a name="p205282023133419"></a>avg_pool2d_backward_out_npu</p>
</td>
</tr>
<tr id="row076693919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16503816826"><a name="p16503816826"></a><a name="p16503816826"></a>715</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19528132313345"><a name="p19528132313345"></a><a name="p19528132313345"></a>avg_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p852815235344"><a name="p852815235344"></a><a name="p852815235344"></a>avg_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row1876633961215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145031316127"><a name="p145031316127"></a><a name="p145031316127"></a>716</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85282230344"><a name="p85282230344"></a><a name="p85282230344"></a>avg_pool3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p552862320346"><a name="p552862320346"></a><a name="p552862320346"></a>avg_pool3d_out_npu</p>
</td>
</tr>
<tr id="row9767103941216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25033165211"><a name="p25033165211"></a><a name="p25033165211"></a>717</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p252822316345"><a name="p252822316345"></a><a name="p252822316345"></a>avg_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15281423143419"><a name="p15281423143419"></a><a name="p15281423143419"></a>avg_pool3d_npu</p>
</td>
</tr>
<tr id="row167671939131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2503116426"><a name="p2503116426"></a><a name="p2503116426"></a>718</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15283235347"><a name="p15283235347"></a><a name="p15283235347"></a>avg_pool3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14528112319343"><a name="p14528112319343"></a><a name="p14528112319343"></a>avg_pool3d_backward_out_npu</p>
</td>
</tr>
<tr id="row1676714395121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6503316326"><a name="p6503316326"></a><a name="p6503316326"></a>719</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p165289230346"><a name="p165289230346"></a><a name="p165289230346"></a>avg_pool3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1528152319340"><a name="p1528152319340"></a><a name="p1528152319340"></a>avg_pool3d_backward_npu</p>
</td>
</tr>
<tr id="row18149195017234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p175032169219"><a name="p175032169219"></a><a name="p175032169219"></a>720</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12528182315340"><a name="p12528182315340"></a><a name="p12528182315340"></a>max_pool2d_with_indices.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1152812231348"><a name="p1152812231348"></a><a name="p1152812231348"></a>max_pool2d_with_indices_out_npu</p>
</td>
</tr>
<tr id="row1614985042313"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7503816528"><a name="p7503816528"></a><a name="p7503816528"></a>721</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1152811230342"><a name="p1152811230342"></a><a name="p1152811230342"></a>max_pool2d_with_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1252813230342"><a name="p1252813230342"></a><a name="p1252813230342"></a>max_pool2d_with_indices_npu</p>
</td>
</tr>
<tr id="row17149115012238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1550401614213"><a name="p1550401614213"></a><a name="p1550401614213"></a>722</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11528423133417"><a name="p11528423133417"></a><a name="p11528423133417"></a>max_pool2d_with_indices_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15528172313343"><a name="p15528172313343"></a><a name="p15528172313343"></a>max_pool2d_with_indices_backward_out_npu</p>
</td>
</tr>
<tr id="row614965016234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1750491617211"><a name="p1750491617211"></a><a name="p1750491617211"></a>723</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p052819238342"><a name="p052819238342"></a><a name="p052819238342"></a>max_pool2d_with_indices_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p252842311342"><a name="p252842311342"></a><a name="p252842311342"></a>max_pool2d_with_indices_backward_npu</p>
</td>
</tr>
<tr id="row8149155011235"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5504121617216"><a name="p5504121617216"></a><a name="p5504121617216"></a>724</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85284235345"><a name="p85284235345"></a><a name="p85284235345"></a>max_pool3d_with_indices.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7528142317342"><a name="p7528142317342"></a><a name="p7528142317342"></a>max_pool3d_with_indices_out_npu</p>
</td>
</tr>
<tr id="row914945052310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15048168211"><a name="p15048168211"></a><a name="p15048168211"></a>725</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p552902317349"><a name="p552902317349"></a><a name="p552902317349"></a>max_pool3d_with_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p152910237342"><a name="p152910237342"></a><a name="p152910237342"></a>max_pool3d_with_indices_npu</p>
</td>
</tr>
<tr id="row414875013236"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450413161926"><a name="p1450413161926"></a><a name="p1450413161926"></a>726</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14529152313344"><a name="p14529152313344"></a><a name="p14529152313344"></a>max_pool3d_with_indices_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145299232343"><a name="p145299232343"></a><a name="p145299232343"></a>max_pool3d_with_indices_backward_out_npu</p>
</td>
</tr>
<tr id="row181481650152310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250416161226"><a name="p250416161226"></a><a name="p250416161226"></a>727</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p052972317347"><a name="p052972317347"></a><a name="p052972317347"></a>max_pool3d_with_indices_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p852932373414"><a name="p852932373414"></a><a name="p852932373414"></a>max_pool3d_with_indices_backward_npu</p>
</td>
</tr>
<tr id="row914810500234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p175042161925"><a name="p175042161925"></a><a name="p175042161925"></a>728</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185291723183417"><a name="p185291723183417"></a><a name="p185291723183417"></a>max_unpool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p652902314349"><a name="p652902314349"></a><a name="p652902314349"></a>max_unpool2d_out_npu</p>
</td>
</tr>
<tr id="row51481550142317"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11504816824"><a name="p11504816824"></a><a name="p11504816824"></a>729</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15529112318349"><a name="p15529112318349"></a><a name="p15529112318349"></a>max_unpool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1052962318344"><a name="p1052962318344"></a><a name="p1052962318344"></a>max_unpool2d_npu</p>
</td>
</tr>
<tr id="row101481250142314"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250471613220"><a name="p250471613220"></a><a name="p250471613220"></a>730</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17529172310346"><a name="p17529172310346"></a><a name="p17529172310346"></a>max_unpool2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9529192315341"><a name="p9529192315341"></a><a name="p9529192315341"></a>max_unpool2d_backward_out_npu</p>
</td>
</tr>
<tr id="row91484505232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1950420161621"><a name="p1950420161621"></a><a name="p1950420161621"></a>731</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0529182311348"><a name="p0529182311348"></a><a name="p0529182311348"></a>max_unpool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p752917233341"><a name="p752917233341"></a><a name="p752917233341"></a>max_unpool2d_backward_npu</p>
</td>
</tr>
<tr id="row2148155019234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p95049161126"><a name="p95049161126"></a><a name="p95049161126"></a>732</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p552915231344"><a name="p552915231344"></a><a name="p552915231344"></a>max_unpool3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1652962363412"><a name="p1652962363412"></a><a name="p1652962363412"></a>max_unpool3d_out_npu</p>
</td>
</tr>
<tr id="row151481250172312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45043161823"><a name="p45043161823"></a><a name="p45043161823"></a>733</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35291923103410"><a name="p35291923103410"></a><a name="p35291923103410"></a>max_unpool3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p55291723153414"><a name="p55291723153414"></a><a name="p55291723153414"></a>max_unpool3d_npu</p>
</td>
</tr>
<tr id="row214811500239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155048163216"><a name="p155048163216"></a><a name="p155048163216"></a>734</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p452911239347"><a name="p452911239347"></a><a name="p452911239347"></a>max_unpool3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2052952319345"><a name="p2052952319345"></a><a name="p2052952319345"></a>max_unpool3d_backward_out_npu</p>
</td>
</tr>
<tr id="row9148450142310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p450431613211"><a name="p450431613211"></a><a name="p450431613211"></a>735</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85291233344"><a name="p85291233344"></a><a name="p85291233344"></a>max_unpool3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1152917237347"><a name="p1152917237347"></a><a name="p1152917237347"></a>max_unpool3d_backward_npu</p>
</td>
</tr>
<tr id="row914819503239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115041816427"><a name="p115041816427"></a><a name="p115041816427"></a>736</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105291123193419"><a name="p105291123193419"></a><a name="p105291123193419"></a>reflection_pad2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1529192333412"><a name="p1529192333412"></a><a name="p1529192333412"></a>reflection_pad2d_out_npu</p>
</td>
</tr>
<tr id="row1514765042314"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p450510161726"><a name="p450510161726"></a><a name="p450510161726"></a>737</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13529202303412"><a name="p13529202303412"></a><a name="p13529202303412"></a>reflection_pad2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p35291123173416"><a name="p35291123173416"></a><a name="p35291123173416"></a>reflection_pad2d_npu</p>
</td>
</tr>
<tr id="row10147145019234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1850515161326"><a name="p1850515161326"></a><a name="p1850515161326"></a>738</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p952916239343"><a name="p952916239343"></a><a name="p952916239343"></a>reflection_pad2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2052912323417"><a name="p2052912323417"></a><a name="p2052912323417"></a>reflection_pad2d_backward_out_npu</p>
</td>
</tr>
<tr id="row131471350162312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14505916423"><a name="p14505916423"></a><a name="p14505916423"></a>739</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p052920238341"><a name="p052920238341"></a><a name="p052920238341"></a>reflection_pad2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p252962317344"><a name="p252962317344"></a><a name="p252962317344"></a>reflection_pad2d_backward_npu</p>
</td>
</tr>
<tr id="row514745013232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p950512161228"><a name="p950512161228"></a><a name="p950512161228"></a>740</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6529172313413"><a name="p6529172313413"></a><a name="p6529172313413"></a>replication_pad2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145291023173417"><a name="p145291023173417"></a><a name="p145291023173417"></a>replication_pad2d_out_npu</p>
</td>
</tr>
<tr id="row16147115062311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25054161225"><a name="p25054161225"></a><a name="p25054161225"></a>741</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p752910235344"><a name="p752910235344"></a><a name="p752910235344"></a>replication_pad2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7530823153414"><a name="p7530823153414"></a><a name="p7530823153414"></a>replication_pad2d_npu</p>
</td>
</tr>
<tr id="row1514755018239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p150501616218"><a name="p150501616218"></a><a name="p150501616218"></a>742</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3530182383410"><a name="p3530182383410"></a><a name="p3530182383410"></a>replication_pad2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15301523143410"><a name="p15301523143410"></a><a name="p15301523143410"></a>replication_pad2d_backward_out_npu</p>
</td>
</tr>
<tr id="row914712503239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p185052161625"><a name="p185052161625"></a><a name="p185052161625"></a>743</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2530182383416"><a name="p2530182383416"></a><a name="p2530182383416"></a>replication_pad2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p145304233347"><a name="p145304233347"></a><a name="p145304233347"></a>replication_pad2d_backward_npu</p>
</td>
</tr>
<tr id="row151473505232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16505716022"><a name="p16505716022"></a><a name="p16505716022"></a>744</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185301623203411"><a name="p185301623203411"></a><a name="p185301623203411"></a>upsample_linear1d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1253019232343"><a name="p1253019232343"></a><a name="p1253019232343"></a>upsample_linear1d_out_npu</p>
</td>
</tr>
<tr id="row171471350182312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250518161825"><a name="p250518161825"></a><a name="p250518161825"></a>745</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p653018232340"><a name="p653018232340"></a><a name="p653018232340"></a>upsample_linear1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11530182353419"><a name="p11530182353419"></a><a name="p11530182353419"></a>upsample_linear1d_npu</p>
</td>
</tr>
<tr id="row12147150142310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3505151619219"><a name="p3505151619219"></a><a name="p3505151619219"></a>746</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9530162318342"><a name="p9530162318342"></a><a name="p9530162318342"></a>upsample_linear1d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195301423123412"><a name="p195301423123412"></a><a name="p195301423123412"></a>upsample_linear1d_backward_npu</p>
</td>
</tr>
<tr id="row101472050152318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350511163213"><a name="p1350511163213"></a><a name="p1350511163213"></a>747</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p185308231344"><a name="p185308231344"></a><a name="p185308231344"></a>upsample_bilinear2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p353010235347"><a name="p353010235347"></a><a name="p353010235347"></a>upsample_bilinear2d_out_npu</p>
</td>
</tr>
<tr id="row31463506231"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350517161214"><a name="p1350517161214"></a><a name="p1350517161214"></a>748</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19530423183418"><a name="p19530423183418"></a><a name="p19530423183418"></a>upsample_bilinear2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10530423133416"><a name="p10530423133416"></a><a name="p10530423133416"></a>upsample_bilinear2d_npu</p>
</td>
</tr>
<tr id="row1814612508238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5505416525"><a name="p5505416525"></a><a name="p5505416525"></a>749</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2053072363415"><a name="p2053072363415"></a><a name="p2053072363415"></a>upsample_bilinear2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1153072318346"><a name="p1153072318346"></a><a name="p1153072318346"></a>upsample_bilinear2d_backward_out_npu</p>
</td>
</tr>
<tr id="row714614509238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9505201610211"><a name="p9505201610211"></a><a name="p9505201610211"></a>750</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p135301823173418"><a name="p135301823173418"></a><a name="p135301823173418"></a>upsample_bilinear2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85301623203415"><a name="p85301623203415"></a><a name="p85301623203415"></a>upsample_bilinear2d_backward_npu</p>
</td>
</tr>
<tr id="row1714605042318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9505171614218"><a name="p9505171614218"></a><a name="p9505171614218"></a>751</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35301123123410"><a name="p35301123123410"></a><a name="p35301123123410"></a>upsample_bicubic2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9530122316348"><a name="p9530122316348"></a><a name="p9530122316348"></a>upsample_bicubic2d_out_npu</p>
</td>
</tr>
<tr id="row111461750132318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125051161024"><a name="p125051161024"></a><a name="p125051161024"></a>752</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195308233346"><a name="p195308233346"></a><a name="p195308233346"></a>upsample_bicubic2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p353014236347"><a name="p353014236347"></a><a name="p353014236347"></a>upsample_bicubic2d_npu</p>
</td>
</tr>
<tr id="row7146185018238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250518161620"><a name="p250518161620"></a><a name="p250518161620"></a>753</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3530123153418"><a name="p3530123153418"></a><a name="p3530123153418"></a>upsample_bicubic2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p953082333419"><a name="p953082333419"></a><a name="p953082333419"></a>upsample_bicubic2d_backward_out_npu</p>
</td>
</tr>
<tr id="row1514675082318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11505191612218"><a name="p11505191612218"></a><a name="p11505191612218"></a>754</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1453092312346"><a name="p1453092312346"></a><a name="p1453092312346"></a>upsample_bicubic2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p353010239346"><a name="p353010239346"></a><a name="p353010239346"></a>upsample_bicubic2d_backward_npu</p>
</td>
</tr>
<tr id="row1814645062311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19505131610215"><a name="p19505131610215"></a><a name="p19505131610215"></a>755</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5530123163413"><a name="p5530123163413"></a><a name="p5530123163413"></a>upsample_trilinear3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17530112316349"><a name="p17530112316349"></a><a name="p17530112316349"></a>upsample_trilinear3d_out_npu</p>
</td>
</tr>
<tr id="row12146135072311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19505216728"><a name="p19505216728"></a><a name="p19505216728"></a>756</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105301723133411"><a name="p105301723133411"></a><a name="p105301723133411"></a>upsample_trilinear3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14530142353412"><a name="p14530142353412"></a><a name="p14530142353412"></a>upsample_trilinear3d_npu</p>
</td>
</tr>
<tr id="row1214625011237"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750511614212"><a name="p750511614212"></a><a name="p750511614212"></a>757</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11530923163414"><a name="p11530923163414"></a><a name="p11530923163414"></a>upsample_trilinear3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17530623103418"><a name="p17530623103418"></a><a name="p17530623103418"></a>upsample_trilinear3d_backward_out_npu</p>
</td>
</tr>
<tr id="row14146155022318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7505101616219"><a name="p7505101616219"></a><a name="p7505101616219"></a>758</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16531132323419"><a name="p16531132323419"></a><a name="p16531132323419"></a>upsample_trilinear3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1453113238343"><a name="p1453113238343"></a><a name="p1453113238343"></a>upsample_trilinear3d_backward_npu</p>
</td>
</tr>
<tr id="row12145250202315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5506141611210"><a name="p5506141611210"></a><a name="p5506141611210"></a>759</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1053112313419"><a name="p1053112313419"></a><a name="p1053112313419"></a>upsample_nearest1d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10531182353417"><a name="p10531182353417"></a><a name="p10531182353417"></a>upsample_nearest1d_out_npu</p>
</td>
</tr>
<tr id="row19145125011236"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p85061716126"><a name="p85061716126"></a><a name="p85061716126"></a>760</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195311230345"><a name="p195311230345"></a><a name="p195311230345"></a>upsample_nearest1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1853142315345"><a name="p1853142315345"></a><a name="p1853142315345"></a>upsample_nearest1d_npu</p>
</td>
</tr>
<tr id="row16145205013238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20506316325"><a name="p20506316325"></a><a name="p20506316325"></a>761</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16531182313341"><a name="p16531182313341"></a><a name="p16531182313341"></a>upsample_nearest1d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165311023133413"><a name="p165311023133413"></a><a name="p165311023133413"></a>upsample_nearest1d_backward_out_npu</p>
</td>
</tr>
<tr id="row1914555052319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155062016921"><a name="p155062016921"></a><a name="p155062016921"></a>762</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12531172363418"><a name="p12531172363418"></a><a name="p12531172363418"></a>upsample_nearest1d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1353192310340"><a name="p1353192310340"></a><a name="p1353192310340"></a>upsample_nearest1d_backward_npu</p>
</td>
</tr>
<tr id="row114511508237"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12506316228"><a name="p12506316228"></a><a name="p12506316228"></a>763</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p25311823123413"><a name="p25311823123413"></a><a name="p25311823123413"></a>upsample_nearest2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p185313234346"><a name="p185313234346"></a><a name="p185313234346"></a>upsample_nearest2d_out_npu</p>
</td>
</tr>
<tr id="row71456502232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05061161223"><a name="p05061161223"></a><a name="p05061161223"></a>764</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8531172353415"><a name="p8531172353415"></a><a name="p8531172353415"></a>upsample_nearest2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1553113233341"><a name="p1553113233341"></a><a name="p1553113233341"></a>upsample_nearest2d_npu</p>
</td>
</tr>
<tr id="row11145115062319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1150651618213"><a name="p1150651618213"></a><a name="p1150651618213"></a>765</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3531182310340"><a name="p3531182310340"></a><a name="p3531182310340"></a>upsample_nearest2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12531192303412"><a name="p12531192303412"></a><a name="p12531192303412"></a>upsample_nearest2d_backward_out_npu</p>
</td>
</tr>
<tr id="row61451350172318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10506151611212"><a name="p10506151611212"></a><a name="p10506151611212"></a>766</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14531172373420"><a name="p14531172373420"></a><a name="p14531172373420"></a>upsample_nearest2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85310235344"><a name="p85310235344"></a><a name="p85310235344"></a>upsample_nearest2d_backward_npu</p>
</td>
</tr>
<tr id="row10145115042317"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p150612162028"><a name="p150612162028"></a><a name="p150612162028"></a>767</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1531152303413"><a name="p1531152303413"></a><a name="p1531152303413"></a>upsample_nearest3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165311223183411"><a name="p165311223183411"></a><a name="p165311223183411"></a>upsample_nearest3d_out_npu</p>
</td>
</tr>
<tr id="row19145165022315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125067162029"><a name="p125067162029"></a><a name="p125067162029"></a>768</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2053142312345"><a name="p2053142312345"></a><a name="p2053142312345"></a>upsample_nearest3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1253112235342"><a name="p1253112235342"></a><a name="p1253112235342"></a>upsample_nearest3d_npu</p>
</td>
</tr>
<tr id="row1314495014234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19506816928"><a name="p19506816928"></a><a name="p19506816928"></a>769</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18531172315341"><a name="p18531172315341"></a><a name="p18531172315341"></a>upsample_nearest3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13531723203418"><a name="p13531723203418"></a><a name="p13531723203418"></a>upsample_nearest3d_backward_out_npu</p>
</td>
</tr>
<tr id="row4144750152311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2506616229"><a name="p2506616229"></a><a name="p2506616229"></a>770</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p253132393419"><a name="p253132393419"></a><a name="p253132393419"></a>upsample_nearest3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4531323163416"><a name="p4531323163416"></a><a name="p4531323163416"></a>upsample_nearest3d_backward_npu</p>
</td>
</tr>
<tr id="row71441550172312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2506141618216"><a name="p2506141618216"></a><a name="p2506141618216"></a>771</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1753192320347"><a name="p1753192320347"></a><a name="p1753192320347"></a>sigmoid_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1253152313343"><a name="p1253152313343"></a><a name="p1253152313343"></a>sigmoid_backward_out_npu</p>
</td>
</tr>
<tr id="row1714475010231"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1550616163213"><a name="p1550616163213"></a><a name="p1550616163213"></a>772</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175311123153414"><a name="p175311123153414"></a><a name="p175311123153414"></a>sigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p153112233346"><a name="p153112233346"></a><a name="p153112233346"></a>sigmoid_backward_npu</p>
</td>
</tr>
<tr id="row514435052316"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p95064161821"><a name="p95064161821"></a><a name="p95064161821"></a>773</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1253115232346"><a name="p1253115232346"></a><a name="p1253115232346"></a>tanh_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1653112231345"><a name="p1653112231345"></a><a name="p1653112231345"></a>tanh_backward_out_npu</p>
</td>
</tr>
<tr id="row1514410509239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p185073162220"><a name="p185073162220"></a><a name="p185073162220"></a>774</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p115311523143418"><a name="p115311523143418"></a><a name="p115311523143418"></a>tanh_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14532123153412"><a name="p14532123153412"></a><a name="p14532123153412"></a>tanh_backward_npu</p>
</td>
</tr>
<tr id="row161441950202312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7507181619213"><a name="p7507181619213"></a><a name="p7507181619213"></a>775</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p165321123113413"><a name="p165321123113413"></a><a name="p165321123113413"></a>slow_conv_transpose2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p753232310348"><a name="p753232310348"></a><a name="p753232310348"></a>slow_conv_transpose2d_out_npu</p>
</td>
</tr>
<tr id="row201444501230"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p650712164215"><a name="p650712164215"></a><a name="p650712164215"></a>776</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1253213234341"><a name="p1253213234341"></a><a name="p1253213234341"></a>slow_conv_transpose2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p25321223153414"><a name="p25321223153414"></a><a name="p25321223153414"></a>slow_conv_transpose2d_npu</p>
</td>
</tr>
<tr id="row111441350162315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115073162027"><a name="p115073162027"></a><a name="p115073162027"></a>777</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p105321523113410"><a name="p105321523113410"></a><a name="p105321523113410"></a>slow_conv_transpose2d_backward.grad_output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7532823113418"><a name="p7532823113418"></a><a name="p7532823113418"></a>slow_conv_transpose2d_backward_out_npu</p>
</td>
</tr>
<tr id="row1914495012313"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p75072167211"><a name="p75072167211"></a><a name="p75072167211"></a>778</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14532142319344"><a name="p14532142319344"></a><a name="p14532142319344"></a>slow_conv_transpose2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17532152373413"><a name="p17532152373413"></a><a name="p17532152373413"></a>slow_conv_transpose2d_backward_npu</p>
</td>
</tr>
<tr id="row91431350182319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115077161821"><a name="p115077161821"></a><a name="p115077161821"></a>779</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3532152310342"><a name="p3532152310342"></a><a name="p3532152310342"></a>thnn_conv2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7532723143414"><a name="p7532723143414"></a><a name="p7532723143414"></a>thnn_conv2d_out_npu</p>
</td>
</tr>
<tr id="row16143150132315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p55075164215"><a name="p55075164215"></a><a name="p55075164215"></a>780</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0532112393410"><a name="p0532112393410"></a><a name="p0532112393410"></a>thnn_conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2053213235348"><a name="p2053213235348"></a><a name="p2053213235348"></a>thnn_conv2d_npu</p>
</td>
</tr>
<tr id="row840154544916"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1341154518498"><a name="p1341154518498"></a><a name="p1341154518498"></a>781</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7532122314349"><a name="p7532122314349"></a><a name="p7532122314349"></a>thnn_conv2d_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9532172373410"><a name="p9532172373410"></a><a name="p9532172373410"></a>thnn_conv2d_forward_out_npu</p>
</td>
</tr>
<tr id="row1279874816495"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p979824820499"><a name="p979824820499"></a><a name="p979824820499"></a>782</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6532192316349"><a name="p6532192316349"></a><a name="p6532192316349"></a>thnn_conv2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14532423133415"><a name="p14532423133415"></a><a name="p14532423133415"></a>thnn_conv2d_forward_npu</p>
</td>
</tr>
<tr id="row78031655104915"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18031855144911"><a name="p18031855144911"></a><a name="p18031855144911"></a>783</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7532723173417"><a name="p7532723173417"></a><a name="p7532723173417"></a>thnn_conv2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2053272323416"><a name="p2053272323416"></a><a name="p2053272323416"></a>thnn_conv2d_backward_npu</p>
</td>
</tr>
<tr id="row19157115292316"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p161574523236"><a name="p161574523236"></a><a name="p161574523236"></a>784</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p45321723183411"><a name="p45321723183411"></a><a name="p45321723183411"></a>thnn_conv_depthwise2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p195321923143417"><a name="p195321923143417"></a><a name="p195321923143417"></a>thnn_conv_depthwise2d_out_npu</p>
</td>
</tr>
<tr id="row72331655162312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p142331855182311"><a name="p142331855182311"></a><a name="p142331855182311"></a>785</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5532172363415"><a name="p5532172363415"></a><a name="p5532172363415"></a>thnn_conv_depthwise2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p753242319346"><a name="p753242319346"></a><a name="p753242319346"></a>thnn_conv_depthwise2d_npu</p>
</td>
</tr>
<tr id="row644295816233"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p644245813239"><a name="p644245813239"></a><a name="p644245813239"></a>786</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p953211238346"><a name="p953211238346"></a><a name="p953211238346"></a>thnn_conv_depthwise2d_forward.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15532923133417"><a name="p15532923133417"></a><a name="p15532923133417"></a>thnn_conv_depthwise2d_forward_out_npu</p>
</td>
</tr>
<tr id="row1226016262418"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p02603212245"><a name="p02603212245"></a><a name="p02603212245"></a>787</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17532323193411"><a name="p17532323193411"></a><a name="p17532323193411"></a>thnn_conv_depthwise2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p125321723103415"><a name="p125321723103415"></a><a name="p125321723103415"></a>thnn_conv_depthwise2d_forward_npu</p>
</td>
</tr>
<tr id="row78092782413"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5300659368"><a name="p5300659368"></a><a name="p5300659368"></a>788</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p35321623193410"><a name="p35321623193410"></a><a name="p35321623193410"></a>thnn_conv_depthwise2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p115321223153416"><a name="p115321223153416"></a><a name="p115321223153416"></a>thnn_conv_depthwise2d_backward_out_npu</p>
</td>
</tr>
<tr id="row18700124417324"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p73001056361"><a name="p73001056361"></a><a name="p73001056361"></a>789</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195321823203418"><a name="p195321823203418"></a><a name="p195321823203418"></a>thnn_conv_depthwise2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1253219239342"><a name="p1253219239342"></a><a name="p1253219239342"></a>thnn_conv_depthwise2d_backward_npu</p>
</td>
</tr>
<tr id="row106701253163210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15300195193612"><a name="p15300195193612"></a><a name="p15300195193612"></a>790</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13532142363411"><a name="p13532142363411"></a><a name="p13532142363411"></a>slow_conv3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p165329233349"><a name="p165329233349"></a><a name="p165329233349"></a>slow_conv3d_out_npu</p>
</td>
</tr>
<tr id="row9277023123319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p230017553617"><a name="p230017553617"></a><a name="p230017553617"></a>791</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175331323153415"><a name="p175331323153415"></a><a name="p175331323153415"></a>slow_conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18533423113412"><a name="p18533423113412"></a><a name="p18533423113412"></a>slow_conv3d_npu</p>
</td>
</tr>
<tr id="row1445232011336"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1930025143617"><a name="p1930025143617"></a><a name="p1930025143617"></a>792</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11533192383410"><a name="p11533192383410"></a><a name="p11533192383410"></a>slow_conv3d_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p253316239349"><a name="p253316239349"></a><a name="p253316239349"></a>slow_conv3d_forward_out_npu</p>
</td>
</tr>
<tr id="row1266291820335"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p130012517362"><a name="p130012517362"></a><a name="p130012517362"></a>793</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7533152316347"><a name="p7533152316347"></a><a name="p7533152316347"></a>slow_conv3d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1753342383414"><a name="p1753342383414"></a><a name="p1753342383414"></a>slow_conv3d_forward_npu</p>
</td>
</tr>
<tr id="row171247167338"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p73001758369"><a name="p73001758369"></a><a name="p73001758369"></a>794</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1853372316346"><a name="p1853372316346"></a><a name="p1853372316346"></a>slow_conv_dilated2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1453312232345"><a name="p1453312232345"></a><a name="p1453312232345"></a>slow_conv_dilated2d_npu</p>
</td>
</tr>
<tr id="row561528193318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13300355369"><a name="p13300355369"></a><a name="p13300355369"></a>795</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1653302353412"><a name="p1653302353412"></a><a name="p1653302353412"></a>slow_conv_dilated2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18533523123413"><a name="p18533523123413"></a><a name="p18533523123413"></a>slow_conv_dilated2d_backward_npu</p>
</td>
</tr>
<tr id="row1951526123316"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p173001518364"><a name="p173001518364"></a><a name="p173001518364"></a>796</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p135339233349"><a name="p135339233349"></a><a name="p135339233349"></a>col2im.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p75331323193419"><a name="p75331323193419"></a><a name="p75331323193419"></a>im2col_backward_out_npu</p>
</td>
</tr>
<tr id="row178121744334"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p33001510368"><a name="p33001510368"></a><a name="p33001510368"></a>797</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p175330237344"><a name="p175330237344"></a><a name="p175330237344"></a>col2im</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1353382343412"><a name="p1353382343412"></a><a name="p1353382343412"></a>im2col_backward_npu</p>
</td>
</tr>
<tr id="row9386164117336"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17301195193610"><a name="p17301195193610"></a><a name="p17301195193610"></a>798</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p85337234343"><a name="p85337234343"></a><a name="p85337234343"></a>col2im_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p85331423113415"><a name="p85331423113415"></a><a name="p85331423113415"></a>im2col_out_npu</p>
</td>
</tr>
<tr id="row354842123319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1301658361"><a name="p1301658361"></a><a name="p1301658361"></a>799</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p75331823123417"><a name="p75331823123417"></a><a name="p75331823123417"></a>col2im_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18533112312342"><a name="p18533112312342"></a><a name="p18533112312342"></a>im2col_npu</p>
</td>
</tr>
<tr id="row8627905339"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p63011257368"><a name="p63011257368"></a><a name="p63011257368"></a>800</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1453312310341"><a name="p1453312310341"></a><a name="p1453312310341"></a>im2col.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p653372318343"><a name="p653372318343"></a><a name="p653372318343"></a>im2col_out_npu</p>
</td>
</tr>
<tr id="row728673916334"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11301115113613"><a name="p11301115113613"></a><a name="p11301115113613"></a>801</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1053314238349"><a name="p1053314238349"></a><a name="p1053314238349"></a>im2col</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1533523123418"><a name="p1533523123418"></a><a name="p1533523123418"></a>im2col_npu</p>
</td>
</tr>
<tr id="row643665843218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p03017563615"><a name="p03017563615"></a><a name="p03017563615"></a>802</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p153352315344"><a name="p153352315344"></a><a name="p153352315344"></a>im2col_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14533122319346"><a name="p14533122319346"></a><a name="p14533122319346"></a>im2col_backward_out_npu</p>
</td>
</tr>
<tr id="row867745613216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p230114519363"><a name="p230114519363"></a><a name="p230114519363"></a>803</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1953392317346"><a name="p1953392317346"></a><a name="p1953392317346"></a>im2col_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1953313232341"><a name="p1953313232341"></a><a name="p1953313232341"></a>im2col_backward_npu</p>
</td>
</tr>
<tr id="row076614919326"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19766164914322"><a name="p19766164914322"></a><a name="p19766164914322"></a>804</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p953362319348"><a name="p953362319348"></a><a name="p953362319348"></a>isfinite</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9533152318349"><a name="p9533152318349"></a><a name="p9533152318349"></a>isfinite_npu</p>
</td>
</tr>
</tbody>
</table>

<h2 id="PyTorch昇腾自定义算子md">PyTorch昇腾自定义算子</h2>

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

