# PyTorch适配算子清单
-   [PyTorch原生算子与昇腾算子对应表](#PyTorch原生算子与昇腾算子对应表)
-   [PyTorch昇腾自定义算子](#PyTorch昇腾自定义算子)
<h2 id="PyTorch原生算子与昇腾算子对应表">PyTorch原生算子与昇腾算子对应表</h2>

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
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0902231815"><a name="p0902231815"></a><a name="p0902231815"></a>dropout</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p890172318116"><a name="p890172318116"></a><a name="p890172318116"></a>dropout_npu</p>
</td>
</tr>
<tr id="row469519391121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1458131614214"><a name="p1458131614214"></a><a name="p1458131614214"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p159082317116"><a name="p159082317116"></a><a name="p159082317116"></a>dropout_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p390172310118"><a name="p390172310118"></a><a name="p390172310118"></a>dropout_npu_</p>
</td>
</tr>
<tr id="row156952394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1145861612218"><a name="p1145861612218"></a><a name="p1145861612218"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p59012314115"><a name="p59012314115"></a><a name="p59012314115"></a>abs</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p890323616"><a name="p890323616"></a><a name="p890323616"></a>abs_npu</p>
</td>
</tr>
<tr id="row17695739101215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134581516024"><a name="p134581516024"></a><a name="p134581516024"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p690142316120"><a name="p690142316120"></a><a name="p690142316120"></a>abs_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p99092318114"><a name="p99092318114"></a><a name="p99092318114"></a>abs_npu_</p>
</td>
</tr>
<tr id="row569517398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1945819162213"><a name="p1945819162213"></a><a name="p1945819162213"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p179052311115"><a name="p179052311115"></a><a name="p179052311115"></a>abs.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6907232019"><a name="p6907232019"></a><a name="p6907232019"></a>abs_out_npu</p>
</td>
</tr>
<tr id="row6695123941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9458816327"><a name="p9458816327"></a><a name="p9458816327"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3901023317"><a name="p3901023317"></a><a name="p3901023317"></a>acos</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p79014231018"><a name="p79014231018"></a><a name="p79014231018"></a>acos_npu</p>
</td>
</tr>
<tr id="row869593910122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124586161729"><a name="p124586161729"></a><a name="p124586161729"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17901823913"><a name="p17901823913"></a><a name="p17901823913"></a>acos_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0906232115"><a name="p0906232115"></a><a name="p0906232115"></a>acos_npu_</p>
</td>
</tr>
<tr id="row16695239121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p745881610211"><a name="p745881610211"></a><a name="p745881610211"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10907231812"><a name="p10907231812"></a><a name="p10907231812"></a>acos.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p990142318111"><a name="p990142318111"></a><a name="p990142318111"></a>acos_out_npu</p>
</td>
</tr>
<tr id="row18696133961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14458161613213"><a name="p14458161613213"></a><a name="p14458161613213"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p890423211"><a name="p890423211"></a><a name="p890423211"></a>adaptive_avg_pool1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18901423814"><a name="p18901423814"></a><a name="p18901423814"></a>adaptive_avg_pool1d_npu</p>
</td>
</tr>
<tr id="row1769693961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p345851614215"><a name="p345851614215"></a><a name="p345851614215"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p890132312118"><a name="p890132312118"></a><a name="p890132312118"></a>add.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p49018238118"><a name="p49018238118"></a><a name="p49018238118"></a>add_npu</p>
</td>
</tr>
<tr id="row1869623951214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1945814161127"><a name="p1945814161127"></a><a name="p1945814161127"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15902239113"><a name="p15902239113"></a><a name="p15902239113"></a>add_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6901231418"><a name="p6901231418"></a><a name="p6901231418"></a>add_npu_</p>
</td>
</tr>
<tr id="row16961439181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13458916224"><a name="p13458916224"></a><a name="p13458916224"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19012231814"><a name="p19012231814"></a><a name="p19012231814"></a>add.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13908239114"><a name="p13908239114"></a><a name="p13908239114"></a>add_out_npu</p>
</td>
</tr>
<tr id="row10696133931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6458116525"><a name="p6458116525"></a><a name="p6458116525"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12919238113"><a name="p12919238113"></a><a name="p12919238113"></a>add.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p39172319115"><a name="p39172319115"></a><a name="p39172319115"></a>add_npu</p>
</td>
</tr>
<tr id="row6696143991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1345814161122"><a name="p1345814161122"></a><a name="p1345814161122"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13910231319"><a name="p13910231319"></a><a name="p13910231319"></a>add_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1591122318113"><a name="p1591122318113"></a><a name="p1591122318113"></a>add_npu_</p>
</td>
</tr>
<tr id="row1969613901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04591116925"><a name="p04591116925"></a><a name="p04591116925"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p209113231814"><a name="p209113231814"></a><a name="p209113231814"></a>addmv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8917231017"><a name="p8917231017"></a><a name="p8917231017"></a>addmv_npu</p>
</td>
</tr>
<tr id="row1169614395122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144592161423"><a name="p144592161423"></a><a name="p144592161423"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10911231117"><a name="p10911231117"></a><a name="p10911231117"></a>addmv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1791112314112"><a name="p1791112314112"></a><a name="p1791112314112"></a>addmv_npu_</p>
</td>
</tr>
<tr id="row2696103911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54596161229"><a name="p54596161229"></a><a name="p54596161229"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1491323415"><a name="p1491323415"></a><a name="p1491323415"></a>addmv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p79152311113"><a name="p79152311113"></a><a name="p79152311113"></a>addmv_out_npu</p>
</td>
</tr>
<tr id="row116976397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04591716922"><a name="p04591716922"></a><a name="p04591716922"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p149112239113"><a name="p149112239113"></a><a name="p149112239113"></a>addr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p169116231016"><a name="p169116231016"></a><a name="p169116231016"></a>addr_npu</p>
</td>
</tr>
<tr id="row1769718393121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p545914161122"><a name="p545914161122"></a><a name="p545914161122"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p139112231913"><a name="p139112231913"></a><a name="p139112231913"></a>addr_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8911323515"><a name="p8911323515"></a><a name="p8911323515"></a>addr_npu_</p>
</td>
</tr>
<tr id="row1669716393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145901617213"><a name="p145901617213"></a><a name="p145901617213"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17914238113"><a name="p17914238113"></a><a name="p17914238113"></a>addr.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p159114231610"><a name="p159114231610"></a><a name="p159114231610"></a>addr_out_npu</p>
</td>
</tr>
<tr id="row1469716399129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1145921614213"><a name="p1145921614213"></a><a name="p1145921614213"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p39116231918"><a name="p39116231918"></a><a name="p39116231918"></a>affine_grid_generator</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p791202314110"><a name="p791202314110"></a><a name="p791202314110"></a>affine_grid_generator_npu</p>
</td>
</tr>
<tr id="row6697143981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194595161421"><a name="p194595161421"></a><a name="p194595161421"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p109116232116"><a name="p109116232116"></a><a name="p109116232116"></a>affine_grid_generator_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1391112317112"><a name="p1391112317112"></a><a name="p1391112317112"></a>affine_grid_generator_backward_npu</p>
</td>
</tr>
<tr id="row5697103931212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p545912161620"><a name="p545912161620"></a><a name="p545912161620"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3918238112"><a name="p3918238112"></a><a name="p3918238112"></a>all.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1491142318116"><a name="p1491142318116"></a><a name="p1491142318116"></a>all_npu</p>
</td>
</tr>
<tr id="row11697133961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174590161428"><a name="p174590161428"></a><a name="p174590161428"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2091823114"><a name="p2091823114"></a><a name="p2091823114"></a>all.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p149111231115"><a name="p149111231115"></a><a name="p149111231115"></a>all_out_npu</p>
</td>
</tr>
<tr id="row13697239171218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p045914169210"><a name="p045914169210"></a><a name="p045914169210"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7912234120"><a name="p7912234120"></a><a name="p7912234120"></a>any.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p59112231013"><a name="p59112231013"></a><a name="p59112231013"></a>any_npu</p>
</td>
</tr>
<tr id="row7698143951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3459916920"><a name="p3459916920"></a><a name="p3459916920"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p791623216"><a name="p791623216"></a><a name="p791623216"></a>any.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17912231012"><a name="p17912231012"></a><a name="p17912231012"></a>any_out_npu</p>
</td>
</tr>
<tr id="row3698133916123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04605168219"><a name="p04605168219"></a><a name="p04605168219"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p29214237115"><a name="p29214237115"></a><a name="p29214237115"></a>arange</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p092623816"><a name="p092623816"></a><a name="p092623816"></a>arange_npu</p>
</td>
</tr>
<tr id="row86981439181215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1446091613214"><a name="p1446091613214"></a><a name="p1446091613214"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1392132310119"><a name="p1392132310119"></a><a name="p1392132310119"></a>arange.start</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p109232317116"><a name="p109232317116"></a><a name="p109232317116"></a>arange_npu</p>
</td>
</tr>
<tr id="row8698203971214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146015163211"><a name="p146015163211"></a><a name="p146015163211"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p192123819"><a name="p192123819"></a><a name="p192123819"></a>arange.start_step</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p792823712"><a name="p792823712"></a><a name="p792823712"></a>arange_npu</p>
</td>
</tr>
<tr id="row1698153941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194608160212"><a name="p194608160212"></a><a name="p194608160212"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1492152315113"><a name="p1492152315113"></a><a name="p1492152315113"></a>arange.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2092162320115"><a name="p2092162320115"></a><a name="p2092162320115"></a>arange_out_npu</p>
</td>
</tr>
<tr id="row4698143917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2460616021"><a name="p2460616021"></a><a name="p2460616021"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p109211231911"><a name="p109211231911"></a><a name="p109211231911"></a>arange.start_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1592102320119"><a name="p1592102320119"></a><a name="p1592102320119"></a>arange_out_npu</p>
</td>
</tr>
<tr id="row1469810393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19460716125"><a name="p19460716125"></a><a name="p19460716125"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18924231313"><a name="p18924231313"></a><a name="p18924231313"></a>_dim_arange</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2921023913"><a name="p2921023913"></a><a name="p2921023913"></a>_dim_arange_npu</p>
</td>
</tr>
<tr id="row17698153919124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174601316821"><a name="p174601316821"></a><a name="p174601316821"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7921323112"><a name="p7921323112"></a><a name="p7921323112"></a>argmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12929239114"><a name="p12929239114"></a><a name="p12929239114"></a>argmax_npu</p>
</td>
</tr>
<tr id="row46981739181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14460141616212"><a name="p14460141616212"></a><a name="p14460141616212"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3921231317"><a name="p3921231317"></a><a name="p3921231317"></a>argmin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p89212314112"><a name="p89212314112"></a><a name="p89212314112"></a>argmin_npu</p>
</td>
</tr>
<tr id="row46981939141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114603161024"><a name="p114603161024"></a><a name="p114603161024"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3924233116"><a name="p3924233116"></a><a name="p3924233116"></a>as_strided</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19215234115"><a name="p19215234115"></a><a name="p19215234115"></a>as_strided_npu</p>
</td>
</tr>
<tr id="row2698339151220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11460816521"><a name="p11460816521"></a><a name="p11460816521"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20922238112"><a name="p20922238112"></a><a name="p20922238112"></a>as_strided_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17921236118"><a name="p17921236118"></a><a name="p17921236118"></a>as_strided_npu_</p>
</td>
</tr>
<tr id="row369911399124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4460916527"><a name="p4460916527"></a><a name="p4460916527"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p109202310111"><a name="p109202310111"></a><a name="p109202310111"></a>asin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p179216238115"><a name="p179216238115"></a><a name="p179216238115"></a>asin_npu</p>
</td>
</tr>
<tr id="row106992394125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446015161828"><a name="p446015161828"></a><a name="p446015161828"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p89210231810"><a name="p89210231810"></a><a name="p89210231810"></a>asin_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p119218231718"><a name="p119218231718"></a><a name="p119218231718"></a>asin_npu_</p>
</td>
</tr>
<tr id="row9699139121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104601161822"><a name="p104601161822"></a><a name="p104601161822"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1192152319110"><a name="p1192152319110"></a><a name="p1192152319110"></a>asin.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5921523210"><a name="p5921523210"></a><a name="p5921523210"></a>asin_out_npu</p>
</td>
</tr>
<tr id="row166991339121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44608161825"><a name="p44608161825"></a><a name="p44608161825"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8927237116"><a name="p8927237116"></a><a name="p8927237116"></a>atan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3921236114"><a name="p3921236114"></a><a name="p3921236114"></a>atan_npu</p>
</td>
</tr>
<tr id="row3699139191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44604161725"><a name="p44604161725"></a><a name="p44604161725"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1292223518"><a name="p1292223518"></a><a name="p1292223518"></a>atan_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12921723613"><a name="p12921723613"></a><a name="p12921723613"></a>atan_npu_</p>
</td>
</tr>
<tr id="row269915391125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10460516425"><a name="p10460516425"></a><a name="p10460516425"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p169211231119"><a name="p169211231119"></a><a name="p169211231119"></a>atan.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p139372313116"><a name="p139372313116"></a><a name="p139372313116"></a>atan_out_npu</p>
</td>
</tr>
<tr id="row869983913127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10460716129"><a name="p10460716129"></a><a name="p10460716129"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1693102316116"><a name="p1693102316116"></a><a name="p1693102316116"></a>baddbmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p129319231417"><a name="p129319231417"></a><a name="p129319231417"></a>baddbmm_npu</p>
</td>
</tr>
<tr id="row46997391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84605163210"><a name="p84605163210"></a><a name="p84605163210"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p129319239110"><a name="p129319239110"></a><a name="p129319239110"></a>baddbmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p109311231312"><a name="p109311231312"></a><a name="p109311231312"></a>baddbmm_npu_</p>
</td>
</tr>
<tr id="row18699143911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846013161021"><a name="p846013161021"></a><a name="p846013161021"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p89313236119"><a name="p89313236119"></a><a name="p89313236119"></a>baddbmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p129315231214"><a name="p129315231214"></a><a name="p129315231214"></a>baddbmm_out_npu</p>
</td>
</tr>
<tr id="row9700163961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11461181616213"><a name="p11461181616213"></a><a name="p11461181616213"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p893523912"><a name="p893523912"></a><a name="p893523912"></a>bartlett_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19931023419"><a name="p19931023419"></a><a name="p19931023419"></a>bartlett_window_npu</p>
</td>
</tr>
<tr id="row57008394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p046141617218"><a name="p046141617218"></a><a name="p046141617218"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p99310232118"><a name="p99310232118"></a><a name="p99310232118"></a>bartlett_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p179316236120"><a name="p179316236120"></a><a name="p179316236120"></a>bartlett_window_npu</p>
</td>
</tr>
<tr id="row20700113951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74615161222"><a name="p74615161222"></a><a name="p74615161222"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10933231016"><a name="p10933231016"></a><a name="p10933231016"></a>batch_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8931623011"><a name="p8931623011"></a><a name="p8931623011"></a>batch_norm_npu_</p>
</td>
</tr>
<tr id="row1070043920122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194614162212"><a name="p194614162212"></a><a name="p194614162212"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p179320235119"><a name="p179320235119"></a><a name="p179320235119"></a>_batch_norm_impl_index</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p393172311119"><a name="p393172311119"></a><a name="p393172311119"></a>_batch_norm_impl_index_npu</p>
</td>
</tr>
<tr id="row1970093931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9461116320"><a name="p9461116320"></a><a name="p9461116320"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p139320231512"><a name="p139320231512"></a><a name="p139320231512"></a>_batch_norm_impl_index_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p49372311111"><a name="p49372311111"></a><a name="p49372311111"></a>_batch_norm_impl_index_backward_npu</p>
</td>
</tr>
<tr id="row270033916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16461151618219"><a name="p16461151618219"></a><a name="p16461151618219"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9935231719"><a name="p9935231719"></a><a name="p9935231719"></a>bernoulli</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p179372313119"><a name="p179372313119"></a><a name="p179372313119"></a>bernoulli_npu</p>
</td>
</tr>
<tr id="row10700339151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4461191610214"><a name="p4461191610214"></a><a name="p4461191610214"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p093192313118"><a name="p093192313118"></a><a name="p093192313118"></a>bernoulli_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7931231814"><a name="p7931231814"></a><a name="p7931231814"></a>bernoulli_npu_</p>
</td>
</tr>
<tr id="row12700539141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146118167215"><a name="p1146118167215"></a><a name="p1146118167215"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0932023419"><a name="p0932023419"></a><a name="p0932023419"></a>bernoulli_.float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1931023413"><a name="p1931023413"></a><a name="p1931023413"></a>bernoulli_npu_</p>
</td>
</tr>
<tr id="row8700203961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2461131611210"><a name="p2461131611210"></a><a name="p2461131611210"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3934233111"><a name="p3934233111"></a><a name="p3934233111"></a>binary_cross_entropy</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p189312235114"><a name="p189312235114"></a><a name="p189312235114"></a>binary_cross_entropy_npu</p>
</td>
</tr>
<tr id="row1770043931218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746161614219"><a name="p1746161614219"></a><a name="p1746161614219"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1393112317115"><a name="p1393112317115"></a><a name="p1393112317115"></a>binary_cross_entropy.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9938231817"><a name="p9938231817"></a><a name="p9938231817"></a>binary_cross_entropy_out_npu</p>
</td>
</tr>
<tr id="row5700139121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84611816526"><a name="p84611816526"></a><a name="p84611816526"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16931723613"><a name="p16931723613"></a><a name="p16931723613"></a>binary_cross_entropy_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p109312315111"><a name="p109312315111"></a><a name="p109312315111"></a>binary_cross_entropy_backward_npu</p>
</td>
</tr>
<tr id="row137012039151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546191619213"><a name="p1546191619213"></a><a name="p1546191619213"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p393112310113"><a name="p393112310113"></a><a name="p393112310113"></a>binary_cross_entropy_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5931231714"><a name="p5931231714"></a><a name="p5931231714"></a>binary_cross_entropy_backward_out_npu</p>
</td>
</tr>
<tr id="row5701143914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746218164216"><a name="p1746218164216"></a><a name="p1746218164216"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p894152314112"><a name="p894152314112"></a><a name="p894152314112"></a>binary_cross_entropy_with_logits</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p494142310118"><a name="p494142310118"></a><a name="p494142310118"></a>binary_cross_entropy_with_logits_npu</p>
</td>
</tr>
<tr id="row18701439171211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17462616221"><a name="p17462616221"></a><a name="p17462616221"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p149413231116"><a name="p149413231116"></a><a name="p149413231116"></a>binary_cross_entropy_with_logits_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1294112311112"><a name="p1294112311112"></a><a name="p1294112311112"></a>binary_cross_entropy_with_logits_backward_npu</p>
</td>
</tr>
<tr id="row5701173912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246216169211"><a name="p1246216169211"></a><a name="p1246216169211"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p694123219"><a name="p694123219"></a><a name="p694123219"></a>bitwise_not</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p189462313114"><a name="p189462313114"></a><a name="p189462313114"></a>bitwise_not_npu</p>
</td>
</tr>
<tr id="row270111390122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164621316724"><a name="p164621316724"></a><a name="p164621316724"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p39402318117"><a name="p39402318117"></a><a name="p39402318117"></a>bitwise_not_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p49414237115"><a name="p49414237115"></a><a name="p49414237115"></a>bitwise_not_npu_</p>
</td>
</tr>
<tr id="row27010399120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14627161629"><a name="p14627161629"></a><a name="p14627161629"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19419235116"><a name="p19419235116"></a><a name="p19419235116"></a>bitwise_not.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p209420231312"><a name="p209420231312"></a><a name="p209420231312"></a>bitwise_not_out_npu</p>
</td>
</tr>
<tr id="row157011339201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15462316429"><a name="p15462316429"></a><a name="p15462316429"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p494123915"><a name="p494123915"></a><a name="p494123915"></a>logical_not</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p49412231218"><a name="p49412231218"></a><a name="p49412231218"></a>logical_not_npu</p>
</td>
</tr>
<tr id="row187011339161218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346214163212"><a name="p346214163212"></a><a name="p346214163212"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p179462314118"><a name="p179462314118"></a><a name="p179462314118"></a>logical_not_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p194923318"><a name="p194923318"></a><a name="p194923318"></a>logical_not_npu_</p>
</td>
</tr>
<tr id="row20701183921218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146271617211"><a name="p1146271617211"></a><a name="p1146271617211"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11944233118"><a name="p11944233118"></a><a name="p11944233118"></a>logical_not.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0941323811"><a name="p0941323811"></a><a name="p0941323811"></a>logical_not_out_npu</p>
</td>
</tr>
<tr id="row177011539151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84621516924"><a name="p84621516924"></a><a name="p84621516924"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1394142315117"><a name="p1394142315117"></a><a name="p1394142315117"></a>logical_and</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1941823915"><a name="p1941823915"></a><a name="p1941823915"></a>logical_and_npu</p>
</td>
</tr>
<tr id="row37015396121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24626162028"><a name="p24626162028"></a><a name="p24626162028"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p159411238112"><a name="p159411238112"></a><a name="p159411238112"></a>logical_and_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p139442312114"><a name="p139442312114"></a><a name="p139442312114"></a>logical_and_npu_</p>
</td>
</tr>
<tr id="row1470243915128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11462101612215"><a name="p11462101612215"></a><a name="p11462101612215"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3944231319"><a name="p3944231319"></a><a name="p3944231319"></a>logical_and.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p69413234112"><a name="p69413234112"></a><a name="p69413234112"></a>logical_and_out_npu</p>
</td>
</tr>
<tr id="row870210392126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144622161322"><a name="p144622161322"></a><a name="p144622161322"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p89417230113"><a name="p89417230113"></a><a name="p89417230113"></a>logical_or</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15941923715"><a name="p15941923715"></a><a name="p15941923715"></a>logical_or_npu</p>
</td>
</tr>
<tr id="row670210390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9462316321"><a name="p9462316321"></a><a name="p9462316321"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9941823516"><a name="p9941823516"></a><a name="p9941823516"></a>logical_or_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p139472319116"><a name="p139472319116"></a><a name="p139472319116"></a>logical_or_npu_</p>
</td>
</tr>
<tr id="row1570215393125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546219161422"><a name="p546219161422"></a><a name="p546219161422"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5942239114"><a name="p5942239114"></a><a name="p5942239114"></a>logical_or.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1194162315116"><a name="p1194162315116"></a><a name="p1194162315116"></a>logical_or_out_npu</p>
</td>
</tr>
<tr id="row18702203919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446291617213"><a name="p446291617213"></a><a name="p446291617213"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15943232012"><a name="p15943232012"></a><a name="p15943232012"></a>blackman_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5943230115"><a name="p5943230115"></a><a name="p5943230115"></a>blackman_window_npu</p>
</td>
</tr>
<tr id="row1870283916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14462151619216"><a name="p14462151619216"></a><a name="p14462151619216"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1994172312114"><a name="p1994172312114"></a><a name="p1994172312114"></a>blackman_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p209519238120"><a name="p209519238120"></a><a name="p209519238120"></a>blackman_window_npu</p>
</td>
</tr>
<tr id="row1870263914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64621016123"><a name="p64621016123"></a><a name="p64621016123"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p119510231514"><a name="p119510231514"></a><a name="p119510231514"></a>bmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16953232110"><a name="p16953232110"></a><a name="p16953232110"></a>bmm_npu</p>
</td>
</tr>
<tr id="row12702103918126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146231616213"><a name="p146231616213"></a><a name="p146231616213"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p29572314118"><a name="p29572314118"></a><a name="p29572314118"></a>bmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6954231119"><a name="p6954231119"></a><a name="p6954231119"></a>bmm_out_npu</p>
</td>
</tr>
<tr id="row97021739111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1046217164214"><a name="p1046217164214"></a><a name="p1046217164214"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1895112315117"><a name="p1895112315117"></a><a name="p1895112315117"></a>cat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p189511235112"><a name="p189511235112"></a><a name="p189511235112"></a>cat_npu</p>
</td>
</tr>
<tr id="row4702439171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14628160211"><a name="p14628160211"></a><a name="p14628160211"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p69510237117"><a name="p69510237117"></a><a name="p69510237117"></a>cat.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p139522311111"><a name="p139522311111"></a><a name="p139522311111"></a>cat_out_npu</p>
</td>
</tr>
<tr id="row12703153917121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54637161021"><a name="p54637161021"></a><a name="p54637161021"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18952231119"><a name="p18952231119"></a><a name="p18952231119"></a>cat.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p59515231317"><a name="p59515231317"></a><a name="p59515231317"></a>cat_npu</p>
</td>
</tr>
<tr id="row1470363911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546371616214"><a name="p1546371616214"></a><a name="p1546371616214"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p395192319113"><a name="p395192319113"></a><a name="p395192319113"></a>cat.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19513231515"><a name="p19513231515"></a><a name="p19513231515"></a>cat_out_npu</p>
</td>
</tr>
<tr id="row170313398129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94631916625"><a name="p94631916625"></a><a name="p94631916625"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p49522310112"><a name="p49522310112"></a><a name="p49522310112"></a>ceil</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15953231912"><a name="p15953231912"></a><a name="p15953231912"></a>ceil_npu</p>
</td>
</tr>
<tr id="row570333911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246314166216"><a name="p1246314166216"></a><a name="p1246314166216"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p79513231417"><a name="p79513231417"></a><a name="p79513231417"></a>ceil_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4954235111"><a name="p4954235111"></a><a name="p4954235111"></a>ceil_npu_</p>
</td>
</tr>
<tr id="row127031039151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1463516628"><a name="p1463516628"></a><a name="p1463516628"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14951923816"><a name="p14951923816"></a><a name="p14951923816"></a>ceil.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17958233112"><a name="p17958233112"></a><a name="p17958233112"></a>ceil_out_npu</p>
</td>
</tr>
<tr id="row147031239181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94637162217"><a name="p94637162217"></a><a name="p94637162217"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p995142315111"><a name="p995142315111"></a><a name="p995142315111"></a>clamp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p209514237113"><a name="p209514237113"></a><a name="p209514237113"></a>clamp_npu</p>
</td>
</tr>
<tr id="row7703143911121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846351612211"><a name="p846351612211"></a><a name="p846351612211"></a>84</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p195223314"><a name="p195223314"></a><a name="p195223314"></a>clamp_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p895132313114"><a name="p895132313114"></a><a name="p895132313114"></a>clamp_npu_</p>
</td>
</tr>
<tr id="row137031396123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154635161025"><a name="p154635161025"></a><a name="p154635161025"></a>85</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6954236118"><a name="p6954236118"></a><a name="p6954236118"></a>clamp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5954231913"><a name="p5954231913"></a><a name="p5954231913"></a>clamp_out_npu</p>
</td>
</tr>
<tr id="row12703133911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146311617218"><a name="p146311617218"></a><a name="p146311617218"></a>86</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p139519231216"><a name="p139519231216"></a><a name="p139519231216"></a>clamp_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5957231213"><a name="p5957231213"></a><a name="p5957231213"></a>clamp_max_npu</p>
</td>
</tr>
<tr id="row37031139181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4463416326"><a name="p4463416326"></a><a name="p4463416326"></a>87</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1952231511"><a name="p1952231511"></a><a name="p1952231511"></a>clamp_max_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1495102310116"><a name="p1495102310116"></a><a name="p1495102310116"></a>clamp_max_npu_</p>
</td>
</tr>
<tr id="row12703123961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7463816828"><a name="p7463816828"></a><a name="p7463816828"></a>88</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8952234119"><a name="p8952234119"></a><a name="p8952234119"></a>clamp_max.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6950231819"><a name="p6950231819"></a><a name="p6950231819"></a>clamp_max_out_npu</p>
</td>
</tr>
<tr id="row170473991217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12463516727"><a name="p12463516727"></a><a name="p12463516727"></a>89</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19602313111"><a name="p19602313111"></a><a name="p19602313111"></a>clamp_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p119672311118"><a name="p119672311118"></a><a name="p119672311118"></a>clamp_min_npu</p>
</td>
</tr>
<tr id="row370416391125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17463181613212"><a name="p17463181613212"></a><a name="p17463181613212"></a>90</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8961323210"><a name="p8961323210"></a><a name="p8961323210"></a>clamp_min_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4961323412"><a name="p4961323412"></a><a name="p4961323412"></a>clamp_min_npu_</p>
</td>
</tr>
<tr id="row12704173941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246381613220"><a name="p1246381613220"></a><a name="p1246381613220"></a>91</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p89620231316"><a name="p89620231316"></a><a name="p89620231316"></a>clamp_min.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p149618231214"><a name="p149618231214"></a><a name="p149618231214"></a>clamp_min_out_npu</p>
</td>
</tr>
<tr id="row6704239131211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14464116423"><a name="p14464116423"></a><a name="p14464116423"></a>92</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1396172319113"><a name="p1396172319113"></a><a name="p1396172319113"></a>constant_pad_nd</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19612313112"><a name="p19612313112"></a><a name="p19612313112"></a>constant_pad_nd_npu</p>
</td>
</tr>
<tr id="row1570493911129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15464141617210"><a name="p15464141617210"></a><a name="p15464141617210"></a>93</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p69618233111"><a name="p69618233111"></a><a name="p69618233111"></a>contiguous</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19968231118"><a name="p19968231118"></a><a name="p19968231118"></a>contiguous_npu</p>
</td>
</tr>
<tr id="row27048393125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104644164219"><a name="p104644164219"></a><a name="p104644164219"></a>94</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1696523617"><a name="p1696523617"></a><a name="p1696523617"></a>convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p996142318111"><a name="p996142318111"></a><a name="p996142318111"></a>convolution_npu</p>
</td>
</tr>
<tr id="row6704173911219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p246411614210"><a name="p246411614210"></a><a name="p246411614210"></a>95</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12961223515"><a name="p12961223515"></a><a name="p12961223515"></a>_convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p159612319117"><a name="p159612319117"></a><a name="p159612319117"></a>_convolution_npu</p>
</td>
</tr>
<tr id="row1070423914123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12464201619213"><a name="p12464201619213"></a><a name="p12464201619213"></a>96</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19969231316"><a name="p19969231316"></a><a name="p19969231316"></a>_convolution_nogroup</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18961323312"><a name="p18961323312"></a><a name="p18961323312"></a>_convolution_nogroup_npu</p>
</td>
</tr>
<tr id="row1704193951216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144646166213"><a name="p144646166213"></a><a name="p144646166213"></a>97</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p169613235118"><a name="p169613235118"></a><a name="p169613235118"></a>conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p139611231213"><a name="p139611231213"></a><a name="p139611231213"></a>conv2d_npu_</p>
</td>
</tr>
<tr id="row14704113914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84643166215"><a name="p84643166215"></a><a name="p84643166215"></a>98</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p396172311120"><a name="p396172311120"></a><a name="p396172311120"></a>conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13963236111"><a name="p13963236111"></a><a name="p13963236111"></a>_conv3d_npu</p>
</td>
</tr>
<tr id="row207047394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154641616627"><a name="p154641616627"></a><a name="p154641616627"></a>99</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1796202319116"><a name="p1796202319116"></a><a name="p1796202319116"></a>conv_tbc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1496182315111"><a name="p1496182315111"></a><a name="p1496182315111"></a>conv_tbc_npu</p>
</td>
</tr>
<tr id="row14705103916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19464181613216"><a name="p19464181613216"></a><a name="p19464181613216"></a>100</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12961523615"><a name="p12961523615"></a><a name="p12961523615"></a>conv_tbc_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p49618233112"><a name="p49618233112"></a><a name="p49618233112"></a>conv_tbc_backward_npu</p>
</td>
</tr>
<tr id="row15705193961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6464131617218"><a name="p6464131617218"></a><a name="p6464131617218"></a>101</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p159611231518"><a name="p159611231518"></a><a name="p159611231518"></a>conv_transpose2d.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p296112316112"><a name="p296112316112"></a><a name="p296112316112"></a>conv_transpose2d_npu_</p>
</td>
</tr>
<tr id="row270513391126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p946419161529"><a name="p946419161529"></a><a name="p946419161529"></a>102</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p139620231315"><a name="p139620231315"></a><a name="p139620231315"></a>copy_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4961823116"><a name="p4961823116"></a><a name="p4961823116"></a>copy_npu_</p>
</td>
</tr>
<tr id="row15705153951215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114647166218"><a name="p114647166218"></a><a name="p114647166218"></a>103</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p129611233120"><a name="p129611233120"></a><a name="p129611233120"></a>cos</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1696142318116"><a name="p1696142318116"></a><a name="p1696142318116"></a>cos_npu</p>
</td>
</tr>
<tr id="row970573915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164642016822"><a name="p164642016822"></a><a name="p164642016822"></a>104</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1496142313111"><a name="p1496142313111"></a><a name="p1496142313111"></a>cos_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p89615231115"><a name="p89615231115"></a><a name="p89615231115"></a>cos_npu_</p>
</td>
</tr>
<tr id="row107052039171216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246413168213"><a name="p1246413168213"></a><a name="p1246413168213"></a>105</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p79717231315"><a name="p79717231315"></a><a name="p79717231315"></a>cos.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p189772314114"><a name="p189772314114"></a><a name="p189772314114"></a>cos_out_npu</p>
</td>
</tr>
<tr id="row17705203951218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04648168215"><a name="p04648168215"></a><a name="p04648168215"></a>106</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1897142315118"><a name="p1897142315118"></a><a name="p1897142315118"></a>cosh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p179752316110"><a name="p179752316110"></a><a name="p179752316110"></a>cosh_npu</p>
</td>
</tr>
<tr id="row1470543918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16465201612216"><a name="p16465201612216"></a><a name="p16465201612216"></a>107</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p159710232019"><a name="p159710232019"></a><a name="p159710232019"></a>cosh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p189719231117"><a name="p189719231117"></a><a name="p189719231117"></a>cosh_npu_</p>
</td>
</tr>
<tr id="row12707133981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13465171617216"><a name="p13465171617216"></a><a name="p13465171617216"></a>108</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p897152320113"><a name="p897152320113"></a><a name="p897152320113"></a>cosh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1697132312115"><a name="p1697132312115"></a><a name="p1697132312115"></a>cosh_out_npu</p>
</td>
</tr>
<tr id="row197089397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19465216628"><a name="p19465216628"></a><a name="p19465216628"></a>109</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p129702316112"><a name="p129702316112"></a><a name="p129702316112"></a>cummin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1973231516"><a name="p1973231516"></a><a name="p1973231516"></a>cummin_npu</p>
</td>
</tr>
<tr id="row147081039121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104652165215"><a name="p104652165215"></a><a name="p104652165215"></a>110</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p179713238113"><a name="p179713238113"></a><a name="p179713238113"></a>cummin.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p119710231016"><a name="p119710231016"></a><a name="p119710231016"></a>cummin_out_npu</p>
</td>
</tr>
<tr id="row1470863918121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1046561620217"><a name="p1046561620217"></a><a name="p1046561620217"></a>111</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15973230119"><a name="p15973230119"></a><a name="p15973230119"></a>cummin.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17974236112"><a name="p17974236112"></a><a name="p17974236112"></a>cummin_npu</p>
</td>
</tr>
<tr id="row8708203951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14651162210"><a name="p14651162210"></a><a name="p14651162210"></a>112</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p179712231012"><a name="p179712231012"></a><a name="p179712231012"></a>cummin.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17970231316"><a name="p17970231316"></a><a name="p17970231316"></a>cummin_out_npu</p>
</td>
</tr>
<tr id="row8708103941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446541617213"><a name="p446541617213"></a><a name="p446541617213"></a>113</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1697523015"><a name="p1697523015"></a><a name="p1697523015"></a>cumprod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12971423511"><a name="p12971423511"></a><a name="p12971423511"></a>cumprod_npu</p>
</td>
</tr>
<tr id="row17708143911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134651116426"><a name="p134651116426"></a><a name="p134651116426"></a>114</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p159714239111"><a name="p159714239111"></a><a name="p159714239111"></a>cumprod.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1597152317117"><a name="p1597152317117"></a><a name="p1597152317117"></a>cumprod_out_npu</p>
</td>
</tr>
<tr id="row11708839101210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12465161617219"><a name="p12465161617219"></a><a name="p12465161617219"></a>115</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10979231015"><a name="p10979231015"></a><a name="p10979231015"></a>cumprod.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p179720235110"><a name="p179720235110"></a><a name="p179720235110"></a>cumprod_npu</p>
</td>
</tr>
<tr id="row1870815396128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p204653163214"><a name="p204653163214"></a><a name="p204653163214"></a>116</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p29713233117"><a name="p29713233117"></a><a name="p29713233117"></a>cumprod.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11979231118"><a name="p11979231118"></a><a name="p11979231118"></a>cumprod_out_npu</p>
</td>
</tr>
<tr id="row77081539121218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15465151619215"><a name="p15465151619215"></a><a name="p15465151619215"></a>117</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p29718230117"><a name="p29718230117"></a><a name="p29718230117"></a>ctc_loss.IntList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p79702319117"><a name="p79702319117"></a><a name="p79702319117"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row18708123911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1346541613215"><a name="p1346541613215"></a><a name="p1346541613215"></a>118</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p169714231513"><a name="p169714231513"></a><a name="p169714231513"></a>ctc_loss.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p697172316111"><a name="p697172316111"></a><a name="p697172316111"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row15708153941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164652161822"><a name="p164652161822"></a><a name="p164652161822"></a>119</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11971923311"><a name="p11971923311"></a><a name="p11971923311"></a>_ctc_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p189719234111"><a name="p189719234111"></a><a name="p189719234111"></a>ctc_loss_npu</p>
</td>
</tr>
<tr id="row147081539111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34651516924"><a name="p34651516924"></a><a name="p34651516924"></a>120</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4982231210"><a name="p4982231210"></a><a name="p4982231210"></a>_ctc_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19811230117"><a name="p19811230117"></a><a name="p19811230117"></a>ctc_loss_backward_npu</p>
</td>
</tr>
<tr id="row47091839121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546510161423"><a name="p546510161423"></a><a name="p546510161423"></a>121</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p698102318116"><a name="p698102318116"></a><a name="p698102318116"></a>fill_diagonal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1698123219"><a name="p1698123219"></a><a name="p1698123219"></a>fill_diagonal_npu_</p>
</td>
</tr>
<tr id="row18709183971211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846571616211"><a name="p1846571616211"></a><a name="p1846571616211"></a>122</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p129818231616"><a name="p129818231616"></a><a name="p129818231616"></a>div.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1398132320113"><a name="p1398132320113"></a><a name="p1398132320113"></a>div_npu</p>
</td>
</tr>
<tr id="row07096390129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1465121614218"><a name="p1465121614218"></a><a name="p1465121614218"></a>123</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10983231912"><a name="p10983231912"></a><a name="p10983231912"></a>div_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p79817231215"><a name="p79817231215"></a><a name="p79817231215"></a>div_npu_</p>
</td>
</tr>
<tr id="row1370903971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446518161627"><a name="p446518161627"></a><a name="p446518161627"></a>124</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p139818231819"><a name="p139818231819"></a><a name="p139818231819"></a>div.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p179852314117"><a name="p179852314117"></a><a name="p179852314117"></a>div_out_npu</p>
</td>
</tr>
<tr id="row070993961220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84653161124"><a name="p84653161124"></a><a name="p84653161124"></a>125</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1981123317"><a name="p1981123317"></a><a name="p1981123317"></a>div.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17981623411"><a name="p17981623411"></a><a name="p17981623411"></a>div_npu</p>
</td>
</tr>
<tr id="row0709143941212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114665161227"><a name="p114665161227"></a><a name="p114665161227"></a>126</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13981232118"><a name="p13981232118"></a><a name="p13981232118"></a>div_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p49812238115"><a name="p49812238115"></a><a name="p49812238115"></a>div_npu_</p>
</td>
</tr>
<tr id="row1570919393128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546613161522"><a name="p1546613161522"></a><a name="p1546613161522"></a>127</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2987236112"><a name="p2987236112"></a><a name="p2987236112"></a>dot</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p698123119"><a name="p698123119"></a><a name="p698123119"></a>dot_npu</p>
</td>
</tr>
<tr id="row670933917126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154661516327"><a name="p154661516327"></a><a name="p154661516327"></a>128</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p39822311118"><a name="p39822311118"></a><a name="p39822311118"></a>dot.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10988231014"><a name="p10988231014"></a><a name="p10988231014"></a>dot_out_npu</p>
</td>
</tr>
<tr id="row770983915126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p546612161524"><a name="p546612161524"></a><a name="p546612161524"></a>129</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p99819237115"><a name="p99819237115"></a><a name="p99819237115"></a>embedding</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p498623417"><a name="p498623417"></a><a name="p498623417"></a>embedding_npu</p>
</td>
</tr>
<tr id="row57091739171214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1746617161425"><a name="p1746617161425"></a><a name="p1746617161425"></a>130</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2986231411"><a name="p2986231411"></a><a name="p2986231411"></a>embedding_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p169918239115"><a name="p169918239115"></a><a name="p169918239115"></a>embedding_backward_npu</p>
</td>
</tr>
<tr id="row1710123916127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p946671618212"><a name="p946671618212"></a><a name="p946671618212"></a>131</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p69952318118"><a name="p69952318118"></a><a name="p69952318118"></a>embedding_dense_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p119914232011"><a name="p119914232011"></a><a name="p119914232011"></a>embedding_dense_backward_npu</p>
</td>
</tr>
<tr id="row1871033917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7466181613217"><a name="p7466181613217"></a><a name="p7466181613217"></a>132</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16998231318"><a name="p16998231318"></a><a name="p16998231318"></a>embedding_renorm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15991923312"><a name="p15991923312"></a><a name="p15991923312"></a>embedding_renorm_npu_</p>
</td>
</tr>
<tr id="row471017396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846691612218"><a name="p1846691612218"></a><a name="p1846691612218"></a>133</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15999231312"><a name="p15999231312"></a><a name="p15999231312"></a>_embedding_bag</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9993236113"><a name="p9993236113"></a><a name="p9993236113"></a>_embedding_bag_npu</p>
</td>
</tr>
<tr id="row87101939181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184668167214"><a name="p184668167214"></a><a name="p184668167214"></a>134</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5991823119"><a name="p5991823119"></a><a name="p5991823119"></a>empty.memory_format</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p699102315112"><a name="p699102315112"></a><a name="p699102315112"></a>empty_npu</p>
</td>
</tr>
<tr id="row7710193911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346618161427"><a name="p346618161427"></a><a name="p346618161427"></a>135</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p189915233116"><a name="p189915233116"></a><a name="p189915233116"></a>resize_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14991423913"><a name="p14991423913"></a><a name="p14991423913"></a>resize_npu_</p>
</td>
</tr>
<tr id="row1871053961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174663161022"><a name="p174663161022"></a><a name="p174663161022"></a>136</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18992239113"><a name="p18992239113"></a><a name="p18992239113"></a>empty_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6996231914"><a name="p6996231914"></a><a name="p6996231914"></a>empty_like_npu</p>
</td>
</tr>
<tr id="row87101439151213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546691616213"><a name="p1546691616213"></a><a name="p1546691616213"></a>137</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p499162317120"><a name="p499162317120"></a><a name="p499162317120"></a>empty_strided</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19999237111"><a name="p19999237111"></a><a name="p19999237111"></a>empty_strided_npu</p>
</td>
</tr>
<tr id="row9710113951217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1846610161326"><a name="p1846610161326"></a><a name="p1846610161326"></a>138</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6996231614"><a name="p6996231614"></a><a name="p6996231614"></a>erf</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8996230115"><a name="p8996230115"></a><a name="p8996230115"></a>erf_npu</p>
</td>
</tr>
<tr id="row4710143961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846614162214"><a name="p846614162214"></a><a name="p846614162214"></a>139</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p79912239116"><a name="p79912239116"></a><a name="p79912239116"></a>erf_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p119920230116"><a name="p119920230116"></a><a name="p119920230116"></a>erf_npu_</p>
</td>
</tr>
<tr id="row107101539181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184671416828"><a name="p184671416828"></a><a name="p184671416828"></a>140</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p189982316110"><a name="p189982316110"></a><a name="p189982316110"></a>erf.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p129992311111"><a name="p129992311111"></a><a name="p129992311111"></a>erf_out_npu</p>
</td>
</tr>
<tr id="row12710739111212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24678168216"><a name="p24678168216"></a><a name="p24678168216"></a>141</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5991423317"><a name="p5991423317"></a><a name="p5991423317"></a>exp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12996231713"><a name="p12996231713"></a><a name="p12996231713"></a>exp_npu</p>
</td>
</tr>
<tr id="row1771193971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24672161213"><a name="p24672161213"></a><a name="p24672161213"></a>142</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p49916231418"><a name="p49916231418"></a><a name="p49916231418"></a>exp_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1699102317115"><a name="p1699102317115"></a><a name="p1699102317115"></a>exp_npu_</p>
</td>
</tr>
<tr id="row12711193917124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146716168216"><a name="p1146716168216"></a><a name="p1146716168216"></a>143</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p699182313112"><a name="p699182313112"></a><a name="p699182313112"></a>exp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13990231419"><a name="p13990231419"></a><a name="p13990231419"></a>exp_out_npu</p>
</td>
</tr>
<tr id="row5711439191211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1246791614218"><a name="p1246791614218"></a><a name="p1246791614218"></a>144</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7992235117"><a name="p7992235117"></a><a name="p7992235117"></a>expm1</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p29992320119"><a name="p29992320119"></a><a name="p29992320119"></a>expm1_npu</p>
</td>
</tr>
<tr id="row8711113910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646741611219"><a name="p1646741611219"></a><a name="p1646741611219"></a>145</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p99919231012"><a name="p99919231012"></a><a name="p99919231012"></a>expm1_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141006236117"><a name="p141006236117"></a><a name="p141006236117"></a>expm1_npu_</p>
</td>
</tr>
<tr id="row107111639131216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p846717161126"><a name="p846717161126"></a><a name="p846717161126"></a>146</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1210011235115"><a name="p1210011235115"></a><a name="p1210011235115"></a>expm1.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17100223611"><a name="p17100223611"></a><a name="p17100223611"></a>expm1_out_npu</p>
</td>
</tr>
<tr id="row18711103921219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12467416324"><a name="p12467416324"></a><a name="p12467416324"></a>147</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p910011234119"><a name="p910011234119"></a><a name="p910011234119"></a>eye</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41008231113"><a name="p41008231113"></a><a name="p41008231113"></a>eye_npu</p>
</td>
</tr>
<tr id="row14711839151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p446713161229"><a name="p446713161229"></a><a name="p446713161229"></a>148</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1610020237111"><a name="p1610020237111"></a><a name="p1610020237111"></a>eye.m</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161002023316"><a name="p161002023316"></a><a name="p161002023316"></a>eye_npu</p>
</td>
</tr>
<tr id="row67121739171219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34682016926"><a name="p34682016926"></a><a name="p34682016926"></a>149</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p181003235117"><a name="p181003235117"></a><a name="p181003235117"></a>eye.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181006231613"><a name="p181006231613"></a><a name="p181006231613"></a>eye_out_npu</p>
</td>
</tr>
<tr id="row167127398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44689161523"><a name="p44689161523"></a><a name="p44689161523"></a>150</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p110012238117"><a name="p110012238117"></a><a name="p110012238117"></a>eye.m_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161001423617"><a name="p161001423617"></a><a name="p161001423617"></a>eye_out_npu</p>
</td>
</tr>
<tr id="row2712123912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154681916524"><a name="p154681916524"></a><a name="p154681916524"></a>151</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19100152318120"><a name="p19100152318120"></a><a name="p19100152318120"></a>fill_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17100102312118"><a name="p17100102312118"></a><a name="p17100102312118"></a>fill_npu_</p>
</td>
</tr>
<tr id="row157121739161214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24685161225"><a name="p24685161225"></a><a name="p24685161225"></a>152</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p121004231912"><a name="p121004231912"></a><a name="p121004231912"></a>fill_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161006231912"><a name="p161006231912"></a><a name="p161006231912"></a>fill_npu_</p>
</td>
</tr>
<tr id="row1171283971211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146881616216"><a name="p1146881616216"></a><a name="p1146881616216"></a>153</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11100102310111"><a name="p11100102310111"></a><a name="p11100102310111"></a>floor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21000231918"><a name="p21000231918"></a><a name="p21000231918"></a>floor_npu</p>
</td>
</tr>
<tr id="row15712439111216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146815164212"><a name="p146815164212"></a><a name="p146815164212"></a>154</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11002231512"><a name="p11002231512"></a><a name="p11002231512"></a>floor_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7100142310117"><a name="p7100142310117"></a><a name="p7100142310117"></a>floor_npu_</p>
</td>
</tr>
<tr id="row18712133915128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p646821618213"><a name="p646821618213"></a><a name="p646821618213"></a>155</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p510013238112"><a name="p510013238112"></a><a name="p510013238112"></a>floor.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161001323618"><a name="p161001323618"></a><a name="p161001323618"></a>floor_out_npu</p>
</td>
</tr>
<tr id="row171243912124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184689161121"><a name="p184689161121"></a><a name="p184689161121"></a>156</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131006237116"><a name="p131006237116"></a><a name="p131006237116"></a>floor_divide</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1210019232019"><a name="p1210019232019"></a><a name="p1210019232019"></a>floor_divide_npu</p>
</td>
</tr>
<tr id="row07121539141212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16468181619218"><a name="p16468181619218"></a><a name="p16468181619218"></a>157</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14100323112"><a name="p14100323112"></a><a name="p14100323112"></a>floor_divide_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1110013237119"><a name="p1110013237119"></a><a name="p1110013237119"></a>floor_divide_npu_</p>
</td>
</tr>
<tr id="row071373901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146814162215"><a name="p1146814162215"></a><a name="p1146814162215"></a>158</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111001323213"><a name="p111001323213"></a><a name="p111001323213"></a>floor_divide.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151001323814"><a name="p151001323814"></a><a name="p151001323814"></a>floor_divide_out_npu</p>
</td>
</tr>
<tr id="row107131393122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546810161821"><a name="p1546810161821"></a><a name="p1546810161821"></a>159</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p181001023919"><a name="p181001023919"></a><a name="p181001023919"></a>floor_divide.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1310052318115"><a name="p1310052318115"></a><a name="p1310052318115"></a>floor_divide_npu</p>
</td>
</tr>
<tr id="row671383921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134681616022"><a name="p134681616022"></a><a name="p134681616022"></a>160</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p710042315118"><a name="p710042315118"></a><a name="p710042315118"></a>floor_divide_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71011823512"><a name="p71011823512"></a><a name="p71011823512"></a>floor_divide_npu_</p>
</td>
</tr>
<tr id="row1171303931210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7468616423"><a name="p7468616423"></a><a name="p7468616423"></a>161</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11010231015"><a name="p11010231015"></a><a name="p11010231015"></a>frac</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2101202317112"><a name="p2101202317112"></a><a name="p2101202317112"></a>frac_npu</p>
</td>
</tr>
<tr id="row117131339161213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646841618215"><a name="p1646841618215"></a><a name="p1646841618215"></a>162</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1510152317118"><a name="p1510152317118"></a><a name="p1510152317118"></a>frac_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151011231112"><a name="p151011231112"></a><a name="p151011231112"></a>frac_npu_</p>
</td>
</tr>
<tr id="row771333941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24691816223"><a name="p24691816223"></a><a name="p24691816223"></a>163</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p910117232019"><a name="p910117232019"></a><a name="p910117232019"></a>frac.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4101192314112"><a name="p4101192314112"></a><a name="p4101192314112"></a>frac_out_npu</p>
</td>
</tr>
<tr id="row371317396123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p346916163210"><a name="p346916163210"></a><a name="p346916163210"></a>164</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191012231317"><a name="p191012231317"></a><a name="p191012231317"></a>full.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7101123715"><a name="p7101123715"></a><a name="p7101123715"></a>full_npu</p>
</td>
</tr>
<tr id="row1871317392121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154692016822"><a name="p154692016822"></a><a name="p154692016822"></a>165</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1110115235119"><a name="p1110115235119"></a><a name="p1110115235119"></a>full</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1910117234114"><a name="p1910117234114"></a><a name="p1910117234114"></a>full_npu</p>
</td>
</tr>
<tr id="row971313918123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546913161320"><a name="p1546913161320"></a><a name="p1546913161320"></a>166</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81011723419"><a name="p81011723419"></a><a name="p81011723419"></a>full.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21011523115"><a name="p21011523115"></a><a name="p21011523115"></a>full_out_npu</p>
</td>
</tr>
<tr id="row2713939191216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18469416720"><a name="p18469416720"></a><a name="p18469416720"></a>167</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1710112313118"><a name="p1710112313118"></a><a name="p1710112313118"></a>grid_sampler</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131014230116"><a name="p131014230116"></a><a name="p131014230116"></a>grid_sampler_npu</p>
</td>
</tr>
<tr id="row107131039161211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1446915162212"><a name="p1446915162212"></a><a name="p1446915162212"></a>168</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18101112317119"><a name="p18101112317119"></a><a name="p18101112317119"></a>grid_sampler_3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2010116231313"><a name="p2010116231313"></a><a name="p2010116231313"></a>grid_sampler_3d_npu</p>
</td>
</tr>
<tr id="row15714103901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44691016722"><a name="p44691016722"></a><a name="p44691016722"></a>169</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81013231215"><a name="p81013231215"></a><a name="p81013231215"></a>grid_sampler_3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7101122317115"><a name="p7101122317115"></a><a name="p7101122317115"></a>grid_sampler_3d_backward_npu</p>
</td>
</tr>
<tr id="row37144394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1146921618216"><a name="p1146921618216"></a><a name="p1146921618216"></a>170</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8101623812"><a name="p8101623812"></a><a name="p8101623812"></a>hann_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161011123114"><a name="p161011123114"></a><a name="p161011123114"></a>hann_window_npu</p>
</td>
</tr>
<tr id="row107141639111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4469121610218"><a name="p4469121610218"></a><a name="p4469121610218"></a>171</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18101523611"><a name="p18101523611"></a><a name="p18101523611"></a>hann_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12101162315110"><a name="p12101162315110"></a><a name="p12101162315110"></a>hann_window_npu</p>
</td>
</tr>
<tr id="row207141396120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1546911161924"><a name="p1546911161924"></a><a name="p1546911161924"></a>172</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12101172314112"><a name="p12101172314112"></a><a name="p12101172314112"></a>hamming_window</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16101152314117"><a name="p16101152314117"></a><a name="p16101152314117"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row2714143971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19469131611216"><a name="p19469131611216"></a><a name="p19469131611216"></a>173</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1710112239114"><a name="p1710112239114"></a><a name="p1710112239114"></a>hamming_window.periodic</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11012231518"><a name="p11012231518"></a><a name="p11012231518"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row871433991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p146910166212"><a name="p146910166212"></a><a name="p146910166212"></a>174</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71014234118"><a name="p71014234118"></a><a name="p71014234118"></a>hamming_window.periodic_alpha</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p010112315113"><a name="p010112315113"></a><a name="p010112315113"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row371493914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1646918161724"><a name="p1646918161724"></a><a name="p1646918161724"></a>175</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31010236113"><a name="p31010236113"></a><a name="p31010236113"></a>hamming_window.periodic_alpha_beta</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1710110231217"><a name="p1710110231217"></a><a name="p1710110231217"></a>hamming_window_npu</p>
</td>
</tr>
<tr id="row471433931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34691316923"><a name="p34691316923"></a><a name="p34691316923"></a>176</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1710210231413"><a name="p1710210231413"></a><a name="p1710210231413"></a>ger</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1010222316119"><a name="p1010222316119"></a><a name="p1010222316119"></a>ger_npu</p>
</td>
</tr>
<tr id="row9714173971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74692161215"><a name="p74692161215"></a><a name="p74692161215"></a>177</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1710213231915"><a name="p1710213231915"></a><a name="p1710213231915"></a>ger.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p191026231612"><a name="p191026231612"></a><a name="p191026231612"></a>ger_out_npu</p>
</td>
</tr>
<tr id="row187141539171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54692016828"><a name="p54692016828"></a><a name="p54692016828"></a>178</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11021423618"><a name="p11021423618"></a><a name="p11021423618"></a>index.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1110210232117"><a name="p1110210232117"></a><a name="p1110210232117"></a>index_npu</p>
</td>
</tr>
<tr id="row1714183941217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144693161523"><a name="p144693161523"></a><a name="p144693161523"></a>179</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p210210231712"><a name="p210210231712"></a><a name="p210210231712"></a>index_put_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6102142315113"><a name="p6102142315113"></a><a name="p6102142315113"></a>index_put_npu_</p>
</td>
</tr>
<tr id="row4715193981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547061610211"><a name="p547061610211"></a><a name="p547061610211"></a>180</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19102182312119"><a name="p19102182312119"></a><a name="p19102182312119"></a>index_put</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71023239118"><a name="p71023239118"></a><a name="p71023239118"></a>index_put_npu</p>
</td>
</tr>
<tr id="row1715193921210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047012161524"><a name="p1047012161524"></a><a name="p1047012161524"></a>181</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p710242316120"><a name="p710242316120"></a><a name="p710242316120"></a>_index_put_impl_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1410218231018"><a name="p1410218231018"></a><a name="p1410218231018"></a>_index_put_impl_npu_</p>
</td>
</tr>
<tr id="row1671583917123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154708168213"><a name="p154708168213"></a><a name="p154708168213"></a>182</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p110215231212"><a name="p110215231212"></a><a name="p110215231212"></a>inverse</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121026231012"><a name="p121026231012"></a><a name="p121026231012"></a>inverse_npu</p>
</td>
</tr>
<tr id="row5715339141220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347041613213"><a name="p1347041613213"></a><a name="p1347041613213"></a>183</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p510218231211"><a name="p510218231211"></a><a name="p510218231211"></a>inverse.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14102323414"><a name="p14102323414"></a><a name="p14102323414"></a>inverse_out_npu</p>
</td>
</tr>
<tr id="row771512390120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19470111613218"><a name="p19470111613218"></a><a name="p19470111613218"></a>184</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81024231111"><a name="p81024231111"></a><a name="p81024231111"></a>isclose</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21021523315"><a name="p21021523315"></a><a name="p21021523315"></a>isclose_npu</p>
</td>
</tr>
<tr id="row14715439151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74701416926"><a name="p74701416926"></a><a name="p74701416926"></a>185</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1102122310112"><a name="p1102122310112"></a><a name="p1102122310112"></a>isnan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121028234116"><a name="p121028234116"></a><a name="p121028234116"></a>isnan_npu</p>
</td>
</tr>
<tr id="row127151139161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14701166211"><a name="p14701166211"></a><a name="p14701166211"></a>186</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7102132312120"><a name="p7102132312120"></a><a name="p7102132312120"></a>is_nonzero</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21026238119"><a name="p21026238119"></a><a name="p21026238119"></a>is_nonzero_npu</p>
</td>
</tr>
<tr id="row137154396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547011161120"><a name="p1547011161120"></a><a name="p1547011161120"></a>187</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1310212230114"><a name="p1310212230114"></a><a name="p1310212230114"></a>kl_div</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4102423219"><a name="p4102423219"></a><a name="p4102423219"></a>kl_div_npu</p>
</td>
</tr>
<tr id="row1071553981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7470161619217"><a name="p7470161619217"></a><a name="p7470161619217"></a>188</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51022023313"><a name="p51022023313"></a><a name="p51022023313"></a>kl_div_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21021223613"><a name="p21021223613"></a><a name="p21021223613"></a>kl_div_backward_npu</p>
</td>
</tr>
<tr id="row1871533901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747014161825"><a name="p747014161825"></a><a name="p747014161825"></a>189</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5102423819"><a name="p5102423819"></a><a name="p5102423819"></a>kthvalue</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111025231610"><a name="p111025231610"></a><a name="p111025231610"></a>kthvalue_npu</p>
</td>
</tr>
<tr id="row14715173914125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174701160210"><a name="p174701160210"></a><a name="p174701160210"></a>190</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1110232313119"><a name="p1110232313119"></a><a name="p1110232313119"></a>kthvalue.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16102172314115"><a name="p16102172314115"></a><a name="p16102172314115"></a>kthvalue_out_npu</p>
</td>
</tr>
<tr id="row07162395125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3470121616216"><a name="p3470121616216"></a><a name="p3470121616216"></a>191</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91023231618"><a name="p91023231618"></a><a name="p91023231618"></a>kthvalue.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20102142312119"><a name="p20102142312119"></a><a name="p20102142312119"></a>kthvalue_npu</p>
</td>
</tr>
<tr id="row15716639181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114709169219"><a name="p114709169219"></a><a name="p114709169219"></a>192</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13103723913"><a name="p13103723913"></a><a name="p13103723913"></a>kthvalue.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1510332314110"><a name="p1510332314110"></a><a name="p1510332314110"></a>kthvalue_out_npu</p>
</td>
</tr>
<tr id="row1671618395124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14706161026"><a name="p14706161026"></a><a name="p14706161026"></a>193</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161036231117"><a name="p161036231117"></a><a name="p161036231117"></a>native_layer_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111032233111"><a name="p111032233111"></a><a name="p111032233111"></a>layer_norm_npu</p>
</td>
</tr>
<tr id="row12716203961220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447161611210"><a name="p1447161611210"></a><a name="p1447161611210"></a>194</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1110312231918"><a name="p1110312231918"></a><a name="p1110312231918"></a>native_layer_norm_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1210318231612"><a name="p1210318231612"></a><a name="p1210318231612"></a>layer_norm_backward_npu</p>
</td>
</tr>
<tr id="row15716183918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347131616211"><a name="p1347131616211"></a><a name="p1347131616211"></a>195</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31033231213"><a name="p31033231213"></a><a name="p31033231213"></a>linspace</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16103823711"><a name="p16103823711"></a><a name="p16103823711"></a>linspace_npu</p>
</td>
</tr>
<tr id="row11716143981216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74711616929"><a name="p74711616929"></a><a name="p74711616929"></a>196</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3103323919"><a name="p3103323919"></a><a name="p3103323919"></a>linspace.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11103122315116"><a name="p11103122315116"></a><a name="p11103122315116"></a>linspace_out_npu</p>
</td>
</tr>
<tr id="row3716193911124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14471116426"><a name="p14471116426"></a><a name="p14471116426"></a>197</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3103162312110"><a name="p3103162312110"></a><a name="p3103162312110"></a>log</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p171039231214"><a name="p171039231214"></a><a name="p171039231214"></a>log_npu</p>
</td>
</tr>
<tr id="row207171039131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447113161821"><a name="p447113161821"></a><a name="p447113161821"></a>198</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1710332314119"><a name="p1710332314119"></a><a name="p1710332314119"></a>log_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p91037233110"><a name="p91037233110"></a><a name="p91037233110"></a>log_npu_</p>
</td>
</tr>
<tr id="row2717113914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647111161126"><a name="p1647111161126"></a><a name="p1647111161126"></a>199</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161031723616"><a name="p161031723616"></a><a name="p161031723616"></a>log.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1610332319116"><a name="p1610332319116"></a><a name="p1610332319116"></a>log_out_npu</p>
</td>
</tr>
<tr id="row771710399126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134716161920"><a name="p134716161920"></a><a name="p134716161920"></a>200</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p910317232018"><a name="p910317232018"></a><a name="p910317232018"></a>log10</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1103523516"><a name="p1103523516"></a><a name="p1103523516"></a>log10_npu</p>
</td>
</tr>
<tr id="row77174392127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4471816729"><a name="p4471816729"></a><a name="p4471816729"></a>201</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1310318231611"><a name="p1310318231611"></a><a name="p1310318231611"></a>log10_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0103223514"><a name="p0103223514"></a><a name="p0103223514"></a>log10_npu_</p>
</td>
</tr>
<tr id="row1971733971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44711816725"><a name="p44711816725"></a><a name="p44711816725"></a>202</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1610316231419"><a name="p1610316231419"></a><a name="p1610316231419"></a>log10.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31034234119"><a name="p31034234119"></a><a name="p31034234119"></a>log10_out_npu</p>
</td>
</tr>
<tr id="row7717939191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447171612212"><a name="p1447171612212"></a><a name="p1447171612212"></a>203</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p810312317116"><a name="p810312317116"></a><a name="p810312317116"></a>log1p</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31031423317"><a name="p31031423317"></a><a name="p31031423317"></a>log1p_npu</p>
</td>
</tr>
<tr id="row7717103981220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17471101610212"><a name="p17471101610212"></a><a name="p17471101610212"></a>204</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10103202311118"><a name="p10103202311118"></a><a name="p10103202311118"></a>log1p_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1810392312114"><a name="p1810392312114"></a><a name="p1810392312114"></a>log1p_npu_</p>
</td>
</tr>
<tr id="row187181439131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647161610218"><a name="p647161610218"></a><a name="p647161610218"></a>205</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8103223311"><a name="p8103223311"></a><a name="p8103223311"></a>log1p.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161035239114"><a name="p161035239114"></a><a name="p161035239114"></a>log1p_out_npu</p>
</td>
</tr>
<tr id="row0718139141218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15471116526"><a name="p15471116526"></a><a name="p15471116526"></a>206</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p201031123714"><a name="p201031123714"></a><a name="p201031123714"></a>log2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12103523116"><a name="p12103523116"></a><a name="p12103523116"></a>log2_npu</p>
</td>
</tr>
<tr id="row571815397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1847118164218"><a name="p1847118164218"></a><a name="p1847118164218"></a>207</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16103172316114"><a name="p16103172316114"></a><a name="p16103172316114"></a>log2_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9104172318110"><a name="p9104172318110"></a><a name="p9104172318110"></a>log2_npu_</p>
</td>
</tr>
<tr id="row187181639141214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14717161323"><a name="p14717161323"></a><a name="p14717161323"></a>208</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11041523915"><a name="p11041523915"></a><a name="p11041523915"></a>log2.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p101043231816"><a name="p101043231816"></a><a name="p101043231816"></a>log2_out_npu</p>
</td>
</tr>
<tr id="row117186395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847114161426"><a name="p847114161426"></a><a name="p847114161426"></a>209</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111044231913"><a name="p111044231913"></a><a name="p111044231913"></a>logspace</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21043231418"><a name="p21043231418"></a><a name="p21043231418"></a>logspace_npu</p>
</td>
</tr>
<tr id="row1071819393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547114160216"><a name="p547114160216"></a><a name="p547114160216"></a>210</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111042023113"><a name="p111042023113"></a><a name="p111042023113"></a>logspace.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13104122319115"><a name="p13104122319115"></a><a name="p13104122319115"></a>logspace_out_npu</p>
</td>
</tr>
<tr id="row37188399121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24713161429"><a name="p24713161429"></a><a name="p24713161429"></a>211</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171042231515"><a name="p171042231515"></a><a name="p171042231515"></a>log_softmax.int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p101043230118"><a name="p101043230118"></a><a name="p101043230118"></a>log_softmax_npu</p>
</td>
</tr>
<tr id="row137187391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194711916925"><a name="p194711916925"></a><a name="p194711916925"></a>212</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131048235113"><a name="p131048235113"></a><a name="p131048235113"></a>log_softmax.Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1310432311115"><a name="p1310432311115"></a><a name="p1310432311115"></a>log_softmax_npu</p>
</td>
</tr>
<tr id="row16718143912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34722016821"><a name="p34722016821"></a><a name="p34722016821"></a>213</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21041237112"><a name="p21041237112"></a><a name="p21041237112"></a>_log_softmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61042231218"><a name="p61042231218"></a><a name="p61042231218"></a>_log_softmax_npu</p>
</td>
</tr>
<tr id="row4718103991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647210161321"><a name="p1647210161321"></a><a name="p1647210161321"></a>214</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16104152320116"><a name="p16104152320116"></a><a name="p16104152320116"></a>_log_softmax_backward_data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p410417232119"><a name="p410417232119"></a><a name="p410417232119"></a>_log_softmax_backward_npu</p>
</td>
</tr>
<tr id="row271833941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74726164211"><a name="p74726164211"></a><a name="p74726164211"></a>215</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p810432313115"><a name="p810432313115"></a><a name="p810432313115"></a>logsumexp</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31042239112"><a name="p31042239112"></a><a name="p31042239112"></a>logsumexp_npu</p>
</td>
</tr>
<tr id="row197181539111220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9472516521"><a name="p9472516521"></a><a name="p9472516521"></a>216</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1310415232014"><a name="p1310415232014"></a><a name="p1310415232014"></a>logsumexp.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3104723016"><a name="p3104723016"></a><a name="p3104723016"></a>logsumexp_out_npu</p>
</td>
</tr>
<tr id="row8719239121220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10472151610219"><a name="p10472151610219"></a><a name="p10472151610219"></a>217</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6104323216"><a name="p6104323216"></a><a name="p6104323216"></a>logsumexp.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18104142319110"><a name="p18104142319110"></a><a name="p18104142319110"></a>logsumexp_npu</p>
</td>
</tr>
<tr id="row18719173916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047217161822"><a name="p2047217161822"></a><a name="p2047217161822"></a>218</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p01042231914"><a name="p01042231914"></a><a name="p01042231914"></a>logsumexp.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p101044231312"><a name="p101044231312"></a><a name="p101044231312"></a>logsumexp_out_npu</p>
</td>
</tr>
<tr id="row2719103912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184729161128"><a name="p184729161128"></a><a name="p184729161128"></a>219</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1710432311111"><a name="p1710432311111"></a><a name="p1710432311111"></a>matmul</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17104122315111"><a name="p17104122315111"></a><a name="p17104122315111"></a>matmul_npu</p>
</td>
</tr>
<tr id="row371910396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347271610213"><a name="p1347271610213"></a><a name="p1347271610213"></a>220</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71043231717"><a name="p71043231717"></a><a name="p71043231717"></a>matmul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19104152319118"><a name="p19104152319118"></a><a name="p19104152319118"></a>matmul_out_npu</p>
</td>
</tr>
<tr id="row371919398126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547211618210"><a name="p547211618210"></a><a name="p547211618210"></a>221</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41045238111"><a name="p41045238111"></a><a name="p41045238111"></a>matrix_power</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1210522318119"><a name="p1210522318119"></a><a name="p1210522318119"></a>matrix_power_npu</p>
</td>
</tr>
<tr id="row9719439151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947219161422"><a name="p1947219161422"></a><a name="p1947219161422"></a>222</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1810542315116"><a name="p1810542315116"></a><a name="p1810542315116"></a>max.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1910552310116"><a name="p1910552310116"></a><a name="p1910552310116"></a>max_npu</p>
</td>
</tr>
<tr id="row2719193921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047213161929"><a name="p1047213161929"></a><a name="p1047213161929"></a>223</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4105623319"><a name="p4105623319"></a><a name="p4105623319"></a>max.dim_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11054231416"><a name="p11054231416"></a><a name="p11054231416"></a>max_out_npu</p>
</td>
</tr>
<tr id="row1471913910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24727161023"><a name="p24727161023"></a><a name="p24727161023"></a>224</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p110512230114"><a name="p110512230114"></a><a name="p110512230114"></a>max_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p101058231617"><a name="p101058231617"></a><a name="p101058231617"></a>max_npu</p>
</td>
</tr>
<tr id="row197191739101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18472151613210"><a name="p18472151613210"></a><a name="p18472151613210"></a>225</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3105723211"><a name="p3105723211"></a><a name="p3105723211"></a>max.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p210518231914"><a name="p210518231914"></a><a name="p210518231914"></a>max_npu</p>
</td>
</tr>
<tr id="row137191939101212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164723162023"><a name="p164723162023"></a><a name="p164723162023"></a>226</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61056239111"><a name="p61056239111"></a><a name="p61056239111"></a>max.names_dim_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61052231815"><a name="p61052231815"></a><a name="p61052231815"></a>max_out_npu</p>
</td>
</tr>
<tr id="row471917394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34727161326"><a name="p34727161326"></a><a name="p34727161326"></a>227</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71051023714"><a name="p71051023714"></a><a name="p71051023714"></a>max_values.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161051623613"><a name="p161051623613"></a><a name="p161051623613"></a>max_npu</p>
</td>
</tr>
<tr id="row5720239191220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5473181619210"><a name="p5473181619210"></a><a name="p5473181619210"></a>228</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6105023817"><a name="p6105023817"></a><a name="p6105023817"></a>max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11105823714"><a name="p11105823714"></a><a name="p11105823714"></a>max_pool2d_npu</p>
</td>
</tr>
<tr id="row172093913121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047391614217"><a name="p2047391614217"></a><a name="p2047391614217"></a>229</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51053231710"><a name="p51053231710"></a><a name="p51053231710"></a>quantized_max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111051223514"><a name="p111051223514"></a><a name="p111051223514"></a>quantized_max_pool2d_npu</p>
</td>
</tr>
<tr id="row1772073901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847391612212"><a name="p847391612212"></a><a name="p847391612212"></a>230</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0105323714"><a name="p0105323714"></a><a name="p0105323714"></a>mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2105112316117"><a name="p2105112316117"></a><a name="p2105112316117"></a>mean_npu</p>
</td>
</tr>
<tr id="row17720163931217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847314161525"><a name="p847314161525"></a><a name="p847314161525"></a>231</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13105923212"><a name="p13105923212"></a><a name="p13105923212"></a>mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1710516231718"><a name="p1710516231718"></a><a name="p1710516231718"></a>mean_npu</p>
</td>
</tr>
<tr id="row9720143916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174734164213"><a name="p174734164213"></a><a name="p174734164213"></a>232</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p010542312111"><a name="p010542312111"></a><a name="p010542312111"></a>mean.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71059231915"><a name="p71059231915"></a><a name="p71059231915"></a>mean_out_npu</p>
</td>
</tr>
<tr id="row17201339131212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7473116621"><a name="p7473116621"></a><a name="p7473116621"></a>233</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13105823512"><a name="p13105823512"></a><a name="p13105823512"></a>mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1710562310112"><a name="p1710562310112"></a><a name="p1710562310112"></a>mean_npu</p>
</td>
</tr>
<tr id="row57201039191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114738167211"><a name="p114738167211"></a><a name="p114738167211"></a>234</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6105923412"><a name="p6105923412"></a><a name="p6105923412"></a>mean.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31051423817"><a name="p31051423817"></a><a name="p31051423817"></a>mean_out_npu</p>
</td>
</tr>
<tr id="row1372013918126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p247310161322"><a name="p247310161322"></a><a name="p247310161322"></a>235</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p410519231418"><a name="p410519231418"></a><a name="p410519231418"></a>median.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1410510235113"><a name="p1410510235113"></a><a name="p1410510235113"></a>median_npu</p>
</td>
</tr>
<tr id="row3720639201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047313163215"><a name="p1047313163215"></a><a name="p1047313163215"></a>236</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p181055230119"><a name="p181055230119"></a><a name="p181055230119"></a>median.dim_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7105192318110"><a name="p7105192318110"></a><a name="p7105192318110"></a>median_out_npu</p>
</td>
</tr>
<tr id="row107201839161213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2473131618212"><a name="p2473131618212"></a><a name="p2473131618212"></a>237</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10106162320116"><a name="p10106162320116"></a><a name="p10106162320116"></a>median.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61061823418"><a name="p61061823418"></a><a name="p61061823418"></a>median_npu</p>
</td>
</tr>
<tr id="row1872083991217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164736161622"><a name="p164736161622"></a><a name="p164736161622"></a>238</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p310613238120"><a name="p310613238120"></a><a name="p310613238120"></a>median.names_dim_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5106823314"><a name="p5106823314"></a><a name="p5106823314"></a>median_out_npu</p>
</td>
</tr>
<tr id="row1172183941219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347314161213"><a name="p1347314161213"></a><a name="p1347314161213"></a>239</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11061223113"><a name="p11061223113"></a><a name="p11061223113"></a>min.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p171063238117"><a name="p171063238117"></a><a name="p171063238117"></a>min_npu</p>
</td>
</tr>
<tr id="row172116399124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747311161124"><a name="p747311161124"></a><a name="p747311161124"></a>240</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p510618232117"><a name="p510618232117"></a><a name="p510618232117"></a>min.dim_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18106152317118"><a name="p18106152317118"></a><a name="p18106152317118"></a>min_out_npu</p>
</td>
</tr>
<tr id="row67218393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4473161612212"><a name="p4473161612212"></a><a name="p4473161612212"></a>241</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141061237119"><a name="p141061237119"></a><a name="p141061237119"></a>min_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51063237112"><a name="p51063237112"></a><a name="p51063237112"></a>min_npu</p>
</td>
</tr>
<tr id="row1672117397122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164741216821"><a name="p164741216821"></a><a name="p164741216821"></a>242</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p210615231310"><a name="p210615231310"></a><a name="p210615231310"></a>min.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p510632310119"><a name="p510632310119"></a><a name="p510632310119"></a>min_npu</p>
</td>
</tr>
<tr id="row1572114394124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174741016525"><a name="p174741016525"></a><a name="p174741016525"></a>243</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5106112313118"><a name="p5106112313118"></a><a name="p5106112313118"></a>min.names_dim_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1910652311112"><a name="p1910652311112"></a><a name="p1910652311112"></a>min_out_npu</p>
</td>
</tr>
<tr id="row8721139131218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447415165218"><a name="p1447415165218"></a><a name="p1447415165218"></a>244</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1210617233117"><a name="p1210617233117"></a><a name="p1210617233117"></a>min_values.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13106423413"><a name="p13106423413"></a><a name="p13106423413"></a>min_npu</p>
</td>
</tr>
<tr id="row1072153917123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174744163217"><a name="p174744163217"></a><a name="p174744163217"></a>245</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2010613233111"><a name="p2010613233111"></a><a name="p2010613233111"></a>mm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17106623310"><a name="p17106623310"></a><a name="p17106623310"></a>mm_npu</p>
</td>
</tr>
<tr id="row37219396128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54741916527"><a name="p54741916527"></a><a name="p54741916527"></a>246</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16106523414"><a name="p16106523414"></a><a name="p16106523414"></a>mm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141069231117"><a name="p141069231117"></a><a name="p141069231117"></a>mm_out_npu</p>
</td>
</tr>
<tr id="row1572183912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447412161427"><a name="p447412161427"></a><a name="p447412161427"></a>247</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p510613239118"><a name="p510613239118"></a><a name="p510613239118"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151066233111"><a name="p151066233111"></a><a name="p151066233111"></a>mode_npu</p>
</td>
</tr>
<tr id="row167216395126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54741416527"><a name="p54741416527"></a><a name="p54741416527"></a>248</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41062231614"><a name="p41062231614"></a><a name="p41062231614"></a>mode.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141067231416"><a name="p141067231416"></a><a name="p141067231416"></a>mode_out_npu</p>
</td>
</tr>
<tr id="row137211039151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64749163218"><a name="p64749163218"></a><a name="p64749163218"></a>249</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11106723714"><a name="p11106723714"></a><a name="p11106723714"></a>mul.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4106523717"><a name="p4106523717"></a><a name="p4106523717"></a>mul_npu</p>
</td>
</tr>
<tr id="row11722153918121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10474161613215"><a name="p10474161613215"></a><a name="p10474161613215"></a>250</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p181061123413"><a name="p181061123413"></a><a name="p181061123413"></a>mul_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1106723613"><a name="p1106723613"></a><a name="p1106723613"></a>mul_npu_</p>
</td>
</tr>
<tr id="row1472273981219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p547414161121"><a name="p547414161121"></a><a name="p547414161121"></a>251</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71061123117"><a name="p71061123117"></a><a name="p71061123117"></a>mul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1010622310120"><a name="p1010622310120"></a><a name="p1010622310120"></a>mul_out_npu</p>
</td>
</tr>
<tr id="row19722103916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047401616212"><a name="p2047401616212"></a><a name="p2047401616212"></a>252</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1910632315119"><a name="p1910632315119"></a><a name="p1910632315119"></a>mul.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71078231715"><a name="p71078231715"></a><a name="p71078231715"></a>mul_npu</p>
</td>
</tr>
<tr id="row16722143916126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p347411161023"><a name="p347411161023"></a><a name="p347411161023"></a>253</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2107723317"><a name="p2107723317"></a><a name="p2107723317"></a>mul_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p310717231413"><a name="p310717231413"></a><a name="p310717231413"></a>mul_npu_</p>
</td>
</tr>
<tr id="row197221239151219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144745161624"><a name="p144745161624"></a><a name="p144745161624"></a>254</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1110752318117"><a name="p1110752318117"></a><a name="p1110752318117"></a>mv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7107182318117"><a name="p7107182318117"></a><a name="p7107182318117"></a>mv_npu</p>
</td>
</tr>
<tr id="row672219394123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547431616212"><a name="p1547431616212"></a><a name="p1547431616212"></a>255</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11107192316111"><a name="p11107192316111"></a><a name="p11107192316111"></a>mv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p310711233116"><a name="p310711233116"></a><a name="p310711233116"></a>mv_out_npu</p>
</td>
</tr>
<tr id="row1872263971219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1747416161923"><a name="p1747416161923"></a><a name="p1747416161923"></a>256</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p910714231119"><a name="p910714231119"></a><a name="p910714231119"></a>narrow_copy</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181072231917"><a name="p181072231917"></a><a name="p181072231917"></a>narrow_copy_npu</p>
</td>
</tr>
<tr id="row12722123915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147418168214"><a name="p1147418168214"></a><a name="p1147418168214"></a>257</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81079231918"><a name="p81079231918"></a><a name="p81079231918"></a>native_batch_norm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111073235113"><a name="p111073235113"></a><a name="p111073235113"></a>batch_norm_npu</p>
</td>
</tr>
<tr id="row187221739191212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5475171617220"><a name="p5475171617220"></a><a name="p5475171617220"></a>258</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p210711234117"><a name="p210711234117"></a><a name="p210711234117"></a>native_batch_norm_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1110712315111"><a name="p1110712315111"></a><a name="p1110712315111"></a>batch_norm_backward_npu</p>
</td>
</tr>
<tr id="row87224394120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647561618215"><a name="p647561618215"></a><a name="p647561618215"></a>259</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1810752319115"><a name="p1810752319115"></a><a name="p1810752319115"></a>_nnpack_spatial_convolution</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p710772313111"><a name="p710772313111"></a><a name="p710772313111"></a>_nnpack_spatial_convolution_npu</p>
</td>
</tr>
<tr id="row107221239181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174754169211"><a name="p174754169211"></a><a name="p174754169211"></a>260</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1610713231614"><a name="p1610713231614"></a><a name="p1610713231614"></a>ones.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p610711230116"><a name="p610711230116"></a><a name="p610711230116"></a>ones_npu</p>
</td>
</tr>
<tr id="row7722153916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54755160212"><a name="p54755160212"></a><a name="p54755160212"></a>261</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p410718231817"><a name="p410718231817"></a><a name="p410718231817"></a>ones</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2107132320117"><a name="p2107132320117"></a><a name="p2107132320117"></a>ones_npu</p>
</td>
</tr>
<tr id="row15723163901215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047571615214"><a name="p1047571615214"></a><a name="p1047571615214"></a>262</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1610712317119"><a name="p1610712317119"></a><a name="p1610712317119"></a>ones.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1910715231615"><a name="p1910715231615"></a><a name="p1910715231615"></a>ones_out_npu</p>
</td>
</tr>
<tr id="row11723139141218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647513161229"><a name="p647513161229"></a><a name="p647513161229"></a>263</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18107152315113"><a name="p18107152315113"></a><a name="p18107152315113"></a>ones_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1110752318118"><a name="p1110752318118"></a><a name="p1110752318118"></a>ones_like_npu</p>
</td>
</tr>
<tr id="row18723183917129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9475111610212"><a name="p9475111610212"></a><a name="p9475111610212"></a>264</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1310718233118"><a name="p1310718233118"></a><a name="p1310718233118"></a>cdist</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p010716231213"><a name="p010716231213"></a><a name="p010716231213"></a>cdist_npu</p>
</td>
</tr>
<tr id="row572323917129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9475616020"><a name="p9475616020"></a><a name="p9475616020"></a>265</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1010712314111"><a name="p1010712314111"></a><a name="p1010712314111"></a>_cdist_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p110712235111"><a name="p110712235111"></a><a name="p110712235111"></a>_cdist_forward_npu</p>
</td>
</tr>
<tr id="row157236395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147517167217"><a name="p147517167217"></a><a name="p147517167217"></a>266</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16107112313117"><a name="p16107112313117"></a><a name="p16107112313117"></a>_cdist_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131077231614"><a name="p131077231614"></a><a name="p131077231614"></a>_cdist_backward_npu</p>
</td>
</tr>
<tr id="row12723539181215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347531612211"><a name="p1347531612211"></a><a name="p1347531612211"></a>267</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191089231916"><a name="p191089231916"></a><a name="p191089231916"></a>pdist</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181085231610"><a name="p181085231610"></a><a name="p181085231610"></a>pdist_npu</p>
</td>
</tr>
<tr id="row972373971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194751160213"><a name="p194751160213"></a><a name="p194751160213"></a>268</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2010820238113"><a name="p2010820238113"></a><a name="p2010820238113"></a>_pdist_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4108112319110"><a name="p4108112319110"></a><a name="p4108112319110"></a>_pdist_forward_npu</p>
</td>
</tr>
<tr id="row87231839101216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10475016423"><a name="p10475016423"></a><a name="p10475016423"></a>269</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1510872316120"><a name="p1510872316120"></a><a name="p1510872316120"></a>randperm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3108112314110"><a name="p3108112314110"></a><a name="p3108112314110"></a>randperm_npu</p>
</td>
</tr>
<tr id="row157231339181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144755161829"><a name="p144755161829"></a><a name="p144755161829"></a>270</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41087231311"><a name="p41087231311"></a><a name="p41087231311"></a>randperm.generator</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10108202312112"><a name="p10108202312112"></a><a name="p10108202312112"></a>randperm_npu</p>
</td>
</tr>
<tr id="row472310397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847514161323"><a name="p847514161323"></a><a name="p847514161323"></a>271</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17108112320110"><a name="p17108112320110"></a><a name="p17108112320110"></a>randperm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181089234118"><a name="p181089234118"></a><a name="p181089234118"></a>randperm_out_npu</p>
</td>
</tr>
<tr id="row372323991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1647518161920"><a name="p1647518161920"></a><a name="p1647518161920"></a>272</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1310872318118"><a name="p1310872318118"></a><a name="p1310872318118"></a>randperm.generator_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p310811239110"><a name="p310811239110"></a><a name="p310811239110"></a>randperm_out_npu</p>
</td>
</tr>
<tr id="row18724183911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04768167214"><a name="p04768167214"></a><a name="p04768167214"></a>273</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111081023818"><a name="p111081023818"></a><a name="p111081023818"></a>range.step</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6108323717"><a name="p6108323717"></a><a name="p6108323717"></a>range_npu</p>
</td>
</tr>
<tr id="row16725639181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p247614161022"><a name="p247614161022"></a><a name="p247614161022"></a>274</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3108112311114"><a name="p3108112311114"></a><a name="p3108112311114"></a>range</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141081423616"><a name="p141081423616"></a><a name="p141081423616"></a>range_npu</p>
</td>
</tr>
<tr id="row87254399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p947613161127"><a name="p947613161127"></a><a name="p947613161127"></a>275</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11108182319111"><a name="p11108182319111"></a><a name="p11108182319111"></a>range.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21083236110"><a name="p21083236110"></a><a name="p21083236110"></a>range_out_npu</p>
</td>
</tr>
<tr id="row3726173971212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147611161727"><a name="p147611161727"></a><a name="p147611161727"></a>276</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1310822313113"><a name="p1310822313113"></a><a name="p1310822313113"></a>reciprocal</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71085231619"><a name="p71085231619"></a><a name="p71085231619"></a>reciprocal_npu</p>
</td>
</tr>
<tr id="row5726103918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3476916627"><a name="p3476916627"></a><a name="p3476916627"></a>277</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13108122315119"><a name="p13108122315119"></a><a name="p13108122315119"></a>reciprocal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2108122314115"><a name="p2108122314115"></a><a name="p2108122314115"></a>reciprocal_npu_</p>
</td>
</tr>
<tr id="row972617399127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24769161524"><a name="p24769161524"></a><a name="p24769161524"></a>278</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7108172310113"><a name="p7108172310113"></a><a name="p7108172310113"></a>reciprocal.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41083231311"><a name="p41083231311"></a><a name="p41083231311"></a>reciprocal_out_npu</p>
</td>
</tr>
<tr id="row1972693961213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16476151615220"><a name="p16476151615220"></a><a name="p16476151615220"></a>279</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4108132313110"><a name="p4108132313110"></a><a name="p4108132313110"></a>neg</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121081023216"><a name="p121081023216"></a><a name="p121081023216"></a>neg_npu</p>
</td>
</tr>
<tr id="row8726133920129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447618168218"><a name="p447618168218"></a><a name="p447618168218"></a>280</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5108023118"><a name="p5108023118"></a><a name="p5108023118"></a>neg_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14108823216"><a name="p14108823216"></a><a name="p14108823216"></a>neg_npu_</p>
</td>
</tr>
<tr id="row12726193914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34765162029"><a name="p34765162029"></a><a name="p34765162029"></a>281</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61089236114"><a name="p61089236114"></a><a name="p61089236114"></a>neg.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61084234113"><a name="p61084234113"></a><a name="p61084234113"></a>neg_out_npu</p>
</td>
</tr>
<tr id="row8726339101213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1547614161929"><a name="p1547614161929"></a><a name="p1547614161929"></a>282</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1510920234110"><a name="p1510920234110"></a><a name="p1510920234110"></a>repeat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p191090232119"><a name="p191090232119"></a><a name="p191090232119"></a>repeat_npu</p>
</td>
</tr>
<tr id="row15726123913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144763161127"><a name="p144763161127"></a><a name="p144763161127"></a>283</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91095231210"><a name="p91095231210"></a><a name="p91095231210"></a>repeat_interleave.self_int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11109182316118"><a name="p11109182316118"></a><a name="p11109182316118"></a>repeat_interleave_npu</p>
</td>
</tr>
<tr id="row1472614394127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147631614219"><a name="p147631614219"></a><a name="p147631614219"></a>284</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61091423412"><a name="p61091423412"></a><a name="p61091423412"></a>round</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16109152319117"><a name="p16109152319117"></a><a name="p16109152319117"></a>round_npu</p>
</td>
</tr>
<tr id="row1072663911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847610167218"><a name="p847610167218"></a><a name="p847610167218"></a>285</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21092239117"><a name="p21092239117"></a><a name="p21092239117"></a>round_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1010913237117"><a name="p1010913237117"></a><a name="p1010913237117"></a>round_npu_</p>
</td>
</tr>
<tr id="row16727239141217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3476191615213"><a name="p3476191615213"></a><a name="p3476191615213"></a>286</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14109523914"><a name="p14109523914"></a><a name="p14109523914"></a>round.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1810914231914"><a name="p1810914231914"></a><a name="p1810914231914"></a>round_out_npu</p>
</td>
</tr>
<tr id="row1572710397121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1476516527"><a name="p1476516527"></a><a name="p1476516527"></a>287</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p201091923916"><a name="p201091923916"></a><a name="p201091923916"></a>relu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p201094231118"><a name="p201094231118"></a><a name="p201094231118"></a>relu_npu</p>
</td>
</tr>
<tr id="row572713392127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04761516824"><a name="p04761516824"></a><a name="p04761516824"></a>288</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101094231413"><a name="p101094231413"></a><a name="p101094231413"></a>relu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31092231115"><a name="p31092231115"></a><a name="p31092231115"></a>relu_npu_</p>
</td>
</tr>
<tr id="row17727639181211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184761916926"><a name="p184761916926"></a><a name="p184761916926"></a>289</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p810915230115"><a name="p810915230115"></a><a name="p810915230115"></a>prelu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41092233115"><a name="p41092233115"></a><a name="p41092233115"></a>prelu_npu</p>
</td>
</tr>
<tr id="row1872783910125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114769161927"><a name="p114769161927"></a><a name="p114769161927"></a>290</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p110919232113"><a name="p110919232113"></a><a name="p110919232113"></a>prelu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181092234111"><a name="p181092234111"></a><a name="p181092234111"></a>prelu_backward_npu</p>
</td>
</tr>
<tr id="row272718396127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947611620210"><a name="p1947611620210"></a><a name="p1947611620210"></a>291</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91091623918"><a name="p91091623918"></a><a name="p91091623918"></a>gelu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11093231113"><a name="p11093231113"></a><a name="p11093231113"></a>gelu_npu</p>
</td>
</tr>
<tr id="row1727163912122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84771716627"><a name="p84771716627"></a><a name="p84771716627"></a>292</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12109823811"><a name="p12109823811"></a><a name="p12109823811"></a>gelu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181096231618"><a name="p181096231618"></a><a name="p181096231618"></a>gelu_backward_npu</p>
</td>
</tr>
<tr id="row9727133901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104774161721"><a name="p104774161721"></a><a name="p104774161721"></a>293</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1410917239112"><a name="p1410917239112"></a><a name="p1410917239112"></a>hardshrink</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31097231712"><a name="p31097231712"></a><a name="p31097231712"></a>hardshrink_npu</p>
</td>
</tr>
<tr id="row17727103918124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124775161217"><a name="p124775161217"></a><a name="p124775161217"></a>294</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61096231914"><a name="p61096231914"></a><a name="p61096231914"></a>hardshrink_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11091231315"><a name="p11091231315"></a><a name="p11091231315"></a>hardshrink_backward_npu</p>
</td>
</tr>
<tr id="row572773917122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84773161021"><a name="p84773161021"></a><a name="p84773161021"></a>295</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141091231415"><a name="p141091231415"></a><a name="p141091231415"></a>rsqrt</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161091423916"><a name="p161091423916"></a><a name="p161091423916"></a>rsqrt_npu</p>
</td>
</tr>
<tr id="row1172723911216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p147711161526"><a name="p147711161526"></a><a name="p147711161526"></a>296</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p201091723314"><a name="p201091723314"></a><a name="p201091723314"></a>rsqrt_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19109723218"><a name="p19109723218"></a><a name="p19109723218"></a>rsqrt_npu_</p>
</td>
</tr>
<tr id="row1672763961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14772161226"><a name="p14772161226"></a><a name="p14772161226"></a>297</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61101423612"><a name="p61101423612"></a><a name="p61101423612"></a>rsqrt.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p511062319111"><a name="p511062319111"></a><a name="p511062319111"></a>rsqrt_out_npu</p>
</td>
</tr>
<tr id="row87284399129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947711161521"><a name="p1947711161521"></a><a name="p1947711161521"></a>298</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3110423318"><a name="p3110423318"></a><a name="p3110423318"></a>selu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p811042312119"><a name="p811042312119"></a><a name="p811042312119"></a>selu_npu</p>
</td>
</tr>
<tr id="row20728839161216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647771615217"><a name="p647771615217"></a><a name="p647771615217"></a>299</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1110182311117"><a name="p1110182311117"></a><a name="p1110182311117"></a>selu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131108231918"><a name="p131108231918"></a><a name="p131108231918"></a>selu_npu_</p>
</td>
</tr>
<tr id="row1072893910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047716161929"><a name="p2047716161929"></a><a name="p2047716161929"></a>300</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1711018231618"><a name="p1711018231618"></a><a name="p1711018231618"></a>celu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19110172310117"><a name="p19110172310117"></a><a name="p19110172310117"></a>celu_npu</p>
</td>
</tr>
<tr id="row1672863915127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184772161020"><a name="p184772161020"></a><a name="p184772161020"></a>301</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12110142315116"><a name="p12110142315116"></a><a name="p12110142315116"></a>celu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13110172318118"><a name="p13110172318118"></a><a name="p13110172318118"></a>celu_npu_</p>
</td>
</tr>
<tr id="row5728739101210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54773168211"><a name="p54773168211"></a><a name="p54773168211"></a>302</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1811182310112"><a name="p1811182310112"></a><a name="p1811182310112"></a>sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81114235117"><a name="p81114235117"></a><a name="p81114235117"></a>sigmoid_npu</p>
</td>
</tr>
<tr id="row15728153941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3477316220"><a name="p3477316220"></a><a name="p3477316220"></a>303</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1411114231318"><a name="p1411114231318"></a><a name="p1411114231318"></a>sigmoid_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1211132311120"><a name="p1211132311120"></a><a name="p1211132311120"></a>sigmoid_npu_</p>
</td>
</tr>
<tr id="row15728239111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19477181614214"><a name="p19477181614214"></a><a name="p19477181614214"></a>304</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p311118231318"><a name="p311118231318"></a><a name="p311118231318"></a>sigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21111723319"><a name="p21111723319"></a><a name="p21111723319"></a>sigmoid_out_npu</p>
</td>
</tr>
<tr id="row67282039151218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15477151614219"><a name="p15477151614219"></a><a name="p15477151614219"></a>305</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111111623616"><a name="p111111623616"></a><a name="p111111623616"></a>sin</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61111123517"><a name="p61111123517"></a><a name="p61111123517"></a>sin_npu</p>
</td>
</tr>
<tr id="row172818396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1347814161129"><a name="p1347814161129"></a><a name="p1347814161129"></a>306</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6111102316117"><a name="p6111102316117"></a><a name="p6111102316117"></a>sin_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p911172314120"><a name="p911172314120"></a><a name="p911172314120"></a>sin_npu_</p>
</td>
</tr>
<tr id="row2728039181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134785167218"><a name="p134785167218"></a><a name="p134785167218"></a>307</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p711114239114"><a name="p711114239114"></a><a name="p711114239114"></a>sin.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161112231312"><a name="p161112231312"></a><a name="p161112231312"></a>sin_out_npu</p>
</td>
</tr>
<tr id="row77287391121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647851613215"><a name="p647851613215"></a><a name="p647851613215"></a>308</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131111231711"><a name="p131111231711"></a><a name="p131111231711"></a>sinh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151111823811"><a name="p151111823811"></a><a name="p151111823811"></a>sinh_npu</p>
</td>
</tr>
<tr id="row772923912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184787161725"><a name="p184787161725"></a><a name="p184787161725"></a>309</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13111162313116"><a name="p13111162313116"></a><a name="p13111162313116"></a>sinh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7111523713"><a name="p7111523713"></a><a name="p7111523713"></a>sinh_npu_</p>
</td>
</tr>
<tr id="row1872912397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p047819161626"><a name="p047819161626"></a><a name="p047819161626"></a>310</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161111723110"><a name="p161111723110"></a><a name="p161111723110"></a>sinh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31113231619"><a name="p31113231619"></a><a name="p31113231619"></a>sinh_out_npu</p>
</td>
</tr>
<tr id="row772923941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p747820160210"><a name="p747820160210"></a><a name="p747820160210"></a>311</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10111123611"><a name="p10111123611"></a><a name="p10111123611"></a>slogdet</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p211114238115"><a name="p211114238115"></a><a name="p211114238115"></a>slogdet_npu</p>
</td>
</tr>
<tr id="row7729143919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174781416028"><a name="p174781416028"></a><a name="p174781416028"></a>312</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11111112318110"><a name="p11111112318110"></a><a name="p11111112318110"></a>softmax.int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71111923411"><a name="p71111923411"></a><a name="p71111923411"></a>softmax_npu</p>
</td>
</tr>
<tr id="row972917397121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1047811169220"><a name="p1047811169220"></a><a name="p1047811169220"></a>313</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1411115231418"><a name="p1411115231418"></a><a name="p1411115231418"></a>softmax.Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51111323611"><a name="p51111323611"></a><a name="p51111323611"></a>softmax_npu</p>
</td>
</tr>
<tr id="row1172903920127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447819161020"><a name="p1447819161020"></a><a name="p1447819161020"></a>314</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6111623915"><a name="p6111623915"></a><a name="p6111623915"></a>_softmax</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p171111123011"><a name="p171111123011"></a><a name="p171111123011"></a>_softmax_npu</p>
</td>
</tr>
<tr id="row12729133920124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2478191614211"><a name="p2478191614211"></a><a name="p2478191614211"></a>315</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p911115238118"><a name="p911115238118"></a><a name="p911115238118"></a>_softmax_backward_data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p511102317117"><a name="p511102317117"></a><a name="p511102317117"></a>_softmax_backward_npu</p>
</td>
</tr>
<tr id="row6729203951211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p847811620215"><a name="p847811620215"></a><a name="p847811620215"></a>316</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p121119231313"><a name="p121119231313"></a><a name="p121119231313"></a>stack</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11114231313"><a name="p11114231313"></a><a name="p11114231313"></a>stack_npu</p>
</td>
</tr>
<tr id="row18729153951219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154781165218"><a name="p154781165218"></a><a name="p154781165218"></a>317</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1511192318116"><a name="p1511192318116"></a><a name="p1511192318116"></a>stack.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1611212234115"><a name="p1611212234115"></a><a name="p1611212234115"></a>stack_out_npu</p>
</td>
</tr>
<tr id="row137291539131215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7478121619218"><a name="p7478121619218"></a><a name="p7478121619218"></a>318</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p211215231218"><a name="p211215231218"></a><a name="p211215231218"></a>sum</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p01125232112"><a name="p01125232112"></a><a name="p01125232112"></a>sum_npu</p>
</td>
</tr>
<tr id="row207291839121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1447871616213"><a name="p1447871616213"></a><a name="p1447871616213"></a>319</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131121323416"><a name="p131121323416"></a><a name="p131121323416"></a>sum.dim_IntList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p311210233113"><a name="p311210233113"></a><a name="p311210233113"></a>sum_npu</p>
</td>
</tr>
<tr id="row77291139121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147818165213"><a name="p1147818165213"></a><a name="p1147818165213"></a>320</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1311219231616"><a name="p1311219231616"></a><a name="p1311219231616"></a>sum.dim_DimnameList</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41121023417"><a name="p41121023417"></a><a name="p41121023417"></a>sum_npu</p>
</td>
</tr>
<tr id="row137301239111219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447821618211"><a name="p447821618211"></a><a name="p447821618211"></a>321</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p011216233116"><a name="p011216233116"></a><a name="p011216233116"></a>sum.IntList_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131122231014"><a name="p131122231014"></a><a name="p131122231014"></a>sum_out_npu</p>
</td>
</tr>
<tr id="row187301139101214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194786167215"><a name="p194786167215"></a><a name="p194786167215"></a>322</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1112132312116"><a name="p1112132312116"></a><a name="p1112132312116"></a>sum.DimnameList_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1311242312110"><a name="p1311242312110"></a><a name="p1311242312110"></a>sum_out_npu</p>
</td>
</tr>
<tr id="row1173053981217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1847931611220"><a name="p1847931611220"></a><a name="p1847931611220"></a>323</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15112122311117"><a name="p15112122311117"></a><a name="p15112122311117"></a>sqrt</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1411262312111"><a name="p1411262312111"></a><a name="p1411262312111"></a>sqrt_npu</p>
</td>
</tr>
<tr id="row20730123981213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1947910165210"><a name="p1947910165210"></a><a name="p1947910165210"></a>324</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101121723419"><a name="p101121723419"></a><a name="p101121723419"></a>sqrt_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31121423511"><a name="p31121423511"></a><a name="p31121423511"></a>sqrt_npu_</p>
</td>
</tr>
<tr id="row773013920126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34791616524"><a name="p34791616524"></a><a name="p34791616524"></a>325</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101128236119"><a name="p101128236119"></a><a name="p101128236119"></a>sqrt.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131121231516"><a name="p131121231516"></a><a name="p131121231516"></a>sqrt_out_npu</p>
</td>
</tr>
<tr id="row1973018398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p647910167219"><a name="p647910167219"></a><a name="p647910167219"></a>326</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31121123613"><a name="p31121123613"></a><a name="p31121123613"></a>std</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10112823716"><a name="p10112823716"></a><a name="p10112823716"></a>std_npu</p>
</td>
</tr>
<tr id="row12730153913122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4479161613216"><a name="p4479161613216"></a><a name="p4479161613216"></a>327</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p5112172318111"><a name="p5112172318111"></a><a name="p5112172318111"></a>std.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14112823516"><a name="p14112823516"></a><a name="p14112823516"></a>std_dim_npu</p>
</td>
</tr>
<tr id="row47301639101218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2047911613210"><a name="p2047911613210"></a><a name="p2047911613210"></a>328</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11112223511"><a name="p11112223511"></a><a name="p11112223511"></a>std_mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1311216231413"><a name="p1311216231413"></a><a name="p1311216231413"></a>std_mean_npu</p>
</td>
</tr>
<tr id="row1873012396124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134791816721"><a name="p134791816721"></a><a name="p134791816721"></a>329</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2011216231519"><a name="p2011216231519"></a><a name="p2011216231519"></a>std_mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6112823816"><a name="p6112823816"></a><a name="p6112823816"></a>std_mean_dim_npu</p>
</td>
</tr>
<tr id="row773043919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10479141613216"><a name="p10479141613216"></a><a name="p10479141613216"></a>330</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11121823810"><a name="p11121823810"></a><a name="p11121823810"></a>std_mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61127236110"><a name="p61127236110"></a><a name="p61127236110"></a>std_mean_names_npu</p>
</td>
</tr>
<tr id="row1473013911211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164795165212"><a name="p164795165212"></a><a name="p164795165212"></a>331</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1711213237118"><a name="p1711213237118"></a><a name="p1711213237118"></a>std.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p711210233118"><a name="p711210233118"></a><a name="p711210233118"></a>std_out_npu</p>
</td>
</tr>
<tr id="row1173173951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124795161215"><a name="p124795161215"></a><a name="p124795161215"></a>332</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1611213231111"><a name="p1611213231111"></a><a name="p1611213231111"></a>std.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1611215231117"><a name="p1611215231117"></a><a name="p1611215231117"></a>std_names_npu</p>
</td>
</tr>
<tr id="row47311839111211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194792161028"><a name="p194792161028"></a><a name="p194792161028"></a>333</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p811342313119"><a name="p811342313119"></a><a name="p811342313119"></a>std.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1711342320115"><a name="p1711342320115"></a><a name="p1711342320115"></a>std_out_npu</p>
</td>
</tr>
<tr id="row7731439171216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447941618212"><a name="p447941618212"></a><a name="p447941618212"></a>334</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p211317233117"><a name="p211317233117"></a><a name="p211317233117"></a>prod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12113182315110"><a name="p12113182315110"></a><a name="p12113182315110"></a>prod_npu</p>
</td>
</tr>
<tr id="row47311439191213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8479101619218"><a name="p8479101619218"></a><a name="p8479101619218"></a>335</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p311316239113"><a name="p311316239113"></a><a name="p311316239113"></a>prod.dim_int</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1111313239111"><a name="p1111313239111"></a><a name="p1111313239111"></a>prod_npu</p>
</td>
</tr>
<tr id="row1373193911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p447917167215"><a name="p447917167215"></a><a name="p447917167215"></a>336</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9113923311"><a name="p9113923311"></a><a name="p9113923311"></a>prod.int_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81131923616"><a name="p81131923616"></a><a name="p81131923616"></a>prod_out_npu</p>
</td>
</tr>
<tr id="row47315396122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19479171620216"><a name="p19479171620216"></a><a name="p19479171620216"></a>337</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1113423316"><a name="p1113423316"></a><a name="p1113423316"></a>prod.dim_Dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p711311231313"><a name="p711311231313"></a><a name="p711311231313"></a>prod_npu</p>
</td>
</tr>
<tr id="row27311139111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124798161629"><a name="p124798161629"></a><a name="p124798161629"></a>338</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41139236117"><a name="p41139236117"></a><a name="p41139236117"></a>prod.Dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141135231316"><a name="p141135231316"></a><a name="p141135231316"></a>prod_out_npu</p>
</td>
</tr>
<tr id="row27312391123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1147919169217"><a name="p1147919169217"></a><a name="p1147919169217"></a>339</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171135231012"><a name="p171135231012"></a><a name="p171135231012"></a>tan</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1113132314113"><a name="p1113132314113"></a><a name="p1113132314113"></a>tan_npu</p>
</td>
</tr>
<tr id="row187311539101213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548014165210"><a name="p1548014165210"></a><a name="p1548014165210"></a>340</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2011318231617"><a name="p2011318231617"></a><a name="p2011318231617"></a>tan_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71131723814"><a name="p71131723814"></a><a name="p71131723814"></a>tan_npu_</p>
</td>
</tr>
<tr id="row187313392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64802161029"><a name="p64802161029"></a><a name="p64802161029"></a>341</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1211313231111"><a name="p1211313231111"></a><a name="p1211313231111"></a>tan.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121134231319"><a name="p121134231319"></a><a name="p121134231319"></a>tan_out_npu</p>
</td>
</tr>
<tr id="row1873183981220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548081614214"><a name="p548081614214"></a><a name="p548081614214"></a>342</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1111319237112"><a name="p1111319237112"></a><a name="p1111319237112"></a>tanh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6113523718"><a name="p6113523718"></a><a name="p6113523718"></a>tanh_npu</p>
</td>
</tr>
<tr id="row12731103951212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948081611215"><a name="p1948081611215"></a><a name="p1948081611215"></a>343</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17113423116"><a name="p17113423116"></a><a name="p17113423116"></a>tanh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1811392315115"><a name="p1811392315115"></a><a name="p1811392315115"></a>tanh_npu_</p>
</td>
</tr>
<tr id="row16732139111215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648091614213"><a name="p1648091614213"></a><a name="p1648091614213"></a>344</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161139231115"><a name="p161139231115"></a><a name="p161139231115"></a>tanh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1111318238111"><a name="p1111318238111"></a><a name="p1111318238111"></a>tanh_out_npu</p>
</td>
</tr>
<tr id="row6732113961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20480151612213"><a name="p20480151612213"></a><a name="p20480151612213"></a>345</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p181138231514"><a name="p181138231514"></a><a name="p181138231514"></a>threshold</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1011318234114"><a name="p1011318234114"></a><a name="p1011318234114"></a>threshold_npu</p>
</td>
</tr>
<tr id="row1473273916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04802161927"><a name="p04802161927"></a><a name="p04802161927"></a>346</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1811313231116"><a name="p1811313231116"></a><a name="p1811313231116"></a>threshold_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p711319231719"><a name="p711319231719"></a><a name="p711319231719"></a>threshold_npu_</p>
</td>
</tr>
<tr id="row673213910127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8480151619212"><a name="p8480151619212"></a><a name="p8480151619212"></a>347</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1711342313118"><a name="p1711342313118"></a><a name="p1711342313118"></a>threshold.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p711310237114"><a name="p711310237114"></a><a name="p711310237114"></a>threshold_out_npu</p>
</td>
</tr>
<tr id="row873263916121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648015161520"><a name="p648015161520"></a><a name="p648015161520"></a>348</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6113132310110"><a name="p6113132310110"></a><a name="p6113132310110"></a>threshold_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p511382313117"><a name="p511382313117"></a><a name="p511382313117"></a>threshold_backward_npu</p>
</td>
</tr>
<tr id="row07321739181219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148016167216"><a name="p1148016167216"></a><a name="p1148016167216"></a>349</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1011412238113"><a name="p1011412238113"></a><a name="p1011412238113"></a>one_hot</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10114112317117"><a name="p10114112317117"></a><a name="p10114112317117"></a>one_hot_npu1</p>
</td>
</tr>
<tr id="row12732163911214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8480516328"><a name="p8480516328"></a><a name="p8480516328"></a>350</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p511416231519"><a name="p511416231519"></a><a name="p511416231519"></a>flip</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111141623918"><a name="p111141623918"></a><a name="p111141623918"></a>flip_npu</p>
</td>
</tr>
<tr id="row117321397120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144801316221"><a name="p144801316221"></a><a name="p144801316221"></a>351</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31142236114"><a name="p31142236114"></a><a name="p31142236114"></a>roll</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16114112313117"><a name="p16114112313117"></a><a name="p16114112313117"></a>roll_npu</p>
</td>
</tr>
<tr id="row873218398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194801316925"><a name="p194801316925"></a><a name="p194801316925"></a>352</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1511412316110"><a name="p1511412316110"></a><a name="p1511412316110"></a>true_divide.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p201143231015"><a name="p201143231015"></a><a name="p201143231015"></a>true_divide_npu</p>
</td>
</tr>
<tr id="row1173243919126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548041618219"><a name="p548041618219"></a><a name="p548041618219"></a>353</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1311410231313"><a name="p1311410231313"></a><a name="p1311410231313"></a>true_divide_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111141523116"><a name="p111141523116"></a><a name="p111141523116"></a>true_divide_npu_</p>
</td>
</tr>
<tr id="row1673219391128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24817161215"><a name="p24817161215"></a><a name="p24817161215"></a>354</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p611414235117"><a name="p611414235117"></a><a name="p611414235117"></a>true_divide.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131145231117"><a name="p131145231117"></a><a name="p131145231117"></a>true_divide_out_npu</p>
</td>
</tr>
<tr id="row117331397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5481111615216"><a name="p5481111615216"></a><a name="p5481111615216"></a>355</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141143237116"><a name="p141143237116"></a><a name="p141143237116"></a>true_divide.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121141323916"><a name="p121141323916"></a><a name="p121141323916"></a>true_divide_npu</p>
</td>
</tr>
<tr id="row6733239131216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18481516224"><a name="p18481516224"></a><a name="p18481516224"></a>356</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3114192316116"><a name="p3114192316116"></a><a name="p3114192316116"></a>true_divide_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p611412238110"><a name="p611412238110"></a><a name="p611412238110"></a>true_divide_npu_</p>
</td>
</tr>
<tr id="row8733113981214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10481716720"><a name="p10481716720"></a><a name="p10481716720"></a>357</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1811418232015"><a name="p1811418232015"></a><a name="p1811418232015"></a>trunc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1711412312117"><a name="p1711412312117"></a><a name="p1711412312117"></a>trunc_npu</p>
</td>
</tr>
<tr id="row107332039151212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64817161211"><a name="p64817161211"></a><a name="p64817161211"></a>358</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1011422319120"><a name="p1011422319120"></a><a name="p1011422319120"></a>trunc_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6114823117"><a name="p6114823117"></a><a name="p6114823117"></a>trunc_npu_</p>
</td>
</tr>
<tr id="row13733339161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64811616724"><a name="p64811616724"></a><a name="p64811616724"></a>359</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151144231017"><a name="p151144231017"></a><a name="p151144231017"></a>trunc.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13114623312"><a name="p13114623312"></a><a name="p13114623312"></a>trunc_out_npu</p>
</td>
</tr>
<tr id="row3733153931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3481101611218"><a name="p3481101611218"></a><a name="p3481101611218"></a>360</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51142231718"><a name="p51142231718"></a><a name="p51142231718"></a>_unique2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1011416231814"><a name="p1011416231814"></a><a name="p1011416231814"></a>_unique2_npu</p>
</td>
</tr>
<tr id="row10733139111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5481111613213"><a name="p5481111613213"></a><a name="p5481111613213"></a>361</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p711492311118"><a name="p711492311118"></a><a name="p711492311118"></a>var</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14114202318112"><a name="p14114202318112"></a><a name="p14114202318112"></a>var_npu</p>
</td>
</tr>
<tr id="row7733739101220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174811716224"><a name="p174811716224"></a><a name="p174811716224"></a>362</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161144235117"><a name="p161144235117"></a><a name="p161144235117"></a>var.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p611452314111"><a name="p611452314111"></a><a name="p611452314111"></a>var_npu</p>
</td>
</tr>
<tr id="row8733183911217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44811516922"><a name="p44811516922"></a><a name="p44811516922"></a>363</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141153231315"><a name="p141153231315"></a><a name="p141153231315"></a>var.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151151023418"><a name="p151151023418"></a><a name="p151151023418"></a>var_out_npu</p>
</td>
</tr>
<tr id="row1733193961217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1248141617211"><a name="p1248141617211"></a><a name="p1248141617211"></a>364</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p911512231014"><a name="p911512231014"></a><a name="p911512231014"></a>var.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p611520231018"><a name="p611520231018"></a><a name="p611520231018"></a>var_npu</p>
</td>
</tr>
<tr id="row157331039111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104817163214"><a name="p104817163214"></a><a name="p104817163214"></a>365</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12115182317119"><a name="p12115182317119"></a><a name="p12115182317119"></a>var.names_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121159233113"><a name="p121159233113"></a><a name="p121159233113"></a>var_out_npu</p>
</td>
</tr>
<tr id="row1573414397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144812161020"><a name="p144812161020"></a><a name="p144812161020"></a>366</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171159233119"><a name="p171159233119"></a><a name="p171159233119"></a>var_mean</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11115142315117"><a name="p11115142315117"></a><a name="p11115142315117"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row11734153931213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1481131615216"><a name="p1481131615216"></a><a name="p1481131615216"></a>367</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1411510234113"><a name="p1411510234113"></a><a name="p1411510234113"></a>var_mean.dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11115202315111"><a name="p11115202315111"></a><a name="p11115202315111"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row173473941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174814160210"><a name="p174814160210"></a><a name="p174814160210"></a>368</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p121151123819"><a name="p121151123819"></a><a name="p121151123819"></a>var_mean.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p711532313112"><a name="p711532313112"></a><a name="p711532313112"></a>var_mean_npu</p>
</td>
</tr>
<tr id="row137341239201216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13481116221"><a name="p13481116221"></a><a name="p13481116221"></a>369</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1811511237111"><a name="p1811511237111"></a><a name="p1811511237111"></a>where.self</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18115192315111"><a name="p18115192315111"></a><a name="p18115192315111"></a>where_npu</p>
</td>
</tr>
<tr id="row19734113931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848161618210"><a name="p848161618210"></a><a name="p848161618210"></a>370</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p011512236116"><a name="p011512236116"></a><a name="p011512236116"></a>where</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15115723511"><a name="p15115723511"></a><a name="p15115723511"></a>where_npu</p>
</td>
</tr>
<tr id="row77341439121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16481516222"><a name="p16481516222"></a><a name="p16481516222"></a>371</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2115223310"><a name="p2115223310"></a><a name="p2115223310"></a>_s_where</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1311522314119"><a name="p1311522314119"></a><a name="p1311522314119"></a>_s_where_npu</p>
</td>
</tr>
<tr id="row147341939121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1448118161522"><a name="p1448118161522"></a><a name="p1448118161522"></a>372</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9115723217"><a name="p9115723217"></a><a name="p9115723217"></a>zeros.names</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13115923516"><a name="p13115923516"></a><a name="p13115923516"></a>zeros_npu</p>
</td>
</tr>
<tr id="row4734439181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34811016329"><a name="p34811016329"></a><a name="p34811016329"></a>373</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111519231016"><a name="p111519231016"></a><a name="p111519231016"></a>zeros</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12115142314119"><a name="p12115142314119"></a><a name="p12115142314119"></a>zeros_npu</p>
</td>
</tr>
<tr id="row47341839121217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164821516622"><a name="p164821516622"></a><a name="p164821516622"></a>374</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p0115142315111"><a name="p0115142315111"></a><a name="p0115142315111"></a>zeros.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16115123316"><a name="p16115123316"></a><a name="p16115123316"></a>zeros_out_npu</p>
</td>
</tr>
<tr id="row2734139131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448291619216"><a name="p448291619216"></a><a name="p448291619216"></a>375</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p811513235119"><a name="p811513235119"></a><a name="p811513235119"></a>zeros_like</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2115162313114"><a name="p2115162313114"></a><a name="p2115162313114"></a>zeros_like_npu</p>
</td>
</tr>
<tr id="row10734153910128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164824161121"><a name="p164824161121"></a><a name="p164824161121"></a>376</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p411514231311"><a name="p411514231311"></a><a name="p411514231311"></a>norm.ScalarOpt_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81151723710"><a name="p81151723710"></a><a name="p81151723710"></a>norm_npu</p>
</td>
</tr>
<tr id="row47351039151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12482201610215"><a name="p12482201610215"></a><a name="p12482201610215"></a>377</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91151223916"><a name="p91151223916"></a><a name="p91151223916"></a>norm.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31151223412"><a name="p31151223412"></a><a name="p31151223412"></a>norm_npu</p>
</td>
</tr>
<tr id="row13735339131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6482116629"><a name="p6482116629"></a><a name="p6482116629"></a>378</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91151623219"><a name="p91151623219"></a><a name="p91151623219"></a>norm.ScalarOpt_dim_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p411562317112"><a name="p411562317112"></a><a name="p411562317112"></a>norm_npu</p>
</td>
</tr>
<tr id="row167351939191219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948217161428"><a name="p1948217161428"></a><a name="p1948217161428"></a>379</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1611562311111"><a name="p1611562311111"></a><a name="p1611562311111"></a>norm.ScalarOpt_dim</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p911610232015"><a name="p911610232015"></a><a name="p911610232015"></a>norm_npu</p>
</td>
</tr>
<tr id="row7735113961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548217161024"><a name="p548217161024"></a><a name="p548217161024"></a>380</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1411682314116"><a name="p1411682314116"></a><a name="p1411682314116"></a>norm.dtype_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16116192316120"><a name="p16116192316120"></a><a name="p16116192316120"></a>norm_out_npu</p>
</td>
</tr>
<tr id="row373503911216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194825167217"><a name="p194825167217"></a><a name="p194825167217"></a>381</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p011612311112"><a name="p011612311112"></a><a name="p011612311112"></a>norm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p171161123219"><a name="p171161123219"></a><a name="p171161123219"></a>norm_out_npu</p>
</td>
</tr>
<tr id="row16735639151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154827161425"><a name="p154827161425"></a><a name="p154827161425"></a>382</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2011622313118"><a name="p2011622313118"></a><a name="p2011622313118"></a>clone</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7116723312"><a name="p7116723312"></a><a name="p7116723312"></a>clone_npu</p>
</td>
</tr>
<tr id="row173518397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948291615214"><a name="p948291615214"></a><a name="p948291615214"></a>383</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p211602319120"><a name="p211602319120"></a><a name="p211602319120"></a>resize_as_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51165234110"><a name="p51165234110"></a><a name="p51165234110"></a>resize_as_npu_</p>
</td>
</tr>
<tr id="row173512397127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194821016020"><a name="p194821016020"></a><a name="p194821016020"></a>384</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p611615234114"><a name="p611615234114"></a><a name="p611615234114"></a>pow.Tensor_Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1811615231817"><a name="p1811615231817"></a><a name="p1811615231817"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row117358394126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348216161828"><a name="p1348216161828"></a><a name="p1348216161828"></a>385</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8116142311114"><a name="p8116142311114"></a><a name="p8116142311114"></a>pow.Tensor_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71161023213"><a name="p71161023213"></a><a name="p71161023213"></a>pow_npu</p>
</td>
</tr>
<tr id="row7735113910125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748214161326"><a name="p1748214161326"></a><a name="p1748214161326"></a>386</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161168232112"><a name="p161168232112"></a><a name="p161168232112"></a>zero_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1111612237116"><a name="p1111612237116"></a><a name="p1111612237116"></a>zero_npu_</p>
</td>
</tr>
<tr id="row20735123919121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1948251617211"><a name="p1948251617211"></a><a name="p1948251617211"></a>387</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101164238117"><a name="p101164238117"></a><a name="p101164238117"></a>sub.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61161523314"><a name="p61161523314"></a><a name="p61161523314"></a>sub_out_npu</p>
</td>
</tr>
<tr id="row4735103911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p748315169219"><a name="p748315169219"></a><a name="p748315169219"></a>388</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p01166231011"><a name="p01166231011"></a><a name="p01166231011"></a>sub.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p101166233113"><a name="p101166233113"></a><a name="p101166233113"></a>sub_npu</p>
</td>
</tr>
<tr id="row137367398126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24831016128"><a name="p24831016128"></a><a name="p24831016128"></a>389</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10116623010"><a name="p10116623010"></a><a name="p10116623010"></a>sub_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9116152313113"><a name="p9116152313113"></a><a name="p9116152313113"></a>sub_npu_</p>
</td>
</tr>
<tr id="row873693961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15483141619216"><a name="p15483141619216"></a><a name="p15483141619216"></a>390</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p201161823813"><a name="p201161823813"></a><a name="p201161823813"></a>sub.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151164231210"><a name="p151164231210"></a><a name="p151164231210"></a>sub_npu</p>
</td>
</tr>
<tr id="row1573610394128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2048301616219"><a name="p2048301616219"></a><a name="p2048301616219"></a>391</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1711602316111"><a name="p1711602316111"></a><a name="p1711602316111"></a>sub_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8116162316116"><a name="p8116162316116"></a><a name="p8116162316116"></a>sub_npu_</p>
</td>
</tr>
<tr id="row17736103910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24836162020"><a name="p24836162020"></a><a name="p24836162020"></a>392</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191167238115"><a name="p191167238115"></a><a name="p191167238115"></a>rsub.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1011612233115"><a name="p1011612233115"></a><a name="p1011612233115"></a>rsub_npu</p>
</td>
</tr>
<tr id="row137361039171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74832162212"><a name="p74832162212"></a><a name="p74832162212"></a>393</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17116112310118"><a name="p17116112310118"></a><a name="p17116112310118"></a>rsub.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p711610237111"><a name="p711610237111"></a><a name="p711610237111"></a>rsub_npu</p>
</td>
</tr>
<tr id="row973673921220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174831161928"><a name="p174831161928"></a><a name="p174831161928"></a>394</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1611617232116"><a name="p1611617232116"></a><a name="p1611617232116"></a>addmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1311617231610"><a name="p1311617231610"></a><a name="p1311617231610"></a>addmm_out_npu</p>
</td>
</tr>
<tr id="row2736539161210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124833164215"><a name="p124833164215"></a><a name="p124833164215"></a>395</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10116172319110"><a name="p10116172319110"></a><a name="p10116172319110"></a>addmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6116223411"><a name="p6116223411"></a><a name="p6116223411"></a>addmm_npu</p>
</td>
</tr>
<tr id="row1736139121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54837161825"><a name="p54837161825"></a><a name="p54837161825"></a>396</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81175231916"><a name="p81175231916"></a><a name="p81175231916"></a>addmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p711716231919"><a name="p711716231919"></a><a name="p711716231919"></a>addmm_npu_</p>
</td>
</tr>
<tr id="row373683917121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948311161028"><a name="p948311161028"></a><a name="p948311161028"></a>397</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p811714231314"><a name="p811714231314"></a><a name="p811714231314"></a>quantize_per_tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41171623110"><a name="p41171623110"></a><a name="p41171623110"></a>quantize_per_tensor_npu</p>
</td>
</tr>
<tr id="row1773611392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048319161218"><a name="p1048319161218"></a><a name="p1048319161218"></a>398</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17117123511"><a name="p17117123511"></a><a name="p17117123511"></a>quantize_per_channel</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61174231311"><a name="p61174231311"></a><a name="p61174231311"></a>quantize_per_channel_npu</p>
</td>
</tr>
<tr id="row2736539161217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124831416728"><a name="p124831416728"></a><a name="p124831416728"></a>399</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51171823913"><a name="p51171823913"></a><a name="p51171823913"></a>to.dtype_layout</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19117323416"><a name="p19117323416"></a><a name="p19117323416"></a>to_npu</p>
</td>
</tr>
<tr id="row11737239181213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54834161021"><a name="p54834161021"></a><a name="p54834161021"></a>400</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14117142317119"><a name="p14117142317119"></a><a name="p14117142317119"></a>to.device</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1711719238110"><a name="p1711719238110"></a><a name="p1711719238110"></a>to_device_npu</p>
</td>
</tr>
<tr id="row17737439171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048312163211"><a name="p1048312163211"></a><a name="p1048312163211"></a>401</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p311714231219"><a name="p311714231219"></a><a name="p311714231219"></a>to.dtype</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p211719237119"><a name="p211719237119"></a><a name="p211719237119"></a>to_dtype_npu</p>
</td>
</tr>
<tr id="row37372039201219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748381616213"><a name="p1748381616213"></a><a name="p1748381616213"></a>402</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4117923417"><a name="p4117923417"></a><a name="p4117923417"></a>to.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5117172315112"><a name="p5117172315112"></a><a name="p5117172315112"></a>to_other_npu</p>
</td>
</tr>
<tr id="row167378397126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2048321613210"><a name="p2048321613210"></a><a name="p2048321613210"></a>403</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1117142317117"><a name="p1117142317117"></a><a name="p1117142317117"></a>_local_scalar_dense</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1111720231819"><a name="p1111720231819"></a><a name="p1111720231819"></a>_local_scalar_dense_npu</p>
</td>
</tr>
<tr id="row10737113912126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14842161621"><a name="p14842161621"></a><a name="p14842161621"></a>404</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1611722319117"><a name="p1611722319117"></a><a name="p1611722319117"></a>lstm.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p911711236117"><a name="p911711236117"></a><a name="p911711236117"></a>lstm_npu</p>
</td>
</tr>
<tr id="row1873733981212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548414164215"><a name="p1548414164215"></a><a name="p1548414164215"></a>405</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151176231513"><a name="p151176231513"></a><a name="p151176231513"></a>lstm.data</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p611718231316"><a name="p611718231316"></a><a name="p611718231316"></a>lstm_npu</p>
</td>
</tr>
<tr id="row673773971216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84847161826"><a name="p84847161826"></a><a name="p84847161826"></a>406</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1411719231112"><a name="p1411719231112"></a><a name="p1411719231112"></a>gru.input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p311782318116"><a name="p311782318116"></a><a name="p311782318116"></a>gru_npu_</p>
</td>
</tr>
<tr id="row4737173910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104846161627"><a name="p104846161627"></a><a name="p104846161627"></a>407</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6117172312112"><a name="p6117172312112"></a><a name="p6117172312112"></a>_pack_padded_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p011718237114"><a name="p011718237114"></a><a name="p011718237114"></a>_pack_padded_sequence_npu</p>
</td>
</tr>
<tr id="row137371539151215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048413161524"><a name="p048413161524"></a><a name="p048413161524"></a>408</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15117132316117"><a name="p15117132316117"></a><a name="p15117132316117"></a>_pad_packed_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181175231211"><a name="p181175231211"></a><a name="p181175231211"></a>_pad_packed_sequence_npu</p>
</td>
</tr>
<tr id="row27371039181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14484121619211"><a name="p14484121619211"></a><a name="p14484121619211"></a>409</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1611752311110"><a name="p1611752311110"></a><a name="p1611752311110"></a>set_.source_Storage</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16117623618"><a name="p16117623618"></a><a name="p16117623618"></a>set_npu_</p>
</td>
</tr>
<tr id="row1173783981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84844163212"><a name="p84844163212"></a><a name="p84844163212"></a>410</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131174237110"><a name="p131174237110"></a><a name="p131174237110"></a>set_.source_Storage_storage_offset</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p911716231715"><a name="p911716231715"></a><a name="p911716231715"></a>set_npu_</p>
</td>
</tr>
<tr id="row673883991219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1548418163214"><a name="p1548418163214"></a><a name="p1548418163214"></a>411</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101171123212"><a name="p101171123212"></a><a name="p101171123212"></a>set_.source_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121175236114"><a name="p121175236114"></a><a name="p121175236114"></a>set_npu_</p>
</td>
</tr>
<tr id="row19738173951218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2484131617213"><a name="p2484131617213"></a><a name="p2484131617213"></a>412</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p201171823619"><a name="p201171823619"></a><a name="p201171823619"></a>set_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81171123519"><a name="p81171123519"></a><a name="p81171123519"></a>set_npu_</p>
</td>
</tr>
<tr id="row27381039101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154841416629"><a name="p154841416629"></a><a name="p154841416629"></a>413</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1411810231415"><a name="p1411810231415"></a><a name="p1411810231415"></a>masked_fill_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11181923317"><a name="p11181923317"></a><a name="p11181923317"></a>masked_fill_npu_</p>
</td>
</tr>
<tr id="row16738939101214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948421616218"><a name="p948421616218"></a><a name="p948421616218"></a>414</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p911815231111"><a name="p911815231111"></a><a name="p911815231111"></a>masked_fill_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141185232110"><a name="p141185232110"></a><a name="p141185232110"></a>masked_fill_npu_</p>
</td>
</tr>
<tr id="row273863917124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p348451610215"><a name="p348451610215"></a><a name="p348451610215"></a>415</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p811814231411"><a name="p811814231411"></a><a name="p811814231411"></a>masked_scatter_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p811822314114"><a name="p811822314114"></a><a name="p811822314114"></a>masked_scatter_npu_</p>
</td>
</tr>
<tr id="row11738183918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84841016325"><a name="p84841016325"></a><a name="p84841016325"></a>416</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13118623718"><a name="p13118623718"></a><a name="p13118623718"></a>view</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5118192317120"><a name="p5118192317120"></a><a name="p5118192317120"></a>view_npu</p>
</td>
</tr>
<tr id="row5738739161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104841316522"><a name="p104841316522"></a><a name="p104841316522"></a>417</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p01180231514"><a name="p01180231514"></a><a name="p01180231514"></a>put_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1911814232012"><a name="p1911814232012"></a><a name="p1911814232012"></a>put_npu_</p>
</td>
</tr>
<tr id="row273811393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24840161020"><a name="p24840161020"></a><a name="p24840161020"></a>418</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1611813237115"><a name="p1611813237115"></a><a name="p1611813237115"></a>index_add_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17118923119"><a name="p17118923119"></a><a name="p17118923119"></a>index_add_npu_</p>
</td>
</tr>
<tr id="row13738739111210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3484101617211"><a name="p3484101617211"></a><a name="p3484101617211"></a>419</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1118102319118"><a name="p1118102319118"></a><a name="p1118102319118"></a>index_add</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1611813231518"><a name="p1611813231518"></a><a name="p1611813231518"></a>index_add_npu</p>
</td>
</tr>
<tr id="row773812399125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5484151613217"><a name="p5484151613217"></a><a name="p5484151613217"></a>420</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p611842312114"><a name="p611842312114"></a><a name="p611842312114"></a>index_add.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7118223512"><a name="p7118223512"></a><a name="p7118223512"></a>index_add_npu</p>
</td>
</tr>
<tr id="row2738123901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64842162216"><a name="p64842162216"></a><a name="p64842162216"></a>421</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111819231617"><a name="p111819231617"></a><a name="p111819231617"></a>index_fill_.int_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1118152320111"><a name="p1118152320111"></a><a name="p1118152320111"></a>index_fill_npu_</p>
</td>
</tr>
<tr id="row27381739161215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124858161217"><a name="p124858161217"></a><a name="p124858161217"></a>422</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p911882315113"><a name="p911882315113"></a><a name="p911882315113"></a>index_fill.int_Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1811817231518"><a name="p1811817231518"></a><a name="p1811817231518"></a>index_fill_npu</p>
</td>
</tr>
<tr id="row197395394127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2485516229"><a name="p2485516229"></a><a name="p2485516229"></a>423</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1111842313113"><a name="p1111842313113"></a><a name="p1111842313113"></a>index_fill_.int_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p411832318119"><a name="p411832318119"></a><a name="p411832318119"></a>index_fill_npu_</p>
</td>
</tr>
<tr id="row97393398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54852161229"><a name="p54852161229"></a><a name="p54852161229"></a>424</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p711810232120"><a name="p711810232120"></a><a name="p711810232120"></a>index_fill.int_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p311818231217"><a name="p311818231217"></a><a name="p311818231217"></a>index_fill_npu</p>
</td>
</tr>
<tr id="row18739193916123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174851116725"><a name="p174851116725"></a><a name="p174851116725"></a>425</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9118162315111"><a name="p9118162315111"></a><a name="p9118162315111"></a>scatter_.src</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p611842310113"><a name="p611842310113"></a><a name="p611842310113"></a>scatter_npu_</p>
</td>
</tr>
<tr id="row19739103918129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5485416227"><a name="p5485416227"></a><a name="p5485416227"></a>426</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p211822318111"><a name="p211822318111"></a><a name="p211822318111"></a>scatter_.value</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p101188231119"><a name="p101188231119"></a><a name="p101188231119"></a>scatter_npu_</p>
</td>
</tr>
<tr id="row157396393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24851316624"><a name="p24851316624"></a><a name="p24851316624"></a>427</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p511814238119"><a name="p511814238119"></a><a name="p511814238119"></a>scatter_add_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16118112310120"><a name="p16118112310120"></a><a name="p16118112310120"></a>scatter_add_npu_</p>
</td>
</tr>
<tr id="row187391139131214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648501620215"><a name="p1648501620215"></a><a name="p1648501620215"></a>428</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p211852314117"><a name="p211852314117"></a><a name="p211852314117"></a>scatter_add</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151181231618"><a name="p151181231618"></a><a name="p151181231618"></a>scatter_add_npu</p>
</td>
</tr>
<tr id="row1273973941211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19485516328"><a name="p19485516328"></a><a name="p19485516328"></a>429</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12118023614"><a name="p12118023614"></a><a name="p12118023614"></a>scatter_add.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p711802319117"><a name="p711802319117"></a><a name="p711802319117"></a>scatter_add_npu</p>
</td>
</tr>
<tr id="row77391439201210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648511163215"><a name="p648511163215"></a><a name="p648511163215"></a>430</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16119152314116"><a name="p16119152314116"></a><a name="p16119152314116"></a>lt_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161199231013"><a name="p161199231013"></a><a name="p161199231013"></a>lt_npu_</p>
</td>
</tr>
<tr id="row3739163911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9485416425"><a name="p9485416425"></a><a name="p9485416425"></a>431</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p511915231513"><a name="p511915231513"></a><a name="p511915231513"></a>lt_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p611918231711"><a name="p611918231711"></a><a name="p611918231711"></a>lt_npu_</p>
</td>
</tr>
<tr id="row37391539141215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12485141616210"><a name="p12485141616210"></a><a name="p12485141616210"></a>432</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p121191423019"><a name="p121191423019"></a><a name="p121191423019"></a>gt_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p211919231511"><a name="p211919231511"></a><a name="p211919231511"></a>gt_npu_</p>
</td>
</tr>
<tr id="row573993941213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44851016222"><a name="p44851016222"></a><a name="p44851016222"></a>433</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151197231818"><a name="p151197231818"></a><a name="p151197231818"></a>gt_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8119112318112"><a name="p8119112318112"></a><a name="p8119112318112"></a>gt_npu_</p>
</td>
</tr>
<tr id="row3740239171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15485141617211"><a name="p15485141617211"></a><a name="p15485141617211"></a>434</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1111902317114"><a name="p1111902317114"></a><a name="p1111902317114"></a>le_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p311916239110"><a name="p311916239110"></a><a name="p311916239110"></a>le_npu_</p>
</td>
</tr>
<tr id="row874013971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94851716128"><a name="p94851716128"></a><a name="p94851716128"></a>435</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1119223513"><a name="p1119223513"></a><a name="p1119223513"></a>le_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p20119923114"><a name="p20119923114"></a><a name="p20119923114"></a>le_npu_</p>
</td>
</tr>
<tr id="row67401395129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3486116926"><a name="p3486116926"></a><a name="p3486116926"></a>436</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1511914231919"><a name="p1511914231919"></a><a name="p1511914231919"></a>ge_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1111942312117"><a name="p1111942312117"></a><a name="p1111942312117"></a>ge_npu_</p>
</td>
</tr>
<tr id="row13740439201212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948613161926"><a name="p948613161926"></a><a name="p948613161926"></a>437</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18119122312112"><a name="p18119122312112"></a><a name="p18119122312112"></a>ge_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1811917233114"><a name="p1811917233114"></a><a name="p1811917233114"></a>ge_npu_</p>
</td>
</tr>
<tr id="row1174013916124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448661614217"><a name="p448661614217"></a><a name="p448661614217"></a>438</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4119162317110"><a name="p4119162317110"></a><a name="p4119162317110"></a>eq_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11191823912"><a name="p11191823912"></a><a name="p11191823912"></a>eq_npu_</p>
</td>
</tr>
<tr id="row774016390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1248616161221"><a name="p1248616161221"></a><a name="p1248616161221"></a>439</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p011992318112"><a name="p011992318112"></a><a name="p011992318112"></a>eq_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p171191523211"><a name="p171191523211"></a><a name="p171191523211"></a>eq_npu_</p>
</td>
</tr>
<tr id="row7740193917126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848620161423"><a name="p1848620161423"></a><a name="p1848620161423"></a>440</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81192230110"><a name="p81192230110"></a><a name="p81192230110"></a>ne_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1111910238115"><a name="p1111910238115"></a><a name="p1111910238115"></a>ne_npu_</p>
</td>
</tr>
<tr id="row67401439171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648691611212"><a name="p1648691611212"></a><a name="p1648691611212"></a>441</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81191223414"><a name="p81191223414"></a><a name="p81191223414"></a>ne_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14119102315112"><a name="p14119102315112"></a><a name="p14119102315112"></a>ne_npu_</p>
</td>
</tr>
<tr id="row1174003910120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194861216228"><a name="p194861216228"></a><a name="p194861216228"></a>442</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141191523315"><a name="p141191523315"></a><a name="p141191523315"></a>bitwise_and.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9119923517"><a name="p9119923517"></a><a name="p9119923517"></a>bitwise_and_out_npu</p>
</td>
</tr>
<tr id="row12740739111211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44861016721"><a name="p44861016721"></a><a name="p44861016721"></a>443</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131190231216"><a name="p131190231216"></a><a name="p131190231216"></a>bitwise_and.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p101201223617"><a name="p101201223617"></a><a name="p101201223617"></a>bitwise_and_out_npu</p>
</td>
</tr>
<tr id="row18740163913121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148616161926"><a name="p1148616161926"></a><a name="p1148616161926"></a>444</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131203232119"><a name="p131203232119"></a><a name="p131203232119"></a>bitwise_and.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p912011231714"><a name="p912011231714"></a><a name="p912011231714"></a>bitwise_and_npu</p>
</td>
</tr>
<tr id="row574163931217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1486816423"><a name="p1486816423"></a><a name="p1486816423"></a>445</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1412020231517"><a name="p1412020231517"></a><a name="p1412020231517"></a>bitwise_and.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151206231613"><a name="p151206231613"></a><a name="p151206231613"></a>bitwise_and_npu</p>
</td>
</tr>
<tr id="row1774114393126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6486111614216"><a name="p6486111614216"></a><a name="p6486111614216"></a>446</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7120152310115"><a name="p7120152310115"></a><a name="p7120152310115"></a>bitwise_and_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12120423118"><a name="p12120423118"></a><a name="p12120423118"></a>bitwise_and_npu_</p>
</td>
</tr>
<tr id="row14741639161211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14869169213"><a name="p14869169213"></a><a name="p14869169213"></a>447</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8120202314113"><a name="p8120202314113"></a><a name="p8120202314113"></a>bitwise_and_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1212011232017"><a name="p1212011232017"></a><a name="p1212011232017"></a>bitwise_and_npu_</p>
</td>
</tr>
<tr id="row0741193911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048613162213"><a name="p048613162213"></a><a name="p048613162213"></a>448</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p712052312116"><a name="p712052312116"></a><a name="p712052312116"></a>__and__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11204231216"><a name="p11204231216"></a><a name="p11204231216"></a>__and___npu</p>
</td>
</tr>
<tr id="row6741839161215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p548619162217"><a name="p548619162217"></a><a name="p548619162217"></a>449</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20120192313113"><a name="p20120192313113"></a><a name="p20120192313113"></a>__and__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1812020236114"><a name="p1812020236114"></a><a name="p1812020236114"></a>__and___npu</p>
</td>
</tr>
<tr id="row9741193991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16486716328"><a name="p16486716328"></a><a name="p16486716328"></a>450</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6120202314120"><a name="p6120202314120"></a><a name="p6120202314120"></a>bitwise_or.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p312019231119"><a name="p312019231119"></a><a name="p312019231119"></a>bitwise_or_out_npu</p>
</td>
</tr>
<tr id="row974103910121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p048613168218"><a name="p048613168218"></a><a name="p048613168218"></a>451</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151206238114"><a name="p151206238114"></a><a name="p151206238114"></a>bitwise_or.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p512010232117"><a name="p512010232117"></a><a name="p512010232117"></a>bitwise_or_out_npu</p>
</td>
</tr>
<tr id="row1741103911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948615169212"><a name="p948615169212"></a><a name="p948615169212"></a>452</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1212019235112"><a name="p1212019235112"></a><a name="p1212019235112"></a>bitwise_or.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p112062313114"><a name="p112062313114"></a><a name="p112062313114"></a>bitwise_or_npu</p>
</td>
</tr>
<tr id="row1674113914126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184861161224"><a name="p184861161224"></a><a name="p184861161224"></a>453</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p912018231116"><a name="p912018231116"></a><a name="p912018231116"></a>bitwise_or.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p612017232120"><a name="p612017232120"></a><a name="p612017232120"></a>bitwise_or_npu</p>
</td>
</tr>
<tr id="row4741839101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174861816527"><a name="p174861816527"></a><a name="p174861816527"></a>454</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p712012311113"><a name="p712012311113"></a><a name="p712012311113"></a>bitwise_or_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1812092316115"><a name="p1812092316115"></a><a name="p1812092316115"></a>bitwise_or_npu_</p>
</td>
</tr>
<tr id="row137421539161220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p948717168215"><a name="p948717168215"></a><a name="p948717168215"></a>455</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p112017239111"><a name="p112017239111"></a><a name="p112017239111"></a>bitwise_or_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1112012318118"><a name="p1112012318118"></a><a name="p1112012318118"></a>bitwise_or_npu_</p>
</td>
</tr>
<tr id="row8742143911218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44871016526"><a name="p44871016526"></a><a name="p44871016526"></a>456</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p312092315118"><a name="p312092315118"></a><a name="p312092315118"></a>__or__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8120923910"><a name="p8120923910"></a><a name="p8120923910"></a>__or___npu</p>
</td>
</tr>
<tr id="row1274263912123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15487716629"><a name="p15487716629"></a><a name="p15487716629"></a>457</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11120122318117"><a name="p11120122318117"></a><a name="p11120122318117"></a>__or__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4120192311111"><a name="p4120192311111"></a><a name="p4120192311111"></a>__or___npu</p>
</td>
</tr>
<tr id="row1374210390127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348719168215"><a name="p1348719168215"></a><a name="p1348719168215"></a>458</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13120132311112"><a name="p13120132311112"></a><a name="p13120132311112"></a>__ior__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p812015231814"><a name="p812015231814"></a><a name="p812015231814"></a>__ior___npu</p>
</td>
</tr>
<tr id="row15742123901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4487316323"><a name="p4487316323"></a><a name="p4487316323"></a>459</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2120523912"><a name="p2120523912"></a><a name="p2120523912"></a>__ior__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p512018231214"><a name="p512018231214"></a><a name="p512018231214"></a>__ior___npu</p>
</td>
</tr>
<tr id="row2742133918128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848781610212"><a name="p1848781610212"></a><a name="p1848781610212"></a>460</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p712117231513"><a name="p712117231513"></a><a name="p712117231513"></a>bitwise_xor.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p712115233120"><a name="p712115233120"></a><a name="p712115233120"></a>bitwise_xor_out_npu</p>
</td>
</tr>
<tr id="row1974273913128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74878161021"><a name="p74878161021"></a><a name="p74878161021"></a>461</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15121142312113"><a name="p15121142312113"></a><a name="p15121142312113"></a>bitwise_xor.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p712118238119"><a name="p712118238119"></a><a name="p712118238119"></a>bitwise_xor_out_npu</p>
</td>
</tr>
<tr id="row274223916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p148717162219"><a name="p148717162219"></a><a name="p148717162219"></a>462</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11211823219"><a name="p11211823219"></a><a name="p11211823219"></a>bitwise_xor.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p012112232011"><a name="p012112232011"></a><a name="p012112232011"></a>bitwise_xor_npu</p>
</td>
</tr>
<tr id="row13742739201212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16487216924"><a name="p16487216924"></a><a name="p16487216924"></a>463</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p312102313116"><a name="p312102313116"></a><a name="p312102313116"></a>bitwise_xor.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p191216232117"><a name="p191216232117"></a><a name="p191216232117"></a>bitwise_xor_npu</p>
</td>
</tr>
<tr id="row3742143941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748719165213"><a name="p1748719165213"></a><a name="p1748719165213"></a>464</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81218238110"><a name="p81218238110"></a><a name="p81218238110"></a>bitwise_xor_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11121123614"><a name="p11121123614"></a><a name="p11121123614"></a>bitwise_xor_npu_</p>
</td>
</tr>
<tr id="row57420390126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348713163210"><a name="p1348713163210"></a><a name="p1348713163210"></a>465</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p412113231010"><a name="p412113231010"></a><a name="p412113231010"></a>bitwise_xor_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61211723611"><a name="p61211723611"></a><a name="p61211723611"></a>bitwise_xor_npu_</p>
</td>
</tr>
<tr id="row197431539141210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18487316622"><a name="p18487316622"></a><a name="p18487316622"></a>466</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61215231911"><a name="p61215231911"></a><a name="p61215231911"></a>__xor__.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4121112319111"><a name="p4121112319111"></a><a name="p4121112319111"></a>__xor___npu</p>
</td>
</tr>
<tr id="row18743173911128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18487116624"><a name="p18487116624"></a><a name="p18487116624"></a>467</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p312117231912"><a name="p312117231912"></a><a name="p312117231912"></a>__xor__.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51211823317"><a name="p51211823317"></a><a name="p51211823317"></a>__xor___npu</p>
</td>
</tr>
<tr id="row15743103916125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14871161324"><a name="p14871161324"></a><a name="p14871161324"></a>468</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1612102319112"><a name="p1612102319112"></a><a name="p1612102319112"></a>atan2_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p912142318118"><a name="p912142318118"></a><a name="p912142318118"></a>atan2_npu_</p>
</td>
</tr>
<tr id="row774363951216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1048711162023"><a name="p1048711162023"></a><a name="p1048711162023"></a>469</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11121923312"><a name="p11121923312"></a><a name="p11121923312"></a>tril_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1012116236117"><a name="p1012116236117"></a><a name="p1012116236117"></a>tril_npu_</p>
</td>
</tr>
<tr id="row47431839151217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13488201612217"><a name="p13488201612217"></a><a name="p13488201612217"></a>470</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111215231817"><a name="p111215231817"></a><a name="p111215231817"></a>triu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111211123413"><a name="p111211123413"></a><a name="p111211123413"></a>triu_npu_</p>
</td>
</tr>
<tr id="row117431739171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44885167216"><a name="p44885167216"></a><a name="p44885167216"></a>471</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12121142320112"><a name="p12121142320112"></a><a name="p12121142320112"></a>renorm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1212111231313"><a name="p1212111231313"></a><a name="p1212111231313"></a>renorm_npu_</p>
</td>
</tr>
<tr id="row14743639201214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p448815161827"><a name="p448815161827"></a><a name="p448815161827"></a>472</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p312114231214"><a name="p312114231214"></a><a name="p312114231214"></a>pow_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1812122317113"><a name="p1812122317113"></a><a name="p1812122317113"></a>pow_npu_</p>
</td>
</tr>
<tr id="row16743183921211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8488131612215"><a name="p8488131612215"></a><a name="p8488131612215"></a>473</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141211923918"><a name="p141211923918"></a><a name="p141211923918"></a>pow_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p112110231216"><a name="p112110231216"></a><a name="p112110231216"></a>pow_npu_</p>
</td>
</tr>
<tr id="row4743103913128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1648841617216"><a name="p1648841617216"></a><a name="p1648841617216"></a>474</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10121172313113"><a name="p10121172313113"></a><a name="p10121172313113"></a>lerp_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71219231116"><a name="p71219231116"></a><a name="p71219231116"></a>lerp_npu_</p>
</td>
</tr>
<tr id="row137431039181217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124884161627"><a name="p124884161627"></a><a name="p124884161627"></a>475</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p712192311116"><a name="p712192311116"></a><a name="p712192311116"></a>lerp_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p812112315116"><a name="p812112315116"></a><a name="p812112315116"></a>lerp_npu_</p>
</td>
</tr>
<tr id="row674333911217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12488111614211"><a name="p12488111614211"></a><a name="p12488111614211"></a>476</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41217231811"><a name="p41217231811"></a><a name="p41217231811"></a>fmod_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9121623113"><a name="p9121623113"></a><a name="p9121623113"></a>fmod_npu_</p>
</td>
</tr>
<tr id="row1374313971215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9488191612213"><a name="p9488191612213"></a><a name="p9488191612213"></a>477</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p112217231819"><a name="p112217231819"></a><a name="p112217231819"></a>fmod_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4122223418"><a name="p4122223418"></a><a name="p4122223418"></a>fmod_npu_</p>
</td>
</tr>
<tr id="row9744133919120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144881516723"><a name="p144881516723"></a><a name="p144881516723"></a>478</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101221423917"><a name="p101221423917"></a><a name="p101221423917"></a>remainder_.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21224230110"><a name="p21224230110"></a><a name="p21224230110"></a>remainder_npu_</p>
</td>
</tr>
<tr id="row37441439121211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348831619216"><a name="p1348831619216"></a><a name="p1348831619216"></a>479</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51225231014"><a name="p51225231014"></a><a name="p51225231014"></a>remainder_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21221423217"><a name="p21221423217"></a><a name="p21221423217"></a>remainder_npu_</p>
</td>
</tr>
<tr id="row1774413961216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15488716822"><a name="p15488716822"></a><a name="p15488716822"></a>480</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p181223231113"><a name="p181223231113"></a><a name="p181223231113"></a>addbmm_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2122723114"><a name="p2122723114"></a><a name="p2122723114"></a>addbmm_npu_</p>
</td>
</tr>
<tr id="row12744153911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1848811161924"><a name="p1848811161924"></a><a name="p1848811161924"></a>481</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7122823815"><a name="p7122823815"></a><a name="p7122823815"></a>addbmm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131222230111"><a name="p131222230111"></a><a name="p131222230111"></a>addbmm_out_npu</p>
</td>
</tr>
<tr id="row197441439141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748814161327"><a name="p1748814161327"></a><a name="p1748814161327"></a>482</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1312219231519"><a name="p1312219231519"></a><a name="p1312219231519"></a>addbmm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18122323217"><a name="p18122323217"></a><a name="p18122323217"></a>addbmm_npu</p>
</td>
</tr>
<tr id="row77441939151220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194883161223"><a name="p194883161223"></a><a name="p194883161223"></a>483</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1812212311118"><a name="p1812212311118"></a><a name="p1812212311118"></a>addcdiv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p712218238119"><a name="p712218238119"></a><a name="p712218238119"></a>addcdiv_npu_</p>
</td>
</tr>
<tr id="row10744193918120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1488111619216"><a name="p1488111619216"></a><a name="p1488111619216"></a>484</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1412212234111"><a name="p1412212234111"></a><a name="p1412212234111"></a>random_.from</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13122142316111"><a name="p13122142316111"></a><a name="p13122142316111"></a>random_npu_</p>
</td>
</tr>
<tr id="row574493911210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4489316329"><a name="p4489316329"></a><a name="p4489316329"></a>485</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8122323613"><a name="p8122323613"></a><a name="p8122323613"></a>random_.to</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2122112313115"><a name="p2122112313115"></a><a name="p2122112313115"></a>random_npu_</p>
</td>
</tr>
<tr id="row4744123901217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1148981611212"><a name="p1148981611212"></a><a name="p1148981611212"></a>486</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p612222318112"><a name="p612222318112"></a><a name="p612222318112"></a>random_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1212222311116"><a name="p1212222311116"></a><a name="p1212222311116"></a>random_npu_</p>
</td>
</tr>
<tr id="row874423915126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848915161423"><a name="p848915161423"></a><a name="p848915161423"></a>487</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91222231913"><a name="p91222231913"></a><a name="p91222231913"></a>uniform_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p212217231518"><a name="p212217231518"></a><a name="p212217231518"></a>uniform_npu_</p>
</td>
</tr>
<tr id="row1774411397122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114898164213"><a name="p114898164213"></a><a name="p114898164213"></a>488</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1212217231318"><a name="p1212217231318"></a><a name="p1212217231318"></a>diag.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1912217234115"><a name="p1912217234115"></a><a name="p1912217234115"></a>diag_out_npu</p>
</td>
</tr>
<tr id="row1074423913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04891916228"><a name="p04891916228"></a><a name="p04891916228"></a>489</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1512215231115"><a name="p1512215231115"></a><a name="p1512215231115"></a>diag</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121221523416"><a name="p121221523416"></a><a name="p121221523416"></a>diag_npu</p>
</td>
</tr>
<tr id="row14745239101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184895164211"><a name="p184895164211"></a><a name="p184895164211"></a>490</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191224238112"><a name="p191224238112"></a><a name="p191224238112"></a>cross.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1612212230111"><a name="p1612212230111"></a><a name="p1612212230111"></a>cross_out_npu</p>
</td>
</tr>
<tr id="row67451339151211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1348901615217"><a name="p1348901615217"></a><a name="p1348901615217"></a>491</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6122112315111"><a name="p6122112315111"></a><a name="p6122112315111"></a>cross</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5122152316118"><a name="p5122152316118"></a><a name="p5122152316118"></a>cross_npu</p>
</td>
</tr>
<tr id="row5745239191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5489316928"><a name="p5489316928"></a><a name="p5489316928"></a>492</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10122623819"><a name="p10122623819"></a><a name="p10122623819"></a>triu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71220231317"><a name="p71220231317"></a><a name="p71220231317"></a>triu_out_npu</p>
</td>
</tr>
<tr id="row174516395120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94891716826"><a name="p94891716826"></a><a name="p94891716826"></a>493</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131226236116"><a name="p131226236116"></a><a name="p131226236116"></a>triu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17123623612"><a name="p17123623612"></a><a name="p17123623612"></a>triu_npu</p>
</td>
</tr>
<tr id="row1174518399127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1748912166214"><a name="p1748912166214"></a><a name="p1748912166214"></a>494</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p201231823914"><a name="p201231823914"></a><a name="p201231823914"></a>tril.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12123723614"><a name="p12123723614"></a><a name="p12123723614"></a>tril_out_npu</p>
</td>
</tr>
<tr id="row16745123913122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p848911612218"><a name="p848911612218"></a><a name="p848911612218"></a>495</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1312312311111"><a name="p1312312311111"></a><a name="p1312312311111"></a>tril</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1312362318114"><a name="p1312362318114"></a><a name="p1312362318114"></a>tril_npu</p>
</td>
</tr>
<tr id="row197450393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p748915161220"><a name="p748915161220"></a><a name="p748915161220"></a>496</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151235231711"><a name="p151235231711"></a><a name="p151235231711"></a>ne.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1012372320119"><a name="p1012372320119"></a><a name="p1012372320119"></a>ne_out_npu</p>
</td>
</tr>
<tr id="row11745739111217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p648914163218"><a name="p648914163218"></a><a name="p648914163218"></a>497</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p912317232013"><a name="p912317232013"></a><a name="p912317232013"></a>ne.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12123162316113"><a name="p12123162316113"></a><a name="p12123162316113"></a>ne_npu</p>
</td>
</tr>
<tr id="row177451439181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134901116021"><a name="p134901116021"></a><a name="p134901116021"></a>498</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p712310237116"><a name="p712310237116"></a><a name="p712310237116"></a>ne.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p912313231613"><a name="p912313231613"></a><a name="p912313231613"></a>ne_out_npu</p>
</td>
</tr>
<tr id="row13745143981210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174903162215"><a name="p174903162215"></a><a name="p174903162215"></a>499</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161230232110"><a name="p161230232110"></a><a name="p161230232110"></a>ne.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10123523410"><a name="p10123523410"></a><a name="p10123523410"></a>ne_npu</p>
</td>
</tr>
<tr id="row9745103914127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2490216726"><a name="p2490216726"></a><a name="p2490216726"></a>500</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17123323911"><a name="p17123323911"></a><a name="p17123323911"></a>eq.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151235230118"><a name="p151235230118"></a><a name="p151235230118"></a>eq_out_npu</p>
</td>
</tr>
<tr id="row1074693914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194901716527"><a name="p194901716527"></a><a name="p194901716527"></a>501</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p412311231917"><a name="p412311231917"></a><a name="p412311231917"></a>eq.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p112362318113"><a name="p112362318113"></a><a name="p112362318113"></a>eq_npu</p>
</td>
</tr>
<tr id="row12746103917120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24908161421"><a name="p24908161421"></a><a name="p24908161421"></a>502</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p312315233113"><a name="p312315233113"></a><a name="p312315233113"></a>eq.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31232238114"><a name="p31232238114"></a><a name="p31232238114"></a>eq_out_npu</p>
</td>
</tr>
<tr id="row1474623981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p64901216329"><a name="p64901216329"></a><a name="p64901216329"></a>503</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1123122312113"><a name="p1123122312113"></a><a name="p1123122312113"></a>eq.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151231223418"><a name="p151231223418"></a><a name="p151231223418"></a>eq_npu</p>
</td>
</tr>
<tr id="row2074613920121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2490116325"><a name="p2490116325"></a><a name="p2490116325"></a>504</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1812312236118"><a name="p1812312236118"></a><a name="p1812312236118"></a>ge.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1412392317120"><a name="p1412392317120"></a><a name="p1412392317120"></a>ge_out_npu</p>
</td>
</tr>
<tr id="row16746839101217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049021616210"><a name="p1049021616210"></a><a name="p1049021616210"></a>505</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p712392319113"><a name="p712392319113"></a><a name="p712392319113"></a>ge.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15123142319117"><a name="p15123142319117"></a><a name="p15123142319117"></a>ge_npu</p>
</td>
</tr>
<tr id="row1674643912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74914162213"><a name="p74914162213"></a><a name="p74914162213"></a>506</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p01231231317"><a name="p01231231317"></a><a name="p01231231317"></a>ge.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111237236115"><a name="p111237236115"></a><a name="p111237236115"></a>ge_out_npu</p>
</td>
</tr>
<tr id="row774653921210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049119161224"><a name="p1049119161224"></a><a name="p1049119161224"></a>507</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13123723212"><a name="p13123723212"></a><a name="p13123723212"></a>ge.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41235237117"><a name="p41235237117"></a><a name="p41235237117"></a>ge_npu</p>
</td>
</tr>
<tr id="row0746339141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2491916325"><a name="p2491916325"></a><a name="p2491916325"></a>508</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1712318231111"><a name="p1712318231111"></a><a name="p1712318231111"></a>le.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1112322318118"><a name="p1112322318118"></a><a name="p1112322318118"></a>le_out_npu</p>
</td>
</tr>
<tr id="row6748143914128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6491101613213"><a name="p6491101613213"></a><a name="p6491101613213"></a>509</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81232023411"><a name="p81232023411"></a><a name="p81232023411"></a>le.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71237231311"><a name="p71237231311"></a><a name="p71237231311"></a>le_npu</p>
</td>
</tr>
<tr id="row67489392120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84911416126"><a name="p84911416126"></a><a name="p84911416126"></a>510</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20124152316115"><a name="p20124152316115"></a><a name="p20124152316115"></a>le.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1912415231515"><a name="p1912415231515"></a><a name="p1912415231515"></a>le_out_npu</p>
</td>
</tr>
<tr id="row5748203971215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6491171620217"><a name="p6491171620217"></a><a name="p6491171620217"></a>511</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1312432313111"><a name="p1312432313111"></a><a name="p1312432313111"></a>le.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1412482319116"><a name="p1412482319116"></a><a name="p1412482319116"></a>le_npu</p>
</td>
</tr>
<tr id="row774883921211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649171619216"><a name="p1649171619216"></a><a name="p1649171619216"></a>512</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2124132310113"><a name="p2124132310113"></a><a name="p2124132310113"></a>gt.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1812432314115"><a name="p1812432314115"></a><a name="p1812432314115"></a>gt_out_npu</p>
</td>
</tr>
<tr id="row17748203901216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184913162217"><a name="p184913162217"></a><a name="p184913162217"></a>513</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11246236112"><a name="p11246236112"></a><a name="p11246236112"></a>gt.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p112402317112"><a name="p112402317112"></a><a name="p112402317112"></a>gt_npu</p>
</td>
</tr>
<tr id="row147481539151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94912161728"><a name="p94912161728"></a><a name="p94912161728"></a>514</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p312415233119"><a name="p312415233119"></a><a name="p312415233119"></a>gt.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10124172313113"><a name="p10124172313113"></a><a name="p10124172313113"></a>gt_out_npu</p>
</td>
</tr>
<tr id="row177481139191214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349117161621"><a name="p349117161621"></a><a name="p349117161621"></a>515</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17124223015"><a name="p17124223015"></a><a name="p17124223015"></a>gt.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1812413236113"><a name="p1812413236113"></a><a name="p1812413236113"></a>gt_npu</p>
</td>
</tr>
<tr id="row87480397120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184911316522"><a name="p184911316522"></a><a name="p184911316522"></a>516</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1112414231214"><a name="p1112414231214"></a><a name="p1112414231214"></a>lt.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141242231119"><a name="p141242231119"></a><a name="p141242231119"></a>lt_out_npu</p>
</td>
</tr>
<tr id="row7748163971216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10491141612217"><a name="p10491141612217"></a><a name="p10491141612217"></a>517</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8124123114"><a name="p8124123114"></a><a name="p8124123114"></a>lt.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1612420234120"><a name="p1612420234120"></a><a name="p1612420234120"></a>lt_npu</p>
</td>
</tr>
<tr id="row0748239151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1491111611219"><a name="p1491111611219"></a><a name="p1491111611219"></a>518</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p181241223113"><a name="p181241223113"></a><a name="p181241223113"></a>lt.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1712418231312"><a name="p1712418231312"></a><a name="p1712418231312"></a>lt_out_npu</p>
</td>
</tr>
<tr id="row12748133913129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149112161722"><a name="p1149112161722"></a><a name="p1149112161722"></a>519</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p201241323117"><a name="p201241323117"></a><a name="p201241323117"></a>lt.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p812417231912"><a name="p812417231912"></a><a name="p812417231912"></a>lt_npu</p>
</td>
</tr>
<tr id="row1474915397124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74921816428"><a name="p74921816428"></a><a name="p74921816428"></a>520</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81243232115"><a name="p81243232115"></a><a name="p81243232115"></a>take.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3124623414"><a name="p3124623414"></a><a name="p3124623414"></a>take_out_npu</p>
</td>
</tr>
<tr id="row18749153921216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649217161928"><a name="p649217161928"></a><a name="p649217161928"></a>521</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p121244231417"><a name="p121244231417"></a><a name="p121244231417"></a>take</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17124623818"><a name="p17124623818"></a><a name="p17124623818"></a>take_npu</p>
</td>
</tr>
<tr id="row1674923991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54922161211"><a name="p54922161211"></a><a name="p54922161211"></a>522</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1124323413"><a name="p1124323413"></a><a name="p1124323413"></a>index_select.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1312413232014"><a name="p1312413232014"></a><a name="p1312413232014"></a>index_select_out_npu</p>
</td>
</tr>
<tr id="row12749153919120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84922161524"><a name="p84922161524"></a><a name="p84922161524"></a>523</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21241123315"><a name="p21241123315"></a><a name="p21241123315"></a>index_select</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p312414231916"><a name="p312414231916"></a><a name="p312414231916"></a>index_select_npu</p>
</td>
</tr>
<tr id="row12749339171214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14492916124"><a name="p14492916124"></a><a name="p14492916124"></a>524</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141241923319"><a name="p141241923319"></a><a name="p141241923319"></a>index_select.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7124192317114"><a name="p7124192317114"></a><a name="p7124192317114"></a>index_select_out_npu</p>
</td>
</tr>
<tr id="row1749193913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8492171612220"><a name="p8492171612220"></a><a name="p8492171612220"></a>525</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61248231814"><a name="p61248231814"></a><a name="p61248231814"></a>index_select.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12124623816"><a name="p12124623816"></a><a name="p12124623816"></a>index_select_npu</p>
</td>
</tr>
<tr id="row1474913393124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1449251616216"><a name="p1449251616216"></a><a name="p1449251616216"></a>526</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17124623918"><a name="p17124623918"></a><a name="p17124623918"></a>masked_select.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12125112315110"><a name="p12125112315110"></a><a name="p12125112315110"></a>masked_select_out_npu</p>
</td>
</tr>
<tr id="row87491639201213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10492181619219"><a name="p10492181619219"></a><a name="p10492181619219"></a>527</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131251623511"><a name="p131251623511"></a><a name="p131251623511"></a>masked_select</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1212511232117"><a name="p1212511232117"></a><a name="p1212511232117"></a>masked_select_npu</p>
</td>
</tr>
<tr id="row0749113919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24922161127"><a name="p24922161127"></a><a name="p24922161127"></a>528</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1912513231114"><a name="p1912513231114"></a><a name="p1912513231114"></a>nonzero.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2012515238115"><a name="p2012515238115"></a><a name="p2012515238115"></a>nonzero_out_npu</p>
</td>
</tr>
<tr id="row374933913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2492816122"><a name="p2492816122"></a><a name="p2492816122"></a>529</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p912512231119"><a name="p912512231119"></a><a name="p912512231119"></a>nonzero</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p91251231118"><a name="p91251231118"></a><a name="p91251231118"></a>nonzero_npu</p>
</td>
</tr>
<tr id="row3749339111220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p94924165210"><a name="p94924165210"></a><a name="p94924165210"></a>530</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17125182316119"><a name="p17125182316119"></a><a name="p17125182316119"></a>gather.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1812512310116"><a name="p1812512310116"></a><a name="p1812512310116"></a>gather_out_npu</p>
</td>
</tr>
<tr id="row117501939131219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194924161022"><a name="p194924161022"></a><a name="p194924161022"></a>531</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111257231612"><a name="p111257231612"></a><a name="p111257231612"></a>gather</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p171258232120"><a name="p171258232120"></a><a name="p171258232120"></a>gather_npu</p>
</td>
</tr>
<tr id="row13750123914124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649217161124"><a name="p1649217161124"></a><a name="p1649217161124"></a>532</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171253231212"><a name="p171253231212"></a><a name="p171253231212"></a>gather.dimname_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11257231119"><a name="p11257231119"></a><a name="p11257231119"></a>gather_out_npu</p>
</td>
</tr>
<tr id="row47504399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114923161120"><a name="p114923161120"></a><a name="p114923161120"></a>533</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51253236113"><a name="p51253236113"></a><a name="p51253236113"></a>gather.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141252023815"><a name="p141252023815"></a><a name="p141252023815"></a>gather_npu</p>
</td>
</tr>
<tr id="row0750163971213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p24921016328"><a name="p24921016328"></a><a name="p24921016328"></a>534</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151254232010"><a name="p151254232010"></a><a name="p151254232010"></a>addcmul.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51254231214"><a name="p51254231214"></a><a name="p51254231214"></a>addcmul_out_npu</p>
</td>
</tr>
<tr id="row07509395124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949213161621"><a name="p949213161621"></a><a name="p949213161621"></a>535</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1412522317115"><a name="p1412522317115"></a><a name="p1412522317115"></a>addcmul</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151250237116"><a name="p151250237116"></a><a name="p151250237116"></a>addcmul_npu</p>
</td>
</tr>
<tr id="row197501839181212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p154921016224"><a name="p154921016224"></a><a name="p154921016224"></a>536</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1812572316117"><a name="p1812572316117"></a><a name="p1812572316117"></a>addcmul_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p112518231517"><a name="p112518231517"></a><a name="p112518231517"></a>addcmul_npu_</p>
</td>
</tr>
<tr id="row1075017392123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18492181618220"><a name="p18492181618220"></a><a name="p18492181618220"></a>537</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1112562310110"><a name="p1112562310110"></a><a name="p1112562310110"></a>addcdiv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12125152310115"><a name="p12125152310115"></a><a name="p12125152310115"></a>addcdiv_out_npu</p>
</td>
</tr>
<tr id="row1375017398124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184927163211"><a name="p184927163211"></a><a name="p184927163211"></a>538</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p712513231216"><a name="p712513231216"></a><a name="p712513231216"></a>addcdiv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181252237116"><a name="p181252237116"></a><a name="p181252237116"></a>addcdiv_npu</p>
</td>
</tr>
<tr id="row0750739111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249316167213"><a name="p249316167213"></a><a name="p249316167213"></a>539</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131254231515"><a name="p131254231515"></a><a name="p131254231515"></a>qr.Q</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1912518231413"><a name="p1912518231413"></a><a name="p1912518231413"></a>qr_out_npu</p>
</td>
</tr>
<tr id="row13750939151216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949317161626"><a name="p949317161626"></a><a name="p949317161626"></a>540</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111251323617"><a name="p111251323617"></a><a name="p111251323617"></a>qr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p91251223512"><a name="p91251223512"></a><a name="p91251223512"></a>qr_npu</p>
</td>
</tr>
<tr id="row1175014398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20493171615215"><a name="p20493171615215"></a><a name="p20493171615215"></a>541</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p112572311115"><a name="p112572311115"></a><a name="p112572311115"></a>multinomial.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p912516231317"><a name="p912516231317"></a><a name="p912516231317"></a>multinomial_out_npu</p>
</td>
</tr>
<tr id="row2751163911122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8493111611212"><a name="p8493111611212"></a><a name="p8493111611212"></a>542</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p512513231916"><a name="p512513231916"></a><a name="p512513231916"></a>multinomial</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21251023218"><a name="p21251023218"></a><a name="p21251023218"></a>multinomial_npu</p>
</td>
</tr>
<tr id="row4751113917122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849312167218"><a name="p849312167218"></a><a name="p849312167218"></a>543</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61256239120"><a name="p61256239120"></a><a name="p61256239120"></a>erfinv</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121267231514"><a name="p121267231514"></a><a name="p121267231514"></a>erfinv_npu</p>
</td>
</tr>
<tr id="row1875113398121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049315161020"><a name="p1049315161020"></a><a name="p1049315161020"></a>544</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1912615232018"><a name="p1912615232018"></a><a name="p1912615232018"></a>erfinv_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p612620231113"><a name="p612620231113"></a><a name="p612620231113"></a>erfinv_npu_</p>
</td>
</tr>
<tr id="row117511339141211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20493141612212"><a name="p20493141612212"></a><a name="p20493141612212"></a>545</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p312622310112"><a name="p312622310112"></a><a name="p312622310112"></a>erfinv.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16126122319114"><a name="p16126122319114"></a><a name="p16126122319114"></a>erfinv_out_npu</p>
</td>
</tr>
<tr id="row47513398125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049321619219"><a name="p1049321619219"></a><a name="p1049321619219"></a>546</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101261231213"><a name="p101261231213"></a><a name="p101261231213"></a>sign</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p912602319111"><a name="p912602319111"></a><a name="p912602319111"></a>sign_npu</p>
</td>
</tr>
<tr id="row177517395121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649311161213"><a name="p649311161213"></a><a name="p649311161213"></a>547</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3126023219"><a name="p3126023219"></a><a name="p3126023219"></a>sign_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15126142315120"><a name="p15126142315120"></a><a name="p15126142315120"></a>sign_npu_</p>
</td>
</tr>
<tr id="row2751193991210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15493016624"><a name="p15493016624"></a><a name="p15493016624"></a>548</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1312652314114"><a name="p1312652314114"></a><a name="p1312652314114"></a>sign.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1912618231115"><a name="p1912618231115"></a><a name="p1912618231115"></a>sign_out_npu</p>
</td>
</tr>
<tr id="row18751203991216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5493191618213"><a name="p5493191618213"></a><a name="p5493191618213"></a>549</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1126192319111"><a name="p1126192319111"></a><a name="p1126192319111"></a>atan2.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21261423415"><a name="p21261423415"></a><a name="p21261423415"></a>atan2_out_npu</p>
</td>
</tr>
<tr id="row875123941218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249316161223"><a name="p1249316161223"></a><a name="p1249316161223"></a>550</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41261023314"><a name="p41261023314"></a><a name="p41261023314"></a>atan2</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5126102312110"><a name="p5126102312110"></a><a name="p5126102312110"></a>atan2_npu</p>
</td>
</tr>
<tr id="row275114391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049314161527"><a name="p2049314161527"></a><a name="p2049314161527"></a>551</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10126132315120"><a name="p10126132315120"></a><a name="p10126132315120"></a>lerp.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p612672310119"><a name="p612672310119"></a><a name="p612672310119"></a>lerp_out_npu</p>
</td>
</tr>
<tr id="row17751123961212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17493916625"><a name="p17493916625"></a><a name="p17493916625"></a>552</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10126623819"><a name="p10126623819"></a><a name="p10126623819"></a>lerp.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3126823015"><a name="p3126823015"></a><a name="p3126823015"></a>lerp_out_npu</p>
</td>
</tr>
<tr id="row1475113393127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10493111614214"><a name="p10493111614214"></a><a name="p10493111614214"></a>553</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61260231117"><a name="p61260231117"></a><a name="p61260231117"></a>lerp.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6126142313113"><a name="p6126142313113"></a><a name="p6126142313113"></a>lerp_npu</p>
</td>
</tr>
<tr id="row1752839141213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104941016626"><a name="p104941016626"></a><a name="p104941016626"></a>554</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p181267231218"><a name="p181267231218"></a><a name="p181267231218"></a>lerp.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151269231412"><a name="p151269231412"></a><a name="p151269231412"></a>lerp_npu</p>
</td>
</tr>
<tr id="row775233991216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1494191618216"><a name="p1494191618216"></a><a name="p1494191618216"></a>555</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14126102317119"><a name="p14126102317119"></a><a name="p14126102317119"></a>histc.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141266231717"><a name="p141266231717"></a><a name="p141266231717"></a>histc_out_npu</p>
</td>
</tr>
<tr id="row3752183961218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p134941616122"><a name="p134941616122"></a><a name="p134941616122"></a>556</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18126223013"><a name="p18126223013"></a><a name="p18126223013"></a>histc</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p912612233115"><a name="p912612233115"></a><a name="p912612233115"></a>histc_npu</p>
</td>
</tr>
<tr id="row1275293918124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1549441619220"><a name="p1549441619220"></a><a name="p1549441619220"></a>557</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131267231614"><a name="p131267231614"></a><a name="p131267231614"></a>fmod.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121261023213"><a name="p121261023213"></a><a name="p121261023213"></a>fmod_out_npu</p>
</td>
</tr>
<tr id="row1275214396124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194941161125"><a name="p194941161125"></a><a name="p194941161125"></a>558</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p612642319114"><a name="p612642319114"></a><a name="p612642319114"></a>fmod.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8126162314110"><a name="p8126162314110"></a><a name="p8126162314110"></a>fmod_npu</p>
</td>
</tr>
<tr id="row1275223951213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13494151614215"><a name="p13494151614215"></a><a name="p13494151614215"></a>559</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161266235119"><a name="p161266235119"></a><a name="p161266235119"></a>fmod.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7126823814"><a name="p7126823814"></a><a name="p7126823814"></a>fmod_out_npu</p>
</td>
</tr>
<tr id="row1475273915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114944161621"><a name="p114944161621"></a><a name="p114944161621"></a>560</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1412752311110"><a name="p1412752311110"></a><a name="p1412752311110"></a>fmod.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p612782318114"><a name="p612782318114"></a><a name="p612782318114"></a>fmod_npu</p>
</td>
</tr>
<tr id="row1575273961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449481619218"><a name="p449481619218"></a><a name="p449481619218"></a>561</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1127723114"><a name="p1127723114"></a><a name="p1127723114"></a>remainder.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2127152315117"><a name="p2127152315117"></a><a name="p2127152315117"></a>remainder_out_npu</p>
</td>
</tr>
<tr id="row275283919128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74941016925"><a name="p74941016925"></a><a name="p74941016925"></a>562</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10127623914"><a name="p10127623914"></a><a name="p10127623914"></a>remainder.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p01271223919"><a name="p01271223919"></a><a name="p01271223919"></a>remainder_npu</p>
</td>
</tr>
<tr id="row15752163931219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249419161427"><a name="p249419161427"></a><a name="p249419161427"></a>563</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17127102315117"><a name="p17127102315117"></a><a name="p17127102315117"></a>remainder.Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1312710236118"><a name="p1312710236118"></a><a name="p1312710236118"></a>remainder_out_npu</p>
</td>
</tr>
<tr id="row15752113921216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p74941916727"><a name="p74941916727"></a><a name="p74941916727"></a>564</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171277232111"><a name="p171277232111"></a><a name="p171277232111"></a>remainder.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8127523115"><a name="p8127523115"></a><a name="p8127523115"></a>remainder_npu</p>
</td>
</tr>
<tr id="row375343912129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184947161222"><a name="p184947161222"></a><a name="p184947161222"></a>565</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p712732315112"><a name="p712732315112"></a><a name="p712732315112"></a>min.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1112712317117"><a name="p1112712317117"></a><a name="p1112712317117"></a>min_out_npu</p>
</td>
</tr>
<tr id="row9753203991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149414160218"><a name="p149414160218"></a><a name="p149414160218"></a>566</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91279231819"><a name="p91279231819"></a><a name="p91279231819"></a>min.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17127152311115"><a name="p17127152311115"></a><a name="p17127152311115"></a>min_npu</p>
</td>
</tr>
<tr id="row575313910120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249551610212"><a name="p1249551610212"></a><a name="p1249551610212"></a>567</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6127192320117"><a name="p6127192320117"></a><a name="p6127192320117"></a>min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1012718230116"><a name="p1012718230116"></a><a name="p1012718230116"></a>min_npu</p>
</td>
</tr>
<tr id="row1675311393122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149510169212"><a name="p1149510169212"></a><a name="p1149510169212"></a>568</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1812720232012"><a name="p1812720232012"></a><a name="p1812720232012"></a>max.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81278232115"><a name="p81278232115"></a><a name="p81278232115"></a>max_out_npu</p>
</td>
</tr>
<tr id="row12753193981217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149514161324"><a name="p1149514161324"></a><a name="p1149514161324"></a>569</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p612752318115"><a name="p612752318115"></a><a name="p612752318115"></a>max.other</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1612710233113"><a name="p1612710233113"></a><a name="p1612710233113"></a>max_npu</p>
</td>
</tr>
<tr id="row27537391124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849591611212"><a name="p849591611212"></a><a name="p849591611212"></a>570</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31272232117"><a name="p31272232117"></a><a name="p31272232117"></a>max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18127142319116"><a name="p18127142319116"></a><a name="p18127142319116"></a>max_npu</p>
</td>
</tr>
<tr id="row1753153911214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194950161529"><a name="p194950161529"></a><a name="p194950161529"></a>571</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8127162318117"><a name="p8127162318117"></a><a name="p8127162318117"></a>median</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1612792320119"><a name="p1612792320119"></a><a name="p1612792320119"></a>median_npu</p>
</td>
</tr>
<tr id="row8753163971217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949518161428"><a name="p949518161428"></a><a name="p949518161428"></a>572</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61271231112"><a name="p61271231112"></a><a name="p61271231112"></a>sort.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p201276233111"><a name="p201276233111"></a><a name="p201276233111"></a>sort_out_npu</p>
</td>
</tr>
<tr id="row1875323910126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p184951516925"><a name="p184951516925"></a><a name="p184951516925"></a>573</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141272023214"><a name="p141272023214"></a><a name="p141272023214"></a>sort</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11271923013"><a name="p11271923013"></a><a name="p11271923013"></a>sort_npu</p>
</td>
</tr>
<tr id="row1775333911120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349512168210"><a name="p349512168210"></a><a name="p349512168210"></a>574</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1312782310110"><a name="p1312782310110"></a><a name="p1312782310110"></a>sort.dimname_values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p3127192310112"><a name="p3127192310112"></a><a name="p3127192310112"></a>sort_out_npu</p>
</td>
</tr>
<tr id="row10754139131211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4495141617213"><a name="p4495141617213"></a><a name="p4495141617213"></a>575</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111273231110"><a name="p111273231110"></a><a name="p111273231110"></a>sort.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p512715231016"><a name="p512715231016"></a><a name="p512715231016"></a>sort_npu</p>
</td>
</tr>
<tr id="row127541139151210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149581615211"><a name="p149581615211"></a><a name="p149581615211"></a>576</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p112711231113"><a name="p112711231113"></a><a name="p112711231113"></a>argsort</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p112712310118"><a name="p112712310118"></a><a name="p112712310118"></a>argsort_npu</p>
</td>
</tr>
<tr id="row17754113913120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144956165214"><a name="p144956165214"></a><a name="p144956165214"></a>577</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1412815231010"><a name="p1412815231010"></a><a name="p1412815231010"></a>argsort.dimname</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11284231416"><a name="p11284231416"></a><a name="p11284231416"></a>argsort_npu</p>
</td>
</tr>
<tr id="row13754639171213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p04954161425"><a name="p04954161425"></a><a name="p04954161425"></a>578</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p912802319113"><a name="p912802319113"></a><a name="p912802319113"></a>topk.values</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p01281723018"><a name="p01281723018"></a><a name="p01281723018"></a>topk_out_npu</p>
</td>
</tr>
<tr id="row18754143914122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44951416822"><a name="p44951416822"></a><a name="p44951416822"></a>579</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111289231911"><a name="p111289231911"></a><a name="p111289231911"></a>topk</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p12128132313111"><a name="p12128132313111"></a><a name="p12128132313111"></a>topk_npu</p>
</td>
</tr>
<tr id="row1675463991214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p104953161219"><a name="p104953161219"></a><a name="p104953161219"></a>580</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14128723112"><a name="p14128723112"></a><a name="p14128723112"></a>all</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1812818231115"><a name="p1812818231115"></a><a name="p1812818231115"></a>all_npu</p>
</td>
</tr>
<tr id="row575410392129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13495131616220"><a name="p13495131616220"></a><a name="p13495131616220"></a>581</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91284233112"><a name="p91284233112"></a><a name="p91284233112"></a>any</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1412812231012"><a name="p1412812231012"></a><a name="p1412812231012"></a>any_npu</p>
</td>
</tr>
<tr id="row157541139121219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164951160219"><a name="p164951160219"></a><a name="p164951160219"></a>582</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91287234113"><a name="p91287234113"></a><a name="p91287234113"></a>renorm.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p191281123716"><a name="p191281123716"></a><a name="p191281123716"></a>renorm_out_npu</p>
</td>
</tr>
<tr id="row1275423991213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1749516161921"><a name="p1749516161921"></a><a name="p1749516161921"></a>583</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1612818231415"><a name="p1612818231415"></a><a name="p1612818231415"></a>renorm</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151284231213"><a name="p151284231213"></a><a name="p151284231213"></a>renorm_npu</p>
</td>
</tr>
<tr id="row5754193914121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2495161616215"><a name="p2495161616215"></a><a name="p2495161616215"></a>584</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81282231916"><a name="p81282231916"></a><a name="p81282231916"></a>unfold</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p612815232111"><a name="p612815232111"></a><a name="p612815232111"></a>unfold</p>
</td>
</tr>
<tr id="row18754113941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1749514161728"><a name="p1749514161728"></a><a name="p1749514161728"></a>585</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p412872312116"><a name="p412872312116"></a><a name="p412872312116"></a>equal</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121281231912"><a name="p121281231912"></a><a name="p121281231912"></a>equal_npu</p>
</td>
</tr>
<tr id="row1175517394122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049571610214"><a name="p2049571610214"></a><a name="p2049571610214"></a>586</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p10128192310120"><a name="p10128192310120"></a><a name="p10128192310120"></a>pow.Tensor_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p171282231118"><a name="p171282231118"></a><a name="p171282231118"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row47551239161216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15496816329"><a name="p15496816329"></a><a name="p15496816329"></a>587</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141281523510"><a name="p141281523510"></a><a name="p141281523510"></a>pow.Tensor_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121281523011"><a name="p121281523011"></a><a name="p121281523011"></a>pow_npu</p>
</td>
</tr>
<tr id="row20755739121216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144967161429"><a name="p144967161429"></a><a name="p144967161429"></a>588</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1212820231118"><a name="p1212820231118"></a><a name="p1212820231118"></a>pow.Scalar_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p712818232114"><a name="p712818232114"></a><a name="p712818232114"></a>pow_out_npu</p>
</td>
</tr>
<tr id="row1675517394128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p549621620217"><a name="p549621620217"></a><a name="p549621620217"></a>589</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p812872315119"><a name="p812872315119"></a><a name="p812872315119"></a>pow.Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151281123419"><a name="p151281123419"></a><a name="p151281123419"></a>pow_npu</p>
</td>
</tr>
<tr id="row17755163920126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p149681614215"><a name="p149681614215"></a><a name="p149681614215"></a>590</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1312816231511"><a name="p1312816231511"></a><a name="p1312816231511"></a>normal_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1812872310113"><a name="p1812872310113"></a><a name="p1812872310113"></a>normal_npu_</p>
</td>
</tr>
<tr id="row167551839111212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144963169210"><a name="p144963169210"></a><a name="p144963169210"></a>591</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1612832312110"><a name="p1612832312110"></a><a name="p1612832312110"></a>normal.Tensor_float_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51287232116"><a name="p51287232116"></a><a name="p51287232116"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row177559399122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p164968169216"><a name="p164968169216"></a><a name="p164968169216"></a>592</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p121281238119"><a name="p121281238119"></a><a name="p121281238119"></a>normal.Tensor_float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161281823616"><a name="p161281823616"></a><a name="p161281823616"></a>normal_npu</p>
</td>
</tr>
<tr id="row9755539121212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349613161228"><a name="p349613161228"></a><a name="p349613161228"></a>593</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1312810231015"><a name="p1312810231015"></a><a name="p1312810231015"></a>normal.float_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181282231113"><a name="p181282231113"></a><a name="p181282231113"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row16755203919122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849611614213"><a name="p849611614213"></a><a name="p849611614213"></a>594</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101290230118"><a name="p101290230118"></a><a name="p101290230118"></a>normal.float_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121292023815"><a name="p121292023815"></a><a name="p121292023815"></a>normal_npu</p>
</td>
</tr>
<tr id="row8755103913125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349681613211"><a name="p349681613211"></a><a name="p349681613211"></a>595</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7129123618"><a name="p7129123618"></a><a name="p7129123618"></a>normal.Tensor_Tensor_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16129423813"><a name="p16129423813"></a><a name="p16129423813"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row16755103912120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349611161928"><a name="p349611161928"></a><a name="p349611161928"></a>596</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1012918231810"><a name="p1012918231810"></a><a name="p1012918231810"></a>normal.Tensor_Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161293239119"><a name="p161293239119"></a><a name="p161293239119"></a>normal_npu</p>
</td>
</tr>
<tr id="row3755183901212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8496151611215"><a name="p8496151611215"></a><a name="p8496151611215"></a>597</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12129152318110"><a name="p12129152318110"></a><a name="p12129152318110"></a>normal.float_float</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2012917238115"><a name="p2012917238115"></a><a name="p2012917238115"></a>normal_npu</p>
</td>
</tr>
<tr id="row1375613961210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13496111615215"><a name="p13496111615215"></a><a name="p13496111615215"></a>598</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51291823412"><a name="p51291823412"></a><a name="p51291823412"></a>normal.float_float_out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p212919231914"><a name="p212919231914"></a><a name="p212919231914"></a>normal_out_npu</p>
</td>
</tr>
<tr id="row1175623951217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1849615161424"><a name="p1849615161424"></a><a name="p1849615161424"></a>599</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p612917231712"><a name="p612917231712"></a><a name="p612917231712"></a>_addr</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11291023916"><a name="p11291023916"></a><a name="p11291023916"></a>_addr_npu</p>
</td>
</tr>
<tr id="row275613921217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44964161029"><a name="p44964161029"></a><a name="p44964161029"></a>600</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p612972315111"><a name="p612972315111"></a><a name="p612972315111"></a>_addr_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1012919234119"><a name="p1012919234119"></a><a name="p1012919234119"></a>_addr_npu_</p>
</td>
</tr>
<tr id="row275623991214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p124966163217"><a name="p124966163217"></a><a name="p124966163217"></a>601</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2129152315120"><a name="p2129152315120"></a><a name="p2129152315120"></a>_addr.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p101292233116"><a name="p101292233116"></a><a name="p101292233116"></a>_addr_out_npu</p>
</td>
</tr>
<tr id="row17756123941211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p349715161628"><a name="p349715161628"></a><a name="p349715161628"></a>602</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1812918231211"><a name="p1812918231211"></a><a name="p1812918231211"></a>_cumsum</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p191291923219"><a name="p191291923219"></a><a name="p191291923219"></a>_cumsum_npu</p>
</td>
</tr>
<tr id="row775611393129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15497111618214"><a name="p15497111618214"></a><a name="p15497111618214"></a>603</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2129023310"><a name="p2129023310"></a><a name="p2129023310"></a>_cumsum.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1412922317118"><a name="p1412922317118"></a><a name="p1412922317118"></a>_cumsum_out_npu</p>
</td>
</tr>
<tr id="row17756183951214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17497141612216"><a name="p17497141612216"></a><a name="p17497141612216"></a>604</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3129162314119"><a name="p3129162314119"></a><a name="p3129162314119"></a>_cumprod</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9129182315117"><a name="p9129182315117"></a><a name="p9129182315117"></a>_cumprod_npu</p>
</td>
</tr>
<tr id="row197561439121218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1349712166211"><a name="p1349712166211"></a><a name="p1349712166211"></a>605</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31291623615"><a name="p31291623615"></a><a name="p31291623615"></a>_cumprod.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61291923717"><a name="p61291923717"></a><a name="p61291923717"></a>_cumprod_out_npu</p>
</td>
</tr>
<tr id="row37563398122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p749720161529"><a name="p749720161529"></a><a name="p749720161529"></a>606</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151292231919"><a name="p151292231919"></a><a name="p151292231919"></a>_var</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31294237112"><a name="p31294237112"></a><a name="p31294237112"></a>_var_npu</p>
</td>
</tr>
<tr id="row1075683921219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p13497141617213"><a name="p13497141617213"></a><a name="p13497141617213"></a>607</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p212914234110"><a name="p212914234110"></a><a name="p212914234110"></a>_amp_non_finite_check_and_unscale_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181299234113"><a name="p181299234113"></a><a name="p181299234113"></a>_amp_non_finite_check_and_unscale_npu_</p>
</td>
</tr>
<tr id="row975733971217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849761616210"><a name="p849761616210"></a><a name="p849761616210"></a>608</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71295232118"><a name="p71295232118"></a><a name="p71295232118"></a>_cat</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11129182318113"><a name="p11129182318113"></a><a name="p11129182318113"></a>_cat_npu</p>
</td>
</tr>
<tr id="row157571439151211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1949741610213"><a name="p1949741610213"></a><a name="p1949741610213"></a>609</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71291823917"><a name="p71291823917"></a><a name="p71291823917"></a>_cat.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p91293239116"><a name="p91293239116"></a><a name="p91293239116"></a>_cat_out_npu</p>
</td>
</tr>
<tr id="row075718391126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54977161324"><a name="p54977161324"></a><a name="p54977161324"></a>610</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51292231314"><a name="p51292231314"></a><a name="p51292231314"></a>_max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p413020233110"><a name="p413020233110"></a><a name="p413020233110"></a>_max_npu</p>
</td>
</tr>
<tr id="row197572391122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3497116121"><a name="p3497116121"></a><a name="p3497116121"></a>611</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19130923117"><a name="p19130923117"></a><a name="p19130923117"></a>_max.max</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p01303237111"><a name="p01303237111"></a><a name="p01303237111"></a>_max_out_npu</p>
</td>
</tr>
<tr id="row15757173917127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18497816524"><a name="p18497816524"></a><a name="p18497816524"></a>612</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16130152320115"><a name="p16130152320115"></a><a name="p16130152320115"></a>_min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p01309231215"><a name="p01309231215"></a><a name="p01309231215"></a>_min_npu</p>
</td>
</tr>
<tr id="row9757039131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849781614219"><a name="p849781614219"></a><a name="p849781614219"></a>613</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11130223611"><a name="p11130223611"></a><a name="p11130223611"></a>_min.min</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1513013233114"><a name="p1513013233114"></a><a name="p1513013233114"></a>_min_out_npu</p>
</td>
</tr>
<tr id="row1757139191214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2049716161224"><a name="p2049716161224"></a><a name="p2049716161224"></a>614</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p513022318117"><a name="p513022318117"></a><a name="p513022318117"></a>mse_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p813012312119"><a name="p813012312119"></a><a name="p813012312119"></a>mse_loss_out_npu</p>
</td>
</tr>
<tr id="row275716390126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1649751617211"><a name="p1649751617211"></a><a name="p1649751617211"></a>615</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p01304231613"><a name="p01304231613"></a><a name="p01304231613"></a>mse_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8130223319"><a name="p8130223319"></a><a name="p8130223319"></a>mse_loss_npu</p>
</td>
</tr>
<tr id="row575717398122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5497316921"><a name="p5497316921"></a><a name="p5497316921"></a>616</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2013016235111"><a name="p2013016235111"></a><a name="p2013016235111"></a>mse_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p313052320111"><a name="p313052320111"></a><a name="p313052320111"></a>mse_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row1275733901211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174972161124"><a name="p174972161124"></a><a name="p174972161124"></a>617</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13130172312119"><a name="p13130172312119"></a><a name="p13130172312119"></a>mse_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111300231413"><a name="p111300231413"></a><a name="p111300231413"></a>mse_loss_backward_npu</p>
</td>
</tr>
<tr id="row475703981210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449771612210"><a name="p449771612210"></a><a name="p449771612210"></a>618</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31308231718"><a name="p31308231718"></a><a name="p31308231718"></a>l1_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17130172314110"><a name="p17130172314110"></a><a name="p17130172314110"></a>l1_loss_out_npu</p>
</td>
</tr>
<tr id="row137581539161214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p949761610212"><a name="p949761610212"></a><a name="p949761610212"></a>619</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81301623719"><a name="p81301623719"></a><a name="p81301623719"></a>l1_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1213010231211"><a name="p1213010231211"></a><a name="p1213010231211"></a>l1_loss_npu</p>
</td>
</tr>
<tr id="row12758113941215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84971316729"><a name="p84971316729"></a><a name="p84971316729"></a>620</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9130152316118"><a name="p9130152316118"></a><a name="p9130152316118"></a>l1_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p151302231311"><a name="p151302231311"></a><a name="p151302231311"></a>l1_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row167581739101219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11497171619218"><a name="p11497171619218"></a><a name="p11497171619218"></a>621</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p313017231313"><a name="p313017231313"></a><a name="p313017231313"></a>l1_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8130323412"><a name="p8130323412"></a><a name="p8130323412"></a>l1_loss_backward_npu</p>
</td>
</tr>
<tr id="row18758143931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p649813161220"><a name="p649813161220"></a><a name="p649813161220"></a>622</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11130182314114"><a name="p11130182314114"></a><a name="p11130182314114"></a>multilabel_margin_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p213013231217"><a name="p213013231217"></a><a name="p213013231217"></a>multilabel_margin_loss_out_npu</p>
</td>
</tr>
<tr id="row77581939111218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p84985160212"><a name="p84985160212"></a><a name="p84985160212"></a>623</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p12130112314118"><a name="p12130112314118"></a><a name="p12130112314118"></a>multilabel_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19130172317111"><a name="p19130172317111"></a><a name="p19130172317111"></a>multilabel_margin_loss_npu</p>
</td>
</tr>
<tr id="row1575893910121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44980167211"><a name="p44980167211"></a><a name="p44980167211"></a>624</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171301723218"><a name="p171301723218"></a><a name="p171301723218"></a>multilabel_margin_loss_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p913013239114"><a name="p913013239114"></a><a name="p913013239114"></a>multilabel_margin_loss_forward_out_npu</p>
</td>
</tr>
<tr id="row675812396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p174986164213"><a name="p174986164213"></a><a name="p174986164213"></a>625</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14130523116"><a name="p14130523116"></a><a name="p14130523116"></a>multilabel_margin_loss_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11130623412"><a name="p11130623412"></a><a name="p11130623412"></a>multilabel_margin_loss_forward_npu</p>
</td>
</tr>
<tr id="row1375812391128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p194985164213"><a name="p194985164213"></a><a name="p194985164213"></a>626</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1313016231212"><a name="p1313016231212"></a><a name="p1313016231212"></a>nll_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1013019231215"><a name="p1013019231215"></a><a name="p1013019231215"></a>nll_loss_out_npu</p>
</td>
</tr>
<tr id="row5758173920123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14498181611220"><a name="p14498181611220"></a><a name="p14498181611220"></a>627</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71307236119"><a name="p71307236119"></a><a name="p71307236119"></a>nll_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p013117231617"><a name="p013117231617"></a><a name="p013117231617"></a>nll_loss_npu</p>
</td>
</tr>
<tr id="row2758113911211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p049861611210"><a name="p049861611210"></a><a name="p049861611210"></a>628</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14131123119"><a name="p14131123119"></a><a name="p14131123119"></a>nll_loss_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9131823913"><a name="p9131823913"></a><a name="p9131823913"></a>nll_loss_forward_out_npu</p>
</td>
</tr>
<tr id="row975863921212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p34981116524"><a name="p34981116524"></a><a name="p34981116524"></a>629</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61311523112"><a name="p61311523112"></a><a name="p61311523112"></a>nll_loss_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p613142316117"><a name="p613142316117"></a><a name="p613142316117"></a>nll_loss_forward_npu</p>
</td>
</tr>
<tr id="row3758339121211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p549818161224"><a name="p549818161224"></a><a name="p549818161224"></a>630</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8131122318116"><a name="p8131122318116"></a><a name="p8131122318116"></a>nll_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p713115234118"><a name="p713115234118"></a><a name="p713115234118"></a>nll_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row18759193912122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8498181611210"><a name="p8498181611210"></a><a name="p8498181611210"></a>631</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p413192319120"><a name="p413192319120"></a><a name="p413192319120"></a>nll_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p413112319112"><a name="p413112319112"></a><a name="p413112319112"></a>nll_loss_backward_npu</p>
</td>
</tr>
<tr id="row5759103991218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1149881614210"><a name="p1149881614210"></a><a name="p1149881614210"></a>632</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81318231719"><a name="p81318231719"></a><a name="p81318231719"></a>nll_loss2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15131323416"><a name="p15131323416"></a><a name="p15131323416"></a>nll_loss2d_out_npu</p>
</td>
</tr>
<tr id="row11759163919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p849819161024"><a name="p849819161024"></a><a name="p849819161024"></a>633</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1613111238113"><a name="p1613111238113"></a><a name="p1613111238113"></a>nll_loss2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10131162319114"><a name="p10131162319114"></a><a name="p10131162319114"></a>nll_loss2d_npu</p>
</td>
</tr>
<tr id="row13759133910129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54981716923"><a name="p54981716923"></a><a name="p54981716923"></a>634</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11131112314116"><a name="p11131112314116"></a><a name="p11131112314116"></a>nll_loss2d_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p013116231911"><a name="p013116231911"></a><a name="p013116231911"></a>nll_loss2d_forward_out_npu</p>
</td>
</tr>
<tr id="row57591039171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114983162219"><a name="p114983162219"></a><a name="p114983162219"></a>635</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1213112234116"><a name="p1213112234116"></a><a name="p1213112234116"></a>nll_loss2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81311623212"><a name="p81311623212"></a><a name="p81311623212"></a>nll_loss2d_forward_npu</p>
</td>
</tr>
<tr id="row20759193931213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10498111612210"><a name="p10498111612210"></a><a name="p10498111612210"></a>636</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61311423716"><a name="p61311423716"></a><a name="p61311423716"></a>nll_loss2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1513111231915"><a name="p1513111231915"></a><a name="p1513111231915"></a>nll_loss2d_backward_out_npu</p>
</td>
</tr>
<tr id="row97594394129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9499111619217"><a name="p9499111619217"></a><a name="p9499111619217"></a>637</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1513112319115"><a name="p1513112319115"></a><a name="p1513112319115"></a>nll_loss2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71311423415"><a name="p71311423415"></a><a name="p71311423415"></a>nll_loss2d_backward_npu</p>
</td>
</tr>
<tr id="row8759339171217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1049912165210"><a name="p1049912165210"></a><a name="p1049912165210"></a>638</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81311323517"><a name="p81311323517"></a><a name="p81311323517"></a>smooth_l1_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1713120233115"><a name="p1713120233115"></a><a name="p1713120233115"></a>smooth_l1_loss_out_npu</p>
</td>
</tr>
<tr id="row6759193961213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p049915161629"><a name="p049915161629"></a><a name="p049915161629"></a>639</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51314232110"><a name="p51314232110"></a><a name="p51314232110"></a>smooth_l1_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1413120234112"><a name="p1413120234112"></a><a name="p1413120234112"></a>smooth_l1_loss_npu</p>
</td>
</tr>
<tr id="row9759339191211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p449951618214"><a name="p449951618214"></a><a name="p449951618214"></a>640</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1813172311118"><a name="p1813172311118"></a><a name="p1813172311118"></a>smooth_l1_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141311323618"><a name="p141311323618"></a><a name="p141311323618"></a>smooth_l1_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row2759183916127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p54999161522"><a name="p54999161522"></a><a name="p54999161522"></a>641</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171311823616"><a name="p171311823616"></a><a name="p171311823616"></a>smooth_l1_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2131623217"><a name="p2131623217"></a><a name="p2131623217"></a>smooth_l1_loss_backward_npu</p>
</td>
</tr>
<tr id="row18760123951219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14999161024"><a name="p14999161024"></a><a name="p14999161024"></a>642</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31311623811"><a name="p31311623811"></a><a name="p31311623811"></a>soft_margin_loss.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p513192312110"><a name="p513192312110"></a><a name="p513192312110"></a>soft_margin_loss_out_npu</p>
</td>
</tr>
<tr id="row18760153913129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144991316028"><a name="p144991316028"></a><a name="p144991316028"></a>643</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101311323111"><a name="p101311323111"></a><a name="p101311323111"></a>soft_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p613116231115"><a name="p613116231115"></a><a name="p613116231115"></a>soft_margin_loss_npu</p>
</td>
</tr>
<tr id="row1776043931215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p249917161724"><a name="p249917161724"></a><a name="p249917161724"></a>644</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1113162316116"><a name="p1113162316116"></a><a name="p1113162316116"></a>soft_margin_loss_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8132162315119"><a name="p8132162315119"></a><a name="p8132162315119"></a>soft_margin_loss_backward_out_npu</p>
</td>
</tr>
<tr id="row9760113961211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1249917162213"><a name="p1249917162213"></a><a name="p1249917162213"></a>645</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p513232313112"><a name="p513232313112"></a><a name="p513232313112"></a>soft_margin_loss_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131321523112"><a name="p131321523112"></a><a name="p131321523112"></a>soft_margin_loss_backward_npu</p>
</td>
</tr>
<tr id="row6760103981211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9499141618217"><a name="p9499141618217"></a><a name="p9499141618217"></a>646</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1613210230119"><a name="p1613210230119"></a><a name="p1613210230119"></a>elu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5132142319112"><a name="p5132142319112"></a><a name="p5132142319112"></a>elu_out_npu</p>
</td>
</tr>
<tr id="row2760143971212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p144990165211"><a name="p144990165211"></a><a name="p144990165211"></a>647</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101328232117"><a name="p101328232117"></a><a name="p101328232117"></a>elu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11321623816"><a name="p11321623816"></a><a name="p11321623816"></a>elu_npu</p>
</td>
</tr>
<tr id="row9760133981218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1499161610211"><a name="p1499161610211"></a><a name="p1499161610211"></a>648</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p121321423711"><a name="p121321423711"></a><a name="p121321423711"></a>elu_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p713212236116"><a name="p713212236116"></a><a name="p713212236116"></a>elu_backward_out_npu</p>
</td>
</tr>
<tr id="row1276053916122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p749918161328"><a name="p749918161328"></a><a name="p749918161328"></a>649</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191327231219"><a name="p191327231219"></a><a name="p191327231219"></a>elu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5132152311117"><a name="p5132152311117"></a><a name="p5132152311117"></a>elu_backward_npu</p>
</td>
</tr>
<tr id="row1076053919126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p114990166220"><a name="p114990166220"></a><a name="p114990166220"></a>650</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1513282312112"><a name="p1513282312112"></a><a name="p1513282312112"></a>elu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p713210233112"><a name="p713210233112"></a><a name="p713210233112"></a>elu_npu_</p>
</td>
</tr>
<tr id="row07605390120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p44991816520"><a name="p44991816520"></a><a name="p44991816520"></a>651</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191327232114"><a name="p191327232114"></a><a name="p191327232114"></a>glu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p013217231712"><a name="p013217231712"></a><a name="p013217231712"></a>glu_out_npu</p>
</td>
</tr>
<tr id="row176093951210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125001916829"><a name="p125001916829"></a><a name="p125001916829"></a>652</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101325231110"><a name="p101325231110"></a><a name="p101325231110"></a>glu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p8132102313117"><a name="p8132102313117"></a><a name="p8132102313117"></a>glu_npu</p>
</td>
</tr>
<tr id="row147611239111213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p165001616523"><a name="p165001616523"></a><a name="p165001616523"></a>653</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1213217232019"><a name="p1213217232019"></a><a name="p1213217232019"></a>glu_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p213212237112"><a name="p213212237112"></a><a name="p213212237112"></a>glu_backward_out_npu</p>
</td>
</tr>
<tr id="row0761839171218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1950014161729"><a name="p1950014161729"></a><a name="p1950014161729"></a>654</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1913216231311"><a name="p1913216231311"></a><a name="p1913216231311"></a>glu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p91327231516"><a name="p91327231516"></a><a name="p91327231516"></a>glu_backward_npu</p>
</td>
</tr>
<tr id="row147617390125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p85005168219"><a name="p85005168219"></a><a name="p85005168219"></a>655</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1113217232119"><a name="p1113217232119"></a><a name="p1113217232119"></a>hardsigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19132182310119"><a name="p19132182310119"></a><a name="p19132182310119"></a>hardsigmoid_out_npu</p>
</td>
</tr>
<tr id="row18761113941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p950031613212"><a name="p950031613212"></a><a name="p950031613212"></a>656</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p113214231012"><a name="p113214231012"></a><a name="p113214231012"></a>hardsigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1713214238114"><a name="p1713214238114"></a><a name="p1713214238114"></a>hardsigmoid_npu</p>
</td>
</tr>
<tr id="row97611639151214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1750012161220"><a name="p1750012161220"></a><a name="p1750012161220"></a>657</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41324234114"><a name="p41324234114"></a><a name="p41324234114"></a>hardsigmoid_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21325234118"><a name="p21325234118"></a><a name="p21325234118"></a>hardsigmoid_npu_</p>
</td>
</tr>
<tr id="row4761173931218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05001816822"><a name="p05001816822"></a><a name="p05001816822"></a>658</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p713262314110"><a name="p713262314110"></a><a name="p713262314110"></a>hardsigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11321823119"><a name="p11321823119"></a><a name="p11321823119"></a>hardsigmoid_backward_npu</p>
</td>
</tr>
<tr id="row9761173917125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2500916321"><a name="p2500916321"></a><a name="p2500916321"></a>659</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p713212236117"><a name="p713212236117"></a><a name="p713212236117"></a>hardtanh.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19132152320114"><a name="p19132152320114"></a><a name="p19132152320114"></a>hardtanh_out_npu</p>
</td>
</tr>
<tr id="row137611839121213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p550041617212"><a name="p550041617212"></a><a name="p550041617212"></a>660</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1613218231814"><a name="p1613218231814"></a><a name="p1613218231814"></a>hardtanh</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p171323236114"><a name="p171323236114"></a><a name="p171323236114"></a>hardtanh_npu</p>
</td>
</tr>
<tr id="row176153961214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750071612214"><a name="p750071612214"></a><a name="p750071612214"></a>661</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161327237117"><a name="p161327237117"></a><a name="p161327237117"></a>hardtanh_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1133192313112"><a name="p1133192313112"></a><a name="p1133192313112"></a>hardtanh_backward_out_npu</p>
</td>
</tr>
<tr id="row77611239121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7500101620218"><a name="p7500101620218"></a><a name="p7500101620218"></a>662</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41333237117"><a name="p41333237117"></a><a name="p41333237117"></a>hardtanh_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1413316233113"><a name="p1413316233113"></a><a name="p1413316233113"></a>hardtanh_backward_npu</p>
</td>
</tr>
<tr id="row376123941214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15005161023"><a name="p15005161023"></a><a name="p15005161023"></a>663</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20133132311116"><a name="p20133132311116"></a><a name="p20133132311116"></a>hardtanh_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41331231413"><a name="p41331231413"></a><a name="p41331231413"></a>hardtanh_npu_</p>
</td>
</tr>
<tr id="row1976173916120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p350020161425"><a name="p350020161425"></a><a name="p350020161425"></a>664</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2133132314113"><a name="p2133132314113"></a><a name="p2133132314113"></a>leaky_relu.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51337236114"><a name="p51337236114"></a><a name="p51337236114"></a>leaky_relu_out_npu</p>
</td>
</tr>
<tr id="row11762339181218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250017161027"><a name="p1250017161027"></a><a name="p1250017161027"></a>665</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p313311231315"><a name="p313311231315"></a><a name="p313311231315"></a>leaky_relu</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6133182311119"><a name="p6133182311119"></a><a name="p6133182311119"></a>leaky_relu_npu</p>
</td>
</tr>
<tr id="row117623394125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135000161828"><a name="p135000161828"></a><a name="p135000161828"></a>666</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13133723313"><a name="p13133723313"></a><a name="p13133723313"></a>leaky_relu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p913314231018"><a name="p913314231018"></a><a name="p913314231018"></a>leaky_relu_backward_npu</p>
</td>
</tr>
<tr id="row57620396121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p55006166220"><a name="p55006166220"></a><a name="p55006166220"></a>667</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151332231013"><a name="p151332231013"></a><a name="p151332231013"></a>leaky_relu_</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9133122311118"><a name="p9133122311118"></a><a name="p9133122311118"></a>leaky_relu_npu_</p>
</td>
</tr>
<tr id="row15762153919129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p850018161820"><a name="p850018161820"></a><a name="p850018161820"></a>668</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p313310238117"><a name="p313310238117"></a><a name="p313310238117"></a>log_sigmoid.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p313320231414"><a name="p313320231414"></a><a name="p313320231414"></a>log_sigmoid_out_npu</p>
</td>
</tr>
<tr id="row0762639141219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19500916222"><a name="p19500916222"></a><a name="p19500916222"></a>669</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p9133112314118"><a name="p9133112314118"></a><a name="p9133112314118"></a>log_sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p113320231311"><a name="p113320231311"></a><a name="p113320231311"></a>log_sigmoid_npu</p>
</td>
</tr>
<tr id="row7762239161219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15001161211"><a name="p15001161211"></a><a name="p15001161211"></a>670</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p8133152319118"><a name="p8133152319118"></a><a name="p8133152319118"></a>log_sigmoid_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p2133423712"><a name="p2133423712"></a><a name="p2133423712"></a>log_sigmoid_forward_out_npu</p>
</td>
</tr>
<tr id="row1776243931220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35003162219"><a name="p35003162219"></a><a name="p35003162219"></a>671</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p013312319117"><a name="p013312319117"></a><a name="p013312319117"></a>log_sigmoid_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1213316232017"><a name="p1213316232017"></a><a name="p1213316232017"></a>log_sigmoid_forward_npu</p>
</td>
</tr>
<tr id="row17621539171212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450071618216"><a name="p1450071618216"></a><a name="p1450071618216"></a>672</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1513320231114"><a name="p1513320231114"></a><a name="p1513320231114"></a>log_sigmoid_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1133152314110"><a name="p1133152314110"></a><a name="p1133152314110"></a>log_sigmoid_backward_out_npu</p>
</td>
</tr>
<tr id="row3762123931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17501121613220"><a name="p17501121613220"></a><a name="p17501121613220"></a>673</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131333231917"><a name="p131333231917"></a><a name="p131333231917"></a>log_sigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p201337239119"><a name="p201337239119"></a><a name="p201337239119"></a>log_sigmoid_backward_npu</p>
</td>
</tr>
<tr id="row1376293918127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p205011116925"><a name="p205011116925"></a><a name="p205011116925"></a>674</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171330231918"><a name="p171330231918"></a><a name="p171330231918"></a>softplus.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61336231519"><a name="p61336231519"></a><a name="p61336231519"></a>softplus_out_npu</p>
</td>
</tr>
<tr id="row117621239181210"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35018161521"><a name="p35018161521"></a><a name="p35018161521"></a>675</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p613382310119"><a name="p613382310119"></a><a name="p613382310119"></a>softplus</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21339232017"><a name="p21339232017"></a><a name="p21339232017"></a>softplus_npu</p>
</td>
</tr>
<tr id="row77631939131212"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p185015162219"><a name="p185015162219"></a><a name="p185015162219"></a>676</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81331237115"><a name="p81331237115"></a><a name="p81331237115"></a>softplus_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p91331823317"><a name="p91331823317"></a><a name="p91331823317"></a>softplus_backward_out_npu</p>
</td>
</tr>
<tr id="row1763183919127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45011216427"><a name="p45011216427"></a><a name="p45011216427"></a>677</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p16133323515"><a name="p16133323515"></a><a name="p16133323515"></a>softplus_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p013342310114"><a name="p013342310114"></a><a name="p013342310114"></a>softplus_backward_npu</p>
</td>
</tr>
<tr id="row11763113911215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7501216828"><a name="p7501216828"></a><a name="p7501216828"></a>678</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191341023517"><a name="p191341023517"></a><a name="p191341023517"></a>softshrink.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p14134923613"><a name="p14134923613"></a><a name="p14134923613"></a>softshrink_out_npu</p>
</td>
</tr>
<tr id="row177631039111214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750112163214"><a name="p750112163214"></a><a name="p750112163214"></a>679</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p713472319112"><a name="p713472319112"></a><a name="p713472319112"></a>softshrink</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p91340231416"><a name="p91340231416"></a><a name="p91340231416"></a>softshrink_npu</p>
</td>
</tr>
<tr id="row16763539181214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4501016221"><a name="p4501016221"></a><a name="p4501016221"></a>680</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3134112312111"><a name="p3134112312111"></a><a name="p3134112312111"></a>softshrink_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1913418231119"><a name="p1913418231119"></a><a name="p1913418231119"></a>softshrink_backward_out_npu</p>
</td>
</tr>
<tr id="row17763939191215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05011116527"><a name="p05011116527"></a><a name="p05011116527"></a>681</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31341623115"><a name="p31341623115"></a><a name="p31341623115"></a>softshrink_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p313422311112"><a name="p313422311112"></a><a name="p313422311112"></a>softshrink_backward_npu</p>
</td>
</tr>
<tr id="row27631039121220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p8501171613213"><a name="p8501171613213"></a><a name="p8501171613213"></a>682</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191345231519"><a name="p191345231519"></a><a name="p191345231519"></a>adaptive_avg_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131345236114"><a name="p131345236114"></a><a name="p131345236114"></a>adaptive_avg_pool2d_out_npu</p>
</td>
</tr>
<tr id="row117631439111219"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11501171616211"><a name="p11501171616211"></a><a name="p11501171616211"></a>683</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71341723013"><a name="p71341723013"></a><a name="p71341723013"></a>adaptive_avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7134172318113"><a name="p7134172318113"></a><a name="p7134172318113"></a>adaptive_avg_pool2d_npu</p>
</td>
</tr>
<tr id="row167636392122"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p105011716824"><a name="p105011716824"></a><a name="p105011716824"></a>684</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21341423715"><a name="p21341423715"></a><a name="p21341423715"></a>_adaptive_avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71341123215"><a name="p71341123215"></a><a name="p71341123215"></a>_adaptive_avg_pool2d_npu</p>
</td>
</tr>
<tr id="row77634398120"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135018161020"><a name="p135018161020"></a><a name="p135018161020"></a>685</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p813413231019"><a name="p813413231019"></a><a name="p813413231019"></a>_adaptive_avg_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1713432311112"><a name="p1713432311112"></a><a name="p1713432311112"></a>adaptive_avg_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row876373921213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p165015161219"><a name="p165015161219"></a><a name="p165015161219"></a>686</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14134132312115"><a name="p14134132312115"></a><a name="p14134132312115"></a>adaptive_avg_pool3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21346231515"><a name="p21346231515"></a><a name="p21346231515"></a>adaptive_avg_pool3d_out_npu</p>
</td>
</tr>
<tr id="row157641939171215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16501816824"><a name="p16501816824"></a><a name="p16501816824"></a>687</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p3134182317115"><a name="p3134182317115"></a><a name="p3134182317115"></a>adaptive_avg_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81345231112"><a name="p81345231112"></a><a name="p81345231112"></a>adaptive_avg_pool3d_npu</p>
</td>
</tr>
<tr id="row676463913126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25013163216"><a name="p25013163216"></a><a name="p25013163216"></a>688</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61343231812"><a name="p61343231812"></a><a name="p61343231812"></a>adaptive_avg_pool3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81341823513"><a name="p81341823513"></a><a name="p81341823513"></a>adaptive_avg_pool3d_backward_out_npu</p>
</td>
</tr>
<tr id="row576412392128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p55021161522"><a name="p55021161522"></a><a name="p55021161522"></a>689</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1013452315112"><a name="p1013452315112"></a><a name="p1013452315112"></a>adaptive_avg_pool3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1213472316120"><a name="p1213472316120"></a><a name="p1213472316120"></a>adaptive_avg_pool3d_backward_npu</p>
</td>
</tr>
<tr id="row0764539201211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250219163212"><a name="p1250219163212"></a><a name="p1250219163212"></a>690</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1113482316116"><a name="p1113482316116"></a><a name="p1113482316116"></a>adaptive_max_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1713416231911"><a name="p1713416231911"></a><a name="p1713416231911"></a>adaptive_max_pool2d_out_npu</p>
</td>
</tr>
<tr id="row6764153914123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35027161327"><a name="p35027161327"></a><a name="p35027161327"></a>691</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1113442311117"><a name="p1113442311117"></a><a name="p1113442311117"></a>adaptive_max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41347238117"><a name="p41347238117"></a><a name="p41347238117"></a>adaptive_max_pool2d_npu</p>
</td>
</tr>
<tr id="row876453916129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15021316426"><a name="p15021316426"></a><a name="p15021316426"></a>692</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61341823112"><a name="p61341823112"></a><a name="p61341823112"></a>adaptive_max_pool2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p9134123719"><a name="p9134123719"></a><a name="p9134123719"></a>adaptive_max_pool2d_backward_out_npu</p>
</td>
</tr>
<tr id="row47641139131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12502216929"><a name="p12502216929"></a><a name="p12502216929"></a>693</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p17135142311110"><a name="p17135142311110"></a><a name="p17135142311110"></a>adaptive_max_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71355239115"><a name="p71355239115"></a><a name="p71355239115"></a>adaptive_max_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row16764039191218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p17502316825"><a name="p17502316825"></a><a name="p17502316825"></a>694</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6135623614"><a name="p6135623614"></a><a name="p6135623614"></a>avg_pool2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31351323717"><a name="p31351323717"></a><a name="p31351323717"></a>avg_pool2d_out_npu</p>
</td>
</tr>
<tr id="row776420399123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p4502171611217"><a name="p4502171611217"></a><a name="p4502171611217"></a>695</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p2135223714"><a name="p2135223714"></a><a name="p2135223714"></a>avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p613532318114"><a name="p613532318114"></a><a name="p613532318114"></a>avg_pool2d_npu</p>
</td>
</tr>
<tr id="row167641939141216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12502716729"><a name="p12502716729"></a><a name="p12502716729"></a>696</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p213552316118"><a name="p213552316118"></a><a name="p213552316118"></a>avg_pool2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p13135623017"><a name="p13135623017"></a><a name="p13135623017"></a>avg_pool2d_backward_out_npu</p>
</td>
</tr>
<tr id="row77645392124"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1250213166210"><a name="p1250213166210"></a><a name="p1250213166210"></a>697</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21356234114"><a name="p21356234114"></a><a name="p21356234114"></a>avg_pool2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111351023118"><a name="p111351023118"></a><a name="p111351023118"></a>avg_pool2d_backward_npu</p>
</td>
</tr>
<tr id="row1576483991213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p35021316429"><a name="p35021316429"></a><a name="p35021316429"></a>698</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p613511231412"><a name="p613511231412"></a><a name="p613511231412"></a>avg_pool3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71351023217"><a name="p71351023217"></a><a name="p71351023217"></a>avg_pool3d_out_npu</p>
</td>
</tr>
<tr id="row1576515398128"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350214169210"><a name="p1350214169210"></a><a name="p1350214169210"></a>699</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p61351123711"><a name="p61351123711"></a><a name="p61351123711"></a>avg_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p5135123313"><a name="p5135123313"></a><a name="p5135123313"></a>avg_pool3d_npu</p>
</td>
</tr>
<tr id="row47651139101216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19502161614214"><a name="p19502161614214"></a><a name="p19502161614214"></a>700</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1513572312111"><a name="p1513572312111"></a><a name="p1513572312111"></a>avg_pool3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1013514231916"><a name="p1013514231916"></a><a name="p1013514231916"></a>avg_pool3d_backward_out_npu</p>
</td>
</tr>
<tr id="row13765939101211"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16502171620211"><a name="p16502171620211"></a><a name="p16502171620211"></a>701</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p513512237111"><a name="p513512237111"></a><a name="p513512237111"></a>avg_pool3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31357231811"><a name="p31357231811"></a><a name="p31357231811"></a>avg_pool3d_backward_npu</p>
</td>
</tr>
<tr id="row1976518395127"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450217169212"><a name="p1450217169212"></a><a name="p1450217169212"></a>702</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131351523116"><a name="p131351523116"></a><a name="p131351523116"></a>max_pool2d_with_indices.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51353231911"><a name="p51353231911"></a><a name="p51353231911"></a>max_pool2d_with_indices_out_npu</p>
</td>
</tr>
<tr id="row1476511392129"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15025163218"><a name="p15025163218"></a><a name="p15025163218"></a>703</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p013592316111"><a name="p013592316111"></a><a name="p013592316111"></a>max_pool2d_with_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0135192310117"><a name="p0135192310117"></a><a name="p0135192310117"></a>max_pool2d_with_indices_npu</p>
</td>
</tr>
<tr id="row0765173961215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145029161423"><a name="p145029161423"></a><a name="p145029161423"></a>704</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91359237110"><a name="p91359237110"></a><a name="p91359237110"></a>max_pool2d_with_indices_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41359235113"><a name="p41359235113"></a><a name="p41359235113"></a>max_pool2d_with_indices_backward_out_npu</p>
</td>
</tr>
<tr id="row0765163915123"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p65022161522"><a name="p65022161522"></a><a name="p65022161522"></a>705</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19135182310118"><a name="p19135182310118"></a><a name="p19135182310118"></a>max_pool2d_with_indices_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p15135123419"><a name="p15135123419"></a><a name="p15135123419"></a>max_pool2d_with_indices_backward_npu</p>
</td>
</tr>
<tr id="row12765639121214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125031016820"><a name="p125031016820"></a><a name="p125031016820"></a>706</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131354237118"><a name="p131354237118"></a><a name="p131354237118"></a>max_pool3d_with_indices.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p413518231115"><a name="p413518231115"></a><a name="p413518231115"></a>max_pool3d_with_indices_out_npu</p>
</td>
</tr>
<tr id="row1476513931214"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45036166220"><a name="p45036166220"></a><a name="p45036166220"></a>707</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7135122312111"><a name="p7135122312111"></a><a name="p7135122312111"></a>max_pool3d_with_indices</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71352231517"><a name="p71352231517"></a><a name="p71352231517"></a>max_pool3d_with_indices_npu</p>
</td>
</tr>
<tr id="row976583971220"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125032016329"><a name="p125032016329"></a><a name="p125032016329"></a>708</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p11355231618"><a name="p11355231618"></a><a name="p11355231618"></a>max_pool3d_with_indices_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p613512314119"><a name="p613512314119"></a><a name="p613512314119"></a>max_pool3d_with_indices_backward_out_npu</p>
</td>
</tr>
<tr id="row2076512399126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p105031516926"><a name="p105031516926"></a><a name="p105031516926"></a>709</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p01354232016"><a name="p01354232016"></a><a name="p01354232016"></a>max_pool3d_with_indices_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p113572311119"><a name="p113572311119"></a><a name="p113572311119"></a>max_pool3d_with_indices_backward_npu</p>
</td>
</tr>
<tr id="row87660397125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155038161329"><a name="p155038161329"></a><a name="p155038161329"></a>710</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91369231116"><a name="p91369231116"></a><a name="p91369231116"></a>reflection_pad2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1613610231114"><a name="p1613610231114"></a><a name="p1613610231114"></a>reflection_pad2d_out_npu</p>
</td>
</tr>
<tr id="row876610396126"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p18503216729"><a name="p18503216729"></a><a name="p18503216729"></a>711</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14136123318"><a name="p14136123318"></a><a name="p14136123318"></a>reflection_pad2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81365231711"><a name="p81365231711"></a><a name="p81365231711"></a>reflection_pad2d_npu</p>
</td>
</tr>
<tr id="row12766339201217"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p135035167217"><a name="p135035167217"></a><a name="p135035167217"></a>712</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14136172314112"><a name="p14136172314112"></a><a name="p14136172314112"></a>replication_pad2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p713602316118"><a name="p713602316118"></a><a name="p713602316118"></a>replication_pad2d_out_npu</p>
</td>
</tr>
<tr id="row976693981213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19503201613220"><a name="p19503201613220"></a><a name="p19503201613220"></a>713</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19136132313116"><a name="p19136132313116"></a><a name="p19136132313116"></a>replication_pad2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17136152312118"><a name="p17136152312118"></a><a name="p17136152312118"></a>replication_pad2d_npu</p>
</td>
</tr>
<tr id="row157661339191218"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115032016329"><a name="p115032016329"></a><a name="p115032016329"></a>714</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p513652320110"><a name="p513652320110"></a><a name="p513652320110"></a>upsample_linear1d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1113682317115"><a name="p1113682317115"></a><a name="p1113682317115"></a>upsample_linear1d_out_npu</p>
</td>
</tr>
<tr id="row076693919125"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16503816826"><a name="p16503816826"></a><a name="p16503816826"></a>715</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p6136102318116"><a name="p6136102318116"></a><a name="p6136102318116"></a>upsample_linear1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16136023216"><a name="p16136023216"></a><a name="p16136023216"></a>upsample_linear1d_npu</p>
</td>
</tr>
<tr id="row1876633961215"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p145031316127"><a name="p145031316127"></a><a name="p145031316127"></a>716</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p813616231417"><a name="p813616231417"></a><a name="p813616231417"></a>upsample_linear1d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81368231516"><a name="p81368231516"></a><a name="p81368231516"></a>upsample_linear1d_backward_npu</p>
</td>
</tr>
<tr id="row9767103941216"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25033165211"><a name="p25033165211"></a><a name="p25033165211"></a>717</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p101365231611"><a name="p101365231611"></a><a name="p101365231611"></a>upsample_bilinear2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p111361236115"><a name="p111361236115"></a><a name="p111361236115"></a>upsample_bilinear2d_out_npu</p>
</td>
</tr>
<tr id="row167671939131213"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2503116426"><a name="p2503116426"></a><a name="p2503116426"></a>718</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p19136192319113"><a name="p19136192319113"></a><a name="p19136192319113"></a>upsample_bilinear2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p0136142310118"><a name="p0136142310118"></a><a name="p0136142310118"></a>upsample_bilinear2d_npu</p>
</td>
</tr>
<tr id="row1676714395121"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p6503316326"><a name="p6503316326"></a><a name="p6503316326"></a>719</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91362231119"><a name="p91362231119"></a><a name="p91362231119"></a>upsample_bilinear2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1136112318110"><a name="p1136112318110"></a><a name="p1136112318110"></a>upsample_bilinear2d_backward_out_npu</p>
</td>
</tr>
<tr id="row18149195017234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p175032169219"><a name="p175032169219"></a><a name="p175032169219"></a>720</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131360237112"><a name="p131360237112"></a><a name="p131360237112"></a>upsample_bilinear2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10136923214"><a name="p10136923214"></a><a name="p10136923214"></a>upsample_bilinear2d_backward_npu</p>
</td>
</tr>
<tr id="row1614985042313"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7503816528"><a name="p7503816528"></a><a name="p7503816528"></a>721</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21364231115"><a name="p21364231115"></a><a name="p21364231115"></a>upsample_bicubic2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p121361023315"><a name="p121361023315"></a><a name="p121361023315"></a>upsample_bicubic2d_out_npu</p>
</td>
</tr>
<tr id="row17149115012238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1550401614213"><a name="p1550401614213"></a><a name="p1550401614213"></a>722</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p13136132311118"><a name="p13136132311118"></a><a name="p13136132311118"></a>upsample_bicubic2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p18136182318115"><a name="p18136182318115"></a><a name="p18136182318115"></a>upsample_bicubic2d_npu</p>
</td>
</tr>
<tr id="row614965016234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1750491617211"><a name="p1750491617211"></a><a name="p1750491617211"></a>723</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51361723116"><a name="p51361723116"></a><a name="p51361723116"></a>upsample_bicubic2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1136162317119"><a name="p1136162317119"></a><a name="p1136162317119"></a>upsample_bicubic2d_backward_out_npu</p>
</td>
</tr>
<tr id="row8149155011235"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5504121617216"><a name="p5504121617216"></a><a name="p5504121617216"></a>724</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111365231418"><a name="p111365231418"></a><a name="p111365231418"></a>upsample_bicubic2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p513620237112"><a name="p513620237112"></a><a name="p513620237112"></a>upsample_bicubic2d_backward_npu</p>
</td>
</tr>
<tr id="row914945052310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p15048168211"><a name="p15048168211"></a><a name="p15048168211"></a>725</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p71368230110"><a name="p71368230110"></a><a name="p71368230110"></a>upsample_trilinear3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p161361723113"><a name="p161361723113"></a><a name="p161361723113"></a>upsample_trilinear3d_out_npu</p>
</td>
</tr>
<tr id="row414875013236"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1450413161926"><a name="p1450413161926"></a><a name="p1450413161926"></a>726</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21361236119"><a name="p21361236119"></a><a name="p21361236119"></a>upsample_trilinear3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p01371237111"><a name="p01371237111"></a><a name="p01371237111"></a>upsample_trilinear3d_npu</p>
</td>
</tr>
<tr id="row181481650152310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250416161226"><a name="p250416161226"></a><a name="p250416161226"></a>727</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191379237119"><a name="p191379237119"></a><a name="p191379237119"></a>upsample_trilinear3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1113715231611"><a name="p1113715231611"></a><a name="p1113715231611"></a>upsample_trilinear3d_backward_out_npu</p>
</td>
</tr>
<tr id="row914810500234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p175042161925"><a name="p175042161925"></a><a name="p175042161925"></a>728</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1913716236112"><a name="p1913716236112"></a><a name="p1913716236112"></a>upsample_trilinear3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1713720231513"><a name="p1713720231513"></a><a name="p1713720231513"></a>upsample_trilinear3d_backward_npu</p>
</td>
</tr>
<tr id="row51481550142317"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11504816824"><a name="p11504816824"></a><a name="p11504816824"></a>729</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p121371723811"><a name="p121371723811"></a><a name="p121371723811"></a>upsample_nearest1d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181371023417"><a name="p181371023417"></a><a name="p181371023417"></a>upsample_nearest1d_out_npu</p>
</td>
</tr>
<tr id="row101481250142314"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250471613220"><a name="p250471613220"></a><a name="p250471613220"></a>730</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161372231510"><a name="p161372231510"></a><a name="p161372231510"></a>upsample_nearest1d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141379237110"><a name="p141379237110"></a><a name="p141379237110"></a>upsample_nearest1d_npu</p>
</td>
</tr>
<tr id="row91484505232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1950420161621"><a name="p1950420161621"></a><a name="p1950420161621"></a>731</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p913714231311"><a name="p913714231311"></a><a name="p913714231311"></a>upsample_nearest1d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p6137323917"><a name="p6137323917"></a><a name="p6137323917"></a>upsample_nearest1d_backward_out_npu</p>
</td>
</tr>
<tr id="row2148155019234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p95049161126"><a name="p95049161126"></a><a name="p95049161126"></a>732</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20137923210"><a name="p20137923210"></a><a name="p20137923210"></a>upsample_nearest1d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1413719236111"><a name="p1413719236111"></a><a name="p1413719236111"></a>upsample_nearest1d_backward_npu</p>
</td>
</tr>
<tr id="row151481250172312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p45043161823"><a name="p45043161823"></a><a name="p45043161823"></a>733</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111378231116"><a name="p111378231116"></a><a name="p111378231116"></a>upsample_nearest2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p16137123915"><a name="p16137123915"></a><a name="p16137123915"></a>upsample_nearest2d_out_npu</p>
</td>
</tr>
<tr id="row214811500239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155048163216"><a name="p155048163216"></a><a name="p155048163216"></a>734</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p171377234115"><a name="p171377234115"></a><a name="p171377234115"></a>upsample_nearest2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p113752314116"><a name="p113752314116"></a><a name="p113752314116"></a>upsample_nearest2d_npu</p>
</td>
</tr>
<tr id="row9148450142310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p450431613211"><a name="p450431613211"></a><a name="p450431613211"></a>735</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18137182317116"><a name="p18137182317116"></a><a name="p18137182317116"></a>upsample_nearest2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p131370233112"><a name="p131370233112"></a><a name="p131370233112"></a>upsample_nearest2d_backward_out_npu</p>
</td>
</tr>
<tr id="row914819503239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p115041816427"><a name="p115041816427"></a><a name="p115041816427"></a>736</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p131372236114"><a name="p131372236114"></a><a name="p131372236114"></a>upsample_nearest2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1613713231712"><a name="p1613713231712"></a><a name="p1613713231712"></a>upsample_nearest2d_backward_npu</p>
</td>
</tr>
<tr id="row1514765042314"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p450510161726"><a name="p450510161726"></a><a name="p450510161726"></a>737</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p91377231114"><a name="p91377231114"></a><a name="p91377231114"></a>upsample_nearest3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1113720231216"><a name="p1113720231216"></a><a name="p1113720231216"></a>upsample_nearest3d_out_npu</p>
</td>
</tr>
<tr id="row10147145019234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1850515161326"><a name="p1850515161326"></a><a name="p1850515161326"></a>738</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p4137523513"><a name="p4137523513"></a><a name="p4137523513"></a>upsample_nearest3d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p71371231510"><a name="p71371231510"></a><a name="p71371231510"></a>upsample_nearest3d_npu</p>
</td>
</tr>
<tr id="row131471350162312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p14505916423"><a name="p14505916423"></a><a name="p14505916423"></a>739</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p151370232113"><a name="p151370232113"></a><a name="p151370232113"></a>upsample_nearest3d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p31373231610"><a name="p31373231610"></a><a name="p31373231610"></a>upsample_nearest3d_backward_out_npu</p>
</td>
</tr>
<tr id="row514745013232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p950512161228"><a name="p950512161228"></a><a name="p950512161228"></a>740</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1513719231716"><a name="p1513719231716"></a><a name="p1513719231716"></a>upsample_nearest3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p513742311114"><a name="p513742311114"></a><a name="p513742311114"></a>upsample_nearest3d_backward_npu</p>
</td>
</tr>
<tr id="row16147115062311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p25054161225"><a name="p25054161225"></a><a name="p25054161225"></a>741</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p313720235115"><a name="p313720235115"></a><a name="p313720235115"></a>sigmoid_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61377234110"><a name="p61377234110"></a><a name="p61377234110"></a>sigmoid_backward_out_npu</p>
</td>
</tr>
<tr id="row1514755018239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p150501616218"><a name="p150501616218"></a><a name="p150501616218"></a>742</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p161371723416"><a name="p161371723416"></a><a name="p161371723416"></a>sigmoid_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p11137102318119"><a name="p11137102318119"></a><a name="p11137102318119"></a>sigmoid_backward_npu</p>
</td>
</tr>
<tr id="row914712503239"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p185052161625"><a name="p185052161625"></a><a name="p185052161625"></a>743</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41377234115"><a name="p41377234115"></a><a name="p41377234115"></a>tanh_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p181381723715"><a name="p181381723715"></a><a name="p181381723715"></a>tanh_backward_out_npu</p>
</td>
</tr>
<tr id="row151473505232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p16505716022"><a name="p16505716022"></a><a name="p16505716022"></a>744</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1613842319113"><a name="p1613842319113"></a><a name="p1613842319113"></a>tanh_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17138192315118"><a name="p17138192315118"></a><a name="p17138192315118"></a>tanh_backward_npu</p>
</td>
</tr>
<tr id="row171471350182312"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250518161825"><a name="p250518161825"></a><a name="p250518161825"></a>745</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21387230111"><a name="p21387230111"></a><a name="p21387230111"></a>slow_conv_transpose2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p4138142311115"><a name="p4138142311115"></a><a name="p4138142311115"></a>slow_conv_transpose2d_out_npu</p>
</td>
</tr>
<tr id="row12147150142310"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p3505151619219"><a name="p3505151619219"></a><a name="p3505151619219"></a>746</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1013819231314"><a name="p1013819231314"></a><a name="p1013819231314"></a>slow_conv_transpose2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p7138182318112"><a name="p7138182318112"></a><a name="p7138182318112"></a>slow_conv_transpose2d_npu</p>
</td>
</tr>
<tr id="row101472050152318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350511163213"><a name="p1350511163213"></a><a name="p1350511163213"></a>747</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1213822319114"><a name="p1213822319114"></a><a name="p1213822319114"></a>slow_conv_transpose2d_backward.grad_output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17138123510"><a name="p17138123510"></a><a name="p17138123510"></a>slow_conv_transpose2d_backward_out_npu</p>
</td>
</tr>
<tr id="row31463506231"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1350517161214"><a name="p1350517161214"></a><a name="p1350517161214"></a>748</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51385236119"><a name="p51385236119"></a><a name="p51385236119"></a>slow_conv_transpose2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1713815231017"><a name="p1713815231017"></a><a name="p1713815231017"></a>slow_conv_transpose2d_backward_npu</p>
</td>
</tr>
<tr id="row1814612508238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5505416525"><a name="p5505416525"></a><a name="p5505416525"></a>749</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p1113816231013"><a name="p1113816231013"></a><a name="p1113816231013"></a>thnn_conv2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p913818231819"><a name="p913818231819"></a><a name="p913818231819"></a>thnn_conv2d_out_npu</p>
</td>
</tr>
<tr id="row714614509238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9505201610211"><a name="p9505201610211"></a><a name="p9505201610211"></a>750</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p7138723712"><a name="p7138723712"></a><a name="p7138723712"></a>thnn_conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21383237118"><a name="p21383237118"></a><a name="p21383237118"></a>thnn_conv2d_npu</p>
</td>
</tr>
<tr id="row1714605042318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p9505171614218"><a name="p9505171614218"></a><a name="p9505171614218"></a>751</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111389231216"><a name="p111389231216"></a><a name="p111389231216"></a>thnn_conv2d_forward.output</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1813810233112"><a name="p1813810233112"></a><a name="p1813810233112"></a>thnn_conv2d_forward_out_npu</p>
</td>
</tr>
<tr id="row111461750132318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125051161024"><a name="p125051161024"></a><a name="p125051161024"></a>752</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p15138823214"><a name="p15138823214"></a><a name="p15138823214"></a>thnn_conv2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p17138142318117"><a name="p17138142318117"></a><a name="p17138142318117"></a>thnn_conv2d_forward_npu</p>
</td>
</tr>
<tr id="row7146185018238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p250518161620"><a name="p250518161620"></a><a name="p250518161620"></a>753</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p18138122311110"><a name="p18138122311110"></a><a name="p18138122311110"></a>thnn_conv2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p19138202319110"><a name="p19138202319110"></a><a name="p19138202319110"></a>thnn_conv2d_backward_npu</p>
</td>
</tr>
<tr id="row1514675082318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p11505191612218"><a name="p11505191612218"></a><a name="p11505191612218"></a>754</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p413814230112"><a name="p413814230112"></a><a name="p413814230112"></a>thnn_conv_depthwise2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p413816231919"><a name="p413816231919"></a><a name="p413816231919"></a>thnn_conv_depthwise2d_out_npu</p>
</td>
</tr>
<tr id="row1814645062311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19505131610215"><a name="p19505131610215"></a><a name="p19505131610215"></a>755</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p20138623612"><a name="p20138623612"></a><a name="p20138623612"></a>thnn_conv_depthwise2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p10138223812"><a name="p10138223812"></a><a name="p10138223812"></a>thnn_conv_depthwise2d_npu</p>
</td>
</tr>
<tr id="row12146135072311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19505216728"><a name="p19505216728"></a><a name="p19505216728"></a>756</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p81384231716"><a name="p81384231716"></a><a name="p81384231716"></a>thnn_conv_depthwise2d_forward.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p313815230112"><a name="p313815230112"></a><a name="p313815230112"></a>thnn_conv_depthwise2d_forward_out_npu</p>
</td>
</tr>
<tr id="row1214625011237"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p750511614212"><a name="p750511614212"></a><a name="p750511614212"></a>757</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p51381823913"><a name="p51381823913"></a><a name="p51381823913"></a>thnn_conv_depthwise2d_forward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p113811236114"><a name="p113811236114"></a><a name="p113811236114"></a>thnn_conv_depthwise2d_forward_npu</p>
</td>
</tr>
<tr id="row14146155022318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p7505101616219"><a name="p7505101616219"></a><a name="p7505101616219"></a>758</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p413812318119"><a name="p413812318119"></a><a name="p413812318119"></a>thnn_conv_depthwise2d_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p141381123614"><a name="p141381123614"></a><a name="p141381123614"></a>thnn_conv_depthwise2d_backward_out_npu</p>
</td>
</tr>
<tr id="row12145250202315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p5506141611210"><a name="p5506141611210"></a><a name="p5506141611210"></a>759</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111386231116"><a name="p111386231116"></a><a name="p111386231116"></a>thnn_conv_depthwise2d_backward.output_mask</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p41391231219"><a name="p41391231219"></a><a name="p41391231219"></a>thnn_conv_depthwise2d_backward_npu</p>
</td>
</tr>
<tr id="row19145125011236"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p85061716126"><a name="p85061716126"></a><a name="p85061716126"></a>760</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p111398231115"><a name="p111398231115"></a><a name="p111398231115"></a>slow_conv_dilated2d</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p81392231913"><a name="p81392231913"></a><a name="p81392231913"></a>slow_conv_dilated2d_npu</p>
</td>
</tr>
<tr id="row16145205013238"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p20506316325"><a name="p20506316325"></a><a name="p20506316325"></a>761</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p21395231715"><a name="p21395231715"></a><a name="p21395231715"></a>slow_conv_dilated2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1013919232115"><a name="p1013919232115"></a><a name="p1013919232115"></a>slow_conv_dilated2d_backward_npu</p>
</td>
</tr>
<tr id="row1914555052319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p155062016921"><a name="p155062016921"></a><a name="p155062016921"></a>762</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p313982319117"><a name="p313982319117"></a><a name="p313982319117"></a>col2im.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p913952316116"><a name="p913952316116"></a><a name="p913952316116"></a>im2col_backward_out_npu</p>
</td>
</tr>
<tr id="row114511508237"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p12506316228"><a name="p12506316228"></a><a name="p12506316228"></a>763</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p141392023519"><a name="p141392023519"></a><a name="p141392023519"></a>col2im</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p191396231016"><a name="p191396231016"></a><a name="p191396231016"></a>im2col_backward_npu</p>
</td>
</tr>
<tr id="row71456502232"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p05061161223"><a name="p05061161223"></a><a name="p05061161223"></a>764</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p14139112318117"><a name="p14139112318117"></a><a name="p14139112318117"></a>col2im_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1013919231414"><a name="p1013919231414"></a><a name="p1013919231414"></a>col2im_backward_out_npu</p>
</td>
</tr>
<tr id="row11145115062319"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p1150651618213"><a name="p1150651618213"></a><a name="p1150651618213"></a>765</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p413911231818"><a name="p413911231818"></a><a name="p413911231818"></a>col2im_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p61391231015"><a name="p61391231015"></a><a name="p61391231015"></a>col2im_backward_npu</p>
</td>
</tr>
<tr id="row61451350172318"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p10506151611212"><a name="p10506151611212"></a><a name="p10506151611212"></a>766</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p41392236115"><a name="p41392236115"></a><a name="p41392236115"></a>im2col.out</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p191391323919"><a name="p191391323919"></a><a name="p191391323919"></a>im2col_out_npu</p>
</td>
</tr>
<tr id="row10145115042317"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p150612162028"><a name="p150612162028"></a><a name="p150612162028"></a>767</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p31391823810"><a name="p31391823810"></a><a name="p31391823810"></a>im2col</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p51391723317"><a name="p51391723317"></a><a name="p51391723317"></a>im2col_npu</p>
</td>
</tr>
<tr id="row19145165022315"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p125067162029"><a name="p125067162029"></a><a name="p125067162029"></a>768</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p913910239118"><a name="p913910239118"></a><a name="p913910239118"></a>im2col_backward.grad_input</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p613910231215"><a name="p613910231215"></a><a name="p613910231215"></a>im2col_backward_out_npu</p>
</td>
</tr>
<tr id="row1314495014234"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p19506816928"><a name="p19506816928"></a><a name="p19506816928"></a>769</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p191395236116"><a name="p191395236116"></a><a name="p191395236116"></a>im2col_backward</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p1513922317116"><a name="p1513922317116"></a><a name="p1513922317116"></a>im2col_backward_npu</p>
</td>
</tr>
<tr id="row4144750152311"><td class="cellrowborder" valign="top" width="8.694379391100702%" headers="mcps1.1.4.1.1 "><p id="p2506616229"><a name="p2506616229"></a><a name="p2506616229"></a>770</p>
</td>
<td class="cellrowborder" valign="top" width="46.18462138953943%" headers="mcps1.1.4.1.2 "><p id="p613912313112"><a name="p613912313112"></a><a name="p613912313112"></a>isfinite</p>
</td>
<td class="cellrowborder" valign="top" width="45.120999219359874%" headers="mcps1.1.4.1.3 "><p id="p21394234113"><a name="p21394234113"></a><a name="p21394234113"></a>isfinite_npu</p>
</td>
</tr>
</tbody>
</table>

<h2 id="PyTorch昇腾自定义算子">PyTorch昇腾自定义算子</h2>

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
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p12853101745"><a name="p12853101745"></a><a name="p12853101745"></a>npu_convolution_transpose</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p78531308412"><a name="p78531308412"></a><a name="p78531308412"></a>npu_convolution_transpose</p>
</td>
</tr>
<tr id="row345954751320"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1937159446"><a name="p1937159446"></a><a name="p1937159446"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p785311011410"><a name="p785311011410"></a><a name="p785311011410"></a>npu_conv_transpose2d</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p17853907412"><a name="p17853907412"></a><a name="p17853907412"></a>convolution_transpose_npu</p>
</td>
</tr>
<tr id="row645954711320"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p103711291640"><a name="p103711291640"></a><a name="p103711291640"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p128531012415"><a name="p128531012415"></a><a name="p128531012415"></a>npu_convolution_transpose_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p7853170549"><a name="p7853170549"></a><a name="p7853170549"></a>convolution_transpose_backward_npu</p>
</td>
</tr>
<tr id="row104591947131315"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p15371791346"><a name="p15371791346"></a><a name="p15371791346"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p585330346"><a name="p585330346"></a><a name="p585330346"></a>npu_convolution</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1685314018415"><a name="p1685314018415"></a><a name="p1685314018415"></a>npu_convolution</p>
</td>
</tr>
<tr id="row17459124711134"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p12371109442"><a name="p12371109442"></a><a name="p12371109442"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p13853901849"><a name="p13853901849"></a><a name="p13853901849"></a>npu_convolution_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1785312013415"><a name="p1785312013415"></a><a name="p1785312013415"></a>npu_convolution_backward</p>
</td>
</tr>
<tr id="row16459847161315"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p537119916419"><a name="p537119916419"></a><a name="p537119916419"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p3853004415"><a name="p3853004415"></a><a name="p3853004415"></a>npu_conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p2853130748"><a name="p2853130748"></a><a name="p2853130748"></a>conv2d_npu</p>
</td>
</tr>
<tr id="row1145915478138"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p103711991749"><a name="p103711991749"></a><a name="p103711991749"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p128531801349"><a name="p128531801349"></a><a name="p128531801349"></a>npu_conv2d.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p178531301248"><a name="p178531301248"></a><a name="p178531301248"></a>conv2d_out_npu</p>
</td>
</tr>
<tr id="row14606476135"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p163721691444"><a name="p163721691444"></a><a name="p163721691444"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p18531701049"><a name="p18531701049"></a><a name="p18531701049"></a>npu_conv2d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p198531806418"><a name="p198531806418"></a><a name="p198531806418"></a>conv2d_backward_npu</p>
</td>
</tr>
<tr id="row446084710139"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p18372159142"><a name="p18372159142"></a><a name="p18372159142"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p385312014414"><a name="p385312014414"></a><a name="p385312014414"></a>npu_conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p198531708413"><a name="p198531708413"></a><a name="p198531708413"></a>conv3d_npu</p>
</td>
</tr>
<tr id="row04607478133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p113729919417"><a name="p113729919417"></a><a name="p113729919417"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p285360344"><a name="p285360344"></a><a name="p285360344"></a>npu_conv3d.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p188531705411"><a name="p188531705411"></a><a name="p188531705411"></a>conv3d_out_npu</p>
</td>
</tr>
<tr id="row9460347191318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5372499412"><a name="p5372499412"></a><a name="p5372499412"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p685313016411"><a name="p685313016411"></a><a name="p685313016411"></a>npu_conv3d_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1853004411"><a name="p1853004411"></a><a name="p1853004411"></a>conv3d_backward_npu</p>
</td>
</tr>
<tr id="row2460174710139"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p937219918416"><a name="p937219918416"></a><a name="p937219918416"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p148541101748"><a name="p148541101748"></a><a name="p148541101748"></a>one_</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p108541901944"><a name="p108541901944"></a><a name="p108541901944"></a>one_npu_</p>
</td>
</tr>
<tr id="row2046034712131"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p19372794415"><a name="p19372794415"></a><a name="p19372794415"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p08546015412"><a name="p08546015412"></a><a name="p08546015412"></a>npu_sort_v2.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p11854805411"><a name="p11854805411"></a><a name="p11854805411"></a>sort_without_indices_out_npu</p>
</td>
</tr>
<tr id="row246010470133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p437229345"><a name="p437229345"></a><a name="p437229345"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p985412017410"><a name="p985412017410"></a><a name="p985412017410"></a>npu_sort_v2</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p685440549"><a name="p685440549"></a><a name="p685440549"></a>sort_without_indices_npu</p>
</td>
</tr>
<tr id="row1246074751311"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p113721095410"><a name="p113721095410"></a><a name="p113721095410"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p118541605415"><a name="p118541605415"></a><a name="p118541605415"></a>npu_format_cast</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p128547017414"><a name="p128547017414"></a><a name="p128547017414"></a>format_cast_npu</p>
</td>
</tr>
<tr id="row546074711139"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p8372994413"><a name="p8372994413"></a><a name="p8372994413"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p285414012410"><a name="p285414012410"></a><a name="p285414012410"></a>npu_format_cast_.acl_format</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p19854130140"><a name="p19854130140"></a><a name="p19854130140"></a>format_cast_npu_</p>
</td>
</tr>
<tr id="row204603471134"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p183721491943"><a name="p183721491943"></a><a name="p183721491943"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p148541701843"><a name="p148541701843"></a><a name="p148541701843"></a>npu_format_cast_.src</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p188543010414"><a name="p188543010414"></a><a name="p188543010414"></a>format_cast_npu_</p>
</td>
</tr>
<tr id="row18460144714136"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p193721891744"><a name="p193721891744"></a><a name="p193721891744"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1854130941"><a name="p1854130941"></a><a name="p1854130941"></a>npu_transpose_to_contiguous</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p148541501541"><a name="p148541501541"></a><a name="p148541501541"></a>transpose_to_contiguous_npu</p>
</td>
</tr>
<tr id="row1460247101318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p537239842"><a name="p537239842"></a><a name="p537239842"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p08541101545"><a name="p08541101545"></a><a name="p08541101545"></a>npu_transpose</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p8854701642"><a name="p8854701642"></a><a name="p8854701642"></a>transpose_npu</p>
</td>
</tr>
<tr id="row846074712133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1537209944"><a name="p1537209944"></a><a name="p1537209944"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p138541808410"><a name="p138541808410"></a><a name="p138541808410"></a>npu_transpose.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p7854190043"><a name="p7854190043"></a><a name="p7854190043"></a>transpose_out_npu</p>
</td>
</tr>
<tr id="row17461164716137"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p9372491449"><a name="p9372491449"></a><a name="p9372491449"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p14854801246"><a name="p14854801246"></a><a name="p14854801246"></a>npu_broadcast</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p128541601348"><a name="p128541601348"></a><a name="p128541601348"></a>broadcast_npu</p>
</td>
</tr>
<tr id="row17461204715132"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p13372191246"><a name="p13372191246"></a><a name="p13372191246"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p285415014417"><a name="p285415014417"></a><a name="p285415014417"></a>npu_broadcast.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p7854701044"><a name="p7854701044"></a><a name="p7854701044"></a>broadcast_out_npu</p>
</td>
</tr>
<tr id="row9461104712136"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p183726915412"><a name="p183726915412"></a><a name="p183726915412"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p0854605414"><a name="p0854605414"></a><a name="p0854605414"></a>npu_dtype_cast</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p885417015419"><a name="p885417015419"></a><a name="p885417015419"></a>dtype_cast_npu</p>
</td>
</tr>
<tr id="row8461114771315"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p63721791410"><a name="p63721791410"></a><a name="p63721791410"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1585450243"><a name="p1585450243"></a><a name="p1585450243"></a>npu_dtype_cast_.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p168541306419"><a name="p168541306419"></a><a name="p168541306419"></a>dtype_cast_npu_</p>
</td>
</tr>
<tr id="row3461104717133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5372199748"><a name="p5372199748"></a><a name="p5372199748"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1085414011411"><a name="p1085414011411"></a><a name="p1085414011411"></a>npu_roi_alignbk</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p4854709412"><a name="p4854709412"></a><a name="p4854709412"></a>roi_align_backward_npu</p>
</td>
</tr>
<tr id="row946154714132"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5372395413"><a name="p5372395413"></a><a name="p5372395413"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p5854200841"><a name="p5854200841"></a><a name="p5854200841"></a>empty_with_format</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p68543019412"><a name="p68543019412"></a><a name="p68543019412"></a>empty_with_format_npu</p>
</td>
</tr>
<tr id="row1146114713137"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p93721916412"><a name="p93721916412"></a><a name="p93721916412"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p4854901345"><a name="p4854901345"></a><a name="p4854901345"></a>empty_with_format.names</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p2085519010416"><a name="p2085519010416"></a><a name="p2085519010416"></a>empty_with_format_npu</p>
</td>
</tr>
<tr id="row146112473133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p2037316914417"><a name="p2037316914417"></a><a name="p2037316914417"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p18855800411"><a name="p18855800411"></a><a name="p18855800411"></a>copy_memory_</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p58551401420"><a name="p58551401420"></a><a name="p58551401420"></a>copy_memory_npu_</p>
</td>
</tr>
<tr id="row1461447161310"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p637339248"><a name="p637339248"></a><a name="p637339248"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p158553018412"><a name="p158553018412"></a><a name="p158553018412"></a>npu_one_hot</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p28551909414"><a name="p28551909414"></a><a name="p28551909414"></a>one_hot_npu</p>
</td>
</tr>
<tr id="row7461134721317"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p7373591343"><a name="p7373591343"></a><a name="p7373591343"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1085519018414"><a name="p1085519018414"></a><a name="p1085519018414"></a>npu_stride_add</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p168557012415"><a name="p168557012415"></a><a name="p168557012415"></a>stride_add_npu</p>
</td>
</tr>
<tr id="row15461147191318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p103731098418"><a name="p103731098418"></a><a name="p103731098418"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p58554012412"><a name="p58554012412"></a><a name="p58554012412"></a>npu_softmax_cross_entropy_with_logits</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p11855202412"><a name="p11855202412"></a><a name="p11855202412"></a>softmax_cross_entropy_with_logits_npu</p>
</td>
</tr>
<tr id="row1046164717135"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p14373391641"><a name="p14373391641"></a><a name="p14373391641"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p48558014416"><a name="p48558014416"></a><a name="p48558014416"></a>npu_softmax_cross_entropy_with_logits_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p198551608414"><a name="p198551608414"></a><a name="p198551608414"></a>softmax_cross_entropy_with_logits_backward_npu</p>
</td>
</tr>
<tr id="row184627475137"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5373591748"><a name="p5373591748"></a><a name="p5373591748"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1285519020410"><a name="p1285519020410"></a><a name="p1285519020410"></a>npu_ps_roi_pooling</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p158559014414"><a name="p158559014414"></a><a name="p158559014414"></a>ps_roi_pooling_npu</p>
</td>
</tr>
<tr id="row24621147121312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p123731099411"><a name="p123731099411"></a><a name="p123731099411"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1685510247"><a name="p1685510247"></a><a name="p1685510247"></a>npu_ps_roi_pooling_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p168551501347"><a name="p168551501347"></a><a name="p168551501347"></a>ps_roi_pooling_backward_npu</p>
</td>
</tr>
<tr id="row204621747161315"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p3373199645"><a name="p3373199645"></a><a name="p3373199645"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p18551011412"><a name="p18551011412"></a><a name="p18551011412"></a>npu_roi_align</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p15855140742"><a name="p15855140742"></a><a name="p15855140742"></a>roi_align_npu</p>
</td>
</tr>
<tr id="row246294719133"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p133731594419"><a name="p133731594419"></a><a name="p133731594419"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p7855501746"><a name="p7855501746"></a><a name="p7855501746"></a>npu_nms_v4</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p685570743"><a name="p685570743"></a><a name="p685570743"></a>nms_v4_npu</p>
</td>
</tr>
<tr id="row84621047181311"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p63731699416"><a name="p63731699416"></a><a name="p63731699416"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1855501415"><a name="p1855501415"></a><a name="p1855501415"></a>npu_lstm</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p4855201943"><a name="p4855201943"></a><a name="p4855201943"></a>lstm_npu</p>
</td>
</tr>
<tr id="row16462174731312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p53731791349"><a name="p53731791349"></a><a name="p53731791349"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p88551809412"><a name="p88551809412"></a><a name="p88551809412"></a>npu_lstm_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1985590749"><a name="p1985590749"></a><a name="p1985590749"></a>lstm_backward_npu</p>
</td>
</tr>
<tr id="row3462124716130"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1037310917412"><a name="p1037310917412"></a><a name="p1037310917412"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p58551806418"><a name="p58551806418"></a><a name="p58551806418"></a>npu_iou</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1685513010413"><a name="p1685513010413"></a><a name="p1685513010413"></a>iou_npu</p>
</td>
</tr>
<tr id="row1046244751316"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p113738910412"><a name="p113738910412"></a><a name="p113738910412"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p285580144"><a name="p285580144"></a><a name="p285580144"></a>npu_ptiou</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p98552010416"><a name="p98552010416"></a><a name="p98552010416"></a>ptiou_npu</p>
</td>
</tr>
<tr id="row1546218475131"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p23731791641"><a name="p23731791641"></a><a name="p23731791641"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p188551801448"><a name="p188551801448"></a><a name="p188551801448"></a>npu_nms_with_mask</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p78551809419"><a name="p78551809419"></a><a name="p78551809419"></a>nms_with_mask_npu</p>
</td>
</tr>
<tr id="row11462847141320"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p63731991745"><a name="p63731991745"></a><a name="p63731991745"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p5855301749"><a name="p5855301749"></a><a name="p5855301749"></a>npu_pad</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p138551404410"><a name="p138551404410"></a><a name="p138551404410"></a>pad_npu</p>
</td>
</tr>
<tr id="row114621347101318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p103736912411"><a name="p103736912411"></a><a name="p103736912411"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p98561003410"><a name="p98561003410"></a><a name="p98561003410"></a>npu_bounding_box_encode</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p188561014416"><a name="p188561014416"></a><a name="p188561014416"></a>bounding_box_encode_npu</p>
</td>
</tr>
<tr id="row585972817535"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p785911289531"><a name="p785911289531"></a><a name="p785911289531"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p188561009417"><a name="p188561009417"></a><a name="p188561009417"></a>npu_bounding_box_decode</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p13856209412"><a name="p13856209412"></a><a name="p13856209412"></a>bounding_box_decode_npu</p>
</td>
</tr>
<tr id="row146351227195319"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p563572745318"><a name="p563572745318"></a><a name="p563572745318"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p985619013418"><a name="p985619013418"></a><a name="p985619013418"></a>npu_gru</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p585640444"><a name="p585640444"></a><a name="p585640444"></a>gru_npu</p>
</td>
</tr>
<tr id="row10604926145316"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1760582611533"><a name="p1760582611533"></a><a name="p1760582611533"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p2856904410"><a name="p2856904410"></a><a name="p2856904410"></a>npu_gru_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1785640542"><a name="p1785640542"></a><a name="p1785640542"></a>gru_backward_npu</p>
</td>
</tr>
<tr id="row1216382525314"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1216319254534"><a name="p1216319254534"></a><a name="p1216319254534"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p15856601943"><a name="p15856601943"></a><a name="p15856601943"></a>npu_set_.source_Storage_storage_offset_format</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p14856804413"><a name="p14856804413"></a><a name="p14856804413"></a>set_npu_</p>
</td>
</tr>
<tr id="row1369152495317"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p369124185315"><a name="p369124185315"></a><a name="p369124185315"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p12856701044"><a name="p12856701044"></a><a name="p12856701044"></a>npu_random_choice_with_mask</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1685618014418"><a name="p1685618014418"></a><a name="p1685618014418"></a>random_choice_with_mask_npu</p>
</td>
</tr>
<tr id="row17636162114539"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p18636152175314"><a name="p18636152175314"></a><a name="p18636152175314"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p8856009418"><a name="p8856009418"></a><a name="p8856009418"></a>npu_batch_nms</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p38561017414"><a name="p38561017414"></a><a name="p38561017414"></a>batch_nms_npu</p>
</td>
</tr>
<tr id="row12499620185311"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p45001320135317"><a name="p45001320135317"></a><a name="p45001320135317"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p18856901048"><a name="p18856901048"></a><a name="p18856901048"></a>npu_slice</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p48564012413"><a name="p48564012413"></a><a name="p48564012413"></a>slice_npu</p>
</td>
</tr>
<tr id="row173888191535"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1338815194530"><a name="p1338815194530"></a><a name="p1338815194530"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p168561101043"><a name="p168561101043"></a><a name="p168561101043"></a>npu_slice.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p17856808418"><a name="p17856808418"></a><a name="p17856808418"></a>slice_out_npu</p>
</td>
</tr>
<tr id="row192551918125314"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p16255171818533"><a name="p16255171818533"></a><a name="p16255171818533"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p188561106418"><a name="p188561106418"></a><a name="p188561106418"></a>npu_dropoutV2</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p5856100349"><a name="p5856100349"></a><a name="p5856100349"></a>dropout_v2_npu</p>
</td>
</tr>
<tr id="row20198181745319"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p12198161765312"><a name="p12198161765312"></a><a name="p12198161765312"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p108568018416"><a name="p108568018416"></a><a name="p108568018416"></a>npu_dropoutV2_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p12856110547"><a name="p12856110547"></a><a name="p12856110547"></a>dropout_v2_backward_npu</p>
</td>
</tr>
<tr id="row1717121610536"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p91711216165319"><a name="p91711216165319"></a><a name="p91711216165319"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p17856110644"><a name="p17856110644"></a><a name="p17856110644"></a>_npu_dropout</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p885617019410"><a name="p885617019410"></a><a name="p885617019410"></a>_dropout_npu</p>
</td>
</tr>
<tr id="row6772114195312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p19772101418536"><a name="p19772101418536"></a><a name="p19772101418536"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p0856302045"><a name="p0856302045"></a><a name="p0856302045"></a>_npu_dropout_inplace</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p28561207410"><a name="p28561207410"></a><a name="p28561207410"></a>_dropout_npu_inplace</p>
</td>
</tr>
<tr id="row1372431312535"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p672491310532"><a name="p672491310532"></a><a name="p672491310532"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p4856100943"><a name="p4856100943"></a><a name="p4856100943"></a>npu_dropout_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p48569013414"><a name="p48569013414"></a><a name="p48569013414"></a>dropout_backward_npu</p>
</td>
</tr>
<tr id="row34271912175319"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1842761225311"><a name="p1842761225311"></a><a name="p1842761225311"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p285660248"><a name="p285660248"></a><a name="p285660248"></a>npu_indexing</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p6856400417"><a name="p6856400417"></a><a name="p6856400417"></a>indexing_npu</p>
</td>
</tr>
<tr id="row6462134711313"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p143731791411"><a name="p143731791411"></a><a name="p143731791411"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1985613019411"><a name="p1985613019411"></a><a name="p1985613019411"></a>npu_indexing.out</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p158561601942"><a name="p158561601942"></a><a name="p158561601942"></a>indexing_out_npu</p>
</td>
</tr>
<tr id="row97791110115313"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p13779310195315"><a name="p13779310195315"></a><a name="p13779310195315"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1857501742"><a name="p1857501742"></a><a name="p1857501742"></a>npu_ifmr</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p13857140149"><a name="p13857140149"></a><a name="p13857140149"></a>ifmr_npu</p>
</td>
</tr>
<tr id="row184631247171312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p2373189647"><a name="p2373189647"></a><a name="p2373189647"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p148576010416"><a name="p148576010416"></a><a name="p148576010416"></a>npu_max.dim</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p12857180844"><a name="p12857180844"></a><a name="p12857180844"></a>max_v1_npu</p>
</td>
</tr>
<tr id="row1346364731320"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p337311910420"><a name="p337311910420"></a><a name="p337311910420"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p385750642"><a name="p385750642"></a><a name="p385750642"></a>npu_max.names_dim</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p685760547"><a name="p685760547"></a><a name="p685760547"></a>max_v1_npu</p>
</td>
</tr>
<tr id="row1463124714138"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p8374392416"><a name="p8374392416"></a><a name="p8374392416"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p148570017417"><a name="p148570017417"></a><a name="p148570017417"></a>npu_scatter</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p78571901941"><a name="p78571901941"></a><a name="p78571901941"></a>scatter_npu</p>
</td>
</tr>
<tr id="row104631747161314"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p23741894415"><a name="p23741894415"></a><a name="p23741894415"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p12857100842"><a name="p12857100842"></a><a name="p12857100842"></a>npu_max_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p198571702042"><a name="p198571702042"></a><a name="p198571702042"></a>max_backward_npu</p>
</td>
</tr>
<tr id="row739518135312"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p3395208125315"><a name="p3395208125315"></a><a name="p3395208125315"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p785710014420"><a name="p785710014420"></a><a name="p785710014420"></a>npu_apply_adam</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p188571001416"><a name="p188571001416"></a><a name="p188571001416"></a>apply_adam_npu</p>
</td>
</tr>
<tr id="row11492352531"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p13492856530"><a name="p13492856530"></a><a name="p13492856530"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p208571701445"><a name="p208571701445"></a><a name="p208571701445"></a>npu_layer_norm_eval</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p38571002041"><a name="p38571002041"></a><a name="p38571002041"></a>layer_norm_eval_npu</p>
</td>
</tr>
<tr id="row4579195725211"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p957975714522"><a name="p957975714522"></a><a name="p957975714522"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1485719013419"><a name="p1485719013419"></a><a name="p1485719013419"></a>npu_alloc_float_status</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p16857707411"><a name="p16857707411"></a><a name="p16857707411"></a>alloc_float_status_npu</p>
</td>
</tr>
<tr id="row1316510579407"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p15165125717406"><a name="p15165125717406"></a><a name="p15165125717406"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p685719017417"><a name="p685719017417"></a><a name="p685719017417"></a>npu_get_float_status</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p285714011418"><a name="p285714011418"></a><a name="p285714011418"></a>get_float_status_npu</p>
</td>
</tr>
<tr id="row238075912407"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p338035912408"><a name="p338035912408"></a><a name="p338035912408"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p1985790648"><a name="p1985790648"></a><a name="p1985790648"></a>npu_clear_float_status</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p58571001245"><a name="p58571001245"></a><a name="p58571001245"></a>clear_float_status_npu</p>
</td>
</tr>
<tr id="row1862019531825"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p9620953028"><a name="p9620953028"></a><a name="p9620953028"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p685770249"><a name="p685770249"></a><a name="p685770249"></a>npu_confusion_transpose</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p5857903416"><a name="p5857903416"></a><a name="p5857903416"></a>confusion_transpose_npu</p>
</td>
</tr>
<tr id="row1728911511238"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p7289125119316"><a name="p7289125119316"></a><a name="p7289125119316"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p168572015417"><a name="p168572015417"></a><a name="p168572015417"></a>npu_confusion_transpose_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p28571101243"><a name="p28571101243"></a><a name="p28571101243"></a>confusion_transpose_backward_npu</p>
</td>
</tr>
<tr id="row360924815311"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5609648036"><a name="p5609648036"></a><a name="p5609648036"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p88571400411"><a name="p88571400411"></a><a name="p88571400411"></a>npu_bmmV2</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p118571000411"><a name="p118571000411"></a><a name="p118571000411"></a>bmm_v2_npu</p>
</td>
</tr>
<tr id="row15706114616318"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p147061846733"><a name="p147061846733"></a><a name="p147061846733"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p188571803410"><a name="p188571803410"></a><a name="p188571803410"></a>fast_gelu</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p13857120341"><a name="p13857120341"></a><a name="p13857120341"></a>fast_gelu_npu</p>
</td>
</tr>
<tr id="row13860184418316"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p486084412314"><a name="p486084412314"></a><a name="p486084412314"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p14857305411"><a name="p14857305411"></a><a name="p14857305411"></a>fast_gelu_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p98571108411"><a name="p98571108411"></a><a name="p98571108411"></a>fast_gelu_backward_npu</p>
</td>
</tr>
<tr id="row155784016310"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p15571640231"><a name="p15571640231"></a><a name="p15571640231"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p88571104413"><a name="p88571104413"></a><a name="p88571104413"></a>npu_sub_sample</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p485715020414"><a name="p485715020414"></a><a name="p485715020414"></a>sub_sample_npu</p>
</td>
</tr>
<tr id="row38693421313"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p178692423316"><a name="p178692423316"></a><a name="p178692423316"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p385811010419"><a name="p385811010419"></a><a name="p385811010419"></a>npu_deformable_conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p14858609414"><a name="p14858609414"></a><a name="p14858609414"></a>deformable_conv2d_npu</p>
</td>
</tr>
<tr id="row8404138936"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p5404193820319"><a name="p5404193820319"></a><a name="p5404193820319"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p138581701749"><a name="p138581701749"></a><a name="p138581701749"></a>npu_deformable_conv2dbk</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p8858900411"><a name="p8858900411"></a><a name="p8858900411"></a>deformable_conv2d_backward_npu</p>
</td>
</tr>
<tr id="row48728361436"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1287253614310"><a name="p1287253614310"></a><a name="p1287253614310"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p385811017418"><a name="p385811017418"></a><a name="p385811017418"></a>npu_mish</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p38581906410"><a name="p38581906410"></a><a name="p38581906410"></a>mish_npu</p>
</td>
</tr>
<tr id="row193801734737"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p163818348318"><a name="p163818348318"></a><a name="p163818348318"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p17858120348"><a name="p17858120348"></a><a name="p17858120348"></a>npu_anchor_response_flags</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p12858160147"><a name="p12858160147"></a><a name="p12858160147"></a>anchor_response_flags_npu</p>
</td>
</tr>
<tr id="row887819321539"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p38781332934"><a name="p38781332934"></a><a name="p38781332934"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p158581001345"><a name="p158581001345"></a><a name="p158581001345"></a>npu_yolo_boxes_encode</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p15858305419"><a name="p15858305419"></a><a name="p15858305419"></a>yolo_boxes_encode_npu</p>
</td>
</tr>
<tr id="row1936802914316"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p43691329937"><a name="p43691329937"></a><a name="p43691329937"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p128581109415"><a name="p128581109415"></a><a name="p128581109415"></a>npu_grid_assign_positive</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p18858209419"><a name="p18858209419"></a><a name="p18858209419"></a>grid_assign_positive_npu</p>
</td>
</tr>
<tr id="row161409311933"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p2014003110319"><a name="p2014003110319"></a><a name="p2014003110319"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p28587010410"><a name="p28587010410"></a><a name="p28587010410"></a>npu_mish_backward</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p2858407414"><a name="p2858407414"></a><a name="p2858407414"></a>mish_backward_npu</p>
</td>
</tr>
<tr id="row124539561211"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p1845314561229"><a name="p1845314561229"></a><a name="p1845314561229"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p4858205414"><a name="p4858205414"></a><a name="p4858205414"></a>npu_normalize_batch</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1685840449"><a name="p1685840449"></a><a name="p1685840449"></a>normalize_batch_npu</p>
</td>
</tr>
<tr id="row47113234310"><td class="cellrowborder" valign="top" width="8.334944884935215%" headers="mcps1.1.4.1.1 "><p id="p14711182310319"><a name="p14711182310319"></a><a name="p14711182310319"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="46.954167472442464%" headers="mcps1.1.4.1.2 "><p id="p12858803419"><a name="p12858803419"></a><a name="p12858803419"></a>npu_masked_fill_range</p>
</td>
<td class="cellrowborder" valign="top" width="44.71088764262232%" headers="mcps1.1.4.1.3 "><p id="p1385819016415"><a name="p1385819016415"></a><a name="p1385819016415"></a>masked_fill_range_npu</p>
</td>
</tr>
</tbody>
</table>

