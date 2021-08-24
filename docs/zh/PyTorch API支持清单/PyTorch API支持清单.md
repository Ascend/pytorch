# PyTorch API支持清单
-   [Tensors](#Tensors)
-   [Generators](#Generators)
-   [Random sampling](#Random-sampling)
-   [Serialization](#Serialization)
-   [Math operations](#Math-operations)
-   [Utilities](#Utilities)
-   [Other](#Other)
-   [torch.Tensor](#torch-Tensor)
-   [Layers \(torch.nn\)](#Layers-torch-nn)
-   [Functions\(torch.nn.functional\)](#Functionstorch-nn-functional)
-   [torch.distributed](#torch-distributed)
-   [NPU和CUDA功能对齐](#NPU和CUDA功能对齐)
<h2 id="Tensors">Tensors</h2>

<a name="table155791359155219"></a>
<table><thead align="left"><tr id="row038419012538"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p960817567348"><a name="p960817567348"></a><a name="p960817567348"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row43857095315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3461923193716"><a name="p3461923193716"></a><a name="p3461923193716"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13908229406"><a name="p13908229406"></a><a name="p13908229406"></a>torch.is_tensor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p59084294011"><a name="p59084294011"></a><a name="p59084294011"></a>是</p>
</td>
</tr>
<tr id="row43858065310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p164611623103720"><a name="p164611623103720"></a><a name="p164611623103720"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p990872134013"><a name="p990872134013"></a><a name="p990872134013"></a>torch.is_storage</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p139081322409"><a name="p139081322409"></a><a name="p139081322409"></a>是</p>
</td>
</tr>
<tr id="row113851010539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p134614238377"><a name="p134614238377"></a><a name="p134614238377"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p209081826409"><a name="p209081826409"></a><a name="p209081826409"></a>torch.is_complex</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p119084274013"><a name="p119084274013"></a><a name="p119084274013"></a>否</p>
</td>
</tr>
<tr id="row438518085316"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4461102312379"><a name="p4461102312379"></a><a name="p4461102312379"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1090852184020"><a name="p1090852184020"></a><a name="p1090852184020"></a>torch.is_floating_point</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1690816284016"><a name="p1690816284016"></a><a name="p1690816284016"></a>是</p>
</td>
</tr>
<tr id="row143859075317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19461323103716"><a name="p19461323103716"></a><a name="p19461323103716"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p890812104014"><a name="p890812104014"></a><a name="p890812104014"></a>torch.set_default_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15908524406"><a name="p15908524406"></a><a name="p15908524406"></a>是</p>
</td>
</tr>
<tr id="row93851001536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11461623163710"><a name="p11461623163710"></a><a name="p11461623163710"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p190814210401"><a name="p190814210401"></a><a name="p190814210401"></a>torch.get_default_dtype</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8908121407"><a name="p8908121407"></a><a name="p8908121407"></a>是</p>
</td>
</tr>
<tr id="row1338619014537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p946112234371"><a name="p946112234371"></a><a name="p946112234371"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p12908226408"><a name="p12908226408"></a><a name="p12908226408"></a>torch.set_default_tensor_type</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p29082210408"><a name="p29082210408"></a><a name="p29082210408"></a>是</p>
</td>
</tr>
<tr id="row93863095319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12461162333717"><a name="p12461162333717"></a><a name="p12461162333717"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1090832164012"><a name="p1090832164012"></a><a name="p1090832164012"></a>torch.numel</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10908526409"><a name="p10908526409"></a><a name="p10908526409"></a>是</p>
</td>
</tr>
<tr id="row53861002533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p184615233371"><a name="p184615233371"></a><a name="p184615233371"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2908112154018"><a name="p2908112154018"></a><a name="p2908112154018"></a>torch.set_printoptions</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p129091722401"><a name="p129091722401"></a><a name="p129091722401"></a>是</p>
</td>
</tr>
<tr id="row1038616085311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p446192314378"><a name="p446192314378"></a><a name="p446192314378"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p490920212404"><a name="p490920212404"></a><a name="p490920212404"></a>torch.set_flush_denormal</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p790912215407"><a name="p790912215407"></a><a name="p790912215407"></a>是</p>
</td>
</tr>
<tr id="row16386170115319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p646182323719"><a name="p646182323719"></a><a name="p646182323719"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p59093274018"><a name="p59093274018"></a><a name="p59093274018"></a>torch.tensor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p99093234017"><a name="p99093234017"></a><a name="p99093234017"></a>是</p>
</td>
</tr>
<tr id="row238620105310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p114621123183710"><a name="p114621123183710"></a><a name="p114621123183710"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2090918274017"><a name="p2090918274017"></a><a name="p2090918274017"></a>torch.sparse_coo_tensor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1990918220404"><a name="p1990918220404"></a><a name="p1990918220404"></a>否</p>
</td>
</tr>
<tr id="row83871705537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1446232318370"><a name="p1446232318370"></a><a name="p1446232318370"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p590922184012"><a name="p590922184012"></a><a name="p590922184012"></a>torch.as_tensor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19097213405"><a name="p19097213405"></a><a name="p19097213405"></a>是</p>
</td>
</tr>
<tr id="row1738750155312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1546262313713"><a name="p1546262313713"></a><a name="p1546262313713"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19094284019"><a name="p19094284019"></a><a name="p19094284019"></a>torch.as_strided</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1990913294016"><a name="p1990913294016"></a><a name="p1990913294016"></a>是</p>
</td>
</tr>
<tr id="row538717035315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p846217239373"><a name="p846217239373"></a><a name="p846217239373"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1490914214405"><a name="p1490914214405"></a><a name="p1490914214405"></a>torch.from_numpy</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3909925400"><a name="p3909925400"></a><a name="p3909925400"></a>是</p>
</td>
</tr>
<tr id="row8387180155310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2462122319377"><a name="p2462122319377"></a><a name="p2462122319377"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19909229408"><a name="p19909229408"></a><a name="p19909229408"></a>torch.zeros</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p159091324404"><a name="p159091324404"></a><a name="p159091324404"></a>是</p>
</td>
</tr>
<tr id="row18387190125313"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1846282317377"><a name="p1846282317377"></a><a name="p1846282317377"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19909127401"><a name="p19909127401"></a><a name="p19909127401"></a>torch.zeros_like</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p59091827402"><a name="p59091827402"></a><a name="p59091827402"></a>是</p>
</td>
</tr>
<tr id="row9388190145310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1146215236373"><a name="p1146215236373"></a><a name="p1146215236373"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1790962194015"><a name="p1790962194015"></a><a name="p1790962194015"></a>torch.ones</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1990982104010"><a name="p1990982104010"></a><a name="p1990982104010"></a>是</p>
</td>
</tr>
<tr id="row1738812045310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p04621423153713"><a name="p04621423153713"></a><a name="p04621423153713"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p490922154013"><a name="p490922154013"></a><a name="p490922154013"></a>torch.ones_like</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3909124409"><a name="p3909124409"></a><a name="p3909124409"></a>是</p>
</td>
</tr>
<tr id="row193881309536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11462182393713"><a name="p11462182393713"></a><a name="p11462182393713"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1090992124019"><a name="p1090992124019"></a><a name="p1090992124019"></a>torch.arange</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1990992174019"><a name="p1990992174019"></a><a name="p1990992174019"></a>是</p>
</td>
</tr>
<tr id="row123882035311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p44621023133712"><a name="p44621023133712"></a><a name="p44621023133712"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6910182194014"><a name="p6910182194014"></a><a name="p6910182194014"></a>torch.range</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1391011264013"><a name="p1391011264013"></a><a name="p1391011264013"></a>是</p>
</td>
</tr>
<tr id="row238819065315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1346272319370"><a name="p1346272319370"></a><a name="p1346272319370"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1391052114017"><a name="p1391052114017"></a><a name="p1391052114017"></a>torch.linspace</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1491019204011"><a name="p1491019204011"></a><a name="p1491019204011"></a>是</p>
</td>
</tr>
<tr id="row63881302531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p24621823163717"><a name="p24621823163717"></a><a name="p24621823163717"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p39104264014"><a name="p39104264014"></a><a name="p39104264014"></a>torch.logspace</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p29104214017"><a name="p29104214017"></a><a name="p29104214017"></a>是</p>
</td>
</tr>
<tr id="row1038913095310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p134621323173719"><a name="p134621323173719"></a><a name="p134621323173719"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17910162104012"><a name="p17910162104012"></a><a name="p17910162104012"></a>torch.eye</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19108254016"><a name="p19108254016"></a><a name="p19108254016"></a>是</p>
</td>
</tr>
<tr id="row1738918085317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p54628235371"><a name="p54628235371"></a><a name="p54628235371"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2091082154012"><a name="p2091082154012"></a><a name="p2091082154012"></a>torch.empty</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1091042174016"><a name="p1091042174016"></a><a name="p1091042174016"></a>是</p>
</td>
</tr>
<tr id="row133896075317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p134627230376"><a name="p134627230376"></a><a name="p134627230376"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1491015210408"><a name="p1491015210408"></a><a name="p1491015210408"></a>torch.empty_like</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15910192194013"><a name="p15910192194013"></a><a name="p15910192194013"></a>是</p>
</td>
</tr>
<tr id="row838911017537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p946202313377"><a name="p946202313377"></a><a name="p946202313377"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13910142144016"><a name="p13910142144016"></a><a name="p13910142144016"></a>torch.empty_strided</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p169109216400"><a name="p169109216400"></a><a name="p169109216400"></a>是</p>
</td>
</tr>
<tr id="row738918015532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5462102317378"><a name="p5462102317378"></a><a name="p5462102317378"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17910122154011"><a name="p17910122154011"></a><a name="p17910122154011"></a>torch.full</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p119109216402"><a name="p119109216402"></a><a name="p119109216402"></a>是</p>
</td>
</tr>
<tr id="row7389190145312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p546382318373"><a name="p546382318373"></a><a name="p546382318373"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10910102144013"><a name="p10910102144013"></a><a name="p10910102144013"></a>torch.full_like</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p69106213402"><a name="p69106213402"></a><a name="p69106213402"></a>是</p>
</td>
</tr>
<tr id="row19389603533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13463122323715"><a name="p13463122323715"></a><a name="p13463122323715"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1091013217405"><a name="p1091013217405"></a><a name="p1091013217405"></a>torch.quantize_per_tensor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1891072184019"><a name="p1891072184019"></a><a name="p1891072184019"></a>是</p>
</td>
</tr>
<tr id="row63901504535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p154631423143719"><a name="p154631423143719"></a><a name="p154631423143719"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p291020214404"><a name="p291020214404"></a><a name="p291020214404"></a>torch.quantize_per_channel</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p109101828407"><a name="p109101828407"></a><a name="p109101828407"></a>否</p>
</td>
</tr>
<tr id="row183907075312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6463162383716"><a name="p6463162383716"></a><a name="p6463162383716"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p29101323401"><a name="p29101323401"></a><a name="p29101323401"></a>torch.cat</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p49111529402"><a name="p49111529402"></a><a name="p49111529402"></a>是</p>
</td>
</tr>
<tr id="row03905011532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1246312318375"><a name="p1246312318375"></a><a name="p1246312318375"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p891114214402"><a name="p891114214402"></a><a name="p891114214402"></a>torch.chunk</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p29110224012"><a name="p29110224012"></a><a name="p29110224012"></a>是</p>
</td>
</tr>
<tr id="row139012012538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8463122317375"><a name="p8463122317375"></a><a name="p8463122317375"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p89116215404"><a name="p89116215404"></a><a name="p89116215404"></a>torch.gather</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p79116217409"><a name="p79116217409"></a><a name="p79116217409"></a>是</p>
</td>
</tr>
<tr id="row339017014538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10463192319375"><a name="p10463192319375"></a><a name="p10463192319375"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13911122174011"><a name="p13911122174011"></a><a name="p13911122174011"></a>torch.index_select</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p139119219408"><a name="p139119219408"></a><a name="p139119219408"></a>是</p>
</td>
</tr>
<tr id="row1139013012539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1846322317378"><a name="p1846322317378"></a><a name="p1846322317378"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p79117213407"><a name="p79117213407"></a><a name="p79117213407"></a>torch.masked_select</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8911920405"><a name="p8911920405"></a><a name="p8911920405"></a>是</p>
</td>
</tr>
<tr id="row1839015014533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14631523203717"><a name="p14631523203717"></a><a name="p14631523203717"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1391152104020"><a name="p1391152104020"></a><a name="p1391152104020"></a>torch.narrow</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12911192114014"><a name="p12911192114014"></a><a name="p12911192114014"></a>是</p>
</td>
</tr>
<tr id="row143901008539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p846342312371"><a name="p846342312371"></a><a name="p846342312371"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p791117219409"><a name="p791117219409"></a><a name="p791117219409"></a>torch.nonzero</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p119115214401"><a name="p119115214401"></a><a name="p119115214401"></a>是</p>
</td>
</tr>
<tr id="row193901901537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20463102315371"><a name="p20463102315371"></a><a name="p20463102315371"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p791117219407"><a name="p791117219407"></a><a name="p791117219407"></a>torch.reshape</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9911721407"><a name="p9911721407"></a><a name="p9911721407"></a>是</p>
</td>
</tr>
<tr id="row123911012535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15463192319377"><a name="p15463192319377"></a><a name="p15463192319377"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p99111322404"><a name="p99111322404"></a><a name="p99111322404"></a>torch.split</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p209115254010"><a name="p209115254010"></a><a name="p209115254010"></a>是</p>
</td>
</tr>
<tr id="row139170115315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1463192316371"><a name="p1463192316371"></a><a name="p1463192316371"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p169112217402"><a name="p169112217402"></a><a name="p169112217402"></a>torch.squeeze</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p129111329407"><a name="p129111329407"></a><a name="p129111329407"></a>是</p>
</td>
</tr>
<tr id="row2391402531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p44635234379"><a name="p44635234379"></a><a name="p44635234379"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10911129405"><a name="p10911129405"></a><a name="p10911129405"></a>torch.stack</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10911122402"><a name="p10911122402"></a><a name="p10911122402"></a>是</p>
</td>
</tr>
<tr id="row139114085311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p746362383711"><a name="p746362383711"></a><a name="p746362383711"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p109119214016"><a name="p109119214016"></a><a name="p109119214016"></a>torch.t</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p691152104016"><a name="p691152104016"></a><a name="p691152104016"></a>是</p>
</td>
</tr>
<tr id="row113918075314"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6463132393711"><a name="p6463132393711"></a><a name="p6463132393711"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9912526407"><a name="p9912526407"></a><a name="p9912526407"></a>torch.take</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p191212104011"><a name="p191212104011"></a><a name="p191212104011"></a>是</p>
</td>
</tr>
<tr id="row53912005535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1146320237370"><a name="p1146320237370"></a><a name="p1146320237370"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19123211403"><a name="p19123211403"></a><a name="p19123211403"></a>torch.transpose</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1691215218400"><a name="p1691215218400"></a><a name="p1691215218400"></a>是</p>
</td>
</tr>
<tr id="row133913019530"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1463202383715"><a name="p1463202383715"></a><a name="p1463202383715"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p591215214408"><a name="p591215214408"></a><a name="p591215214408"></a>torch.unbind</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p791217254020"><a name="p791217254020"></a><a name="p791217254020"></a>是</p>
</td>
</tr>
<tr id="row163915095317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p174639231377"><a name="p174639231377"></a><a name="p174639231377"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17912122194016"><a name="p17912122194016"></a><a name="p17912122194016"></a>torch.unsqueeze</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p109123219403"><a name="p109123219403"></a><a name="p109123219403"></a>是</p>
</td>
</tr>
<tr id="row73911205534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p114642233377"><a name="p114642233377"></a><a name="p114642233377"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15912122204018"><a name="p15912122204018"></a><a name="p15912122204018"></a>torch.where</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p199122210404"><a name="p199122210404"></a><a name="p199122210404"></a>是</p>
</td>
</tr>
</tbody>
</table>

<h2 id="Generators">Generators</h2>

<a name="table155791359155219"></a>
<table><thead align="left"><tr id="row038419012538"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p16697450143718"><a name="p16697450143718"></a><a name="p16697450143718"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row43857095315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p297125514377"><a name="p297125514377"></a><a name="p297125514377"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2819162154210"><a name="p2819162154210"></a><a name="p2819162154210"></a>torch._C.Generator</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p168191122427"><a name="p168191122427"></a><a name="p168191122427"></a>否</p>
</td>
</tr>
<tr id="row43858065310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p597175543718"><a name="p597175543718"></a><a name="p597175543718"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6819132124218"><a name="p6819132124218"></a><a name="p6819132124218"></a>torch._C.Generator.device</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7819182114210"><a name="p7819182114210"></a><a name="p7819182114210"></a>否</p>
</td>
</tr>
<tr id="row113851010539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p159715553720"><a name="p159715553720"></a><a name="p159715553720"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p108201621424"><a name="p108201621424"></a><a name="p108201621424"></a>torch._C.Generator.get_state</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1882014210423"><a name="p1882014210423"></a><a name="p1882014210423"></a>否</p>
</td>
</tr>
<tr id="row438518085316"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1597555183713"><a name="p1597555183713"></a><a name="p1597555183713"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p88205211428"><a name="p88205211428"></a><a name="p88205211428"></a>torch._C.Generator.initial_seed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p20820423421"><a name="p20820423421"></a><a name="p20820423421"></a>否</p>
</td>
</tr>
<tr id="row143859075317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p09725543717"><a name="p09725543717"></a><a name="p09725543717"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2820821425"><a name="p2820821425"></a><a name="p2820821425"></a>torch._C.Generator.manual_seed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p108201827429"><a name="p108201827429"></a><a name="p108201827429"></a>否</p>
</td>
</tr>
<tr id="row93851001536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p497145553715"><a name="p497145553715"></a><a name="p497145553715"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p88202022429"><a name="p88202022429"></a><a name="p88202022429"></a>torch._C.Generator.seed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p138201725421"><a name="p138201725421"></a><a name="p138201725421"></a>否</p>
</td>
</tr>
<tr id="row1338619014537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p49765513717"><a name="p49765513717"></a><a name="p49765513717"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16820192124218"><a name="p16820192124218"></a><a name="p16820192124218"></a>torch._C.Generator.set_state</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p982032104213"><a name="p982032104213"></a><a name="p982032104213"></a>否</p>
</td>
</tr>
</tbody>
</table>

<h2 id="Random-sampling">Random sampling</h2>

<a name="table155791359155219"></a>
<table><thead align="left"><tr id="row038419012538"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p382722114388"><a name="p382722114388"></a><a name="p382722114388"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row43857095315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13204617173914"><a name="p13204617173914"></a><a name="p13204617173914"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p127171230154512"><a name="p127171230154512"></a><a name="p127171230154512"></a>torch.seed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p17717830194513"><a name="p17717830194513"></a><a name="p17717830194513"></a>否</p>
</td>
</tr>
<tr id="row43858065310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1720431713391"><a name="p1720431713391"></a><a name="p1720431713391"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1371753034519"><a name="p1371753034519"></a><a name="p1371753034519"></a>torch.manual_seed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19717103054512"><a name="p19717103054512"></a><a name="p19717103054512"></a>否</p>
</td>
</tr>
<tr id="row113851010539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13205151783910"><a name="p13205151783910"></a><a name="p13205151783910"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19717103074516"><a name="p19717103074516"></a><a name="p19717103074516"></a>torch.initial_seed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p871713064519"><a name="p871713064519"></a><a name="p871713064519"></a>否</p>
</td>
</tr>
<tr id="row438518085316"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11205101703919"><a name="p11205101703919"></a><a name="p11205101703919"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p147171030134511"><a name="p147171030134511"></a><a name="p147171030134511"></a>torch.get_rng_state</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p671714300458"><a name="p671714300458"></a><a name="p671714300458"></a>否</p>
</td>
</tr>
<tr id="row143859075317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1920551714393"><a name="p1920551714393"></a><a name="p1920551714393"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9717153020451"><a name="p9717153020451"></a><a name="p9717153020451"></a>torch.set_rng_state</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8717143016452"><a name="p8717143016452"></a><a name="p8717143016452"></a>否</p>
</td>
</tr>
<tr id="row93851001536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12058170399"><a name="p12058170399"></a><a name="p12058170399"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p571814301455"><a name="p571814301455"></a><a name="p571814301455"></a>torch.torch.default_generator</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13718630154516"><a name="p13718630154516"></a><a name="p13718630154516"></a>否</p>
</td>
</tr>
<tr id="row1338619014537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1820551712395"><a name="p1820551712395"></a><a name="p1820551712395"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p127184309457"><a name="p127184309457"></a><a name="p127184309457"></a>torch.bernoulli</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p167181308456"><a name="p167181308456"></a><a name="p167181308456"></a>是</p>
</td>
</tr>
<tr id="row93863095319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16205517123920"><a name="p16205517123920"></a><a name="p16205517123920"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1371873074520"><a name="p1371873074520"></a><a name="p1371873074520"></a>torch.multinomial</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7718103014515"><a name="p7718103014515"></a><a name="p7718103014515"></a>是</p>
</td>
</tr>
<tr id="row53861002533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p020512176391"><a name="p020512176391"></a><a name="p020512176391"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p147181030174511"><a name="p147181030174511"></a><a name="p147181030174511"></a>torch.normal</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p197182030144520"><a name="p197182030144520"></a><a name="p197182030144520"></a>是</p>
</td>
</tr>
<tr id="row1038616085311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13205141720397"><a name="p13205141720397"></a><a name="p13205141720397"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19718103011453"><a name="p19718103011453"></a><a name="p19718103011453"></a>torch.poisson</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1771823014511"><a name="p1771823014511"></a><a name="p1771823014511"></a>否</p>
</td>
</tr>
<tr id="row16386170115319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p02056179390"><a name="p02056179390"></a><a name="p02056179390"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p3718133024518"><a name="p3718133024518"></a><a name="p3718133024518"></a>torch.rand</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p117181230194515"><a name="p117181230194515"></a><a name="p117181230194515"></a>是</p>
</td>
</tr>
<tr id="row238620105310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12205191753913"><a name="p12205191753913"></a><a name="p12205191753913"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15718143010457"><a name="p15718143010457"></a><a name="p15718143010457"></a>torch.rand_like</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p271819303453"><a name="p271819303453"></a><a name="p271819303453"></a>是</p>
</td>
</tr>
<tr id="row83871705537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p920519176399"><a name="p920519176399"></a><a name="p920519176399"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13718183018455"><a name="p13718183018455"></a><a name="p13718183018455"></a>torch.randint</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10718103013459"><a name="p10718103013459"></a><a name="p10718103013459"></a>是</p>
</td>
</tr>
<tr id="row1738750155312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p72051178398"><a name="p72051178398"></a><a name="p72051178398"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10718173012452"><a name="p10718173012452"></a><a name="p10718173012452"></a>torch.randint_like</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p4718130154520"><a name="p4718130154520"></a><a name="p4718130154520"></a>是</p>
</td>
</tr>
<tr id="row538717035315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p720515171392"><a name="p720515171392"></a><a name="p720515171392"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1771833018453"><a name="p1771833018453"></a><a name="p1771833018453"></a>torch.randn</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1471812303455"><a name="p1471812303455"></a><a name="p1471812303455"></a>是</p>
</td>
</tr>
<tr id="row8387180155310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p720551723915"><a name="p720551723915"></a><a name="p720551723915"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p771810301456"><a name="p771810301456"></a><a name="p771810301456"></a>torch.randn_like</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p071833054513"><a name="p071833054513"></a><a name="p071833054513"></a>是</p>
</td>
</tr>
<tr id="row18387190125313"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1920511718390"><a name="p1920511718390"></a><a name="p1920511718390"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p127195305457"><a name="p127195305457"></a><a name="p127195305457"></a>torch.randperm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p127191830184510"><a name="p127191830184510"></a><a name="p127191830184510"></a>是</p>
</td>
</tr>
<tr id="row9388190145310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p82051517173915"><a name="p82051517173915"></a><a name="p82051517173915"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p471915306451"><a name="p471915306451"></a><a name="p471915306451"></a>torch.Tensor.bernoulli_()</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p620972719507"><a name="p620972719507"></a><a name="p620972719507"></a>是</p>
</td>
</tr>
<tr id="row1738812045310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p82051717173915"><a name="p82051717173915"></a><a name="p82051717173915"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1771933034510"><a name="p1771933034510"></a><a name="p1771933034510"></a>torch.Tensor.bernoulli_()</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18896193117503"><a name="p18896193117503"></a><a name="p18896193117503"></a>是</p>
</td>
</tr>
<tr id="row193881309536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3205161710396"><a name="p3205161710396"></a><a name="p3205161710396"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p771953074517"><a name="p771953074517"></a><a name="p771953074517"></a>torch.Tensor.exponential_()</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1871913014520"><a name="p1871913014520"></a><a name="p1871913014520"></a>否</p>
</td>
</tr>
<tr id="row123882035311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5205161716396"><a name="p5205161716396"></a><a name="p5205161716396"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1171911300458"><a name="p1171911300458"></a><a name="p1171911300458"></a>torch.Tensor.geometric_()</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p77196306457"><a name="p77196306457"></a><a name="p77196306457"></a>否</p>
</td>
</tr>
<tr id="row238819065315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6206181783919"><a name="p6206181783919"></a><a name="p6206181783919"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6719230194519"><a name="p6719230194519"></a><a name="p6719230194519"></a>torch.Tensor.log_normal_()</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p171915308454"><a name="p171915308454"></a><a name="p171915308454"></a>否</p>
</td>
</tr>
<tr id="row63881302531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2206117123914"><a name="p2206117123914"></a><a name="p2206117123914"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p147191030194514"><a name="p147191030194514"></a><a name="p147191030194514"></a>torch.Tensor.normal_()</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p157201307459"><a name="p157201307459"></a><a name="p157201307459"></a>是</p>
</td>
</tr>
<tr id="row1038913095310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p162060174396"><a name="p162060174396"></a><a name="p162060174396"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p127206309452"><a name="p127206309452"></a><a name="p127206309452"></a>torch.Tensor.random_()</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0720203074518"><a name="p0720203074518"></a><a name="p0720203074518"></a>是</p>
</td>
</tr>
<tr id="row1738918085317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p52061417193914"><a name="p52061417193914"></a><a name="p52061417193914"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p37202308458"><a name="p37202308458"></a><a name="p37202308458"></a>torch.Tensor.uniform_()</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p072023054514"><a name="p072023054514"></a><a name="p072023054514"></a>是</p>
</td>
</tr>
<tr id="row133896075317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p152062017123914"><a name="p152062017123914"></a><a name="p152062017123914"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1372063017451"><a name="p1372063017451"></a><a name="p1372063017451"></a>torch.quasirandom.SobolEngine</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1972013019458"><a name="p1972013019458"></a><a name="p1972013019458"></a>否</p>
</td>
</tr>
<tr id="row838911017537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p320610175392"><a name="p320610175392"></a><a name="p320610175392"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1372033074519"><a name="p1372033074519"></a><a name="p1372033074519"></a>torch.quasirandom.SobolEngine.draw</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p27201030184519"><a name="p27201030184519"></a><a name="p27201030184519"></a>否</p>
</td>
</tr>
<tr id="row738918015532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p820614176397"><a name="p820614176397"></a><a name="p820614176397"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10720193024516"><a name="p10720193024516"></a><a name="p10720193024516"></a>torch.quasirandom.SobolEngine.fast_forward</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18720133084520"><a name="p18720133084520"></a><a name="p18720133084520"></a>否</p>
</td>
</tr>
<tr id="row7389190145312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8206191733917"><a name="p8206191733917"></a><a name="p8206191733917"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2720930184520"><a name="p2720930184520"></a><a name="p2720930184520"></a>torch.quasirandom.SobolEngine.reset</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5720103019455"><a name="p5720103019455"></a><a name="p5720103019455"></a>否</p>
</td>
</tr>
</tbody>
</table>

<h2 id="Serialization">Serialization</h2>

<a name="table155791359155219"></a>
<table><thead align="left"><tr id="row038419012538"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p12952172303918"><a name="p12952172303918"></a><a name="p12952172303918"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row43857095315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p69526235395"><a name="p69526235395"></a><a name="p69526235395"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p597074315716"><a name="p597074315716"></a><a name="p597074315716"></a>torch.save</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15970143135715"><a name="p15970143135715"></a><a name="p15970143135715"></a>是</p>
</td>
</tr>
<tr id="row43858065310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p495272353914"><a name="p495272353914"></a><a name="p495272353914"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1097014434578"><a name="p1097014434578"></a><a name="p1097014434578"></a>torch.load</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1797054313571"><a name="p1797054313571"></a><a name="p1797054313571"></a>是</p>
</td>
</tr>
</tbody>
</table>

<h2 id="Math-operations">Math operations</h2>

<a name="table155791359155219"></a>
<table><thead align="left"><tr id="row038419012538"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p7641165016399"><a name="p7641165016399"></a><a name="p7641165016399"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row43857095315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4619742184017"><a name="p4619742184017"></a><a name="p4619742184017"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1867133211015"><a name="p1867133211015"></a><a name="p1867133211015"></a>torch.abs</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9867632900"><a name="p9867632900"></a><a name="p9867632900"></a>是</p>
</td>
</tr>
<tr id="row43858065310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11619164211402"><a name="p11619164211402"></a><a name="p11619164211402"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p158671032900"><a name="p158671032900"></a><a name="p158671032900"></a>torch.acos</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p198673320014"><a name="p198673320014"></a><a name="p198673320014"></a>是</p>
</td>
</tr>
<tr id="row113851010539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p116191342144010"><a name="p116191342144010"></a><a name="p116191342144010"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p78678328016"><a name="p78678328016"></a><a name="p78678328016"></a>torch.add</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p158671532209"><a name="p158671532209"></a><a name="p158671532209"></a>是</p>
</td>
</tr>
<tr id="row438518085316"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15620134217403"><a name="p15620134217403"></a><a name="p15620134217403"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15867832302"><a name="p15867832302"></a><a name="p15867832302"></a>torch.addcdiv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1086716321203"><a name="p1086716321203"></a><a name="p1086716321203"></a>是</p>
</td>
</tr>
<tr id="row143859075317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1662094211407"><a name="p1662094211407"></a><a name="p1662094211407"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2867203213017"><a name="p2867203213017"></a><a name="p2867203213017"></a>torch.addcmul</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1418195216527"><a name="p1418195216527"></a><a name="p1418195216527"></a>是</p>
</td>
</tr>
<tr id="row93851001536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p166201542134012"><a name="p166201542134012"></a><a name="p166201542134012"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p208678321909"><a name="p208678321909"></a><a name="p208678321909"></a>torch.angle</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6867632400"><a name="p6867632400"></a><a name="p6867632400"></a>否</p>
</td>
</tr>
<tr id="row1338619014537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p146201425408"><a name="p146201425408"></a><a name="p146201425408"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1086717321105"><a name="p1086717321105"></a><a name="p1086717321105"></a>torch.asin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p086718321609"><a name="p086718321609"></a><a name="p086718321609"></a>是</p>
</td>
</tr>
<tr id="row93863095319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p196200428400"><a name="p196200428400"></a><a name="p196200428400"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p586810327019"><a name="p586810327019"></a><a name="p586810327019"></a>torch.atan</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p386816328014"><a name="p386816328014"></a><a name="p386816328014"></a>是</p>
</td>
</tr>
<tr id="row53861002533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2062044213409"><a name="p2062044213409"></a><a name="p2062044213409"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p28681532301"><a name="p28681532301"></a><a name="p28681532301"></a>torch.atan2</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1786863210010"><a name="p1786863210010"></a><a name="p1786863210010"></a>是</p>
</td>
</tr>
<tr id="row1038616085311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5620144220407"><a name="p5620144220407"></a><a name="p5620144220407"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p486803216010"><a name="p486803216010"></a><a name="p486803216010"></a>torch.bitwise_not</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p197612210531"><a name="p197612210531"></a><a name="p197612210531"></a>是</p>
</td>
</tr>
<tr id="row16386170115319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14620164294015"><a name="p14620164294015"></a><a name="p14620164294015"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6868932902"><a name="p6868932902"></a><a name="p6868932902"></a>torch.bitwise_and</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p52591727185310"><a name="p52591727185310"></a><a name="p52591727185310"></a>是</p>
</td>
</tr>
<tr id="row238620105310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17620942144014"><a name="p17620942144014"></a><a name="p17620942144014"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p98683321203"><a name="p98683321203"></a><a name="p98683321203"></a>torch.bitwise_or</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p268984315532"><a name="p268984315532"></a><a name="p268984315532"></a>是</p>
</td>
</tr>
<tr id="row83871705537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p662015425400"><a name="p662015425400"></a><a name="p662015425400"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13868143211011"><a name="p13868143211011"></a><a name="p13868143211011"></a>torch.bitwise_xor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p201312468537"><a name="p201312468537"></a><a name="p201312468537"></a>是</p>
</td>
</tr>
<tr id="row1738750155312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1462034211407"><a name="p1462034211407"></a><a name="p1462034211407"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p886803219010"><a name="p886803219010"></a><a name="p886803219010"></a>torch.ceil</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p986816322018"><a name="p986816322018"></a><a name="p986816322018"></a>是</p>
</td>
</tr>
<tr id="row538717035315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p96201242134015"><a name="p96201242134015"></a><a name="p96201242134015"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1486812322015"><a name="p1486812322015"></a><a name="p1486812322015"></a>torch.clamp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p686883218010"><a name="p686883218010"></a><a name="p686883218010"></a>是</p>
</td>
</tr>
<tr id="row8387180155310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3620174215409"><a name="p3620174215409"></a><a name="p3620174215409"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p38685321902"><a name="p38685321902"></a><a name="p38685321902"></a>torch.conj</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1868832907"><a name="p1868832907"></a><a name="p1868832907"></a>否</p>
</td>
</tr>
<tr id="row18387190125313"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3620104284015"><a name="p3620104284015"></a><a name="p3620104284015"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p986820321504"><a name="p986820321504"></a><a name="p986820321504"></a>torch.cos</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p58686328019"><a name="p58686328019"></a><a name="p58686328019"></a>是</p>
</td>
</tr>
<tr id="row9388190145310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1162019428408"><a name="p1162019428408"></a><a name="p1162019428408"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p286816329011"><a name="p286816329011"></a><a name="p286816329011"></a>torch.cosh</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p48681132603"><a name="p48681132603"></a><a name="p48681132603"></a>是</p>
</td>
</tr>
<tr id="row1738812045310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p26205425403"><a name="p26205425403"></a><a name="p26205425403"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p886973210016"><a name="p886973210016"></a><a name="p886973210016"></a>torch.div</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p98691632500"><a name="p98691632500"></a><a name="p98691632500"></a>是</p>
</td>
</tr>
<tr id="row193881309536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p86201425402"><a name="p86201425402"></a><a name="p86201425402"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8869113217010"><a name="p8869113217010"></a><a name="p8869113217010"></a>torch.digamma</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15869932706"><a name="p15869932706"></a><a name="p15869932706"></a>否</p>
</td>
</tr>
<tr id="row123882035311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18620184218406"><a name="p18620184218406"></a><a name="p18620184218406"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p118691432301"><a name="p118691432301"></a><a name="p118691432301"></a>torch.erf</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1086993212014"><a name="p1086993212014"></a><a name="p1086993212014"></a>是</p>
</td>
</tr>
<tr id="row238819065315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15620124224017"><a name="p15620124224017"></a><a name="p15620124224017"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1086983218018"><a name="p1086983218018"></a><a name="p1086983218018"></a>torch.erfc</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p108691832605"><a name="p108691832605"></a><a name="p108691832605"></a>否</p>
</td>
</tr>
<tr id="row63881302531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10620442114014"><a name="p10620442114014"></a><a name="p10620442114014"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p886918321019"><a name="p886918321019"></a><a name="p886918321019"></a>torch.erfinv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p187126477544"><a name="p187126477544"></a><a name="p187126477544"></a>是</p>
</td>
</tr>
<tr id="row1038913095310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10621174218406"><a name="p10621174218406"></a><a name="p10621174218406"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p986913321704"><a name="p986913321704"></a><a name="p986913321704"></a>torch.exp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9869232906"><a name="p9869232906"></a><a name="p9869232906"></a>是</p>
</td>
</tr>
<tr id="row1738918085317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p116213422403"><a name="p116213422403"></a><a name="p116213422403"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1786916321502"><a name="p1786916321502"></a><a name="p1786916321502"></a>torch.expm1</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6869193214011"><a name="p6869193214011"></a><a name="p6869193214011"></a>是</p>
</td>
</tr>
<tr id="row133896075317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17621842184014"><a name="p17621842184014"></a><a name="p17621842184014"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p198693321017"><a name="p198693321017"></a><a name="p198693321017"></a>torch.floor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p98694328010"><a name="p98694328010"></a><a name="p98694328010"></a>是</p>
</td>
</tr>
<tr id="row838911017537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1562134214401"><a name="p1562134214401"></a><a name="p1562134214401"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2869632409"><a name="p2869632409"></a><a name="p2869632409"></a>torch.floor_divide</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1869232204"><a name="p1869232204"></a><a name="p1869232204"></a>是</p>
</td>
</tr>
<tr id="row738918015532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1562115422403"><a name="p1562115422403"></a><a name="p1562115422403"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1686914321800"><a name="p1686914321800"></a><a name="p1686914321800"></a>torch.fmod</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p28698321017"><a name="p28698321017"></a><a name="p28698321017"></a>是</p>
</td>
</tr>
<tr id="row7389190145312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1662174216403"><a name="p1662174216403"></a><a name="p1662174216403"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188701321302"><a name="p188701321302"></a><a name="p188701321302"></a>torch.frac</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1687003211011"><a name="p1687003211011"></a><a name="p1687003211011"></a>是</p>
</td>
</tr>
<tr id="row19389603533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3621742134012"><a name="p3621742134012"></a><a name="p3621742134012"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p208703321306"><a name="p208703321306"></a><a name="p208703321306"></a>torch.imag</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1987019321809"><a name="p1987019321809"></a><a name="p1987019321809"></a>否</p>
</td>
</tr>
<tr id="row63901504535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9621642154012"><a name="p9621642154012"></a><a name="p9621642154012"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5870163211012"><a name="p5870163211012"></a><a name="p5870163211012"></a>torch.lerp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p02024338559"><a name="p02024338559"></a><a name="p02024338559"></a>是</p>
</td>
</tr>
<tr id="row183907075312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p962119422402"><a name="p962119422402"></a><a name="p962119422402"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188706329019"><a name="p188706329019"></a><a name="p188706329019"></a>torch.lgamma</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18711832201"><a name="p18711832201"></a><a name="p18711832201"></a>否</p>
</td>
</tr>
<tr id="row03905011532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1762144214020"><a name="p1762144214020"></a><a name="p1762144214020"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p687113321205"><a name="p687113321205"></a><a name="p687113321205"></a>torch.log</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p158711432209"><a name="p158711432209"></a><a name="p158711432209"></a>是</p>
</td>
</tr>
<tr id="row139012012538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7621184218406"><a name="p7621184218406"></a><a name="p7621184218406"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p3871632907"><a name="p3871632907"></a><a name="p3871632907"></a>torch.log10</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14871143219012"><a name="p14871143219012"></a><a name="p14871143219012"></a>是</p>
</td>
</tr>
<tr id="row339017014538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10621242164017"><a name="p10621242164017"></a><a name="p10621242164017"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p787112321309"><a name="p787112321309"></a><a name="p787112321309"></a>torch.log1p</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p78719321401"><a name="p78719321401"></a><a name="p78719321401"></a>是</p>
</td>
</tr>
<tr id="row1139013012539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p106211942134016"><a name="p106211942134016"></a><a name="p106211942134016"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17871113210017"><a name="p17871113210017"></a><a name="p17871113210017"></a>torch.log2</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p78715321104"><a name="p78715321104"></a><a name="p78715321104"></a>是</p>
</td>
</tr>
<tr id="row1839015014533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16211142194015"><a name="p16211142194015"></a><a name="p16211142194015"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p087110321207"><a name="p087110321207"></a><a name="p087110321207"></a>torch.logical_and</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1687143216015"><a name="p1687143216015"></a><a name="p1687143216015"></a>是</p>
</td>
</tr>
<tr id="row143901008539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11621842184017"><a name="p11621842184017"></a><a name="p11621842184017"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p48710325010"><a name="p48710325010"></a><a name="p48710325010"></a>torch.logical_not</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1987115327019"><a name="p1987115327019"></a><a name="p1987115327019"></a>是</p>
</td>
</tr>
<tr id="row193901901537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14621124224019"><a name="p14621124224019"></a><a name="p14621124224019"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1087118321904"><a name="p1087118321904"></a><a name="p1087118321904"></a>torch.logical_or</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8871193217012"><a name="p8871193217012"></a><a name="p8871193217012"></a>是</p>
</td>
</tr>
<tr id="row123911012535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1962134216401"><a name="p1962134216401"></a><a name="p1962134216401"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1887116325014"><a name="p1887116325014"></a><a name="p1887116325014"></a>torch.logical_xor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1987114322014"><a name="p1987114322014"></a><a name="p1987114322014"></a>是</p>
</td>
</tr>
<tr id="row139170115315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7621174204013"><a name="p7621174204013"></a><a name="p7621174204013"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1087115321901"><a name="p1087115321901"></a><a name="p1087115321901"></a>torch.mul</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6871103214019"><a name="p6871103214019"></a><a name="p6871103214019"></a>是</p>
</td>
</tr>
<tr id="row2391402531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8622742174010"><a name="p8622742174010"></a><a name="p8622742174010"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8871123220015"><a name="p8871123220015"></a><a name="p8871123220015"></a>torch.mvlgamma</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2871173218012"><a name="p2871173218012"></a><a name="p2871173218012"></a>否</p>
</td>
</tr>
<tr id="row139114085311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19622114210402"><a name="p19622114210402"></a><a name="p19622114210402"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1871183213019"><a name="p1871183213019"></a><a name="p1871183213019"></a>torch.neg</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p387117321002"><a name="p387117321002"></a><a name="p387117321002"></a>是</p>
</td>
</tr>
<tr id="row113918075314"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1362294213402"><a name="p1362294213402"></a><a name="p1362294213402"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p98721132405"><a name="p98721132405"></a><a name="p98721132405"></a>torch.polygamma</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p38728325016"><a name="p38728325016"></a><a name="p38728325016"></a>否</p>
</td>
</tr>
<tr id="row53912005535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p96228426403"><a name="p96228426403"></a><a name="p96228426403"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p48723327012"><a name="p48723327012"></a><a name="p48723327012"></a>torch.pow</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p887215322011"><a name="p887215322011"></a><a name="p887215322011"></a>是</p>
</td>
</tr>
<tr id="row133913019530"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p46221242134011"><a name="p46221242134011"></a><a name="p46221242134011"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1387216329019"><a name="p1387216329019"></a><a name="p1387216329019"></a>torch.real</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p58728321009"><a name="p58728321009"></a><a name="p58728321009"></a>是</p>
</td>
</tr>
<tr id="row163915095317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1362284213404"><a name="p1362284213404"></a><a name="p1362284213404"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p118720325012"><a name="p118720325012"></a><a name="p118720325012"></a>torch.reciprocal</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p68721932203"><a name="p68721932203"></a><a name="p68721932203"></a>是</p>
</td>
</tr>
<tr id="row73911205534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17622134210406"><a name="p17622134210406"></a><a name="p17622134210406"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1587218321011"><a name="p1587218321011"></a><a name="p1587218321011"></a>torch.remainder</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p108721132506"><a name="p108721132506"></a><a name="p108721132506"></a>是</p>
</td>
</tr>
<tr id="row83911506532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1062218420405"><a name="p1062218420405"></a><a name="p1062218420405"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1087213321012"><a name="p1087213321012"></a><a name="p1087213321012"></a>torch.round</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14872932408"><a name="p14872932408"></a><a name="p14872932408"></a>是</p>
</td>
</tr>
<tr id="row19392170185319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p262234244019"><a name="p262234244019"></a><a name="p262234244019"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p58721832702"><a name="p58721832702"></a><a name="p58721832702"></a>torch.rsqrt</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18724322016"><a name="p18724322016"></a><a name="p18724322016"></a>是</p>
</td>
</tr>
<tr id="row339218019535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p462210420406"><a name="p462210420406"></a><a name="p462210420406"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p20872123213014"><a name="p20872123213014"></a><a name="p20872123213014"></a>torch.sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p787273217015"><a name="p787273217015"></a><a name="p787273217015"></a>是</p>
</td>
</tr>
<tr id="row739213012531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p186225420404"><a name="p186225420404"></a><a name="p186225420404"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19872103219017"><a name="p19872103219017"></a><a name="p19872103219017"></a>torch.sign</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10872153218010"><a name="p10872153218010"></a><a name="p10872153218010"></a>是</p>
</td>
</tr>
<tr id="row4392110105317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p462219425408"><a name="p462219425408"></a><a name="p462219425408"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p787213217010"><a name="p787213217010"></a><a name="p787213217010"></a>torch.sin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p487220321001"><a name="p487220321001"></a><a name="p487220321001"></a>是</p>
</td>
</tr>
<tr id="row1939219020532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20622144211409"><a name="p20622144211409"></a><a name="p20622144211409"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p138721232809"><a name="p138721232809"></a><a name="p138721232809"></a>torch.sinh</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18872163219015"><a name="p18872163219015"></a><a name="p18872163219015"></a>是</p>
</td>
</tr>
<tr id="row63923025317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13622174244010"><a name="p13622174244010"></a><a name="p13622174244010"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5872123214011"><a name="p5872123214011"></a><a name="p5872123214011"></a>torch.sqrt</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p108721332200"><a name="p108721332200"></a><a name="p108721332200"></a>是</p>
</td>
</tr>
<tr id="row2039219014530"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p962217428404"><a name="p962217428404"></a><a name="p962217428404"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1287314321602"><a name="p1287314321602"></a><a name="p1287314321602"></a>torch.square</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p487316324010"><a name="p487316324010"></a><a name="p487316324010"></a>是</p>
</td>
</tr>
<tr id="row23921006538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p136221342104018"><a name="p136221342104018"></a><a name="p136221342104018"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188736320016"><a name="p188736320016"></a><a name="p188736320016"></a>torch.tan</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1387316329015"><a name="p1387316329015"></a><a name="p1387316329015"></a>是</p>
</td>
</tr>
<tr id="row8392140185315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p0622184234017"><a name="p0622184234017"></a><a name="p0622184234017"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p187323215011"><a name="p187323215011"></a><a name="p187323215011"></a>torch.tanh</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5873232607"><a name="p5873232607"></a><a name="p5873232607"></a>是</p>
</td>
</tr>
<tr id="row1739310165312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1062216427406"><a name="p1062216427406"></a><a name="p1062216427406"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1387311321603"><a name="p1387311321603"></a><a name="p1387311321603"></a>torch.true_divide</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p287383220014"><a name="p287383220014"></a><a name="p287383220014"></a>是</p>
</td>
</tr>
<tr id="row339315085320"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8622164215404"><a name="p8622164215404"></a><a name="p8622164215404"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p138731132709"><a name="p138731132709"></a><a name="p138731132709"></a>torch.trunc</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5873103211018"><a name="p5873103211018"></a><a name="p5873103211018"></a>是</p>
</td>
</tr>
<tr id="row11393170155312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11623194211407"><a name="p11623194211407"></a><a name="p11623194211407"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p4873203218015"><a name="p4873203218015"></a><a name="p4873203218015"></a>torch.argmax</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1687311321003"><a name="p1687311321003"></a><a name="p1687311321003"></a>是</p>
</td>
</tr>
<tr id="row9393401537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p862319421404"><a name="p862319421404"></a><a name="p862319421404"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13873332404"><a name="p13873332404"></a><a name="p13873332404"></a>torch.argmin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p188733321107"><a name="p188733321107"></a><a name="p188733321107"></a>是</p>
</td>
</tr>
<tr id="row203931000538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p262334210404"><a name="p262334210404"></a><a name="p262334210404"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1187353217015"><a name="p1187353217015"></a><a name="p1187353217015"></a>torch.dist</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p11873123212018"><a name="p11873123212018"></a><a name="p11873123212018"></a>是</p>
</td>
</tr>
<tr id="row23939025317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12623194215403"><a name="p12623194215403"></a><a name="p12623194215403"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1187333213018"><a name="p1187333213018"></a><a name="p1187333213018"></a>torch.logsumexp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p25151354311"><a name="p25151354311"></a><a name="p25151354311"></a>是</p>
</td>
</tr>
<tr id="row1339413025317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p126231442124012"><a name="p126231442124012"></a><a name="p126231442124012"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9873032308"><a name="p9873032308"></a><a name="p9873032308"></a>torch.mean</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p68735321605"><a name="p68735321605"></a><a name="p68735321605"></a>是</p>
</td>
</tr>
<tr id="row93943075317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16623154264018"><a name="p16623154264018"></a><a name="p16623154264018"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p48731032900"><a name="p48731032900"></a><a name="p48731032900"></a>torch.median</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p065114561718"><a name="p065114561718"></a><a name="p065114561718"></a>是</p>
</td>
</tr>
<tr id="row10394200145311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3623204213407"><a name="p3623204213407"></a><a name="p3623204213407"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p88737320012"><a name="p88737320012"></a><a name="p88737320012"></a>torch.mode</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p20873113216017"><a name="p20873113216017"></a><a name="p20873113216017"></a>否</p>
</td>
</tr>
<tr id="row63941804534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4623164224010"><a name="p4623164224010"></a><a name="p4623164224010"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13874532109"><a name="p13874532109"></a><a name="p13874532109"></a>torch.norm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p987419321508"><a name="p987419321508"></a><a name="p987419321508"></a>是</p>
</td>
</tr>
<tr id="row23942055320"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15623124254016"><a name="p15623124254016"></a><a name="p15623124254016"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p387423214010"><a name="p387423214010"></a><a name="p387423214010"></a>torch.prod</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p787418328019"><a name="p787418328019"></a><a name="p787418328019"></a>是</p>
</td>
</tr>
<tr id="row1139414017534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5623194218402"><a name="p5623194218402"></a><a name="p5623194218402"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p187433219015"><a name="p187433219015"></a><a name="p187433219015"></a>torch.std</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18741132302"><a name="p18741132302"></a><a name="p18741132302"></a>是</p>
</td>
</tr>
<tr id="row1394609538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p162364213402"><a name="p162364213402"></a><a name="p162364213402"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2874032704"><a name="p2874032704"></a><a name="p2874032704"></a>torch.std_mean</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p138743321900"><a name="p138743321900"></a><a name="p138743321900"></a>是</p>
</td>
</tr>
<tr id="row143947005311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2623144294020"><a name="p2623144294020"></a><a name="p2623144294020"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1987493219015"><a name="p1987493219015"></a><a name="p1987493219015"></a>torch.sum</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p28743325019"><a name="p28743325019"></a><a name="p28743325019"></a>是</p>
</td>
</tr>
<tr id="row339411011531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p76231742114020"><a name="p76231742114020"></a><a name="p76231742114020"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15874532802"><a name="p15874532802"></a><a name="p15874532802"></a>torch.unique</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p11874143217012"><a name="p11874143217012"></a><a name="p11874143217012"></a>是</p>
</td>
</tr>
<tr id="row1039518035310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p462364215403"><a name="p462364215403"></a><a name="p462364215403"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p187414322010"><a name="p187414322010"></a><a name="p187414322010"></a>torch.unique_consecutive</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3874143211010"><a name="p3874143211010"></a><a name="p3874143211010"></a>否</p>
</td>
</tr>
<tr id="row1539510018533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p162324213402"><a name="p162324213402"></a><a name="p162324213402"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1587433214015"><a name="p1587433214015"></a><a name="p1587433214015"></a>torch.var</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1874123216013"><a name="p1874123216013"></a><a name="p1874123216013"></a>否</p>
</td>
</tr>
<tr id="row339580135314"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6623104234014"><a name="p6623104234014"></a><a name="p6623104234014"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5874232503"><a name="p5874232503"></a><a name="p5874232503"></a>torch.var_mean</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p198742323010"><a name="p198742323010"></a><a name="p198742323010"></a>否</p>
</td>
</tr>
<tr id="row2039510015315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4623154210404"><a name="p4623154210404"></a><a name="p4623154210404"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p78741932907"><a name="p78741932907"></a><a name="p78741932907"></a>torch.allclose</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p208746327017"><a name="p208746327017"></a><a name="p208746327017"></a>是</p>
</td>
</tr>
<tr id="row153957075319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p26237429405"><a name="p26237429405"></a><a name="p26237429405"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p887418321902"><a name="p887418321902"></a><a name="p887418321902"></a>torch.argsort</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1287516321000"><a name="p1287516321000"></a><a name="p1287516321000"></a>是</p>
</td>
</tr>
<tr id="row193957014539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p96231842164019"><a name="p96231842164019"></a><a name="p96231842164019"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8875143218015"><a name="p8875143218015"></a><a name="p8875143218015"></a>torch.eq</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p28756322012"><a name="p28756322012"></a><a name="p28756322012"></a>是</p>
</td>
</tr>
<tr id="row33955015319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1262415421406"><a name="p1262415421406"></a><a name="p1262415421406"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p108751032405"><a name="p108751032405"></a><a name="p108751032405"></a>torch.equal</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p4875143215013"><a name="p4875143215013"></a><a name="p4875143215013"></a>是</p>
</td>
</tr>
<tr id="row83951035310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1762474274018"><a name="p1762474274018"></a><a name="p1762474274018"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p68757321007"><a name="p68757321007"></a><a name="p68757321007"></a>torch.ge</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p118759320015"><a name="p118759320015"></a><a name="p118759320015"></a>是</p>
</td>
</tr>
<tr id="row239510145313"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14624742154010"><a name="p14624742154010"></a><a name="p14624742154010"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p78756321007"><a name="p78756321007"></a><a name="p78756321007"></a>torch.gt</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p168756329013"><a name="p168756329013"></a><a name="p168756329013"></a>是</p>
</td>
</tr>
<tr id="row1239650105316"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1362454219402"><a name="p1362454219402"></a><a name="p1362454219402"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p0875153211016"><a name="p0875153211016"></a><a name="p0875153211016"></a>torch.isfinite</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6875532109"><a name="p6875532109"></a><a name="p6875532109"></a>是</p>
</td>
</tr>
<tr id="row1939613055319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p462414426405"><a name="p462414426405"></a><a name="p462414426405"></a>84</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p48751321010"><a name="p48751321010"></a><a name="p48751321010"></a>torch.isinf</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p387553218012"><a name="p387553218012"></a><a name="p387553218012"></a>是</p>
</td>
</tr>
<tr id="row339614019537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7624742174013"><a name="p7624742174013"></a><a name="p7624742174013"></a>85</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1287523214012"><a name="p1287523214012"></a><a name="p1287523214012"></a>torch.isnan</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p987513321018"><a name="p987513321018"></a><a name="p987513321018"></a>是</p>
</td>
</tr>
<tr id="row4396130195317"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13624144244016"><a name="p13624144244016"></a><a name="p13624144244016"></a>86</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p128757327017"><a name="p128757327017"></a><a name="p128757327017"></a>torch.kthvalue</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1487593212011"><a name="p1487593212011"></a><a name="p1487593212011"></a>是</p>
</td>
</tr>
<tr id="row103961903533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5624124224011"><a name="p5624124224011"></a><a name="p5624124224011"></a>87</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1887514324012"><a name="p1887514324012"></a><a name="p1887514324012"></a>torch.le</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3875103211014"><a name="p3875103211014"></a><a name="p3875103211014"></a>是</p>
</td>
</tr>
<tr id="row4396202532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p76249428406"><a name="p76249428406"></a><a name="p76249428406"></a>88</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p198754326016"><a name="p198754326016"></a><a name="p198754326016"></a>torch.lt</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p587517321015"><a name="p587517321015"></a><a name="p587517321015"></a>是</p>
</td>
</tr>
<tr id="row123968015310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12624144274015"><a name="p12624144274015"></a><a name="p12624144274015"></a>89</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8875153219011"><a name="p8875153219011"></a><a name="p8875153219011"></a>torch.max</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p487510321002"><a name="p487510321002"></a><a name="p487510321002"></a>是</p>
</td>
</tr>
<tr id="row139614010530"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1662404213408"><a name="p1662404213408"></a><a name="p1662404213408"></a>90</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188751132300"><a name="p188751132300"></a><a name="p188751132300"></a>torch.min</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1187515321505"><a name="p1187515321505"></a><a name="p1187515321505"></a>是</p>
</td>
</tr>
<tr id="row839711065319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p26244424406"><a name="p26244424406"></a><a name="p26244424406"></a>91</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8876183213010"><a name="p8876183213010"></a><a name="p8876183213010"></a>torch.ne</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1587612321007"><a name="p1587612321007"></a><a name="p1587612321007"></a>是</p>
</td>
</tr>
<tr id="row1239711075320"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p106246426405"><a name="p106246426405"></a><a name="p106246426405"></a>92</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p387613321704"><a name="p387613321704"></a><a name="p387613321704"></a>torch.sort</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p987653214019"><a name="p987653214019"></a><a name="p987653214019"></a>是</p>
</td>
</tr>
<tr id="row139715065315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p56241842114020"><a name="p56241842114020"></a><a name="p56241842114020"></a>93</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p587683210011"><a name="p587683210011"></a><a name="p587683210011"></a>torch.topk</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2087613327014"><a name="p2087613327014"></a><a name="p2087613327014"></a>是</p>
</td>
</tr>
<tr id="row439713018536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2062464215406"><a name="p2062464215406"></a><a name="p2062464215406"></a>94</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1487620321006"><a name="p1487620321006"></a><a name="p1487620321006"></a>torch.fft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p178765323015"><a name="p178765323015"></a><a name="p178765323015"></a>否</p>
</td>
</tr>
<tr id="row139719014531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p162434234019"><a name="p162434234019"></a><a name="p162434234019"></a>95</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11876332702"><a name="p11876332702"></a><a name="p11876332702"></a>torch.ifft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2876232201"><a name="p2876232201"></a><a name="p2876232201"></a>否</p>
</td>
</tr>
<tr id="row193971014532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6624194264018"><a name="p6624194264018"></a><a name="p6624194264018"></a>96</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19876183215012"><a name="p19876183215012"></a><a name="p19876183215012"></a>torch.rfft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p38761321007"><a name="p38761321007"></a><a name="p38761321007"></a>否</p>
</td>
</tr>
<tr id="row18397120185318"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p662434274016"><a name="p662434274016"></a><a name="p662434274016"></a>97</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15876113214011"><a name="p15876113214011"></a><a name="p15876113214011"></a>torch.irfft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p28763320016"><a name="p28763320016"></a><a name="p28763320016"></a>否</p>
</td>
</tr>
<tr id="row639718095313"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17624642164011"><a name="p17624642164011"></a><a name="p17624642164011"></a>98</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188762032902"><a name="p188762032902"></a><a name="p188762032902"></a>torch.stft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p168761322016"><a name="p168761322016"></a><a name="p168761322016"></a>否</p>
</td>
</tr>
<tr id="row5397180125320"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p96252042124020"><a name="p96252042124020"></a><a name="p96252042124020"></a>99</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16876103214012"><a name="p16876103214012"></a><a name="p16876103214012"></a>torch.bartlett_window</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1230179150"><a name="p1230179150"></a><a name="p1230179150"></a>是</p>
</td>
</tr>
<tr id="row15398302537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1062594215408"><a name="p1062594215408"></a><a name="p1062594215408"></a>100</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p148765322011"><a name="p148765322011"></a><a name="p148765322011"></a>torch.blackman_window</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p523510161857"><a name="p523510161857"></a><a name="p523510161857"></a>是</p>
</td>
</tr>
<tr id="row1939890135314"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13625242134011"><a name="p13625242134011"></a><a name="p13625242134011"></a>101</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p198762323017"><a name="p198762323017"></a><a name="p198762323017"></a>torch.hamming_window</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15907317756"><a name="p15907317756"></a><a name="p15907317756"></a>是</p>
</td>
</tr>
<tr id="row839814012534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15625104220405"><a name="p15625104220405"></a><a name="p15625104220405"></a>102</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p0876732903"><a name="p0876732903"></a><a name="p0876732903"></a>torch.hann_window</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p195391419553"><a name="p195391419553"></a><a name="p195391419553"></a>是</p>
</td>
</tr>
<tr id="row20398600534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14625164213409"><a name="p14625164213409"></a><a name="p14625164213409"></a>103</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p88767321011"><a name="p88767321011"></a><a name="p88767321011"></a>torch.bincount</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p78771132408"><a name="p78771132408"></a><a name="p78771132408"></a>否</p>
</td>
</tr>
<tr id="row123981708536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1625042174017"><a name="p1625042174017"></a><a name="p1625042174017"></a>104</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p78771832003"><a name="p78771832003"></a><a name="p78771832003"></a>torch.broadcast_tensors</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1887712321509"><a name="p1887712321509"></a><a name="p1887712321509"></a>是</p>
</td>
</tr>
<tr id="row5398809537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9625164244010"><a name="p9625164244010"></a><a name="p9625164244010"></a>105</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p787717321103"><a name="p787717321103"></a><a name="p787717321103"></a>torch.cartesian_prod</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p28775321309"><a name="p28775321309"></a><a name="p28775321309"></a>是</p>
</td>
</tr>
<tr id="row103980011534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5625154220407"><a name="p5625154220407"></a><a name="p5625154220407"></a>106</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1287712324015"><a name="p1287712324015"></a><a name="p1287712324015"></a>torch.cdist</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15901434518"><a name="p15901434518"></a><a name="p15901434518"></a>是</p>
</td>
</tr>
<tr id="row639812045319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17625342184013"><a name="p17625342184013"></a><a name="p17625342184013"></a>107</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p387710322012"><a name="p387710322012"></a><a name="p387710322012"></a>torch.combinations</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p128781832308"><a name="p128781832308"></a><a name="p128781832308"></a>否</p>
</td>
</tr>
<tr id="row139814015318"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11625104211401"><a name="p11625104211401"></a><a name="p11625104211401"></a>108</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1487819321201"><a name="p1487819321201"></a><a name="p1487819321201"></a>torch.cross</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p78783323016"><a name="p78783323016"></a><a name="p78783323016"></a>是</p>
</td>
</tr>
<tr id="row17399307538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1462594284015"><a name="p1462594284015"></a><a name="p1462594284015"></a>109</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p48781832301"><a name="p48781832301"></a><a name="p48781832301"></a>torch.cummax</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p158788321506"><a name="p158788321506"></a><a name="p158788321506"></a>否</p>
</td>
</tr>
<tr id="row839916045311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16625134212409"><a name="p16625134212409"></a><a name="p16625134212409"></a>110</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p78781832607"><a name="p78781832607"></a><a name="p78781832607"></a>torch.cummin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19878183219019"><a name="p19878183219019"></a><a name="p19878183219019"></a>是</p>
</td>
</tr>
<tr id="row13996015537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14625144214010"><a name="p14625144214010"></a><a name="p14625144214010"></a>111</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p38788329012"><a name="p38788329012"></a><a name="p38788329012"></a>torch.cumprod</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15878432108"><a name="p15878432108"></a><a name="p15878432108"></a>是</p>
</td>
</tr>
<tr id="row239980105314"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p136251442144010"><a name="p136251442144010"></a><a name="p136251442144010"></a>112</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p138781032100"><a name="p138781032100"></a><a name="p138781032100"></a>torch.cumsum</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p198788321707"><a name="p198788321707"></a><a name="p198788321707"></a>是</p>
</td>
</tr>
<tr id="row103996085310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p762534234011"><a name="p762534234011"></a><a name="p762534234011"></a>113</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p287816323017"><a name="p287816323017"></a><a name="p287816323017"></a>torch.diag</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p133116391963"><a name="p133116391963"></a><a name="p133116391963"></a>是</p>
</td>
</tr>
<tr id="row3399801538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1162594244014"><a name="p1162594244014"></a><a name="p1162594244014"></a>114</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5878173212011"><a name="p5878173212011"></a><a name="p5878173212011"></a>torch.diag_embed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p128789321704"><a name="p128789321704"></a><a name="p128789321704"></a>是</p>
</td>
</tr>
<tr id="row339970165312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8625174216404"><a name="p8625174216404"></a><a name="p8625174216404"></a>115</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p178788321201"><a name="p178788321201"></a><a name="p178788321201"></a>torch.diagflat</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5879183213013"><a name="p5879183213013"></a><a name="p5879183213013"></a>是</p>
</td>
</tr>
<tr id="row204001504538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16625104224013"><a name="p16625104224013"></a><a name="p16625104224013"></a>116</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14879193218015"><a name="p14879193218015"></a><a name="p14879193218015"></a>torch.diagonal</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2879432208"><a name="p2879432208"></a><a name="p2879432208"></a>是</p>
</td>
</tr>
<tr id="row18400609537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p262524212409"><a name="p262524212409"></a><a name="p262524212409"></a>117</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1187973212015"><a name="p1187973212015"></a><a name="p1187973212015"></a>torch.einsum</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p887917321709"><a name="p887917321709"></a><a name="p887917321709"></a>是</p>
</td>
</tr>
<tr id="row740010017539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6625442144018"><a name="p6625442144018"></a><a name="p6625442144018"></a>118</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p587983215012"><a name="p587983215012"></a><a name="p587983215012"></a>torch.flatten</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p487983214019"><a name="p487983214019"></a><a name="p487983214019"></a>是</p>
</td>
</tr>
<tr id="row1340014016533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p762624204019"><a name="p762624204019"></a><a name="p762624204019"></a>119</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1587993212016"><a name="p1587993212016"></a><a name="p1587993212016"></a>torch.flip</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5879163218011"><a name="p5879163218011"></a><a name="p5879163218011"></a>是</p>
</td>
</tr>
<tr id="row64001702538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p462634234015"><a name="p462634234015"></a><a name="p462634234015"></a>120</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1287923220015"><a name="p1287923220015"></a><a name="p1287923220015"></a>torch.rot90</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p108795325017"><a name="p108795325017"></a><a name="p108795325017"></a>是</p>
</td>
</tr>
<tr id="row640013014535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p462610428404"><a name="p462610428404"></a><a name="p462610428404"></a>121</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p118791732902"><a name="p118791732902"></a><a name="p118791732902"></a>torch.histc</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p28791532205"><a name="p28791532205"></a><a name="p28791532205"></a>否</p>
</td>
</tr>
<tr id="row64001101538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p462694284017"><a name="p462694284017"></a><a name="p462694284017"></a>122</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p98790321607"><a name="p98790321607"></a><a name="p98790321607"></a>torch.meshgrid</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9879133219014"><a name="p9879133219014"></a><a name="p9879133219014"></a>是</p>
</td>
</tr>
<tr id="row1040030115315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1262694216404"><a name="p1262694216404"></a><a name="p1262694216404"></a>123</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p7879123213016"><a name="p7879123213016"></a><a name="p7879123213016"></a>torch.renorm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p287919321105"><a name="p287919321105"></a><a name="p287919321105"></a>是</p>
</td>
</tr>
<tr id="row18400009533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1962654254018"><a name="p1962654254018"></a><a name="p1962654254018"></a>124</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1287916321801"><a name="p1287916321801"></a><a name="p1287916321801"></a>torch.repeat_interleave</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9879632404"><a name="p9879632404"></a><a name="p9879632404"></a>否</p>
</td>
</tr>
<tr id="row19400190165310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12626124213408"><a name="p12626124213408"></a><a name="p12626124213408"></a>125</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p178794327015"><a name="p178794327015"></a><a name="p178794327015"></a>torch.roll</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p138791832202"><a name="p138791832202"></a><a name="p138791832202"></a>否</p>
</td>
</tr>
<tr id="row340115095310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1262634294020"><a name="p1262634294020"></a><a name="p1262634294020"></a>126</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8879732401"><a name="p8879732401"></a><a name="p8879732401"></a>torch.tensordot</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1387916326017"><a name="p1387916326017"></a><a name="p1387916326017"></a>是</p>
</td>
</tr>
<tr id="row6401200115311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p662616424409"><a name="p662616424409"></a><a name="p662616424409"></a>127</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p487910328017"><a name="p487910328017"></a><a name="p487910328017"></a>torch.trace</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p118791321104"><a name="p118791321104"></a><a name="p118791321104"></a>否</p>
</td>
</tr>
<tr id="row104011002534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1762624219403"><a name="p1762624219403"></a><a name="p1762624219403"></a>128</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1087918324016"><a name="p1087918324016"></a><a name="p1087918324016"></a>torch.tril</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13873173591312"><a name="p13873173591312"></a><a name="p13873173591312"></a>是</p>
</td>
</tr>
<tr id="row2040119012536"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10626114218405"><a name="p10626114218405"></a><a name="p10626114218405"></a>129</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1088018321506"><a name="p1088018321506"></a><a name="p1088018321506"></a>torch.tril_indices</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7880332001"><a name="p7880332001"></a><a name="p7880332001"></a>否</p>
</td>
</tr>
<tr id="row1840115025310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p146264425403"><a name="p146264425403"></a><a name="p146264425403"></a>130</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11880132505"><a name="p11880132505"></a><a name="p11880132505"></a>torch.triu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p488033215019"><a name="p488033215019"></a><a name="p488033215019"></a>是</p>
</td>
</tr>
<tr id="row14011102531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p106261429409"><a name="p106261429409"></a><a name="p106261429409"></a>131</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p28801032606"><a name="p28801032606"></a><a name="p28801032606"></a>torch.triu_indices</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p168802321904"><a name="p168802321904"></a><a name="p168802321904"></a>否</p>
</td>
</tr>
<tr id="row640119035312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11626204224011"><a name="p11626204224011"></a><a name="p11626204224011"></a>132</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p198801132409"><a name="p198801132409"></a><a name="p198801132409"></a>torch.addbmm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1321401816154"><a name="p1321401816154"></a><a name="p1321401816154"></a>是</p>
</td>
</tr>
<tr id="row240117085315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p662614213408"><a name="p662614213408"></a><a name="p662614213408"></a>133</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p788093216017"><a name="p788093216017"></a><a name="p788093216017"></a>torch.addmm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0984101961511"><a name="p0984101961511"></a><a name="p0984101961511"></a>是</p>
</td>
</tr>
<tr id="row184025005313"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p66261442194012"><a name="p66261442194012"></a><a name="p66261442194012"></a>134</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1988016323011"><a name="p1988016323011"></a><a name="p1988016323011"></a>torch.addmv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p118752219159"><a name="p118752219159"></a><a name="p118752219159"></a>是</p>
</td>
</tr>
<tr id="row114023035316"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12626442124015"><a name="p12626442124015"></a><a name="p12626442124015"></a>135</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2880183210011"><a name="p2880183210011"></a><a name="p2880183210011"></a>torch.addr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p183491324171511"><a name="p183491324171511"></a><a name="p183491324171511"></a>是</p>
</td>
</tr>
<tr id="row144022085310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p762644216401"><a name="p762644216401"></a><a name="p762644216401"></a>136</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1988019322017"><a name="p1988019322017"></a><a name="p1988019322017"></a>torch.baddbmm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p58801323017"><a name="p58801323017"></a><a name="p58801323017"></a>是</p>
</td>
</tr>
<tr id="row1440217015311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1626742104018"><a name="p1626742104018"></a><a name="p1626742104018"></a>137</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p98806321906"><a name="p98806321906"></a><a name="p98806321906"></a>torch.bmm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1688015321205"><a name="p1688015321205"></a><a name="p1688015321205"></a>是</p>
</td>
</tr>
<tr id="row1340490205319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1562744211408"><a name="p1562744211408"></a><a name="p1562744211408"></a>138</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6880123217010"><a name="p6880123217010"></a><a name="p6880123217010"></a>torch.chain_matmul</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p788023216010"><a name="p788023216010"></a><a name="p788023216010"></a>是</p>
</td>
</tr>
<tr id="row1940410015314"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8627942104010"><a name="p8627942104010"></a><a name="p8627942104010"></a>139</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p178803323015"><a name="p178803323015"></a><a name="p178803323015"></a>torch.cholesky</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p168808323017"><a name="p168808323017"></a><a name="p168808323017"></a>否</p>
</td>
</tr>
<tr id="row54041206535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p562794213402"><a name="p562794213402"></a><a name="p562794213402"></a>140</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p198805328020"><a name="p198805328020"></a><a name="p198805328020"></a>torch.cholesky_inverse</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p188073215015"><a name="p188073215015"></a><a name="p188073215015"></a>否</p>
</td>
</tr>
<tr id="row1440519085311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1462764218403"><a name="p1462764218403"></a><a name="p1462764218403"></a>141</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18881183219012"><a name="p18881183219012"></a><a name="p18881183219012"></a>torch.cholesky_solve</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0881163219016"><a name="p0881163219016"></a><a name="p0881163219016"></a>否</p>
</td>
</tr>
<tr id="row9405301534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16627642194012"><a name="p16627642194012"></a><a name="p16627642194012"></a>142</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p158813321502"><a name="p158813321502"></a><a name="p158813321502"></a>torch.dot</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p58817321900"><a name="p58817321900"></a><a name="p58817321900"></a>否</p>
</td>
</tr>
<tr id="row94051306531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11627242104014"><a name="p11627242104014"></a><a name="p11627242104014"></a>143</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p118811232705"><a name="p118811232705"></a><a name="p118811232705"></a>torch.eig</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1888123218014"><a name="p1888123218014"></a><a name="p1888123218014"></a>否</p>
</td>
</tr>
<tr id="row540511025311"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p162764284014"><a name="p162764284014"></a><a name="p162764284014"></a>144</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p08817321207"><a name="p08817321207"></a><a name="p08817321207"></a>torch.geqrf</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p588119321404"><a name="p588119321404"></a><a name="p588119321404"></a>否</p>
</td>
</tr>
<tr id="row17405206531"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16271342184012"><a name="p16271342184012"></a><a name="p16271342184012"></a>145</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1888117321103"><a name="p1888117321103"></a><a name="p1888117321103"></a>torch.ger</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p48818321609"><a name="p48818321609"></a><a name="p48818321609"></a>是</p>
</td>
</tr>
<tr id="row1440520015530"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13627442194016"><a name="p13627442194016"></a><a name="p13627442194016"></a>146</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p788163210010"><a name="p788163210010"></a><a name="p788163210010"></a>torch.inverse</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1381973018373"><a name="p1381973018373"></a><a name="p1381973018373"></a>是</p>
</td>
</tr>
<tr id="row1340515045313"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p462712423409"><a name="p462712423409"></a><a name="p462712423409"></a>147</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p98815327017"><a name="p98815327017"></a><a name="p98815327017"></a>torch.det</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p488112321805"><a name="p488112321805"></a><a name="p488112321805"></a>否</p>
</td>
</tr>
<tr id="row124056017532"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p762784217401"><a name="p762784217401"></a><a name="p762784217401"></a>148</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1588114321300"><a name="p1588114321300"></a><a name="p1588114321300"></a>torch.logdet</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p128813322013"><a name="p128813322013"></a><a name="p128813322013"></a>否</p>
</td>
</tr>
<tr id="row74056012538"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p106271342164011"><a name="p106271342164011"></a><a name="p106271342164011"></a>149</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16881632902"><a name="p16881632902"></a><a name="p16881632902"></a>torch.slogdet</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9881133215014"><a name="p9881133215014"></a><a name="p9881133215014"></a>是</p>
</td>
</tr>
<tr id="row24061102533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13627164217405"><a name="p13627164217405"></a><a name="p13627164217405"></a>150</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1488117321506"><a name="p1488117321506"></a><a name="p1488117321506"></a>torch.lstsq</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1988133216018"><a name="p1988133216018"></a><a name="p1988133216018"></a>否</p>
</td>
</tr>
<tr id="row04064011533"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p156271442164015"><a name="p156271442164015"></a><a name="p156271442164015"></a>151</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1088115322002"><a name="p1088115322002"></a><a name="p1088115322002"></a>torch.lu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14881832304"><a name="p14881832304"></a><a name="p14881832304"></a>否</p>
</td>
</tr>
<tr id="row18406160115310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15627942134015"><a name="p15627942134015"></a><a name="p15627942134015"></a>152</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18812032302"><a name="p18812032302"></a><a name="p18812032302"></a>torch.lu_solve</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12881153215010"><a name="p12881153215010"></a><a name="p12881153215010"></a>否</p>
</td>
</tr>
<tr id="row4406130195312"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p66271542104018"><a name="p66271542104018"></a><a name="p66271542104018"></a>153</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p178812321509"><a name="p178812321509"></a><a name="p178812321509"></a>torch.lu_unpack</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14881193213017"><a name="p14881193213017"></a><a name="p14881193213017"></a>否</p>
</td>
</tr>
<tr id="row4406190145310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8627204244017"><a name="p8627204244017"></a><a name="p8627204244017"></a>154</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p588263212014"><a name="p588263212014"></a><a name="p588263212014"></a>torch.matmul</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1388218328016"><a name="p1388218328016"></a><a name="p1388218328016"></a>是</p>
</td>
</tr>
<tr id="row6406305530"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p166271842184015"><a name="p166271842184015"></a><a name="p166271842184015"></a>155</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p4882103211012"><a name="p4882103211012"></a><a name="p4882103211012"></a>torch.matrix_power</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p28821232108"><a name="p28821232108"></a><a name="p28821232108"></a>是</p>
</td>
</tr>
<tr id="row14406160125318"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16271742194015"><a name="p16271742194015"></a><a name="p16271742194015"></a>156</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p38827323019"><a name="p38827323019"></a><a name="p38827323019"></a>torch.matrix_rank</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16882532306"><a name="p16882532306"></a><a name="p16882532306"></a>否</p>
</td>
</tr>
<tr id="row17406500530"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1062711429402"><a name="p1062711429402"></a><a name="p1062711429402"></a>157</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5882032305"><a name="p5882032305"></a><a name="p5882032305"></a>torch.mm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p11882163210019"><a name="p11882163210019"></a><a name="p11882163210019"></a>是</p>
</td>
</tr>
<tr id="row64061205534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1162814423407"><a name="p1162814423407"></a><a name="p1162814423407"></a>158</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188219321603"><a name="p188219321603"></a><a name="p188219321603"></a>torch.mv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p126884235428"><a name="p126884235428"></a><a name="p126884235428"></a>是</p>
</td>
</tr>
<tr id="row34071203537"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5628342184015"><a name="p5628342184015"></a><a name="p5628342184015"></a>159</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p888217321202"><a name="p888217321202"></a><a name="p888217321202"></a>torch.orgqr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p148829321201"><a name="p148829321201"></a><a name="p148829321201"></a>否</p>
</td>
</tr>
<tr id="row104071105535"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4628142144018"><a name="p4628142144018"></a><a name="p4628142144018"></a>160</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p148821332703"><a name="p148821332703"></a><a name="p148821332703"></a>torch.ormqr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p138821832804"><a name="p138821832804"></a><a name="p138821832804"></a>否</p>
</td>
</tr>
<tr id="row154071206539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p106280429405"><a name="p106280429405"></a><a name="p106280429405"></a>161</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11882332702"><a name="p11882332702"></a><a name="p11882332702"></a>torch.pinverse</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p588213328019"><a name="p588213328019"></a><a name="p588213328019"></a>否</p>
</td>
</tr>
<tr id="row740711045315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p56281242204011"><a name="p56281242204011"></a><a name="p56281242204011"></a>162</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1488253220014"><a name="p1488253220014"></a><a name="p1488253220014"></a>torch.qr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p20882432402"><a name="p20882432402"></a><a name="p20882432402"></a>是</p>
</td>
</tr>
<tr id="row64071004534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3628042144017"><a name="p3628042144017"></a><a name="p3628042144017"></a>163</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188824321902"><a name="p188824321902"></a><a name="p188824321902"></a>torch.solve</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p118821232404"><a name="p118821232404"></a><a name="p118821232404"></a>否</p>
</td>
</tr>
<tr id="row1840718075310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17628144212409"><a name="p17628144212409"></a><a name="p17628144212409"></a>164</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1088213329013"><a name="p1088213329013"></a><a name="p1088213329013"></a>torch.svd</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6882153210010"><a name="p6882153210010"></a><a name="p6882153210010"></a>否</p>
</td>
</tr>
<tr id="row74071017534"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7628124218408"><a name="p7628124218408"></a><a name="p7628124218408"></a>165</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16882173210017"><a name="p16882173210017"></a><a name="p16882173210017"></a>torch.svd_lowrank</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p988243219014"><a name="p988243219014"></a><a name="p988243219014"></a>否</p>
</td>
</tr>
<tr id="row340717035314"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19628842174018"><a name="p19628842174018"></a><a name="p19628842174018"></a>166</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p48822326015"><a name="p48822326015"></a><a name="p48822326015"></a>torch.pca_lowrank</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2883432508"><a name="p2883432508"></a><a name="p2883432508"></a>否</p>
</td>
</tr>
<tr id="row1840712045314"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12628842184016"><a name="p12628842184016"></a><a name="p12628842184016"></a>167</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p3883432304"><a name="p3883432304"></a><a name="p3883432304"></a>torch.symeig</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8883163218013"><a name="p8883163218013"></a><a name="p8883163218013"></a>否</p>
</td>
</tr>
<tr id="row154081406539"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3628204212404"><a name="p3628204212404"></a><a name="p3628204212404"></a>168</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188831232801"><a name="p188831232801"></a><a name="p188831232801"></a>torch.lobpcg</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p118831732602"><a name="p118831732602"></a><a name="p118831732602"></a>否</p>
</td>
</tr>
<tr id="row4408110165319"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1462894214408"><a name="p1462894214408"></a><a name="p1462894214408"></a>169</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2088319328011"><a name="p2088319328011"></a><a name="p2088319328011"></a>torch.trapz</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p168837321706"><a name="p168837321706"></a><a name="p168837321706"></a>是</p>
</td>
</tr>
<tr id="row11408180115316"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7628164214408"><a name="p7628164214408"></a><a name="p7628164214408"></a>170</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18831532508"><a name="p18831532508"></a><a name="p18831532508"></a>torch.triangular_solve</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2088333216013"><a name="p2088333216013"></a><a name="p2088333216013"></a>否</p>
</td>
</tr>
</tbody>
</table>

<h2 id="Utilities">Utilities</h2>

<a name="table155791359155219"></a>
<table><thead align="left"><tr id="row038419012538"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p22242584402"><a name="p22242584402"></a><a name="p22242584402"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row43857095315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p122445854019"><a name="p122445854019"></a><a name="p122445854019"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p87813018118"><a name="p87813018118"></a><a name="p87813018118"></a>torch.compiled_with_cxx11_abi</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p187843018115"><a name="p187843018115"></a><a name="p187843018115"></a>是</p>
</td>
</tr>
<tr id="row43858065310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2224155854020"><a name="p2224155854020"></a><a name="p2224155854020"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1578230915"><a name="p1578230915"></a><a name="p1578230915"></a>torch.result_type</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p107813301117"><a name="p107813301117"></a><a name="p107813301117"></a>是</p>
</td>
</tr>
<tr id="row3997191714119"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12241958124018"><a name="p12241958124018"></a><a name="p12241958124018"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p137815301612"><a name="p137815301612"></a><a name="p137815301612"></a>torch.can_cast</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p197812306116"><a name="p197812306116"></a><a name="p197812306116"></a>是</p>
</td>
</tr>
<tr id="row168295201413"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p132241558204019"><a name="p132241558204019"></a><a name="p132241558204019"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1785302113"><a name="p1785302113"></a><a name="p1785302113"></a>torch.promote_types</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p378173014114"><a name="p378173014114"></a><a name="p378173014114"></a>是</p>
</td>
</tr>
</tbody>
</table>

<h2 id="Other">Other</h2>

<a name="table155791359155219"></a>
<table><thead align="left"><tr id="row038419012538"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p143036271411"><a name="p143036271411"></a><a name="p143036271411"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row43857095315"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9572125434116"><a name="p9572125434116"></a><a name="p9572125434116"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p101541879210"><a name="p101541879210"></a><a name="p101541879210"></a>torch.no_grad</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p915419720217"><a name="p915419720217"></a><a name="p915419720217"></a>是</p>
</td>
</tr>
<tr id="row43858065310"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p257235454116"><a name="p257235454116"></a><a name="p257235454116"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1215477324"><a name="p1215477324"></a><a name="p1215477324"></a>torch.enable_grad</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1315415719214"><a name="p1315415719214"></a><a name="p1315415719214"></a>是</p>
</td>
</tr>
<tr id="row3997191714119"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18572125424115"><a name="p18572125424115"></a><a name="p18572125424115"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p121542071422"><a name="p121542071422"></a><a name="p121542071422"></a>torch.set_grad_enabled</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p515419720210"><a name="p515419720210"></a><a name="p515419720210"></a>是</p>
</td>
</tr>
<tr id="row168295201413"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4573115418419"><a name="p4573115418419"></a><a name="p4573115418419"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p161554713220"><a name="p161554713220"></a><a name="p161554713220"></a>torch.get_num_threads</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1415537625"><a name="p1415537625"></a><a name="p1415537625"></a>是</p>
</td>
</tr>
<tr id="row776650825"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18573175494114"><a name="p18573175494114"></a><a name="p18573175494114"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p81551071429"><a name="p81551071429"></a><a name="p81551071429"></a>torch.set_num_threads</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p131551571121"><a name="p131551571121"></a><a name="p131551571121"></a>是</p>
</td>
</tr>
<tr id="row155010451113"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18573155414110"><a name="p18573155414110"></a><a name="p18573155414110"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p41551671624"><a name="p41551671624"></a><a name="p41551671624"></a>torch.get_num_interop_threads</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p141552071210"><a name="p141552071210"></a><a name="p141552071210"></a>是</p>
</td>
</tr>
<tr id="row198851747015"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8573254184111"><a name="p8573254184111"></a><a name="p8573254184111"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p51551373215"><a name="p51551373215"></a><a name="p51551373215"></a>torch.set_num_interop_threads</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1215587720"><a name="p1215587720"></a><a name="p1215587720"></a>是</p>
</td>
</tr>
</tbody>
</table>

<h2 id="torch-Tensor">torch.Tensor</h2>

<a name="table1895120221777"></a>
<table><thead align="left"><tr id="row1595142219714"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p182711720423"><a name="p182711720423"></a><a name="p182711720423"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1995219221178"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p97113014438"><a name="p97113014438"></a><a name="p97113014438"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p75211139186"><a name="p75211139186"></a><a name="p75211139186"></a>torch.Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p195211395816"><a name="p195211395816"></a><a name="p195211395816"></a>是</p>
</td>
</tr>
<tr id="row199521822673"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1711800432"><a name="p1711800432"></a><a name="p1711800432"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14521183914819"><a name="p14521183914819"></a><a name="p14521183914819"></a>torch.Tensor.new_tensor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1652111394813"><a name="p1652111394813"></a><a name="p1652111394813"></a>是</p>
</td>
</tr>
<tr id="row17952152215715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14727004319"><a name="p14727004319"></a><a name="p14727004319"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18521939284"><a name="p18521939284"></a><a name="p18521939284"></a>torch.Tensor.new_full</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6521339384"><a name="p6521339384"></a><a name="p6521339384"></a>是</p>
</td>
</tr>
<tr id="row195216227713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p87220144320"><a name="p87220144320"></a><a name="p87220144320"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p552111398812"><a name="p552111398812"></a><a name="p552111398812"></a>torch.Tensor.new_empty</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p552120391185"><a name="p552120391185"></a><a name="p552120391185"></a>是</p>
</td>
</tr>
<tr id="row159525227716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p272708434"><a name="p272708434"></a><a name="p272708434"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19522183913811"><a name="p19522183913811"></a><a name="p19522183913811"></a>torch.Tensor.new_ones</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p11522939187"><a name="p11522939187"></a><a name="p11522939187"></a>是</p>
</td>
</tr>
<tr id="row795215221271"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p572100174319"><a name="p572100174319"></a><a name="p572100174319"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1952215391383"><a name="p1952215391383"></a><a name="p1952215391383"></a>torch.Tensor.new_zeros</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p152216393817"><a name="p152216393817"></a><a name="p152216393817"></a>是</p>
</td>
</tr>
<tr id="row695222220712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p207217016437"><a name="p207217016437"></a><a name="p207217016437"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p105222391984"><a name="p105222391984"></a><a name="p105222391984"></a>torch.Tensor.is_cuda</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18522739580"><a name="p18522739580"></a><a name="p18522739580"></a>是</p>
</td>
</tr>
<tr id="row1795282218716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8723017436"><a name="p8723017436"></a><a name="p8723017436"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p352233912811"><a name="p352233912811"></a><a name="p352233912811"></a>torch.Tensor.is_quantized</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85227391486"><a name="p85227391486"></a><a name="p85227391486"></a>是</p>
</td>
</tr>
<tr id="row1395217220719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18721304435"><a name="p18721304435"></a><a name="p18721304435"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p105228393819"><a name="p105228393819"></a><a name="p105228393819"></a>torch.Tensor.device</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p452243919811"><a name="p452243919811"></a><a name="p452243919811"></a>是</p>
</td>
</tr>
<tr id="row159521322578"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10721803436"><a name="p10721803436"></a><a name="p10721803436"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p20522173920813"><a name="p20522173920813"></a><a name="p20522173920813"></a>torch.Tensor.ndim</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p652213391583"><a name="p652213391583"></a><a name="p652213391583"></a>是</p>
</td>
</tr>
<tr id="row2095218223710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1172190154317"><a name="p1172190154317"></a><a name="p1172190154317"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p152210391580"><a name="p152210391580"></a><a name="p152210391580"></a>torch.Tensor.T</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1052213920817"><a name="p1052213920817"></a><a name="p1052213920817"></a>是</p>
</td>
</tr>
<tr id="row595219228718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p07212024319"><a name="p07212024319"></a><a name="p07212024319"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p12522113917819"><a name="p12522113917819"></a><a name="p12522113917819"></a>torch.Tensor.abs</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p55221139181"><a name="p55221139181"></a><a name="p55221139181"></a>是</p>
</td>
</tr>
<tr id="row169521022471"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p472160104314"><a name="p472160104314"></a><a name="p472160104314"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11522539688"><a name="p11522539688"></a><a name="p11522539688"></a>torch.Tensor.abs_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85223398819"><a name="p85223398819"></a><a name="p85223398819"></a>是</p>
</td>
</tr>
<tr id="row1795214221772"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4735044317"><a name="p4735044317"></a><a name="p4735044317"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14522239780"><a name="p14522239780"></a><a name="p14522239780"></a>torch.Tensor.acos</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18522839981"><a name="p18522839981"></a><a name="p18522839981"></a>是</p>
</td>
</tr>
<tr id="row1195313225711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5739013431"><a name="p5739013431"></a><a name="p5739013431"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p25220398814"><a name="p25220398814"></a><a name="p25220398814"></a>torch.Tensor.acos_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8522183911815"><a name="p8522183911815"></a><a name="p8522183911815"></a>是</p>
</td>
</tr>
<tr id="row1953922272"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p137318074312"><a name="p137318074312"></a><a name="p137318074312"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1152216391080"><a name="p1152216391080"></a><a name="p1152216391080"></a>torch.Tensor.add</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p17522439286"><a name="p17522439286"></a><a name="p17522439286"></a>是</p>
</td>
</tr>
<tr id="row159531822075"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10739018436"><a name="p10739018436"></a><a name="p10739018436"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1252213911814"><a name="p1252213911814"></a><a name="p1252213911814"></a>torch.Tensor.add_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p352273913815"><a name="p352273913815"></a><a name="p352273913815"></a>是</p>
</td>
</tr>
<tr id="row18953192213717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p573405432"><a name="p573405432"></a><a name="p573405432"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p652393920813"><a name="p652393920813"></a><a name="p652393920813"></a>torch.Tensor.addbmm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2523103915811"><a name="p2523103915811"></a><a name="p2523103915811"></a>是</p>
</td>
</tr>
<tr id="row12953222577"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p137310124311"><a name="p137310124311"></a><a name="p137310124311"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14523113918810"><a name="p14523113918810"></a><a name="p14523113918810"></a>torch.Tensor.addbmm_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p25239391684"><a name="p25239391684"></a><a name="p25239391684"></a>是</p>
</td>
</tr>
<tr id="row8953202211717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2735017434"><a name="p2735017434"></a><a name="p2735017434"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6523143920817"><a name="p6523143920817"></a><a name="p6523143920817"></a>torch.Tensor.addcdiv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10523163916815"><a name="p10523163916815"></a><a name="p10523163916815"></a>是</p>
</td>
</tr>
<tr id="row1895319228716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p873200433"><a name="p873200433"></a><a name="p873200433"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p452333913819"><a name="p452333913819"></a><a name="p452333913819"></a>torch.Tensor.addcdiv_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1152393918813"><a name="p1152393918813"></a><a name="p1152393918813"></a>是</p>
</td>
</tr>
<tr id="row7953132213717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6733020437"><a name="p6733020437"></a><a name="p6733020437"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16523239785"><a name="p16523239785"></a><a name="p16523239785"></a>torch.Tensor.addcmul</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p4523203919810"><a name="p4523203919810"></a><a name="p4523203919810"></a>是</p>
</td>
</tr>
<tr id="row139538224711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p197312034311"><a name="p197312034311"></a><a name="p197312034311"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p85239393819"><a name="p85239393819"></a><a name="p85239393819"></a>torch.Tensor.addcmul_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p165231398810"><a name="p165231398810"></a><a name="p165231398810"></a>是</p>
</td>
</tr>
<tr id="row1495382217720"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1773140104313"><a name="p1773140104313"></a><a name="p1773140104313"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p4523639487"><a name="p4523639487"></a><a name="p4523639487"></a>torch.Tensor.addmm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1252373916813"><a name="p1252373916813"></a><a name="p1252373916813"></a>是</p>
</td>
</tr>
<tr id="row1995318221379"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1747014317"><a name="p1747014317"></a><a name="p1747014317"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p175231339986"><a name="p175231339986"></a><a name="p175231339986"></a>torch.Tensor.addmm_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p35233391188"><a name="p35233391188"></a><a name="p35233391188"></a>是</p>
</td>
</tr>
<tr id="row49531322571"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p174150174311"><a name="p174150174311"></a><a name="p174150174311"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p25231839684"><a name="p25231839684"></a><a name="p25231839684"></a>torch.Tensor.addmv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p155232391982"><a name="p155232391982"></a><a name="p155232391982"></a>是</p>
</td>
</tr>
<tr id="row19953722671"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19741708430"><a name="p19741708430"></a><a name="p19741708430"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1152310399814"><a name="p1152310399814"></a><a name="p1152310399814"></a>torch.Tensor.addmv_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18523163918819"><a name="p18523163918819"></a><a name="p18523163918819"></a>是</p>
</td>
</tr>
<tr id="row195319221571"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20746094310"><a name="p20746094310"></a><a name="p20746094310"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1352317398813"><a name="p1352317398813"></a><a name="p1352317398813"></a>torch.Tensor.addr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1052313914813"><a name="p1052313914813"></a><a name="p1052313914813"></a>是</p>
</td>
</tr>
<tr id="row179531622472"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16746084313"><a name="p16746084313"></a><a name="p16746084313"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1952317391682"><a name="p1952317391682"></a><a name="p1952317391682"></a>torch.Tensor.addr_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p75231839384"><a name="p75231839384"></a><a name="p75231839384"></a>是</p>
</td>
</tr>
<tr id="row159531122179"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17419014432"><a name="p17419014432"></a><a name="p17419014432"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1252313917816"><a name="p1252313917816"></a><a name="p1252313917816"></a>torch.Tensor.allclose</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p752416391887"><a name="p752416391887"></a><a name="p752416391887"></a>是</p>
</td>
</tr>
<tr id="row1195411220715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17470194316"><a name="p17470194316"></a><a name="p17470194316"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10524123912810"><a name="p10524123912810"></a><a name="p10524123912810"></a>torch.Tensor.angle</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1152473916811"><a name="p1152473916811"></a><a name="p1152473916811"></a>否</p>
</td>
</tr>
<tr id="row39541227713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5742006436"><a name="p5742006436"></a><a name="p5742006436"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p0524439887"><a name="p0524439887"></a><a name="p0524439887"></a>torch.Tensor.apply_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1552417391780"><a name="p1552417391780"></a><a name="p1552417391780"></a>否</p>
</td>
</tr>
<tr id="row1895452219717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p674008431"><a name="p674008431"></a><a name="p674008431"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10524113919818"><a name="p10524113919818"></a><a name="p10524113919818"></a>torch.Tensor.argmax</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p65249391810"><a name="p65249391810"></a><a name="p65249391810"></a>是</p>
</td>
</tr>
<tr id="row109542220714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p107412018437"><a name="p107412018437"></a><a name="p107412018437"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p452413913811"><a name="p452413913811"></a><a name="p452413913811"></a>torch.Tensor.argmin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13524133916811"><a name="p13524133916811"></a><a name="p13524133916811"></a>是</p>
</td>
</tr>
<tr id="row159541622578"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p274120104311"><a name="p274120104311"></a><a name="p274120104311"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p155248398811"><a name="p155248398811"></a><a name="p155248398811"></a>torch.Tensor.argsort</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p185241539082"><a name="p185241539082"></a><a name="p185241539082"></a>是</p>
</td>
</tr>
<tr id="row15954522373"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p97519014313"><a name="p97519014313"></a><a name="p97519014313"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p0524183920812"><a name="p0524183920812"></a><a name="p0524183920812"></a>torch.Tensor.asin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1752453911819"><a name="p1752453911819"></a><a name="p1752453911819"></a>是</p>
</td>
</tr>
<tr id="row179544221179"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p775607430"><a name="p775607430"></a><a name="p775607430"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p135247394814"><a name="p135247394814"></a><a name="p135247394814"></a>torch.Tensor.asin_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2524939488"><a name="p2524939488"></a><a name="p2524939488"></a>是</p>
</td>
</tr>
<tr id="row19954622773"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p127510194311"><a name="p127510194311"></a><a name="p127510194311"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p252419392813"><a name="p252419392813"></a><a name="p252419392813"></a>torch.Tensor.as_strided</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p052419391818"><a name="p052419391818"></a><a name="p052419391818"></a>是</p>
</td>
</tr>
<tr id="row19542221775"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p187560134316"><a name="p187560134316"></a><a name="p187560134316"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p155241239684"><a name="p155241239684"></a><a name="p155241239684"></a>torch.Tensor.atan</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p152413395818"><a name="p152413395818"></a><a name="p152413395818"></a>是</p>
</td>
</tr>
<tr id="row179540221471"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12757016437"><a name="p12757016437"></a><a name="p12757016437"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p252414392816"><a name="p252414392816"></a><a name="p252414392816"></a>torch.Tensor.atan2</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1852411391589"><a name="p1852411391589"></a><a name="p1852411391589"></a>是</p>
</td>
</tr>
<tr id="row159544228715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1757064317"><a name="p1757064317"></a><a name="p1757064317"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1152410396813"><a name="p1152410396813"></a><a name="p1152410396813"></a>torch.Tensor.atan2_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10524639186"><a name="p10524639186"></a><a name="p10524639186"></a>是</p>
</td>
</tr>
<tr id="row39548222717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1475140134312"><a name="p1475140134312"></a><a name="p1475140134312"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p145240391481"><a name="p145240391481"></a><a name="p145240391481"></a>torch.Tensor.atan_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85241339381"><a name="p85241339381"></a><a name="p85241339381"></a>是</p>
</td>
</tr>
<tr id="row1995418225714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p0759094310"><a name="p0759094310"></a><a name="p0759094310"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10525143918816"><a name="p10525143918816"></a><a name="p10525143918816"></a>torch.Tensor.baddbmm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p145251939682"><a name="p145251939682"></a><a name="p145251939682"></a>是</p>
</td>
</tr>
<tr id="row1295418221272"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p77513054314"><a name="p77513054314"></a><a name="p77513054314"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1652516391812"><a name="p1652516391812"></a><a name="p1652516391812"></a>torch.Tensor.baddbmm_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2525163919819"><a name="p2525163919819"></a><a name="p2525163919819"></a>是</p>
</td>
</tr>
<tr id="row109545225714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1375110104317"><a name="p1375110104317"></a><a name="p1375110104317"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16525039683"><a name="p16525039683"></a><a name="p16525039683"></a>torch.Tensor.bernoulli</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115257392082"><a name="p115257392082"></a><a name="p115257392082"></a>是</p>
</td>
</tr>
<tr id="row395416224716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p775120104317"><a name="p775120104317"></a><a name="p775120104317"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11525103917819"><a name="p11525103917819"></a><a name="p11525103917819"></a>torch.Tensor.bernoulli_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10513194594716"><a name="p10513194594716"></a><a name="p10513194594716"></a>是</p>
</td>
</tr>
<tr id="row295519221279"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20751054315"><a name="p20751054315"></a><a name="p20751054315"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9525739885"><a name="p9525739885"></a><a name="p9525739885"></a>torch.Tensor.bfloat16</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1752517391086"><a name="p1752517391086"></a><a name="p1752517391086"></a>否</p>
</td>
</tr>
<tr id="row1295516221479"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p197516010433"><a name="p197516010433"></a><a name="p197516010433"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p352516391881"><a name="p352516391881"></a><a name="p352516391881"></a>torch.Tensor.bincount</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1752512391381"><a name="p1752512391381"></a><a name="p1752512391381"></a>否</p>
</td>
</tr>
<tr id="row4955132218718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10756044314"><a name="p10756044314"></a><a name="p10756044314"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1452510399815"><a name="p1452510399815"></a><a name="p1452510399815"></a>torch.Tensor.bitwise_not</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7525739881"><a name="p7525739881"></a><a name="p7525739881"></a>是</p>
</td>
</tr>
<tr id="row1795513221072"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17751104430"><a name="p17751104430"></a><a name="p17751104430"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p195251739682"><a name="p195251739682"></a><a name="p195251739682"></a>torch.Tensor.bitwise_not_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1652510396813"><a name="p1652510396813"></a><a name="p1652510396813"></a>是</p>
</td>
</tr>
<tr id="row095518221678"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p37600194311"><a name="p37600194311"></a><a name="p37600194311"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1352511391181"><a name="p1352511391181"></a><a name="p1352511391181"></a>torch.Tensor.bitwise_and</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1652512390812"><a name="p1652512390812"></a><a name="p1652512390812"></a>是</p>
</td>
</tr>
<tr id="row1295518221875"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p376180184311"><a name="p376180184311"></a><a name="p376180184311"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1552583910814"><a name="p1552583910814"></a><a name="p1552583910814"></a>torch.Tensor.bitwise_and_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p175258398820"><a name="p175258398820"></a><a name="p175258398820"></a>是</p>
</td>
</tr>
<tr id="row695517221718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p37616044315"><a name="p37616044315"></a><a name="p37616044315"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p352593919818"><a name="p352593919818"></a><a name="p352593919818"></a>torch.Tensor.bitwise_or</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19525193918813"><a name="p19525193918813"></a><a name="p19525193918813"></a>是</p>
</td>
</tr>
<tr id="row12955102216719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19768074317"><a name="p19768074317"></a><a name="p19768074317"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1152511391186"><a name="p1152511391186"></a><a name="p1152511391186"></a>torch.Tensor.bitwise_or_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p75251139784"><a name="p75251139784"></a><a name="p75251139784"></a>是</p>
</td>
</tr>
<tr id="row10955162217714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p137614064320"><a name="p137614064320"></a><a name="p137614064320"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p185257396814"><a name="p185257396814"></a><a name="p185257396814"></a>torch.Tensor.bitwise_xor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p145257393816"><a name="p145257393816"></a><a name="p145257393816"></a>是</p>
</td>
</tr>
<tr id="row39551522778"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19766013434"><a name="p19766013434"></a><a name="p19766013434"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1552613391689"><a name="p1552613391689"></a><a name="p1552613391689"></a>torch.Tensor.bitwise_xor_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6526183914813"><a name="p6526183914813"></a><a name="p6526183914813"></a>是</p>
</td>
</tr>
<tr id="row1895513221720"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p147660164310"><a name="p147660164310"></a><a name="p147660164310"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p115261839688"><a name="p115261839688"></a><a name="p115261839688"></a>torch.Tensor.bmm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14526153910812"><a name="p14526153910812"></a><a name="p14526153910812"></a>是</p>
</td>
</tr>
<tr id="row595502218718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8760064320"><a name="p8760064320"></a><a name="p8760064320"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p452610391981"><a name="p452610391981"></a><a name="p452610391981"></a>torch.Tensor.bool</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p352615391488"><a name="p352615391488"></a><a name="p352615391488"></a>是</p>
</td>
</tr>
<tr id="row129554221076"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18762094317"><a name="p18762094317"></a><a name="p18762094317"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1952610391781"><a name="p1952610391781"></a><a name="p1952610391781"></a>torch.Tensor.byte</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115264396810"><a name="p115264396810"></a><a name="p115264396810"></a>是</p>
</td>
</tr>
<tr id="row89556228711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p27620104312"><a name="p27620104312"></a><a name="p27620104312"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1752663911819"><a name="p1752663911819"></a><a name="p1752663911819"></a>torch.Tensor.cauchy_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p75265391817"><a name="p75265391817"></a><a name="p75265391817"></a>否</p>
</td>
</tr>
<tr id="row8955202216715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p576150174310"><a name="p576150174310"></a><a name="p576150174310"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p155269391481"><a name="p155269391481"></a><a name="p155269391481"></a>torch.Tensor.ceil</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1852616392812"><a name="p1852616392812"></a><a name="p1852616392812"></a>是</p>
</td>
</tr>
<tr id="row695519223712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p67640174317"><a name="p67640174317"></a><a name="p67640174317"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p115268391483"><a name="p115268391483"></a><a name="p115268391483"></a>torch.Tensor.ceil_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1052616391484"><a name="p1052616391484"></a><a name="p1052616391484"></a>是</p>
</td>
</tr>
<tr id="row595611221714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p0762015435"><a name="p0762015435"></a><a name="p0762015435"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p55261399810"><a name="p55261399810"></a><a name="p55261399810"></a>torch.Tensor.char</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1652619391883"><a name="p1652619391883"></a><a name="p1652619391883"></a>是</p>
</td>
</tr>
<tr id="row179561522674"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1376110184310"><a name="p1376110184310"></a><a name="p1376110184310"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14526173916817"><a name="p14526173916817"></a><a name="p14526173916817"></a>torch.Tensor.cholesky</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13526193912812"><a name="p13526193912812"></a><a name="p13526193912812"></a>否</p>
</td>
</tr>
<tr id="row99561922874"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p87611015437"><a name="p87611015437"></a><a name="p87611015437"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p252643914818"><a name="p252643914818"></a><a name="p252643914818"></a>torch.Tensor.cholesky_inverse</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p452615399820"><a name="p452615399820"></a><a name="p452615399820"></a>否</p>
</td>
</tr>
<tr id="row18956102216715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3761201437"><a name="p3761201437"></a><a name="p3761201437"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8526439288"><a name="p8526439288"></a><a name="p8526439288"></a>torch.Tensor.cholesky_solve</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p165261939886"><a name="p165261939886"></a><a name="p165261939886"></a>否</p>
</td>
</tr>
<tr id="row4956722677"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p67610016437"><a name="p67610016437"></a><a name="p67610016437"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1652619390818"><a name="p1652619390818"></a><a name="p1652619390818"></a>torch.Tensor.chunk</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1352616399818"><a name="p1352616399818"></a><a name="p1352616399818"></a>是</p>
</td>
</tr>
<tr id="row1895613229710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p167615011436"><a name="p167615011436"></a><a name="p167615011436"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p952611399811"><a name="p952611399811"></a><a name="p952611399811"></a>torch.Tensor.clamp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p155271396812"><a name="p155271396812"></a><a name="p155271396812"></a>是</p>
</td>
</tr>
<tr id="row1895616221873"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5768014318"><a name="p5768014318"></a><a name="p5768014318"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1252710391281"><a name="p1252710391281"></a><a name="p1252710391281"></a>torch.Tensor.clamp_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1852711391085"><a name="p1852711391085"></a><a name="p1852711391085"></a>是</p>
</td>
</tr>
<tr id="row119562229719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p57613024311"><a name="p57613024311"></a><a name="p57613024311"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15279399815"><a name="p15279399815"></a><a name="p15279399815"></a>torch.Tensor.clone</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1527103916820"><a name="p1527103916820"></a><a name="p1527103916820"></a>是</p>
</td>
</tr>
<tr id="row1295672219711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5771307437"><a name="p5771307437"></a><a name="p5771307437"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16527939481"><a name="p16527939481"></a><a name="p16527939481"></a>torch.Tensor.contiguous</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85271539482"><a name="p85271539482"></a><a name="p85271539482"></a>是</p>
</td>
</tr>
<tr id="row29561322074"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p57716034319"><a name="p57716034319"></a><a name="p57716034319"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p55270396816"><a name="p55270396816"></a><a name="p55270396816"></a>torch.Tensor.copy_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p252711392087"><a name="p252711392087"></a><a name="p252711392087"></a>是</p>
</td>
</tr>
<tr id="row7956142216710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8774084319"><a name="p8774084319"></a><a name="p8774084319"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2527039386"><a name="p2527039386"></a><a name="p2527039386"></a>torch.Tensor.conj</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1852715391683"><a name="p1852715391683"></a><a name="p1852715391683"></a>否</p>
</td>
</tr>
<tr id="row18956162212712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17775013435"><a name="p17775013435"></a><a name="p17775013435"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p752715397816"><a name="p752715397816"></a><a name="p752715397816"></a>torch.Tensor.cos</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p35271839482"><a name="p35271839482"></a><a name="p35271839482"></a>是</p>
</td>
</tr>
<tr id="row595662210716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4771701431"><a name="p4771701431"></a><a name="p4771701431"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1152715391486"><a name="p1152715391486"></a><a name="p1152715391486"></a>torch.Tensor.cos_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p65277399815"><a name="p65277399815"></a><a name="p65277399815"></a>是</p>
</td>
</tr>
<tr id="row179561221673"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14771704433"><a name="p14771704433"></a><a name="p14771704433"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1552712391684"><a name="p1552712391684"></a><a name="p1552712391684"></a>torch.Tensor.cosh</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p75271839283"><a name="p75271839283"></a><a name="p75271839283"></a>是</p>
</td>
</tr>
<tr id="row2956322671"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p187770134316"><a name="p187770134316"></a><a name="p187770134316"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p115271439585"><a name="p115271439585"></a><a name="p115271439585"></a>torch.Tensor.cosh_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15276391089"><a name="p15276391089"></a><a name="p15276391089"></a>是</p>
</td>
</tr>
<tr id="row59571222979"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p177713010434"><a name="p177713010434"></a><a name="p177713010434"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p185271239981"><a name="p185271239981"></a><a name="p185271239981"></a>torch.Tensor.cpu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p20527113918820"><a name="p20527113918820"></a><a name="p20527113918820"></a>是</p>
</td>
</tr>
<tr id="row129575227715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11777020432"><a name="p11777020432"></a><a name="p11777020432"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p25271539781"><a name="p25271539781"></a><a name="p25271539781"></a>torch.Tensor.cross</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85272391881"><a name="p85272391881"></a><a name="p85272391881"></a>是</p>
</td>
</tr>
<tr id="row3957132217714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p177190184318"><a name="p177190184318"></a><a name="p177190184318"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p652712399818"><a name="p652712399818"></a><a name="p652712399818"></a>torch.Tensor.cuda</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p175271639083"><a name="p175271639083"></a><a name="p175271639083"></a>否</p>
</td>
</tr>
<tr id="row1395732218716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2077190104310"><a name="p2077190104310"></a><a name="p2077190104310"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p7527139782"><a name="p7527139782"></a><a name="p7527139782"></a>torch.Tensor.cummax</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85281339382"><a name="p85281339382"></a><a name="p85281339382"></a>否</p>
</td>
</tr>
<tr id="row159571822579"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p67715019438"><a name="p67715019438"></a><a name="p67715019438"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p65281639289"><a name="p65281639289"></a><a name="p65281639289"></a>torch.Tensor.cummin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12528139088"><a name="p12528139088"></a><a name="p12528139088"></a>是</p>
</td>
</tr>
<tr id="row16957522876"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14770013438"><a name="p14770013438"></a><a name="p14770013438"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1352815391184"><a name="p1352815391184"></a><a name="p1352815391184"></a>torch.Tensor.cumprod</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19528183910816"><a name="p19528183910816"></a><a name="p19528183910816"></a>是</p>
</td>
</tr>
<tr id="row189571822378"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1377110104312"><a name="p1377110104312"></a><a name="p1377110104312"></a>84</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1552811398810"><a name="p1552811398810"></a><a name="p1552811398810"></a>torch.Tensor.cumsum</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p17528133918810"><a name="p17528133918810"></a><a name="p17528133918810"></a>是</p>
</td>
</tr>
<tr id="row119577224716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p177714018434"><a name="p177714018434"></a><a name="p177714018434"></a>85</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6528183918810"><a name="p6528183918810"></a><a name="p6528183918810"></a>torch.Tensor.data_ptr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p155281391813"><a name="p155281391813"></a><a name="p155281391813"></a>是</p>
</td>
</tr>
<tr id="row2095742213717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p77711004318"><a name="p77711004318"></a><a name="p77711004318"></a>86</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p175287397811"><a name="p175287397811"></a><a name="p175287397811"></a>torch.Tensor.dequantize</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15281539382"><a name="p15281539382"></a><a name="p15281539382"></a>否</p>
</td>
</tr>
<tr id="row129571322279"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p977150144319"><a name="p977150144319"></a><a name="p977150144319"></a>87</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16528739482"><a name="p16528739482"></a><a name="p16528739482"></a>torch.Tensor.det</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p552819391816"><a name="p552819391816"></a><a name="p552819391816"></a>否</p>
</td>
</tr>
<tr id="row1795717221876"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17771901434"><a name="p17771901434"></a><a name="p17771901434"></a>88</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1652873912810"><a name="p1652873912810"></a><a name="p1652873912810"></a>torch.Tensor.dense_dim</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p752873913817"><a name="p752873913817"></a><a name="p752873913817"></a>否</p>
</td>
</tr>
<tr id="row1295710223719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p117719011435"><a name="p117719011435"></a><a name="p117719011435"></a>89</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p852823913810"><a name="p852823913810"></a><a name="p852823913810"></a>torch.Tensor.diag</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1152819391483"><a name="p1152819391483"></a><a name="p1152819391483"></a>是</p>
</td>
</tr>
<tr id="row9957122219719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1678150104317"><a name="p1678150104317"></a><a name="p1678150104317"></a>90</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p125287391780"><a name="p125287391780"></a><a name="p125287391780"></a>torch.Tensor.diag_embed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1052815392089"><a name="p1052815392089"></a><a name="p1052815392089"></a>是</p>
</td>
</tr>
<tr id="row3957622973"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1781808437"><a name="p1781808437"></a><a name="p1781808437"></a>91</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p952811392813"><a name="p952811392813"></a><a name="p952811392813"></a>torch.Tensor.diagflat</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p05281639389"><a name="p05281639389"></a><a name="p05281639389"></a>是</p>
</td>
</tr>
<tr id="row095772218711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p87817014316"><a name="p87817014316"></a><a name="p87817014316"></a>92</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p352813914812"><a name="p352813914812"></a><a name="p352813914812"></a>torch.Tensor.diagonal</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p55281939683"><a name="p55281939683"></a><a name="p55281939683"></a>是</p>
</td>
</tr>
<tr id="row1095722220711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p678180174310"><a name="p678180174310"></a><a name="p678180174310"></a>93</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14528139481"><a name="p14528139481"></a><a name="p14528139481"></a>torch.Tensor.fill_diagonal_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9528439983"><a name="p9528439983"></a><a name="p9528439983"></a>是</p>
</td>
</tr>
<tr id="row795810227710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14781606436"><a name="p14781606436"></a><a name="p14781606436"></a>94</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1452918396815"><a name="p1452918396815"></a><a name="p1452918396815"></a>torch.Tensor.digamma</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14529123910816"><a name="p14529123910816"></a><a name="p14529123910816"></a>否</p>
</td>
</tr>
<tr id="row209582228716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p678170164313"><a name="p678170164313"></a><a name="p678170164313"></a>95</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p135295391485"><a name="p135295391485"></a><a name="p135295391485"></a>torch.Tensor.digamma_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15529143912811"><a name="p15529143912811"></a><a name="p15529143912811"></a>否</p>
</td>
</tr>
<tr id="row17958162212714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15781107436"><a name="p15781107436"></a><a name="p15781107436"></a>96</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1052916393820"><a name="p1052916393820"></a><a name="p1052916393820"></a>torch.Tensor.dim</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p35299392815"><a name="p35299392815"></a><a name="p35299392815"></a>是</p>
</td>
</tr>
<tr id="row1295811225719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p197860174319"><a name="p197860174319"></a><a name="p197860174319"></a>97</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p105291339786"><a name="p105291339786"></a><a name="p105291339786"></a>torch.Tensor.dist</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p125298392088"><a name="p125298392088"></a><a name="p125298392088"></a>是</p>
</td>
</tr>
<tr id="row595812221271"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p47880134310"><a name="p47880134310"></a><a name="p47880134310"></a>98</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p55291939584"><a name="p55291939584"></a><a name="p55291939584"></a>torch.Tensor.div</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10529163914816"><a name="p10529163914816"></a><a name="p10529163914816"></a>是</p>
</td>
</tr>
<tr id="row159584226718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20781409438"><a name="p20781409438"></a><a name="p20781409438"></a>99</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5529183918810"><a name="p5529183918810"></a><a name="p5529183918810"></a>torch.Tensor.div_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9529193914814"><a name="p9529193914814"></a><a name="p9529193914814"></a>是</p>
</td>
</tr>
<tr id="row795817221072"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1378160164320"><a name="p1378160164320"></a><a name="p1378160164320"></a>100</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p752983919818"><a name="p752983919818"></a><a name="p752983919818"></a>torch.Tensor.dot</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115291939589"><a name="p115291939589"></a><a name="p115291939589"></a>否</p>
</td>
</tr>
<tr id="row199583220716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p878009436"><a name="p878009436"></a><a name="p878009436"></a>101</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1152920392810"><a name="p1152920392810"></a><a name="p1152920392810"></a>torch.Tensor.double</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p165291339681"><a name="p165291339681"></a><a name="p165291339681"></a>否</p>
</td>
</tr>
<tr id="row895812216718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7799024318"><a name="p7799024318"></a><a name="p7799024318"></a>102</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16529439383"><a name="p16529439383"></a><a name="p16529439383"></a>torch.Tensor.eig</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1252923914813"><a name="p1252923914813"></a><a name="p1252923914813"></a>否</p>
</td>
</tr>
<tr id="row1795818220713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p177990164311"><a name="p177990164311"></a><a name="p177990164311"></a>103</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p125294391985"><a name="p125294391985"></a><a name="p125294391985"></a>torch.Tensor.element_size</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115298392811"><a name="p115298392811"></a><a name="p115298392811"></a>是</p>
</td>
</tr>
<tr id="row395814221774"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p157919014436"><a name="p157919014436"></a><a name="p157919014436"></a>104</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p652916390815"><a name="p652916390815"></a><a name="p652916390815"></a>torch.Tensor.eq</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1152918394814"><a name="p1152918394814"></a><a name="p1152918394814"></a>是</p>
</td>
</tr>
<tr id="row395812226717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1479110194315"><a name="p1479110194315"></a><a name="p1479110194315"></a>105</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p452920392820"><a name="p452920392820"></a><a name="p452920392820"></a>torch.Tensor.eq_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p252914391982"><a name="p252914391982"></a><a name="p252914391982"></a>是</p>
</td>
</tr>
<tr id="row1595811227720"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p379140194313"><a name="p379140194313"></a><a name="p379140194313"></a>106</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p152915393814"><a name="p152915393814"></a><a name="p152915393814"></a>torch.Tensor.equal</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p195301339085"><a name="p195301339085"></a><a name="p195301339085"></a>是</p>
</td>
</tr>
<tr id="row1295811228713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p57915018436"><a name="p57915018436"></a><a name="p57915018436"></a>107</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11530239884"><a name="p11530239884"></a><a name="p11530239884"></a>torch.Tensor.erf</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10530113918819"><a name="p10530113918819"></a><a name="p10530113918819"></a>是</p>
</td>
</tr>
<tr id="row16958822975"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9799014313"><a name="p9799014313"></a><a name="p9799014313"></a>108</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p25302392088"><a name="p25302392088"></a><a name="p25302392088"></a>torch.Tensor.erf_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p35304393813"><a name="p35304393813"></a><a name="p35304393813"></a>是</p>
</td>
</tr>
<tr id="row1595911222077"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16795034316"><a name="p16795034316"></a><a name="p16795034316"></a>109</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p053011391483"><a name="p053011391483"></a><a name="p053011391483"></a>torch.Tensor.erfc</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p45301639887"><a name="p45301639887"></a><a name="p45301639887"></a>是</p>
</td>
</tr>
<tr id="row16959132214716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p187918094316"><a name="p187918094316"></a><a name="p187918094316"></a>110</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1553013910815"><a name="p1553013910815"></a><a name="p1553013910815"></a>torch.Tensor.erfc_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p853033913811"><a name="p853033913811"></a><a name="p853033913811"></a>是</p>
</td>
</tr>
<tr id="row5959922375"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1679403434"><a name="p1679403434"></a><a name="p1679403434"></a>111</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13530123911817"><a name="p13530123911817"></a><a name="p13530123911817"></a>torch.Tensor.erfinv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1453013915817"><a name="p1453013915817"></a><a name="p1453013915817"></a>是</p>
</td>
</tr>
<tr id="row69591222672"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p479008439"><a name="p479008439"></a><a name="p479008439"></a>112</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8530339484"><a name="p8530339484"></a><a name="p8530339484"></a>torch.Tensor.erfinv_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1553019393818"><a name="p1553019393818"></a><a name="p1553019393818"></a>是</p>
</td>
</tr>
<tr id="row1895915221278"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18090134317"><a name="p18090134317"></a><a name="p18090134317"></a>113</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p65301439889"><a name="p65301439889"></a><a name="p65301439889"></a>torch.Tensor.exp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p853020397810"><a name="p853020397810"></a><a name="p853020397810"></a>是</p>
</td>
</tr>
<tr id="row1395942213712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8801603433"><a name="p8801603433"></a><a name="p8801603433"></a>114</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18530173912813"><a name="p18530173912813"></a><a name="p18530173912813"></a>torch.Tensor.exp_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p17530639785"><a name="p17530639785"></a><a name="p17530639785"></a>是</p>
</td>
</tr>
<tr id="row395910221872"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p168000124313"><a name="p168000124313"></a><a name="p168000124313"></a>115</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p95301739788"><a name="p95301739788"></a><a name="p95301739788"></a>torch.Tensor.expm1</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1653016391183"><a name="p1653016391183"></a><a name="p1653016391183"></a>是</p>
</td>
</tr>
<tr id="row395911229712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1280306432"><a name="p1280306432"></a><a name="p1280306432"></a>116</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p4530139080"><a name="p4530139080"></a><a name="p4530139080"></a>torch.Tensor.expm1_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1353017397819"><a name="p1353017397819"></a><a name="p1353017397819"></a>是</p>
</td>
</tr>
<tr id="row1495911221871"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p180170194313"><a name="p180170194313"></a><a name="p180170194313"></a>117</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1053013391286"><a name="p1053013391286"></a><a name="p1053013391286"></a>torch.Tensor.expand</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1653012392812"><a name="p1653012392812"></a><a name="p1653012392812"></a>是</p>
</td>
</tr>
<tr id="row095912221073"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12807054318"><a name="p12807054318"></a><a name="p12807054318"></a>118</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p205303391386"><a name="p205303391386"></a><a name="p205303391386"></a>torch.Tensor.expand_as</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p25311139489"><a name="p25311139489"></a><a name="p25311139489"></a>是</p>
</td>
</tr>
<tr id="row179591322774"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p108015012437"><a name="p108015012437"></a><a name="p108015012437"></a>119</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9531939985"><a name="p9531939985"></a><a name="p9531939985"></a>torch.Tensor.exponential_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p653119399812"><a name="p653119399812"></a><a name="p653119399812"></a>否</p>
</td>
</tr>
<tr id="row2095982210718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2080130134320"><a name="p2080130134320"></a><a name="p2080130134320"></a>120</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p753163912815"><a name="p753163912815"></a><a name="p753163912815"></a>torch.Tensor.fft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p153114391480"><a name="p153114391480"></a><a name="p153114391480"></a>否</p>
</td>
</tr>
<tr id="row12959322577"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14801508436"><a name="p14801508436"></a><a name="p14801508436"></a>121</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17531339386"><a name="p17531339386"></a><a name="p17531339386"></a>torch.Tensor.fill_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p75311939086"><a name="p75311939086"></a><a name="p75311939086"></a>是</p>
</td>
</tr>
<tr id="row1995919221176"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2802018434"><a name="p2802018434"></a><a name="p2802018434"></a>122</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p853112396818"><a name="p853112396818"></a><a name="p853112396818"></a>torch.Tensor.flatten</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1531439287"><a name="p1531439287"></a><a name="p1531439287"></a>是</p>
</td>
</tr>
<tr id="row169591822874"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p38030104312"><a name="p38030104312"></a><a name="p38030104312"></a>123</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9531539887"><a name="p9531539887"></a><a name="p9531539887"></a>torch.Tensor.flip</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15311391384"><a name="p15311391384"></a><a name="p15311391384"></a>是</p>
</td>
</tr>
<tr id="row139595227710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p198112014316"><a name="p198112014316"></a><a name="p198112014316"></a>124</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1953113919812"><a name="p1953113919812"></a><a name="p1953113919812"></a>torch.Tensor.float</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p65317391087"><a name="p65317391087"></a><a name="p65317391087"></a>是</p>
</td>
</tr>
<tr id="row1696011221771"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p68118010435"><a name="p68118010435"></a><a name="p68118010435"></a>125</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8531113912810"><a name="p8531113912810"></a><a name="p8531113912810"></a>torch.Tensor.floor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1453119391789"><a name="p1453119391789"></a><a name="p1453119391789"></a>是</p>
</td>
</tr>
<tr id="row996016226716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12811902435"><a name="p12811902435"></a><a name="p12811902435"></a>126</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1853153917814"><a name="p1853153917814"></a><a name="p1853153917814"></a>torch.Tensor.floor_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p153116391880"><a name="p153116391880"></a><a name="p153116391880"></a>是</p>
</td>
</tr>
<tr id="row1096017221072"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p168116054312"><a name="p168116054312"></a><a name="p168116054312"></a>127</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p12531639282"><a name="p12531639282"></a><a name="p12531639282"></a>torch.Tensor.floor_divide</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p453112391786"><a name="p453112391786"></a><a name="p453112391786"></a>是</p>
</td>
</tr>
<tr id="row189601322779"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1281110204311"><a name="p1281110204311"></a><a name="p1281110204311"></a>128</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p75311039185"><a name="p75311039185"></a><a name="p75311039185"></a>torch.Tensor.floor_divide_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p95313393817"><a name="p95313393817"></a><a name="p95313393817"></a>是</p>
</td>
</tr>
<tr id="row1496042214711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8814016435"><a name="p8814016435"></a><a name="p8814016435"></a>129</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p165317391688"><a name="p165317391688"></a><a name="p165317391688"></a>torch.Tensor.fmod</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p053153910816"><a name="p053153910816"></a><a name="p053153910816"></a>是</p>
</td>
</tr>
<tr id="row199601722575"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p78111024314"><a name="p78111024314"></a><a name="p78111024314"></a>130</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1153112391287"><a name="p1153112391287"></a><a name="p1153112391287"></a>torch.Tensor.fmod_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p253117391186"><a name="p253117391186"></a><a name="p253117391186"></a>是</p>
</td>
</tr>
<tr id="row4960102215717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p128111014319"><a name="p128111014319"></a><a name="p128111014319"></a>131</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p253115395810"><a name="p253115395810"></a><a name="p253115395810"></a>torch.Tensor.frac</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6532113910819"><a name="p6532113910819"></a><a name="p6532113910819"></a>是</p>
</td>
</tr>
<tr id="row1596019222718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p178110114314"><a name="p178110114314"></a><a name="p178110114314"></a>132</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p185321839285"><a name="p185321839285"></a><a name="p185321839285"></a>torch.Tensor.frac_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p175321739880"><a name="p175321739880"></a><a name="p175321739880"></a>是</p>
</td>
</tr>
<tr id="row11960132212713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3813017436"><a name="p3813017436"></a><a name="p3813017436"></a>133</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p3532163914812"><a name="p3532163914812"></a><a name="p3532163914812"></a>torch.Tensor.gather</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1253233919813"><a name="p1253233919813"></a><a name="p1253233919813"></a>是</p>
</td>
</tr>
<tr id="row49601722072"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16811606439"><a name="p16811606439"></a><a name="p16811606439"></a>134</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19532183912813"><a name="p19532183912813"></a><a name="p19532183912813"></a>torch.Tensor.ge</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1953223917813"><a name="p1953223917813"></a><a name="p1953223917813"></a>是</p>
</td>
</tr>
<tr id="row39605221174"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16823020433"><a name="p16823020433"></a><a name="p16823020433"></a>135</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p12532339786"><a name="p12532339786"></a><a name="p12532339786"></a>torch.Tensor.ge_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10532153917818"><a name="p10532153917818"></a><a name="p10532153917818"></a>是</p>
</td>
</tr>
<tr id="row59601622872"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p48280164313"><a name="p48280164313"></a><a name="p48280164313"></a>136</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1553214396819"><a name="p1553214396819"></a><a name="p1553214396819"></a>torch.Tensor.geometric_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15532839883"><a name="p15532839883"></a><a name="p15532839883"></a>否</p>
</td>
</tr>
<tr id="row4960422872"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1382708436"><a name="p1382708436"></a><a name="p1382708436"></a>137</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p165321139586"><a name="p165321139586"></a><a name="p165321139586"></a>torch.Tensor.geqrf</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p125326393818"><a name="p125326393818"></a><a name="p125326393818"></a>否</p>
</td>
</tr>
<tr id="row69604222713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p88211016434"><a name="p88211016434"></a><a name="p88211016434"></a>138</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1053243917820"><a name="p1053243917820"></a><a name="p1053243917820"></a>torch.Tensor.ger</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1138152141213"><a name="p1138152141213"></a><a name="p1138152141213"></a>是</p>
</td>
</tr>
<tr id="row1496012221713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20824084320"><a name="p20824084320"></a><a name="p20824084320"></a>139</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p195326391815"><a name="p195326391815"></a><a name="p195326391815"></a>torch.Tensor.get_device</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p553212399818"><a name="p553212399818"></a><a name="p553212399818"></a>是</p>
</td>
</tr>
<tr id="row19611722075"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8822010432"><a name="p8822010432"></a><a name="p8822010432"></a>140</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p65321394812"><a name="p65321394812"></a><a name="p65321394812"></a>torch.Tensor.gt</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19532113911816"><a name="p19532113911816"></a><a name="p19532113911816"></a>是</p>
</td>
</tr>
<tr id="row2961822376"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p68280194313"><a name="p68280194313"></a><a name="p68280194313"></a>141</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p185329391282"><a name="p185329391282"></a><a name="p185329391282"></a>torch.Tensor.gt_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p55328391689"><a name="p55328391689"></a><a name="p55328391689"></a>是</p>
</td>
</tr>
<tr id="row89611221375"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17821094310"><a name="p17821094310"></a><a name="p17821094310"></a>142</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p85326399813"><a name="p85326399813"></a><a name="p85326399813"></a>torch.Tensor.half</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p253212392081"><a name="p253212392081"></a><a name="p253212392081"></a>是</p>
</td>
</tr>
<tr id="row1696115229710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12829012435"><a name="p12829012435"></a><a name="p12829012435"></a>143</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14532153912817"><a name="p14532153912817"></a><a name="p14532153912817"></a>torch.Tensor.hardshrink</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1853218391819"><a name="p1853218391819"></a><a name="p1853218391819"></a>是</p>
</td>
</tr>
<tr id="row1096182219715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p68216054320"><a name="p68216054320"></a><a name="p68216054320"></a>144</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8533839786"><a name="p8533839786"></a><a name="p8533839786"></a>torch.Tensor.histc</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p653319394816"><a name="p653319394816"></a><a name="p653319394816"></a>否</p>
</td>
</tr>
<tr id="row1196132211718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p183701431"><a name="p183701431"></a><a name="p183701431"></a>145</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p155336391982"><a name="p155336391982"></a><a name="p155336391982"></a>torch.Tensor.ifft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p205334391282"><a name="p205334391282"></a><a name="p205334391282"></a>否</p>
</td>
</tr>
<tr id="row159611722176"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1283805432"><a name="p1283805432"></a><a name="p1283805432"></a>146</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19533173913816"><a name="p19533173913816"></a><a name="p19533173913816"></a>torch.Tensor.index_add_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p195336391389"><a name="p195336391389"></a><a name="p195336391389"></a>是</p>
</td>
</tr>
<tr id="row29611222179"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p383150104312"><a name="p383150104312"></a><a name="p383150104312"></a>147</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p4533143916819"><a name="p4533143916819"></a><a name="p4533143916819"></a>torch.Tensor.index_add</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85337391189"><a name="p85337391189"></a><a name="p85337391189"></a>是</p>
</td>
</tr>
<tr id="row119610222711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1583308436"><a name="p1583308436"></a><a name="p1583308436"></a>148</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p75331639881"><a name="p75331639881"></a><a name="p75331639881"></a>torch.Tensor.index_copy_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p553383914815"><a name="p553383914815"></a><a name="p553383914815"></a>是</p>
</td>
</tr>
<tr id="row5961172211719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p128340114315"><a name="p128340114315"></a><a name="p128340114315"></a>149</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13533143911819"><a name="p13533143911819"></a><a name="p13533143911819"></a>torch.Tensor.index_copy</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p185334391187"><a name="p185334391187"></a><a name="p185334391187"></a>是</p>
</td>
</tr>
<tr id="row179613221674"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p383604430"><a name="p383604430"></a><a name="p383604430"></a>150</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p20533173911811"><a name="p20533173911811"></a><a name="p20533173911811"></a>torch.Tensor.index_fill_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2089795991218"><a name="p2089795991218"></a><a name="p2089795991218"></a>是</p>
</td>
</tr>
<tr id="row096115221574"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1483120114313"><a name="p1483120114313"></a><a name="p1483120114313"></a>151</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p135331939783"><a name="p135331939783"></a><a name="p135331939783"></a>torch.Tensor.index_fill</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p45331439786"><a name="p45331439786"></a><a name="p45331439786"></a>是</p>
</td>
</tr>
<tr id="row17961922271"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p198320194311"><a name="p198320194311"></a><a name="p198320194311"></a>152</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15332039786"><a name="p15332039786"></a><a name="p15332039786"></a>torch.Tensor.index_put_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p135331939186"><a name="p135331939186"></a><a name="p135331939186"></a>是</p>
</td>
</tr>
<tr id="row996112221675"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1783601437"><a name="p1783601437"></a><a name="p1783601437"></a>153</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1253343911816"><a name="p1253343911816"></a><a name="p1253343911816"></a>torch.Tensor.index_put</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p105331639483"><a name="p105331639483"></a><a name="p105331639483"></a>是</p>
</td>
</tr>
<tr id="row7961522574"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11839012434"><a name="p11839012434"></a><a name="p11839012434"></a>154</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1953313391682"><a name="p1953313391682"></a><a name="p1953313391682"></a>torch.Tensor.index_select</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115331539888"><a name="p115331539888"></a><a name="p115331539888"></a>是</p>
</td>
</tr>
<tr id="row696115221873"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1083603434"><a name="p1083603434"></a><a name="p1083603434"></a>155</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p053318395815"><a name="p053318395815"></a><a name="p053318395815"></a>torch.Tensor.indices</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p853353918817"><a name="p853353918817"></a><a name="p853353918817"></a>否</p>
</td>
</tr>
<tr id="row109627226711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p68410184311"><a name="p68410184311"></a><a name="p68410184311"></a>156</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p195330397812"><a name="p195330397812"></a><a name="p195330397812"></a>torch.Tensor.int</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16533153916817"><a name="p16533153916817"></a><a name="p16533153916817"></a>是</p>
</td>
</tr>
<tr id="row17962522574"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p48420154313"><a name="p48420154313"></a><a name="p48420154313"></a>157</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1453433915816"><a name="p1453433915816"></a><a name="p1453433915816"></a>torch.Tensor.int_repr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1053433919815"><a name="p1053433919815"></a><a name="p1053433919815"></a>否</p>
</td>
</tr>
<tr id="row29624221774"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1884205437"><a name="p1884205437"></a><a name="p1884205437"></a>158</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p953415391180"><a name="p953415391180"></a><a name="p953415391180"></a>torch.Tensor.inverse</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8254155310133"><a name="p8254155310133"></a><a name="p8254155310133"></a>是</p>
</td>
</tr>
<tr id="row19962172211716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p158419064314"><a name="p158419064314"></a><a name="p158419064314"></a>159</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p12534439387"><a name="p12534439387"></a><a name="p12534439387"></a>torch.Tensor.irfft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p135348391080"><a name="p135348391080"></a><a name="p135348391080"></a>否</p>
</td>
</tr>
<tr id="row296214225716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p984180184311"><a name="p984180184311"></a><a name="p984180184311"></a>160</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1853453913818"><a name="p1853453913818"></a><a name="p1853453913818"></a>torch.Tensor.is_contiguous</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1353411391888"><a name="p1353411391888"></a><a name="p1353411391888"></a>是</p>
</td>
</tr>
<tr id="row129621227718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p208480164312"><a name="p208480164312"></a><a name="p208480164312"></a>161</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p753433910811"><a name="p753433910811"></a><a name="p753433910811"></a>torch.Tensor.is_complex</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3534183915818"><a name="p3534183915818"></a><a name="p3534183915818"></a>是</p>
</td>
</tr>
<tr id="row1096212221070"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1484404439"><a name="p1484404439"></a><a name="p1484404439"></a>162</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5534183915818"><a name="p5534183915818"></a><a name="p5534183915818"></a>torch.Tensor.is_floating_point</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16534133917815"><a name="p16534133917815"></a><a name="p16534133917815"></a>是</p>
</td>
</tr>
<tr id="row696232213714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p38411024315"><a name="p38411024315"></a><a name="p38411024315"></a>163</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1953433917820"><a name="p1953433917820"></a><a name="p1953433917820"></a>torch.Tensor.is_pinned</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p353473910815"><a name="p353473910815"></a><a name="p353473910815"></a>是</p>
</td>
</tr>
<tr id="row896222212714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1684902436"><a name="p1684902436"></a><a name="p1684902436"></a>164</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p153433913813"><a name="p153433913813"></a><a name="p153433913813"></a>torch.Tensor.is_set_to</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p75341939686"><a name="p75341939686"></a><a name="p75341939686"></a>否</p>
</td>
</tr>
<tr id="row396214221579"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1384180174317"><a name="p1384180174317"></a><a name="p1384180174317"></a>165</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p105341639188"><a name="p105341639188"></a><a name="p105341639188"></a>torch.Tensor.is_shared</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p05341039583"><a name="p05341039583"></a><a name="p05341039583"></a>是</p>
</td>
</tr>
<tr id="row1596216221175"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1684200204318"><a name="p1684200204318"></a><a name="p1684200204318"></a>166</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1753443913813"><a name="p1753443913813"></a><a name="p1753443913813"></a>torch.Tensor.is_signed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p195341739484"><a name="p195341739484"></a><a name="p195341739484"></a>是</p>
</td>
</tr>
<tr id="row1496211222712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1885306436"><a name="p1885306436"></a><a name="p1885306436"></a>167</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1753413398816"><a name="p1753413398816"></a><a name="p1753413398816"></a>torch.Tensor.is_sparse</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115341739788"><a name="p115341739788"></a><a name="p115341739788"></a>是</p>
</td>
</tr>
<tr id="row3962152210717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14851016431"><a name="p14851016431"></a><a name="p14851016431"></a>168</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1953473911810"><a name="p1953473911810"></a><a name="p1953473911810"></a>torch.Tensor.item</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15346391982"><a name="p15346391982"></a><a name="p15346391982"></a>是</p>
</td>
</tr>
<tr id="row19962172216713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p28513054311"><a name="p28513054311"></a><a name="p28513054311"></a>169</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10534539788"><a name="p10534539788"></a><a name="p10534539788"></a>torch.Tensor.kthvalue</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1453417394816"><a name="p1453417394816"></a><a name="p1453417394816"></a>是</p>
</td>
</tr>
<tr id="row189621022177"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1085130154314"><a name="p1085130154314"></a><a name="p1085130154314"></a>170</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1453516391186"><a name="p1453516391186"></a><a name="p1453516391186"></a>torch.Tensor.le</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85351839388"><a name="p85351839388"></a><a name="p85351839388"></a>是</p>
</td>
</tr>
<tr id="row18962202213713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p48513014316"><a name="p48513014316"></a><a name="p48513014316"></a>171</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p105351839885"><a name="p105351839885"></a><a name="p105351839885"></a>torch.Tensor.le_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p105353397819"><a name="p105353397819"></a><a name="p105353397819"></a>是</p>
</td>
</tr>
<tr id="row29631222979"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15851301436"><a name="p15851301436"></a><a name="p15851301436"></a>172</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p453519391687"><a name="p453519391687"></a><a name="p453519391687"></a>torch.Tensor.lerp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p185356392088"><a name="p185356392088"></a><a name="p185356392088"></a>是</p>
</td>
</tr>
<tr id="row139637221971"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p98515014437"><a name="p98515014437"></a><a name="p98515014437"></a>173</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p553583917820"><a name="p553583917820"></a><a name="p553583917820"></a>torch.Tensor.lerp_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18627142151520"><a name="p18627142151520"></a><a name="p18627142151520"></a>是</p>
</td>
</tr>
<tr id="row13963222072"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p385120144314"><a name="p385120144314"></a><a name="p385120144314"></a>174</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1353517392086"><a name="p1353517392086"></a><a name="p1353517392086"></a>torch.Tensor.lgamma</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1253518391819"><a name="p1253518391819"></a><a name="p1253518391819"></a>否</p>
</td>
</tr>
<tr id="row296352220716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1685004433"><a name="p1685004433"></a><a name="p1685004433"></a>175</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1853515399818"><a name="p1853515399818"></a><a name="p1853515399818"></a>torch.Tensor.lgamma_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1753516391282"><a name="p1753516391282"></a><a name="p1753516391282"></a>否</p>
</td>
</tr>
<tr id="row0963192219715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p138514010437"><a name="p138514010437"></a><a name="p138514010437"></a>176</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p953512391815"><a name="p953512391815"></a><a name="p953512391815"></a>torch.Tensor.log</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1253518393814"><a name="p1253518393814"></a><a name="p1253518393814"></a>是</p>
</td>
</tr>
<tr id="row16963192217714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1852084311"><a name="p1852084311"></a><a name="p1852084311"></a>177</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1853511391589"><a name="p1853511391589"></a><a name="p1853511391589"></a>torch.Tensor.log_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115358391389"><a name="p115358391389"></a><a name="p115358391389"></a>是</p>
</td>
</tr>
<tr id="row0963112220718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16865011435"><a name="p16865011435"></a><a name="p16865011435"></a>178</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p753573912811"><a name="p753573912811"></a><a name="p753573912811"></a>torch.Tensor.logdet</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p553583914819"><a name="p553583914819"></a><a name="p553583914819"></a>否</p>
</td>
</tr>
<tr id="row596310223712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p986120144318"><a name="p986120144318"></a><a name="p986120144318"></a>179</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p12535113920816"><a name="p12535113920816"></a><a name="p12535113920816"></a>torch.Tensor.log10</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p753519392080"><a name="p753519392080"></a><a name="p753519392080"></a>是</p>
</td>
</tr>
<tr id="row109631622570"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p186160154311"><a name="p186160154311"></a><a name="p186160154311"></a>180</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17535183910816"><a name="p17535183910816"></a><a name="p17535183910816"></a>torch.Tensor.log10_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19535103911815"><a name="p19535103911815"></a><a name="p19535103911815"></a>是</p>
</td>
</tr>
<tr id="row1296322217716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11864044318"><a name="p11864044318"></a><a name="p11864044318"></a>181</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p135351439481"><a name="p135351439481"></a><a name="p135351439481"></a>torch.Tensor.log1p</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3535039786"><a name="p3535039786"></a><a name="p3535039786"></a>是</p>
</td>
</tr>
<tr id="row169631222475"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3861001431"><a name="p3861001431"></a><a name="p3861001431"></a>182</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1953620399810"><a name="p1953620399810"></a><a name="p1953620399810"></a>torch.Tensor.log1p_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p135361839483"><a name="p135361839483"></a><a name="p135361839483"></a>是</p>
</td>
</tr>
<tr id="row129631220715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p158660104312"><a name="p158660104312"></a><a name="p158660104312"></a>183</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p85361439986"><a name="p85361439986"></a><a name="p85361439986"></a>torch.Tensor.log2</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7536039285"><a name="p7536039285"></a><a name="p7536039285"></a>是</p>
</td>
</tr>
<tr id="row14963522575"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p138617044318"><a name="p138617044318"></a><a name="p138617044318"></a>184</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1753614394812"><a name="p1753614394812"></a><a name="p1753614394812"></a>torch.Tensor.log2_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12536193916818"><a name="p12536193916818"></a><a name="p12536193916818"></a>是</p>
</td>
</tr>
<tr id="row0963192217710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p28680174311"><a name="p28680174311"></a><a name="p28680174311"></a>185</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p353611396817"><a name="p353611396817"></a><a name="p353611396817"></a>torch.Tensor.log_normal_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p85361339788"><a name="p85361339788"></a><a name="p85361339788"></a>是</p>
</td>
</tr>
<tr id="row1396352211714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p38618015435"><a name="p38618015435"></a><a name="p38618015435"></a>186</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5536739885"><a name="p5536739885"></a><a name="p5536739885"></a>torch.Tensor.logsumexp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p366193013167"><a name="p366193013167"></a><a name="p366193013167"></a>是</p>
</td>
</tr>
<tr id="row596417221271"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6866019439"><a name="p6866019439"></a><a name="p6866019439"></a>187</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10536739584"><a name="p10536739584"></a><a name="p10536739584"></a>torch.Tensor.logical_and</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1153614391888"><a name="p1153614391888"></a><a name="p1153614391888"></a>是</p>
</td>
</tr>
<tr id="row396413221779"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p38613064310"><a name="p38613064310"></a><a name="p38613064310"></a>188</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1153619393818"><a name="p1153619393818"></a><a name="p1153619393818"></a>torch.Tensor.logical_and_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18536039386"><a name="p18536039386"></a><a name="p18536039386"></a>是</p>
</td>
</tr>
<tr id="row159645221876"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p0871205438"><a name="p0871205438"></a><a name="p0871205438"></a>189</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p953618391820"><a name="p953618391820"></a><a name="p953618391820"></a>torch.Tensor.logical_not</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1853693910815"><a name="p1853693910815"></a><a name="p1853693910815"></a>是</p>
</td>
</tr>
<tr id="row89649229712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p188710019436"><a name="p188710019436"></a><a name="p188710019436"></a>190</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p553663910815"><a name="p553663910815"></a><a name="p553663910815"></a>torch.Tensor.logical_not_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p75361739389"><a name="p75361739389"></a><a name="p75361739389"></a>是</p>
</td>
</tr>
<tr id="row10964172212710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p78715004313"><a name="p78715004313"></a><a name="p78715004313"></a>191</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p253618392813"><a name="p253618392813"></a><a name="p253618392813"></a>torch.Tensor.logical_or</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p953616394813"><a name="p953616394813"></a><a name="p953616394813"></a>是</p>
</td>
</tr>
<tr id="row119644229719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19871701438"><a name="p19871701438"></a><a name="p19871701438"></a>192</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10536439588"><a name="p10536439588"></a><a name="p10536439588"></a>torch.Tensor.logical_or_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p20536203916819"><a name="p20536203916819"></a><a name="p20536203916819"></a>是</p>
</td>
</tr>
<tr id="row196415221879"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p98790184312"><a name="p98790184312"></a><a name="p98790184312"></a>193</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p135362395816"><a name="p135362395816"></a><a name="p135362395816"></a>torch.Tensor.logical_xor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p155367391083"><a name="p155367391083"></a><a name="p155367391083"></a>否</p>
</td>
</tr>
<tr id="row696414225714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p118713094318"><a name="p118713094318"></a><a name="p118713094318"></a>194</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p65361739389"><a name="p65361739389"></a><a name="p65361739389"></a>torch.Tensor.logical_xor_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10536939884"><a name="p10536939884"></a><a name="p10536939884"></a>否</p>
</td>
</tr>
<tr id="row1296419221879"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p118715016430"><a name="p118715016430"></a><a name="p118715016430"></a>195</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p185370391787"><a name="p185370391787"></a><a name="p185370391787"></a>torch.Tensor.long</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1153773916816"><a name="p1153773916816"></a><a name="p1153773916816"></a>是</p>
</td>
</tr>
<tr id="row1596413226720"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p287190154314"><a name="p287190154314"></a><a name="p287190154314"></a>196</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1553716392814"><a name="p1553716392814"></a><a name="p1553716392814"></a>torch.Tensor.lstsq</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p653715398817"><a name="p653715398817"></a><a name="p653715398817"></a>否</p>
</td>
</tr>
<tr id="row119649221976"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1871003439"><a name="p1871003439"></a><a name="p1871003439"></a>197</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p145371639389"><a name="p145371639389"></a><a name="p145371639389"></a>torch.Tensor.lt</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1537203914812"><a name="p1537203914812"></a><a name="p1537203914812"></a>是</p>
</td>
</tr>
<tr id="row189649221719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7878024319"><a name="p7878024319"></a><a name="p7878024319"></a>198</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p553711393812"><a name="p553711393812"></a><a name="p553711393812"></a>torch.Tensor.lt_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13537123914812"><a name="p13537123914812"></a><a name="p13537123914812"></a>是</p>
</td>
</tr>
<tr id="row096452216717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3880094315"><a name="p3880094315"></a><a name="p3880094315"></a>199</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18537183914819"><a name="p18537183914819"></a><a name="p18537183914819"></a>torch.Tensor.lu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1153783919816"><a name="p1153783919816"></a><a name="p1153783919816"></a>是</p>
</td>
</tr>
<tr id="row1296482212720"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p28820094312"><a name="p28820094312"></a><a name="p28820094312"></a>200</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p353783919820"><a name="p353783919820"></a><a name="p353783919820"></a>torch.Tensor.lu_solve</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p95371539886"><a name="p95371539886"></a><a name="p95371539886"></a>是</p>
</td>
</tr>
<tr id="row11964122218719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p68870174318"><a name="p68870174318"></a><a name="p68870174318"></a>201</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p185371139385"><a name="p185371139385"></a><a name="p185371139385"></a>torch.Tensor.map_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1153720391782"><a name="p1153720391782"></a><a name="p1153720391782"></a>否</p>
</td>
</tr>
<tr id="row19964122374"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10881084315"><a name="p10881084315"></a><a name="p10881084315"></a>202</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p135371639486"><a name="p135371639486"></a><a name="p135371639486"></a>torch.Tensor.masked_scatter_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p65372391581"><a name="p65372391581"></a><a name="p65372391581"></a>是</p>
</td>
</tr>
<tr id="row199653222713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8889024312"><a name="p8889024312"></a><a name="p8889024312"></a>203</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14537163911815"><a name="p14537163911815"></a><a name="p14537163911815"></a>torch.Tensor.masked_scatter</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1653712394815"><a name="p1653712394815"></a><a name="p1653712394815"></a>是</p>
</td>
</tr>
<tr id="row4965162211710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1788904436"><a name="p1788904436"></a><a name="p1788904436"></a>204</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p653763913811"><a name="p653763913811"></a><a name="p653763913811"></a>torch.Tensor.masked_fill_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p165371039182"><a name="p165371039182"></a><a name="p165371039182"></a>是</p>
</td>
</tr>
<tr id="row1396515221478"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1388140204318"><a name="p1388140204318"></a><a name="p1388140204318"></a>205</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p175378397813"><a name="p175378397813"></a><a name="p175378397813"></a>torch.Tensor.masked_fill</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p45371399817"><a name="p45371399817"></a><a name="p45371399817"></a>是</p>
</td>
</tr>
<tr id="row196542215720"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p98830114320"><a name="p98830114320"></a><a name="p98830114320"></a>206</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p95373391188"><a name="p95373391188"></a><a name="p95373391188"></a>torch.Tensor.masked_select</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19537173913813"><a name="p19537173913813"></a><a name="p19537173913813"></a>是</p>
</td>
</tr>
<tr id="row1996519226719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2088150164316"><a name="p2088150164316"></a><a name="p2088150164316"></a>207</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17537193911811"><a name="p17537193911811"></a><a name="p17537193911811"></a>torch.Tensor.matmul</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p105379391813"><a name="p105379391813"></a><a name="p105379391813"></a>是</p>
</td>
</tr>
<tr id="row159653221476"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p208815019439"><a name="p208815019439"></a><a name="p208815019439"></a>208</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1253863911815"><a name="p1253863911815"></a><a name="p1253863911815"></a>torch.Tensor.matrix_power</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1453818391589"><a name="p1453818391589"></a><a name="p1453818391589"></a>是</p>
</td>
</tr>
<tr id="row096562216719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3881705435"><a name="p3881705435"></a><a name="p3881705435"></a>209</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p253823920814"><a name="p253823920814"></a><a name="p253823920814"></a>torch.Tensor.max</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p165388391988"><a name="p165388391988"></a><a name="p165388391988"></a>是</p>
</td>
</tr>
<tr id="row1796512221873"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9891084316"><a name="p9891084316"></a><a name="p9891084316"></a>210</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1653810391384"><a name="p1653810391384"></a><a name="p1653810391384"></a>torch.Tensor.mean</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p105384398819"><a name="p105384398819"></a><a name="p105384398819"></a>是</p>
</td>
</tr>
<tr id="row2965922771"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p158910174310"><a name="p158910174310"></a><a name="p158910174310"></a>211</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p55387397810"><a name="p55387397810"></a><a name="p55387397810"></a>torch.Tensor.median</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p25381239582"><a name="p25381239582"></a><a name="p25381239582"></a>是</p>
</td>
</tr>
<tr id="row39651722376"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18890020434"><a name="p18890020434"></a><a name="p18890020434"></a>212</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8538113915819"><a name="p8538113915819"></a><a name="p8538113915819"></a>torch.Tensor.min</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8538133920816"><a name="p8538133920816"></a><a name="p8538133920816"></a>是</p>
</td>
</tr>
<tr id="row1396519221675"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p138970164315"><a name="p138970164315"></a><a name="p138970164315"></a>213</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p253853918815"><a name="p253853918815"></a><a name="p253853918815"></a>torch.Tensor.mm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p195385391287"><a name="p195385391287"></a><a name="p195385391287"></a>是</p>
</td>
</tr>
<tr id="row1596512221715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p289208439"><a name="p289208439"></a><a name="p289208439"></a>214</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p053814391884"><a name="p053814391884"></a><a name="p053814391884"></a>torch.Tensor.mode</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p453812391811"><a name="p453812391811"></a><a name="p453812391811"></a>否</p>
</td>
</tr>
<tr id="row1796510221271"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14894024314"><a name="p14894024314"></a><a name="p14894024314"></a>215</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p85386398819"><a name="p85386398819"></a><a name="p85386398819"></a>torch.Tensor.mul</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p195381392089"><a name="p195381392089"></a><a name="p195381392089"></a>是</p>
</td>
</tr>
<tr id="row996502211713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6891005434"><a name="p6891005434"></a><a name="p6891005434"></a>216</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p653820391085"><a name="p653820391085"></a><a name="p653820391085"></a>torch.Tensor.mul_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115381439381"><a name="p115381439381"></a><a name="p115381439381"></a>是</p>
</td>
</tr>
<tr id="row179651223711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5898014438"><a name="p5898014438"></a><a name="p5898014438"></a>217</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p115386391883"><a name="p115386391883"></a><a name="p115386391883"></a>torch.Tensor.multinomial</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p122941220189"><a name="p122941220189"></a><a name="p122941220189"></a>是</p>
</td>
</tr>
<tr id="row99651322976"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p689150134317"><a name="p689150134317"></a><a name="p689150134317"></a>218</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p553817391788"><a name="p553817391788"></a><a name="p553817391788"></a>torch.Tensor.mv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p95380391882"><a name="p95380391882"></a><a name="p95380391882"></a>是</p>
</td>
</tr>
<tr id="row149660221070"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11908011433"><a name="p11908011433"></a><a name="p11908011433"></a>219</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p145388395811"><a name="p145388395811"></a><a name="p145388395811"></a>torch.Tensor.mvlgamma</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19538439383"><a name="p19538439383"></a><a name="p19538439383"></a>否</p>
</td>
</tr>
<tr id="row13966172213711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6901203439"><a name="p6901203439"></a><a name="p6901203439"></a>220</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p953823917816"><a name="p953823917816"></a><a name="p953823917816"></a>torch.Tensor.mvlgamma_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19539183919811"><a name="p19539183919811"></a><a name="p19539183919811"></a>否</p>
</td>
</tr>
<tr id="row139669221379"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p129010013436"><a name="p129010013436"></a><a name="p129010013436"></a>221</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p95397396811"><a name="p95397396811"></a><a name="p95397396811"></a>torch.Tensor.narrow</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p155394395817"><a name="p155394395817"></a><a name="p155394395817"></a>是</p>
</td>
</tr>
<tr id="row189663229715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1290180184310"><a name="p1290180184310"></a><a name="p1290180184310"></a>222</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p125394398814"><a name="p125394398814"></a><a name="p125394398814"></a>torch.Tensor.narrow_copy</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p353923916811"><a name="p353923916811"></a><a name="p353923916811"></a>是</p>
</td>
</tr>
<tr id="row1966322674"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p139017014316"><a name="p139017014316"></a><a name="p139017014316"></a>223</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1853913915815"><a name="p1853913915815"></a><a name="p1853913915815"></a>torch.Tensor.ndimension</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p653916396813"><a name="p653916396813"></a><a name="p653916396813"></a>是</p>
</td>
</tr>
<tr id="row89665229713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20901902439"><a name="p20901902439"></a><a name="p20901902439"></a>224</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15539539387"><a name="p15539539387"></a><a name="p15539539387"></a>torch.Tensor.ne</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p145399394817"><a name="p145399394817"></a><a name="p145399394817"></a>是</p>
</td>
</tr>
<tr id="row129662022776"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15908054315"><a name="p15908054315"></a><a name="p15908054315"></a>225</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p55392039680"><a name="p55392039680"></a><a name="p55392039680"></a>torch.Tensor.ne_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p115397391817"><a name="p115397391817"></a><a name="p115397391817"></a>是</p>
</td>
</tr>
<tr id="row12966122210713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10903013432"><a name="p10903013432"></a><a name="p10903013432"></a>226</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p95398392817"><a name="p95398392817"></a><a name="p95398392817"></a>torch.Tensor.neg</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p25396391181"><a name="p25396391181"></a><a name="p25396391181"></a>是</p>
</td>
</tr>
<tr id="row179669226717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1591170184318"><a name="p1591170184318"></a><a name="p1591170184318"></a>227</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p153920391086"><a name="p153920391086"></a><a name="p153920391086"></a>torch.Tensor.neg_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p55391639686"><a name="p55391639686"></a><a name="p55391639686"></a>是</p>
</td>
</tr>
<tr id="row396692219718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p109190154311"><a name="p109190154311"></a><a name="p109190154311"></a>228</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p053933916813"><a name="p053933916813"></a><a name="p053933916813"></a>torch.Tensor.nelement</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18539239286"><a name="p18539239286"></a><a name="p18539239286"></a>是</p>
</td>
</tr>
<tr id="row1196620226711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p09115012432"><a name="p09115012432"></a><a name="p09115012432"></a>229</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p195395391986"><a name="p195395391986"></a><a name="p195395391986"></a>torch.Tensor.nonzero</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7539739182"><a name="p7539739182"></a><a name="p7539739182"></a>是</p>
</td>
</tr>
<tr id="row8966822478"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p391160114317"><a name="p391160114317"></a><a name="p391160114317"></a>230</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19540153916818"><a name="p19540153916818"></a><a name="p19540153916818"></a>torch.Tensor.norm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p554063919820"><a name="p554063919820"></a><a name="p554063919820"></a>是</p>
</td>
</tr>
<tr id="row99661022277"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15913017431"><a name="p15913017431"></a><a name="p15913017431"></a>231</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p254011395817"><a name="p254011395817"></a><a name="p254011395817"></a>torch.Tensor.normal_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p20540639982"><a name="p20540639982"></a><a name="p20540639982"></a>是</p>
</td>
</tr>
<tr id="row69669221717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12911208432"><a name="p12911208432"></a><a name="p12911208432"></a>232</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p254103915812"><a name="p254103915812"></a><a name="p254103915812"></a>torch.Tensor.numel</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p05413391180"><a name="p05413391180"></a><a name="p05413391180"></a>是</p>
</td>
</tr>
<tr id="row1796672218715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2917034314"><a name="p2917034314"></a><a name="p2917034314"></a>233</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p554173920811"><a name="p554173920811"></a><a name="p554173920811"></a>torch.Tensor.numpy</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1154112391483"><a name="p1154112391483"></a><a name="p1154112391483"></a>否</p>
</td>
</tr>
<tr id="row1696682210715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1391120184312"><a name="p1391120184312"></a><a name="p1391120184312"></a>234</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p115414395820"><a name="p115414395820"></a><a name="p115414395820"></a>torch.Tensor.orgqr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p954115398813"><a name="p954115398813"></a><a name="p954115398813"></a>否</p>
</td>
</tr>
<tr id="row1596792220717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p69311094317"><a name="p69311094317"></a><a name="p69311094317"></a>235</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p125413391583"><a name="p125413391583"></a><a name="p125413391583"></a>torch.Tensor.ormqr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2541139588"><a name="p2541139588"></a><a name="p2541139588"></a>否</p>
</td>
</tr>
<tr id="row1096715221171"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p119311054311"><a name="p119311054311"></a><a name="p119311054311"></a>236</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p115412039585"><a name="p115412039585"></a><a name="p115412039585"></a>torch.Tensor.permute</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p17541113913820"><a name="p17541113913820"></a><a name="p17541113913820"></a>是</p>
</td>
</tr>
<tr id="row9967142213720"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18931106432"><a name="p18931106432"></a><a name="p18931106432"></a>237</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p75416393818"><a name="p75416393818"></a><a name="p75416393818"></a>torch.Tensor.pin_memory</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3541123919813"><a name="p3541123919813"></a><a name="p3541123919813"></a>否</p>
</td>
</tr>
<tr id="row189673221178"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p393702438"><a name="p393702438"></a><a name="p393702438"></a>238</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p554116393814"><a name="p554116393814"></a><a name="p554116393814"></a>torch.Tensor.pinverse</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1254116396811"><a name="p1254116396811"></a><a name="p1254116396811"></a>否</p>
</td>
</tr>
<tr id="row29671220714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6941702438"><a name="p6941702438"></a><a name="p6941702438"></a>239</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p454113391987"><a name="p454113391987"></a><a name="p454113391987"></a>torch.Tensor.polygamma</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1954113916817"><a name="p1954113916817"></a><a name="p1954113916817"></a>否</p>
</td>
</tr>
<tr id="row149679224714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p189413020433"><a name="p189413020433"></a><a name="p189413020433"></a>240</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17541339582"><a name="p17541339582"></a><a name="p17541339582"></a>torch.Tensor.polygamma_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16541153910820"><a name="p16541153910820"></a><a name="p16541153910820"></a>否</p>
</td>
</tr>
<tr id="row109677221674"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17941008434"><a name="p17941008434"></a><a name="p17941008434"></a>241</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1554118394810"><a name="p1554118394810"></a><a name="p1554118394810"></a>torch.Tensor.pow</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p25411639889"><a name="p25411639889"></a><a name="p25411639889"></a>是</p>
</td>
</tr>
<tr id="row59673224718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1394304432"><a name="p1394304432"></a><a name="p1394304432"></a>242</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1054120391810"><a name="p1054120391810"></a><a name="p1054120391810"></a>torch.Tensor.pow_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p654114391083"><a name="p654114391083"></a><a name="p654114391083"></a>是</p>
</td>
</tr>
<tr id="row19968182210710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5949064316"><a name="p5949064316"></a><a name="p5949064316"></a>243</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p105411391884"><a name="p105411391884"></a><a name="p105411391884"></a>torch.Tensor.prod</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15541639189"><a name="p15541639189"></a><a name="p15541639189"></a>是</p>
</td>
</tr>
<tr id="row109685225710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p169430184312"><a name="p169430184312"></a><a name="p169430184312"></a>244</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p554133910817"><a name="p554133910817"></a><a name="p554133910817"></a>torch.Tensor.put_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1542203920815"><a name="p1542203920815"></a><a name="p1542203920815"></a>是</p>
</td>
</tr>
<tr id="row4968102219710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p09413016436"><a name="p09413016436"></a><a name="p09413016436"></a>245</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1654283915817"><a name="p1654283915817"></a><a name="p1654283915817"></a>torch.Tensor.qr</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p254220391781"><a name="p254220391781"></a><a name="p254220391781"></a>是</p>
</td>
</tr>
<tr id="row149681222179"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4948014317"><a name="p4948014317"></a><a name="p4948014317"></a>246</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p354253914819"><a name="p354253914819"></a><a name="p354253914819"></a>torch.Tensor.qscheme</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14542639783"><a name="p14542639783"></a><a name="p14542639783"></a>否</p>
</td>
</tr>
<tr id="row179681622277"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p69413017437"><a name="p69413017437"></a><a name="p69413017437"></a>247</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p05428398818"><a name="p05428398818"></a><a name="p05428398818"></a>torch.Tensor.q_scale</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p05421639489"><a name="p05421639489"></a><a name="p05421639489"></a>否</p>
</td>
</tr>
<tr id="row296817222714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3948010439"><a name="p3948010439"></a><a name="p3948010439"></a>248</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6542839888"><a name="p6542839888"></a><a name="p6542839888"></a>torch.Tensor.q_zero_point</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1542133918816"><a name="p1542133918816"></a><a name="p1542133918816"></a>否</p>
</td>
</tr>
<tr id="row1796918222717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p794190204316"><a name="p794190204316"></a><a name="p794190204316"></a>249</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p254214391584"><a name="p254214391584"></a><a name="p254214391584"></a>torch.Tensor.q_per_channel_scales</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p654214391088"><a name="p654214391088"></a><a name="p654214391088"></a>否</p>
</td>
</tr>
<tr id="row29693225718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p194130204310"><a name="p194130204310"></a><a name="p194130204310"></a>250</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p75421039286"><a name="p75421039286"></a><a name="p75421039286"></a>torch.Tensor.q_per_channel_zero_points</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2542439084"><a name="p2542439084"></a><a name="p2542439084"></a>否</p>
</td>
</tr>
<tr id="row1796992210714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12941019437"><a name="p12941019437"></a><a name="p12941019437"></a>251</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9542163914816"><a name="p9542163914816"></a><a name="p9542163914816"></a>torch.Tensor.q_per_channel_axis</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3542173910812"><a name="p3542173910812"></a><a name="p3542173910812"></a>否</p>
</td>
</tr>
<tr id="row1696910221475"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3941807435"><a name="p3941807435"></a><a name="p3941807435"></a>252</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15542163917818"><a name="p15542163917818"></a><a name="p15542163917818"></a>torch.Tensor.random_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1754253913816"><a name="p1754253913816"></a><a name="p1754253913816"></a>是</p>
</td>
</tr>
<tr id="row2969172218710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19945054314"><a name="p19945054314"></a><a name="p19945054314"></a>253</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2542103912817"><a name="p2542103912817"></a><a name="p2542103912817"></a>torch.Tensor.reciprocal</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7542163911811"><a name="p7542163911811"></a><a name="p7542163911811"></a>是</p>
</td>
</tr>
<tr id="row996920221719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p149490184319"><a name="p149490184319"></a><a name="p149490184319"></a>254</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11542439182"><a name="p11542439182"></a><a name="p11542439182"></a>torch.Tensor.reciprocal_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p165426391081"><a name="p165426391081"></a><a name="p165426391081"></a>是</p>
</td>
</tr>
<tr id="row189693221875"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p179511010435"><a name="p179511010435"></a><a name="p179511010435"></a>255</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1654213918818"><a name="p1654213918818"></a><a name="p1654213918818"></a>torch.Tensor.record_stream</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p17542339983"><a name="p17542339983"></a><a name="p17542339983"></a>否</p>
</td>
</tr>
<tr id="row496972218715"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11951507438"><a name="p11951507438"></a><a name="p11951507438"></a>256</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13542183914813"><a name="p13542183914813"></a><a name="p13542183914813"></a>torch.Tensor.remainder</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p9542123911817"><a name="p9542123911817"></a><a name="p9542123911817"></a>是</p>
</td>
</tr>
<tr id="row1296922214714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1895180184320"><a name="p1895180184320"></a><a name="p1895180184320"></a>257</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2054316396817"><a name="p2054316396817"></a><a name="p2054316396817"></a>torch.Tensor.remainder_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p4543439982"><a name="p4543439982"></a><a name="p4543439982"></a>是</p>
</td>
</tr>
<tr id="row69694221374"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p29510015439"><a name="p29510015439"></a><a name="p29510015439"></a>258</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p254318391084"><a name="p254318391084"></a><a name="p254318391084"></a>torch.Tensor.renorm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1643113114200"><a name="p1643113114200"></a><a name="p1643113114200"></a>是</p>
</td>
</tr>
<tr id="row119697222072"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1595100434"><a name="p1595100434"></a><a name="p1595100434"></a>259</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17543153919817"><a name="p17543153919817"></a><a name="p17543153919817"></a>torch.Tensor.renorm_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13673103322011"><a name="p13673103322011"></a><a name="p13673103322011"></a>是</p>
</td>
</tr>
<tr id="row2096982219718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1951005435"><a name="p1951005435"></a><a name="p1951005435"></a>260</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1554373914814"><a name="p1554373914814"></a><a name="p1554373914814"></a>torch.Tensor.repeat</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19543193911818"><a name="p19543193911818"></a><a name="p19543193911818"></a>是</p>
</td>
</tr>
<tr id="row39694221879"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16953014314"><a name="p16953014314"></a><a name="p16953014314"></a>261</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p85436391784"><a name="p85436391784"></a><a name="p85436391784"></a>torch.Tensor.repeat_interleave</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p95430391983"><a name="p95430391983"></a><a name="p95430391983"></a>是</p>
</td>
</tr>
<tr id="row896910226711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p59519014317"><a name="p59519014317"></a><a name="p59519014317"></a>262</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p205433391087"><a name="p205433391087"></a><a name="p205433391087"></a>torch.Tensor.requires_grad_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p254319394819"><a name="p254319394819"></a><a name="p254319394819"></a>是</p>
</td>
</tr>
<tr id="row796914222710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18953017433"><a name="p18953017433"></a><a name="p18953017433"></a>263</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16543539985"><a name="p16543539985"></a><a name="p16543539985"></a>torch.Tensor.reshape</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18543639386"><a name="p18543639386"></a><a name="p18543639386"></a>是</p>
</td>
</tr>
<tr id="row49691122371"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1595180164318"><a name="p1595180164318"></a><a name="p1595180164318"></a>264</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1954319398812"><a name="p1954319398812"></a><a name="p1954319398812"></a>torch.Tensor.reshape_as</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p4543939481"><a name="p4543939481"></a><a name="p4543939481"></a>是</p>
</td>
</tr>
<tr id="row1197016221711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p49550134313"><a name="p49550134313"></a><a name="p49550134313"></a>265</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p954314391180"><a name="p954314391180"></a><a name="p954314391180"></a>torch.Tensor.resize_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p35438399814"><a name="p35438399814"></a><a name="p35438399814"></a>是</p>
</td>
</tr>
<tr id="row20970102212714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p99517064315"><a name="p99517064315"></a><a name="p99517064315"></a>266</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p454317392812"><a name="p454317392812"></a><a name="p454317392812"></a>torch.Tensor.resize_as_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5543439489"><a name="p5543439489"></a><a name="p5543439489"></a>是</p>
</td>
</tr>
<tr id="row1597017229718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p159520144316"><a name="p159520144316"></a><a name="p159520144316"></a>267</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p75437390816"><a name="p75437390816"></a><a name="p75437390816"></a>torch.Tensor.rfft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p054314396819"><a name="p054314396819"></a><a name="p054314396819"></a>否</p>
</td>
</tr>
<tr id="row797042215717"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p119514044313"><a name="p119514044313"></a><a name="p119514044313"></a>268</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p145434394810"><a name="p145434394810"></a><a name="p145434394810"></a>torch.Tensor.roll</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19543739486"><a name="p19543739486"></a><a name="p19543739486"></a>否</p>
</td>
</tr>
<tr id="row397019221976"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p89518014316"><a name="p89518014316"></a><a name="p89518014316"></a>269</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19543039984"><a name="p19543039984"></a><a name="p19543039984"></a>torch.Tensor.rot90</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p354383914812"><a name="p354383914812"></a><a name="p354383914812"></a>是</p>
</td>
</tr>
<tr id="row997082219719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p69511017431"><a name="p69511017431"></a><a name="p69511017431"></a>270</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p195442391885"><a name="p195442391885"></a><a name="p195442391885"></a>torch.Tensor.round</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6544439783"><a name="p6544439783"></a><a name="p6544439783"></a>是</p>
</td>
</tr>
<tr id="row397016221173"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p89560114314"><a name="p89560114314"></a><a name="p89560114314"></a>271</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p95446391189"><a name="p95446391189"></a><a name="p95446391189"></a>torch.Tensor.round_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p954411391189"><a name="p954411391189"></a><a name="p954411391189"></a>是</p>
</td>
</tr>
<tr id="row1597019228712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p39520018433"><a name="p39520018433"></a><a name="p39520018433"></a>272</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5544183917813"><a name="p5544183917813"></a><a name="p5544183917813"></a>torch.Tensor.rsqrt</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p25444391284"><a name="p25444391284"></a><a name="p25444391284"></a>是</p>
</td>
</tr>
<tr id="row49701122174"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18956024313"><a name="p18956024313"></a><a name="p18956024313"></a>273</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p185443394812"><a name="p185443394812"></a><a name="p185443394812"></a>torch.Tensor.rsqrt_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p654416399816"><a name="p654416399816"></a><a name="p654416399816"></a>是</p>
</td>
</tr>
<tr id="row197032210716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p09540184317"><a name="p09540184317"></a><a name="p09540184317"></a>274</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19544839688"><a name="p19544839688"></a><a name="p19544839688"></a>torch.Tensor.scatter</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2544143917816"><a name="p2544143917816"></a><a name="p2544143917816"></a>是</p>
</td>
</tr>
<tr id="row39707226710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p196504434"><a name="p196504434"></a><a name="p196504434"></a>275</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2544183912819"><a name="p2544183912819"></a><a name="p2544183912819"></a>torch.Tensor.scatter_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1544203919814"><a name="p1544203919814"></a><a name="p1544203919814"></a>是</p>
</td>
</tr>
<tr id="row119701322075"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p69615016435"><a name="p69615016435"></a><a name="p69615016435"></a>276</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1154463910812"><a name="p1154463910812"></a><a name="p1154463910812"></a>torch.Tensor.scatter_add_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1354413396815"><a name="p1354413396815"></a><a name="p1354413396815"></a>是</p>
</td>
</tr>
<tr id="row1497042219714"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18960064315"><a name="p18960064315"></a><a name="p18960064315"></a>277</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1454423916811"><a name="p1454423916811"></a><a name="p1454423916811"></a>torch.Tensor.scatter_add</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18544153916820"><a name="p18544153916820"></a><a name="p18544153916820"></a>是</p>
</td>
</tr>
<tr id="row13970162215718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p69618094313"><a name="p69618094313"></a><a name="p69618094313"></a>278</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2054410391689"><a name="p2054410391689"></a><a name="p2054410391689"></a>torch.Tensor.select</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p154417392810"><a name="p154417392810"></a><a name="p154417392810"></a>是</p>
</td>
</tr>
<tr id="row1497011222716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6961604434"><a name="p6961604434"></a><a name="p6961604434"></a>279</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1254420391484"><a name="p1254420391484"></a><a name="p1254420391484"></a>torch.Tensor.set_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p135441239882"><a name="p135441239882"></a><a name="p135441239882"></a>是</p>
</td>
</tr>
<tr id="row1397122215711"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p159617010436"><a name="p159617010436"></a><a name="p159617010436"></a>280</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p354473914818"><a name="p354473914818"></a><a name="p354473914818"></a>torch.Tensor.share_memory_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16544163911816"><a name="p16544163911816"></a><a name="p16544163911816"></a>否</p>
</td>
</tr>
<tr id="row29718221173"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1796905436"><a name="p1796905436"></a><a name="p1796905436"></a>281</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p7544439985"><a name="p7544439985"></a><a name="p7544439985"></a>torch.Tensor.short</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p254473918820"><a name="p254473918820"></a><a name="p254473918820"></a>是</p>
</td>
</tr>
<tr id="row99711322279"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p119670164317"><a name="p119670164317"></a><a name="p119670164317"></a>282</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8544239184"><a name="p8544239184"></a><a name="p8544239184"></a>torch.Tensor.sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1654413391787"><a name="p1654413391787"></a><a name="p1654413391787"></a>是</p>
</td>
</tr>
<tr id="row1971192217710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p179640184318"><a name="p179640184318"></a><a name="p179640184318"></a>283</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p354518393819"><a name="p354518393819"></a><a name="p354518393819"></a>torch.Tensor.sigmoid_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p155456391583"><a name="p155456391583"></a><a name="p155456391583"></a>是</p>
</td>
</tr>
<tr id="row79713221871"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15961808439"><a name="p15961808439"></a><a name="p15961808439"></a>284</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1554511395814"><a name="p1554511395814"></a><a name="p1554511395814"></a>torch.Tensor.sign</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p95456391785"><a name="p95456391785"></a><a name="p95456391785"></a>是</p>
</td>
</tr>
<tr id="row797112225716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16961705434"><a name="p16961705434"></a><a name="p16961705434"></a>285</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p165458391483"><a name="p165458391483"></a><a name="p165458391483"></a>torch.Tensor.sign_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1854511391980"><a name="p1854511391980"></a><a name="p1854511391980"></a>是</p>
</td>
</tr>
<tr id="row1797115220716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p199615013430"><a name="p199615013430"></a><a name="p199615013430"></a>286</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1254511391085"><a name="p1254511391085"></a><a name="p1254511391085"></a>torch.Tensor.sin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p654511391280"><a name="p654511391280"></a><a name="p654511391280"></a>是</p>
</td>
</tr>
<tr id="row197118228713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8961044310"><a name="p8961044310"></a><a name="p8961044310"></a>287</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p105459394812"><a name="p105459394812"></a><a name="p105459394812"></a>torch.Tensor.sin_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p754514391985"><a name="p754514391985"></a><a name="p754514391985"></a>是</p>
</td>
</tr>
<tr id="row7971172214718"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5961044315"><a name="p5961044315"></a><a name="p5961044315"></a>288</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p154512395813"><a name="p154512395813"></a><a name="p154512395813"></a>torch.Tensor.sinh</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p155457390815"><a name="p155457390815"></a><a name="p155457390815"></a>是</p>
</td>
</tr>
<tr id="row12971152220713"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p196209432"><a name="p196209432"></a><a name="p196209432"></a>289</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p175451839888"><a name="p175451839888"></a><a name="p175451839888"></a>torch.Tensor.sinh_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p454518393815"><a name="p454518393815"></a><a name="p454518393815"></a>是</p>
</td>
</tr>
<tr id="row11971122219712"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19968074312"><a name="p19968074312"></a><a name="p19968074312"></a>290</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p454516391481"><a name="p454516391481"></a><a name="p454516391481"></a>torch.Tensor.size</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p165452039086"><a name="p165452039086"></a><a name="p165452039086"></a>是</p>
</td>
</tr>
<tr id="row139711722178"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p129615084310"><a name="p129615084310"></a><a name="p129615084310"></a>291</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p65456391181"><a name="p65456391181"></a><a name="p65456391181"></a>torch.Tensor.slogdet</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p17545839986"><a name="p17545839986"></a><a name="p17545839986"></a>否</p>
</td>
</tr>
<tr id="row119719221710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p149619014316"><a name="p149619014316"></a><a name="p149619014316"></a>292</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p05459391686"><a name="p05459391686"></a><a name="p05459391686"></a>torch.Tensor.solve</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1754553911819"><a name="p1754553911819"></a><a name="p1754553911819"></a>否</p>
</td>
</tr>
<tr id="row397142217716"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1196605433"><a name="p1196605433"></a><a name="p1196605433"></a>293</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p135451394818"><a name="p135451394818"></a><a name="p135451394818"></a>torch.Tensor.sort</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1054511392813"><a name="p1054511392813"></a><a name="p1054511392813"></a>是</p>
</td>
</tr>
<tr id="row1997119221073"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1396190134317"><a name="p1396190134317"></a><a name="p1396190134317"></a>294</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p105451239183"><a name="p105451239183"></a><a name="p105451239183"></a>torch.Tensor.split</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18545539783"><a name="p18545539783"></a><a name="p18545539783"></a>是</p>
</td>
</tr>
<tr id="row397120226719"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p69713064312"><a name="p69713064312"></a><a name="p69713064312"></a>295</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p05458391589"><a name="p05458391589"></a><a name="p05458391589"></a>torch.Tensor.sparse_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1754511391087"><a name="p1754511391087"></a><a name="p1754511391087"></a>否</p>
</td>
</tr>
<tr id="row179729221473"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p397180174320"><a name="p397180174320"></a><a name="p397180174320"></a>296</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1754616391815"><a name="p1754616391815"></a><a name="p1754616391815"></a>torch.Tensor.sparse_dim</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p125461939589"><a name="p125461939589"></a><a name="p125461939589"></a>否</p>
</td>
</tr>
<tr id="row797217221174"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2971808436"><a name="p2971808436"></a><a name="p2971808436"></a>297</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p35462391684"><a name="p35462391684"></a><a name="p35462391684"></a>torch.Tensor.sqrt</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10546183917810"><a name="p10546183917810"></a><a name="p10546183917810"></a>是</p>
</td>
</tr>
<tr id="row149729223720"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p189780124313"><a name="p189780124313"></a><a name="p189780124313"></a>298</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p155461439385"><a name="p155461439385"></a><a name="p155461439385"></a>torch.Tensor.sqrt_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p054613396817"><a name="p054613396817"></a><a name="p054613396817"></a>是</p>
</td>
</tr>
<tr id="row69721422376"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p89716014430"><a name="p89716014430"></a><a name="p89716014430"></a>299</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10546153920819"><a name="p10546153920819"></a><a name="p10546153920819"></a>torch.Tensor.square</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15546103919819"><a name="p15546103919819"></a><a name="p15546103919819"></a>是</p>
</td>
</tr>
<tr id="row81092038775"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2097100174313"><a name="p2097100174313"></a><a name="p2097100174313"></a>300</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19809193412916"><a name="p19809193412916"></a><a name="p19809193412916"></a>torch.Tensor.square_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p38091934599"><a name="p38091934599"></a><a name="p38091934599"></a>是</p>
</td>
</tr>
<tr id="row39408441576"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p159750204316"><a name="p159750204316"></a><a name="p159750204316"></a>301</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p980915348911"><a name="p980915348911"></a><a name="p980915348911"></a>torch.Tensor.squeeze</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18091034494"><a name="p18091034494"></a><a name="p18091034494"></a>是</p>
</td>
</tr>
<tr id="row18555347279"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4970014439"><a name="p4970014439"></a><a name="p4970014439"></a>302</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p108093348910"><a name="p108093348910"></a><a name="p108093348910"></a>torch.Tensor.squeeze_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p138091034192"><a name="p138091034192"></a><a name="p138091034192"></a>是</p>
</td>
</tr>
<tr id="row106811519710"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4972017433"><a name="p4972017433"></a><a name="p4972017433"></a>303</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15810113414914"><a name="p15810113414914"></a><a name="p15810113414914"></a>torch.Tensor.std</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p68101534891"><a name="p68101534891"></a><a name="p68101534891"></a>是</p>
</td>
</tr>
<tr id="row194319591984"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p49711084318"><a name="p49711084318"></a><a name="p49711084318"></a>304</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p48101341790"><a name="p48101341790"></a><a name="p48101341790"></a>torch.Tensor.stft</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0810234797"><a name="p0810234797"></a><a name="p0810234797"></a>否</p>
</td>
</tr>
<tr id="row195887318912"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p99720134318"><a name="p99720134318"></a><a name="p99720134318"></a>305</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p981093411910"><a name="p981093411910"></a><a name="p981093411910"></a>torch.Tensor.storage</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12810153415911"><a name="p12810153415911"></a><a name="p12810153415911"></a>是</p>
</td>
</tr>
<tr id="row157651831290"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1597104435"><a name="p1597104435"></a><a name="p1597104435"></a>306</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p881018341398"><a name="p881018341398"></a><a name="p881018341398"></a>torch.Tensor.storage_offset</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1081043414914"><a name="p1081043414914"></a><a name="p1081043414914"></a>是</p>
</td>
</tr>
<tr id="row29327311915"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14979014317"><a name="p14979014317"></a><a name="p14979014317"></a>307</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2081033417916"><a name="p2081033417916"></a><a name="p2081033417916"></a>torch.Tensor.storage_type</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15810134291"><a name="p15810134291"></a><a name="p15810134291"></a>是</p>
</td>
</tr>
<tr id="row9848419911"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p179710054313"><a name="p179710054313"></a><a name="p179710054313"></a>308</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p168105347911"><a name="p168105347911"></a><a name="p168105347911"></a>torch.Tensor.stride</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p118105342917"><a name="p118105342917"></a><a name="p118105342917"></a>是</p>
</td>
</tr>
<tr id="row3237142919"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p897505439"><a name="p897505439"></a><a name="p897505439"></a>309</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p20810173411919"><a name="p20810173411919"></a><a name="p20810173411919"></a>torch.Tensor.sub</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p138101034498"><a name="p138101034498"></a><a name="p138101034498"></a>是</p>
</td>
</tr>
<tr id="row103815414917"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p39717054319"><a name="p39717054319"></a><a name="p39717054319"></a>310</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1681053419911"><a name="p1681053419911"></a><a name="p1681053419911"></a>torch.Tensor.sub_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1381017341094"><a name="p1381017341094"></a><a name="p1381017341094"></a>是</p>
</td>
</tr>
<tr id="row135321341697"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p297702435"><a name="p297702435"></a><a name="p297702435"></a>311</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2810634497"><a name="p2810634497"></a><a name="p2810634497"></a>torch.Tensor.sum</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p208101034093"><a name="p208101034093"></a><a name="p208101034093"></a>是</p>
</td>
</tr>
<tr id="row126762041792"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p159720019431"><a name="p159720019431"></a><a name="p159720019431"></a>312</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1181043419920"><a name="p1181043419920"></a><a name="p1181043419920"></a>torch.Tensor.sum_to_size</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5810163417914"><a name="p5810163417914"></a><a name="p5810163417914"></a>是</p>
</td>
</tr>
<tr id="row7829941596"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10972008436"><a name="p10972008436"></a><a name="p10972008436"></a>313</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p78108341695"><a name="p78108341695"></a><a name="p78108341695"></a>torch.Tensor.svd</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p198103344916"><a name="p198103344916"></a><a name="p198103344916"></a>否</p>
</td>
</tr>
<tr id="row149731641091"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p89819019431"><a name="p89819019431"></a><a name="p89819019431"></a>314</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p88108349918"><a name="p88108349918"></a><a name="p88108349918"></a>torch.Tensor.symeig</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0810173417913"><a name="p0810173417913"></a><a name="p0810173417913"></a>否</p>
</td>
</tr>
<tr id="row2133351910"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19816004317"><a name="p19816004317"></a><a name="p19816004317"></a>315</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p081015340910"><a name="p081015340910"></a><a name="p081015340910"></a>torch.Tensor.t</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1810183410912"><a name="p1810183410912"></a><a name="p1810183410912"></a>是</p>
</td>
</tr>
<tr id="row8276751195"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p59880124316"><a name="p59880124316"></a><a name="p59880124316"></a>316</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p178111341094"><a name="p178111341094"></a><a name="p178111341094"></a>torch.Tensor.t_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p38119341999"><a name="p38119341999"></a><a name="p38119341999"></a>是</p>
</td>
</tr>
<tr id="row11430125291"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16981204437"><a name="p16981204437"></a><a name="p16981204437"></a>317</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1781115343914"><a name="p1781115343914"></a><a name="p1781115343914"></a>torch.Tensor.to</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2081112345915"><a name="p2081112345915"></a><a name="p2081112345915"></a>是</p>
</td>
</tr>
<tr id="row2583851694"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20989013435"><a name="p20989013435"></a><a name="p20989013435"></a>318</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14811534294"><a name="p14811534294"></a><a name="p14811534294"></a>torch.Tensor.to_mkldnn</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10811103411917"><a name="p10811103411917"></a><a name="p10811103411917"></a>否</p>
</td>
</tr>
<tr id="row137330511911"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p109870194317"><a name="p109870194317"></a><a name="p109870194317"></a>319</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p281115341297"><a name="p281115341297"></a><a name="p281115341297"></a>torch.Tensor.take</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p108111434295"><a name="p108111434295"></a><a name="p108111434295"></a>是</p>
</td>
</tr>
<tr id="row8876254918"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p119810019432"><a name="p119810019432"></a><a name="p119810019432"></a>320</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9811173412915"><a name="p9811173412915"></a><a name="p9811173412915"></a>torch.Tensor.tan</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p118111034891"><a name="p118111034891"></a><a name="p118111034891"></a>是</p>
</td>
</tr>
<tr id="row102818613911"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15981014430"><a name="p15981014430"></a><a name="p15981014430"></a>321</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1811634990"><a name="p1811634990"></a><a name="p1811634990"></a>torch.Tensor.tan_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p108114341791"><a name="p108114341791"></a><a name="p108114341791"></a>是</p>
</td>
</tr>
<tr id="row16172126798"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p129818013437"><a name="p129818013437"></a><a name="p129818013437"></a>322</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1481114341595"><a name="p1481114341595"></a><a name="p1481114341595"></a>torch.Tensor.tanh</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1811134395"><a name="p1811134395"></a><a name="p1811134395"></a>是</p>
</td>
</tr>
<tr id="row183173615915"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12986014312"><a name="p12986014312"></a><a name="p12986014312"></a>323</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1581163420912"><a name="p1581163420912"></a><a name="p1581163420912"></a>torch.Tensor.tanh_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0811103417911"><a name="p0811103417911"></a><a name="p0811103417911"></a>是</p>
</td>
</tr>
<tr id="row204601265912"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p99860194320"><a name="p99860194320"></a><a name="p99860194320"></a>324</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1981118346911"><a name="p1981118346911"></a><a name="p1981118346911"></a>torch.Tensor.tolist</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p28118341898"><a name="p28118341898"></a><a name="p28118341898"></a>是</p>
</td>
</tr>
<tr id="row060414612918"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1698110184316"><a name="p1698110184316"></a><a name="p1698110184316"></a>325</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p88116342917"><a name="p88116342917"></a><a name="p88116342917"></a>torch.Tensor.topk</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p38111234695"><a name="p38111234695"></a><a name="p38111234695"></a>是</p>
</td>
</tr>
<tr id="row1775617616915"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1398190204319"><a name="p1398190204319"></a><a name="p1398190204319"></a>326</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p20811123415913"><a name="p20811123415913"></a><a name="p20811123415913"></a>torch.Tensor.to_sparse</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p681319343916"><a name="p681319343916"></a><a name="p681319343916"></a>否</p>
</td>
</tr>
<tr id="row990116614911"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8988012435"><a name="p8988012435"></a><a name="p8988012435"></a>327</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p12813934691"><a name="p12813934691"></a><a name="p12813934691"></a>torch.Tensor.trace</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p78131334791"><a name="p78131334791"></a><a name="p78131334791"></a>否</p>
</td>
</tr>
<tr id="row260976916"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16984017435"><a name="p16984017435"></a><a name="p16984017435"></a>328</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p188131342917"><a name="p188131342917"></a><a name="p188131342917"></a>torch.Tensor.transpose</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p17813113418918"><a name="p17813113418918"></a><a name="p17813113418918"></a>是</p>
</td>
</tr>
<tr id="row18212171791"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1198100194319"><a name="p1198100194319"></a><a name="p1198100194319"></a>329</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p281333412917"><a name="p281333412917"></a><a name="p281333412917"></a>torch.Tensor.transpose_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p88131634395"><a name="p88131634395"></a><a name="p88131634395"></a>是</p>
</td>
</tr>
<tr id="row0364271490"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1498100104318"><a name="p1498100104318"></a><a name="p1498100104318"></a>330</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p7813123413917"><a name="p7813123413917"></a><a name="p7813123413917"></a>torch.Tensor.triangular_solve</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p188131234099"><a name="p188131234099"></a><a name="p188131234099"></a>否</p>
</td>
</tr>
<tr id="row252410713915"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p498190164318"><a name="p498190164318"></a><a name="p498190164318"></a>331</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18813123410918"><a name="p18813123410918"></a><a name="p18813123410918"></a>torch.Tensor.tril</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3728021202412"><a name="p3728021202412"></a><a name="p3728021202412"></a>是</p>
</td>
</tr>
<tr id="row2669672099"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19850164312"><a name="p19850164312"></a><a name="p19850164312"></a>332</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p081313414914"><a name="p081313414914"></a><a name="p081313414914"></a>torch.Tensor.tril_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p192721923172412"><a name="p192721923172412"></a><a name="p192721923172412"></a>是</p>
</td>
</tr>
<tr id="row178281373911"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11985054316"><a name="p11985054316"></a><a name="p11985054316"></a>333</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p3813123414915"><a name="p3813123414915"></a><a name="p3813123414915"></a>torch.Tensor.triu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p88139341691"><a name="p88139341691"></a><a name="p88139341691"></a>是</p>
</td>
</tr>
<tr id="row498013719920"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p79814004318"><a name="p79814004318"></a><a name="p79814004318"></a>334</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p68148344918"><a name="p68148344918"></a><a name="p68148344918"></a>torch.Tensor.triu_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p581413341795"><a name="p581413341795"></a><a name="p581413341795"></a>是</p>
</td>
</tr>
<tr id="row213312810918"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p199916014317"><a name="p199916014317"></a><a name="p199916014317"></a>335</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18145348913"><a name="p18145348913"></a><a name="p18145348913"></a>torch.Tensor.true_divide</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p681423410915"><a name="p681423410915"></a><a name="p681423410915"></a>是</p>
</td>
</tr>
<tr id="row19302681691"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15991801433"><a name="p15991801433"></a><a name="p15991801433"></a>336</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19814534194"><a name="p19814534194"></a><a name="p19814534194"></a>torch.Tensor.true_divide_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p181413340914"><a name="p181413340914"></a><a name="p181413340914"></a>是</p>
</td>
</tr>
<tr id="row2445138594"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1999190174311"><a name="p1999190174311"></a><a name="p1999190174311"></a>337</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p58141344910"><a name="p58141344910"></a><a name="p58141344910"></a>torch.Tensor.trunc</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6543537192419"><a name="p6543537192419"></a><a name="p6543537192419"></a>是</p>
</td>
</tr>
<tr id="row10604486914"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10991903435"><a name="p10991903435"></a><a name="p10991903435"></a>338</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16814163418911"><a name="p16814163418911"></a><a name="p16814163418911"></a>torch.Tensor.trunc_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3611839172410"><a name="p3611839172410"></a><a name="p3611839172410"></a>是</p>
</td>
</tr>
<tr id="row14757188999"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16991017430"><a name="p16991017430"></a><a name="p16991017430"></a>339</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p78148344913"><a name="p78148344913"></a><a name="p78148344913"></a>torch.Tensor.type</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p58148341091"><a name="p58148341091"></a><a name="p58148341091"></a>是</p>
</td>
</tr>
<tr id="row1791611811913"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p89915018430"><a name="p89915018430"></a><a name="p89915018430"></a>340</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p138141234692"><a name="p138141234692"></a><a name="p138141234692"></a>torch.Tensor.type_as</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2814034196"><a name="p2814034196"></a><a name="p2814034196"></a>是</p>
</td>
</tr>
<tr id="row188417914911"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p399110194317"><a name="p399110194317"></a><a name="p399110194317"></a>341</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p118141634699"><a name="p118141634699"></a><a name="p118141634699"></a>torch.Tensor.unbind</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14814434991"><a name="p14814434991"></a><a name="p14814434991"></a>是</p>
</td>
</tr>
<tr id="row1622019094"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11991405432"><a name="p11991405432"></a><a name="p11991405432"></a>342</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1814634991"><a name="p1814634991"></a><a name="p1814634991"></a>torch.Tensor.unfold</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13814143413910"><a name="p13814143413910"></a><a name="p13814143413910"></a>是</p>
</td>
</tr>
<tr id="row183881491297"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p129913014320"><a name="p129913014320"></a><a name="p129913014320"></a>343</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p48144347917"><a name="p48144347917"></a><a name="p48144347917"></a>torch.Tensor.uniform_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1181493417913"><a name="p1181493417913"></a><a name="p1181493417913"></a>是</p>
</td>
</tr>
<tr id="row7549491199"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p109915013436"><a name="p109915013436"></a><a name="p109915013436"></a>344</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p3814234292"><a name="p3814234292"></a><a name="p3814234292"></a>torch.Tensor.unique</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p188141434597"><a name="p188141434597"></a><a name="p188141434597"></a>是</p>
</td>
</tr>
<tr id="row771715914917"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p899180104312"><a name="p899180104312"></a><a name="p899180104312"></a>345</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16814183411911"><a name="p16814183411911"></a><a name="p16814183411911"></a>torch.Tensor.unique_consecutive</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1181420345918"><a name="p1181420345918"></a><a name="p1181420345918"></a>否</p>
</td>
</tr>
<tr id="row13876191890"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p299100144310"><a name="p299100144310"></a><a name="p299100144310"></a>346</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p981419343913"><a name="p981419343913"></a><a name="p981419343913"></a>torch.Tensor.unsqueeze</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p78141334392"><a name="p78141334392"></a><a name="p78141334392"></a>是</p>
</td>
</tr>
<tr id="row946131018912"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2099190154314"><a name="p2099190154314"></a><a name="p2099190154314"></a>347</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p178151934490"><a name="p178151934490"></a><a name="p178151934490"></a>torch.Tensor.unsqueeze_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16815113417911"><a name="p16815113417911"></a><a name="p16815113417911"></a>是</p>
</td>
</tr>
<tr id="row32041410997"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9998013433"><a name="p9998013433"></a><a name="p9998013433"></a>348</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p108155343915"><a name="p108155343915"></a><a name="p108155343915"></a>torch.Tensor.values</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1081543412919"><a name="p1081543412919"></a><a name="p1081543412919"></a>否</p>
</td>
</tr>
<tr id="row236414108918"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1999018431"><a name="p1999018431"></a><a name="p1999018431"></a>349</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p681514343919"><a name="p681514343919"></a><a name="p681514343919"></a>torch.Tensor.var</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14815183411911"><a name="p14815183411911"></a><a name="p14815183411911"></a>否</p>
</td>
</tr>
<tr id="row135241410695"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8996015436"><a name="p8996015436"></a><a name="p8996015436"></a>350</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p198156349920"><a name="p198156349920"></a><a name="p198156349920"></a>torch.Tensor.view</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1815183417917"><a name="p1815183417917"></a><a name="p1815183417917"></a>是</p>
</td>
</tr>
<tr id="row268511103912"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p179915020433"><a name="p179915020433"></a><a name="p179915020433"></a>351</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p148155341398"><a name="p148155341398"></a><a name="p148155341398"></a>torch.Tensor.view_as</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1481523416912"><a name="p1481523416912"></a><a name="p1481523416912"></a>是</p>
</td>
</tr>
<tr id="row178373105917"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16991802438"><a name="p16991802438"></a><a name="p16991802438"></a>352</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18157343917"><a name="p18157343917"></a><a name="p18157343917"></a>torch.Tensor.where</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8815163411915"><a name="p8815163411915"></a><a name="p8815163411915"></a>是</p>
</td>
</tr>
<tr id="row0989910292"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p49911014315"><a name="p49911014315"></a><a name="p49911014315"></a>353</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p208154349915"><a name="p208154349915"></a><a name="p208154349915"></a>torch.Tensor.zero_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8815193412911"><a name="p8815193412911"></a><a name="p8815193412911"></a>是</p>
</td>
</tr>
<tr id="row1014141111918"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p189950204317"><a name="p189950204317"></a><a name="p189950204317"></a>354</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p881533417915"><a name="p881533417915"></a><a name="p881533417915"></a>torch.BoolTensor</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2815193419911"><a name="p2815193419911"></a><a name="p2815193419911"></a>是</p>
</td>
</tr>
<tr id="row1302511697"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p99910144311"><a name="p99910144311"></a><a name="p99910144311"></a>355</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p98159340911"><a name="p98159340911"></a><a name="p98159340911"></a>torch.BoolTensor.all</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p19815153414912"><a name="p19815153414912"></a><a name="p19815153414912"></a>是</p>
</td>
</tr>
<tr id="row1046981117910"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9100903439"><a name="p9100903439"></a><a name="p9100903439"></a>356</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17815113411911"><a name="p17815113411911"></a><a name="p17815113411911"></a>torch.BoolTensor.any</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p88155341496"><a name="p88155341496"></a><a name="p88155341496"></a>是</p>
</td>
</tr>
</tbody>
</table>

<h2 id="Layers-torch-nn">Layers \(torch.nn\)</h2>

<a name="table8705205810121"></a>
<table><thead align="left"><tr id="row2070595881212"><th class="cellrowborder" valign="top" width="8.000000000000002%" id="mcps1.1.4.1.1"><p id="p8163151816433"><a name="p8163151816433"></a><a name="p8163151816433"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="71.50000000000001%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20.5%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row11705105816125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1312305310432"><a name="p1312305310432"></a><a name="p1312305310432"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p125949131510"><a name="p125949131510"></a><a name="p125949131510"></a>torch.nn.Parameter</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p75948171514"><a name="p75948171514"></a><a name="p75948171514"></a>是</p>
</td>
</tr>
<tr id="row870525861213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1123253164317"><a name="p1123253164317"></a><a name="p1123253164317"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p155951141515"><a name="p155951141515"></a><a name="p155951141515"></a>torch.nn.Module</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p115951214155"><a name="p115951214155"></a><a name="p115951214155"></a>是</p>
</td>
</tr>
<tr id="row87051658141219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p0123145334316"><a name="p0123145334316"></a><a name="p0123145334316"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1859510161517"><a name="p1859510161517"></a><a name="p1859510161517"></a>torch.nn.Module.add_module</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p17595171131511"><a name="p17595171131511"></a><a name="p17595171131511"></a>是</p>
</td>
</tr>
<tr id="row147051958111214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p31231753194316"><a name="p31231753194316"></a><a name="p31231753194316"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p35951214152"><a name="p35951214152"></a><a name="p35951214152"></a>torch.nn.Module.apply</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p55951410157"><a name="p55951410157"></a><a name="p55951410157"></a>是</p>
</td>
</tr>
<tr id="row11705558181218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11123165354311"><a name="p11123165354311"></a><a name="p11123165354311"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p25951117156"><a name="p25951117156"></a><a name="p25951117156"></a>torch.nn.Module.bfloat16</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p17595121161511"><a name="p17595121161511"></a><a name="p17595121161511"></a>否</p>
</td>
</tr>
<tr id="row1370516586127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p91231053144311"><a name="p91231053144311"></a><a name="p91231053144311"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p115951714159"><a name="p115951714159"></a><a name="p115951714159"></a>torch.nn.Module.buffers</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p16595141121517"><a name="p16595141121517"></a><a name="p16595141121517"></a>是</p>
</td>
</tr>
<tr id="row19706185811121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p13123195374314"><a name="p13123195374314"></a><a name="p13123195374314"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p659551131512"><a name="p659551131512"></a><a name="p659551131512"></a>torch.nn.Module.children</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p19595101161516"><a name="p19595101161516"></a><a name="p19595101161516"></a>是</p>
</td>
</tr>
<tr id="row6706158111215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9123105344319"><a name="p9123105344319"></a><a name="p9123105344319"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p25957141517"><a name="p25957141517"></a><a name="p25957141517"></a>torch.nn.Module.cpu</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1859515181510"><a name="p1859515181510"></a><a name="p1859515181510"></a>是</p>
</td>
</tr>
<tr id="row9706185811214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p512395344314"><a name="p512395344314"></a><a name="p512395344314"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p759518131511"><a name="p759518131511"></a><a name="p759518131511"></a>torch.nn.Module.cuda</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p4595191201514"><a name="p4595191201514"></a><a name="p4595191201514"></a>否</p>
</td>
</tr>
<tr id="row13706758101211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1612320539438"><a name="p1612320539438"></a><a name="p1612320539438"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1659581141510"><a name="p1659581141510"></a><a name="p1659581141510"></a>torch.nn.Module.double</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p65950119154"><a name="p65950119154"></a><a name="p65950119154"></a>否</p>
</td>
</tr>
<tr id="row1970617588129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p212325310438"><a name="p212325310438"></a><a name="p212325310438"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p45954101519"><a name="p45954101519"></a><a name="p45954101519"></a>torch.nn.Module.dump_patches</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p759517121513"><a name="p759517121513"></a><a name="p759517121513"></a>是</p>
</td>
</tr>
<tr id="row47061258131213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p101239531437"><a name="p101239531437"></a><a name="p101239531437"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p75951815152"><a name="p75951815152"></a><a name="p75951815152"></a>torch.nn.Module.eval</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p059511151519"><a name="p059511151519"></a><a name="p059511151519"></a>是</p>
</td>
</tr>
<tr id="row20706195819124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1912365319432"><a name="p1912365319432"></a><a name="p1912365319432"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p195958111154"><a name="p195958111154"></a><a name="p195958111154"></a>torch.nn.Module.extra_repr</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1759512151513"><a name="p1759512151513"></a><a name="p1759512151513"></a>是</p>
</td>
</tr>
<tr id="row770625817125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p712335313435"><a name="p712335313435"></a><a name="p712335313435"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p17595311155"><a name="p17595311155"></a><a name="p17595311155"></a>torch.nn.Module.float</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p25955151520"><a name="p25955151520"></a><a name="p25955151520"></a>是</p>
</td>
</tr>
<tr id="row3706058171218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1012395304318"><a name="p1012395304318"></a><a name="p1012395304318"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p105962116150"><a name="p105962116150"></a><a name="p105962116150"></a>torch.nn.Module.forward</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p12596141141512"><a name="p12596141141512"></a><a name="p12596141141512"></a>是</p>
</td>
</tr>
<tr id="row2070615583120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p17123953174313"><a name="p17123953174313"></a><a name="p17123953174313"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p19596411154"><a name="p19596411154"></a><a name="p19596411154"></a>torch.nn.Module.half</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p13596161191511"><a name="p13596161191511"></a><a name="p13596161191511"></a>是</p>
</td>
</tr>
<tr id="row1170635811219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1612445374310"><a name="p1612445374310"></a><a name="p1612445374310"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p165961119156"><a name="p165961119156"></a><a name="p165961119156"></a>torch.nn.Module.load_state_dict</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p759651111514"><a name="p759651111514"></a><a name="p759651111514"></a>是</p>
</td>
</tr>
<tr id="row197061058181216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p13124753144314"><a name="p13124753144314"></a><a name="p13124753144314"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p11596141111520"><a name="p11596141111520"></a><a name="p11596141111520"></a>torch.nn.Module.modules</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p35962151514"><a name="p35962151514"></a><a name="p35962151514"></a>是</p>
</td>
</tr>
<tr id="row970615587122"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p412435314316"><a name="p412435314316"></a><a name="p412435314316"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p55962117152"><a name="p55962117152"></a><a name="p55962117152"></a>torch.nn.Module.named_buffers</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p559611116151"><a name="p559611116151"></a><a name="p559611116151"></a>是</p>
</td>
</tr>
<tr id="row170615589128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p101241853184316"><a name="p101241853184316"></a><a name="p101241853184316"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p18596131131515"><a name="p18596131131515"></a><a name="p18596131131515"></a>torch.nn.Module.named_children</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1859612171520"><a name="p1859612171520"></a><a name="p1859612171520"></a>是</p>
</td>
</tr>
<tr id="row1970618585124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p161243539434"><a name="p161243539434"></a><a name="p161243539434"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p6596101131512"><a name="p6596101131512"></a><a name="p6596101131512"></a>torch.nn.Module.named_modules</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1359617117157"><a name="p1359617117157"></a><a name="p1359617117157"></a>是</p>
</td>
</tr>
<tr id="row5706358131211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p121241353164316"><a name="p121241353164316"></a><a name="p121241353164316"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p145968161518"><a name="p145968161518"></a><a name="p145968161518"></a>torch.nn.Module.named_parameters</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p195966171511"><a name="p195966171511"></a><a name="p195966171511"></a>是</p>
</td>
</tr>
<tr id="row37071758121218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1712485374319"><a name="p1712485374319"></a><a name="p1712485374319"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1959616117158"><a name="p1959616117158"></a><a name="p1959616117158"></a>torch.nn.Module.parameters</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p759612110159"><a name="p759612110159"></a><a name="p759612110159"></a>是</p>
</td>
</tr>
<tr id="row6707758161219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p912455324318"><a name="p912455324318"></a><a name="p912455324318"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p165961111156"><a name="p165961111156"></a><a name="p165961111156"></a>torch.nn.Module.register_backward_hook</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p193164121552"><a name="p193164121552"></a><a name="p193164121552"></a>是</p>
</td>
</tr>
<tr id="row8707458161212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1712517538437"><a name="p1712517538437"></a><a name="p1712517538437"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1759611121516"><a name="p1759611121516"></a><a name="p1759611121516"></a>torch.nn.Module.register_buffer</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p35961215159"><a name="p35961215159"></a><a name="p35961215159"></a>是</p>
</td>
</tr>
<tr id="row1270712587122"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1912505317431"><a name="p1912505317431"></a><a name="p1912505317431"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p75965111519"><a name="p75965111519"></a><a name="p75965111519"></a>torch.nn.Module.register_forward_hook</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p0646814358"><a name="p0646814358"></a><a name="p0646814358"></a>是</p>
</td>
</tr>
<tr id="row157071158191212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p12125165354318"><a name="p12125165354318"></a><a name="p12125165354318"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p105968114153"><a name="p105968114153"></a><a name="p105968114153"></a>torch.nn.Module.register_forward_pre_hook</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p14264716755"><a name="p14264716755"></a><a name="p14264716755"></a>是</p>
</td>
</tr>
<tr id="row1870765801212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9125155304318"><a name="p9125155304318"></a><a name="p9125155304318"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p359751201517"><a name="p359751201517"></a><a name="p359751201517"></a>torch.nn.Module.register_parameter</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p2597312155"><a name="p2597312155"></a><a name="p2597312155"></a>是</p>
</td>
</tr>
<tr id="row1470715810129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p41252053184319"><a name="p41252053184319"></a><a name="p41252053184319"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p195970181512"><a name="p195970181512"></a><a name="p195970181512"></a>torch.nn.Module.requires_grad_</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p6597913150"><a name="p6597913150"></a><a name="p6597913150"></a>是</p>
</td>
</tr>
<tr id="row1870755871217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p101257537431"><a name="p101257537431"></a><a name="p101257537431"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p9597141131510"><a name="p9597141131510"></a><a name="p9597141131510"></a>torch.nn.Module.state_dict</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p75970111515"><a name="p75970111515"></a><a name="p75970111515"></a>是</p>
</td>
</tr>
<tr id="row17071358141219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1712515324312"><a name="p1712515324312"></a><a name="p1712515324312"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p175977117156"><a name="p175977117156"></a><a name="p175977117156"></a>torch.nn.Module.to</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1659714111515"><a name="p1659714111515"></a><a name="p1659714111515"></a>是</p>
</td>
</tr>
<tr id="row1070725831212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p4125453164318"><a name="p4125453164318"></a><a name="p4125453164318"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p175971018153"><a name="p175971018153"></a><a name="p175971018153"></a>torch.nn.Module.train</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1059712116151"><a name="p1059712116151"></a><a name="p1059712116151"></a>是</p>
</td>
</tr>
<tr id="row13707185810127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p121251353174311"><a name="p121251353174311"></a><a name="p121251353174311"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p17597111131515"><a name="p17597111131515"></a><a name="p17597111131515"></a>torch.nn.Module.type</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p155979117155"><a name="p155979117155"></a><a name="p155979117155"></a>是</p>
</td>
</tr>
<tr id="row170725811124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p612515530432"><a name="p612515530432"></a><a name="p612515530432"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p059715115155"><a name="p059715115155"></a><a name="p059715115155"></a>torch.nn.Module.zero_grad</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1959711118155"><a name="p1959711118155"></a><a name="p1959711118155"></a>是</p>
</td>
</tr>
<tr id="row4707358191215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p181258531439"><a name="p181258531439"></a><a name="p181258531439"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p559710131516"><a name="p559710131516"></a><a name="p559710131516"></a>torch.nn.Sequential</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p259711181518"><a name="p259711181518"></a><a name="p259711181518"></a>是</p>
</td>
</tr>
<tr id="row170717583124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p81261531436"><a name="p81261531436"></a><a name="p81261531436"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p559710111155"><a name="p559710111155"></a><a name="p559710111155"></a>torch.nn.ModuleList</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p14597515152"><a name="p14597515152"></a><a name="p14597515152"></a>是</p>
</td>
</tr>
<tr id="row8707135871217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p101262537436"><a name="p101262537436"></a><a name="p101262537436"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p165972112152"><a name="p165972112152"></a><a name="p165972112152"></a>torch.nn.ModuleList.append</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p35977120158"><a name="p35977120158"></a><a name="p35977120158"></a>是</p>
</td>
</tr>
<tr id="row5707358141215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p71266531434"><a name="p71266531434"></a><a name="p71266531434"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p159711112158"><a name="p159711112158"></a><a name="p159711112158"></a>torch.nn.ModuleList.extend</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p165971191510"><a name="p165971191510"></a><a name="p165971191510"></a>是</p>
</td>
</tr>
<tr id="row11708115819121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111261453184317"><a name="p111261453184317"></a><a name="p111261453184317"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p13597418154"><a name="p13597418154"></a><a name="p13597418154"></a>torch.nn.ModuleList.insert</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1059710111516"><a name="p1059710111516"></a><a name="p1059710111516"></a>是</p>
</td>
</tr>
<tr id="row13708135816124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1412616534435"><a name="p1412616534435"></a><a name="p1412616534435"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1559701101514"><a name="p1559701101514"></a><a name="p1559701101514"></a>torch.nn.ModuleDict</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p85971918151"><a name="p85971918151"></a><a name="p85971918151"></a>是</p>
</td>
</tr>
<tr id="row17708758191210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p51266539430"><a name="p51266539430"></a><a name="p51266539430"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p2059813181519"><a name="p2059813181519"></a><a name="p2059813181519"></a>torch.nn.ModuleDict.clear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p55981011155"><a name="p55981011155"></a><a name="p55981011155"></a>是</p>
</td>
</tr>
<tr id="row870825821218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p14126165315433"><a name="p14126165315433"></a><a name="p14126165315433"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p75983114154"><a name="p75983114154"></a><a name="p75983114154"></a>torch.nn.ModuleDict.items</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p16598513154"><a name="p16598513154"></a><a name="p16598513154"></a>是</p>
</td>
</tr>
<tr id="row12708145818128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p91261853144311"><a name="p91261853144311"></a><a name="p91261853144311"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p65985181515"><a name="p65985181515"></a><a name="p65985181515"></a>torch.nn.ModuleDict.keys</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p55988120153"><a name="p55988120153"></a><a name="p55988120153"></a>是</p>
</td>
</tr>
<tr id="row1870885811120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10126853114314"><a name="p10126853114314"></a><a name="p10126853114314"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1659831131513"><a name="p1659831131513"></a><a name="p1659831131513"></a>torch.nn.ModuleDict.pop</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p165981114152"><a name="p165981114152"></a><a name="p165981114152"></a>是</p>
</td>
</tr>
<tr id="row970855841214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1212617539439"><a name="p1212617539439"></a><a name="p1212617539439"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p8598131161511"><a name="p8598131161511"></a><a name="p8598131161511"></a>torch.nn.ModuleDict.update</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p16598214151"><a name="p16598214151"></a><a name="p16598214151"></a>是</p>
</td>
</tr>
<tr id="row1070805801215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p2127155384319"><a name="p2127155384319"></a><a name="p2127155384319"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1259812191511"><a name="p1259812191511"></a><a name="p1259812191511"></a>torch.nn.ModuleDict.values</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p18598161161513"><a name="p18598161161513"></a><a name="p18598161161513"></a>是</p>
</td>
</tr>
<tr id="row1670855831219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1712715354317"><a name="p1712715354317"></a><a name="p1712715354317"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1559821151516"><a name="p1559821151516"></a><a name="p1559821151516"></a>torch.nn.ParameterList</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p12598011155"><a name="p12598011155"></a><a name="p12598011155"></a>是</p>
</td>
</tr>
<tr id="row670815589121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p51271153144318"><a name="p51271153144318"></a><a name="p51271153144318"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p65987101520"><a name="p65987101520"></a><a name="p65987101520"></a>torch.nn.ParameterList.append</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p17598131191513"><a name="p17598131191513"></a><a name="p17598131191513"></a>是</p>
</td>
</tr>
<tr id="row3708258131219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111275532436"><a name="p111275532436"></a><a name="p111275532436"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1859815161515"><a name="p1859815161515"></a><a name="p1859815161515"></a>torch.nn.ParameterList.extend</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p759810118155"><a name="p759810118155"></a><a name="p759810118155"></a>是</p>
</td>
</tr>
<tr id="row1470835820122"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p161279539439"><a name="p161279539439"></a><a name="p161279539439"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p10598214159"><a name="p10598214159"></a><a name="p10598214159"></a>torch.nn.ParameterDict</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p05981610158"><a name="p05981610158"></a><a name="p05981610158"></a>是</p>
</td>
</tr>
<tr id="row10708058101217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10127153204317"><a name="p10127153204317"></a><a name="p10127153204317"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1359811111158"><a name="p1359811111158"></a><a name="p1359811111158"></a>torch.nn.ParameterDict.clear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p15598914159"><a name="p15598914159"></a><a name="p15598914159"></a>是</p>
</td>
</tr>
<tr id="row77085584121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1912717536431"><a name="p1912717536431"></a><a name="p1912717536431"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p185985111151"><a name="p185985111151"></a><a name="p185985111151"></a>torch.nn.ParameterDict.items</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p25981514154"><a name="p25981514154"></a><a name="p25981514154"></a>是</p>
</td>
</tr>
<tr id="row1970835817126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p2012785317439"><a name="p2012785317439"></a><a name="p2012785317439"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1559814151513"><a name="p1559814151513"></a><a name="p1559814151513"></a>torch.nn.ParameterDict.keys</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p145984161510"><a name="p145984161510"></a><a name="p145984161510"></a>是</p>
</td>
</tr>
<tr id="row127081058121210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p15127145316436"><a name="p15127145316436"></a><a name="p15127145316436"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p559991101514"><a name="p559991101514"></a><a name="p559991101514"></a>torch.nn.ParameterDict.pop</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p19599911156"><a name="p19599911156"></a><a name="p19599911156"></a>是</p>
</td>
</tr>
<tr id="row1670919584124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p0127145319433"><a name="p0127145319433"></a><a name="p0127145319433"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p55995191519"><a name="p55995191519"></a><a name="p55995191519"></a>torch.nn.ParameterDict.update</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p759981141517"><a name="p759981141517"></a><a name="p759981141517"></a>是</p>
</td>
</tr>
<tr id="row7709558101218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p181271653114320"><a name="p181271653114320"></a><a name="p181271653114320"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p14599814153"><a name="p14599814153"></a><a name="p14599814153"></a>torch.nn.ParameterDict.values</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p16599101171510"><a name="p16599101171510"></a><a name="p16599101171510"></a>是</p>
</td>
</tr>
<tr id="row2709175891213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p91271531430"><a name="p91271531430"></a><a name="p91271531430"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p9599121131515"><a name="p9599121131515"></a><a name="p9599121131515"></a>torch.nn.Conv1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p859917112157"><a name="p859917112157"></a><a name="p859917112157"></a>是</p>
</td>
</tr>
<tr id="row16709205871211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p141276533437"><a name="p141276533437"></a><a name="p141276533437"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p185997111158"><a name="p185997111158"></a><a name="p185997111158"></a>torch.nn.Conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p17599514157"><a name="p17599514157"></a><a name="p17599514157"></a>是</p>
</td>
</tr>
<tr id="row270925814121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111281053164313"><a name="p111281053164313"></a><a name="p111281053164313"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p959911141519"><a name="p959911141519"></a><a name="p959911141519"></a>torch.nn.Conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p759912115154"><a name="p759912115154"></a><a name="p759912115154"></a>是</p>
</td>
</tr>
<tr id="row470965801214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p91281653134312"><a name="p91281653134312"></a><a name="p91281653134312"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1959971161512"><a name="p1959971161512"></a><a name="p1959971161512"></a>torch.nn.ConvTranspose1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p65991110155"><a name="p65991110155"></a><a name="p65991110155"></a>否</p>
</td>
</tr>
<tr id="row1709145831212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p3128185304320"><a name="p3128185304320"></a><a name="p3128185304320"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p259931131511"><a name="p259931131511"></a><a name="p259931131511"></a>torch.nn.ConvTranspose2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p195998119156"><a name="p195998119156"></a><a name="p195998119156"></a>是</p>
</td>
</tr>
<tr id="row12709658121213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p5128125319434"><a name="p5128125319434"></a><a name="p5128125319434"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1959920113155"><a name="p1959920113155"></a><a name="p1959920113155"></a>torch.nn.ConvTranspose3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p195996181517"><a name="p195996181517"></a><a name="p195996181517"></a>否</p>
</td>
</tr>
<tr id="row1370910584126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p71281353164320"><a name="p71281353164320"></a><a name="p71281353164320"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p12599191111514"><a name="p12599191111514"></a><a name="p12599191111514"></a>torch.nn.Unfold</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p35991515153"><a name="p35991515153"></a><a name="p35991515153"></a>否</p>
</td>
</tr>
<tr id="row670975811212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p512811538431"><a name="p512811538431"></a><a name="p512811538431"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1559911114150"><a name="p1559911114150"></a><a name="p1559911114150"></a>torch.nn.Fold</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1459915117151"><a name="p1459915117151"></a><a name="p1459915117151"></a>是</p>
</td>
</tr>
<tr id="row770925815121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p2128195324310"><a name="p2128195324310"></a><a name="p2128195324310"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p15599410156"><a name="p15599410156"></a><a name="p15599410156"></a>torch.nn.MaxPool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p7599131181511"><a name="p7599131181511"></a><a name="p7599131181511"></a>是</p>
</td>
</tr>
<tr id="row16709758121219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p15128453164319"><a name="p15128453164319"></a><a name="p15128453164319"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p260015121517"><a name="p260015121517"></a><a name="p260015121517"></a>torch.nn.MaxPool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p5600611159"><a name="p5600611159"></a><a name="p5600611159"></a>是</p>
</td>
</tr>
<tr id="row10709858191213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9128175313430"><a name="p9128175313430"></a><a name="p9128175313430"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p360015115159"><a name="p360015115159"></a><a name="p360015115159"></a>torch.nn.MaxPool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p4600212159"><a name="p4600212159"></a><a name="p4600212159"></a>是</p>
</td>
</tr>
<tr id="row87091258121218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1712855315438"><a name="p1712855315438"></a><a name="p1712855315438"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p760018191514"><a name="p760018191514"></a><a name="p760018191514"></a>torch.nn.MaxUnpool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1460012118157"><a name="p1460012118157"></a><a name="p1460012118157"></a>否</p>
</td>
</tr>
<tr id="row270945891211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p13128165384318"><a name="p13128165384318"></a><a name="p13128165384318"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p17600191111510"><a name="p17600191111510"></a><a name="p17600191111510"></a>torch.nn.MaxUnpool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p935153282615"><a name="p935153282615"></a><a name="p935153282615"></a>否</p>
</td>
</tr>
<tr id="row1770915813123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p812835394314"><a name="p812835394314"></a><a name="p812835394314"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p26001315158"><a name="p26001315158"></a><a name="p26001315158"></a>torch.nn.MaxUnpool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p2600617159"><a name="p2600617159"></a><a name="p2600617159"></a>否</p>
</td>
</tr>
<tr id="row1670905812123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p512875316433"><a name="p512875316433"></a><a name="p512875316433"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1160016110156"><a name="p1160016110156"></a><a name="p1160016110156"></a>torch.nn.AvgPool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1560017113153"><a name="p1560017113153"></a><a name="p1560017113153"></a>是</p>
</td>
</tr>
<tr id="row371035841218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1128195319432"><a name="p1128195319432"></a><a name="p1128195319432"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p360014171515"><a name="p360014171515"></a><a name="p360014171515"></a>torch.nn.AvgPool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1560011131512"><a name="p1560011131512"></a><a name="p1560011131512"></a>是</p>
</td>
</tr>
<tr id="row1871015587122"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p41291531438"><a name="p41291531438"></a><a name="p41291531438"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1760021201512"><a name="p1760021201512"></a><a name="p1760021201512"></a>torch.nn.AvgPool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p860019181515"><a name="p860019181515"></a><a name="p860019181515"></a>是</p>
</td>
</tr>
<tr id="row187109589121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p19129195310433"><a name="p19129195310433"></a><a name="p19129195310433"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p206008131511"><a name="p206008131511"></a><a name="p206008131511"></a>torch.nn.FractionalMaxPool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p460013131512"><a name="p460013131512"></a><a name="p460013131512"></a>否</p>
</td>
</tr>
<tr id="row16710145813124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p121299533432"><a name="p121299533432"></a><a name="p121299533432"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p136001812155"><a name="p136001812155"></a><a name="p136001812155"></a>torch.nn.LPPool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p9986518111616"><a name="p9986518111616"></a><a name="p9986518111616"></a>是</p>
</td>
</tr>
<tr id="row1071012587121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111294530439"><a name="p111294530439"></a><a name="p111294530439"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p166001117159"><a name="p166001117159"></a><a name="p166001117159"></a>torch.nn.LPPool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p560019119152"><a name="p560019119152"></a><a name="p560019119152"></a>是</p>
</td>
</tr>
<tr id="row671035861215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p15129145394318"><a name="p15129145394318"></a><a name="p15129145394318"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1060015115153"><a name="p1060015115153"></a><a name="p1060015115153"></a>torch.nn.AdaptiveMaxPool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p76000151516"><a name="p76000151516"></a><a name="p76000151516"></a>否</p>
</td>
</tr>
<tr id="row271005819123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p61291953174316"><a name="p61291953174316"></a><a name="p61291953174316"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1760015161515"><a name="p1760015161515"></a><a name="p1760015161515"></a>torch.nn.AdaptiveMaxPool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1660017111519"><a name="p1660017111519"></a><a name="p1660017111519"></a>否</p>
</td>
</tr>
<tr id="row15710165831215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p51292535432"><a name="p51292535432"></a><a name="p51292535432"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1360110115151"><a name="p1360110115151"></a><a name="p1360110115151"></a>torch.nn.AdaptiveMaxPool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136011213157"><a name="p136011213157"></a><a name="p136011213157"></a>否</p>
</td>
</tr>
<tr id="row1071075841210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p61291853194312"><a name="p61291853194312"></a><a name="p61291853194312"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p9601515152"><a name="p9601515152"></a><a name="p9601515152"></a>torch.nn.AdaptiveAvgPool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136013151517"><a name="p136013151517"></a><a name="p136013151517"></a>是</p>
</td>
</tr>
<tr id="row971075831218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p51293536433"><a name="p51293536433"></a><a name="p51293536433"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1960117116152"><a name="p1960117116152"></a><a name="p1960117116152"></a>torch.nn.AdaptiveAvgPool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1360151101517"><a name="p1360151101517"></a><a name="p1360151101517"></a>是</p>
</td>
</tr>
<tr id="row10710115810128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p31297535439"><a name="p31297535439"></a><a name="p31297535439"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p20601161161520"><a name="p20601161161520"></a><a name="p20601161161520"></a>torch.nn.AdaptiveAvgPool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p18601314153"><a name="p18601314153"></a><a name="p18601314153"></a>否</p>
</td>
</tr>
<tr id="row4710258171220"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p151291853194314"><a name="p151291853194314"></a><a name="p151291853194314"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1460113111153"><a name="p1460113111153"></a><a name="p1460113111153"></a>torch.nn.ReflectionPad1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p176011217151"><a name="p176011217151"></a><a name="p176011217151"></a>否</p>
</td>
</tr>
<tr id="row1371095814124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p712965312438"><a name="p712965312438"></a><a name="p712965312438"></a>84</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p760113181511"><a name="p760113181511"></a><a name="p760113181511"></a>torch.nn.ReflectionPad2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p4601201141519"><a name="p4601201141519"></a><a name="p4601201141519"></a>否</p>
</td>
</tr>
<tr id="row271085831215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p191291653154315"><a name="p191291653154315"></a><a name="p191291653154315"></a>85</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p0601916158"><a name="p0601916158"></a><a name="p0601916158"></a>torch.nn.ReplicationPad1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p156011316150"><a name="p156011316150"></a><a name="p156011316150"></a>否</p>
</td>
</tr>
<tr id="row771017588125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p512918532435"><a name="p512918532435"></a><a name="p512918532435"></a>86</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p36011419159"><a name="p36011419159"></a><a name="p36011419159"></a>torch.nn.ReplicationPad2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p46011819152"><a name="p46011819152"></a><a name="p46011819152"></a>否</p>
</td>
</tr>
<tr id="row87107585120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p19129135314436"><a name="p19129135314436"></a><a name="p19129135314436"></a>87</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p146011121519"><a name="p146011121519"></a><a name="p146011121519"></a>torch.nn.ReplicationPad3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p106011141512"><a name="p106011141512"></a><a name="p106011141512"></a>否</p>
</td>
</tr>
<tr id="row19710458181218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p21301053174311"><a name="p21301053174311"></a><a name="p21301053174311"></a>88</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p146013120156"><a name="p146013120156"></a><a name="p146013120156"></a>torch.nn.ZeroPad2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p860111131519"><a name="p860111131519"></a><a name="p860111131519"></a>是</p>
</td>
</tr>
<tr id="row0711205881212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11130145314432"><a name="p11130145314432"></a><a name="p11130145314432"></a>89</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1860191121512"><a name="p1860191121512"></a><a name="p1860191121512"></a>torch.nn.ConstantPad1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p660113119156"><a name="p660113119156"></a><a name="p660113119156"></a>是</p>
</td>
</tr>
<tr id="row471114581121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p513017539439"><a name="p513017539439"></a><a name="p513017539439"></a>90</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p15601518158"><a name="p15601518158"></a><a name="p15601518158"></a>torch.nn.ConstantPad2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p166011817158"><a name="p166011817158"></a><a name="p166011817158"></a>是</p>
</td>
</tr>
<tr id="row127111458141213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p171309530437"><a name="p171309530437"></a><a name="p171309530437"></a>91</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p56011511156"><a name="p56011511156"></a><a name="p56011511156"></a>torch.nn.ConstantPad3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1960121181518"><a name="p1960121181518"></a><a name="p1960121181518"></a>是</p>
</td>
</tr>
<tr id="row27114589122"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1213025364319"><a name="p1213025364319"></a><a name="p1213025364319"></a>92</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1460214117157"><a name="p1460214117157"></a><a name="p1460214117157"></a>torch.nn.ELU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p46024118156"><a name="p46024118156"></a><a name="p46024118156"></a>是</p>
</td>
</tr>
<tr id="row77111058131213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p413065364319"><a name="p413065364319"></a><a name="p413065364319"></a>93</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p660271181513"><a name="p660271181513"></a><a name="p660271181513"></a>torch.nn.Hardshrink</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p460211114151"><a name="p460211114151"></a><a name="p460211114151"></a>是</p>
</td>
</tr>
<tr id="row171145811128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p0130053164310"><a name="p0130053164310"></a><a name="p0130053164310"></a>94</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p18602913156"><a name="p18602913156"></a><a name="p18602913156"></a>torch.nn.Hardtanh</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p260218181512"><a name="p260218181512"></a><a name="p260218181512"></a>是</p>
</td>
</tr>
<tr id="row1171112585128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p01301053144315"><a name="p01301053144315"></a><a name="p01301053144315"></a>95</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p060217115156"><a name="p060217115156"></a><a name="p060217115156"></a>torch.nn.LeakyReLU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p156024101518"><a name="p156024101518"></a><a name="p156024101518"></a>是</p>
</td>
</tr>
<tr id="row7711558151217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p81301953174311"><a name="p81301953174311"></a><a name="p81301953174311"></a>96</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16602018151"><a name="p16602018151"></a><a name="p16602018151"></a>torch.nn.LogSigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1660221141515"><a name="p1660221141515"></a><a name="p1660221141515"></a>是</p>
</td>
</tr>
<tr id="row37111858121212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1513085394318"><a name="p1513085394318"></a><a name="p1513085394318"></a>97</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p2602191141520"><a name="p2602191141520"></a><a name="p2602191141520"></a>torch.nn.MultiheadAttention</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p126025116153"><a name="p126025116153"></a><a name="p126025116153"></a>否</p>
</td>
</tr>
<tr id="row1171185813129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9130185374311"><a name="p9130185374311"></a><a name="p9130185374311"></a>98</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p8602171131518"><a name="p8602171131518"></a><a name="p8602171131518"></a>torch.nn.PReLU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p106023111159"><a name="p106023111159"></a><a name="p106023111159"></a>否</p>
</td>
</tr>
<tr id="row371145831217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p121312053104320"><a name="p121312053104320"></a><a name="p121312053104320"></a>99</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p7602151161513"><a name="p7602151161513"></a><a name="p7602151161513"></a>torch.nn.ReLU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1260201121517"><a name="p1260201121517"></a><a name="p1260201121517"></a>是</p>
</td>
</tr>
<tr id="row167112583124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11341753174316"><a name="p11341753174316"></a><a name="p11341753174316"></a>100</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p660217116158"><a name="p660217116158"></a><a name="p660217116158"></a>torch.nn.ReLU6</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p46027171512"><a name="p46027171512"></a><a name="p46027171512"></a>是</p>
</td>
</tr>
<tr id="row1871135818121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1413405316438"><a name="p1413405316438"></a><a name="p1413405316438"></a>101</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116022117159"><a name="p116022117159"></a><a name="p116022117159"></a>torch.nn.RReLU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p26025151514"><a name="p26025151514"></a><a name="p26025151514"></a>否</p>
</td>
</tr>
<tr id="row167111858181217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10134115316435"><a name="p10134115316435"></a><a name="p10134115316435"></a>102</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p36023119157"><a name="p36023119157"></a><a name="p36023119157"></a>torch.nn.SELU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p860213161511"><a name="p860213161511"></a><a name="p860213161511"></a>是</p>
</td>
</tr>
<tr id="row1771185811219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p3134135311438"><a name="p3134135311438"></a><a name="p3134135311438"></a>103</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p18602131181513"><a name="p18602131181513"></a><a name="p18602131181513"></a>torch.nn.CELU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p26021214159"><a name="p26021214159"></a><a name="p26021214159"></a>是</p>
</td>
</tr>
<tr id="row177111058121213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p61341253144314"><a name="p61341253144314"></a><a name="p61341253144314"></a>104</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16602815152"><a name="p16602815152"></a><a name="p16602815152"></a>torch.nn.GELU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p176031919154"><a name="p176031919154"></a><a name="p176031919154"></a>是</p>
</td>
</tr>
<tr id="row07111858101218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p713485319434"><a name="p713485319434"></a><a name="p713485319434"></a>105</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p8603201161520"><a name="p8603201161520"></a><a name="p8603201161520"></a>torch.nn.Sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p86031010152"><a name="p86031010152"></a><a name="p86031010152"></a>是</p>
</td>
</tr>
<tr id="row1171225819121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p15134185312439"><a name="p15134185312439"></a><a name="p15134185312439"></a>106</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p6603916158"><a name="p6603916158"></a><a name="p6603916158"></a>torch.nn.Softplus</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p86031171513"><a name="p86031171513"></a><a name="p86031171513"></a>否</p>
</td>
</tr>
<tr id="row3712125815125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p41341353194313"><a name="p41341353194313"></a><a name="p41341353194313"></a>107</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p17603911153"><a name="p17603911153"></a><a name="p17603911153"></a>torch.nn.Softshrink</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p6603151191510"><a name="p6603151191510"></a><a name="p6603151191510"></a>是，SoftShrink场景暂不支持</p>
</td>
</tr>
<tr id="row67121458121211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p313415320431"><a name="p313415320431"></a><a name="p313415320431"></a>108</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p11603017153"><a name="p11603017153"></a><a name="p11603017153"></a>torch.nn.Softsign</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p5603171111520"><a name="p5603171111520"></a><a name="p5603171111520"></a>是</p>
</td>
</tr>
<tr id="row57127589124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p41342053184319"><a name="p41342053184319"></a><a name="p41342053184319"></a>109</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1560319119157"><a name="p1560319119157"></a><a name="p1560319119157"></a>torch.nn.Tanh</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1060315118150"><a name="p1060315118150"></a><a name="p1060315118150"></a>是</p>
</td>
</tr>
<tr id="row371212585125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1413465314319"><a name="p1413465314319"></a><a name="p1413465314319"></a>110</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1460331141516"><a name="p1460331141516"></a><a name="p1460331141516"></a>torch.nn.Tanhshrink</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p116035120159"><a name="p116035120159"></a><a name="p116035120159"></a>是</p>
</td>
</tr>
<tr id="row9712175818128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p713415535432"><a name="p713415535432"></a><a name="p713415535432"></a>111</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p56036117150"><a name="p56036117150"></a><a name="p56036117150"></a>torch.nn.Threshold</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p36038171518"><a name="p36038171518"></a><a name="p36038171518"></a>是</p>
</td>
</tr>
<tr id="row071211588127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1713435314311"><a name="p1713435314311"></a><a name="p1713435314311"></a>112</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p126031212155"><a name="p126031212155"></a><a name="p126031212155"></a>torch.nn.Softmin</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p960320161511"><a name="p960320161511"></a><a name="p960320161511"></a>是</p>
</td>
</tr>
<tr id="row117121758201210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1813515334316"><a name="p1813515334316"></a><a name="p1813515334316"></a>113</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1660314111520"><a name="p1660314111520"></a><a name="p1660314111520"></a>torch.nn.Softmax</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p116031614151"><a name="p116031614151"></a><a name="p116031614151"></a>是</p>
</td>
</tr>
<tr id="row871219584125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p0135135364314"><a name="p0135135364314"></a><a name="p0135135364314"></a>114</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1060318141517"><a name="p1060318141517"></a><a name="p1060318141517"></a>torch.nn.Softmax2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p2603313157"><a name="p2603313157"></a><a name="p2603313157"></a>是</p>
</td>
</tr>
<tr id="row187121583124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111352538438"><a name="p111352538438"></a><a name="p111352538438"></a>115</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p13603171201517"><a name="p13603171201517"></a><a name="p13603171201517"></a>torch.nn.LogSoftmax</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p2603121201512"><a name="p2603121201512"></a><a name="p2603121201512"></a>否</p>
</td>
</tr>
<tr id="row67121858141217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p7135165316439"><a name="p7135165316439"></a><a name="p7135165316439"></a>116</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1860315171515"><a name="p1860315171515"></a><a name="p1860315171515"></a>torch.nn.AdaptiveLogSoftmaxWithLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p11603116159"><a name="p11603116159"></a><a name="p11603116159"></a>否</p>
</td>
</tr>
<tr id="row147121758171218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1413513533433"><a name="p1413513533433"></a><a name="p1413513533433"></a>117</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p12603817150"><a name="p12603817150"></a><a name="p12603817150"></a>torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p11604191171515"><a name="p11604191171515"></a><a name="p11604191171515"></a>否</p>
</td>
</tr>
<tr id="row47121058191212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p131354532435"><a name="p131354532435"></a><a name="p131354532435"></a>118</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116047115150"><a name="p116047115150"></a><a name="p116047115150"></a>torch.nn.AdaptiveLogSoftmaxWithLoss.predict</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1560411181511"><a name="p1560411181511"></a><a name="p1560411181511"></a>否</p>
</td>
</tr>
<tr id="row671265819128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10135115310434"><a name="p10135115310434"></a><a name="p10135115310434"></a>119</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1460419116159"><a name="p1460419116159"></a><a name="p1460419116159"></a>torch.nn.BatchNorm1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p146045115153"><a name="p146045115153"></a><a name="p146045115153"></a>是</p>
</td>
</tr>
<tr id="row14712758101213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9135153184310"><a name="p9135153184310"></a><a name="p9135153184310"></a>120</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p86041816156"><a name="p86041816156"></a><a name="p86041816156"></a>torch.nn.BatchNorm2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136042015153"><a name="p136042015153"></a><a name="p136042015153"></a>是</p>
</td>
</tr>
<tr id="row177121158151220"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p913555310433"><a name="p913555310433"></a><a name="p913555310433"></a>121</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p14604918159"><a name="p14604918159"></a><a name="p14604918159"></a>torch.nn.BatchNorm3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p156047113157"><a name="p156047113157"></a><a name="p156047113157"></a>否</p>
</td>
</tr>
<tr id="row2712205841219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1513515324314"><a name="p1513515324314"></a><a name="p1513515324314"></a>122</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p3604121191513"><a name="p3604121191513"></a><a name="p3604121191513"></a>torch.nn.GroupNorm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p8614171162113"><a name="p8614171162113"></a><a name="p8614171162113"></a>是</p>
</td>
</tr>
<tr id="row147131058131210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10135165304311"><a name="p10135165304311"></a><a name="p10135165304311"></a>123</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p156043115155"><a name="p156043115155"></a><a name="p156043115155"></a>torch.nn.SyncBatchNorm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p760413121513"><a name="p760413121513"></a><a name="p760413121513"></a>否</p>
</td>
</tr>
<tr id="row12713658131212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p41351853134318"><a name="p41351853134318"></a><a name="p41351853134318"></a>124</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p12604151171516"><a name="p12604151171516"></a><a name="p12604151171516"></a>torch.nn.SyncBatchNorm.convert_sync_batchnorm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1760471141517"><a name="p1760471141517"></a><a name="p1760471141517"></a>否</p>
</td>
</tr>
<tr id="row187131058121214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9135145354314"><a name="p9135145354314"></a><a name="p9135145354314"></a>125</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p4604171191515"><a name="p4604171191515"></a><a name="p4604171191515"></a>torch.nn.InstanceNorm1d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p14604211157"><a name="p14604211157"></a><a name="p14604211157"></a>是</p>
</td>
</tr>
<tr id="row3713105819123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1413516531437"><a name="p1413516531437"></a><a name="p1413516531437"></a>126</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p560411115154"><a name="p560411115154"></a><a name="p560411115154"></a>torch.nn.InstanceNorm2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p106045111151"><a name="p106045111151"></a><a name="p106045111151"></a>是</p>
</td>
</tr>
<tr id="row15713125817124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p15136205315435"><a name="p15136205315435"></a><a name="p15136205315435"></a>127</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p96047151511"><a name="p96047151511"></a><a name="p96047151511"></a>torch.nn.InstanceNorm3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p186041113155"><a name="p186041113155"></a><a name="p186041113155"></a>是</p>
</td>
</tr>
<tr id="row371325810126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p813645384319"><a name="p813645384319"></a><a name="p813645384319"></a>128</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p56046113159"><a name="p56046113159"></a><a name="p56046113159"></a>torch.nn.LayerNorm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p36041011159"><a name="p36041011159"></a><a name="p36041011159"></a>是</p>
</td>
</tr>
<tr id="row1671355815125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p713635314435"><a name="p713635314435"></a><a name="p713635314435"></a>129</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p3604915151"><a name="p3604915151"></a><a name="p3604915151"></a>torch.nn.LocalResponseNorm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p4604513155"><a name="p4604513155"></a><a name="p4604513155"></a>是</p>
</td>
</tr>
<tr id="row371315881214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p51361953194313"><a name="p51361953194313"></a><a name="p51361953194313"></a>130</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1060461151518"><a name="p1060461151518"></a><a name="p1060461151518"></a>torch.nn.RNNBase</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p06050191516"><a name="p06050191516"></a><a name="p06050191516"></a>是</p>
</td>
</tr>
<tr id="row671355814123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10136115317431"><a name="p10136115317431"></a><a name="p10136115317431"></a>131</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1960551181510"><a name="p1960551181510"></a><a name="p1960551181510"></a>torch.nn.RNNBase.flatten_parameters</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p860511111151"><a name="p860511111151"></a><a name="p860511111151"></a>是</p>
</td>
</tr>
<tr id="row137131758141211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p19136653104315"><a name="p19136653104315"></a><a name="p19136653104315"></a>132</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p15605191121511"><a name="p15605191121511"></a><a name="p15605191121511"></a>torch.nn.RNN</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p186144514221"><a name="p186144514221"></a><a name="p186144514221"></a>是</p>
</td>
</tr>
<tr id="row6713558101217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1213605384316"><a name="p1213605384316"></a><a name="p1213605384316"></a>133</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16052116159"><a name="p16052116159"></a><a name="p16052116159"></a>torch.nn.LSTM</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p13774122824812"><a name="p13774122824812"></a><a name="p13774122824812"></a>是，DynamicRNN场景暂不支持</p>
</td>
</tr>
<tr id="row571335815120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p21361053164313"><a name="p21361053164313"></a><a name="p21361053164313"></a>134</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p36052012151"><a name="p36052012151"></a><a name="p36052012151"></a>torch.nn.GRU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p166059117151"><a name="p166059117151"></a><a name="p166059117151"></a>是，DynamicGRUV2场景暂不支持</p>
</td>
</tr>
<tr id="row1671345881214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1613610533433"><a name="p1613610533433"></a><a name="p1613610533433"></a>135</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1060514114156"><a name="p1060514114156"></a><a name="p1060514114156"></a>torch.nn.RNNCell</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1834922302211"><a name="p1834922302211"></a><a name="p1834922302211"></a>是</p>
</td>
</tr>
<tr id="row1171311586121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1413610533439"><a name="p1413610533439"></a><a name="p1413610533439"></a>136</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p106059114157"><a name="p106059114157"></a><a name="p106059114157"></a>torch.nn.LSTMCell</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p86058141517"><a name="p86058141517"></a><a name="p86058141517"></a>是</p>
</td>
</tr>
<tr id="row1771335871214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9136125394316"><a name="p9136125394316"></a><a name="p9136125394316"></a>137</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p96055117159"><a name="p96055117159"></a><a name="p96055117159"></a>torch.nn.GRUCell</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p787614286221"><a name="p787614286221"></a><a name="p787614286221"></a>是</p>
</td>
</tr>
<tr id="row47130585127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111364536431"><a name="p111364536431"></a><a name="p111364536431"></a>138</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p86051913155"><a name="p86051913155"></a><a name="p86051913155"></a>torch.nn.Transformer</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p8605151121514"><a name="p8605151121514"></a><a name="p8605151121514"></a>否</p>
</td>
</tr>
<tr id="row8713558161217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p113675313435"><a name="p113675313435"></a><a name="p113675313435"></a>139</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p176054111511"><a name="p176054111511"></a><a name="p176054111511"></a>torch.nn.Transformer.forward</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p96056118154"><a name="p96056118154"></a><a name="p96056118154"></a>否</p>
</td>
</tr>
<tr id="row97132583120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111366536430"><a name="p111366536430"></a><a name="p111366536430"></a>140</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1560512141510"><a name="p1560512141510"></a><a name="p1560512141510"></a>torch.nn.Transformer.generate_square_subsequent_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1760511151510"><a name="p1760511151510"></a><a name="p1760511151510"></a>否</p>
</td>
</tr>
<tr id="row147141158141213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1713616534435"><a name="p1713616534435"></a><a name="p1713616534435"></a>141</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p26051812157"><a name="p26051812157"></a><a name="p26051812157"></a>torch.nn.TransformerEncoder</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1172765612219"><a name="p1172765612219"></a><a name="p1172765612219"></a>是</p>
</td>
</tr>
<tr id="row10714185817126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p0136453184315"><a name="p0136453184315"></a><a name="p0136453184315"></a>142</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p860517111151"><a name="p860517111151"></a><a name="p860517111151"></a>torch.nn.TransformerEncoder.forward</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p3320658182213"><a name="p3320658182213"></a><a name="p3320658182213"></a>是</p>
</td>
</tr>
<tr id="row19714858181217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p16136155315433"><a name="p16136155315433"></a><a name="p16136155315433"></a>143</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1360510113151"><a name="p1360510113151"></a><a name="p1360510113151"></a>torch.nn.TransformerDecoder</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p36067161514"><a name="p36067161514"></a><a name="p36067161514"></a>否</p>
</td>
</tr>
<tr id="row4714115881212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p813611538437"><a name="p813611538437"></a><a name="p813611538437"></a>144</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p146063114151"><a name="p146063114151"></a><a name="p146063114151"></a>torch.nn.TransformerDecoder.forward</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1260613110156"><a name="p1260613110156"></a><a name="p1260613110156"></a>否</p>
</td>
</tr>
<tr id="row14714165841214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1013665318436"><a name="p1013665318436"></a><a name="p1013665318436"></a>145</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p560601181517"><a name="p560601181517"></a><a name="p560601181517"></a>torch.nn.TransformerEncoderLayer</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p2606014156"><a name="p2606014156"></a><a name="p2606014156"></a>是</p>
</td>
</tr>
<tr id="row11714958111216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p141365532431"><a name="p141365532431"></a><a name="p141365532431"></a>146</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p260661121517"><a name="p260661121517"></a><a name="p260661121517"></a>torch.nn.TransformerEncoderLayer.forward</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p20606111191510"><a name="p20606111191510"></a><a name="p20606111191510"></a>是</p>
</td>
</tr>
<tr id="row4714145821214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p171371453164317"><a name="p171371453164317"></a><a name="p171371453164317"></a>147</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p96061214155"><a name="p96061214155"></a><a name="p96061214155"></a>torch.nn.TransformerDecoderLayer</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p460613114151"><a name="p460613114151"></a><a name="p460613114151"></a>否</p>
</td>
</tr>
<tr id="row15714105816128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p17137853194313"><a name="p17137853194313"></a><a name="p17137853194313"></a>148</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1760631171511"><a name="p1760631171511"></a><a name="p1760631171511"></a>torch.nn.TransformerDecoderLayer.forward</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1560613131517"><a name="p1560613131517"></a><a name="p1560613131517"></a>否</p>
</td>
</tr>
<tr id="row15714958171216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p413712532436"><a name="p413712532436"></a><a name="p413712532436"></a>149</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p10606191171511"><a name="p10606191171511"></a><a name="p10606191171511"></a>torch.nn.Identity</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p96061716156"><a name="p96061716156"></a><a name="p96061716156"></a>是</p>
</td>
</tr>
<tr id="row471485812127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p15137115354310"><a name="p15137115354310"></a><a name="p15137115354310"></a>150</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1760610181519"><a name="p1760610181519"></a><a name="p1760610181519"></a>torch.nn.Linear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p116067121513"><a name="p116067121513"></a><a name="p116067121513"></a>是</p>
</td>
</tr>
<tr id="row13714155813121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1513713532434"><a name="p1513713532434"></a><a name="p1513713532434"></a>151</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p560612111520"><a name="p560612111520"></a><a name="p560612111520"></a>torch.nn.Bilinear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p126061816152"><a name="p126061816152"></a><a name="p126061816152"></a>是</p>
</td>
</tr>
<tr id="row67141458111216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11371453194315"><a name="p11371453194315"></a><a name="p11371453194315"></a>152</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p11606171191515"><a name="p11606171191515"></a><a name="p11606171191515"></a>torch.nn.Dropout</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p560616110155"><a name="p560616110155"></a><a name="p560616110155"></a>是</p>
</td>
</tr>
<tr id="row77146582129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p16137105374314"><a name="p16137105374314"></a><a name="p16137105374314"></a>153</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p13606161151517"><a name="p13606161151517"></a><a name="p13606161151517"></a>torch.nn.Dropout2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p196066118158"><a name="p196066118158"></a><a name="p196066118158"></a>是</p>
</td>
</tr>
<tr id="row19714195881218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p14137195315432"><a name="p14137195315432"></a><a name="p14137195315432"></a>154</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1606171201516"><a name="p1606171201516"></a><a name="p1606171201516"></a>torch.nn.Dropout3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p17606713159"><a name="p17606713159"></a><a name="p17606713159"></a>是</p>
</td>
</tr>
<tr id="row187146583127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1113715538433"><a name="p1113715538433"></a><a name="p1113715538433"></a>155</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p560611110151"><a name="p560611110151"></a><a name="p560611110151"></a>torch.nn.AlphaDropout</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p176066113156"><a name="p176066113156"></a><a name="p176066113156"></a>是</p>
</td>
</tr>
<tr id="row1071495811212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p191374537437"><a name="p191374537437"></a><a name="p191374537437"></a>156</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p36061715155"><a name="p36061715155"></a><a name="p36061715155"></a>torch.nn.Embedding</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p186070181518"><a name="p186070181518"></a><a name="p186070181518"></a>是</p>
</td>
</tr>
<tr id="row1771435831219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1513745314316"><a name="p1513745314316"></a><a name="p1513745314316"></a>157</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1560712110154"><a name="p1560712110154"></a><a name="p1560712110154"></a>torch.nn.Embedding.from_pretrained</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p13607131191515"><a name="p13607131191515"></a><a name="p13607131191515"></a>是</p>
</td>
</tr>
<tr id="row1715058101212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1213765354314"><a name="p1213765354314"></a><a name="p1213765354314"></a>158</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p26071719157"><a name="p26071719157"></a><a name="p26071719157"></a>torch.nn.EmbeddingBag</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p460713110153"><a name="p460713110153"></a><a name="p460713110153"></a>否</p>
</td>
</tr>
<tr id="row1671515820128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p16137153164319"><a name="p16137153164319"></a><a name="p16137153164319"></a>159</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p060710113154"><a name="p060710113154"></a><a name="p060710113154"></a>torch.nn.EmbeddingBag.from_pretrained</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p186071015156"><a name="p186071015156"></a><a name="p186071015156"></a>否</p>
</td>
</tr>
<tr id="row571595891216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p81371153104312"><a name="p81371153104312"></a><a name="p81371153104312"></a>160</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p56071215156"><a name="p56071215156"></a><a name="p56071215156"></a>torch.nn.CosineSimilarity</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p06074111515"><a name="p06074111515"></a><a name="p06074111515"></a>是</p>
</td>
</tr>
<tr id="row187151589123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p713775364320"><a name="p713775364320"></a><a name="p713775364320"></a>161</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p36078171514"><a name="p36078171514"></a><a name="p36078171514"></a>torch.nn.PairwiseDistance</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p360713119155"><a name="p360713119155"></a><a name="p360713119155"></a>是</p>
</td>
</tr>
<tr id="row12715175816127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p213814532433"><a name="p213814532433"></a><a name="p213814532433"></a>162</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16607912159"><a name="p16607912159"></a><a name="p16607912159"></a>torch.nn.L1Loss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p160719141517"><a name="p160719141517"></a><a name="p160719141517"></a>是</p>
</td>
</tr>
<tr id="row271514581121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9138125364315"><a name="p9138125364315"></a><a name="p9138125364315"></a>163</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p10607219158"><a name="p10607219158"></a><a name="p10607219158"></a>torch.nn.MSELoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p36071219159"><a name="p36071219159"></a><a name="p36071219159"></a>是</p>
</td>
</tr>
<tr id="row871512583127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p013865344318"><a name="p013865344318"></a><a name="p013865344318"></a>164</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p2607912154"><a name="p2607912154"></a><a name="p2607912154"></a>torch.nn.CrossEntropyLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1460711171518"><a name="p1460711171518"></a><a name="p1460711171518"></a>是</p>
</td>
</tr>
<tr id="row2071513588120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p71381053194315"><a name="p71381053194315"></a><a name="p71381053194315"></a>165</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p2607151161510"><a name="p2607151161510"></a><a name="p2607151161510"></a>torch.nn.CTCLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p260721121515"><a name="p260721121515"></a><a name="p260721121515"></a>是</p>
</td>
</tr>
<tr id="row19715358171219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1713865317437"><a name="p1713865317437"></a><a name="p1713865317437"></a>166</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16076115154"><a name="p16076115154"></a><a name="p16076115154"></a>torch.nn.NLLLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p26079191515"><a name="p26079191515"></a><a name="p26079191515"></a>是</p>
</td>
</tr>
<tr id="row177151458121210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11381753104316"><a name="p11381753104316"></a><a name="p11381753104316"></a>167</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p26071410155"><a name="p26071410155"></a><a name="p26071410155"></a>torch.nn.PoissonNLLLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p12607111181517"><a name="p12607111181517"></a><a name="p12607111181517"></a>是</p>
</td>
</tr>
<tr id="row107158582124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p18138105314315"><a name="p18138105314315"></a><a name="p18138105314315"></a>168</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p76076141514"><a name="p76076141514"></a><a name="p76076141514"></a>torch.nn.KLDivLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p46071317159"><a name="p46071317159"></a><a name="p46071317159"></a>是</p>
</td>
</tr>
<tr id="row18715125831218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p16138553104310"><a name="p16138553104310"></a><a name="p16138553104310"></a>169</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p960851131519"><a name="p960851131519"></a><a name="p960851131519"></a>torch.nn.BCELoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p15608111181511"><a name="p15608111181511"></a><a name="p15608111181511"></a>是</p>
</td>
</tr>
<tr id="row371510589120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1138353154311"><a name="p1138353154311"></a><a name="p1138353154311"></a>170</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p76082151515"><a name="p76082151515"></a><a name="p76082151515"></a>torch.nn.BCEWithLogitsLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p146081317159"><a name="p146081317159"></a><a name="p146081317159"></a>是</p>
</td>
</tr>
<tr id="row11715155851210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p7138145319438"><a name="p7138145319438"></a><a name="p7138145319438"></a>171</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p960801151517"><a name="p960801151517"></a><a name="p960801151517"></a>torch.nn.MarginRankingLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p16997184119324"><a name="p16997184119324"></a><a name="p16997184119324"></a>是</p>
</td>
</tr>
<tr id="row971515588121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p313845319435"><a name="p313845319435"></a><a name="p313845319435"></a>172</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p160810115157"><a name="p160810115157"></a><a name="p160810115157"></a>torch.nn.HingeEmbeddingLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1460881111516"><a name="p1460881111516"></a><a name="p1460881111516"></a>是</p>
</td>
</tr>
<tr id="row371515811216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p17138185304311"><a name="p17138185304311"></a><a name="p17138185304311"></a>173</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p06085119155"><a name="p06085119155"></a><a name="p06085119155"></a>torch.nn.MultiLabelMarginLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p8608161101520"><a name="p8608161101520"></a><a name="p8608161101520"></a>否</p>
</td>
</tr>
<tr id="row1671516585121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p21386539437"><a name="p21386539437"></a><a name="p21386539437"></a>174</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p8608121121518"><a name="p8608121121518"></a><a name="p8608121121518"></a>torch.nn.SmoothL1Loss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1560841111517"><a name="p1560841111517"></a><a name="p1560841111517"></a>是</p>
</td>
</tr>
<tr id="row8716175841217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1413855314312"><a name="p1413855314312"></a><a name="p1413855314312"></a>175</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p560820114155"><a name="p560820114155"></a><a name="p560820114155"></a>torch.nn.SoftMarginLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1460821191515"><a name="p1460821191515"></a><a name="p1460821191515"></a>否</p>
</td>
</tr>
<tr id="row15716758181216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1713885354310"><a name="p1713885354310"></a><a name="p1713885354310"></a>176</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p060819111513"><a name="p060819111513"></a><a name="p060819111513"></a>torch.nn.MultiLabelSoftMarginLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p13743239349"><a name="p13743239349"></a><a name="p13743239349"></a>是</p>
</td>
</tr>
<tr id="row67169582125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p4138853124310"><a name="p4138853124310"></a><a name="p4138853124310"></a>177</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p46081131518"><a name="p46081131518"></a><a name="p46081131518"></a>torch.nn.CosineEmbeddingLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p96081012157"><a name="p96081012157"></a><a name="p96081012157"></a>是</p>
</td>
</tr>
<tr id="row3716158111215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p71381553114318"><a name="p71381553114318"></a><a name="p71381553114318"></a>178</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p360801131515"><a name="p360801131515"></a><a name="p360801131515"></a>torch.nn.MultiMarginLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p16608161131511"><a name="p16608161131511"></a><a name="p16608161131511"></a>否</p>
</td>
</tr>
<tr id="row6716165861215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p18138135394310"><a name="p18138135394310"></a><a name="p18138135394310"></a>179</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p8608121151519"><a name="p8608121151519"></a><a name="p8608121151519"></a>torch.nn.TripletMarginLoss</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p7608141151519"><a name="p7608141151519"></a><a name="p7608141151519"></a>是</p>
</td>
</tr>
<tr id="row1071617586126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p151383537433"><a name="p151383537433"></a><a name="p151383537433"></a>180</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1360871161520"><a name="p1360871161520"></a><a name="p1360871161520"></a>torch.nn.PixelShuffle</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1160815171513"><a name="p1160815171513"></a><a name="p1160815171513"></a>是</p>
</td>
</tr>
<tr id="row5716175811220"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p4139135315438"><a name="p4139135315438"></a><a name="p4139135315438"></a>181</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p860813191510"><a name="p860813191510"></a><a name="p860813191510"></a>torch.nn.Upsample</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1560815181510"><a name="p1560815181510"></a><a name="p1560815181510"></a>是</p>
</td>
</tr>
<tr id="row97161458171220"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p17139453104319"><a name="p17139453104319"></a><a name="p17139453104319"></a>182</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p126095117152"><a name="p126095117152"></a><a name="p126095117152"></a>torch.nn.UpsamplingNearest2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p26098118151"><a name="p26098118151"></a><a name="p26098118151"></a>是</p>
</td>
</tr>
<tr id="row771614587129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p913995310436"><a name="p913995310436"></a><a name="p913995310436"></a>183</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p9609171101514"><a name="p9609171101514"></a><a name="p9609171101514"></a>torch.nn.UpsamplingBilinear2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p146093141517"><a name="p146093141517"></a><a name="p146093141517"></a>是</p>
</td>
</tr>
<tr id="row8716205817129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p13139195334311"><a name="p13139195334311"></a><a name="p13139195334311"></a>184</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116091219155"><a name="p116091219155"></a><a name="p116091219155"></a>torch.nn.DataParallel</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p15609211151"><a name="p15609211151"></a><a name="p15609211151"></a>否</p>
</td>
</tr>
<tr id="row47161158141218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p313975304319"><a name="p313975304319"></a><a name="p313975304319"></a>185</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p15609111181513"><a name="p15609111181513"></a><a name="p15609111181513"></a>torch.nn.parallel.DistributedDataParallel</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p156091213158"><a name="p156091213158"></a><a name="p156091213158"></a>是</p>
</td>
</tr>
<tr id="row1071615891214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1013915324316"><a name="p1013915324316"></a><a name="p1013915324316"></a>186</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p9609131141512"><a name="p9609131141512"></a><a name="p9609131141512"></a>torch.nn.parallel.DistributedDataParallel.no_sync</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1460971151517"><a name="p1460971151517"></a><a name="p1460971151517"></a>是</p>
</td>
</tr>
<tr id="row14716105815127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p913910539432"><a name="p913910539432"></a><a name="p913910539432"></a>187</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p126091019159"><a name="p126091019159"></a><a name="p126091019159"></a>torch.nn.utils.clip_grad_norm_</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p560914118159"><a name="p560914118159"></a><a name="p560914118159"></a>否</p>
</td>
</tr>
<tr id="row6716145818120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1413945319438"><a name="p1413945319438"></a><a name="p1413945319438"></a>188</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1760916101511"><a name="p1760916101511"></a><a name="p1760916101511"></a>torch.nn.utils.clip_grad_value_</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p06091118157"><a name="p06091118157"></a><a name="p06091118157"></a>否</p>
</td>
</tr>
<tr id="row19716105817125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p131391153164315"><a name="p131391153164315"></a><a name="p131391153164315"></a>189</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p960918161511"><a name="p960918161511"></a><a name="p960918161511"></a>torch.nn.utils.parameters_to_vector</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1560910111157"><a name="p1560910111157"></a><a name="p1560910111157"></a>是</p>
</td>
</tr>
<tr id="row2071615581128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p213985320431"><a name="p213985320431"></a><a name="p213985320431"></a>190</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p460918141518"><a name="p460918141518"></a><a name="p460918141518"></a>torch.nn.utils.vector_to_parameters</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p6609101121518"><a name="p6609101121518"></a><a name="p6609101121518"></a>是</p>
</td>
</tr>
<tr id="row10716135810126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1913925314318"><a name="p1913925314318"></a><a name="p1913925314318"></a>197</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116091714150"><a name="p116091714150"></a><a name="p116091714150"></a>torch.nn.utils.prune.PruningContainer</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p13609101181519"><a name="p13609101181519"></a><a name="p13609101181519"></a>是</p>
</td>
</tr>
<tr id="row371605811128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1613914535432"><a name="p1613914535432"></a><a name="p1613914535432"></a>198</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p7609111131516"><a name="p7609111131516"></a><a name="p7609111131516"></a>torch.nn.utils.prune.PruningContainer.add_pruning_method</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136095181520"><a name="p136095181520"></a><a name="p136095181520"></a>否</p>
</td>
</tr>
<tr id="row1071719583123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p61398532430"><a name="p61398532430"></a><a name="p61398532430"></a>199</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p160911111517"><a name="p160911111517"></a><a name="p160911111517"></a>torch.nn.utils.prune.PruningContainer.apply</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p15609214155"><a name="p15609214155"></a><a name="p15609214155"></a>是</p>
</td>
</tr>
<tr id="row1717185811212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p31391553164318"><a name="p31391553164318"></a><a name="p31391553164318"></a>200</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1360911118156"><a name="p1360911118156"></a><a name="p1360911118156"></a>torch.nn.utils.prune.PruningContainer.apply_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1860916112157"><a name="p1860916112157"></a><a name="p1860916112157"></a>否</p>
</td>
</tr>
<tr id="row12717195817127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1813919530435"><a name="p1813919530435"></a><a name="p1813919530435"></a>201</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p11610171151514"><a name="p11610171151514"></a><a name="p11610171151514"></a>torch.nn.utils.prune.PruningContainer.compute_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p10610121161510"><a name="p10610121161510"></a><a name="p10610121161510"></a>是</p>
</td>
</tr>
<tr id="row197171658181216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p17140253124315"><a name="p17140253124315"></a><a name="p17140253124315"></a>202</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p261010112155"><a name="p261010112155"></a><a name="p261010112155"></a>torch.nn.utils.prune.PruningContainer.prune</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p76102015157"><a name="p76102015157"></a><a name="p76102015157"></a>是</p>
</td>
</tr>
<tr id="row371705813120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p414095384312"><a name="p414095384312"></a><a name="p414095384312"></a>203</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p146101419155"><a name="p146101419155"></a><a name="p146101419155"></a>torch.nn.utils.prune.PruningContainer.remove</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1761014120157"><a name="p1761014120157"></a><a name="p1761014120157"></a>否</p>
</td>
</tr>
<tr id="row771719581125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p314015319432"><a name="p314015319432"></a><a name="p314015319432"></a>204</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p17610810156"><a name="p17610810156"></a><a name="p17610810156"></a>torch.nn.utils.prune.Identity</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p96101101510"><a name="p96101101510"></a><a name="p96101101510"></a>是</p>
</td>
</tr>
<tr id="row20717658121215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11140195374314"><a name="p11140195374314"></a><a name="p11140195374314"></a>205</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p146107101514"><a name="p146107101514"></a><a name="p146107101514"></a>torch.nn.utils.prune.Identity.apply</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p96103191514"><a name="p96103191514"></a><a name="p96103191514"></a>是</p>
</td>
</tr>
<tr id="row671735861210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p8140653104310"><a name="p8140653104310"></a><a name="p8140653104310"></a>206</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p161014113158"><a name="p161014113158"></a><a name="p161014113158"></a>torch.nn.utils.prune.Identity.apply_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p861021131513"><a name="p861021131513"></a><a name="p861021131513"></a>否</p>
</td>
</tr>
<tr id="row87172058141218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10140553154313"><a name="p10140553154313"></a><a name="p10140553154313"></a>207</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1561016151511"><a name="p1561016151511"></a><a name="p1561016151511"></a>torch.nn.utils.prune.Identity.prune</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p12610014159"><a name="p12610014159"></a><a name="p12610014159"></a>是</p>
</td>
</tr>
<tr id="row107179586125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p214095384316"><a name="p214095384316"></a><a name="p214095384316"></a>208</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p166101151520"><a name="p166101151520"></a><a name="p166101151520"></a>torch.nn.utils.prune.Identity.remove</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p66101510156"><a name="p66101510156"></a><a name="p66101510156"></a>否</p>
</td>
</tr>
<tr id="row571715582128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p141408535439"><a name="p141408535439"></a><a name="p141408535439"></a>209</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p46109171512"><a name="p46109171512"></a><a name="p46109171512"></a>torch.nn.utils.prune.RandomUnstructured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p861014111510"><a name="p861014111510"></a><a name="p861014111510"></a>是</p>
</td>
</tr>
<tr id="row67171588121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p31402539437"><a name="p31402539437"></a><a name="p31402539437"></a>210</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p6610214151"><a name="p6610214151"></a><a name="p6610214151"></a>torch.nn.utils.prune.RandomUnstructured.apply</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p15610101201518"><a name="p15610101201518"></a><a name="p15610101201518"></a>否</p>
</td>
</tr>
<tr id="row10717458191215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1314010532437"><a name="p1314010532437"></a><a name="p1314010532437"></a>211</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1610131131510"><a name="p1610131131510"></a><a name="p1610131131510"></a>torch.nn.utils.prune.RandomUnstructured.apply_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p96105115153"><a name="p96105115153"></a><a name="p96105115153"></a>否</p>
</td>
</tr>
<tr id="row127171158141211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p414012534437"><a name="p414012534437"></a><a name="p414012534437"></a>212</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p861081111516"><a name="p861081111516"></a><a name="p861081111516"></a>torch.nn.utils.prune.RandomUnstructured.prune</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136104131512"><a name="p136104131512"></a><a name="p136104131512"></a>否</p>
</td>
</tr>
<tr id="row9717058121214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p13140195394318"><a name="p13140195394318"></a><a name="p13140195394318"></a>213</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p561013171516"><a name="p561013171516"></a><a name="p561013171516"></a>torch.nn.utils.prune.RandomUnstructured.remove</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p126103161516"><a name="p126103161516"></a><a name="p126103161516"></a>否</p>
</td>
</tr>
<tr id="row1771775851220"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p914075354312"><a name="p914075354312"></a><a name="p914075354312"></a>214</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p661121151517"><a name="p661121151517"></a><a name="p661121151517"></a>torch.nn.utils.prune.L1Unstructured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p6611517154"><a name="p6611517154"></a><a name="p6611517154"></a>是</p>
</td>
</tr>
<tr id="row197177587126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p15140205384312"><a name="p15140205384312"></a><a name="p15140205384312"></a>215</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p66111113152"><a name="p66111113152"></a><a name="p66111113152"></a>torch.nn.utils.prune.L1Unstructured.apply</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1761114121518"><a name="p1761114121518"></a><a name="p1761114121518"></a>否</p>
</td>
</tr>
<tr id="row2718125811121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p314195344312"><a name="p314195344312"></a><a name="p314195344312"></a>216</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p4611131171519"><a name="p4611131171519"></a><a name="p4611131171519"></a>torch.nn.utils.prune.L1Unstructured.apply_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p061111114154"><a name="p061111114154"></a><a name="p061111114154"></a>否</p>
</td>
</tr>
<tr id="row37181558171216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1814185311434"><a name="p1814185311434"></a><a name="p1814185311434"></a>217</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p86111114153"><a name="p86111114153"></a><a name="p86111114153"></a>torch.nn.utils.prune.L1Unstructured.prune</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1361119112158"><a name="p1361119112158"></a><a name="p1361119112158"></a>否</p>
</td>
</tr>
<tr id="row187181158191220"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p19141753114318"><a name="p19141753114318"></a><a name="p19141753114318"></a>218</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1161116111512"><a name="p1161116111512"></a><a name="p1161116111512"></a>torch.nn.utils.prune.L1Unstructured.remove</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136117117159"><a name="p136117117159"></a><a name="p136117117159"></a>否</p>
</td>
</tr>
<tr id="row3718125881211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p61418530431"><a name="p61418530431"></a><a name="p61418530431"></a>219</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p26112016159"><a name="p26112016159"></a><a name="p26112016159"></a>torch.nn.utils.prune.RandomStructured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p461116141513"><a name="p461116141513"></a><a name="p461116141513"></a>是</p>
</td>
</tr>
<tr id="row197182586122"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p5141155344310"><a name="p5141155344310"></a><a name="p5141155344310"></a>220</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116115117156"><a name="p116115117156"></a><a name="p116115117156"></a>torch.nn.utils.prune.RandomStructured.apply</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p12611191191517"><a name="p12611191191517"></a><a name="p12611191191517"></a>是</p>
</td>
</tr>
<tr id="row371825821210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p514155314315"><a name="p514155314315"></a><a name="p514155314315"></a>221</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16115118159"><a name="p16115118159"></a><a name="p16115118159"></a>torch.nn.utils.prune.RandomStructured.apply_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p16611121111516"><a name="p16611121111516"></a><a name="p16611121111516"></a>否</p>
</td>
</tr>
<tr id="row8718175881210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p91414536433"><a name="p91414536433"></a><a name="p91414536433"></a>222</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1061114117152"><a name="p1061114117152"></a><a name="p1061114117152"></a>torch.nn.utils.prune.RandomStructured.compute_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1261171141514"><a name="p1261171141514"></a><a name="p1261171141514"></a>是</p>
</td>
</tr>
<tr id="row4718758141210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p6141145374311"><a name="p6141145374311"></a><a name="p6141145374311"></a>223</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1061112112152"><a name="p1061112112152"></a><a name="p1061112112152"></a>torch.nn.utils.prune.RandomStructured.prune</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p961131121517"><a name="p961131121517"></a><a name="p961131121517"></a>是</p>
</td>
</tr>
<tr id="row12718145814126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1414111535437"><a name="p1414111535437"></a><a name="p1414111535437"></a>224</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1261112120154"><a name="p1261112120154"></a><a name="p1261112120154"></a>torch.nn.utils.prune.RandomStructured.remove</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p156115141517"><a name="p156115141517"></a><a name="p156115141517"></a>否</p>
</td>
</tr>
<tr id="row57181158111216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p6141175316431"><a name="p6141175316431"></a><a name="p6141175316431"></a>225</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p761119118153"><a name="p761119118153"></a><a name="p761119118153"></a>torch.nn.utils.prune.LnStructured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1761120113156"><a name="p1761120113156"></a><a name="p1761120113156"></a>是</p>
</td>
</tr>
<tr id="row137181658171217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p12141125310434"><a name="p12141125310434"></a><a name="p12141125310434"></a>226</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p86124191519"><a name="p86124191519"></a><a name="p86124191519"></a>torch.nn.utils.prune.LnStructured.apply</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p6612117158"><a name="p6612117158"></a><a name="p6612117158"></a>否</p>
</td>
</tr>
<tr id="row16718558111215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1214118535433"><a name="p1214118535433"></a><a name="p1214118535433"></a>227</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p176126120152"><a name="p176126120152"></a><a name="p176126120152"></a>torch.nn.utils.prune.LnStructured.apply_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p14612181181516"><a name="p14612181181516"></a><a name="p14612181181516"></a>否</p>
</td>
</tr>
<tr id="row271875891218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111411853184319"><a name="p111411853184319"></a><a name="p111411853184319"></a>228</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p961271111510"><a name="p961271111510"></a><a name="p961271111510"></a>torch.nn.utils.prune.LnStructured.compute_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1612111159"><a name="p1612111159"></a><a name="p1612111159"></a>否</p>
</td>
</tr>
<tr id="row5718358151219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p16141105354320"><a name="p16141105354320"></a><a name="p16141105354320"></a>229</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p561241111512"><a name="p561241111512"></a><a name="p561241111512"></a>torch.nn.utils.prune.LnStructured.prune</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p176126115154"><a name="p176126115154"></a><a name="p176126115154"></a>否</p>
</td>
</tr>
<tr id="row471835819125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10141115364312"><a name="p10141115364312"></a><a name="p10141115364312"></a>230</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1961211119154"><a name="p1961211119154"></a><a name="p1961211119154"></a>torch.nn.utils.prune.LnStructured.remove</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p161215120154"><a name="p161215120154"></a><a name="p161215120154"></a>否</p>
</td>
</tr>
<tr id="row20719175871211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p414165317432"><a name="p414165317432"></a><a name="p414165317432"></a>231</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p26124111157"><a name="p26124111157"></a><a name="p26124111157"></a>torch.nn.utils.prune.CustomFromMask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p2061214151513"><a name="p2061214151513"></a><a name="p2061214151513"></a>是</p>
</td>
</tr>
<tr id="row3719105831213"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p131411953164311"><a name="p131411953164311"></a><a name="p131411953164311"></a>232</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1161261161514"><a name="p1161261161514"></a><a name="p1161261161514"></a>torch.nn.utils.prune.CustomFromMask.apply</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p3612131171513"><a name="p3612131171513"></a><a name="p3612131171513"></a>是</p>
</td>
</tr>
<tr id="row17719858131218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p5141155320431"><a name="p5141155320431"></a><a name="p5141155320431"></a>233</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p761219118158"><a name="p761219118158"></a><a name="p761219118158"></a>torch.nn.utils.prune.CustomFromMask.apply_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p261213121512"><a name="p261213121512"></a><a name="p261213121512"></a>否</p>
</td>
</tr>
<tr id="row1071925811127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p14141115310435"><a name="p14141115310435"></a><a name="p14141115310435"></a>234</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p36121813155"><a name="p36121813155"></a><a name="p36121813155"></a>torch.nn.utils.prune.CustomFromMask.prune</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1861215113152"><a name="p1861215113152"></a><a name="p1861215113152"></a>是</p>
</td>
</tr>
<tr id="row187191758181210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p614110535432"><a name="p614110535432"></a><a name="p614110535432"></a>235</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p196121412154"><a name="p196121412154"></a><a name="p196121412154"></a>torch.nn.utils.prune.CustomFromMask.remove</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p461210181518"><a name="p461210181518"></a><a name="p461210181518"></a>否</p>
</td>
</tr>
<tr id="row157191358191216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p814217533437"><a name="p814217533437"></a><a name="p814217533437"></a>236</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p161271111510"><a name="p161271111510"></a><a name="p161271111510"></a>torch.nn.utils.prune.identity</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1161251191517"><a name="p1161251191517"></a><a name="p1161251191517"></a>是</p>
</td>
</tr>
<tr id="row1671955881217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p2142185344313"><a name="p2142185344313"></a><a name="p2142185344313"></a>237</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p36127111514"><a name="p36127111514"></a><a name="p36127111514"></a>torch.nn.utils.prune.random_unstructured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1361215110154"><a name="p1361215110154"></a><a name="p1361215110154"></a>否</p>
</td>
</tr>
<tr id="row47199581125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p514215313432"><a name="p514215313432"></a><a name="p514215313432"></a>238</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p6612613152"><a name="p6612613152"></a><a name="p6612613152"></a>torch.nn.utils.prune.l1_unstructured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p176125161513"><a name="p176125161513"></a><a name="p176125161513"></a>否</p>
</td>
</tr>
<tr id="row10719115814125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10142153184317"><a name="p10142153184317"></a><a name="p10142153184317"></a>239</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p361321141511"><a name="p361321141511"></a><a name="p361321141511"></a>torch.nn.utils.prune.random_structured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p106132161511"><a name="p106132161511"></a><a name="p106132161511"></a>是</p>
</td>
</tr>
<tr id="row4719145831217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p814235315435"><a name="p814235315435"></a><a name="p814235315435"></a>240</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p761313119153"><a name="p761313119153"></a><a name="p761313119153"></a>torch.nn.utils.prune.ln_structured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p8613514159"><a name="p8613514159"></a><a name="p8613514159"></a>否</p>
</td>
</tr>
<tr id="row10719195811124"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1142105304316"><a name="p1142105304316"></a><a name="p1142105304316"></a>241</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1161317110156"><a name="p1161317110156"></a><a name="p1161317110156"></a>torch.nn.utils.prune.global_unstructured</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p761491121519"><a name="p761491121519"></a><a name="p761491121519"></a>否</p>
</td>
</tr>
<tr id="row197191558191217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1414275314312"><a name="p1414275314312"></a><a name="p1414275314312"></a>242</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p461415116151"><a name="p461415116151"></a><a name="p461415116151"></a>torch.nn.utils.prune.custom_from_mask</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1461419131517"><a name="p1461419131517"></a><a name="p1461419131517"></a>是</p>
</td>
</tr>
<tr id="row14719358131214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p41422053204317"><a name="p41422053204317"></a><a name="p41422053204317"></a>243</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16614111111512"><a name="p16614111111512"></a><a name="p16614111111512"></a>torch.nn.utils.prune.remove</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1161417101518"><a name="p1161417101518"></a><a name="p1161417101518"></a>是</p>
</td>
</tr>
<tr id="row1719258171210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1914245314310"><a name="p1914245314310"></a><a name="p1914245314310"></a>244</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p56144141519"><a name="p56144141519"></a><a name="p56144141519"></a>torch.nn.utils.prune.is_pruned</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p961401161513"><a name="p961401161513"></a><a name="p961401161513"></a>是</p>
</td>
</tr>
<tr id="row1971965881215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p14142115310433"><a name="p14142115310433"></a><a name="p14142115310433"></a>245</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1561431171519"><a name="p1561431171519"></a><a name="p1561431171519"></a>torch.nn.utils.weight_norm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1561410118151"><a name="p1561410118151"></a><a name="p1561410118151"></a>是</p>
</td>
</tr>
<tr id="row77198583125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p4142205384310"><a name="p4142205384310"></a><a name="p4142205384310"></a>246</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p10614417158"><a name="p10614417158"></a><a name="p10614417158"></a>torch.nn.utils.remove_weight_norm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p86143191517"><a name="p86143191517"></a><a name="p86143191517"></a>是</p>
</td>
</tr>
<tr id="row107198580126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p41421353114315"><a name="p41421353114315"></a><a name="p41421353114315"></a>247</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p186141914158"><a name="p186141914158"></a><a name="p186141914158"></a>torch.nn.utils.spectral_norm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136144111515"><a name="p136144111515"></a><a name="p136144111515"></a>是</p>
</td>
</tr>
<tr id="row14719165861212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p2014215532435"><a name="p2014215532435"></a><a name="p2014215532435"></a>248</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116147118152"><a name="p116147118152"></a><a name="p116147118152"></a>torch.nn.utils.remove_spectral_norm</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p36141116156"><a name="p36141116156"></a><a name="p36141116156"></a>否</p>
</td>
</tr>
<tr id="row772085817126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p214211531439"><a name="p214211531439"></a><a name="p214211531439"></a>249</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p661419141517"><a name="p661419141517"></a><a name="p661419141517"></a>torch.nn.utils.rnn.PackedSequence</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p13614715150"><a name="p13614715150"></a><a name="p13614715150"></a>是</p>
</td>
</tr>
<tr id="row372019589122"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p014335374311"><a name="p014335374311"></a><a name="p014335374311"></a>250</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p3614151201520"><a name="p3614151201520"></a><a name="p3614151201520"></a>torch.nn.utils.rnn.pack_padded_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p761416171519"><a name="p761416171519"></a><a name="p761416171519"></a>是</p>
</td>
</tr>
<tr id="row772095801218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p191431253144320"><a name="p191431253144320"></a><a name="p191431253144320"></a>251</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p561416115159"><a name="p561416115159"></a><a name="p561416115159"></a>torch.nn.utils.rnn.pad_packed_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p146145121515"><a name="p146145121515"></a><a name="p146145121515"></a>否</p>
</td>
</tr>
<tr id="row1172045811216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1114320535437"><a name="p1114320535437"></a><a name="p1114320535437"></a>252</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p961421121516"><a name="p961421121516"></a><a name="p961421121516"></a>torch.nn.utils.rnn.pad_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p96144121510"><a name="p96144121510"></a><a name="p96144121510"></a>是</p>
</td>
</tr>
<tr id="row1872045861211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p414355334316"><a name="p414355334316"></a><a name="p414355334316"></a>253</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p661411161518"><a name="p661411161518"></a><a name="p661411161518"></a>torch.nn.utils.rnn.pack_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p561481121515"><a name="p561481121515"></a><a name="p561481121515"></a>否</p>
</td>
</tr>
<tr id="row8720195851218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p12143165364315"><a name="p12143165364315"></a><a name="p12143165364315"></a>254</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p361501121516"><a name="p361501121516"></a><a name="p361501121516"></a>torch.nn.Flatten</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p196151116152"><a name="p196151116152"></a><a name="p196151116152"></a>是</p>
</td>
</tr>
<tr id="row172095814121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10143135319438"><a name="p10143135319438"></a><a name="p10143135319438"></a>255</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p126152131516"><a name="p126152131516"></a><a name="p126152131516"></a>torch.quantization.quantize</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p166159121513"><a name="p166159121513"></a><a name="p166159121513"></a>否</p>
</td>
</tr>
<tr id="row87201258101211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p191431853124313"><a name="p191431853124313"></a><a name="p191431853124313"></a>256</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116156113157"><a name="p116156113157"></a><a name="p116156113157"></a>torch.quantization.quantize_dynamic</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p661510110154"><a name="p661510110154"></a><a name="p661510110154"></a>否</p>
</td>
</tr>
<tr id="row187203583129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p14143125384319"><a name="p14143125384319"></a><a name="p14143125384319"></a>257</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p46153191516"><a name="p46153191516"></a><a name="p46153191516"></a>torch.quantization.quantize_qat</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1961513118152"><a name="p1961513118152"></a><a name="p1961513118152"></a>否</p>
</td>
</tr>
<tr id="row0720185810126"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p114310536433"><a name="p114310536433"></a><a name="p114310536433"></a>258</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1461519171519"><a name="p1461519171519"></a><a name="p1461519171519"></a>torch.quantization.prepare</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p661513113159"><a name="p661513113159"></a><a name="p661513113159"></a>否</p>
</td>
</tr>
<tr id="row13720858141216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11432053174319"><a name="p11432053174319"></a><a name="p11432053174319"></a>259</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1861515119153"><a name="p1861515119153"></a><a name="p1861515119153"></a>torch.quantization.prepare_qat</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p126159110151"><a name="p126159110151"></a><a name="p126159110151"></a>否</p>
</td>
</tr>
<tr id="row1972095811214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p121434536433"><a name="p121434536433"></a><a name="p121434536433"></a>260</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p06158112156"><a name="p06158112156"></a><a name="p06158112156"></a>torch.quantization.convert</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p206159171520"><a name="p206159171520"></a><a name="p206159171520"></a>否</p>
</td>
</tr>
<tr id="row1972035881210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p18143653184315"><a name="p18143653184315"></a><a name="p18143653184315"></a>261</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1261517110150"><a name="p1261517110150"></a><a name="p1261517110150"></a>torch.quantization.QConfig</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p206158112154"><a name="p206158112154"></a><a name="p206158112154"></a>否</p>
</td>
</tr>
<tr id="row16720165881214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p17143105316430"><a name="p17143105316430"></a><a name="p17143105316430"></a>262</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p2615151141516"><a name="p2615151141516"></a><a name="p2615151141516"></a>torch.quantization.QConfigDynamic</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1661511131514"><a name="p1661511131514"></a><a name="p1661511131514"></a>否</p>
</td>
</tr>
<tr id="row9720175816120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1714345320432"><a name="p1714345320432"></a><a name="p1714345320432"></a>263</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p561520151519"><a name="p561520151519"></a><a name="p561520151519"></a>torch.quantization.fuse_modules</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1061513191516"><a name="p1061513191516"></a><a name="p1061513191516"></a>否</p>
</td>
</tr>
<tr id="row872065881214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1414313535435"><a name="p1414313535435"></a><a name="p1414313535435"></a>264</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p11615181161516"><a name="p11615181161516"></a><a name="p11615181161516"></a>torch.quantization.QuantStub</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1261515113154"><a name="p1261515113154"></a><a name="p1261515113154"></a>否</p>
</td>
</tr>
<tr id="row12720158201220"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1214385364310"><a name="p1214385364310"></a><a name="p1214385364310"></a>265</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116152017156"><a name="p116152017156"></a><a name="p116152017156"></a>torch.quantization.DeQuantStub</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1861551171510"><a name="p1861551171510"></a><a name="p1861551171510"></a>否</p>
</td>
</tr>
<tr id="row2072095861217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p21431853134310"><a name="p21431853134310"></a><a name="p21431853134310"></a>266</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p176158117158"><a name="p176158117158"></a><a name="p176158117158"></a>torch.quantization.QuantWrapper</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1561561121511"><a name="p1561561121511"></a><a name="p1561561121511"></a>否</p>
</td>
</tr>
<tr id="row572165871211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p8143953104320"><a name="p8143953104320"></a><a name="p8143953104320"></a>267</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p2616511155"><a name="p2616511155"></a><a name="p2616511155"></a>torch.quantization.add_quant_dequant</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p161617114156"><a name="p161617114156"></a><a name="p161617114156"></a>否</p>
</td>
</tr>
<tr id="row15721958201216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11144153154315"><a name="p11144153154315"></a><a name="p11144153154315"></a>268</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p961615119154"><a name="p961615119154"></a><a name="p961615119154"></a>torch.quantization.add_observer_</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136161614152"><a name="p136161614152"></a><a name="p136161614152"></a>否</p>
</td>
</tr>
<tr id="row17211858171214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p16144145394317"><a name="p16144145394317"></a><a name="p16144145394317"></a>269</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p166161117151"><a name="p166161117151"></a><a name="p166161117151"></a>torch.quantization.swap_module</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1361615111153"><a name="p1361615111153"></a><a name="p1361615111153"></a>否</p>
</td>
</tr>
<tr id="row8721145817125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1314419534436"><a name="p1314419534436"></a><a name="p1314419534436"></a>270</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p561618111154"><a name="p561618111154"></a><a name="p561618111154"></a>torch.quantization.propagate_qconfig_</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p18616191181511"><a name="p18616191181511"></a><a name="p18616191181511"></a>否</p>
</td>
</tr>
<tr id="row1172125816123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p4144953174313"><a name="p4144953174313"></a><a name="p4144953174313"></a>271</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p156161811158"><a name="p156161811158"></a><a name="p156161811158"></a>torch.quantization.default_eval_fn</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p5616111111518"><a name="p5616111111518"></a><a name="p5616111111518"></a>否</p>
</td>
</tr>
<tr id="row17211558171214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p13144135374314"><a name="p13144135374314"></a><a name="p13144135374314"></a>272</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1361617119151"><a name="p1361617119151"></a><a name="p1361617119151"></a>torch.quantization.MinMaxObserver</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p9616612150"><a name="p9616612150"></a><a name="p9616612150"></a>否</p>
</td>
</tr>
<tr id="row2721155881217"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p914455344311"><a name="p914455344311"></a><a name="p914455344311"></a>273</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p116162119159"><a name="p116162119159"></a><a name="p116162119159"></a>torch.quantization.MovingAverageMinMaxObserver</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1261615111154"><a name="p1261615111154"></a><a name="p1261615111154"></a>否</p>
</td>
</tr>
<tr id="row1721125811125"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p91447534431"><a name="p91447534431"></a><a name="p91447534431"></a>274</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p56177131517"><a name="p56177131517"></a><a name="p56177131517"></a>torch.quantization.PerChannelMinMaxObserver</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p106171718154"><a name="p106171718154"></a><a name="p106171718154"></a>否</p>
</td>
</tr>
<tr id="row7721135861218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1914415354314"><a name="p1914415354314"></a><a name="p1914415354314"></a>275</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p261761181517"><a name="p261761181517"></a><a name="p261761181517"></a>torch.quantization.MovingAveragePerChannelMinMaxObserver</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p7617161101512"><a name="p7617161101512"></a><a name="p7617161101512"></a>否</p>
</td>
</tr>
<tr id="row13721858201215"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1014418533438"><a name="p1014418533438"></a><a name="p1014418533438"></a>276</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1661710115159"><a name="p1661710115159"></a><a name="p1661710115159"></a>torch.quantization.HistogramObserver</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1761771111513"><a name="p1761771111513"></a><a name="p1761771111513"></a>否</p>
</td>
</tr>
<tr id="row572125813129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p4144753134319"><a name="p4144753134319"></a><a name="p4144753134319"></a>277</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1617611150"><a name="p1617611150"></a><a name="p1617611150"></a>torch.quantization.FakeQuantize</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1961714171517"><a name="p1961714171517"></a><a name="p1961714171517"></a>否</p>
</td>
</tr>
<tr id="row8721125818128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p111441053144315"><a name="p111441053144315"></a><a name="p111441053144315"></a>278</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p36170113150"><a name="p36170113150"></a><a name="p36170113150"></a>torch.quantization.NoopObserver</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p561711115157"><a name="p561711115157"></a><a name="p561711115157"></a>否</p>
</td>
</tr>
<tr id="row1472105811214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p114415318439"><a name="p114415318439"></a><a name="p114415318439"></a>279</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1961791181510"><a name="p1961791181510"></a><a name="p1961791181510"></a>torch.quantization.get_observer_dict</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p196171317157"><a name="p196171317157"></a><a name="p196171317157"></a>否</p>
</td>
</tr>
<tr id="row47211058111216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p21441553194314"><a name="p21441553194314"></a><a name="p21441553194314"></a>280</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p13617416152"><a name="p13617416152"></a><a name="p13617416152"></a>torch.quantization.RecordingObserver</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p14617201161512"><a name="p14617201161512"></a><a name="p14617201161512"></a>否</p>
</td>
</tr>
<tr id="row1972120586127"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p31441653184315"><a name="p31441653184315"></a><a name="p31441653184315"></a>281</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p196171315159"><a name="p196171315159"></a><a name="p196171315159"></a>torch.nn.intrinsic.ConvBn2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p86173116150"><a name="p86173116150"></a><a name="p86173116150"></a>是</p>
</td>
</tr>
<tr id="row0721458191220"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1414495334311"><a name="p1414495334311"></a><a name="p1414495334311"></a>282</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p186176111150"><a name="p186176111150"></a><a name="p186176111150"></a>torch.nn.intrinsic.ConvBnReLU2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1061710111515"><a name="p1061710111515"></a><a name="p1061710111515"></a>是</p>
</td>
</tr>
<tr id="row5721758131214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p71444536436"><a name="p71444536436"></a><a name="p71444536436"></a>283</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p861714191515"><a name="p861714191515"></a><a name="p861714191515"></a>torch.nn.intrinsic.ConvReLU2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p19617131201515"><a name="p19617131201515"></a><a name="p19617131201515"></a>是</p>
</td>
</tr>
<tr id="row17223587123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p314555316431"><a name="p314555316431"></a><a name="p314555316431"></a>284</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p126173119156"><a name="p126173119156"></a><a name="p126173119156"></a>torch.nn.intrinsic.ConvReLU3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p96177131515"><a name="p96177131515"></a><a name="p96177131515"></a>否</p>
</td>
</tr>
<tr id="row272212586120"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p114575334310"><a name="p114575334310"></a><a name="p114575334310"></a>285</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p561811101518"><a name="p561811101518"></a><a name="p561811101518"></a>torch.nn.intrinsic.LinearReLU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p136182120157"><a name="p136182120157"></a><a name="p136182120157"></a>是</p>
</td>
</tr>
<tr id="row672265841219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p414565364315"><a name="p414565364315"></a><a name="p414565364315"></a>286</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p136185119150"><a name="p136185119150"></a><a name="p136185119150"></a>torch.nn.intrinsic.qat.ConvBn2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p6618112158"><a name="p6618112158"></a><a name="p6618112158"></a>否</p>
</td>
</tr>
<tr id="row147221758101211"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p2145653164313"><a name="p2145653164313"></a><a name="p2145653164313"></a>287</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p176185111151"><a name="p176185111151"></a><a name="p176185111151"></a>torch.nn.intrinsic.qat.ConvBnReLU2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p9618314156"><a name="p9618314156"></a><a name="p9618314156"></a>否</p>
</td>
</tr>
<tr id="row18722105841219"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1714555334311"><a name="p1714555334311"></a><a name="p1714555334311"></a>288</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p146182015152"><a name="p146182015152"></a><a name="p146182015152"></a>torch.nn.intrinsic.qat.ConvReLU2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1861871151511"><a name="p1861871151511"></a><a name="p1861871151511"></a>否</p>
</td>
</tr>
<tr id="row1972235815123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p19145253164316"><a name="p19145253164316"></a><a name="p19145253164316"></a>289</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p2061815171511"><a name="p2061815171511"></a><a name="p2061815171511"></a>torch.nn.intrinsic.qat.LinearReLU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p461810118152"><a name="p461810118152"></a><a name="p461810118152"></a>否</p>
</td>
</tr>
<tr id="row137221458161210"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p191451153204318"><a name="p191451153204318"></a><a name="p191451153204318"></a>290</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1861810161511"><a name="p1861810161511"></a><a name="p1861810161511"></a>torch.nn.intrinsic.quantized.ConvReLU2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p361891121516"><a name="p361891121516"></a><a name="p361891121516"></a>否</p>
</td>
</tr>
<tr id="row197228580123"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p51450538438"><a name="p51450538438"></a><a name="p51450538438"></a>291</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p761861191516"><a name="p761861191516"></a><a name="p761861191516"></a>torch.nn.intrinsic.quantized.ConvReLU3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p961816114154"><a name="p961816114154"></a><a name="p961816114154"></a>否</p>
</td>
</tr>
<tr id="row872225818128"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p214513537432"><a name="p214513537432"></a><a name="p214513537432"></a>292</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p561812117155"><a name="p561812117155"></a><a name="p561812117155"></a>torch.nn.intrinsic.quantized.LinearReLU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p186183171514"><a name="p186183171514"></a><a name="p186183171514"></a>否</p>
</td>
</tr>
<tr id="row177221585121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p12145153174314"><a name="p12145153174314"></a><a name="p12145153174314"></a>293</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p56183116151"><a name="p56183116151"></a><a name="p56183116151"></a>torch.nn.qat.Conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p17618216155"><a name="p17618216155"></a><a name="p17618216155"></a>否</p>
</td>
</tr>
<tr id="row8722105811129"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p114585310436"><a name="p114585310436"></a><a name="p114585310436"></a>294</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p106181511151"><a name="p106181511151"></a><a name="p106181511151"></a>torch.nn.qat.Conv2d.from_float</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1261841121520"><a name="p1261841121520"></a><a name="p1261841121520"></a>否</p>
</td>
</tr>
<tr id="row117221358171212"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10145953174316"><a name="p10145953174316"></a><a name="p10145953174316"></a>295</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p56181151516"><a name="p56181151516"></a><a name="p56181151516"></a>torch.nn.qat.Linear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p561815118154"><a name="p561815118154"></a><a name="p561815118154"></a>否</p>
</td>
</tr>
<tr id="row1272216581122"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p14145125313438"><a name="p14145125313438"></a><a name="p14145125313438"></a>296</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1618141151514"><a name="p1618141151514"></a><a name="p1618141151514"></a>torch.nn.qat.Linear.from_float</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p19618151171510"><a name="p19618151171510"></a><a name="p19618151171510"></a>否</p>
</td>
</tr>
<tr id="row1972215841216"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1314655364319"><a name="p1314655364319"></a><a name="p1314655364319"></a>297</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p19618112151"><a name="p19618112151"></a><a name="p19618112151"></a>torch.nn.quantized.functional.relu</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p156186191515"><a name="p156186191515"></a><a name="p156186191515"></a>否</p>
</td>
</tr>
<tr id="row20722145814121"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p191462534437"><a name="p191462534437"></a><a name="p191462534437"></a>298</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1461921121513"><a name="p1461921121513"></a><a name="p1461921121513"></a>torch.nn.quantized.functional.linear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p176197131511"><a name="p176197131511"></a><a name="p176197131511"></a>否</p>
</td>
</tr>
<tr id="row5722115841218"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p101461553124311"><a name="p101461553124311"></a><a name="p101461553124311"></a>299</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p2619416158"><a name="p2619416158"></a><a name="p2619416158"></a>torch.nn.quantized.functional.conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p17619716153"><a name="p17619716153"></a><a name="p17619716153"></a>否</p>
</td>
</tr>
<tr id="row27221658181214"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1014635324313"><a name="p1014635324313"></a><a name="p1014635324313"></a>300</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p134974137165"><a name="p134974137165"></a><a name="p134974137165"></a>torch.nn.quantized.functional.conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p649713136162"><a name="p649713136162"></a><a name="p649713136162"></a>否</p>
</td>
</tr>
<tr id="row3902096167"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p214685317433"><a name="p214685317433"></a><a name="p214685317433"></a>301</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p15497101331614"><a name="p15497101331614"></a><a name="p15497101331614"></a>torch.nn.quantized.functional.max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p144971713141615"><a name="p144971713141615"></a><a name="p144971713141615"></a>否</p>
</td>
</tr>
<tr id="row5258598166"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p2146453134316"><a name="p2146453134316"></a><a name="p2146453134316"></a>302</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p6497201351610"><a name="p6497201351610"></a><a name="p6497201351610"></a>torch.nn.quantized.functional.adaptive_avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p349731311610"><a name="p349731311610"></a><a name="p349731311610"></a>否</p>
</td>
</tr>
<tr id="row16419894168"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1714614536433"><a name="p1714614536433"></a><a name="p1714614536433"></a>303</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1497151341614"><a name="p1497151341614"></a><a name="p1497151341614"></a>torch.nn.quantized.functional.avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p149791318162"><a name="p149791318162"></a><a name="p149791318162"></a>否</p>
</td>
</tr>
<tr id="row195624919163"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p3146153164312"><a name="p3146153164312"></a><a name="p3146153164312"></a>304</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1549701318168"><a name="p1549701318168"></a><a name="p1549701318168"></a>torch.nn.quantized.functional.interpolate</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p18497121321611"><a name="p18497121321611"></a><a name="p18497121321611"></a>否</p>
</td>
</tr>
<tr id="row6706179101612"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1114611532433"><a name="p1114611532433"></a><a name="p1114611532433"></a>305</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p20497113121618"><a name="p20497113121618"></a><a name="p20497113121618"></a>torch.nn.quantized.functional.upsample</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p6497161331618"><a name="p6497161331618"></a><a name="p6497161331618"></a>否</p>
</td>
</tr>
<tr id="row1485910915169"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p10146175384310"><a name="p10146175384310"></a><a name="p10146175384310"></a>306</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16497713101615"><a name="p16497713101615"></a><a name="p16497713101615"></a>torch.nn.quantized.functional.upsample_bilinear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p44981131161"><a name="p44981131161"></a><a name="p44981131161"></a>否</p>
</td>
</tr>
<tr id="row129952913165"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p4146453164317"><a name="p4146453164317"></a><a name="p4146453164317"></a>307</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p15498181371610"><a name="p15498181371610"></a><a name="p15498181371610"></a>torch.nn.quantized.functional.upsample_nearest</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1649801301612"><a name="p1649801301612"></a><a name="p1649801301612"></a>否</p>
</td>
</tr>
<tr id="row141391110181612"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p2146125317439"><a name="p2146125317439"></a><a name="p2146125317439"></a>308</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p14981013161611"><a name="p14981013161611"></a><a name="p14981013161611"></a>torch.nn.quantized.ReLU</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p184981713161613"><a name="p184981713161613"></a><a name="p184981713161613"></a>否</p>
</td>
</tr>
<tr id="row22754101163"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p9146155314317"><a name="p9146155314317"></a><a name="p9146155314317"></a>309</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1249818137163"><a name="p1249818137163"></a><a name="p1249818137163"></a>torch.nn.quantized.ReLU6</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p19498131320160"><a name="p19498131320160"></a><a name="p19498131320160"></a>否</p>
</td>
</tr>
<tr id="row14435510191617"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11146185324316"><a name="p11146185324316"></a><a name="p11146185324316"></a>310</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p14498131320167"><a name="p14498131320167"></a><a name="p14498131320167"></a>torch.nn.quantized.Conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p3498213191615"><a name="p3498213191615"></a><a name="p3498213191615"></a>否</p>
</td>
</tr>
<tr id="row45808105169"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p11461553154312"><a name="p11461553154312"></a><a name="p11461553154312"></a>311</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p14498151311615"><a name="p14498151311615"></a><a name="p14498151311615"></a>torch.nn.quantized.Conv2d.from_float</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p16498181331620"><a name="p16498181331620"></a><a name="p16498181331620"></a>否</p>
</td>
</tr>
<tr id="row1773971021612"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p31461535438"><a name="p31461535438"></a><a name="p31461535438"></a>312</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p74983133168"><a name="p74983133168"></a><a name="p74983133168"></a>torch.nn.quantized.Conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1749811131164"><a name="p1749811131164"></a><a name="p1749811131164"></a>否</p>
</td>
</tr>
<tr id="row4882610151612"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p8146953174310"><a name="p8146953174310"></a><a name="p8146953174310"></a>313</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p17498171315169"><a name="p17498171315169"></a><a name="p17498171315169"></a>torch.nn.quantized.Conv3d.from_float</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1449815131164"><a name="p1449815131164"></a><a name="p1449815131164"></a>否</p>
</td>
</tr>
<tr id="row15021117163"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p614665384319"><a name="p614665384319"></a><a name="p614665384319"></a>314</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p16498513171612"><a name="p16498513171612"></a><a name="p16498513171612"></a>torch.nn.quantized.FloatFunctional</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p2498111361615"><a name="p2498111361615"></a><a name="p2498111361615"></a>否</p>
</td>
</tr>
<tr id="row16194161113162"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1314625344316"><a name="p1314625344316"></a><a name="p1314625344316"></a>315</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p18498313161615"><a name="p18498313161615"></a><a name="p18498313161615"></a>torch.nn.quantized.QFunctional</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p14498121361615"><a name="p14498121361615"></a><a name="p14498121361615"></a>否</p>
</td>
</tr>
<tr id="row14340311101614"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1146145354317"><a name="p1146145354317"></a><a name="p1146145354317"></a>316</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p34981913151618"><a name="p34981913151618"></a><a name="p34981913151618"></a>torch.nn.quantized.Quantize</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1249811131169"><a name="p1249811131169"></a><a name="p1249811131169"></a>否</p>
</td>
</tr>
<tr id="row64994112163"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p101471253114318"><a name="p101471253114318"></a><a name="p101471253114318"></a>317</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p11498513151614"><a name="p11498513151614"></a><a name="p11498513151614"></a>torch.nn.quantized.DeQuantize</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p10498191314162"><a name="p10498191314162"></a><a name="p10498191314162"></a>否</p>
</td>
</tr>
<tr id="row1063441116167"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p01471753154310"><a name="p01471753154310"></a><a name="p01471753154310"></a>318</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p124981113111617"><a name="p124981113111617"></a><a name="p124981113111617"></a>torch.nn.quantized.Linear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p449861316169"><a name="p449861316169"></a><a name="p449861316169"></a>否</p>
</td>
</tr>
<tr id="row99841449161517"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p121473534435"><a name="p121473534435"></a><a name="p121473534435"></a>319</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p24982013161617"><a name="p24982013161617"></a><a name="p24982013161617"></a>torch.nn.quantized.Linear.from_float</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p1349941312169"><a name="p1349941312169"></a><a name="p1349941312169"></a>否</p>
</td>
</tr>
<tr id="row10146115071517"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p71477530437"><a name="p71477530437"></a><a name="p71477530437"></a>320</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p1949919132168"><a name="p1949919132168"></a><a name="p1949919132168"></a>torch.nn.quantized.dynamic.Linear</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p5499413141612"><a name="p5499413141612"></a><a name="p5499413141612"></a>否</p>
</td>
</tr>
<tr id="row1576825018156"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p41479533438"><a name="p41479533438"></a><a name="p41479533438"></a>321</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p149911331610"><a name="p149911331610"></a><a name="p149911331610"></a>torch.nn.quantized.dynamic.Linear.from_float</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p14499131319163"><a name="p14499131319163"></a><a name="p14499131319163"></a>否</p>
</td>
</tr>
<tr id="row1582515508158"><td class="cellrowborder" valign="top" width="8.000000000000002%" headers="mcps1.1.4.1.1 "><p id="p1014718531438"><a name="p1014718531438"></a><a name="p1014718531438"></a>322</p>
</td>
<td class="cellrowborder" valign="top" width="71.50000000000001%" headers="mcps1.1.4.1.2 "><p id="p349961361612"><a name="p349961361612"></a><a name="p349961361612"></a>torch.nn.quantized.dynamic.LSTM</p>
</td>
<td class="cellrowborder" valign="top" width="20.5%" headers="mcps1.1.4.1.3 "><p id="p949951321612"><a name="p949951321612"></a><a name="p949951321612"></a>否</p>
</td>
</tr>
</tbody>
</table>

<h2 id="Functionstorch-nn-functional">Functions\(torch.nn.functional\)</h2>

<a name="table9911195161614"></a>
<table><thead align="left"><tr id="row591155115160"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p1339212174419"><a name="p1339212174419"></a><a name="p1339212174419"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row12911651101612"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7251937134415"><a name="p7251937134415"></a><a name="p7251937134415"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14292145412171"><a name="p14292145412171"></a><a name="p14292145412171"></a>torch.nn.functional.conv1d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p152921954111710"><a name="p152921954111710"></a><a name="p152921954111710"></a>是</p>
</td>
</tr>
<tr id="row10912951131610"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p42516377446"><a name="p42516377446"></a><a name="p42516377446"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p122926544171"><a name="p122926544171"></a><a name="p122926544171"></a>torch.nn.functional.conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p82929544177"><a name="p82929544177"></a><a name="p82929544177"></a>是</p>
</td>
</tr>
<tr id="row9912145115168"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p152511637144420"><a name="p152511637144420"></a><a name="p152511637144420"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p182921654151714"><a name="p182921654151714"></a><a name="p182921654151714"></a>torch.nn.functional.conv3d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p142921854111710"><a name="p142921854111710"></a><a name="p142921854111710"></a>是</p>
</td>
</tr>
<tr id="row149120516169"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2025115375442"><a name="p2025115375442"></a><a name="p2025115375442"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p929295416174"><a name="p929295416174"></a><a name="p929295416174"></a>torch.nn.functional.conv_transpose1d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p102927544178"><a name="p102927544178"></a><a name="p102927544178"></a>是</p>
</td>
</tr>
<tr id="row3912195101614"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1925173717449"><a name="p1925173717449"></a><a name="p1925173717449"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p112926547173"><a name="p112926547173"></a><a name="p112926547173"></a>torch.nn.functional.conv_transpose2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5292115431717"><a name="p5292115431717"></a><a name="p5292115431717"></a>是</p>
</td>
</tr>
<tr id="row199121251191618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1925273784413"><a name="p1925273784413"></a><a name="p1925273784413"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8292185411175"><a name="p8292185411175"></a><a name="p8292185411175"></a>torch.nn.functional.conv_transpose3d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p129285431714"><a name="p129285431714"></a><a name="p129285431714"></a>否</p>
</td>
</tr>
<tr id="row1391295120167"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p325273714444"><a name="p325273714444"></a><a name="p325273714444"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2292254161717"><a name="p2292254161717"></a><a name="p2292254161717"></a>torch.nn.functional.unfold</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5292155416171"><a name="p5292155416171"></a><a name="p5292155416171"></a>否</p>
</td>
</tr>
<tr id="row9912195151615"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p525217370442"><a name="p525217370442"></a><a name="p525217370442"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1929214548179"><a name="p1929214548179"></a><a name="p1929214548179"></a>torch.nn.functional.fold</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p629265415170"><a name="p629265415170"></a><a name="p629265415170"></a>是</p>
</td>
</tr>
<tr id="row1591213519165"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p152521637174419"><a name="p152521637174419"></a><a name="p152521637174419"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15292105451714"><a name="p15292105451714"></a><a name="p15292105451714"></a>torch.nn.functional.avg_pool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16292754161716"><a name="p16292754161716"></a><a name="p16292754161716"></a>是</p>
</td>
</tr>
<tr id="row6912155117163"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p182521437164420"><a name="p182521437164420"></a><a name="p182521437164420"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p329295401713"><a name="p329295401713"></a><a name="p329295401713"></a>torch.nn.functional.avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15292195417176"><a name="p15292195417176"></a><a name="p15292195417176"></a>是</p>
</td>
</tr>
<tr id="row129122051111617"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10252163713444"><a name="p10252163713444"></a><a name="p10252163713444"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p329225481715"><a name="p329225481715"></a><a name="p329225481715"></a>torch.nn.functional.avg_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7292165420176"><a name="p7292165420176"></a><a name="p7292165420176"></a>是</p>
</td>
</tr>
<tr id="row29128516164"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p825283716449"><a name="p825283716449"></a><a name="p825283716449"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p429220540178"><a name="p429220540178"></a><a name="p429220540178"></a>torch.nn.functional.max_pool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p829210544176"><a name="p829210544176"></a><a name="p829210544176"></a>是</p>
</td>
</tr>
<tr id="row10912051121618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7252133794418"><a name="p7252133794418"></a><a name="p7252133794418"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17292754181712"><a name="p17292754181712"></a><a name="p17292754181712"></a>torch.nn.functional.max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p42936548177"><a name="p42936548177"></a><a name="p42936548177"></a>是</p>
</td>
</tr>
<tr id="row291285161613"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p825215377447"><a name="p825215377447"></a><a name="p825215377447"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1229385410170"><a name="p1229385410170"></a><a name="p1229385410170"></a>torch.nn.functional.max_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3293554181716"><a name="p3293554181716"></a><a name="p3293554181716"></a>是</p>
</td>
</tr>
<tr id="row2912195191612"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p42521237134411"><a name="p42521237134411"></a><a name="p42521237134411"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p129395411172"><a name="p129395411172"></a><a name="p129395411172"></a>torch.nn.functional.max_unpool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1029305413179"><a name="p1029305413179"></a><a name="p1029305413179"></a>否</p>
</td>
</tr>
<tr id="row99121451141610"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1725293744420"><a name="p1725293744420"></a><a name="p1725293744420"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1829320545173"><a name="p1829320545173"></a><a name="p1829320545173"></a>torch.nn.functional.max_unpool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1129320547173"><a name="p1129320547173"></a><a name="p1129320547173"></a>否</p>
</td>
</tr>
<tr id="row49123512166"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p152521637174416"><a name="p152521637174416"></a><a name="p152521637174416"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16293185411715"><a name="p16293185411715"></a><a name="p16293185411715"></a>torch.nn.functional.max_unpool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1293754131711"><a name="p1293754131711"></a><a name="p1293754131711"></a>否</p>
</td>
</tr>
<tr id="row20912651141614"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p125243718440"><a name="p125243718440"></a><a name="p125243718440"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p4293954151710"><a name="p4293954151710"></a><a name="p4293954151710"></a>torch.nn.functional.lp_pool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1656163275419"><a name="p1656163275419"></a><a name="p1656163275419"></a>是</p>
</td>
</tr>
<tr id="row39137519169"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18252193710441"><a name="p18252193710441"></a><a name="p18252193710441"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1293054121710"><a name="p1293054121710"></a><a name="p1293054121710"></a>torch.nn.functional.lp_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12150113415410"><a name="p12150113415410"></a><a name="p12150113415410"></a>是</p>
</td>
</tr>
<tr id="row1691313515168"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4252037144411"><a name="p4252037144411"></a><a name="p4252037144411"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1429313541178"><a name="p1429313541178"></a><a name="p1429313541178"></a>torch.nn.functional.adaptive_max_pool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13779666475"><a name="p13779666475"></a><a name="p13779666475"></a>是</p>
</td>
</tr>
<tr id="row1891325111612"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1125283714411"><a name="p1125283714411"></a><a name="p1125283714411"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p429318545177"><a name="p429318545177"></a><a name="p429318545177"></a>torch.nn.functional.adaptive_max_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p152931454201719"><a name="p152931454201719"></a><a name="p152931454201719"></a>是</p>
</td>
</tr>
<tr id="row1991316512169"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p162521378441"><a name="p162521378441"></a><a name="p162521378441"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1029313541177"><a name="p1029313541177"></a><a name="p1029313541177"></a>torch.nn.functional.adaptive_max_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p32939542172"><a name="p32939542172"></a><a name="p32939542172"></a>否</p>
</td>
</tr>
<tr id="row49134510168"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1252153718446"><a name="p1252153718446"></a><a name="p1252153718446"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p929305491713"><a name="p929305491713"></a><a name="p929305491713"></a>torch.nn.functional.adaptive_avg_pool1d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p82931954111717"><a name="p82931954111717"></a><a name="p82931954111717"></a>是</p>
</td>
</tr>
<tr id="row59131551141613"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p625283784412"><a name="p625283784412"></a><a name="p625283784412"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19293195401718"><a name="p19293195401718"></a><a name="p19293195401718"></a>torch.nn.functional.adaptive_avg_pool2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12939544171"><a name="p12939544171"></a><a name="p12939544171"></a>是</p>
</td>
</tr>
<tr id="row4913125114162"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p925233754410"><a name="p925233754410"></a><a name="p925233754410"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1629395461716"><a name="p1629395461716"></a><a name="p1629395461716"></a>torch.nn.functional.adaptive_avg_pool3d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8294125417172"><a name="p8294125417172"></a><a name="p8294125417172"></a>是</p>
</td>
</tr>
<tr id="row1591365115165"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p425233734418"><a name="p425233734418"></a><a name="p425233734418"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p629412543176"><a name="p629412543176"></a><a name="p629412543176"></a>torch.nn.functional.threshold</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p829425491711"><a name="p829425491711"></a><a name="p829425491711"></a>是</p>
</td>
</tr>
<tr id="row79131351121620"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1825313375441"><a name="p1825313375441"></a><a name="p1825313375441"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17294195415175"><a name="p17294195415175"></a><a name="p17294195415175"></a>torch.nn.functional.threshold_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13294354131710"><a name="p13294354131710"></a><a name="p13294354131710"></a>是</p>
</td>
</tr>
<tr id="row291375118160"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p14253153719445"><a name="p14253153719445"></a><a name="p14253153719445"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p229475481710"><a name="p229475481710"></a><a name="p229475481710"></a>torch.nn.functional.relu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p62945543177"><a name="p62945543177"></a><a name="p62945543177"></a>是</p>
</td>
</tr>
<tr id="row59132051151618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p17253173754418"><a name="p17253173754418"></a><a name="p17253173754418"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1629412541179"><a name="p1629412541179"></a><a name="p1629412541179"></a>torch.nn.functional.relu_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18294105420179"><a name="p18294105420179"></a><a name="p18294105420179"></a>是</p>
</td>
</tr>
<tr id="row139131951171616"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1325343717446"><a name="p1325343717446"></a><a name="p1325343717446"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p22941354171714"><a name="p22941354171714"></a><a name="p22941354171714"></a>torch.nn.functional.hardtanh</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p129425415178"><a name="p129425415178"></a><a name="p129425415178"></a>是</p>
</td>
</tr>
<tr id="row4913251171618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p82531637144416"><a name="p82531637144416"></a><a name="p82531637144416"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p829419544170"><a name="p829419544170"></a><a name="p829419544170"></a>torch.nn.functional.hardtanh_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14294135491710"><a name="p14294135491710"></a><a name="p14294135491710"></a>是</p>
</td>
</tr>
<tr id="row16913185181614"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5253137174413"><a name="p5253137174413"></a><a name="p5253137174413"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p329414547171"><a name="p329414547171"></a><a name="p329414547171"></a>torch.nn.functional.relu6</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p192944549170"><a name="p192944549170"></a><a name="p192944549170"></a>是</p>
</td>
</tr>
<tr id="row891365121620"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1925383714444"><a name="p1925383714444"></a><a name="p1925383714444"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p182941541174"><a name="p182941541174"></a><a name="p182941541174"></a>torch.nn.functional.elu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p102944546173"><a name="p102944546173"></a><a name="p102944546173"></a>是</p>
</td>
</tr>
<tr id="row891335118163"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2253153754415"><a name="p2253153754415"></a><a name="p2253153754415"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1129445413178"><a name="p1129445413178"></a><a name="p1129445413178"></a>torch.nn.functional.elu_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0294105421712"><a name="p0294105421712"></a><a name="p0294105421712"></a>是</p>
</td>
</tr>
<tr id="row2913135120167"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p142531437194414"><a name="p142531437194414"></a><a name="p142531437194414"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10294125431715"><a name="p10294125431715"></a><a name="p10294125431715"></a>torch.nn.functional.selu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1729410544173"><a name="p1729410544173"></a><a name="p1729410544173"></a>是</p>
</td>
</tr>
<tr id="row0914451171619"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9253037184416"><a name="p9253037184416"></a><a name="p9253037184416"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1294165421710"><a name="p1294165421710"></a><a name="p1294165421710"></a>torch.nn.functional.celu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p202953543177"><a name="p202953543177"></a><a name="p202953543177"></a>是</p>
</td>
</tr>
<tr id="row2091415181613"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p112561337144416"><a name="p112561337144416"></a><a name="p112561337144416"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17295105411179"><a name="p17295105411179"></a><a name="p17295105411179"></a>torch.nn.functional.leaky_relu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p14295105471718"><a name="p14295105471718"></a><a name="p14295105471718"></a>是</p>
</td>
</tr>
<tr id="row189141516162"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p132568372446"><a name="p132568372446"></a><a name="p132568372446"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p32951954151717"><a name="p32951954151717"></a><a name="p32951954151717"></a>torch.nn.functional.leaky_relu_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2295115416175"><a name="p2295115416175"></a><a name="p2295115416175"></a>是</p>
</td>
</tr>
<tr id="row1591405112168"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1825618376446"><a name="p1825618376446"></a><a name="p1825618376446"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1029525413170"><a name="p1029525413170"></a><a name="p1029525413170"></a>torch.nn.functional.prelu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p18295105416171"><a name="p18295105416171"></a><a name="p18295105416171"></a>是</p>
</td>
</tr>
<tr id="row19914155119169"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4256143764412"><a name="p4256143764412"></a><a name="p4256143764412"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2029545410176"><a name="p2029545410176"></a><a name="p2029545410176"></a>torch.nn.functional.rrelu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2295754111712"><a name="p2295754111712"></a><a name="p2295754111712"></a>否</p>
</td>
</tr>
<tr id="row1191485111617"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1925616374449"><a name="p1925616374449"></a><a name="p1925616374449"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1429565461718"><a name="p1429565461718"></a><a name="p1429565461718"></a>torch.nn.functional.rrelu_</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1629519545173"><a name="p1629519545173"></a><a name="p1629519545173"></a>否</p>
</td>
</tr>
<tr id="row4914165141612"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p4256183715448"><a name="p4256183715448"></a><a name="p4256183715448"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p7295155413171"><a name="p7295155413171"></a><a name="p7295155413171"></a>torch.nn.functional.glu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p182951654191715"><a name="p182951654191715"></a><a name="p182951654191715"></a>是</p>
</td>
</tr>
<tr id="row19149515164"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1625610377441"><a name="p1625610377441"></a><a name="p1625610377441"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11295125411173"><a name="p11295125411173"></a><a name="p11295125411173"></a>torch.nn.functional.gelu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p6295155410170"><a name="p6295155410170"></a><a name="p6295155410170"></a>是</p>
</td>
</tr>
<tr id="row69148514163"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p52561937144419"><a name="p52561937144419"></a><a name="p52561937144419"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p16295185491716"><a name="p16295185491716"></a><a name="p16295185491716"></a>torch.nn.functional.logsigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12295115491716"><a name="p12295115491716"></a><a name="p12295115491716"></a>是</p>
</td>
</tr>
<tr id="row1091445112160"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p32565370445"><a name="p32565370445"></a><a name="p32565370445"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p11295254111710"><a name="p11295254111710"></a><a name="p11295254111710"></a>torch.nn.functional.hardshrink</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0295554111710"><a name="p0295554111710"></a><a name="p0295554111710"></a>是</p>
</td>
</tr>
<tr id="row169141951171620"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1425619370443"><a name="p1425619370443"></a><a name="p1425619370443"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p62951154141710"><a name="p62951154141710"></a><a name="p62951154141710"></a>torch.nn.functional.tanhshrink</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p329512549175"><a name="p329512549175"></a><a name="p329512549175"></a>是</p>
</td>
</tr>
<tr id="row1791425131613"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19256153719444"><a name="p19256153719444"></a><a name="p19256153719444"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p152951354181711"><a name="p152951354181711"></a><a name="p152951354181711"></a>torch.nn.functional.softsign</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16295205418175"><a name="p16295205418175"></a><a name="p16295205418175"></a>是</p>
</td>
</tr>
<tr id="row14914651191619"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p182561737174417"><a name="p182561737174417"></a><a name="p182561737174417"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5295125416173"><a name="p5295125416173"></a><a name="p5295125416173"></a>torch.nn.functional.softplus</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p0295125412171"><a name="p0295125412171"></a><a name="p0295125412171"></a>是</p>
</td>
</tr>
<tr id="row291425113164"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p52571037184414"><a name="p52571037184414"></a><a name="p52571037184414"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p4296354101713"><a name="p4296354101713"></a><a name="p4296354101713"></a>torch.nn.functional.softmin</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p11296254111718"><a name="p11296254111718"></a><a name="p11296254111718"></a>是</p>
</td>
</tr>
<tr id="row1491415151616"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8257113784419"><a name="p8257113784419"></a><a name="p8257113784419"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5296954111715"><a name="p5296954111715"></a><a name="p5296954111715"></a>torch.nn.functional.softmax</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1829625413175"><a name="p1829625413175"></a><a name="p1829625413175"></a>是</p>
</td>
</tr>
<tr id="row791455121610"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1925713711446"><a name="p1925713711446"></a><a name="p1925713711446"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1296165441720"><a name="p1296165441720"></a><a name="p1296165441720"></a>torch.nn.functional.softshrink</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1229615491719"><a name="p1229615491719"></a><a name="p1229615491719"></a>否</p>
</td>
</tr>
<tr id="row13914051181618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p725712372447"><a name="p725712372447"></a><a name="p725712372447"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p829625411177"><a name="p829625411177"></a><a name="p829625411177"></a>torch.nn.functional.gumbel_softmax</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p142962540177"><a name="p142962540177"></a><a name="p142962540177"></a>否</p>
</td>
</tr>
<tr id="row6915175112164"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p425713375447"><a name="p425713375447"></a><a name="p425713375447"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1729613548176"><a name="p1729613548176"></a><a name="p1729613548176"></a>torch.nn.functional.log_softmax</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p142961546179"><a name="p142961546179"></a><a name="p142961546179"></a>是</p>
</td>
</tr>
<tr id="row109156512166"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p102578379443"><a name="p102578379443"></a><a name="p102578379443"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1329675441715"><a name="p1329675441715"></a><a name="p1329675441715"></a>torch.nn.functional.tanh</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1296854151715"><a name="p1296854151715"></a><a name="p1296854151715"></a>是</p>
</td>
</tr>
<tr id="row12915185114167"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p42571437144415"><a name="p42571437144415"></a><a name="p42571437144415"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1529665441716"><a name="p1529665441716"></a><a name="p1529665441716"></a>torch.nn.functional.sigmoid</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1629685415174"><a name="p1629685415174"></a><a name="p1629685415174"></a>是</p>
</td>
</tr>
<tr id="row12915155117168"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1625773704410"><a name="p1625773704410"></a><a name="p1625773704410"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1296135401717"><a name="p1296135401717"></a><a name="p1296135401717"></a>torch.nn.functional.batch_norm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8296125420173"><a name="p8296125420173"></a><a name="p8296125420173"></a>是</p>
</td>
</tr>
<tr id="row49154517162"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13257183774416"><a name="p13257183774416"></a><a name="p13257183774416"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p9296754121715"><a name="p9296754121715"></a><a name="p9296754121715"></a>torch.nn.functional.instance_norm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2029695491716"><a name="p2029695491716"></a><a name="p2029695491716"></a>是</p>
</td>
</tr>
<tr id="row1191514513166"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p102571137184416"><a name="p102571137184416"></a><a name="p102571137184416"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13297115417174"><a name="p13297115417174"></a><a name="p13297115417174"></a>torch.nn.functional.layer_norm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1029735415179"><a name="p1029735415179"></a><a name="p1029735415179"></a>是</p>
</td>
</tr>
<tr id="row11915151151610"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19257173774419"><a name="p19257173774419"></a><a name="p19257173774419"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p829718543174"><a name="p829718543174"></a><a name="p829718543174"></a>torch.nn.functional.local_response_norm</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p82971454151710"><a name="p82971454151710"></a><a name="p82971454151710"></a>是</p>
</td>
</tr>
<tr id="row17915155118164"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8257437194418"><a name="p8257437194418"></a><a name="p8257437194418"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p132976547175"><a name="p132976547175"></a><a name="p132976547175"></a>torch.nn.functional.normalize</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p72971549170"><a name="p72971549170"></a><a name="p72971549170"></a>是</p>
</td>
</tr>
<tr id="row10915185151620"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1125723714411"><a name="p1125723714411"></a><a name="p1125723714411"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p129765471713"><a name="p129765471713"></a><a name="p129765471713"></a>torch.nn.functional.linear</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10297145421719"><a name="p10297145421719"></a><a name="p10297145421719"></a>是</p>
</td>
</tr>
<tr id="row12915125181620"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2025715376445"><a name="p2025715376445"></a><a name="p2025715376445"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p6297954151712"><a name="p6297954151712"></a><a name="p6297954151712"></a>torch.nn.functional.bilinear</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p8297205411719"><a name="p8297205411719"></a><a name="p8297205411719"></a>是</p>
</td>
</tr>
<tr id="row691585121610"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p52571537164420"><a name="p52571537164420"></a><a name="p52571537164420"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19297185415171"><a name="p19297185415171"></a><a name="p19297185415171"></a>torch.nn.functional.dropout</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3297554121719"><a name="p3297554121719"></a><a name="p3297554121719"></a>是</p>
</td>
</tr>
<tr id="row19915551101616"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6258537174420"><a name="p6258537174420"></a><a name="p6258537174420"></a>64</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1629795431711"><a name="p1629795431711"></a><a name="p1629795431711"></a>torch.nn.functional.alpha_dropout</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13297185413175"><a name="p13297185413175"></a><a name="p13297185413175"></a>是</p>
</td>
</tr>
<tr id="row191510513169"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1125823774412"><a name="p1125823774412"></a><a name="p1125823774412"></a>65</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p629775414172"><a name="p629775414172"></a><a name="p629775414172"></a>torch.nn.functional.dropout2d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p10297125411711"><a name="p10297125411711"></a><a name="p10297125411711"></a>否</p>
</td>
</tr>
<tr id="row139152051101619"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p125843717440"><a name="p125843717440"></a><a name="p125843717440"></a>66</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13297125417175"><a name="p13297125417175"></a><a name="p13297125417175"></a>torch.nn.functional.dropout3d</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15297185431714"><a name="p15297185431714"></a><a name="p15297185431714"></a>否</p>
</td>
</tr>
<tr id="row18915195112165"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3258143711447"><a name="p3258143711447"></a><a name="p3258143711447"></a>67</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1629705421712"><a name="p1629705421712"></a><a name="p1629705421712"></a>torch.nn.functional.embedding</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p62971554151719"><a name="p62971554151719"></a><a name="p62971554151719"></a>是</p>
</td>
</tr>
<tr id="row149151051131614"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p152582377445"><a name="p152582377445"></a><a name="p152582377445"></a>68</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p112977546178"><a name="p112977546178"></a><a name="p112977546178"></a>torch.nn.functional.embedding_bag</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16297854191710"><a name="p16297854191710"></a><a name="p16297854191710"></a>否</p>
</td>
</tr>
<tr id="row10916851201620"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9258183712441"><a name="p9258183712441"></a><a name="p9258183712441"></a>69</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15297454131710"><a name="p15297454131710"></a><a name="p15297454131710"></a>torch.nn.functional.one_hot</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p192981054161719"><a name="p192981054161719"></a><a name="p192981054161719"></a>是</p>
</td>
</tr>
<tr id="row09161651151613"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1925843713448"><a name="p1925843713448"></a><a name="p1925843713448"></a>70</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p729875415177"><a name="p729875415177"></a><a name="p729875415177"></a>torch.nn.functional.pairwise_distance</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p42981754111710"><a name="p42981754111710"></a><a name="p42981754111710"></a>是</p>
</td>
</tr>
<tr id="row591645121619"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9258437184418"><a name="p9258437184418"></a><a name="p9258437184418"></a>71</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p162985549179"><a name="p162985549179"></a><a name="p162985549179"></a>torch.nn.functional.cosine_similarity</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1429835471711"><a name="p1429835471711"></a><a name="p1429835471711"></a>是</p>
</td>
</tr>
<tr id="row139161251191612"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p82581376444"><a name="p82581376444"></a><a name="p82581376444"></a>72</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p122981154181713"><a name="p122981154181713"></a><a name="p122981154181713"></a>torch.nn.functional.pdist</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p3298105415175"><a name="p3298105415175"></a><a name="p3298105415175"></a>是</p>
</td>
</tr>
<tr id="row16916151171616"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p192582371445"><a name="p192582371445"></a><a name="p192582371445"></a>73</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p629820548177"><a name="p629820548177"></a><a name="p629820548177"></a>torch.nn.functional.binary_cross_entropy</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p5298155431714"><a name="p5298155431714"></a><a name="p5298155431714"></a>是</p>
</td>
</tr>
<tr id="row89161551181617"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p825863713444"><a name="p825863713444"></a><a name="p825863713444"></a>74</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p729819540175"><a name="p729819540175"></a><a name="p729819540175"></a>torch.nn.functional.binary_cross_entropy_with_logits</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12298954181710"><a name="p12298954181710"></a><a name="p12298954181710"></a>是</p>
</td>
</tr>
<tr id="row39161951101617"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p425833734418"><a name="p425833734418"></a><a name="p425833734418"></a>75</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1229875451713"><a name="p1229875451713"></a><a name="p1229875451713"></a>torch.nn.functional.poisson_nll_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p162981254191716"><a name="p162981254191716"></a><a name="p162981254191716"></a>是</p>
</td>
</tr>
<tr id="row12916175120167"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13258173715446"><a name="p13258173715446"></a><a name="p13258173715446"></a>76</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p112981354181718"><a name="p112981354181718"></a><a name="p112981354181718"></a>torch.nn.functional.cosine_embedding_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1229812545178"><a name="p1229812545178"></a><a name="p1229812545178"></a>是</p>
</td>
</tr>
<tr id="row591615181615"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1925863714417"><a name="p1925863714417"></a><a name="p1925863714417"></a>77</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14298185416176"><a name="p14298185416176"></a><a name="p14298185416176"></a>torch.nn.functional.cross_entropy</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p172981454151714"><a name="p172981454151714"></a><a name="p172981454151714"></a>是</p>
</td>
</tr>
<tr id="row991635120169"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p20258237134414"><a name="p20258237134414"></a><a name="p20258237134414"></a>78</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p529855414175"><a name="p529855414175"></a><a name="p529855414175"></a>torch.nn.functional.ctc_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p122987548172"><a name="p122987548172"></a><a name="p122987548172"></a>是</p>
</td>
</tr>
<tr id="row1691655191618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3258537194416"><a name="p3258537194416"></a><a name="p3258537194416"></a>79</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p829814545177"><a name="p829814545177"></a><a name="p829814545177"></a>torch.nn.functional.hinge_embedding_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1298854131717"><a name="p1298854131717"></a><a name="p1298854131717"></a>是</p>
</td>
</tr>
<tr id="row1191685131615"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p225873712445"><a name="p225873712445"></a><a name="p225873712445"></a>80</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p729855417172"><a name="p729855417172"></a><a name="p729855417172"></a>torch.nn.functional.kl_div</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p13298954201718"><a name="p13298954201718"></a><a name="p13298954201718"></a>是</p>
</td>
</tr>
<tr id="row1191610510169"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5258837174414"><a name="p5258837174414"></a><a name="p5258837174414"></a>81</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p029812544170"><a name="p029812544170"></a><a name="p029812544170"></a>torch.nn.functional.l1_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2298135412173"><a name="p2298135412173"></a><a name="p2298135412173"></a>是</p>
</td>
</tr>
<tr id="row18916105110169"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p0258163713447"><a name="p0258163713447"></a><a name="p0258163713447"></a>82</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p192991654131713"><a name="p192991654131713"></a><a name="p192991654131713"></a>torch.nn.functional.mse_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p629914542179"><a name="p629914542179"></a><a name="p629914542179"></a>是</p>
</td>
</tr>
<tr id="row1491655117161"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p18258173724411"><a name="p18258173724411"></a><a name="p18258173724411"></a>83</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p52991554171715"><a name="p52991554171715"></a><a name="p52991554171715"></a>torch.nn.functional.margin_ranking_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p16299125415170"><a name="p16299125415170"></a><a name="p16299125415170"></a>是</p>
</td>
</tr>
<tr id="row1891635120166"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p2025943714449"><a name="p2025943714449"></a><a name="p2025943714449"></a>84</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8299175491720"><a name="p8299175491720"></a><a name="p8299175491720"></a>torch.nn.functional.multilabel_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p7299175421711"><a name="p7299175421711"></a><a name="p7299175421711"></a>是</p>
</td>
</tr>
<tr id="row179161751131619"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1725917372441"><a name="p1725917372441"></a><a name="p1725917372441"></a>85</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17299145419175"><a name="p17299145419175"></a><a name="p17299145419175"></a>torch.nn.functional.multilabel_soft_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p102991754121711"><a name="p102991754121711"></a><a name="p102991754121711"></a>是</p>
</td>
</tr>
<tr id="row16917135171618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15259113715446"><a name="p15259113715446"></a><a name="p15259113715446"></a>86</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1429915546178"><a name="p1429915546178"></a><a name="p1429915546178"></a>torch.nn.functional.multi_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p92996545177"><a name="p92996545177"></a><a name="p92996545177"></a>否</p>
</td>
</tr>
<tr id="row591713518161"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p102592378448"><a name="p102592378448"></a><a name="p102592378448"></a>87</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p229995413177"><a name="p229995413177"></a><a name="p229995413177"></a>torch.nn.functional.nll_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p429945412175"><a name="p429945412175"></a><a name="p429945412175"></a>是</p>
</td>
</tr>
<tr id="row89171951141614"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p725933718448"><a name="p725933718448"></a><a name="p725933718448"></a>88</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p8299354151719"><a name="p8299354151719"></a><a name="p8299354151719"></a>torch.nn.functional.smooth_l1_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15299185414173"><a name="p15299185414173"></a><a name="p15299185414173"></a>是</p>
</td>
</tr>
<tr id="row8917105171612"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p19259193714410"><a name="p19259193714410"></a><a name="p19259193714410"></a>89</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1629955416178"><a name="p1629955416178"></a><a name="p1629955416178"></a>torch.nn.functional.soft_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p42991454101717"><a name="p42991454101717"></a><a name="p42991454101717"></a>是</p>
</td>
</tr>
<tr id="row691785121618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p72591937164412"><a name="p72591937164412"></a><a name="p72591937164412"></a>90</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p172991854111717"><a name="p172991854111717"></a><a name="p172991854111717"></a>torch.nn.functional.triplet_margin_loss</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1529935417179"><a name="p1529935417179"></a><a name="p1529935417179"></a>是</p>
</td>
</tr>
<tr id="row1791714519166"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p132591337114413"><a name="p132591337114413"></a><a name="p132591337114413"></a>91</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p10299115418173"><a name="p10299115418173"></a><a name="p10299115418173"></a>torch.nn.functional.pixel_shuffle</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1529965481710"><a name="p1529965481710"></a><a name="p1529965481710"></a>是</p>
</td>
</tr>
<tr id="row14917165114163"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p425919371449"><a name="p425919371449"></a><a name="p425919371449"></a>92</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p192991554121713"><a name="p192991554121713"></a><a name="p192991554121713"></a>torch.nn.functional.pad</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p2299145414171"><a name="p2299145414171"></a><a name="p2299145414171"></a>是</p>
</td>
</tr>
<tr id="row49171551121611"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p16259737104412"><a name="p16259737104412"></a><a name="p16259737104412"></a>93</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19299195461711"><a name="p19299195461711"></a><a name="p19299195461711"></a>torch.nn.functional.interpolate</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p829945411711"><a name="p829945411711"></a><a name="p829945411711"></a>否</p>
</td>
</tr>
<tr id="row7917115141613"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p7259183704414"><a name="p7259183704414"></a><a name="p7259183704414"></a>94</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1029955411713"><a name="p1029955411713"></a><a name="p1029955411713"></a>torch.nn.functional.upsample</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p163004549178"><a name="p163004549178"></a><a name="p163004549178"></a>否</p>
</td>
</tr>
<tr id="row39171451111618"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p192590377448"><a name="p192590377448"></a><a name="p192590377448"></a>95</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18300205491716"><a name="p18300205491716"></a><a name="p18300205491716"></a>torch.nn.functional.upsample_nearest</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p33003541178"><a name="p33003541178"></a><a name="p33003541178"></a>否</p>
</td>
</tr>
<tr id="row139171051201617"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1825973774413"><a name="p1825973774413"></a><a name="p1825973774413"></a>96</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p14300125441718"><a name="p14300125441718"></a><a name="p14300125441718"></a>torch.nn.functional.upsample_bilinear</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p63001554191711"><a name="p63001554191711"></a><a name="p63001554191711"></a>是</p>
</td>
</tr>
<tr id="row12917145131610"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1259133719449"><a name="p1259133719449"></a><a name="p1259133719449"></a>97</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p18300175411179"><a name="p18300175411179"></a><a name="p18300175411179"></a>torch.nn.functional.grid_sample</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p193001854151716"><a name="p193001854151716"></a><a name="p193001854151716"></a>是</p>
</td>
</tr>
<tr id="row19171651151610"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6259103716441"><a name="p6259103716441"></a><a name="p6259103716441"></a>98</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19300145441713"><a name="p19300145441713"></a><a name="p19300145441713"></a>torch.nn.functional.affine_grid</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1630018547177"><a name="p1630018547177"></a><a name="p1630018547177"></a>否</p>
</td>
</tr>
<tr id="row1917751201612"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1225973714411"><a name="p1225973714411"></a><a name="p1225973714411"></a>99</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p13300155411174"><a name="p13300155411174"></a><a name="p13300155411174"></a>torch.nn.parallel.data_parallel</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15300195419176"><a name="p15300195419176"></a><a name="p15300195419176"></a>否</p>
</td>
</tr>
</tbody>
</table>

<h2 id="torch-distributed">torch.distributed</h2>

<a name="table2583230111811"></a>
<table><thead align="left"><tr id="row11583113018186"><th class="cellrowborder" valign="top" width="10%" id="mcps1.1.4.1.1"><p id="p1356195214418"><a name="p1356195214418"></a><a name="p1356195214418"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="70%" id="mcps1.1.4.1.2"><p id="p9571151720222"><a name="p9571151720222"></a><a name="p9571151720222"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="20%" id="mcps1.1.4.1.3"><p id="p2038513010539"><a name="p2038513010539"></a><a name="p2038513010539"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row058316307186"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6509102811457"><a name="p6509102811457"></a><a name="p6509102811457"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p073713142115"><a name="p073713142115"></a><a name="p073713142115"></a>torch.distributed.init_process_group</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p473715110218"><a name="p473715110218"></a><a name="p473715110218"></a>是</p>
</td>
</tr>
<tr id="row1458443011810"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p3509428134519"><a name="p3509428134519"></a><a name="p3509428134519"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p137376162115"><a name="p137376162115"></a><a name="p137376162115"></a>torch.distributed.Backend</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1373717182112"><a name="p1373717182112"></a><a name="p1373717182112"></a>是</p>
</td>
</tr>
<tr id="row16584133091819"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1650915281452"><a name="p1650915281452"></a><a name="p1650915281452"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p177379115217"><a name="p177379115217"></a><a name="p177379115217"></a>torch.distributed.get_backend</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p4737510218"><a name="p4737510218"></a><a name="p4737510218"></a>是</p>
</td>
</tr>
<tr id="row2584103061812"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p10509528144519"><a name="p10509528144519"></a><a name="p10509528144519"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p7737131172110"><a name="p7737131172110"></a><a name="p7737131172110"></a>torch.distributed.get_rank</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1373791192113"><a name="p1373791192113"></a><a name="p1373791192113"></a>是</p>
</td>
</tr>
<tr id="row1358443011180"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8509172834517"><a name="p8509172834517"></a><a name="p8509172834517"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1273791172119"><a name="p1273791172119"></a><a name="p1273791172119"></a>torch.distributed.get_world_size</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12737617215"><a name="p12737617215"></a><a name="p12737617215"></a>是</p>
</td>
</tr>
<tr id="row15584133013184"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p250916282459"><a name="p250916282459"></a><a name="p250916282459"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p2737131122111"><a name="p2737131122111"></a><a name="p2737131122111"></a>torch.distributed.is_initialized</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p15737415215"><a name="p15737415215"></a><a name="p15737415215"></a>是</p>
</td>
</tr>
<tr id="row55841730171818"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p05091228124512"><a name="p05091228124512"></a><a name="p05091228124512"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p77377102112"><a name="p77377102112"></a><a name="p77377102112"></a>torch.distributed.is_mpi_available</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p473713112217"><a name="p473713112217"></a><a name="p473713112217"></a>是</p>
</td>
</tr>
<tr id="row1258413301183"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6509122818458"><a name="p6509122818458"></a><a name="p6509122818458"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1973861162115"><a name="p1973861162115"></a><a name="p1973861162115"></a>torch.distributed.is_nccl_available</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p273819112215"><a name="p273819112215"></a><a name="p273819112215"></a>是</p>
</td>
</tr>
<tr id="row1584330181815"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p65091728144516"><a name="p65091728144516"></a><a name="p65091728144516"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p073814112214"><a name="p073814112214"></a><a name="p073814112214"></a>torch.distributed.new_group</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1473813119211"><a name="p1473813119211"></a><a name="p1473813119211"></a>是</p>
</td>
</tr>
<tr id="row1858473019182"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9509202811454"><a name="p9509202811454"></a><a name="p9509202811454"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p57380122119"><a name="p57380122119"></a><a name="p57380122119"></a>torch.distributed.send</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p073811112113"><a name="p073811112113"></a><a name="p073811112113"></a>否</p>
</td>
</tr>
<tr id="row958443011183"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1050962819454"><a name="p1050962819454"></a><a name="p1050962819454"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p87383112219"><a name="p87383112219"></a><a name="p87383112219"></a>torch.distributed.recv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1173817102119"><a name="p1173817102119"></a><a name="p1173817102119"></a>否</p>
</td>
</tr>
<tr id="row1458411302187"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1250932811455"><a name="p1250932811455"></a><a name="p1250932811455"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p873810182114"><a name="p873810182114"></a><a name="p873810182114"></a>torch.distributed.isend</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1073841182115"><a name="p1073841182115"></a><a name="p1073841182115"></a>否</p>
</td>
</tr>
<tr id="row658483012187"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p75091281455"><a name="p75091281455"></a><a name="p75091281455"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p7738151152113"><a name="p7738151152113"></a><a name="p7738151152113"></a>torch.distributed.irecv</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p107381318210"><a name="p107381318210"></a><a name="p107381318210"></a>否</p>
</td>
</tr>
<tr id="row0584153071814"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p8509928134512"><a name="p8509928134512"></a><a name="p8509928134512"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p187381317218"><a name="p187381317218"></a><a name="p187381317218"></a>is_completed</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1073814116216"><a name="p1073814116216"></a><a name="p1073814116216"></a>是</p>
</td>
</tr>
<tr id="row15584530191812"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p050919284454"><a name="p050919284454"></a><a name="p050919284454"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p117388142114"><a name="p117388142114"></a><a name="p117388142114"></a>wait</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p12738131152120"><a name="p12738131152120"></a><a name="p12738131152120"></a>是</p>
</td>
</tr>
<tr id="row158443071816"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p6510152844518"><a name="p6510152844518"></a><a name="p6510152844518"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p5738181152113"><a name="p5738181152113"></a><a name="p5738181152113"></a>torch.distributed.broadcast</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1373871112118"><a name="p1373871112118"></a><a name="p1373871112118"></a>是</p>
</td>
</tr>
<tr id="row1558493041819"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p11510172844518"><a name="p11510172844518"></a><a name="p11510172844518"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17387117214"><a name="p17387117214"></a><a name="p17387117214"></a>torch.distributed.all_reduce</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p673815110213"><a name="p673815110213"></a><a name="p673815110213"></a>是</p>
</td>
</tr>
<tr id="row15584143041815"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15510828104516"><a name="p15510828104516"></a><a name="p15510828104516"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1473816112217"><a name="p1473816112217"></a><a name="p1473816112217"></a>torch.distributed.reduce</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p873820182120"><a name="p873820182120"></a><a name="p873820182120"></a>否</p>
</td>
</tr>
<tr id="row45841930111817"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1851032884516"><a name="p1851032884516"></a><a name="p1851032884516"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p15738121152119"><a name="p15738121152119"></a><a name="p15738121152119"></a>torch.distributed.all_gather</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p107381118213"><a name="p107381118213"></a><a name="p107381118213"></a>是</p>
</td>
</tr>
<tr id="row14585113017183"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p13510202864513"><a name="p13510202864513"></a><a name="p13510202864513"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p773814112116"><a name="p773814112116"></a><a name="p773814112116"></a>torch.distributed.gather</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1973910172110"><a name="p1973910172110"></a><a name="p1973910172110"></a>否</p>
</td>
</tr>
<tr id="row12585113019181"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p5510628144513"><a name="p5510628144513"></a><a name="p5510628144513"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p873981102115"><a name="p873981102115"></a><a name="p873981102115"></a>torch.distributed.scatter</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p157397142115"><a name="p157397142115"></a><a name="p157397142115"></a>否</p>
</td>
</tr>
<tr id="row9585143001815"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p451012281455"><a name="p451012281455"></a><a name="p451012281455"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p373921132110"><a name="p373921132110"></a><a name="p373921132110"></a>torch.distributed.barrier</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1073910142111"><a name="p1073910142111"></a><a name="p1073910142111"></a>是</p>
</td>
</tr>
<tr id="row1958553015187"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12510132894516"><a name="p12510132894516"></a><a name="p12510132894516"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p17739216213"><a name="p17739216213"></a><a name="p17739216213"></a>torch.distributed.ReduceOp</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p11739141152119"><a name="p11739141152119"></a><a name="p11739141152119"></a>是</p>
</td>
</tr>
<tr id="row1258573016189"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1551052814457"><a name="p1551052814457"></a><a name="p1551052814457"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p19739116211"><a name="p19739116211"></a><a name="p19739116211"></a>torch.distributed.reduce_op</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p873912152118"><a name="p873912152118"></a><a name="p873912152118"></a>是</p>
</td>
</tr>
<tr id="row185851530171817"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1151012286451"><a name="p1151012286451"></a><a name="p1151012286451"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p37395122116"><a name="p37395122116"></a><a name="p37395122116"></a>torch.distributed.broadcast_multigpu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p773911132110"><a name="p773911132110"></a><a name="p773911132110"></a>否</p>
</td>
</tr>
<tr id="row1958563071814"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p1751042820454"><a name="p1751042820454"></a><a name="p1751042820454"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p12739615218"><a name="p12739615218"></a><a name="p12739615218"></a>torch.distributed.all_reduce_multigpu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1773912172113"><a name="p1773912172113"></a><a name="p1773912172113"></a>否</p>
</td>
</tr>
<tr id="row75851630171818"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p12510172814456"><a name="p12510172814456"></a><a name="p12510172814456"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p573918122114"><a name="p573918122114"></a><a name="p573918122114"></a>torch.distributed.reduce_multigpu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p97391716215"><a name="p97391716215"></a><a name="p97391716215"></a>否</p>
</td>
</tr>
<tr id="row165851730191816"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p751032844515"><a name="p751032844515"></a><a name="p751032844515"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p177398115218"><a name="p177398115218"></a><a name="p177398115218"></a>torch.distributed.all_gather_multigpu</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p1873918122117"><a name="p1873918122117"></a><a name="p1873918122117"></a>否</p>
</td>
</tr>
<tr id="row16585630151811"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p9510428144510"><a name="p9510428144510"></a><a name="p9510428144510"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p177399110214"><a name="p177399110214"></a><a name="p177399110214"></a>torch.distributed.launch</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p107407182112"><a name="p107407182112"></a><a name="p107407182112"></a>是</p>
</td>
</tr>
<tr id="row155851630161818"><td class="cellrowborder" valign="top" width="10%" headers="mcps1.1.4.1.1 "><p id="p15510162816454"><a name="p15510162816454"></a><a name="p15510162816454"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="70%" headers="mcps1.1.4.1.2 "><p id="p1740131182110"><a name="p1740131182110"></a><a name="p1740131182110"></a>torch.multiprocessing.spawn</p>
</td>
<td class="cellrowborder" valign="top" width="20%" headers="mcps1.1.4.1.3 "><p id="p074116119214"><a name="p074116119214"></a><a name="p074116119214"></a>是</p>
</td>
</tr>
</tbody>
</table>

<h2 id="NPU和CUDA功能对齐">NPU和CUDA功能对齐</h2>

<a name="table1125623115220"></a>
<table><thead align="left"><tr id="row72566317225"><th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.1.5.1.1"><p id="p15384153811454"><a name="p15384153811454"></a><a name="p15384153811454"></a>序号</p>
</th>
<th class="cellrowborder" valign="top" width="36.99%" id="mcps1.1.5.1.2"><p id="p4981944112214"><a name="p4981944112214"></a><a name="p4981944112214"></a>API名称</p>
</th>
<th class="cellrowborder" valign="top" width="36.99%" id="mcps1.1.5.1.3"><p id="p169812445222"><a name="p169812445222"></a><a name="p169812445222"></a>npu对应API名称</p>
</th>
<th class="cellrowborder" valign="top" width="18.5%" id="mcps1.1.5.1.4"><p id="p1698194412221"><a name="p1698194412221"></a><a name="p1698194412221"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row32560313226"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1345817184613"><a name="p1345817184613"></a><a name="p1345817184613"></a>1</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p141501633192320"><a name="p141501633192320"></a><a name="p141501633192320"></a>torch.cuda.current_blas_handle</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p17150733112310"><a name="p17150733112310"></a><a name="p17150733112310"></a>torch.npu.current_blas_handle</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1812175542317"><a name="p1812175542317"></a><a name="p1812175542317"></a>否</p>
</td>
</tr>
<tr id="row1525693182216"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p345817194616"><a name="p345817194616"></a><a name="p345817194616"></a>2</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p815010337236"><a name="p815010337236"></a><a name="p815010337236"></a>torch.cuda.current_device</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p415043332316"><a name="p415043332316"></a><a name="p415043332316"></a>torch.npu.current_device</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p4128554235"><a name="p4128554235"></a><a name="p4128554235"></a>是</p>
</td>
</tr>
<tr id="row142561231202218"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p845877144620"><a name="p845877144620"></a><a name="p845877144620"></a>3</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p9150203310236"><a name="p9150203310236"></a><a name="p9150203310236"></a>torch.cuda.current_stream</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p315023312316"><a name="p315023312316"></a><a name="p315023312316"></a>torch.npu.current_stream</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p21295515230"><a name="p21295515230"></a><a name="p21295515230"></a>否</p>
</td>
</tr>
<tr id="row15256123115227"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p20458575465"><a name="p20458575465"></a><a name="p20458575465"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p101508337231"><a name="p101508337231"></a><a name="p101508337231"></a>torch.cuda.default_stream</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p3150533152316"><a name="p3150533152316"></a><a name="p3150533152316"></a>torch.npu.default_stream</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1512855152313"><a name="p1512855152313"></a><a name="p1512855152313"></a>是</p>
</td>
</tr>
<tr id="row9257123114223"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1045837114616"><a name="p1045837114616"></a><a name="p1045837114616"></a>5</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p115033352312"><a name="p115033352312"></a><a name="p115033352312"></a>torch.cuda.device</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p5150133315237"><a name="p5150133315237"></a><a name="p5150133315237"></a>torch.npu.device</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p151225516232"><a name="p151225516232"></a><a name="p151225516232"></a>否</p>
</td>
</tr>
<tr id="row12257203152219"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p34581775469"><a name="p34581775469"></a><a name="p34581775469"></a>6</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p715053382316"><a name="p715053382316"></a><a name="p715053382316"></a>torch.cuda.device_count</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p4150143312236"><a name="p4150143312236"></a><a name="p4150143312236"></a>torch.npu.device_count</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1112125552315"><a name="p1112125552315"></a><a name="p1112125552315"></a>是</p>
</td>
</tr>
<tr id="row182571431162217"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p4458197134618"><a name="p4458197134618"></a><a name="p4458197134618"></a>7</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p01504331237"><a name="p01504331237"></a><a name="p01504331237"></a>torch.cuda.device_of</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p141501633182315"><a name="p141501633182315"></a><a name="p141501633182315"></a>torch.npu.device_of</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p141275552318"><a name="p141275552318"></a><a name="p141275552318"></a>否</p>
</td>
</tr>
<tr id="row6257173113221"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p045814713464"><a name="p045814713464"></a><a name="p045814713464"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p91501833152313"><a name="p91501833152313"></a><a name="p91501833152313"></a>torch.cuda.get_device_capability</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p171501433142316"><a name="p171501433142316"></a><a name="p171501433142316"></a>torch.npu.get_device_capability</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p141365511238"><a name="p141365511238"></a><a name="p141365511238"></a>否</p>
</td>
</tr>
<tr id="row225723152216"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p8458577465"><a name="p8458577465"></a><a name="p8458577465"></a>9</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p15150183319231"><a name="p15150183319231"></a><a name="p15150183319231"></a>torch.cuda.get_device_name</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p015063342316"><a name="p015063342316"></a><a name="p015063342316"></a>torch.npu.get_device_name</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p41314559239"><a name="p41314559239"></a><a name="p41314559239"></a>否</p>
</td>
</tr>
<tr id="row6257931152210"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p94582714615"><a name="p94582714615"></a><a name="p94582714615"></a>10</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p13150123392318"><a name="p13150123392318"></a><a name="p13150123392318"></a>torch.cuda.init</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1415012334239"><a name="p1415012334239"></a><a name="p1415012334239"></a>torch.npu.init</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p9131855112316"><a name="p9131855112316"></a><a name="p9131855112316"></a>是</p>
</td>
</tr>
<tr id="row10257143110220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p104581371469"><a name="p104581371469"></a><a name="p104581371469"></a>11</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p18150153392311"><a name="p18150153392311"></a><a name="p18150153392311"></a>torch.cuda.ipc_collect</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p8150203318236"><a name="p8150203318236"></a><a name="p8150203318236"></a>torch.npu.ipc_collect</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p713185516239"><a name="p713185516239"></a><a name="p713185516239"></a>否</p>
</td>
</tr>
<tr id="row225763122217"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1045818714619"><a name="p1045818714619"></a><a name="p1045818714619"></a>12</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p8151833162312"><a name="p8151833162312"></a><a name="p8151833162312"></a>torch.cuda.is_available</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p201511833172312"><a name="p201511833172312"></a><a name="p201511833172312"></a>torch.npu.is_available</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p111312553238"><a name="p111312553238"></a><a name="p111312553238"></a>是</p>
</td>
</tr>
<tr id="row4257133114221"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p445815710463"><a name="p445815710463"></a><a name="p445815710463"></a>13</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p4151033102317"><a name="p4151033102317"></a><a name="p4151033102317"></a>torch.cuda.is_initialized</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p18151533152314"><a name="p18151533152314"></a><a name="p18151533152314"></a>torch.npu.is_initialized</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p151316556239"><a name="p151316556239"></a><a name="p151316556239"></a>是</p>
</td>
</tr>
<tr id="row62581831122218"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p145820718466"><a name="p145820718466"></a><a name="p145820718466"></a>14</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p19151333132317"><a name="p19151333132317"></a><a name="p19151333132317"></a>torch.cuda.set_device</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p41511433112315"><a name="p41511433112315"></a><a name="p41511433112315"></a>torch.npu.set_device</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p2130555233"><a name="p2130555233"></a><a name="p2130555233"></a>部分支持</p>
</td>
</tr>
<tr id="row1258153152215"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1345837174616"><a name="p1345837174616"></a><a name="p1345837174616"></a>15</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p6151103312319"><a name="p6151103312319"></a><a name="p6151103312319"></a>torch.cuda.stream</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1915193322320"><a name="p1915193322320"></a><a name="p1915193322320"></a>torch.npu.stream</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1413455112310"><a name="p1413455112310"></a><a name="p1413455112310"></a>是</p>
</td>
</tr>
<tr id="row14258193162213"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p13458474461"><a name="p13458474461"></a><a name="p13458474461"></a>16</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1715143312235"><a name="p1715143312235"></a><a name="p1715143312235"></a>torch.cuda.synchronize</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p4151133316230"><a name="p4151133316230"></a><a name="p4151133316230"></a>torch.npu.synchronize</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p013145511237"><a name="p013145511237"></a><a name="p013145511237"></a>是</p>
</td>
</tr>
<tr id="row162586317227"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p24581178466"><a name="p24581178466"></a><a name="p24581178466"></a>17</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1515183382317"><a name="p1515183382317"></a><a name="p1515183382317"></a>torch.cuda.get_rng_state</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p14151733142319"><a name="p14151733142319"></a><a name="p14151733142319"></a>torch.npu.get_rng_state</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p91335562317"><a name="p91335562317"></a><a name="p91335562317"></a>否</p>
</td>
</tr>
<tr id="row15258163132214"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p64581177462"><a name="p64581177462"></a><a name="p64581177462"></a>18</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p6151033122317"><a name="p6151033122317"></a><a name="p6151033122317"></a>torch.cuda.get_rng_state_all</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p141511033202316"><a name="p141511033202316"></a><a name="p141511033202316"></a>torch.npu.get_rng_state_all</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p141319550231"><a name="p141319550231"></a><a name="p141319550231"></a>否</p>
</td>
</tr>
<tr id="row1258163110226"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p645811704611"><a name="p645811704611"></a><a name="p645811704611"></a>19</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p18151193311235"><a name="p18151193311235"></a><a name="p18151193311235"></a>torch.cuda.set_rng_state</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p415113337236"><a name="p415113337236"></a><a name="p415113337236"></a>torch.npu.set_rng_state</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p7135559234"><a name="p7135559234"></a><a name="p7135559234"></a>否</p>
</td>
</tr>
<tr id="row192581431202210"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p10458147134612"><a name="p10458147134612"></a><a name="p10458147134612"></a>20</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p315111338230"><a name="p315111338230"></a><a name="p315111338230"></a>torch.cuda.set_rng_state_all</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p14151173392316"><a name="p14151173392316"></a><a name="p14151173392316"></a>torch.npu.set_rng_state_all</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p913955152311"><a name="p913955152311"></a><a name="p913955152311"></a>否</p>
</td>
</tr>
<tr id="row122581314220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p11458473468"><a name="p11458473468"></a><a name="p11458473468"></a>21</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p215118330233"><a name="p215118330233"></a><a name="p215118330233"></a>torch.cuda.manual_seed</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p181513333231"><a name="p181513333231"></a><a name="p181513333231"></a>torch.npu.manual_seed</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p131410550237"><a name="p131410550237"></a><a name="p131410550237"></a>否</p>
</td>
</tr>
<tr id="row425863117226"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p144594784612"><a name="p144594784612"></a><a name="p144594784612"></a>22</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p91513335237"><a name="p91513335237"></a><a name="p91513335237"></a>torch.cuda.manual_seed_all</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p815119339237"><a name="p815119339237"></a><a name="p815119339237"></a>torch.npu.manual_seed_all</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p19144552233"><a name="p19144552233"></a><a name="p19144552233"></a>否</p>
</td>
</tr>
<tr id="row1425814310220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p34594794618"><a name="p34594794618"></a><a name="p34594794618"></a>23</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p715203312313"><a name="p715203312313"></a><a name="p715203312313"></a>torch.cuda.seed</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p17152153382313"><a name="p17152153382313"></a><a name="p17152153382313"></a>torch.npu.seed</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p10141553234"><a name="p10141553234"></a><a name="p10141553234"></a>否</p>
</td>
</tr>
<tr id="row15259123116220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p124592764614"><a name="p124592764614"></a><a name="p124592764614"></a>24</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p151527338234"><a name="p151527338234"></a><a name="p151527338234"></a>torch.cuda.seed_all</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p8152113302319"><a name="p8152113302319"></a><a name="p8152113302319"></a>torch.npu.seed_all</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1514165514237"><a name="p1514165514237"></a><a name="p1514165514237"></a>否</p>
</td>
</tr>
<tr id="row5259143120221"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p645911711460"><a name="p645911711460"></a><a name="p645911711460"></a>25</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1915283332312"><a name="p1915283332312"></a><a name="p1915283332312"></a>torch.cuda.initial_seed</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p51521333202313"><a name="p51521333202313"></a><a name="p51521333202313"></a>torch.npu.initial_seed</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p4141755182320"><a name="p4141755182320"></a><a name="p4141755182320"></a>否</p>
</td>
</tr>
<tr id="row1825963192220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p44595717463"><a name="p44595717463"></a><a name="p44595717463"></a>26</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p12152333122316"><a name="p12152333122316"></a><a name="p12152333122316"></a>torch.cuda.comm.broadcast</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p215215339235"><a name="p215215339235"></a><a name="p215215339235"></a>torch.npu.comm.broadcast</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p13141555182318"><a name="p13141555182318"></a><a name="p13141555182318"></a>否</p>
</td>
</tr>
<tr id="row202591431192219"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1945997154619"><a name="p1945997154619"></a><a name="p1945997154619"></a>27</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p715263372317"><a name="p715263372317"></a><a name="p715263372317"></a>torch.cuda.comm.broadcast_coalesced</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1152333142310"><a name="p1152333142310"></a><a name="p1152333142310"></a>torch.npu.comm.broadcast_coalesced</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p161455516233"><a name="p161455516233"></a><a name="p161455516233"></a>否</p>
</td>
</tr>
<tr id="row1625910312227"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p164591273461"><a name="p164591273461"></a><a name="p164591273461"></a>28</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p10152103311239"><a name="p10152103311239"></a><a name="p10152103311239"></a>torch.cuda.comm.reduce_add</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p181521533122319"><a name="p181521533122319"></a><a name="p181521533122319"></a>torch.npu.comm.reduce_add</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p81495582310"><a name="p81495582310"></a><a name="p81495582310"></a>否</p>
</td>
</tr>
<tr id="row6259133116226"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p5459772464"><a name="p5459772464"></a><a name="p5459772464"></a>29</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p615213312233"><a name="p615213312233"></a><a name="p615213312233"></a>torch.cuda.comm.scatter</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1215219336235"><a name="p1215219336235"></a><a name="p1215219336235"></a>torch.npu.comm.scatter</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p2014455162317"><a name="p2014455162317"></a><a name="p2014455162317"></a>否</p>
</td>
</tr>
<tr id="row125912312221"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1345920718464"><a name="p1345920718464"></a><a name="p1345920718464"></a>30</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1215217337233"><a name="p1215217337233"></a><a name="p1215217337233"></a>torch.cuda.comm.gather</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p715220332235"><a name="p715220332235"></a><a name="p715220332235"></a>torch.npu.comm.gather</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p614155514239"><a name="p614155514239"></a><a name="p614155514239"></a>否</p>
</td>
</tr>
<tr id="row12591314225"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p174593715461"><a name="p174593715461"></a><a name="p174593715461"></a>31</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p3152233132310"><a name="p3152233132310"></a><a name="p3152233132310"></a>torch.cuda.Stream</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p17152133342310"><a name="p17152133342310"></a><a name="p17152133342310"></a>torch.npu.Stream</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p9141556234"><a name="p9141556234"></a><a name="p9141556234"></a>是</p>
</td>
</tr>
<tr id="row19259431162218"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p154591714462"><a name="p154591714462"></a><a name="p154591714462"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p16153133322319"><a name="p16153133322319"></a><a name="p16153133322319"></a>torch.cuda.Stream.query</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p11531333233"><a name="p11531333233"></a><a name="p11531333233"></a>torch.npu.Stream.query</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p51465502318"><a name="p51465502318"></a><a name="p51465502318"></a>否</p>
</td>
</tr>
<tr id="row14260203182216"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p04591715463"><a name="p04591715463"></a><a name="p04591715463"></a>33</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p161531933192314"><a name="p161531933192314"></a><a name="p161531933192314"></a>torch.cuda.Stream.record_event</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p0153193342313"><a name="p0153193342313"></a><a name="p0153193342313"></a>torch.npu.Stream.record_event</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p111445582318"><a name="p111445582318"></a><a name="p111445582318"></a>是</p>
</td>
</tr>
<tr id="row15260133117222"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p44596718460"><a name="p44596718460"></a><a name="p44596718460"></a>34</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p131531033132319"><a name="p131531033132319"></a><a name="p131531033132319"></a>torch.cuda.Stream.synchronize</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p17153533182313"><a name="p17153533182313"></a><a name="p17153533182313"></a>torch.npu.Stream.synchronize</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1014195510231"><a name="p1014195510231"></a><a name="p1014195510231"></a>是</p>
</td>
</tr>
<tr id="row2260631192216"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p245947164613"><a name="p245947164613"></a><a name="p245947164613"></a>35</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p11531233132314"><a name="p11531233132314"></a><a name="p11531233132314"></a>torch.cuda.Stream.wait_event</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1915323312316"><a name="p1915323312316"></a><a name="p1915323312316"></a>torch.npu.Stream.wait_event</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p151475552311"><a name="p151475552311"></a><a name="p151475552311"></a>是</p>
</td>
</tr>
<tr id="row226012313225"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1045916718464"><a name="p1045916718464"></a><a name="p1045916718464"></a>36</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p81541733122313"><a name="p81541733122313"></a><a name="p81541733122313"></a>torch.cuda.Stream.wait_stream</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p18154113318230"><a name="p18154113318230"></a><a name="p18154113318230"></a>torch.npu.Stream.wait_stream</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p111414558234"><a name="p111414558234"></a><a name="p111414558234"></a>是</p>
</td>
</tr>
<tr id="row1626003119223"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p124593712464"><a name="p124593712464"></a><a name="p124593712464"></a>37</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p5154833102318"><a name="p5154833102318"></a><a name="p5154833102318"></a>torch.cuda.Event</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p17154123342311"><a name="p17154123342311"></a><a name="p17154123342311"></a>torch.npu.Event</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p61514557238"><a name="p61514557238"></a><a name="p61514557238"></a>是</p>
</td>
</tr>
<tr id="row16260203113228"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1145916734616"><a name="p1145916734616"></a><a name="p1145916734616"></a>38</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p51540332230"><a name="p51540332230"></a><a name="p51540332230"></a>torch.cuda.Event.elapsed_time</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p181541033112312"><a name="p181541033112312"></a><a name="p181541033112312"></a>torch.npu.Event.elapsed_time</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1515195592316"><a name="p1515195592316"></a><a name="p1515195592316"></a>是</p>
</td>
</tr>
<tr id="row72601931192214"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1545977154618"><a name="p1545977154618"></a><a name="p1545977154618"></a>39</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1815411334239"><a name="p1815411334239"></a><a name="p1815411334239"></a>torch.cuda.Event.from_ipc_handle</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p131545335232"><a name="p131545335232"></a><a name="p131545335232"></a>torch.npu.Event.from_ipc_handle</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p141575520235"><a name="p141575520235"></a><a name="p141575520235"></a>否</p>
</td>
</tr>
<tr id="row112601731102216"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1445977164619"><a name="p1445977164619"></a><a name="p1445977164619"></a>40</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p8154173319237"><a name="p8154173319237"></a><a name="p8154173319237"></a>torch.cuda.Event.ipc_handle</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1154433142317"><a name="p1154433142317"></a><a name="p1154433142317"></a>torch.npu.Event.ipc_handle</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p3159559237"><a name="p3159559237"></a><a name="p3159559237"></a>否</p>
</td>
</tr>
<tr id="row1226093182217"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p9460170461"><a name="p9460170461"></a><a name="p9460170461"></a>41</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1015413331237"><a name="p1015413331237"></a><a name="p1015413331237"></a>torch.cuda.Event.query</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p15154933152310"><a name="p15154933152310"></a><a name="p15154933152310"></a>torch.npu.Event.query</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p181565517238"><a name="p181565517238"></a><a name="p181565517238"></a>是</p>
</td>
</tr>
<tr id="row19260123142213"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p0460157184617"><a name="p0460157184617"></a><a name="p0460157184617"></a>42</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1615413338234"><a name="p1615413338234"></a><a name="p1615413338234"></a>torch.cuda.Event.record</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p15154143302313"><a name="p15154143302313"></a><a name="p15154143302313"></a>torch.npu.Event.record</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p01518558230"><a name="p01518558230"></a><a name="p01518558230"></a>是</p>
</td>
</tr>
<tr id="row726116318228"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p13460197144619"><a name="p13460197144619"></a><a name="p13460197144619"></a>43</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p11154113332317"><a name="p11154113332317"></a><a name="p11154113332317"></a>torch.cuda.Event.synchronize</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p151541233152312"><a name="p151541233152312"></a><a name="p151541233152312"></a>torch.npu.Event.synchronize</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p115955122311"><a name="p115955122311"></a><a name="p115955122311"></a>是</p>
</td>
</tr>
<tr id="row16261203112227"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p946013764614"><a name="p946013764614"></a><a name="p946013764614"></a>44</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p191541033122318"><a name="p191541033122318"></a><a name="p191541033122318"></a>torch.cuda.Event.wait</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p6154143316232"><a name="p6154143316232"></a><a name="p6154143316232"></a>torch.npu.Event.wait</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1151755112311"><a name="p1151755112311"></a><a name="p1151755112311"></a>是</p>
</td>
</tr>
<tr id="row3261031202219"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p046019774610"><a name="p046019774610"></a><a name="p046019774610"></a>45</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p21549334232"><a name="p21549334232"></a><a name="p21549334232"></a>torch.cuda.empty_cache</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p151540332232"><a name="p151540332232"></a><a name="p151540332232"></a>torch.npu.empty_cache</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p13157558234"><a name="p13157558234"></a><a name="p13157558234"></a>是</p>
</td>
</tr>
<tr id="row182611431132218"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p16460137194613"><a name="p16460137194613"></a><a name="p16460137194613"></a>46</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1415413336232"><a name="p1415413336232"></a><a name="p1415413336232"></a>torch.cuda.memory_stats</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p17154173319238"><a name="p17154173319238"></a><a name="p17154173319238"></a>torch.npu.memory_stats</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1715655192311"><a name="p1715655192311"></a><a name="p1715655192311"></a>是</p>
</td>
</tr>
<tr id="row42611331122220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p746017164620"><a name="p746017164620"></a><a name="p746017164620"></a>47</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1315419331231"><a name="p1315419331231"></a><a name="p1315419331231"></a>torch.cuda.memory_summary</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p5155133316231"><a name="p5155133316231"></a><a name="p5155133316231"></a>torch.npu.memory_summary</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p1115855132315"><a name="p1115855132315"></a><a name="p1115855132315"></a>是</p>
</td>
</tr>
<tr id="row226133132212"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p946097174613"><a name="p946097174613"></a><a name="p946097174613"></a>48</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p19155183372314"><a name="p19155183372314"></a><a name="p19155183372314"></a>torch.cuda.memory_snapshot</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1155103310239"><a name="p1155103310239"></a><a name="p1155103310239"></a>torch.npu.memory_snapshot</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p2015355172319"><a name="p2015355172319"></a><a name="p2015355172319"></a>是</p>
</td>
</tr>
<tr id="row2261133112229"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1146017720469"><a name="p1146017720469"></a><a name="p1146017720469"></a>49</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p7155833122311"><a name="p7155833122311"></a><a name="p7155833122311"></a>torch.cuda.memory_allocated</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p6155183312318"><a name="p6155183312318"></a><a name="p6155183312318"></a>torch.npu.memory_allocated</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p61512556238"><a name="p61512556238"></a><a name="p61512556238"></a>是</p>
</td>
</tr>
<tr id="row1726113142220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p4460472465"><a name="p4460472465"></a><a name="p4460472465"></a>50</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p11155173319232"><a name="p11155173319232"></a><a name="p11155173319232"></a>torch.cuda.max_memory_allocated</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p9155733142313"><a name="p9155733142313"></a><a name="p9155733142313"></a>torch.npu.max_memory_allocated</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p151585516235"><a name="p151585516235"></a><a name="p151585516235"></a>是</p>
</td>
</tr>
<tr id="row172619313226"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p184608764618"><a name="p184608764618"></a><a name="p184608764618"></a>51</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p12155433182317"><a name="p12155433182317"></a><a name="p12155433182317"></a>torch.cuda.reset_max_memory_allocated</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p12155633142317"><a name="p12155633142317"></a><a name="p12155633142317"></a>torch.npu.reset_max_memory_allocated</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p19151155162311"><a name="p19151155162311"></a><a name="p19151155162311"></a>是</p>
</td>
</tr>
<tr id="row162615314222"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1146012784610"><a name="p1146012784610"></a><a name="p1146012784610"></a>52</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p215518330239"><a name="p215518330239"></a><a name="p215518330239"></a>torch.cuda.memory_reserved</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p3155143362310"><a name="p3155143362310"></a><a name="p3155143362310"></a>torch.npu.memory_reserved</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p41595512311"><a name="p41595512311"></a><a name="p41595512311"></a>是</p>
</td>
</tr>
<tr id="row1262183172219"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p0460271468"><a name="p0460271468"></a><a name="p0460271468"></a>53</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p15155153318234"><a name="p15155153318234"></a><a name="p15155153318234"></a>torch.cuda.max_memory_reserved</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p17155123392319"><a name="p17155123392319"></a><a name="p17155123392319"></a>torch.npu.max_memory_reserved</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p11159559235"><a name="p11159559235"></a><a name="p11159559235"></a>是</p>
</td>
</tr>
<tr id="row1026215311225"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p64601673465"><a name="p64601673465"></a><a name="p64601673465"></a>54</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p181551533142312"><a name="p181551533142312"></a><a name="p181551533142312"></a>torch.cuda.memory_cached</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p815511337231"><a name="p815511337231"></a><a name="p815511337231"></a>torch.npu.memory_cached</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p11505562316"><a name="p11505562316"></a><a name="p11505562316"></a>是</p>
</td>
</tr>
<tr id="row112621631202218"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1746077124616"><a name="p1746077124616"></a><a name="p1746077124616"></a>55</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p71551833102319"><a name="p71551833102319"></a><a name="p71551833102319"></a>torch.cuda.max_memory_cached</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p201554339239"><a name="p201554339239"></a><a name="p201554339239"></a>torch.npu.max_memory_cached</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p171513552230"><a name="p171513552230"></a><a name="p171513552230"></a>是</p>
</td>
</tr>
<tr id="row19262103117223"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p12460127194617"><a name="p12460127194617"></a><a name="p12460127194617"></a>56</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p11155133317239"><a name="p11155133317239"></a><a name="p11155133317239"></a>torch.cuda.reset_max_memory_cached</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p71554338237"><a name="p71554338237"></a><a name="p71554338237"></a>torch.npu.reset_max_memory_cached</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p015155542313"><a name="p015155542313"></a><a name="p015155542313"></a>是</p>
</td>
</tr>
<tr id="row13262103115220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p15460137194610"><a name="p15460137194610"></a><a name="p15460137194610"></a>57</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p9155233102318"><a name="p9155233102318"></a><a name="p9155233102318"></a>torch.cuda.nvtx.mark</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p5155333152312"><a name="p5155333152312"></a><a name="p5155333152312"></a>torch.npu.nvtx.mark</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p11611558231"><a name="p11611558231"></a><a name="p11611558231"></a>否</p>
</td>
</tr>
<tr id="row3262193122220"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p10460107134616"><a name="p10460107134616"></a><a name="p10460107134616"></a>58</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p6155173319233"><a name="p6155173319233"></a><a name="p6155173319233"></a>torch.cuda.nvtx.range_push</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p3156173332319"><a name="p3156173332319"></a><a name="p3156173332319"></a>torch.npu.nvtx.range_push</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p7161855192311"><a name="p7161855192311"></a><a name="p7161855192311"></a>否</p>
</td>
</tr>
<tr id="row2262231202212"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p64609718468"><a name="p64609718468"></a><a name="p64609718468"></a>59</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p13156153312310"><a name="p13156153312310"></a><a name="p13156153312310"></a>torch.cuda.nvtx.range_pop</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p14156333162315"><a name="p14156333162315"></a><a name="p14156333162315"></a>torch.npu.nvtx.range_pop</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p7163558235"><a name="p7163558235"></a><a name="p7163558235"></a>否</p>
</td>
</tr>
<tr id="row20262173152212"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p4460147114614"><a name="p4460147114614"></a><a name="p4460147114614"></a>60</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p6156173312312"><a name="p6156173312312"></a><a name="p6156173312312"></a>torch.cuda._sleep</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p215603362313"><a name="p215603362313"></a><a name="p215603362313"></a>torch.npu._sleep</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p181612550234"><a name="p181612550234"></a><a name="p181612550234"></a>否</p>
</td>
</tr>
<tr id="row726216318223"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p3460127184614"><a name="p3460127184614"></a><a name="p3460127184614"></a>61</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p121561233122313"><a name="p121561233122313"></a><a name="p121561233122313"></a>torch.cuda.Stream.priority_range</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1615643332318"><a name="p1615643332318"></a><a name="p1615643332318"></a>torch.npu.Stream.priority_range</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p131685514235"><a name="p131685514235"></a><a name="p131685514235"></a>否</p>
</td>
</tr>
<tr id="row9263103182219"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p2461157164610"><a name="p2461157164610"></a><a name="p2461157164610"></a>62</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p1215616339233"><a name="p1215616339233"></a><a name="p1215616339233"></a>torch.cuda.get_device_properties</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p20156833152314"><a name="p20156833152314"></a><a name="p20156833152314"></a>torch.npu.get_device_properties</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p416105516232"><a name="p416105516232"></a><a name="p416105516232"></a>否</p>
</td>
</tr>
<tr id="row72631131192217"><td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.1.5.1.1 "><p id="p1446117784610"><a name="p1446117784610"></a><a name="p1446117784610"></a>63</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.2 "><p id="p121561133162318"><a name="p121561133162318"></a><a name="p121561133162318"></a>torch.cuda.amp.GradScaler</p>
</td>
<td class="cellrowborder" valign="top" width="36.99%" headers="mcps1.1.5.1.3 "><p id="p1415673311237"><a name="p1415673311237"></a><a name="p1415673311237"></a>torch.npu.amp.GradScaler</p>
</td>
<td class="cellrowborder" valign="top" width="18.5%" headers="mcps1.1.5.1.4 "><p id="p71675582320"><a name="p71675582320"></a><a name="p71675582320"></a>否</p>
</td>
</tr>
</tbody>
</table>

>![](public_sys-resources/icon-note.gif) **说明：** 
>torch.npu.set\_device\(\)接口只支持在程序开始的位置通过set\_device进行指定，不支持多次指定和with torch.npu.device\(id\)方式的device切换

