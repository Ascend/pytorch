# PyTorch Installation Guide
-   [Overview](#overviewmd)
-   [Manual Build and Installation](#manual-build-and-installationmd)
    -   [Prerequisites](#prerequisitesmd)
    -   [Installing the PyTorch Framework](#installing-the-pytorch-frameworkmd)
    -   [Configuring Environment Variables](#configuring-environment-variablesmd)
    -   [Installing the Mixed Precision Module](#installing-the-mixed-precision-modulemd)
-   [References](#referencesmd)
    -   [Installing CMake](#installing-cmakemd)
    -   [How Do I Install GCC 7.3.0?](#how-do-i-install-gcc-7-3-0md)
    -   [What to Do If "torch 1.5.0xxxx" and "torchvision" Do Not Match When torch-\*.whl Is Installed?](#what-to-do-if-torch-1-5-0xxxx-and-torchvision-do-not-match-when-torch--whl-is-installedmd)
<h2 id="overviewmd">Overview</h2>

When setting up the environment for PyTorch model development and running, you can manually build and install the modules adapted to the PyTorch framework on a server.

**Figure  1**  Environment setup process<a name="en-us_topic_0000001119176876_fig1938918396117"></a>  


![](figures/210926103326800.png)

<h2 id="manual-build-and-installationmd">Manual Build and Installation</h2>

-   **[Prerequisites](#prerequisitesmd)**  

-   **[Installing the PyTorch Framework](#installing-the-pytorch-frameworkmd)**  

-   **[Configuring Environment Variables](#configuring-environment-variablesmd)**  

-   **[Installing the Mixed Precision Module](#installing-the-mixed-precision-modulemd)**  


<h3 id="prerequisitesmd">Prerequisites</h3>

#### Prerequisites<a name="en-us_topic_0000001105856382_en-us_topic_0275872734_section108914373254"></a>

-   The development or operating environment of CANN has been installed. For details, see the _CANN Software Installation Guide_.
-   CMake 3.12.0 or later has been installed. For details about how to install CMake, see  [Installing CMake](#installing-cmakemd).
-   GCC 7.3.0 or later has been installed. For details about how to install and use GCC 7.3.0, see  [How Do I Install GCC 7.3.0?](#how-do-i-install-gcc-7-3-0md).
-   Python 3.7.5 or 3.8 has been installed.
-   Note that PyTorch 1.5 does not support Python 3.9 build and installation.
-   The Patch and Git tools have been installed in the environment. To install the tools for Ubuntu and CentOS, run the following commands:
    -   Ubuntu

        ```
        apt-get install patch
        apt-get install git
        ```

    -   CentOS

        ```
        yum install patch
        yum install git
        ```



<h3 id="installing-the-pytorch-frameworkmd">Installing the PyTorch Framework</h3>

#### Installation Process<a name="en-us_topic_0000001152776301_section1611810384557"></a>

1.  Log in to the server as the  **root**  user or a non-root user.
2.  Run the following commands in sequence to install the PyTorch dependencies.

    If you install Python and its dependencies as a non-root user, add  **--user**  at the end of each command in this step. Example command:  **pip3.7 install pyyaml --user**.

    ```
    pip3 install pyyaml
    pip3 install wheel
    ```

3.  Obtain the PyTorch source code.

    1.  Run the following command to obtain the PyTorch source code adapted to Ascend AI Processors and switch to the required branch:

        ```
        git clone https://gitee.com/ascend/pytorch.git
        # By default, the master branch is used. If other branches are required, run the git checkout command to switch to that branch.
        # git checkout -b v1.5.0 remotes/origin/v1.5.0

        ```

        The directory structure of the downloaded source code is as follows:

        ```
        ├── patch                            # Directory of the patch adapted to Ascend AI Processors
        │   ├── pytorch1.5.0_npu.patch      #  Patch for PyTorch 1.5.0
        ├── pytorch1.5.0                     # Source code and test directory for PyTorch 1.5.0
        │   ├── access_control_test.py
        │   ├── src                         # Source code directory
        │   └── test                        # Directory for storing test cases
        └── scripts                          # Build and create a directory.
        ```

    2.  Obtain the native PyTorch source code from the root directory  **/pytorch**  of the current repository.

            ```
            git clone -b v1.5.0 --depth=1 https://github.com/pytorch/pytorch.git
            ```

    3.  Go to the native PyTorch code directory  **pytorch**  and obtain the PyTorch passive dependency code.

        ```
        cd  pytorch
        git submodule sync
        git submodule update --init --recursive
        ```

    >![](public_sys-resources/icon-note.gif) **NOTE:** 
    >Due to network fluctuation, it may take a long time to obtain the source code. If no error is reported after the download is complete, the PyTorch and third-party code on which PyTorch depends are generated.

4.  Generate the PyTorch installation package adapted to Ascend AI Processors.
    1.  Go to the  **pytorch/scripts**  directory and run the conversion script to generate full code adapted to Ascend AI Processors.

        ```
        cd ../scripts
        # If PyTorch 1.5.0 is installed:
        bash gen.sh

        The full code adapted to Ascend AI Processors is generated in the  **pytorch/pytorch**  directory.

    2.  Go to the  **pytorch/pytorch/**  directory and install the dependency library.

        ```
        cd ../pytorch
        pip3 install -r requirements.txt
        ```

    3.  Build and generate the binary installation package of PyTorch.

        ```
        bash build.sh --python=3.7
        or
        bash build.sh --python=3.8
        or
        bash build.sh --python=3.9    # PyTorch 1.5 does not support build and installation using Python 3.9.

        ```

        Specify the Python version in the environment for build. The generated binary package is stored in the current dist directory  **pytorch/pytorch/dist**.

5.  <a name="en-us_topic_0000001152776301_li49671667141"></a>Install PyTorch.

    Go to the  **pytorch/pytorch/dist**  directory and run the following command to install PyTorch:

    ```
    cd dist/
    pip3 install --upgrade torch-1.5.0+ascend.post5-cp37-cp37m-linux_{arch}.whl
    ```

    _\{**arch\}**_  indicates the architecture information. The value can be  **aarch64**  or  **x86\_64**.

    >![](public_sys-resources/icon-note.gif) **NOTE:** 
    >To upgrade PyTorch in the environment, uninstall the PyTorch software package installed in the environment and then perform  [Step 5 Install PyTorch](#en-us_topic_0000001152776301_li49671667141). Run the following command to check whether PyTorch has been installed:
    >**pip3 list | grep torch**


<h3 id="configuring-environment-variablesmd">Configuring Environment Variables</h3>

After the software packages are installed, configure environment variables to use Ascend PyTorch. For details about the environment variables, see  [Table 1](#en-us_topic_0000001152616261_table42017516135).

1.  Configure the operating environment variables by running the following command in the  **root**  directory of the PyTorch source code adapted to Ascend AI Processors:

    ```
    source pytorch/env.sh
    ```

2.  Select a proper HCCL initialization mode based on the actual scenario and configure the corresponding environment variables.

    ```
    # Scenario 1: Single-node scenario
    export HCCL_WHITELIST_DISABLE=1 # Disable the HCCL trustlist.
    # Scenario 2: Multi-node scenario
    export HCCL_WHITELIST_DISABLE=1 # Disable the HCCL trustlist.
    export HCCL_IF_IP="1.1.1.1"  # Replace 1.1.1.1 with the actual NIC IP address of the host. Ensure that the NIC IP addresses in use can communicate with each other in the cluster.
    ```

3.  \(Optional\) Configure function or performance environment variables in the NPU scenario. The variables are disabled by default.

    ```
    export DYNAMIC_COMPILE_ENABLE=1  # (Optional) Dynamic shape feature, which is used when the shape changes. To enable this function, set the value to 1.
    export COMBINED_ENABLE=1 # (Optional) Optimizes the scenario where two inconsecutive operators are combined. To enable this function, set the value to 1.
    export TRI_COMBINED_ENABLE=1 # (Optional) Optimizes the scenario where three inconsecutive operators are combined. To enable this function, set the value to 1.
    export ACL_DUMP_DATA=1 # (Optional) Operator data dump function, which is used for debugging. To enable this function, set the value to 1.
    export DYNAMIC_OP="ADD#MUL" #  (Optional) Operator implementation. The performance of the ADD and MUL operators varies depending on the scenarios.
    ```

4.  \(Optional\) If the system is openEuler or its inherited OS, such as UOS, run the following command to cancel CPU core binding.

    ```
    # unset GOMP_CPU_AFFINITY
    ```


**Table  1**  Description of environment variables

<a name="en-us_topic_0000001152616261_table42017516135"></a>
<table><thead align="left"><tr id="en-us_topic_0000001152616261_row16198951191317"><th class="cellrowborder" valign="top" width="55.48%" id="mcps1.2.3.1.1"><p id="en-us_topic_0000001152616261_p51981251161315"><a name="en-us_topic_0000001152616261_p51981251161315"></a><a name="en-us_topic_0000001152616261_p51981251161315"></a>Environment Variable</p>
</th>
<th class="cellrowborder" valign="top" width="44.519999999999996%" id="mcps1.2.3.1.2"><p id="en-us_topic_0000001152616261_p9198135114133"><a name="en-us_topic_0000001152616261_p9198135114133"></a><a name="en-us_topic_0000001152616261_p9198135114133"></a>Description</p>
</th>
</tr>
</thead>
<tbody><tr id="en-us_topic_0000001152616261_row6882121917329"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p688241953218"><a name="en-us_topic_0000001152616261_p688241953218"></a><a name="en-us_topic_0000001152616261_p688241953218"></a>LD_LIBRARY_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p1888291915322"><a name="en-us_topic_0000001152616261_p1888291915322"></a><a name="en-us_topic_0000001152616261_p1888291915322"></a>Dynamic library search path. Set this variable based on the preceding example.</p>
<p id="p1292181892120"><a name="p1292181892120"></a><a name="p1292181892120"></a>If you need to upgrade GCC in OSs such as CentOS, Debian, and BC-Linux, add <strong id="b6163826603"><a name="b6163826603"></a><a name="b6163826603"></a><em id="i161631926505"><a name="i161631926505"></a><a name="i161631926505"></a>${install_path}</em>/lib64</strong> to the <span class="parmname" id="parmname161637265015"><a name="parmname161637265015"></a><a name="parmname161637265015"></a><b>LD_LIBRARY_PATH</b></span> variable of the dynamic library search path. Replace <em id="i01649261704"><a name="i01649261704"></a><a name="i01649261704"></a><strong id="b1216317261109"><a name="b1216317261109"></a><a name="b1216317261109"></a>{install_path}</strong></em> with the GCC installation path. For details, see <a href="#how-do-i-install-gcc-7-3-0md#en-us_topic_0000001135347812_en-us_topic_0000001173199577_en-us_topic_0000001172534867_en-us_topic_0276688294_li9745165315131">5</a>.</p>
</td>
</tr>
<tr id="en-us_topic_0000001152616261_row16194175523010"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p16195185523019"><a name="en-us_topic_0000001152616261_p16195185523019"></a><a name="en-us_topic_0000001152616261_p16195185523019"></a>PYTHONPATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p19637083322"><a name="en-us_topic_0000001152616261_p19637083322"></a><a name="en-us_topic_0000001152616261_p19637083322"></a>Python search path. Set this variable based on the preceding example.</p>
</td>
</tr>
<tr id="en-us_topic_0000001152616261_row2954102119329"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p195452113218"><a name="en-us_topic_0000001152616261_p195452113218"></a><a name="en-us_topic_0000001152616261_p195452113218"></a>PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p964914893211"><a name="en-us_topic_0000001152616261_p964914893211"></a><a name="en-us_topic_0000001152616261_p964914893211"></a>Executable program search path. Set this variable based on the preceding example.</p>
</td>
</tr>
<tr id="en-us_topic_0000001152616261_row58592816294"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p1886016892913"><a name="en-us_topic_0000001152616261_p1886016892913"></a><a name="en-us_topic_0000001152616261_p1886016892913"></a>ASCEND_OPP_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p28608892915"><a name="en-us_topic_0000001152616261_p28608892915"></a><a name="en-us_topic_0000001152616261_p28608892915"></a>Operator package (OPP) root directory. Set this variable based on the preceding example.</p>
</td>
</tr>
<tr id="en-us_topic_0000001152616261_row144592037903"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p104601373014"><a name="en-us_topic_0000001152616261_p104601373014"></a><a name="en-us_topic_0000001152616261_p104601373014"></a>OPTION_EXEC_EXTERN_PLUGIN_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p1046013716017"><a name="en-us_topic_0000001152616261_p1046013716017"></a><a name="en-us_topic_0000001152616261_p1046013716017"></a>Path of the operator information library.</p>
</td>
</tr>
<tr id="en-us_topic_0000001152616261_row16184379493"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p131851873492"><a name="en-us_topic_0000001152616261_p131851873492"></a><a name="en-us_topic_0000001152616261_p131851873492"></a>ASCEND_AICPU_PATH</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p181851575497"><a name="en-us_topic_0000001152616261_p181851575497"></a><a name="en-us_topic_0000001152616261_p181851575497"></a>Path of the AI CPU operator package.</p>
</td>
</tr>
<tr id="en-us_topic_0000001152616261_row1680820246202"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p4809112415207"><a name="en-us_topic_0000001152616261_p4809112415207"></a><a name="en-us_topic_0000001152616261_p4809112415207"></a>HCCL_WHITELIST_DISABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p952814428206"><a name="en-us_topic_0000001152616261_p952814428206"></a><a name="en-us_topic_0000001152616261_p952814428206"></a>Whether to enable the communication trustlist when the HCCL is used.</p>
<a name="ul928845132310"></a><a name="ul928845132310"></a><ul id="ul928845132310"><li><strong id="b12793231525"><a name="b12793231525"></a><a name="b12793231525"></a>0</strong>: enable the trustlist. The HCCL communication trustlist does not need to be verified.</li><li><strong id="b1146142619212"><a name="b1146142619212"></a><a name="b1146142619212"></a>1</strong>: disable the trustlist. The HCCL communication trustlist needs to be verified.</li></ul>
<p id="en-us_topic_0000001152616261_p5809162416201"><a name="en-us_topic_0000001152616261_p5809162416201"></a><a name="en-us_topic_0000001152616261_p5809162416201"></a>The default value is <strong id="en-us_topic_0000001152616261_b1270332516435"><a name="en-us_topic_0000001152616261_b1270332516435"></a><a name="en-us_topic_0000001152616261_b1270332516435"></a>0</strong>, indicating that the trustlist is enabled by default.</p>
</td>
</tr>
<tr id="en-us_topic_0000001152616261_row0671137162115"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p4671203792114"><a name="en-us_topic_0000001152616261_p4671203792114"></a><a name="en-us_topic_0000001152616261_p4671203792114"></a>HCCL_IF_IP</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p1822165982114"><a name="en-us_topic_0000001152616261_p1822165982114"></a><a name="en-us_topic_0000001152616261_p1822165982114"></a>IP address of the NIC for initializing communication in the HCCL.</p>
<a name="ul2676102292415"></a><a name="ul2676102292415"></a><ul id="ul2676102292415"><li>The IP address is in dotted decimal notation.</li><li>Currently, only the host NIC is supported.</li></ul>
<p id="en-us_topic_0000001152616261_p1167163719217"><a name="en-us_topic_0000001152616261_p1167163719217"></a><a name="en-us_topic_0000001152616261_p1167163719217"></a>By default, the host communication NICs are selected in the following sequence: NICs other than Docker/local NICs (in ascending alphabetical order of NIC names) &gt; Docker NICs &gt; local NICs.</p>
</td>
</tr>
<tr id="en-us_topic_0000001152616261_row1371356152313"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p15106173103120"><a name="p15106173103120"></a><a name="p15106173103120"></a>PTCOPY_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p71055317311"><a name="p71055317311"></a><a name="p71055317311"></a>Use the PTCopy operator mode to accelerate the operation of converting non-contiguous tensors to contiguous tensors and copy process. You are advised to set the value to <strong id="b114621515513"><a name="b114621515513"></a><a name="b114621515513"></a>1</strong> to enable this function. </p>
</td>
</tr>
<tr id="row743212132309"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p17433111312307"><a name="p17433111312307"></a><a name="p17433111312307"></a>ASCEND_SLOG_PRINT_TO_STDOUT</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p6433151393018"><a name="p6433151393018"></a><a name="p6433151393018"></a>(Optional) Whether to enable the log printing function.</p>
<a name="ul760201917473"></a><a name="ul760201917473"></a><ul id="ul760201917473"><li><strong id="b1955232216215"><a name="b1955232216215"></a><a name="b1955232216215"></a>0</strong>: uses the default log output mode.</li><li><strong id="b22913272211"><a name="b22913272211"></a><a name="b22913272211"></a>1</strong>: prints the logs to the screen.</li><li>Other values: invalid</li></ul>
</td>
</tr>
<tr id="row19237171814300"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p14238161893019"><a name="p14238161893019"></a><a name="p14238161893019"></a>ASCEND_GLOBAL_LOG_LEVEL</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p223841810303"><a name="p223841810303"></a><a name="p223841810303"></a>Sets the global log level of app logs.</p>
<a name="ul175714586453"></a><a name="ul175714586453"></a><ul id="ul175714586453"><li><strong id="b94698561835"><a name="b94698561835"></a><a name="b94698561835"></a>0</strong>: DEBUG</li><li><strong id="b1612516597314"><a name="b1612516597314"></a><a name="b1612516597314"></a>1</strong>: INFO</li><li><strong id="b82051911547"><a name="b82051911547"></a><a name="b82051911547"></a>2</strong>: WARNING</li><li><strong id="b04351021945"><a name="b04351021945"></a><a name="b04351021945"></a>3</strong>: ERROR</li><li><strong id="b116153316418"><a name="b116153316418"></a><a name="b116153316418"></a>4</strong>: NULL (no log output)</li><li>Other values: invalid</li></ul>
</td>
</tr>
<tr id="row1348192313303"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p1734815235305"><a name="p1734815235305"></a><a name="p1734815235305"></a>ASCEND_GLOBAL_EVENT_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p12348202373018"><a name="p12348202373018"></a><a name="p12348202373018"></a>Whether to enable event logging for apps.</p>
<a name="ul416352114610"></a><a name="ul416352114610"></a><ul id="ul416352114610"><li><strong id="b425610311940"><a name="b425610311940"></a><a name="b425610311940"></a>0</strong>: disabled</li><li><strong id="b198984321741"><a name="b198984321741"></a><a name="b198984321741"></a>1</strong>: enabled</li><li>Other values: invalid</li></ul>
</td>
</tr>
<tr id="row17878184693015"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p1878194683016"><a name="p1878194683016"></a><a name="p1878194683016"></a>DYNAMIC_COMPILE_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p1887894620304"><a name="p1887894620304"></a><a name="p1887894620304"></a>(Optional) Dynamic shape feature, which is used when the shape changes. To enable this function, set the value to <strong id="b11281447194117"><a name="b11281447194117"></a><a name="b11281447194117"></a>1</strong>.</p>
</td>
</tr>
<tr id="row78312162301"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p1832171673019"><a name="p1832171673019"></a><a name="p1832171673019"></a>COMBINED_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p583261643014"><a name="p583261643014"></a><a name="p583261643014"></a>(Optional) Optimizes the scenario where two inconsecutive operators are combined. To enable this function, set the value to <strong id="b19824817612"><a name="b19824817612"></a><a name="b19824817612"></a>1</strong>.</p>
</td>
</tr>
<tr id="row17630155212342"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p66309527341"><a name="p66309527341"></a><a name="p66309527341"></a>RI_COMBINED_ENABLE</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p19630185220345"><a name="p19630185220345"></a><a name="p19630185220345"></a>(Optional) Optimizes the scenario where three inconsecutive operators are combined. To enable this function, set the value to <strong id="b10522121418612"><a name="b10522121418612"></a><a name="b10522121418612"></a>1</strong>.</p>
</td>
</tr>
<tr id="row183041355123411"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p730435533415"><a name="p730435533415"></a><a name="p730435533415"></a>ACL_DUMP_DATA</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p16304105533412"><a name="p16304105533412"></a><a name="p16304105533412"></a>(Optional) Operator data dump function, which is used for debugging. To enable this function, set the value to <strong id="b1861154419918"><a name="b1861154419918"></a><a name="b1861154419918"></a>1</strong>.</p>
</td>
</tr>
<tr id="row27481914203518"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="p674813144357"><a name="p674813144357"></a><a name="p674813144357"></a>DYNAMIC_OP</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="p974891414353"><a name="p974891414353"></a><a name="p974891414353"></a>(Optional) Operator implementation. The performance of the ADD and MUL operators varies depending on the scenarios. This parameter is not set by default.</p>
</td>
</tr>
<tr id="row19173161510309"><td class="cellrowborder" valign="top" width="55.48%" headers="mcps1.2.3.1.1 "><p id="en-us_topic_0000001152616261_p16711563237"><a name="en-us_topic_0000001152616261_p16711563237"></a><a name="en-us_topic_0000001152616261_p16711563237"></a>unset GOMP_CPU_AFFINITY</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.3.1.2 "><p id="en-us_topic_0000001152616261_p0711356152317"><a name="en-us_topic_0000001152616261_p0711356152317"></a><a name="en-us_topic_0000001152616261_p0711356152317"></a>(Optional) If the system is openEuler or its inherited OS, such as UOS, run this command to cancel CPU core binding.</p>
</td>
</tr>
</tbody>
</table>

<h3 id="installing-the-mixed-precision-modulemd">Installing the Mixed Precision Module</h3>

#### Prerequisites<a name="en-us_topic_0000001106176190_section3225481020"></a>

1.  Ensure that the PyTorch framework adapted to Ascend AI Processors in the operating environment can be used properly.
2.  Before building and installing Apex, you have configured the environment variables on which the build depends. See  [Configuring Environment Variables](#configuring-environment-variablesmd).

#### Installation Process<a name="en-us_topic_0000001106176190_section11880164819567"></a>

1.  Log in to the server as the  **root**  user or a non-root user.
2.  Obtain the Apex source code.

    1.  Run the following command to obtain the Apex source code adapted to Ascend AI Processors:

        ```
        git clone https://gitee.com/ascend/apex.git
        ```

        The directory structure of the downloaded source code is as follows:

        ```
        apex
        │ ├─patch             # Directory of the patch adapted to Ascend AI Processors
        │    ├─npu.patch
        │ ├─scripts           # Build and creation directory
        │    ├─gen.sh
        │ ├─src               # Source code directory
        │ ├─tests              # Directory for storing test cases
        ```

    2.  Run the following commands to go to the  **apex**  directory and obtain the native Apex source code:

        ```
        cd apex
        git clone https://github.com/NVIDIA/apex.git
        ```

        After the native Apex source code is downloaded, the main directory structure of the code is as follows:

        ```
        apex
        │ ├─apex              # Directory for storing the native Apex code
        │ ├─patch             # Directory of the patch adapted to Ascend AI Processors
        │    ├─npu.patch
        │ ├─scripts           # Build and creation directory
        │    ├─gen.sh
        │ ├─src               # Source code directory
        │ ├─tests              # Directory for storing test cases
        ```

    3.  Go to the native Apex code directory, that is,  **apex/apex**. Switch to the code branch whose commit ID is 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a.

        ```
        cd apex
        git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
        ```

    >![](public_sys-resources/icon-note.gif) **NOTE:** 
    >Due to network fluctuation, it may take a long time to obtain the source code.

3.  Generate the Apex installation package adapted to Ascend AI Processors.
    1.  Go to the  **apex/scripts**  directory and run the conversion script to generate full code adapted to Ascend AI Processors.

        ```
        cd ../scripts
        bash gen.sh
        ```

        The full code adapted to Ascend AI Processors is generated in the  **apex/apex**  directory.

    2.  Go to the full code directory  **apex/apex**, and build and generate the binary installation package of Apex.

        ```
        cd ../apex
        python3 setup.py --cpp_ext --npu_float_status bdist_wheel
        ```

        The Python version must be the same as that used by PyTorch. The generated binary package is stored in the current  **dist**  directory, that is,  **apex/apex/dist**.

4.  <a name="en-us_topic_0000001106176190_li425495374416"></a>Install Apex.

    Go to the  **apex/apex/dist**  directory and run the following command to install Apex:

    ```
    pip3 install --upgrade apex-0.1+ascend-cp37-cp37m-linux_{arch}.whl
    ```

    _\{**arch\}**_  indicates the architecture information. The value can be  **aarch64**  or  **x86\_64**.

    >![](public_sys-resources/icon-note.gif) **NOTE:** 
    >To upgrade PyTorch in the environment, uninstall the PyTorch software package installed in the environment and then perform  [Step 4 Install Apex](#en-us_topic_0000001106176190_li425495374416). Run the following command to check whether PyTorch has been installed:
    >**pip3 list | grep apex**


<h2 id="referencesmd">References</h2>

-   **[Installing CMake](#installing-cmakemd)**  

-   **[How Do I Install GCC 7.3.0?](#how-do-i-install-gcc-7-3-0md)**  

-   **[What to Do If "torch 1.5.0xxxx" and "torchvision" Do Not Match When torch-\*.whl Is Installed?](#what-to-do-if-torch-1-5-0xxxx-and-torchvision-do-not-match-when-torch--whl-is-installedmd)**  


<h3 id="installing-cmakemd">Installing CMake</h3>

The following describes how to install CMake 3.12.1.

1.  Obtain the CMake software package.

    ```
    wget https://cmake.org/files/v3.12/cmake-3.12.1.tar.gz --no-check-certificate
    ```

2.  Decompress the package and go to the software package directory.

    ```
    tar -xf cmake-3.12.1.tar.gz
    cd cmake-3.12.1/
    ```

3.  Run the configuration, build, and installation commands.

    ```
    ./configure --prefix=/usr/local/cmake
    make && make install
    ```

4.  Set the soft link.

    ```
    ln -s /usr/local/cmake/bin/cmake /usr/bin/cmake
    ```

5.  Check whether CMake has been installed.
    ```
    cmake --version
    ```

    If the message "cmake version 3.12.1" is displayed, the installation is successful.


<h3 id="how-do-i-install-gcc-7-3-0md">How Do I Install GCC 7.3.0?</h3>

Perform the following steps as the  **root**  user.

1.  Download  **gcc-7.3.0.tar.gz**  from  [https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz](https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz).
2.  GCC installation requires adequate temporary space. Run the following command to clear the  **/tmp**  directory in advance:

    ```
    sudo rm -rf /tmp/*
    ```

3.  Install the dependency package. \(CentOS and Ubuntu are used as examples.\)
    -   For CentOS, run the following command:

        ```
        yum install bzip2    
        ```

    -   For Ubuntu, run the following command:

        ```
        apt-get install bzip2    
        ```

4.  Build and install GCC.
    1.  Go to the directory where the source package  **gcc-7.3.0.tar.gz**  is located and run the following command to decompress it:

        ```
        tar -zxvf gcc-7.3.0.tar.gz
        ```

    2.  Go to the extracted directory and run the following command to download the GCC dependency packages:

        ```
        cd gcc-7.3.0
        ./contrib/download_prerequisites
        ```

        If an error is reported during the command execution, run the following commands in the  **gcc-7.3.0/**  directory to download the dependency packages:

        ```
        wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2
        wget http://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.4.tar.bz2
        wget http://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz
        wget http://gcc.gnu.org/pub/gcc/infrastructure/isl-0.16.1.tar.bz2
        ```

        After the preceding dependencies are downloaded, run the following command again:

        ```
        ./contrib/download_prerequisites
        ```

        If the validation fails, check whether the dependency packages are repeatedly downloaded. The packages should be downloaded at a time.

    3.  <a name="en-us_topic_0000001135347812_en-us_topic_0000001173199577_en-us_topic_0000001172534867_en-us_topic_0276688294_li1649343041310"></a>Run the configuration, build, and installation commands.

        ```
        ./configure --enable-languages=c,c++ --disable-multilib --with-system-zlib --prefix=/usr/local/linux_gcc7.3.0
        make -j15    # Check the number of CPUs by running grep -w processor /proc/cpuinfo|wc -l. In this example, the number is 15.
        make install    
        ```

        >![](public_sys-resources/icon-notice.gif) **NOTICE:** 
        >The  **--prefix**  option is used to specify the linux\_gcc7.3.0 installation path, which is configurable. Do not set it to  **/usr/local**  or  **/usr**, which is the default installation path for the GCC installed by using the software source. Otherwise, a conflict occurs and the original GCC compilation environment of the system is damaged. In this example, the installation path is set to  **/usr/local/linux\_gcc7.3.0**.


5.  Set the environment variable.

    Training must be performed in the compilation environment with GCC upgraded. Therefore, configure the following environment variable in your training script:

    ```
    export LD_LIBRARY_PATH=${install_path}/lib64:${LD_LIBRARY_PATH}
    ```

    **$\{install\_path\}**  indicates the GCC 7.3.0 installation path configured in  [3](#en-us_topic_0000001135347812_en-us_topic_0000001173199577_en-us_topic_0000001172534867_en-us_topic_0276688294_li1649343041310). In this example, the GCC 7.3.0 installation path is  **/usr/local/gcc7.3.0/**.

    >![](public_sys-resources/icon-note.gif) **NOTE:** 
    >Skip this step if you do not need to use the compilation environment with GCC upgraded.


<h3 id="what-to-do-if-torch-1-5-0xxxx-and-torchvision-do-not-match-when-torch--whl-is-installedmd">What to Do If "torch 1.5.0xxxx" and "torchvision" Do Not Match When torch-\*.whl Is Installed?</h3>

#### Symptom<a name="en-us_topic_0000001105856364_en-us_topic_0175549220_section197270431505"></a>

During the installation of  **torch-**_\*_**.whl**, the message "ERROR: torchvision 0.6.0 has requirement torch==1.5.0, but you'll have torch 1.5.0a0+1977093 which is incompatible" is displayed.

![](figures/en-us_image_0000001190081735.png)

#### Possible Causes<a name="en-us_topic_0000001105856364_en-us_topic_0175549220_section169499490501"></a>

When the PyTorch is installed, the version check is automatically triggered. The version of the torchvision installed in the environment is 0.6.0. During the check, it is found that the version of the  **torch-**_\*_**.whl**  is inconsistent with the required version 1.5.0. As a result, an error message is displayed, but the installation is successful.

#### Solution<a name="en-us_topic_0000001105856364_section108142031907"></a>

This problem has no impact on the actual result, and no action is required.

