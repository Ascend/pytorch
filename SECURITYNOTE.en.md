# Ascend Extension for PyTorch Plug-in Security Statement

## System security hardening

You are advised to enable ASLR (level 2), also called all random address space layout randomization, in the system. To enable ASLR, perform the following operations:

```bash
echo 2 > /proc/sys/kernel/randomize_va_space
```

## Running User Suggestions

To ensure security and minimize permissions, you are not advised to use torch_npu as an administrator account such as root.

## File permission control

1. It is recommended that users take security measures such as permission control for sensitive files, such as files required by training, files saved during training, private data of users, and business assets, for example, write permission control for dataset files in multi-user dataset sharing scenarios and data file permission control in profiler scenarios. For details about the permission, see the [File Permission Reference](#file-permission-reference) Perform the settings.
2. The profiler tool in torch_npu generates performance record files. The file permission is 640, and the folder permission is 750. You can control the permission on the generated files as required.
3. User permission control is required during installation and use. For details, see the [File Permission Reference](#file-permission-reference) To save installation or uninstallation logs, add the --log FILE parameter to the end of the installation or uninstallation command. The permission on the FILE file and directory must be controlled.
4. Permissions on files generated during the running of the PyTorch framework, for example, files saved by the torch.save interface, depend on system settings. It is recommended that the current script execution user control the permission on the generated file as required. For details about the permission, see the [File Permission Reference](#file-permission-reference). You can use umask to control default permissions to reduce security risks such as privilege escalation. You are advised to set umask in hosts (including host machines) and containers to 0027 or later to improve security.

### File Permission Reference

| Type                                                                        | Linux Permission Reference Max. |
| --------------------------------------------------------------------------- | ------------------------------- |
| User Home Directory                                                         | 750 (rwxr-x---)                 |
| Program files (including script files and library files)                    | 550 (r-xr-x---)                 |
| Program File Directory                                                      | 550 (r-xr-x---)                 |
| Configuration File                                                          | 640 (rw-r-----)                 |
| Configuration File Directory                                                | 750 (rwxr-x---)                 |
| Log files (recorded or archived)                                            | 440 (r--r-----)                 |
| Log File (Recording)                                                        | 640 (rw-r-----)                 |
| Log File Directory                                                          | 750 (rwxr-x---)                 |
| Debug File                                                                  | 640 (rw-r-----)                 |
| Debug File Directory                                                        | 750 (rwxr-x---)                 |
| Temporary File Directory                                                    | 750 (rwxr-x---)                 |
| Maintaining the Upgrade File Directory                                      | 770 (rwxrwx---)                 |
| Service data file                                                           | 640 (rw-r-----)                 |
| Service data file directory                                                 | 750 (rwxr-x---)                 |
| Key components, private keys, certificates, and ciphertext file directories | 700 (rwx------)                 |
| Key component, private key, certificate, and encryption ciphertext          | 600 (rw-------)                 |
| Encryption and decryption interfaces and encryption and decryption scripts  | 500 (r-x------)                 |

## Debugging tool declaration

torch_npu integrates the performance analysis tool profiler:

- Integration reason: Align with the native PyTorch support capability, provide the NPU PyTorch framework development performance analysis capability, and accelerate the performance analysis and debugging process.
- Application scenario: By default, performance data is not collected. If you need to perform performance analysis, you can add the Ascend Extension for PyTorch Profiler API to the model training script to collect performance data during training and export visualized performance data files after the training is complete.
- Risk warning: This function will generate performance data locally. Therefore, you need to enhance the protection of related performance data. Use this function when you need to analyze model performance. After the analysis is complete, disable this function in a timely manner. For details about the Profiler tool, see the [Introduction to the PyTorch Performance Analysis Tool](https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/trainingmigrguide/performance_tuning_0014.html).

## Data Security Statement

1. Data needs to be loaded and saved during the use of PyTorch. Some APIs, such as torch.load, torch.jit.load, and torch.distributed.scatter_object_list, use the risk module pickle, which may cause data risks. For details, see the [torch.load](https://pytorch.org/docs/main/generated/torch.load.html#torch.load), [collective-functions](https://pytorch.org/docs/main/distributed.html#collective-functions) Understand the specific risks.
2. Ascend Extension for PyTorch depends on the basic capabilities of CANN to implement functions such as AOE performance optimization, operator dump, and log recording. Users need to pay attention to the permission control on files generated by the preceding functions to enhance data protection.

## Building a Security Statement

torch_npu supports source code compilation and installation. During compilation, third-party libraries are downloaded and shell scripts are executed. During compilation, temporary program files and compilation directories are generated. Users can control the permissions of files in the source code directory as required to reduce security risks.

## Operation Safety Statement

1. You are advised to compile training scripts based on the resource status of the running environment. If the training script does not match the resource status, for example, the memory size for loading datasets exceeds the memory capacity limit, or the data generated by the training script locally exceeds the disk space size, errors may occur and the process exits unexpectedly.
2. When the PyTorch and torch_npu run abnormally, the process exits and the error information is printed. This is normal. You are advised to locate the fault cause based on the error message, including setting operator synchronization, viewing CANN logs, and parsing the generated core dump file.
3. The distributed nature of PyTorch and torch_npu applies only to internal communication. For performance reasons, these distributed features do not contain any authorization protocols and send unencrypted messages. For details about the distributed PyTorch feature and security precautions, see the [using-distributed-features](https://github.com/pytorch/pytorch/security#using-distributed-features).

## Public IP Address Statement

The configuration file and script of torch_npu exist [Public IP address](#public-ip-address).

### Public IP address

| Type                     | Open Source Address                                                                                                                                                                                                            | File Name                               | Public IP address/Public URL/Domain name/Email address                         | Usage Description                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/ascend/op-plugin.git                                       | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/GitHub_Trending/go/googletest.git                          | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/ascend/torchair.git                                        | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/ascend/Tensorpipe.git                                      | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/GitHub_Trending/fm/fmt.git                                 | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/GitHub_Trending/js/json.git                                | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/gh_mirrors/to/torch-mlir.git                               | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/cann/runtime.git                                           | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/cann/ge.git                                                | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .gitmodules                             | https://gitcode.com/cann/graph-autofusion.git                                  | Dependent open-source code repository                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | ci\\docker\\X86\\Dockerfile             | https://mirrors.huaweicloud.com/repository/pypi/simple                         | Docker configuration file, which is used to configure the pip source.                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | ci\\docker\\X86\\Dockerfile             | https://download.pytorch.org/whl/cpu                                           | Docker configuration source, which is used to configure the torch download connection.               |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | ci\\docker\\ARM\\Dockerfile             | https://mirrors.huaweicloud.com/repository/pypi/simple                         | Docker configuration file, which is used to configure the pip source.                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | ci\\docker\\X86\\Dockerfile             | https://mirrors.wlnmp.com/centos/Centos7-aliyun-altarch.repo                   | Docker configuration file, which is used to configure the yum source.                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | ci\\docker\\ARM\\Dockerfile             | https://mirrors.wlnmp.com/centos/Centos7-aliyun-altarch.repo                   | Docker configuration file, which is used to configure the yum source.                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | .github\\workflows\\_build-and-test.yml | https://mirrors.huaweicloud.com/repository/pypi/simple                         | Workflow configuration file, which is used to configure the pip source.                              |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | setup.cfg                               | https://gitcode.com/ascend/pytorch                                             | Input parameter of the URL for packing the whl.                                                      |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | setup.cfg                               | https://gitcode.com/ascend/pytorch/tags                                        | Input parameter download_url for packing the whl.                                                    |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | third_party\\op-plugin\\ci\\build.sh    | https://gitcode.com/ascend/pytorch.git                                         | The compilation script obtains code based on the torch_npu repository address and compiles the code. |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | third_party\\op-plugin\\ci\\exec_ut.sh  | https://gitcode.com/ascend/pytorch.git                                         | The UT script pulls code based on the torch_npu repository address and performs the UT test.         |
| Open source introduction | https://github.com/pytorch/pytorch/blob/release/2.13/test/nn/test_convolution.py https://github.com/pytorch/pytorch/blob/release/2.13/test/test_mps.py https://github.com/pytorch/pytorch/blob/main/test/test_serialization.py | test\\url.ini                           | https://download.pytorch.org/test_data/legacy_conv2d.pt                        | Used for downloading related PT files by the test script.                                            |
| Open source introduction | https://github.com/pytorch/pytorch/blob/release/2.13/test/test_serialization.py                                                                                                                                                | test\\url.ini                           | https://download.pytorch.org/test_data/legacy_serialized.pt                    | Used for downloading related PT files by the test script.                                            |
| Open source introduction | https://github.com/pytorch/pytorch/blob/release/2.13/test/test_serialization.py                                                                                                                                                | test\\url.ini                           | https://download.pytorch.org/test_data/gpu_tensors.pt                          | Used for downloading related PT files by the test script.                                            |
| Open source introduction | https://github.com/pytorch/pytorch/blob/release/2.13/test/onnx/test_utility_funs.py                                                                                                                                            | test\\url.ini                           | https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml          | Link to an issue                                                                                     |
| Open source introduction | https://github.com/pytorch/pytorch/blob/release/2.13/test/test_nn.py https://github.com/pytorch/pytorch/blob/main/test/test_serialization.py                                                                                   | test\\url.ini                           | https://download.pytorch.org/test_data/linear.pt                               | Used to download related PT files by the test script.                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | torch_npu\\npu\\config.yaml             | https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl | Download Path of the Flame Map Script                                                                |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | test\\requirements.txt                  | https://download.pytorch.org/whl/nightly/cpu                                   | Download link, which is used to download the torch-cpu version.                                      |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | requirements.txt                        | https://download.pytorch.org/whl/nightly/cpu                                   | Download link, which is used to download the torch-cpu version.                                      |
| Huawei-developed         | Not involved.                                                                                                                                                                                                                  | test\\get_synchronized_files.sh         | https://github.com/pytorch/pytorch.git                                         | Download link, which is used to download PyTorch test cases.                                         |

## Public Interface Statement

Ascend Extension for PyTorch is a PyTorch adaptation plug-in that allows users to use PyTorch to perform training and inference on Ascend devices. After the adaptation, the Ascend Extension for PyTorch supports the native PyTorch APIs. In addition to the native PyTorch APIs, Ascend Extension for PyTorch provides some customized APIs, including custom operator, affinity library, and other APIs. PyTorch APIs and customized API connections are supported. For details, see the Customization API Reference And also thePyTorch Native Interface List.

Reference [PyTorch Community Open Interface Specifications](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation), Ascend Extension for PyTorch provides an external customized interface. If a function appears to conform to the standard for public interfaces and is presented in the document, the interface is public. Otherwise, ask the community before using the function whether the function is indeed an exposed or accidentally exposed interface, because these unexposed interfaces may be modified or deleted in the future.

The Ascend Extension for PyTorch project is jointly developed by C++ and Python. Currently, only Python interfaces are provided for formal interfaces except in the Libtorch scenario. The DLL in the torch_npu binary package does not directly provide services. The exposed interfaces are for internal use. It is not recommended.

## Communication security hardening

The PyTorch distributed training service requires communication between devices. By default, the ports enabled for communication are all 0s listening. To reduce security risks, you are advised to perform security hardening in this scenario, for example, using iptables to configure firewalls. Restrict external access to the ports used by the distributed training before the distributed training starts and clear the firewall rules after the distributed training is complete.

1. Firewall Rule Setting and Remove Reference Script Template
    
    - For details about how to set firewall rules, see the following script:
    
        ```bash
        #!/bin/bash
        set -x
        
        #Port number to limit
        port={端口号}
        
        #Clear Old Rules
        iptables -D INPUT -p tcp -j {规则名}
        iptables -F {规则名}
        iptables -X {规则名}
        
        #Create a new rule chain
        iptables -t filter -N {规则名}
        
        #Configure the trustlist in the multi-node cluster scenario to allow other nodes to access the listening port of the active node.
        #Add a rule that allows a specific IP address range to the rule chain
        iptables -t filter -A {规则名} -i eno1 -p tcp --dport $port -s {允许外部访问的IP} -j ACCEPT
        
        #Disabling External Addresses to Access Distributed Training Ports
        #Add a rule for denying other IP addresses to the PORT-LIMIT-RULE rule chain.
        iptables -t filter -A {规则名} -i {要限制的网卡名} -p tcp --dport $port -j DROP
        
        #Passing Traffic to a Rule Chain
        iptables -I INPUT -p tcp -j {规则名}
        ```
    
    - Remove the firewall rule. For details, see the following script:
    
        ```bash
        #!/bin/bash
        set -x
        #Clear Rule
        iptables -D INPUT -p tcp -j {规则名}
        iptables -F {规则名}
        iptables -X {规则名}
        ```

2. Firewall Rule Setting and Removal Reference Script Example
    
    1. Set the firewall for a specific port. In the script, the port number is the port to be restricted. For details about the port number in the PyTorch distributed training, see.[Communication Matrix Information](#communication-matrix-information); The restricted network adapter name is the network adapter used by the server for distributed communication, and the IP address that allows external access is the IP address of the distributed training server. You can run the ifconfig command to view the network adapter and server IP address. In the following command output, eth0 indicates the network adapter name and 192.168.1.1 indicates the server IP address.
        
        ```bash
        #ifconfig
        eth0
            inet addr:192.168.1.1 Bcast:192.168.1.255 Mask:255.255.255.0
            inet6 addr: fe80::230:64ee:ef1a:c1a/64 Scope:Link
        ```

    2. Assume that the IP address of the active server is 192.168.1.1, the other server that requires distributed training is 192.168.1.2, and the training port is 29510.
        
        - For details about how to set firewall rules, see the following script:
        
            ```bash
            #!/bin/bash
            set -x
            
            #Set the listening port.
            port=29510
            
            #Clear old rules
            iptables -D INPUT -p tcp -j PORT-LIMIT-RULE
            iptables -F PORT-LIMIT-RULE
            iptables -X PORT-LIMIT-RULE
            
            #Create a PORT-LIMIT-RULE rule chain.
            iptables -t filter -N PORT-LIMIT-RULE
            
            #Set a trustlist in the multi-node cluster scenario to allow the IP address 192.168.1.2 to access the active node.
            #To add a rule that allows a specific IP address range in the PORT-LIMIT-RULE rule chain, run the following command:
            iptables -t filter -A PORT-LIMIT-RULE -i eno1 -p tcp --dport $port -s 192.168.1.2 -j ACCEPT
            
            #Disabling external addresses from accessing distributed training ports
            #Add a rule for denying other IP addresses to the PORT-LIMIT-RULE rule chain.
            iptables -t filter -A PORT-LIMIT-RULE -i eth0 -p tcp --dport $port -j DROP
            
            #Pass the traffic to the PORT-LIMIT-RULE rule chain.
            iptables -I INPUT -p tcp -j PORT-LIMIT-RULE
            ```
        
        - Remove the firewall rule. For details, see the following script:
        
            ```bash
            #!/bin/bash
            set -x
            #Clear Rule
            iptables -D INPUT -p tcp -j PORT-LIMIT-RULE
            iptables -F PORT-LIMIT-RULE
            iptables -X PORT-LIMIT-RULE
            ```

## Communication Matrix

PyTorch provides the distributed training capability and supports training in single-node and multi-node scenarios. Network communication is required. PyTorch uses TCP for communication, and torch_npu uses the HCCL in CANN to communicate with NPU devices. For details about the communication ports, see the [Communication Matrix Information](#communication-matrix-information). To ensure the network security of communication between nodes, you can use iptables to mitigate security risks. For details, see the [Communication security hardening](#communication-security-hardening) Perform network security hardening.

### Communication Matrix Information

| Component                                            | PyTorch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | HCCL                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source Device                                        | Server running torch_npu                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Server running torch_npu                                                                                                                                                                                                                                                                                                                                                                        |
| Source IP address                                    | Device IP address                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Device IP address                                                                                                                                                                                                                                                                                                                                                                               |
| Source Port                                          | The operating system automatically allocates resources. The allocation range is determined by the operating system configuration.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | The default value is 60000 and the value range is \[1024, 65520\]. You can use the HCCL_IF_BASE_PORT environment variable to specify the start port number of the host NIC. After the setting, the system uses 16 ports starting from the start port number by default.                                                                                                                         |
| Destination device                                   | Server running torch_npu                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Server running torch_npu                                                                                                                                                                                                                                                                                                                                                                        |
| Destination IP address                               | Device IP address                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Device IP address                                                                                                                                                                                                                                                                                                                                                                               |
| Destination port (listening)                         | The default port number is 29500 and 29400. You can set the port number.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | The default value is 60000 and the value range is \[1024, 65520\]. You can use the HCCL_IF_BASE_PORT environment variable to specify the start port number of the host NIC. After the setting, the system uses 16 ports starting from the start port number by default.                                                                                                                         |
| Agreement                                            | TCP                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | TCP                                                                                                                                                                                                                                                                                                                                                                                             |
| Port Description                                     | In the distributed scenario, if 1. torchrun/torch.distributed.launch (1) backend is set to static (backend by default), the destination port (29500 by default) is used to receive and send data, and the source port is used to receive and send data. If master-addr is used to specify an address and master-port is used to specify the port (2) backend as c10d, the destination port (29400 by default) is used to receive and send data, and the source port is used to receive and send data. Use rdzv_endpoint to specify the address and port number in the format of address:port number. 2. torch_npu_run: The destination port (29500 by default) is used to receive and send data, and the source port is used to receive and send data. Run the master-addr command to specify the IP address and the master-port command to specify the port. | The default value is 60000 and the value range is \[1024, 65520\]. You can use the HCCL_IF_BASE_PORT environment variable to specify the start port number of the host network adapter. After the setting, the system uses 16 ports starting from the start port number by default.                                                                                                             |
| Indicates whether the listening port can be changed. | Yes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Yes                                                                                                                                                                                                                                                                                                                                                                                             |
| Authentication mode                                  | No authentication mode                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | No authentication mode                                                                                                                                                                                                                                                                                                                                                                          |
| Encryption mode                                      | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | None                                                                                                                                                                                                                                                                                                                                                                                            |
| Home Plane                                           | Not involved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Not involved.                                                                                                                                                                                                                                                                                                                                                                                   |
| Version                                              | All versions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | All versions                                                                                                                                                                                                                                                                                                                                                                                    |
| Special Scenarios                                    | None                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | None                                                                                                                                                                                                                                                                                                                                                                                            |
| Remarks                                              | The communication process is controlled by the open-source software PyTorch. Configure the PyTorch native settings. For details, see the [PyTorch Documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility). The source port is automatically allocated by the operating system, and the allocation range is determined by the configuration of the operating system. For example, ubuntu, the source port is specified by the /proc/sys/net/ipv4/ipv4_local_port_range file and can be viewed by running the cat /proc/sys/net/ipv4/ipv4_local_port_range or sysctl net.ipv4.ip_local_port_range command.                                                                                                                                                                                                                                 | This communication process is controlled by the HCCL component in the CANN. The torch_npu component does not control the communication process. For details about the port range, see the [Environment Variable Reference](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0001.html) Execution Related > Aggregate Communications > HCCL_IF_BASE_PORT |

## Vulnerability Mechanism Description

The Ascend Extension for PyTorch community attaches great importance to the security of the community version. A vulnerability management specialist is assigned to handle vulnerability-related affairs. To build a more secure E2E AI tool chain, we welcome you to participate.

### Vulnerability Handling Process

For each security vulnerability, the Ascend Extension for PyTorch community arranges personnel to trace and handle the vulnerability. 

The following sections focus on the process of vulnerability reporting, vulnerability assessment, and vulnerability disclosure.

### Vulnerability Reporting

You can contact the Ascend Extension for PyTorch community team by submitting an issue. We will arrange security vulnerability specialist personnel to contact you as soon as possible. Note: To ensure security, do not describe specific information related to security and privacy in the issue.

#### Reporting response

1. The Ascend Extension for PyTorch community will confirm, analyze, and report security vulnerabilities within three working days, and initiate the security handling process.
2. The Ascend Extension for PyTorch security team distributes and follows up on security vulnerability issues after they are identified.
3. We will update the report in a timely manner as security vulnerability issues are classified, identified, fixed, and released.

### Vulnerability Assessment

The CVSS standard is widely used in the industry to evaluate the severity of vulnerabilities. When using CVSS v3.1 to evaluate vulnerabilities, Ascend Extension for PyTorch needs to set vulnerability attack scenarios and evaluate the impact based on the actual impact of the attack scenarios. Vulnerability severity evaluation refers to the evaluation of the vulnerability's difficulty in exploiting and the impact on confidentiality, integrity, and usability after exploitation, and a score is generated.

#### Vulnerability Assessment Criteria

The Ascend Extension for PyTorch evaluates the severity level of a vulnerability using the following vector:

- Attack Vector (AV): indicates the "remoteness" of the attack and how to exploit the vulnerability.
- Attack Complexity (AC): Describe the difficulty of executing an attack and the factors required to successfully execute the attack.
- User interaction (UI): Determines whether the attack requires user participation.
- Required Privileges (PR): Log the level of user authentication required for a successful attack.
- Scope (S): Determines whether an attacker can influence components with different privilege levels.
- Confidentiality (C): measures the impact of information disclosure to unauthorized parties.
- Integrity (I): measures the impact caused by information tampering.
- Availability (A): measures how much users are affected when they need to access data or services.

#### Assessment Principles

- Evaluate the severity of the vulnerability, not the risk.
- The evaluation must be based on the attack scenario and ensure that the attack can affect the confidentiality, integrity, and availability of the system.
- If multiple attack scenarios are involved, the attack scenario with the highest CVSS score should be used as the basis.
- If the library is embedded and invoked in the vulnerability, determine the attack scenario of the vulnerability based on the usage mode of the library in the product and evaluate the vulnerability.
- If the security defect cannot be triggered or does not affect the CIA (confidentiality, integrity, and availability), the CVSS score is 0.

#### Assessment Procedure

To evaluate the vulnerability severity, perform the following steps:

1. Set possible attack scenarios and score based on attack scenarios.
2. Identify the Vulnerable Component and the Impact Component.
3. Select the value of the basic indicator.
    
    - Exploitable metrics (attack vector, attack complexity, required permissions, user interaction, scope) Select a metric value based on the vulnerability component.
    - Impact metrics (confidentiality, integrity, availability) reflect either the impact on the vulnerable component or the impact on the affected component, whichever is the most severe.

#### Severity levels

| **Severity Rating** | **CVSS score (Score)** | **Vulnerability fixing duration** |
| ------------------- | ---------------------- | --------------------------------- |
| Critical            | 9.0-10.0               | 7 days                            |
| High                | 7.0-8.9                | 14 days                           |
| Medium (Medium)     | 4.0-6.9                | 30 days                           |
| Low (Low)           | 0.1-3.9                | 30 days                           |

### Vulnerability disclosure

After the security vulnerability is fixed, the Ascend Extension for PyTorch community releases the SA and SN. The security notice includes the technical details, type, reporter, CVE ID, affected version, and fixed version of the vulnerability. To protect the security of Ascend Extension for PyTorch users, the Ascend Extension for PyTorch community does not publicly disclose, discuss, or confirm security issues with Ascend Extension for PyTorch products until the Ascend Extension for PyTorch community has conducted investigations, remediation, and security bulletins.

### Appendixes

#### Security bulletin (SA)

Currently, the version is being maintained and has no security vulnerability.

#### Safety Instructions (SN)

Vulnerabilities of third-party open-source components: None
