#pragma once

#include <map>

namespace  c10_npu::acl {
    
    static std::map<int, std::string> error_code_map = {
        {100000, "Parameter verification failed.\n\
            Check whether the input parameter value of the interface is correct."},
        {100001, "ACL uninitialized.\n\
            (1)Check whether the acl.init interface has been invoked for initialization. Ensure that the acl.init \
            interface has been invoked before other pyACL interfaces.\n\
            (2)Check whether the initialization interface of the corresponding function has been invoked, for example, \
            the acl.mdl.init_dump interface for initializing Dump and the acl.prof.init interface for initializing \
            Profiling."},
        {100002, "Repeated initialization or repeated loading.\n\
            Check whether the corresponding interface is repeatedly invoked for initialization or loading."},
        {100003, "Invalid file.\n\
            Check whether the file exists and can be accessed."},
        {100004, "Failed to write the file.\n\
            Check whether the file path exists and whether the file has the write permission."},
        {100005, "Invalid file size.\n\
            Check whether the file size meets the interface requirements."},
        {100006, "Failed to parse the file.\n\
            Check whether the file content is valid."},
        {100007, "The file is missing parameters.\n\
            Check whether the file content is complete."},
        {100008, "Invalid file parameter.\n\
            Check whether the parameter values in the file are correct."},
        {100009, "Invalid dump configuration.\n\
            Check whether the dump configuration in the acl.init interface configuration file is correct. For details, \
            see 'Preparing Comparison Data > Preparing Offline Model Dump Data Files' in the <Precision Comparison Tool\
             User Guide>."},
        {100010, "Invalid Profiling configuration.\n\
            Check whether the profiling configuration is correct."},
        {100011, "Invalid model ID.\n\
            Check whether the model ID is correct and whether the model is correctly loaded."},
        {100012, "Failed to deserialize the model.\n\
            The model may not match the current version. Convert the model again by referring to the <ATC Tool User \
            Guide>."},
        {100013, "Failed to parse the model.\n\
            The model may not match the current version. Convert the model again by referring to the <ATC Tool User \
            Guide>."},
        {100014, "Failed to read the model.\n\
            Check whether the model file exists and can be accessed."},
        {100015, "Invalid model size.\n\
            The model file is invalid, Convert the model again by referring to the <ATC Tool User Guide>."},
        {100016, "The model is missing parameters.\n\
            The model may not match the current version. Convert the model again by referring to the <ATC Tool User \
            Guide>."},
        {100017, "The input for the model does not match.\n\
            Check whether the model input is correct."},
        {100018, "The output of the model does not match.\n\
            Check whether the output of the model is correct."},
        {100019, "Model is not-dynamic.\n\
            Check whether the current model supports dynamic scenarios. If not, convert the model again by referring \
            to the <ATC Tool User Guide>."},
        {100020, "The type of a single operator does not match.\n\
            Check whether the operator type is correct."},
        {100021, "The input of a single operator does not match.\n\
            Check whether the operator input is correct."},
        {100022, "The output of a single operator does not match.\n\
            Check whether the operator output is correct."},
        {100023, "The attributes of a single operator do not match.\n\
            Check whether the operator attributes are correct."},
        {100024, "Single operator not found.\n\
            Check whether the operator type is supported."},
        {100025, "Failed to load a single operator.\n\
            The model may not match the current version. Convert the model again by referring to the <ATC Tool User \
            Guide>."},
        {100026, "Unsupported data type.\n\
            Check whether the data type exists or is currently supported."},
        {100027, "The format does not match.\n\
            Check whether the format is correct."},
        {100028, "When the operator interface is compiled in binary selection mode, the operator haven't registered a \
        selector.\n\
            Check whether the acl.op.register_compile_func interface is invoked to register the operator selector."},
        {100029, "During operator compilation, the operator kernel haven't registered.\n\
            Check whether the acl.op.create_kernel interface is invoked to register the operator kernel."},
        {100030, "When the operator interface is compiled in binary selection mode, the operator is registered \
        repeatedly.\n\
            Check whether the acl.op.register_compile_func interface is repeatedly invoked to register the operator \
            selector."},
        {100031, "During operator compilation, the operator kernel is registered repeatedly.\n\
            Check whether the acl.op.create_kernel interface is repeatedly invoked to register the operator kernel."},
        {100032, "Invalid queue ID.\n\
            Check whether the queue ID is correct."},
        {100033, "Duplicate subscription.\n\
            Check whether the acl.rt.subscribe_report interface is invoked repeatedly for the same stream."},
        {100034, "The stream is not subscribed.\n\
            Check whether the acl.rt.subscribe_report interface has been invoked.\n\
            [notice] This return code will be discarded in later versions. Use the 'ACL_ERROR_RT_STREAM_NO_CB_REG' \
            return code."},
        {100035, "The thread is not subscribed.\n\
            Check whether the acl.rt.subscribe_report interface has been invoked.\n\
            [notice] This return code will be discarded in later versions. Use the 'ACL_ERROR_RT_THREAD_SUBSCRIBE' \
            return code."},
        {100036, "Waiting for callback time out.\n\
            (1)Check whether the acl.rt.launch_callback interface has been invoked to deliver the callback task.\n\
            (2)Check whether the timeout interval in the acl.rt.process_report interface is proper.\n\
            (3)Check whether the callback task has been processed. If the callback task has been processed but the \
            acl.rt.process_report interface is still invoked, optimize the code logic.\n\
            [notice] This return code will be discarded in later versions. Use the 'ACL_ERROR_RT_REPORT_TIMEOUT' \
            return code."},
        {100037, "Repeated deinitialization.\n\
            Check whether the acl.finalize interface is repeatedly invoked for deinitialization."},
        {100038, "The static AIPP configuration information does not exist.\n\
            When invoking the 'acl.mdl.get_first_aipp_info' interface, ensure that the index value is correct.\n\
            [notice] This return code will be discarded in later versions. Use the 'ACL_ERROR_GE_AIPP_NOT_EXIST' \
            return code."},
        {100039, "The dynamic library path configured before running the application is the path of the \
        compilation stub, not the correct dynamic library path.\n\
            Check the configuration of the dynamic library path and ensure that the dynamic library in running mode \
            is used."},
        {100040, "Group is not set.\n\
            Check if the aclrtSetGroup interface has been called."},
        {100041, "No corresponding Group is created.\n\
            Check whether the Group ID set during the interface invocation is within the supported range. \
            The value range of the Group ID is [0, (number of groups -1)]. You can invoke the aclrtGetGroupCount \
            interface to obtain the number of groups."},
        {100042, "A profiling data collection task exists.\n\
            Check whether multiple profiling configurations are delivered to the same device.\n\
            For details, see the Profiling pyACL API. (Performance data is collected and flushed to disks through \
            the Profiling pyACL API.) The code logic is adjusted based on the interface invoking requirements and \
            interface invoking sequence in."},
        {100043, "The acl.prof.init interface is not used for analysis initialization.\n\
            Check the interface invoking sequence for collecting and analyzing data and analyze the pyACL API by \
            referring to.\n\
            See (Performance data collected and flushed to disks by analyzing the pyACL API) Description in."},
        {100044, "A task for obtaining dump data exists.\n\
            Before invoking the 'acl.mdl.init_dump', 'acl.mdl.set_dump', 'acl.mdl.finalize_dump' interfaces to \
            configure dump information: Check whether the acl.init interface has been invoked to configure the \
            dump information.\n\
            If yes, adjust the code logic and retain one method to configure the dump information."},
        {100045, "The acl.mdl.init_dump interface is not used to initialize the dump.\n\
            Check the invoking sequence of the interfaces for obtaining dump data. For details, see the description \
            of the acl.mdl.init_dump interface."},
        {148046, "Subscribe to the same model repeatedly.\n\
            Check the API invoking sequence. See (Performance data collected and flushed to disks by analyzing \
            the pyACL API) Description in."},
        {148047, "A conflict occurs in invoking the interface for collecting performance data.\n\
            The interfaces for collecting profiling performance data in the two modes cannot be invoked crossly.\n\
            The 'acl.prof.model_subscribe', 'acl.prof.get_op_*', 'acl.prof.model_un_subscribe' interface cannot be \
            invoked between the 'acl.prof.init' and 'acl.prof.finalize' interfaces.\n\
            The 'acl.prof.init', 'acl.prof.start', 'acl.prof.stop', and 'acl.prof.finalize' interfaces cannot be \
            invoked between the 'acl.prof.model_subscribe' and 'acl.prof.model_un_subscribe' interfaces."},
        {148048, "Invalid operator cache information aging configuration.\n\
            Check the aging configuration of the operator cache information. For details, see the configuration \
            description and example in acl.init."},
        {148049, "The 'ASCEND_OPP_PATH' environment variable is not set, or the value of the environment variable \
        is incorrect.\n\
            Check whether the 'ASCEND_OPP_PATH' environment variable is set and whether the value of the environment \
            variable is the installation path of the opp software package."},
        {148050, "The operator does not support dynamic Shape.\n\
            Check whether the shape of the operator in the single-operator model file is dynamic. If yes, change the \
            shape to a fixed one.\n\
            Check whether the shape of aclTensorDesc is dynamic during operator compilation. If yes, create \
            aclTensorDesc based on the fixed shape."},
        {148051, "Related resources have not been released.\n\
            This error code is returned if the related channel is not destroyed when the channel description \
            information is being destroyed. Check whether the channel associated with the channel description \
            is destroyed."},
        {148052, "Input image encoding format (such as arithmetic encoding and progressive encoding) that is not \
        supported by the JPEGD function.\n\
            When JPEGD image decoding is implemented, only Huffman encoding is supported.\n\
            The color space of the original image before compression is YUV. The ratio of pixel components is \
            4:4:4, 4:2:2, 4:2:0, 4:0:0, or 4:4:0.\n\
            Arithmetic coding, progressive JPEG, and JPEG2000 are not supported."},
        {200000, "Failed to apply for memory.\n\
            Check the available memory in the hardware environment."},
        {200001, "The interface does not support this function.\n\
            Check whether the invoked interface is supported."},
        {200002, "Invalid device.\n\
            Check whether the device exists.\n\
            [notice] This return code will be discarded in later versions. Use the 'ACL_ERROR_RT_INVALID_DEVICEID' \
            return code."},
        {200003, "The memory address is not aligned.\n\
            Check whether the memory address meets the interface requirements."},
        {200004, "The resources do not match.\n\
            Check whether the correct resources such as Stream and Context are transferred when the interface is \
            invoked."},
        {200005, "Invalid resource handle.\n\
            Check whether the transferred resources such as Stream and Context are destroyed or occupied when the \
            interface is invoked."},
        {200006, "This feature is not supported.\n\
            Rectify the fault based on the error information in the log or contact Huawei technical support. For \
            details about logs, see the Log Reference."},
        {200007, "Unsupported profiling configurations are delivered.\n\
            Check whether the profiling configuration is correct by referring to the description in \
            'acl.prof.create_config'."},
        {300000, "The storage limit is exceeded.\n\
            Check the remaining storage space in the hardware environment."},
        {500000, "Unknown internal error.\n\
            Rectify the fault based on the error information in the log or contact Huawei technical support. \
            For details about logs, see the Log Reference."},
        {500001, "The internal ACL of the system is incorrect.\n\
            Rectify the fault based on the error information in the log or contact Huawei technical support. \
            For details about logs, see the Log Reference."},
        {500002, "A GE error occurs in the system.\n\
            Rectify the fault based on the error information in the log or contact Huawei technical support. \
            For details about logs, see the Log Reference."},
        {500003, "The internal RUNTIME of the system is incorrect.\n\
            Rectify the fault based on the error information in the log or contact Huawei technical support. \
            For details about logs, see the Log Reference."},
        {500004, "An internal DRV (Driver) error occurs.\n\
            Rectify the fault based on the error information in the log or contact Huawei technical support. \
            For details about logs, see the Log Reference."},
        {500005, "Profiling error.\n\
            Rectify the fault based on the error information in the log or contact Huawei technical support. \
            For details about logs, see the Log Reference."},
    };
} /* npu_errmsg */
