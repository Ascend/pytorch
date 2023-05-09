#ifndef INC_EXTERNAL_GE_GE_API_H_
#define INC_EXTERNAL_GE_GE_API_H_

#include <map>
#include <string>
#include <vector>

#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"
#include "graph/graph.h"
#include "graph/tensor.h"

namespace ge {
typedef uint32_t (*pCallBackFunc)(uint32_t graph_id, const std::map<std::string, ge::Tensor> &params_list);

namespace session {
typedef uint32_t (*pCallBackFunc)(uint32_t graph_id, const std::map<AscendString, ge::Tensor> &params_list);
}

// Initialize GE
ATTRIBUTED_DEPRECATED(GE_FUNC_VISIBILITY Status GEInitialize(const std::map<AscendString, AscendString> &))
GE_FUNC_VISIBILITY Status GEInitialize(const std::map<std::string, std::string> &options);

GE_FUNC_VISIBILITY Status GEInitialize(const std::map<AscendString, AscendString> &options);

// Finalize GE, release all resources
GE_FUNC_VISIBILITY Status GEFinalize();

GE_FUNC_VISIBILITY std::string GEGetErrorMsg();

GE_FUNC_VISIBILITY std::string GEGetWarningMsg();

class GE_FUNC_VISIBILITY Session {
    public:
    ATTRIBUTED_DEPRECATED(Session(const std::map<AscendString, AscendString> &))
    explicit Session(const std::map<std::string, std::string> &options);

    explicit Session(const std::map<AscendString, AscendString> &options);

    ~Session();

    ///
    /// @ingroup client
    /// @brief add a graph with a specific graphId
    /// @param [in] graphId graph id
    /// @return Status result of function
    ///
    Status AddGraph(uint32_t graphId, const Graph &graph);

    ///
    /// @ingroup client
    /// @brief add a graph with a specific graphId and graphOptions
    /// @param [in] graphId graph id
    /// @param [in] graph the graph
    /// @param [in] options graph options
    /// @return Status result of function
    ///
    ATTRIBUTED_DEPRECATED(Status AddGraph(uint32_t, const Graph &, const std::map<AscendString, AscendString> &))
    Status AddGraph(uint32_t graphId, const Graph &graph, const std::map<std::string, std::string> &options);

    ///
    /// @ingroup client
    /// @brief add a graph with a specific graphId and graphOptions
    /// @param [in] graphId graph id
    /// @param [in] graph the graph
    /// @param [in] options graph options
    /// @return Status result of function
    ///
    Status AddGraph(uint32_t graphId, const Graph &graph, const std::map<AscendString, AscendString> &options);

    ///
    /// @ingroup client
    /// @brief add a copy graph with a specific graphId
    /// @param [in] graphId graph id
    /// @param [in] graph the graph
    /// @return Status result of function
    ///
    Status AddGraphWithCopy(uint32_t graph_id, const Graph &graph);

    ///
    /// @ingroup client
    /// @brief add a copy graph with a specific graphId and graphOptions
    /// @param [in] graphId graph id
    /// @param [in] graph the graph
    /// @param [in] options graph options
    /// @return Status result of function
    ///
    Status AddGraphWithCopy(uint32_t graph_id, const Graph &graph, const std::map<AscendString, AscendString> &options);

    ///
    /// @ingroup ge_graph
    /// @brief remove a graph of the session with specific session id
    /// @param [in] graphId graph id
    /// @return Status result of function
    ///
    Status RemoveGraph(uint32_t graphId);

    ///
    /// @ingroup ge_graph
    /// @brief run a graph of the session with specific session id
    /// @param [in] graphId graph id
    /// @param [in] inputs input data
    /// @param [out] outputs output data
    /// @return Status result of function
    ///
    Status RunGraph(uint32_t graphId, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs);

    ///
    /// @ingroup ge_graph
    /// @brief run a graph of the session with specific session id and specific stream asynchronously
    /// @param [in] graph_id graph id
    /// @param [in] stream specific stream
    /// @param [in] inputs input data
    /// @param [out] outputs output data
    /// @return Status result of function
    ///
    Status RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<Tensor> &inputs,
    std::vector<Tensor> &outputs);

    ///
    /// @ingroup ge_graph
    /// @brief build graph in the session with specific session id
    /// @param [in] graphId: graph id
    /// @param [in] inputs: input data
    /// @return Status result of function
    ///
    Status BuildGraph(uint32_t graphId, const std::vector<InputTensorInfo> &inputs);

    Status BuildGraph(uint32_t graphId, const std::vector<ge::Tensor> &inputs);  /*lint !e148*/

    ///
    /// @ingroup ge_graph
    /// @brief run graph in the session with specific session id asynchronously
    /// @param [in] graphId: graph id
    /// @param [in] inputs: input data
    /// @param [out] callback: callback while runing graph has been finished.
    ///                        The callback function will not be checked.
    ///                        Please ensure that the implementation of the function is trusted.
    /// @return Status result of function
    ///
    Status RunGraphAsync(uint32_t graphId, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback);

    ///
    /// @ingroup ge_graph
    /// @brief get variables in the session with specific session id
    /// @param [in] var_names: variable names
    /// @param [out] var_values: variable values
    /// @return Status result of function
    ///
    ATTRIBUTED_DEPRECATED(Status GetVariables(const std::vector<std::string> &, std::vector<Tensor> &))
    Status GetVariables(const std::vector<std::string> &var_names, std::vector<Tensor> &var_values);

    ///
    /// @ingroup ge_graph
    /// @brief get variables in the session with specific session id
    /// @param [in] var_names: variable names
    /// @param [out] var_values: variable values
    /// @return Status result of function
    ///
    Status GetVariables(const std::vector<AscendString> &var_names, std::vector<Tensor> &var_values);

    ///
    /// @ingroup ge_graph
    /// @brief register callback func with specific summary or checkpoint by users
    /// @param [in] key: func key
    /// @param [in] callback: callback  specific summary or checkpoint.
    ///                       The callback function will not be checked.
    ///                       Please ensure that the implementation of the function is trusted.
    /// @return Status result of function
    ///
    ATTRIBUTED_DEPRECATED(Status RegisterCallBackFunc(const char *, const session::pCallBackFunc &))
    Status RegisterCallBackFunc(const std::string &key, const pCallBackFunc &callback);

    Status RegisterCallBackFunc(const char *key, const session::pCallBackFunc &callback);

    bool IsGraphNeedRebuild(uint32_t graphId);

    private:
    uint64_t sessionId_;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GE_GE_API_H_
