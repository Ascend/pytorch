#pragma once

#include <string>
#include <utility>
#include <torch/torch.h>

namespace torch_npu {
namespace afd {
#pragma pack(push, 1)
struct ScheduleContext {
    struct CommonArea {
        uint32_t session_num;
        uint32_t micro_batch_num;
        uint32_t micro_batch_size;
        uint32_t selected_expert_num;
        uint32_t expert_num; // expert num per layer, include route expert and share expert
        uint32_t attn_to_ffn_token_size; // each token in ffn window data area space size, align to 512.
        uint32_t ffn_to_attn_token_size; // each token in attention window data area space size, align to 512.
        int32_t schedule_mode;  // 0:just ffn, 1:just attention, 2:ffn+attention
        int8_t reserve0[96];   // padding to 128 bytes
    };
    struct ControlArea {
        int32_t run_flag;     // 0: exit 1: running
        int8_t reserve2[124]; // padding to 128 bytes
    };
    struct FfnArea {
        // ffn area
        uint64_t token_info_buf;
        uint64_t token_info_buf_size;
        uint64_t token_data_buf;
        uint64_t token_data_buf_size;
        uint64_t polling_index;
        int8_t reserve3[88];

        // ffn out area
        uint64_t layer_ids_buf;
        uint64_t layer_ids_buf_size;
        uint64_t session_ids_buf;
        uint64_t session_ids_buf_size;
        uint64_t micro_batch_ids_buf;
        uint64_t micro_batch_ids_buf_size;
        uint64_t expert_ids_buf;
        uint64_t expert_ids_buf_size;
        uint32_t out_num;
        int8_t reserve4[60];
    };

    struct AttentionArea {
        // attention area
        uint64_t token_info_buf;  // point to a int64 dev mem
        uint64_t token_info_buf_size;
        uint64_t token_data_buf;  // point to a int64 dev mem
        uint64_t token_data_buf_size;
        uint32_t micro_batch_id;
        int8_t reserve5[92];
    };

    // common area
    CommonArea common;
    ControlArea control;
    AttentionArea attention;
    FfnArea ffn;
    // reserve area
    int8_t reserve6[384];  // padding to 1024 bytes
};
static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");
#pragma pack(pop)

class ScheduleContextHolder {
public:
    ScheduleContextHolder(int32_t schedule_mode, uint32_t session_num, uint32_t micro_batch_num,
                          uint32_t micro_batch_size, uint32_t selected_expert_num, uint32_t expert_num,
                          uint32_t attn_to_ffn_token_size, uint32_t ffn_to_attn_token_size, uint64_t ffn_window,
                          uint64_t ffn_window_size, uint64_t attention_window, uint64_t attention_window_size);

    ~ScheduleContextHolder() = default;

    uint32_t Init();

    std::pair<uint32_t, at::Tensor> GetContextTensor() const;

    uint32_t StopSchedule();

    std::string GetScheduleContextInfo() const;

private:
    bool CheckParams() const;

    bool CheckFfnParams() const;

    bool CheckAttentionParams() const;

    uint32_t InitFfn();

    uint32_t InitAttention();

    uint32_t InitFfnTokenInfoBuf() const;

    uint64_t CalcFfnTokenInfoSize() const;

    uint64_t CalcAttentionTokenInfoSize() const;

    uint32_t AllocAndAssignDevMem();

    uint32_t GetScheduleContextFromDev(ScheduleContext &context) const;

    static std::string ToString(const ScheduleContext &context);

    uint64_t ffn_window_ = 0;
    uint64_t ffn_window_size_ = 0;
    uint64_t attention_window_ = 0;
    uint64_t attention_window_size_ = 0;
    bool init_flag_ = false;
    ScheduleContext context_{};  // host ptr

    at::Tensor context_tensor_;
    at::Tensor workspace_tensor_;

    uint64_t workspace_size_ = 0;
};
}
}