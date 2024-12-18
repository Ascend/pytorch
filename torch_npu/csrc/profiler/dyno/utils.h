//
// Created by liyou on 2024/12/3.
//
#pragma once
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <c10/util/Exception.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
namespace torch_npu {
namespace profiler {
inline int32_t GetProcessId()
{
    int32_t pid = 0;
    pid = static_cast<int32_t>(getpid());
    return pid;
}

inline std::pair<int32_t, std::string> GetParentPidAndCommand(int32_t pid)
{
    std::string fileName = "/proc/" + std::to_string(pid) + "/stat";
    std::ifstream statFile(fileName);
    if (!statFile) {
        return std::make_pair(0, "");
    }
    int32_t parentPid = 0;
    std::string command;
    std::string line;
    if (std::getline(statFile, line)) {
        sscanf(line.c_str(), "%*d (%[^)]) %*c %d", command.data(), &parentPid);
        ASCEND_LOGI("Success to get parent pid %d", parentPid);
        return std::make_pair(parentPid, command);
    }
    ASCEND_LOGW("Failed to parse /proc/%d/stat", pid);
    return std::make_pair(0, "");
}

constexpr int MaxParentPids = 5;
inline std::vector<std::pair<int32_t, std::string>> GetPidCommandPairsofAncestors()
{
    std::vector<std::pair<int32_t, std::string>> process_pids_and_cmds;
    process_pids_and_cmds.reserve(MaxParentPids + 1);
    int32_t current_pid = GetProcessId();
    for (int i = 0; i <= MaxParentPids && (i == 0 || current_pid > 1); i++) {
        std::pair<int32_t, std::string> parent_pid_and_cmd = GetParentPidAndCommand(current_pid);
        process_pids_and_cmds.push_back(std::make_pair(current_pid, parent_pid_and_cmd.second));
        current_pid = parent_pid_and_cmd.first;
    }
    return process_pids_and_cmds;
}

inline std::vector<int32_t> GetPids()
{
    const auto &pids = GetPidCommandPairsofAncestors();
    std::vector<int32_t> res;
    res.reserve(pids.size());
    for (const auto &pidPair : pids) {
        res.push_back(pidPair.first);
    }
    return res;
}
inline std::string GenerateUuidV4()
{
    static std::random_device randomDevice;
    static std::mt19937 gen(randomDevice());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream stringStream;
    stringStream << std::hex;
    for (int i = 0; i < 8; i++) {
        stringStream << dis(gen);
    }
    stringStream << "-";
    for (int j = 0; j < 4; j++) {
        stringStream << dis(gen);
    }
    stringStream << "-4";
    for (int k = 0; k < 3; k++) {
        stringStream << dis(gen);
    }
    stringStream << "-";
    stringStream << dis2(gen);
    for (int m = 0; m < 3; m++) {
        stringStream << dis(gen);
    }
    stringStream << "-";
    for (int n = 0; n < 12; n++) {
        stringStream << dis(gen);
    }
    return stringStream.str();
}
} // namespace profiler
} // namespace torch_npu
