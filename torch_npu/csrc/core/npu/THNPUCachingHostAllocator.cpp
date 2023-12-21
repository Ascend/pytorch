// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/npu_log.h"
#include <c10/util/Logging.h>
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

#include <Python.h>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "torch_npu/csrc/core/npu/THNPUCachingHostAllocator.h"
#include "third_party/acl/inc/acl/acl.h"

namespace {
struct BlockSize {
    size_t size; // allocation size
    void *ptr;   // host memory pointer

    explicit BlockSize(size_t size, void *ptr = nullptr) : size(size), ptr(ptr) {}
};

struct Block : public BlockSize {
    bool allocated;  // true if the block is currently allocated
    int event_count; // number of outstanding npu events
    std::unordered_set<c10_npu::NPUStream> streams;
    Block(size_t size, void *ptr, bool allocated)
        : BlockSize(size, ptr), allocated(allocated), event_count(0), streams() {}
};

static bool BlockComparator(const BlockSize &a, const BlockSize &b)
{
    // sort by size, break ties with pointer
    if (a.size != b.size) {
        return a.size < b.size;
    }
    return reinterpret_cast<uintptr_t>(a.ptr) < reinterpret_cast<uintptr_t>(b.ptr);
}

struct HostAllocator {
    using Comparison = bool (*)(const BlockSize &, const BlockSize &);

    // lock around all operations
    std::mutex mutex;

    // blocks by pointer
    std::unordered_map<void *, Block> blocks;

    // pointers that are ready to be allocated (event_count=0)
    std::set<BlockSize, Comparison> available;

    // outstanding ACL events
    std::deque<std::pair<aclrtEvent, void *>> npu_events;

    // record events
    std::mutex record_mutex;
    std::set<aclrtEvent> complete_events;

    HostAllocator() : available(BlockComparator) {}

    aclError malloc(void **ptr, size_t size)
    {
        std::lock_guard<std::mutex> lock(mutex);

        // process outstanding npu events which may have occurred
        aclError err = processEvents();
        if (err != ACL_ERROR_NONE) {
            return err;
        }

        // search for the smallest block which can hold this allocation
        BlockSize search_key(size);
        auto it = available.lower_bound(search_key);
        if (it != available.end()) {
            Block &block = blocks.at(it->ptr);
            AT_ASSERT(!block.allocated && block.event_count == 0);
            block.allocated = true;
            *ptr = block.ptr;
            available.erase(it);
            return ACL_ERROR_NONE;
        }

        *ptr = nullptr;

        // allocate a new block if no cached allocation is found
        err = aclrtMallocHost(ptr, size);
        if (err != ACL_ERROR_NONE) {
            return err;
        }

        blocks.insert({*ptr, Block(size, *ptr, true)});
        return ACL_ERROR_NONE;
    }

    aclError free(void *ptr)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ptr) {
            return ACL_ERROR_NONE;
        }

        // process outstanding npu events which may have occurred
        aclError err = processEvents();
        if (err != ACL_ERROR_NONE) {
            return err;
        }

        auto it = blocks.find(ptr);
        AT_ASSERT(it != blocks.end());

        Block &block = it->second;
        AT_ASSERT(block.allocated);

        // free (on valid memory) shouldn't fail, so mark unallocated before
        // we process the streams.
        block.allocated = false;

        // insert npu events for each stream on which this block was used. This
        err = insertEvents(block);
        if (err != ACL_ERROR_NONE) {
            return err;
        }

        if (block.event_count == 0) {
            // the block can be re-used if there are no outstanding npu events
            available.insert(block);
        }
        return ACL_ERROR_NONE;
    }

    aclError recordEvent(void *ptr, c10_npu::NPUStream stream)
    {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = blocks.find(ptr);
        if (it == blocks.end()) {
            // Sync when host memory is allocated by malloc
            aclError error = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream);
            if (error != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                AT_ERROR("ACL stream synchronize failed.");
                return error;
            }
            return ACL_ERROR_NONE;
        }

        Block &block = it->second;
        AT_ASSERT(block.allocated);

        block.streams.insert(stream);
        return ACL_ERROR_NONE;
    }

    bool isPinndPtr(void *ptr)
    {
        std::lock_guard<std::mutex> lock(mutex);
        return blocks.find(ptr) != blocks.end();
    }

    void insertCompleteEvent(aclrtEvent event)
    {
        if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
            std::lock_guard<std::mutex> lock(record_mutex);
            complete_events.insert(event);
        }
    }

    bool findAndEraseCompleteEvent(aclrtEvent event)
    {
        if (c10_npu::option::OptionsManager::CheckQueueEnable()) {
            std::lock_guard<std::mutex> lock(record_mutex);
            auto it = complete_events.find(event);
            if (it == complete_events.end()) {
                return false;
            }
            complete_events.erase(it);
        }
        return true;
    }

    aclError processEvents()
    {
        // Process outstanding npuEvents. Events that are completed are removed
        // from the queue, and the 'event_count' for the corresponding allocation
        // is decremented. Stops at the first event which has not been completed.
        // Since events on different devices or streams may occur out of order,
        // the processing of some events may be delayed.
        while (!npu_events.empty()) {
            auto &e = npu_events.front();
            aclrtEvent event = e.first;
            // when TASK_QUEUE_ENABLE is set, pytorch thread can destroy event
            // after acl thread has launched record event task
            if (!findAndEraseCompleteEvent(event)) {
                break;
            }
            c10_npu::acl::aclrtEventRecordedStatus status =
                c10_npu::acl::ACL_EVENT_RECORDED_STATUS_NOT_READY;
            aclError err = c10_npu::acl::AclQueryEventRecordedStatus(event, &status);
            if (err != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                insertCompleteEvent(event);
                return err;
            }
            if (status != c10_npu::acl::ACL_EVENT_RECORDED_STATUS_COMPLETE) {
                insertCompleteEvent(event);
                break;
            }

            err = aclrtDestroyEvent(event);
            if (err != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                insertCompleteEvent(event);
                return err;
            }
            ASCEND_LOGI("aclrtDestroyEvent is successfully executed.");

            Block &block = blocks.at(e.second);
            block.event_count--;
            if (block.event_count == 0 && !block.allocated) {
                available.insert(block);
            }
            npu_events.pop_front();
        }
        return ACL_ERROR_NONE;
    }

    void emptyCache()
    {
        std::lock_guard<std::mutex> lock(mutex);

        // remove events for freed blocks
        for (auto it = npu_events.begin(); it != npu_events.end(); ++it) {
            aclrtEvent event = it->first;
            Block &block = blocks.at(it->second);
            if (!block.allocated) {
                if (aclrtDestroyEvent(event) != ACL_ERROR_NONE) {
                    C10_NPU_SHOW_ERR_MSG();
                    NPU_LOGW("destory acl event fail");
                }
                ASCEND_LOGI("aclrtDestroyEvent is successfully executed.");
                block.event_count--;
            }
        }

        // all npu_events have been processed
        npu_events.clear();

        // clear list of available blocks
        available.clear();

        // free and erase non-allocated blocks
        for (auto it = blocks.begin(); it != blocks.end();) {
            Block &block = it->second;
            if (aclrtFreeHost(block.ptr) != ACL_ERROR_NONE) {
                NPU_LOGE("free host pin failed!");
            }
            if (!block.allocated) {
                it = blocks.erase(it);
            } else {
                block.streams.clear();
                ++it;
            }
        }
    }

    aclError insertEvents(Block &block)
    {
        aclError err = ACL_ERROR_NONE;

        int prev_device = 0;
        err = c10_npu::GetDevice(&prev_device);
        if (err != ACL_ERROR_NONE)
            return err;

        std::unordered_set<c10_npu::NPUStream> streams(std::move(block.streams));
        for (auto it = streams.begin(); it != streams.end(); ++it) {
            err = c10_npu::SetDevice(it->device_index());
            if (err != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                break;
            }

            aclrtEvent event = nullptr;
            err = c10_npu::acl::AclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE);
            if (err != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                break;
            }
            err = c10_npu::queue::HostAllocatorLaunchRecordEventTask(event, *it);
            if (err != ACL_ERROR_NONE)
                break;

            block.event_count++;
            npu_events.emplace_back(event, block.ptr);
        }

        c10_npu::SetDevice(prev_device);

        return err;
    }
};
} // namespace
static HostAllocator allocator;

aclError THNPUCachingHostAllocator_recordEvent(
    void *ptr,
    c10_npu::NPUStream stream)
{
    return allocator.recordEvent(ptr, stream);
}

void THNPUCachingHostAllocator_insertCompleteEvent(aclrtEvent event)
{
    return allocator.insertCompleteEvent(event);
}

bool THNPUCachingHostAllocator_isPinndPtr(void *ptr)
{
    return allocator.isPinndPtr(ptr);
}

void THNPUCachingHostAllocator_emptyCache()
{
    allocator.emptyCache();
}

static void THNPUCachingHostDeleter(void *ptr)
{
    // check the current thread have hold GIL Lock.
    if (PyGILState_Check()) {
        // the current thread should not hold GIL.
        Py_BEGIN_ALLOW_THREADS
            allocator.free(ptr);
        Py_END_ALLOW_THREADS
    } else {
        allocator.free(ptr);
    }
}

struct THNPUCachingHostAllocator final : public at::Allocator {
    at::DataPtr allocate(size_t size) const override
    {
        AT_ASSERT(size >= 0);
        void *ptr = nullptr;
        if (allocator.malloc(&ptr, size) != ACL_ERROR_NONE) {
            NPU_LOGE("allocate host pinned memory fail");
        }
        return {ptr, ptr, &THNPUCachingHostDeleter, at::DeviceType::CPU};
    }
    at::DeleterFnPtr raw_deleter() const override
    {
        return &THNPUCachingHostDeleter;
    }
};

static THNPUCachingHostAllocator thnpu_caching_host_allocator;
at::Allocator *getTHNPUCachingHostAllocator()
{
    return &thnpu_caching_host_allocator;
}

c10::Allocator *getPinnedMemoryAllocator()
{
    C10_LOG_API_USAGE_ONCE("aten.init.npu");
    c10_npu::NpuSysCtrl::SysStatus status =
        c10_npu::NpuSysCtrl::GetInstance().Initialize();
    if (status != c10_npu::NpuSysCtrl::SysStatus::INIT_SUCC) {
        NPU_LOGE("Npu init fail.");
    }
    return getTHNPUCachingHostAllocator();
}