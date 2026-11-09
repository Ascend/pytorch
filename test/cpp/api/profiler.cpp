// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <c10/util/Exception.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ActivityBuffers.h"
#include "Config.h"
#include "output_base.h"
#include "torch_npu/csrc/profiler/npu_activity_profiler.h"

namespace torch_npu {
namespace profiler {
namespace {

class MockActivityLogger : public libkineto::ActivityLogger {
 public:
  MOCK_METHOD(void, handleDeviceInfo, (const libkineto::DeviceInfo&, int64_t), (override));
  MOCK_METHOD(void, handleResourceInfo, (const libkineto::ResourceInfo&, int64_t), (override));
  MOCK_METHOD(void, handleOverheadInfo, (const libkineto::ActivityLogger::OverheadInfo&, int64_t), (override));
  MOCK_METHOD(void, handleTraceSpan, (const libkineto::TraceSpan&), (override));
  MOCK_METHOD(void, handleActivity, (const libkineto::ITraceActivity&), (override));
  MOCK_METHOD(void, handleGenericActivity, (const libkineto::GenericTraceActivity&), (override));
  MOCK_METHOD(
      void,
      handleTraceStart,
      ((const std::unordered_map<std::string, std::string>&), const std::string&),
      (override));
  MOCK_METHOD(void, finalizeMemoryTrace, (const std::string&, const libkineto::Config&), (override));
  // The owning argument requires a complete ActivityBuffers type even with an empty body.
  void finalizeTrace(const libkineto::Config&, std::unique_ptr<libkineto::ActivityBuffers>, int64_t) override {}
};

class FakeNPUActivityProfiler : public NPUActivityProfiler {
 public:
  const char* name() const override {
    return "fake";
  }
  void start() override {
    ++starts;
    status_ = libkineto::TraceStatus::RECORDING;
  }
  void stop() override {
    ++stops;
    status_ = libkineto::TraceStatus::PROCESSING;
  }
  std::vector<std::string> errors() override {
    return {};
  }
  void processTrace(libkineto::ActivityLogger&) override {
    ++traces;
  }
  void processTrace(libkineto::ActivityLogger&, libkineto::getLinkedActivityCallback, int64_t, int64_t) override {
    ++traces;
  }
  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override {
    return nullptr;
  }
  std::vector<libkineto::ResourceInfo> getResourceInfos() override {
    return {};
  }
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override {
    return nullptr;
  }

  int starts = 0;
  int stops = 0;
  int traces = 0;
};

std::unique_ptr<libkineto::Config> configWithBackend(const char* backend) {
  auto config = std::make_unique<libkineto::Config>();
  if (backend != nullptr) {
    EXPECT_TRUE(config->parse(std::string("CUSTOM_CONFIG={\"backend\":\"") + backend + "\"}"));
  }
  return config;
}

std::unique_ptr<libkineto::Config> configWithJson(const std::string& json) {
  auto config = std::make_unique<libkineto::Config>();
  EXPECT_TRUE(config->parse("CUSTOM_CONFIG=" + json));
  return config;
}

TEST(NPUActivityProfilerFactoryTest, SelectsBackendAndPluginCreatesManager) {
  struct TestCase {
    const char* config;
    const char* expected;
  };
  for (const auto& test :
       std::vector<TestCase>{{nullptr, "mspti"}, {"mspti", "mspti"}, {"aclprof", "aclprof"}, {"invalid", "mspti"}}) {
    auto config = configWithBackend(test.config);
    auto profiler = NPUActivityProfilerFactory::create({libkineto::ActivityType::CONCURRENT_KERNEL}, *config);
    ASSERT_NE(profiler, nullptr);
    EXPECT_STREQ(profiler->name(), test.expected);
  }

  NPUActivityProfilerPlugin plugin;
  auto config = configWithBackend("aclprof");
  auto session = plugin.configure({libkineto::ActivityType::CONCURRENT_KERNEL}, *config);
  ASSERT_NE(session, nullptr);
  EXPECT_NE(dynamic_cast<PrivateUse1ProfilerManager*>(session.get()), nullptr);
}

TEST(NPUActivityProfilerFactoryTest, ConfigurationMatrixAndRetention) {
  struct Case {
    const char* json;
    const char* backend;
    const char* level;
    std::vector<std::string> exports;
  };
  for (const auto& test : std::vector<Case>{
           {"", "mspti", "", {}},
           {"{}", "mspti", "", {}},
           {"invalid JSON", "mspti", "", {}},
           {"[]", "mspti", "", {}},
           {R"({"backend":true})", "mspti", "", {}},
           {R"({"backend":"aclprof"})", "aclprof", "", {}},
           {R"({"profiler_level":"Level1"})", "mspti", "Level1", {}},
           {R"({"export_type":["db"]})", "mspti", "", {"db"}},
           {R"({"profiler_level":"Level0","export_type":["text"]})", "mspti", "Level0", {"text"}},
           {R"({"backend":"mspti","profiler_level":"Level2","export_type":["db","text"]})",
            "mspti",
            "Level2",
            {"db", "text"}},
           {R"({"backend":"invalid","profiler_level":"Level_none","export_type":["db"]})",
            "mspti",
            "Level_none",
            {"db"}},
           {R"({"backend":"aclprof","profiler_level":"Level1","export_type":["db"]})", "aclprof", "Level1", {"db"}},
           {R"({"backend":"aclprof","export_type":["text","db","text"]})", "aclprof", "", {"db", "text"}}}) {
    SCOPED_TRACE(test.json);
    auto config = configWithJson(test.json);
    auto profiler = NPUActivityProfilerFactory::create({libkineto::ActivityType::CONCURRENT_KERNEL}, *config);
    ASSERT_NE(profiler, nullptr);
    EXPECT_STREQ(profiler->name(), test.backend);
    EXPECT_EQ(profiler->experimentalConfig().backend, test.backend);
    EXPECT_EQ(profiler->experimentalConfig().profilerLevel, test.level);
    EXPECT_EQ(profiler->experimentalConfig().exportTypes, test.exports);
  }
}

TEST(NPUActivityProfilerFactoryTest, RejectsMalformedOptions) {
  for (const auto& json :
       {R"({"profiler_level":1})",
        R"({"profiler_level":true})",
        R"({"profiler_level":null})",
        R"({"profiler_level":[]})",
        R"({"profiler_level":"Level3"})",
        R"({"export_type":null})",
        R"({"export_type":"db"})",
        R"({"export_type":[]})",
        R"({"export_type":{}})",
        R"({"export_type":["csv"]})",
        R"({"export_type":["db",false]})"}) {
    SCOPED_TRACE(json);
    auto config = configWithJson(json);
    EXPECT_THROW(NPUActivityProfilerFactory::create({libkineto::ActivityType::CONCURRENT_KERNEL}, *config), c10::Error);
  }
}

TEST(NPUActivityProfilerFactoryTest, SessionConfigurationIsOwnedAndIndependent) {
  const auto activities = std::set<libkineto::ActivityType>{libkineto::ActivityType::CONCURRENT_KERNEL};
  auto config = configWithJson(R"({"backend":"aclprof","profiler_level":"Level1","export_type":["db"]})");
  auto first = NPUActivityProfilerFactory::create(activities, *config);
  config = configWithJson(R"({"backend":"mspti","profiler_level":"Level2","export_type":["text"]})");
  auto second = NPUActivityProfilerFactory::create(activities, *config);
  config.reset();
  EXPECT_EQ(first->experimentalConfig().profilerLevel, "Level1");
  EXPECT_EQ(first->experimentalConfig().exportTypes, std::vector<std::string>{"db"});
  EXPECT_EQ(second->experimentalConfig().profilerLevel, "Level2");
  EXPECT_EQ(second->experimentalConfig().exportTypes, std::vector<std::string>{"text"});
  auto empty = configWithBackend(nullptr);
  auto third = NPUActivityProfilerFactory::create(activities, *empty);
  EXPECT_STREQ(third->name(), "mspti");
  EXPECT_TRUE(third->experimentalConfig().profilerLevel.empty());
  EXPECT_TRUE(third->experimentalConfig().exportTypes.empty());
  EXPECT_EQ(NPUActivityProfilerFactory::create({}, *empty), nullptr);

  NPUActivityProfilerPlugin plugin;
  config = configWithJson(R"({"backend":"aclprof","profiler_level":"Level1","export_type":["db"]})");
  auto session = plugin.configure(0, 1, activities, *config);
  ASSERT_NE(dynamic_cast<PrivateUse1ProfilerManager*>(session.get()), nullptr);
  session->start();
  session->stop();
  EXPECT_EQ(session->status(), libkineto::TraceStatus::PROCESSING);
}

TEST(PrivateUse1ProfilerManagerTest, DelegatesLifecycleAndTrace) {
  auto fake = std::make_unique<FakeNPUActivityProfiler>();
  auto* backend = fake.get();
  PrivateUse1ProfilerManager manager(std::move(fake));
  ::testing::NiceMock<MockActivityLogger> logger;

  EXPECT_EQ(manager.status(), libkineto::TraceStatus::READY);
  manager.start();
  EXPECT_EQ(manager.status(), libkineto::TraceStatus::RECORDING);
  manager.processTrace(logger);
  manager.processTrace(logger, nullptr, 1, 2);
  manager.stop();
  EXPECT_EQ(manager.status(), libkineto::TraceStatus::PROCESSING);
  EXPECT_EQ(backend->starts, 1);
  EXPECT_EQ(backend->stops, 1);
  EXPECT_EQ(backend->traces, 2);
}

TEST(AclprofActivityProfilerTest, SequentialStubSessionsAreIndependentAndEmpty) {
  for (int session = 0; session < 2; ++session) {
    auto config = configWithBackend("aclprof");
    auto profiler = NPUActivityProfilerFactory::create({libkineto::ActivityType::CONCURRENT_KERNEL}, *config);
    ASSERT_NE(profiler, nullptr);
    EXPECT_EQ(profiler->status(), libkineto::TraceStatus::READY);
    profiler->start();
    EXPECT_EQ(profiler->status(), libkineto::TraceStatus::RECORDING);
    profiler->stop();
    EXPECT_EQ(profiler->status(), libkineto::TraceStatus::PROCESSING);
    ::testing::NiceMock<MockActivityLogger> logger;
    EXPECT_CALL(logger, handleGenericActivity(::testing::_)).Times(0);
    profiler->processTrace(logger);
  }
}

} // namespace
} // namespace profiler
} // namespace  torch_npu
