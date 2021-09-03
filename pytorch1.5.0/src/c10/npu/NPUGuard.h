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

#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/npu/NPUMacros.h>
#include <c10/npu/impl/NPUGuardImpl.h>

#include <cstddef>

namespace c10 {
namespace npu {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for NPU.  It accepts
/// integer indices (interpreting them as NPU devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// NPUSetDevice/NPUGetDevice calls); however, it can only be used
/// from code that links against NPU directly.
struct NPUGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit NPUGuard() = delete;

  /// Set the current NPU device to the passed device index.
  explicit NPUGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current NPU device to the passed device.  Errors if the passed
  /// device is not a NPU device.
  explicit NPUGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  NPUGuard(const NPUGuard&) = delete;
  NPUGuard& operator=(const NPUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  NPUGuard(NPUGuard&& other) = delete;
  NPUGuard& operator=(NPUGuard&& other) = delete;

  /// Sets the NPU device to the given device.  Errors if the given device
  /// is not a NPU device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the NPU device to the given device.  Errors if the given device
  /// is not a NPU device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the NPU device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::NPUGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for NPU.  See
/// NPUGuard for when you can use this.
struct OptionalNPUGuard {
  /// Create an uninitialized OptionalNPUGuard.
  explicit OptionalNPUGuard() : guard_() {}

  /// Set the current NPU device to the passed Device, if it is not nullopt.
  explicit OptionalNPUGuard(optional<Device> device_opt) : guard_(device_opt) {}

  /// Set the current NPU device to the passed device index, if it is not
  /// nullopt
  explicit OptionalNPUGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalNPUGuard(const OptionalNPUGuard&) = delete;
  OptionalNPUGuard& operator=(const OptionalNPUGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalNPUGuard(OptionalNPUGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalNPUGuard& operator=(OptionalNPUGuard&& other) = delete;

  /// Sets the NPU device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a NPU
  /// device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the NPU device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a NPU device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the NPU device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original NPU device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<impl::NPUGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for NPU.  See NPUGuard
/// for when you can use this.
struct NPUStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit NPUStreamGuard() = delete;

  /// Set the current NPU device to the device associated with the passed
  /// stream, and set the current NPU stream on that device to the passed
  /// stream. Errors if the Stream is not a NPU stream.
  explicit NPUStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  NPUStreamGuard(const NPUStreamGuard&) = delete;
  NPUStreamGuard& operator=(const NPUStreamGuard&) = delete;

  /// Move is disallowed, as NPUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  NPUStreamGuard(NPUStreamGuard&& other) = delete;
  NPUStreamGuard& operator=(NPUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a NPU stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on NPU, use NPUMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the NPU stream that was set at the time the guard was constructed.
  NPUStream original_stream() const {
    return NPUStream(NPUStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent NPU stream that was set using this device guard,
  /// either from construction, or via set_stream.
  NPUStream current_stream() const {
    return NPUStream(NPUStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent NPU device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the NPU device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::NPUGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for NPU.  See NPUGuard
/// for when you can use this.
struct OptionalNPUStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalNPUStreamGuard() : guard_() {}

  /// Set the current NPU device to the device associated with the passed
  /// stream, and set the current NPU stream on that device to the passed
  /// stream. Errors if the Stream is not a NPU stream.
  explicit OptionalNPUStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalNPUStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalNPUStreamGuard(const OptionalNPUStreamGuard&) = delete;
  OptionalNPUStreamGuard& operator=(const OptionalNPUStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalNPUStreamGuard(OptionalNPUStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalNPUStreamGuard& operator=(OptionalNPUStreamGuard&& other) = delete;

  /// Resets the currently set NPU stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the NPU stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  optional<NPUStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(NPUStream(NPUStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Returns the most recent NPU stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  optional<NPUStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(NPUStream(NPUStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Restore the original NPU device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::NPUGuardImpl> guard_;
};
} // namespace npu
} // namespace c10