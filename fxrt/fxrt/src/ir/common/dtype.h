#ifndef __IR_COMMON_DTYPE_H__
#define __IR_COMMON_DTYPE_H__

#include <cstdint>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace fxrt {
namespace ir {

/**
 * @brief Represents a data type.
 */
struct DataType {
  /**
   * @brief Enumeration of supported data types. The value equal to torch scalar type.
   */
  enum Type : int8_t {
    Unknown = -1, ///< Unknown data type
    UInt8 = 0, ///< 8-bit unsigned integer
    Int8 = 1, ///< 8-bit signed integer
    Int16 = 2, ///< 16-bit signed integer
    Int32 = 3, ///< 32-bit signed integer
    Int64 = 4, ///< 64-bit signed integer
    Float16 = 5, ///< 16-bit floating point
    Float32 = 6, ///< 32-bit floating point
    Float64 = 7, ///< 64-bit floating point
    Complex64 = 10, ///< 64-bit complex floating point
    Bool = 11, ///< Boolean
    QInt8 = 12, ///< 8-bit quantized integer
    QUInt4x2 = 13, ///< 8-bit quantized integer
    BFloat16 = 15, ///< 16-bit bfloating point
  };

  Type value; ///< The underlying enum value.

  /**
   * @brief Default constructor.
   */
  DataType() : value(Unknown) {}

  /**
   * @brief Constructs a DataType from a Type enum.
   * @param v The enum value.
   */
  DataType(Type v) : value(v) {} // NOLINT(runtime/explicit)

  /**
   * @brief Allows for using DataType in a switch statement.
   */
  operator Type() const {
    return value;
  }

  /**
   * @brief Gets the size of a data type in bytes.
   * @return The size in bytes.
   * @throws std::runtime_error if the data type is unsupported.
   */
  size_t GetSize() const {
    constexpr size_t kComplexFactor = 2;
    switch (value) {
      case Float16:
        return 2;
      case BFloat16:
        return 2;
      case Float32:
        return 4;
      case Float64:
        return 8;
      case Complex64:
        return kComplexFactor * sizeof(float);
      case Int8:
        return 1;
      case Int16:
        return 2;
      case Int32:
        return 4;
      case Int64:
        return 8;
      case UInt8:
        return 1;
      case QInt8:
        return 1;
      case QUInt4x2:
        return 1;
      case Bool:
        return 1;
      default:
        throw std::runtime_error("Unsupported data type");
    }
  }

  /**
   * @brief Converts a data type to its string representation.
   * @return The string representation.
   * @throws std::runtime_error if the data type is unsupported.
   */
  std::string ToString() const {
    switch (value) {
      case Float16:
        return "float16";
      case BFloat16:
        return "bfloat16";
      case Float32:
        return "float32";
      case Float64:
        return "float64";
      case Complex64:
        return "complex64";
      case Int8:
        return "int8";
      case Int16:
        return "int16";
      case Int32:
        return "int32";
      case Int64:
        return "int64";
      case UInt8:
        return "uint8";
      case QInt8:
        return "qint8";
      case QUInt4x2:
        return "quint4x2";
      case Bool:
        return "bool";
      case Unknown:
        return "unknown";
      default:
        throw std::runtime_error("Unsupported data type");
    }
  }

  /**
   * @brief Converts a string to the corresponding DataType.
   * @param str The string representation of the data type.
   * @return The corresponding DataType.
   * @throws std::runtime_error if the string does not correspond to a valid type.
   */
  static DataType FromString(const std::string& str) {
    static const std::unordered_map<std::string, Type> kStringToTypeMap = {
        {"float16", Float16},     {"f16", Float16}, // float16
        {"bfloat16", BFloat16},   {"bf16", BFloat16}, // bfloat16
        {"float32", Float32},     {"f32", Float32}, // float32
        {"float64", Float64},     {"f64", Float64}, // float64
        {"complex64", Complex64}, {"c64", Complex64}, // complex64
        {"int8", Int8},           {"i8", Int8},       {"si8", Int8}, // int8
        {"int16", Int16},         {"i16", Int16},     {"si16", Int16}, // int16
        {"int32", Int32},         {"i32", Int32},     {"si32", Int32}, // int32
        {"int64", Int64},         {"i64", Int64},     {"si64", Int64}, // int64
        {"uint8", UInt8},         {"ui8", UInt8}, // uint8
        {"qint8", QInt8}, // qint8
        {"quint4x2", QUInt4x2}, // quint4x2
        {"bool", Bool},           {"i1", Bool}, // bool
        {"unknown", Unknown}, // unknown
    };
    auto it = kStringToTypeMap.find(str);
    if (it != kStringToTypeMap.end()) {
      return DataType(it->second);
    }
    throw std::runtime_error("Unsupported data type string: " + str);
  }
};

} // namespace ir
} // namespace fxrt

#endif // __IR_COMMON_DTYPE_H__
