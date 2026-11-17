/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSER_DEFAULT_H_INCLUDED
#define PARAM_PARSER_DEFAULT_H_INCLUDED

#include "param/parser_common.h"
#include "param/utils.h"

#include <type_traits>
#include <limits>
#include <cstdlib>
#include <cerrno>

// Parsers for default types


// 默认所有类型都不支持，对于支持的类型进行特化
// Primary template - catch all for unsupported types
template <typename T>
struct ncclParamParserDefault {
  static ncclResult_t resolve(const char*, T&) {
    return ncclInvalidArgument;
  }

  static bool validate(const T&) {
    return false;
  }

  static std::string toString(const T&) {
    return "<unsupported>";
  }

  static constexpr const char* desc = "Unsupported parser";
};

// Specialization for bool
template <>
struct ncclParamParserDefault<bool> {
  // 兼容识别多种布尔表示
  static ncclResult_t resolve(const char* input, bool& out) {
    if (input == nullptr) return ncclInvalidArgument;
    std::string s(input);
    using nccl::param::utils::iequals;
    if (s == "1" || iequals(s, "T") || iequals(s, "TRUE")) {
      out = true;
      return ncclSuccess;
    }
    if (s == "0" || iequals(s, "F") || iequals(s, "FALSE")) {
      out = false;
      return ncclSuccess;
    }
    return ncclInvalidArgument;
  }

  static bool validate(const bool&) {
    return true;
  }

  static std::string toString(const bool& value) {
    return value ? "TRUE" : "FALSE";
  }

  static constexpr const char* desc = "Boolean: 1/T/TRUE or 0/F/FALSE";
};

// Specialization for const char*
// Note: This parser returns a pointer into the provided input; ncclParam<const char*>
// owns/copies the string into internal storage in ncclParam
template <>
struct ncclParamParserDefault<const char*> {
  static ncclResult_t resolve(const char* input, const char*& out) {
    out = input;
    return ncclSuccess;
  }

  static bool validate(const char* const&) {
    return true;
  }

  static std::string toString(const char* const& value) {
    return value ? std::string(value) : std::string();
  }

  static constexpr const char* desc = "String";
};

// 所有整数类型共用的解析逻辑，整数解析器基类
// Helper base for integer types
template <typename T>
struct ncclIntegerParser {
  // 区分有符号整数和无符号整数，分别用 std::strtoll 和 std::strtoull 转换
  static ncclResult_t resolve(const char* input, T& out) {
    if (input == nullptr || *input == '\0') return ncclInvalidArgument;
    char* endPtr = nullptr;
    errno = 0;
    // 判断是不是有符号整数
    if NCCL_PARAM_IF_CONSTEXPR (std::is_signed<T>::value) {
      long long val = std::strtoll(input, &endPtr, 10); // endPtr 保存第一个未解析字符的指针
      // errno == ERANGE	超出 long long 范围
      // errno == EINVAL	转换参数非法，例如空字符串
      if (endPtr == input || *endPtr != '\0' || errno == ERANGE || errno == EINVAL) return ncclInvalidArgument;
      if (val >= std::numeric_limits<T>::min() && val <= std::numeric_limits<T>::max()) {
        out = static_cast<T>(val);  // 这里没有验证整数值是否在 T 的范围内
      } else {
        return ncclInvalidArgument;
      }
    } else {
      unsigned long long val = std::strtoull(input, &endPtr, 10);
      if (endPtr == input || *endPtr != '\0' || errno == ERANGE || errno == EINVAL) return ncclInvalidArgument;
      if (val >= std::numeric_limits<T>::min() && val <= std::numeric_limits<T>::max()) {
        out = static_cast<T>(val);
      } else {
        return ncclInvalidArgument;
      }
    }
    return ncclSuccess;
  }

  // 这里验证无意义，因为 static_cast 已经确保了值在 T 的范围内
  static bool validate(const T& val) {
    return true;
  }

  static std::string toString(const T& value) {
    return std::to_string(value);
  }

  static constexpr const char* desc = "Integer";
};

// Explicit specializations for integer types
template <>
struct ncclParamParserDefault<int8_t> : ncclIntegerParser<int8_t> {};
template <>
struct ncclParamParserDefault<int16_t> : ncclIntegerParser<int16_t> {};
template <>
struct ncclParamParserDefault<int32_t> : ncclIntegerParser<int32_t> {};
template <>
struct ncclParamParserDefault<int64_t> : ncclIntegerParser<int64_t> {};
template <>
struct ncclParamParserDefault<uint8_t> : ncclIntegerParser<uint8_t> {};
template <>
struct ncclParamParserDefault<uint16_t> : ncclIntegerParser<uint16_t> {};
template <>
struct ncclParamParserDefault<uint32_t> : ncclIntegerParser<uint32_t> {};
template <>
struct ncclParamParserDefault<uint64_t> : ncclIntegerParser<uint64_t> {};

// ============================================================================
// nccl::param::parser — adapt static methods to function-pointer signatures
// ============================================================================
namespace nccl {
namespace param {
namespace parser {


// 适配 ncclParamParser<T> 的函数签名，即增加一个 void* ctx 参数
template <typename T>
ncclResult_t defaultResolve(const void*, const char* input, T& out) {
  return ncclParamParserDefault<T>::resolve(input, out);
}

template <typename T>
bool defaultValidate(const void*, const T& val) {
  return ncclParamParserDefault<T>::validate(val);
}

template <typename T>
std::string defaultToString(const void*, const T& val) {
  return ncclParamParserDefault<T>::toString(val);
}

template <typename T>
struct boundedCtx {
  T lower;
  T upper;
};

template <typename T>
bool boundedValidate(const void* ctx, const T& val) {
  auto* b = static_cast<const boundedCtx<T>*>(ctx);
  return val >= b->lower && val <= b->upper;
}

} // namespace parser
} // namespace param
} // namespace nccl

// ============================================================================
// Factory for default parser of type T
// ============================================================================
// 默认 parser 的工厂函数，返回一个静态单例 parser
template <typename T>
const ncclParamParser<T>& ncclParamDefault() {
  using namespace nccl::param::parser;
  // 每个类型 T 只创建一份默认 parser
  static const ncclParamParser<T> instance{defaultResolve<T>, defaultValidate<T>, defaultToString<T>, nullptr,
                                           ncclParamParserDefault<T>::desc};
  return instance;
}

// ============================================================================
// Bounded Parser Factory
// ============================================================================

// 创建一个“有范围限制”的 parser
// ncclParamBounded: is based on default parser with upper and lower bounds
// using resolve and toString of the default parser, customize validate function
template <typename T>
ncclParamParser<T> ncclParamBounded(T lower, T upper) {
  using namespace nccl::param::parser;
  auto ctx = std::make_shared<boundedCtx<T>>(boundedCtx<T>{lower, upper});
  std::string d = "Integer in range [" + std::to_string(lower) + ", " + std::to_string(upper) + "]";
  return {defaultResolve<T>, boundedValidate<T>, defaultToString<T>, std::move(ctx), std::move(d)};
}

template <typename T>
ncclParamParser<T> ncclParamBounded(T lower) {
  return ncclParamBounded(lower, std::numeric_limits<T>::max());
}

#endif /* PARAM_PARSER_DEFAULT_H_INCLUDED */
