/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSER_COMMON_H_INCLUDED
#define PARAM_PARSER_COMMON_H_INCLUDED

#include "nccl.h"
#include "debug.h"

#include <string>
#include <memory>
#include <array>
#include <cstring>

// 把字符串解析成类型 T 的解析器
// ncclParamParser: Runtime parser interface ncclParam depends on
template <typename T>
struct ncclParamParser {
  // 把字符串解析成 T 类型的值
  using resolveFn_t = ncclResult_t (*)(const void*, const char*, T&);
  // 检查解析后的值是否合法
  using validateFn_t = bool (*)(const void*, const T&);
  // 把当前参数值转换回字符串
  using toStringFn_t = std::string (*)(const void*, const T&);

  resolveFn_t resolveFn = nullptr;
  validateFn_t validateFn = nullptr;
  toStringFn_t toStringFn = nullptr;
  // 保存解析器所需的额外数据，例如枚举解析器的选项表
  std::shared_ptr<const void> ctx;  // owns factory state; nullptr for stateless parsers
  std::string desc;  // Description of accepted values

  // 包装方法，把 ctx.get() 传进去
  // Wrapper methods — preserve parser.resolve(...) call syntax
  ncclResult_t resolve(const char* input, T& out) const {
    return resolveFn(ctx.get(), input, out);
  }
  bool validate(const T& val) const {
    return validateFn(ctx.get(), val);
  }
  std::string toString(const T& val) const {
    return toStringFn(ctx.get(), val);
  }
  explicit operator bool() const {
    return resolveFn != nullptr;
  }
};

// Empty braces yield a null ncclParamParser<T>; the ncclParam constructor
// detects this and fills from ncclParamDefault<T>().
#define NCCL_PARAM_DEFAULT \
  { \
  }

// Option Builder for enum or bitset types
//
// Usage:
//   auto opts = makeOptions(
//     makeOption<int32_t>("OFF",  0, "Disable feature"),
//     makeOption<int32_t>("ON",   1, "Enable feature"),
//     makeOption<int32_t>("AUTO", 2)      // no description
//   );
//   // opts is ncclOptionSet<int32_t, 3> containing:
//   //   {{"OFF", 0, "Disable feature"}, {"ON", 1, "Enable feature"}, {"AUTO", 2, nullptr}}
//   auto parser = ncclParamOneOf(opts);   // or ncclParamBitsetOf<EnumT>(opts)
template <typename T>
struct ncclOption {
  const char* name;
  T value;
  const char* desc;  // Per-option description (nullptr when no description)
};

// 固定大小 std::arrary 保存所有选项
// ncclOptionSet: Fixed-size option set (compile-time N, zero heap allocation)
template <typename T, size_t N>
struct ncclOptionSet {
  std::array<ncclOption<T>, N> options;

  constexpr const ncclOption<T>* begin() const {
    return options.data();
  }
  constexpr const ncclOption<T>* end() const {
    return options.data() + N;
  }
  constexpr size_t size() const {
    return N;
  }
};

// 检查选项不重复
// assert no two options share the same name.
template <typename T, size_t N>
inline void ncclOptionSetAssertUnique(const ncclOptionSet<T, N>& opts) {
  for (size_t i = 0; i < N - 1; i++) {
    for (size_t j = i + 1; j < N; j++) {
      if (std::strcmp(opts.options[i].name, opts.options[j].name) == 0) {
        WARN("PARAM: Duplicate option name \"%s\"", opts.options[i].name);
      }
    }
  }
}

// makeOption: Create an option (2-arg: no description)
template <typename T>
ncclOption<T> makeOption(const char* name, T value) {
  return {name, value, nullptr};
}

// makeOption: Create an option with description (3-arg)
template <typename T>
ncclOption<T> makeOption(const char* name, T value, const char* desc) {
  return {name, value, desc};
}

// makeOptions: Create a fixed-size option set from variadic arguments
template <typename T, typename... Args>
auto makeOptions(ncclOption<T> first, Args... rest) -> ncclOptionSet<T, 1 + sizeof...(Args)> {
  ncclOptionSet<T, 1 + sizeof...(Args)> opts{{{first, rest...}}};
  ncclOptionSetAssertUnique(opts);
  return opts;
}

#endif /* PARAM_PARSER_COMMON_H_INCLUDED */
