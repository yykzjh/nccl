/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#include <stdlib.h>
#include <stdint.h>
#include <algorithm> // For std::min/std::max
#include "nccl.h"

#if defined(NCCL_OS_LINUX)
#ifdef PROFAPI
#define NCCL_API(ret, func, args...) \
  // __attribute__((visibility("default"))) 为声明附加额外属性，函数符号在.so动态库中对外可见
  // __attribute__((alias(#func))) 附加函数别名，即p##func和func指向同一个函数
  // __attribute__((weak)) 表示弱符号，即如果其他地方有同名函数，则优先使用其他地方的函数，profiler 插件定义自己的 ncclAllReduce（强符号），自动覆盖 NCCL 的弱实现
  extern "C" __attribute__((visibility("default"))) __attribute__((alias(#func))) ret p##func(args); \
  extern "C" __attribute__((visibility("default"))) __attribute__((weak)) ret func(args)
#else
#define NCCL_API(ret, func, args...) extern "C" __attribute__((visibility("default"))) ret func(args)
#endif // end PROFAPI
#else
/* Windows and other non-Linux: use standard variadic macro (no visibility attribute) */
#define NCCL_API(ret, func, ...) extern "C" ret func(__VA_ARGS__)
#endif // end NCCL_OS_LINUX

#include "debug.h"
#include "checks.h"
#include "cudawrap.h"
#include "alloc.h"
#include "utils.h"
#include "param.h"
#include "nvtx.h"

#endif // end include guard
