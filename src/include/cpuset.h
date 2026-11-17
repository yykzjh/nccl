/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CPUSET_H_
#define NCCL_CPUSET_H_

#include "nccl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef NCCL_OS_LINUX
#include <sched.h>
#elif defined(NCCL_OS_WINDOWS)
#define CPU_SETSIZE 64 // Windows uses DWORD_PTR for affinity
#endif

// Convert local_cpus, e.g. 0003ff,f0003fff to ncclAffinity.
// The bitmask is divided into chunks of 32 bits, each of them represented by 8 hex number.
/*
bitmask 被切成 32 位一段、每段用 8 个十六进制数表示，逗号分隔，最高位段在最左。
所以 0003ff,f0003fff 里 0003ff 是高位段（管 CPU 32–63）、f0003fff 是低位段（管 CPU 0–31）。
*/
// 每个 chunk 用 32 位（uint32_t）表示
#define U32_LEN 32 // using uint32_t
// 需要几个 chunk，Linux 1024/32 = 32 个、Windows 64/32 = 2 个
#define CPU_SET_N_U32 (CPU_SETSIZE / U32_LEN)


// 把 Linux sysfs 的十六进制 CPU 掩码字符串（例如 0003ff,f0003fff）解析成 ncclAffinity 集合，最终供 ncclTopoGetCpuAffinity 绑核用。
static ncclResult_t ncclStrToCpuset(const char* maskStr, ncclAffinity* set) {
  // 一个 chunk 数组（Linux 32 个、Windows 2 个）清零，用来暂存解析出的各 32 位段
  uint32_t cpumasks[CPU_SET_N_U32] = {0};

  // 字符串 → chunk 数组（高位在前，倒着填）
  // transform the string into an array of 32 bit masks, starting with the highest mask
  int m = CPU_SET_N_U32;  // 数组倒着填写
  char* str = strdup(maskStr);  // 临时拷贝字符串
  char* token = strtok(str, ",");  // 逗号分隔
  while (token != NULL && m > 0) {
    cpumasks[--m] = strtoul(token, NULL, /*base = hex*/ 16);
    token = strtok(NULL, ",");
  }
  free(str);

  // chunk 数组 → CPU 集合（从最低有效段起，展开每一位）
  // list all the CPUs as part of the CPU set, starting with the lowest mask (= current value of m)
  ncclOsCpuZero(*set);  // 清空 cpu mask
  for (int a = 0; (a + m) < CPU_SET_N_U32; a++) {  // 从有数据的 chunk 开始遍历
    // each mask is U32_LEN CPUs, list them all if the bit is on
    for (int i = 0; i < U32_LEN; ++i) {  // 遍历每个 chunk 中的 bit
      // 该 bit 对应的 cpu 为 1，调用 glibc 宏设置 cpu mask
      if (cpumasks[a + m] & (1UL << i)) ncclOsCpuSet(*set, i + a * U32_LEN);
    }
  }
  return ncclSuccess;
}

static char* ncclCpusetToRangeStr(ncclAffinity* mask, char* str, size_t len) {
  int c = 0;
  int start = -1;
  // Iterate through all possible CPU bits plus one extra position
  for (int cpu = 0; cpu <= CPU_SETSIZE; cpu++) {
    int isSet = (cpu == CPU_SETSIZE) ? 0 : ncclOsCpuIsSet(*mask, cpu);
    // Start of a new range
    if (isSet && start == -1) {
      start = cpu;
    }
    // End of a range, add comma between ranges
    if (!isSet && start != -1) {
      if (cpu - 1 == start) {
        c += snprintf(str + c, len - c, "%s%d", c ? "," : "", start);
      } else {
        c += snprintf(str + c, len - c, "%s%d-%d", c ? "," : "", start, cpu - 1);
      }
      if (c >= len - 1) break;
      start = -1;
    }
  }
  if (c == 0) str[0] = '\0';
  return str;
}

static ncclResult_t ncclStrListToCpuset(const char* userStr, ncclAffinity* mask) {
  // reset the CPU set
  ncclOsCpuZero(*mask);
  const char delim[] = ",";
  char* str = strdup(userStr);
  char* token = strtok(str, delim);
  while (token != NULL) {
    uint64_t cpu = strtoull(token, NULL, 0);
    ncclOsCpuSet(*mask, cpu);
    token = strtok(NULL, delim);
  }
  free(str);
  return ncclSuccess;
}

static ncclResult_t ncclCpusetToStrList(ncclAffinity* mask, char* str, size_t len) {
  if (len == 0) return ncclSuccess;
  str[0] = '\0';
  int count = 0;
  for (uint64_t id = 0; id < CPU_SETSIZE; ++id) {
    if (ncclOsCpuIsSet(*mask, id)) {
      snprintf(str + strlen(str), len - strlen(str), "%s%lu", (count++ == 0) ? "" : ",", id);
    }
  }
  return ncclSuccess;
}

#endif
