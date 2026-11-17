/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "nvtx.h"
#include "param.h"

// { profiler 中显示的名称, 枚举实际数值, flags：是否将枚举值看作位标志，如果为 1，可能出现 0x05（= 0x01 | 0x04），表示 "Read | Exec"}
static constexpr const nvtxPayloadEnum_t NvtxEnumRedSchema[] = {
  {"Sum", ncclSum, 0}, {"Product", ncclProd, 0}, {"Max", ncclMax, 0}, {"Min", ncclMin, 0}, {"Avg", ncclAvg, 0}
};

NCCL_PARAM(NvtxDisable, "NVTX_DISABLE", 0);

// 这个函数向 NVTX 注册一张枚举值到可读名称的映射表，让 profiler 在展示 ncclAllReduce、ncclReduce、
// ncclReduceScatter 的 payload 时，把 ncclRedOp_t 显示成 Sum、Max 这类文字，而不是原始整数 0、2
// Must be called before the first call to any reduction operation.
void initNvtxRegisteredEnums() {
  // 用户指定 NCCL_NVTX_DISABLE=1 显式禁用 NVTX 时直接返回，不注册
  // Register schemas and strings
  if (ncclParamNvtxDisable()) {
    return;
  }

  /*
  构造枚举属性结构体：
    + fieldMask：告诉 NVTX 这个结构体中哪些字段是真正填写了的，未设置位对应的字段被忽略。
      - ENTRIES：枚举条目数组
      - NUM_ENTRIES：条目数量
      - SIZE：枚举类型的字节大小
      - SCHEMA_ID：该枚举 schema 的 id
    + 第二个字段 NULL：对应枚举 schema 的名称，这里不需要，所以传空。
    + 
  */ 
  constexpr const nvtxPayloadEnumAttr_t eAttr{NVTX_PAYLOAD_ENUM_ATTR_ENTRIES | NVTX_PAYLOAD_ENUM_ATTR_NUM_ENTRIES |
                                                NVTX_PAYLOAD_ENUM_ATTR_SIZE | NVTX_PAYLOAD_ENUM_ATTR_SCHEMA_ID,
                                              NULL,
                                              NvtxEnumRedSchema,
                                              // std::extent<T>::value 返回数组第一维的元素个数，定义在 <type_traits>
                                              std::extent<decltype(NvtxEnumRedSchema)>::value,
                                              sizeof(ncclRedOp_t),
                                              // id 分配空间与结构体 schema 共用同一段编号，24 + STATIC_START
                                              NVTX_PAYLOAD_ENTRY_NCCL_REDOP,
                                              // 结构体尾部的保留/扩展字段，当前不使用
                                              nullptr};

  // nvtx3::domain::get<T>() 会为 nccl_domain 类型返回一个进程内单例的 domain handle。
  // domain 的意义是给所有 NCCL 的 NVTX 对象加上命名空间，让 profiler 里的 NCCL 数据与其他库（如 cuDNN、用户自己的标记）区分开
  nvtxPayloadEnumRegister(nvtx3::domain::get<nccl_domain>(), &eAttr);
}
