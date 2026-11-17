/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_HOST_H_
#define _NCCL_DEVICE_HOST_H_

#include "impl/comm__types.h"

typedef struct ncclDevResourceRequirements {
  struct ncclDevResourceRequirements* next;
  size_t bufferSize, bufferAlign;
  ncclDevResourceHandle_t* outBufferHandle; // If non-null, target assigned during ncclDevCommCreate.
  int ginSignalCount;
  int ginCounterCount;
  ncclGinSignal_t* outGinSignalStart;
  ncclGinCounter_t* outGinCounterStart;
} ncclDevResourceRequirements_t;

typedef struct ncclTeamRequirements {
  struct ncclTeamRequirements* next;
  ncclTeam_t team;
  bool multimem;
  ncclMultimemHandle_t* outMultimemHandle; // If non-null, target assigned during ncclDevCommCreate.
} ncclTeamRequirements_t;

/*
设备通信器创建时所需的资源清单，也是 ncclCommQueryProperties 的输入侧对应结构。
流程是先用 ncclCommQueryProperties 查出能提供什么，再填这份 requirements 声明要什么，
最后交给 ncclDevCommCreate 兑现。
*/
typedef struct ncclDevCommRequirements {
  /* attributes that users should never touch. */
  size_t size;  // 调用方结构体的真实字节数
  unsigned int magic;  // 默认值 NCCL_API_MAGIC，初始化标记，用于分辨“这是被初始化宏填过的结构体”和“这是一块没清的野内存”
  unsigned int version;  // 默认值 NCCL_VERSION_CODE，即应用编译期的头文件版本

  /* attributes that users are able to customize. */
  ncclDevResourceRequirements_t* resourceRequirementsList;  // 通用资源申请书，每个节点声明一块带对齐要求的缓冲，并可顺带申请信号与计数器
  ncclTeamRequirements_t* teamRequirementsList;  // 按 team 申请多播句柄

  // 是否要求 LSA team 内的所有 rank 支持 multimem
  bool lsaMultimem; // Enable multimem on lsa team

  /*
  复合屏障数量，一次申请三份资源
    hybridLsaBarrier 走 ncclLsaBarrierCreateRequirement，LSA 范围的内存屏障。
    hybridRailGinBarrier 走 ncclGinBarrierCreateRequirement，rail 范围的网络屏障，信号数 n * railTeam.nRanks。
    hybridWorldGinBarrier 走 ncclGinBarrierCreateRequirement，world 范围的网络屏障，但当连接类型为 NCCL_GIN_CONNECTION_RAIL 时数量被强制置 0。
  */
  int barrierCount;
  int lsaBarrierCount;  // 纯 LSA 范围的内存屏障数量，不涉及网络
  int railGinBarrierCount;  // 纯 rail 范围的网络屏障数量，只在 rail team 内同步，不要求全互连

  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;

  bool ginForceEnable;  // 历史兼容开关，默认 false，已废弃

  // GIN 上下文数量，默认 4，注释明示只是 hint
  int ginContextCount; // This is a hint, the actual context count in the devcomm may not match.
  // 申请多少个索引式网络信号，可用编号是 [0, n)。信号是 GIN 侧最基础的完成通知原语，内核用它做跨节点到达确认
  int ginSignalCount; // Guaranteed to start at id=0
  // 申请多少个网络计数器，用于累计式而非一次性通知的语义
  int ginCounterCount; // Guaranteed to start at id=0
  /*
  默认 NCCL_GIN_CONNECTION_NONE。三态含义是 NONE 不启用 GIN，FULL 每个 rank 与所有其他 rank 建连，
  RAIL 只与同 rail 的 rank 建连。它是是否启用 GIN 的真正开关，影响面很广。
  */
  // 以上是 2.30.7 的三种基础模式；当前版本还支持 CUSTOM_STRIDE / CUSTOM_ARRAY。
  ncclGinConnectionType_t ginConnectionType;
  // 默认 false。本 devComm 的上下文是否独占、不与其它 devComm 共享
  bool ginExclusiveContexts;
  int ginQueueDepth;  // 默认 0。GIN 队列深度，决定单个 QP 上可同时挂多少未完成请求
  int ginTrafficClass;  // 覆盖 GIN 流量的 traffic class，用于在 RoCE 上做 QoS 区分

  int worldGinBarrierCount;  // world 范围的 GIN 屏障数量，跨节点全连接语义，能被任意 rank 到任意 rank 的屏障使用

  // Set to false if GIN strong signals will not be needed by the kernels using this devComm (defaults to true).
  // When false, the use of GIN strong signals results in undefined behavior.
  /*
  语义是“要求强信号语义”。这是个能力协商位，不是性能优化位。legacy 指索引式信号这条老路径，新路径是 VA 信号。
  置假则使用强信号导致未定义行为，也就是说置假后内核若仍按强语义去用，行为是未定义，不是被忽略也不是报错。
  实践上只在确认内核不会用到强信号时才置假，收益是放宽对插件的能力要求。
  */
  bool ginStrongSignalsRequired;

  // Set to false if GIN VA signals will not be needed by the kernels using this devComm (defaults to true).
  // When false, the use of GIN VA signals results in undefined behavior.
  // 语义同上，针对 VA 信号，即以对称内存窗口地址而非信号编号寻址的那类信号
  bool ginVaSignalsRequired;

  // Stride of ranks to connect for GIN if ginConnectionType is NCCL_GIN_CONNECTION_CUSTOM_STRIDE.
  int ginCustomStride;

  ncclGinType_t ginType;
  // If true, initialize the devComm assuming the version of the device code is the same
  // as the runtime version of the NCCL library (i.e., the device code is JIT-compiled).
  // When true, the DevComm must be allocated according to devCommRuntimeVersionSize.
  bool useRuntimeVersion;

  int cftCaps; // Bitmask of ncclCftCap_t values
  int cftBarrierCount;

  // List of ginCustomArrayCount world ranks to connect to. Only valid if ginConnectionType
  // is NCCL_GIN_CONNECTION_CUSTOM_ARRAY. Both ranks of a pair must request the connection (peer j
  // appearing in rank i's ginCustomArray requires peer i to appear in rank j's ginCustomArray).
  int* ginCustomArray;
  int ginCustomArrayCount;
} ncclDevCommRequirements_t;

// clang-format off: maintain hand-formatted code
#define NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER {                               \
    sizeof(ncclDevCommRequirements_t),           /* size */                    \
    NCCL_API_MAGIC,                              /* magic */                   \
    NCCL_VERSION_CODE,                           /* version */                 \
    NULL,                                        /* resourceRequirementsList*/ \
    NULL,                                        /* teamRequirementsList */    \
    false,                                       /* lsaMultimem */             \
    0,                                           /* barrierCount */            \
    0,                                           /* lsaBarrierCount */         \
    0,                                           /* railGinBarrierCount */     \
    0,                                           /* lsaLLA2ABlockCount */      \
    0,                                           /* lsaLLA2ASlotCount */       \
    false,                                       /* ginForceEnable */          \
    4,                                           /* ginContextCount */         \
    0,                                           /* ginSignalCount */          \
    0,                                           /* ginCounterCount */         \
    NCCL_GIN_CONNECTION_NONE,                    /* ginConnectionType */       \
    false,                                       /* ginExclusiveContexts */    \
    0,                                           /* ginQueueDepth */           \
    NCCL_CONFIG_UNDEF_INT,                       /* ginTrafficClass */         \
    0,                                           /* worldGinBarrierCount */    \
    true,                                        /* ginStrongSignalsRequired */ \
    true,                                        /* ginVaSignalsRequired */     \
    1,                                           /* ginCustomStride      */     \
    NCCL_GIN_TYPE_NONE,                          /* ginType */                  \
    false,                                       /* useRuntimeVersion */        \
    NCCL_CFT_NONE,                               /* cftCaps */                 \
    0,                                           /* cftBarrierCount */         \
    NULL,                                        /* ginCustomArray */         \
    0,                                           /* ginCustomArrayCount */    \
}
// clang-format on

/*
设备 API（Device API）对外的能力查询结果结构体，也是一份跨版本的 ABI 契约。
纯 host 侧的输出参数，由 ncclCommQueryProperties 负责填充，应用只读不写。
在设计上遵循 NCCL 设备 API 的统一范式，与同一文件里的 ncclDevCommRequirements（输入侧）配对：
  输入侧 ncclDevCommRequirements 用 NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER 初始化，声明“我要什么资源”。
  输出侧 ncclCommProperties 用 NCCL_COMM_PROPERTIES_INITIALIZER 初始化，回答“这个 communicator 能给什么”。
*/
typedef struct ncclCommProperties {
  /* internal use only */
  size_t size;  // 初始化宏写入 sizeof(ncclCommProperties_t)
  unsigned int magic;  // 初始化宏写入 NCCL_API_MAGIC，专门拦“结构体没初始化、成员是野值”的调用
  // 下方 40/52 字节的例子来自 2.30.7；当前结构大小以 sizeof(ncclCommProperties_t) 为准。
  /*
  初始化宏写入 NCCL_VERSION_CODE，version 承担两个职责：
    兼容分发。ncclCommQueryProperties 把它交给 getNcclVersionCompat，后者在 devCommCompat[] 表里按 [minVersion, maxVersion] 区间线性查找，
  命中哪条就用哪条的回调集。所以它记录的是应用编译时头文件的版本，不是运行时库的版本，两者可能不同，这正是整套兼容逻辑存在的前提。
    越界门禁。老结构体只有 40 字节，新结构体 52 字节，ncclCommQueryProperties 里用 props->version > NCCL_VERSION(2, 29, 3)
  来决定要不要写尾部三个新字段。这个判断与业务无关，纯粹是防止写穿应用侧那块较小的内存。
  */
  unsigned int version;

  /* attributes for users. */
  int rank;
  int nRanks;
  int cudaDev;
  int nvmlDev;
  bool deviceApiSupport;
  bool multimemSupport;
  ncclGinType_t ginType;
  // 表示整个 communicator 被切成几个 LSA team，取值来自 ncclDevrInitOnce 里的 comm->nRanks / devr->lsaSize，即“世界大小除以单个 LSA 队大小”。
  int nLsaTeams;
  bool hostRmaSupport;
  ncclGinType_t railedGinType;
  uint64_t commHash;
  int ginMinStride;
  ncclGinConnectionType_t ginConnectionType;
  bool ginSupport[64]; // ginSupport[i] is true if gin type i is supported
  size_t devCommRuntimeVersionSize;
  bool cftSupport;
  bool cftMulticastSupport;
  bool cftCountedSupport;
} ncclCommProperties_t;

#define NCCL_COMM_PROPERTIES_INITIALIZER \
  { \
    sizeof(ncclCommProperties_t),                  /* size */ \
    NCCL_API_MAGIC,                                /* magic */ \
    NCCL_VERSION_CODE,                             /* version */ \
  }

////////////////////////////////////////////////////////////////////////////////
// Device communicator API:

NCCL_EXTERN_C ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props);
NCCL_EXTERN_C ncclResult_t ncclDevCommCreate(ncclComm_t comm, ncclDevCommRequirements_t const* reqs,
                                             ncclDevComm_t* outDevComm);
NCCL_EXTERN_C ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm_t const* devComm);

////////////////////////////////////////////////////////////////////////////////
// Window API:

// VA pointer based query functions for host code
NCCL_EXTERN_C ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr);
NCCL_EXTERN_C ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset,
                                                        ncclMultimemHandle_t multimem, void** outPtr);
NCCL_EXTERN_C ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr);
NCCL_EXTERN_C ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr);

// CFT handle based query functions for host code
NCCL_EXTERN_C ncclResult_t ncclGetMultimemDeviceLeInfo(ncclWindow_t window, size_t offset, ncclCftLeId* leId,
                                                       size_t* leOffset);
NCCL_EXTERN_C ncclResult_t ncclGetCftDeviceLeInfo(ncclWindow_t window, size_t offset, int peerCft, ncclTeam_t cftTeam,
                                                  ncclCftLeId* leId, size_t* leOffset);
NCCL_EXTERN_C ncclResult_t ncclGetPeerDeviceLeInfo(ncclWindow_t window, size_t offset, int peerWorld, ncclCftLeId* leId,
                                                   size_t* leOffset);

////////////////////////////////////////////////////////////////////////////////
// Team API:

NCCL_EXTERN_C ncclTeam_t ncclTeamWorld(ncclComm_t comm);
NCCL_EXTERN_C ncclTeam_t ncclTeamLsa(ncclComm_t comm);
#if __cplusplus
NCCL_EXTERN_C ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode = NCCL_CFT_TEAM_FLAT);
#else
NCCL_EXTERN_C ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode);
#endif
NCCL_EXTERN_C ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm);
NCCL_EXTERN_C int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank);
NCCL_EXTERN_C int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank);
NCCL_EXTERN_C ncclTeam_t ncclTeamRail(ncclComm_t comm);

////////////////////////////////////////////////////////////////////////////////
// Device resource requirement API:

NCCL_EXTERN_C ncclResult_t ncclLsaBarrierCreateRequirement(
  ncclTeam_t team, int nBarriers, ncclLsaBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);
NCCL_EXTERN_C ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                                           ncclGinBarrierHandle_t* outHandle,
                                                           ncclDevResourceRequirements_t* outReq);
NCCL_EXTERN_C ncclResult_t ncclCftBarrierCreateRequirement(
  ncclTeam_t team, int nBarriers, ncclCftBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);
NCCL_EXTERN_C int ncclLLA2ACalcSlots(int maxElts, int maxEltSize);
NCCL_EXTERN_C ncclResult_t ncclLLA2ACreateRequirement(int nBlocks, int nSlots, ncclLLA2AHandle_t* outHandle,
                                                      ncclDevResourceRequirements_t* outReq);

#if defined(NCCL_OS_WINDOWS)
NCCL_EXTERN_C ncclResult_t ncclGinOutboxCreateRequirement(int nBlocks, int size_log2, ncclGinOutboxHandle* outHandle,
                                                          ncclDevResourceRequirements* outReq);
NCCL_EXTERN_C ncclResult_t ncclGinInboxA2ACreateRequirement(
  ncclTeam peers, int nBlocks, int size_log2, ncclGinInboxA2AHandle* outHandle, ncclDevResourceRequirements* outReq);
#endif

#endif // _NCCL_DEVICE_HOST_H_
