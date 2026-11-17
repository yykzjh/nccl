/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CORE_H_
#define _NCCL_DEVICE_CORE_H_
#include <nccl.h>
#include "coop.h"
#include "utility.h"

struct ncclDevComm;
typedef struct ncclDevComm ncclDevComm_t;

struct ncclTeam;
typedef struct ncclTeam ncclTeam_t;

// typedef struct ncclWindow_vidmem* ncclWindow_t; // in nccl.h
typedef struct ncclWindow_vidmem ncclWindow_vidmem_t;

struct ncclMultimemHandle;
typedef struct ncclMultimemHandle ncclMultimemHandle_t;

typedef uint32_t ncclDevResourceHandle;
typedef ncclDevResourceHandle ncclDevResourceHandle_t;

typedef uint32_t ncclGinSignal_t;
typedef uint32_t ncclGinCounter_t;

typedef struct {
  uint64_t opaque[2];
} ncclGinRequest_t;

struct ncclLsaBarrierHandle;
typedef struct ncclLsaBarrierHandle ncclLsaBarrierHandle_t;

struct ncclGinBarrierHandle;
typedef struct ncclGinBarrierHandle ncclGinBarrierHandle_t;

struct ncclLLA2AHandle;
typedef struct ncclLLA2AHandle ncclLLA2AHandle_t;

struct ncclTeam {
  int nRanks, rank, stride;
};

#if __cplusplus
template <typename T>
struct ncclSymPtr;
#endif

#if __cplusplus
struct ncclTeamTagWorld {};
struct ncclTeamTagLsa {};
struct ncclTeamTagRail {};
#endif

struct ncclDevCommRequirements;
typedef struct ncclDevCommRequirements ncclDevCommRequirements_t;

struct ncclDevResourceRequirements;
typedef struct ncclDevResourceRequirements ncclDevResourceRequirements_t;

struct ncclTeamRequirements;
typedef struct ncclTeamRequirements ncclTeamRequirements_t;

struct ncclCommProperties;
typedef struct ncclCommProperties ncclCommProperties_t;

typedef enum {
  NCCL_GIN_CONNECTION_NONE,
  NCCL_GIN_CONNECTION_FULL,
  NCCL_GIN_CONNECTION_RAIL,
} ncclGinConnectionType_t;

/*
设备通信器创建时所需的资源清单，也是 ncclCommQueryProperties 的输入侧对应结构。
流程是先用 ncclCommQueryProperties 查出能提供什么，再填这份 requirements 声明要什么，
最后交给 ncclDevCommCreate 兑现。
*/
struct ncclDevCommRequirements {
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
  ncclGinConnectionType_t ginConnectionType;
  // 默认 false。本 devComm 的上下文是否独占、不与其它 devComm 共享
  bool ginExclusiveContexts;
  int ginQueueDepth;  // 默认 0。GIN 队列深度，决定单个 QP 上可同时挂多少未完成请求
  int ginTrafficClass;  // 覆盖 GIN 流量的 traffic class，用于在 RoCE 上做 QoS 区分

  int worldGinBarrierCount;  // world 范围的 GIN 屏障数量，跨节点全连接语义，能被任意 rank 到任意 rank 的屏障使用

  /*
  语义是“要求强信号语义”。这是个能力协商位，不是性能优化位。legacy 指索引式信号这条老路径，新路径是 VA 信号。
  置假则使用强信号导致未定义行为，也就是说置假后内核若仍按强语义去用，行为是未定义，不是被忽略也不是报错。
  实践上只在确认内核不会用到强信号时才置假，收益是放宽对插件的能力要求。
  */
  // Set to false if GIN strong signals will not be needed by the kernels using this devComm (defaults to true).
  // When false, the use of GIN strong signals results in undefined behavior.
  bool ginStrongSignalsRequired;

  // 语义同上，针对 VA 信号，即以对称内存窗口地址而非信号编号寻址的那类信号
  // Set to false if GIN VA signals will not be needed by the kernels using this devComm (defaults to true).
  // When false, the use of GIN VA signals results in undefined behavior.
  bool ginVaSignalsRequired;
};

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
}
// clang-format on

struct ncclDevResourceRequirements {
  ncclDevResourceRequirements_t* next;
  size_t bufferSize, bufferAlign;
  ncclDevResourceHandle_t* outBufferHandle; // If non-null, target assigned during ncclDevCommCreate.
  int ginSignalCount;
  int ginCounterCount;
  ncclGinSignal_t* outGinSignalStart;
  ncclGinCounter_t* outGinCounterStart;
};

struct ncclTeamRequirements {
  ncclTeamRequirements_t* next;
  ncclTeam_t team;
  bool multimem;
  ncclMultimemHandle_t* outMultimemHandle; // If non-null, target assigned during ncclDevCommCreate.
};

#define NCCL_COMM_PROPERTIES_INITIALIZER \
  { \
    sizeof(ncclCommProperties_t),                  /* size */ \
    NCCL_API_MAGIC,                                /* magic */ \
    NCCL_VERSION_CODE,                             /* version */ \
  }

typedef enum {
  NCCL_GIN_TYPE_NONE = 0,
  NCCL_GIN_TYPE_PROXY = 2, // intentially not 1. Must match NCCL_NET_DEVICE_GIN_PROXY for backward compatibility
  NCCL_GIN_TYPE_GDAKI = 3, // intentially not 2. Must match NCCL_NET_DEVICE_GIN_GDAKI for backward compatibility
  NCCL_GIN_TYPE_GPI = 4, // Must match NCCL_NET_DEVICE_GIN_GPI
} ncclGinType_t;

/*
设备 API（Device API）对外的能力查询结果结构体，也是一份跨版本的 ABI 契约。
纯 host 侧的输出参数，由 ncclCommQueryProperties 负责填充，应用只读不写。
在设计上遵循 NCCL 设备 API 的统一范式，与同一文件里的 ncclDevCommRequirements（输入侧）配对：
  输入侧 ncclDevCommRequirements 用 NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER 初始化，声明“我要什么资源”。
  输出侧 ncclCommProperties 用 NCCL_COMM_PROPERTIES_INITIALIZER 初始化，回答“这个 communicator 能给什么”。
*/
struct ncclCommProperties {
  /* internal use only */
  size_t size;  // 初始化宏写入 sizeof(ncclCommProperties_t)
  unsigned int magic;  // 初始化宏写入 NCCL_API_MAGIC，专门拦“结构体没初始化、成员是野值”的调用
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
  /*
  isAllCudaP2p && WinEnable && cuMemEnable && (globalGinSupport != NONE || isOneLsaTeams) 合取。
  换句话说它要求“全 rank 之间的 CUDA P2P 都通、window 机制启用、cuMem 可用”，再叠加 GIN 或单 LSA 二者之一。
  */
  bool deviceApiSupport;
  /*
  comm->nvlsSupport && !comm->p2pCrossClique，NVLS 多播依赖 NVLink Switch 域内的地址翻译，
  跨 MNNVL clique 之后同一条多播地址不再映射到同一组 rank。
  */
  bool multimemSupport;
  // globalGinSupport != FULL ? NONE : ginState.ginType，要 FULL，即全互连 GIN，对应 requirements 里的 NCCL_GIN_CONNECTION_FULL
  ncclGinType_t ginType;
  // 表示整个 communicator 被切成几个 LSA team，取值来自 ncclDevrInitOnce 里的 comm->nRanks / devr->lsaSize，即“世界大小除以单个 LSA 队大小”。
  int nLsaTeams;
  bool hostRmaSupport;
  // globalGinSupport == NONE ? NONE : ginState.ginType，只要不是 NONE，即含 RAIL 部分连通，对应 NCCL_GIN_CONNECTION_RAIL
  ncclGinType_t railedGinType;
};

NCCL_EXTERN_C __host__ ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props);
NCCL_EXTERN_C __host__ ncclResult_t ncclDevCommCreate(ncclComm_t comm, ncclDevCommRequirements_t const* reqs,
                                                      ncclDevComm_t* outDevComm);
NCCL_EXTERN_C __host__ ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm_t const* devComm);

NCCL_EXTERN_C __host__ ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset,
                                                                 ncclMultimemHandle_t multimem, void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank,
                                                            void** outPtr);
NCCL_EXTERN_C __host__ ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer,
                                                             void** outPtr);

////////////////////////////////////////////////////////////////////////////////
// Team API:
#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamWorld(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamWorld(ncclComm_t comm);
#endif

#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamLsa(ncclComm_t comm);
#endif

NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE bool ncclTeamRankIsMember(ncclTeam_t a, ncclTeam_t b, int bPeer);
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToTeam(ncclTeam_t a, ncclTeam_t b, int bPeer);

#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToWorld(ncclDevComm const&, ncclTeam, int rank);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ int ncclTeamRankToWorld(ncclComm_t comm, ncclTeam_t team, int rank);
#endif

#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankToLsa(ncclDevComm const&, ncclTeam, int rank);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ int ncclTeamRankToLsa(ncclComm_t comm, ncclTeam_t team, int rank);
#endif

NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamInnerFactor(ncclTeam_t parent, int innerSize);
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam_t ncclTeamOuterFactor(ncclTeam_t parent, int innerSize);

// Interpret each team as a set of ranks. This function assumes that `subset`
// is a subset of `parent`. Thus the number of ranks in the set difference of
// `parent` minus `subset` is `super.nRanks - subset.nRanks`. Given `index` this
// function returns the index'th element of `parent` minus `subset`.
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE int ncclTeamRankInDifference(ncclTeam_t parent, ncclTeam_t subset, int index);

// Equivalent to ncclTeamOuterFactor of lsa team.
#if __cplusplus
NCCL_IR_EXTERN_C NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamRail(ncclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
NCCL_EXTERN_C __host__ ncclTeam_t ncclTeamRail(ncclComm_t comm);
#endif

// Get offset of resource buffer within `comm.resourceWindow`.
NCCL_EXTERN_C NCCL_HOST_DEVICE_INLINE size_t ncclGetResourceBufferOffset(ncclDevResourceHandle_t h);

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGetResourceBuffer(ncclDevComm const&, ncclDevResourceHandle);
#endif

////////////////////////////////////////////////////////////////////////////////
// Window API:

#if NCCL_CHECK_CUDACC
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLocalPointer(ncclWindow_t w, size_t offset);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
NCCL_DEVICE_INLINE void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset,
                                                                 ncclMultimemHandle mmHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const&);
#endif

#if NCCL_CHECK_CUDACC
// Convenience for combining ncclGet***Pointer() with resource handle.
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLocalPointer(ncclDevComm const&, ncclDevResourceHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaPointer(ncclDevComm const&, ncclDevResourceHandle,
                                                                          int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferPeerPointer(ncclDevComm const&, ncclDevResourceHandle,
                                                                           ncclTeam, int peer);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferMultimemPointer(
  ncclDevComm const&, ncclDevResourceHandle, ncclMultimemHandle);
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetResourceBufferLsaMultimemPointer(ncclDevComm const&,
                                                                                  ncclDevResourceHandle);
#endif

#endif // _NCCL_DEVICE_CORE_H_
