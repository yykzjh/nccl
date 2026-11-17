/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_V14_H_
#define GIN_V14_H_
#include "nccl_net.h"

typedef struct {
  /*
  信号强度 —— supportsStrongSignals(强 vs 弱)
    weak signal：只保证与它绑定的那一次 put 已完成，单个 put 的完成通知
    strong signal：还对同一 context/peer 上此前的操作建立更强的完成顺序，带 release 栅栏的阶段 barrier
  */
  bool supportsStrongSignals;
  /*
  信号寻址方式 —— supportsVASignals(INDEXED vs VA)，描述信号计数器落在哪个地址
    type	                          信号落点	                        描述符字段
    NCCL_GIN_SIGNAL_TYPE_NONE	      不发信号	                            —
    NCCL_GIN_SIGNAL_TYPE_INDEXED	  NCCL 预分配的信号槽池里的第 N 个	   indexedSignal.signalId
    NCCL_GIN_SIGNAL_TYPE_VA	        用户 window 里的任意虚拟地址	      vaSignal.{signalWindow, signalOffset, ncclWindow}
    INDEXED(基线)：在建 devComm 时申请 ginSignalCount 个信号槽，NCCL 统一管理这组计数器，put 时用 signalId 指定“递增第几号槽”。
  这是经典模型，一组由框架托管的固定计数器。
    VA(Virtual Address，加强版)：信号不再落在托管槽池，而是落在用户指定 window 内的某个虚拟地址(signalWindow + signalOffset)。
  device API 里对应 ncclGin_VASignalInc/Add、ncclGin_StrongVASignalInc、ncclGin_WeakVASignalInc 等，
  注意它们都比 INDEXED 版多带 signalWindow/signalOffset 两个参数。
    VA 信号的好处是灵活：可以把完成计数器放在任意内存布局里(例如和数据共处一个 window、或按用户自定义结构排布)，而不局限于 NCCL 的槽池。
  代价是实现更重，proxy 后端里 VA 信号必须单独占一个 GFD(GIN 工作描述符)：
  */
  bool supportsVASignals;
} ncclGinProperties_v14_t;

typedef struct {
  int nSignals;
  int nCounters;
  int nContexts;
  int queueDepth;
  int trafficClass;
  int backendVersion;
  int rankStride;
} ncclGinConfig_v14_t;

typedef struct {
  // Name of the GIN support (mainly for logs)
  const char* name;
  // Initialize the GIN support.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing GIN operations.
  ncclResult_t (*devices)(int* ndev);
  // Get the GIN properties.
  ncclResult_t (*getGinProperties)(ncclGinProperties_v14_t* ginProps);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Create a group for GIN operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Create device-side GIN context. devHandle will be passed to device code.
  ncclResult_t (*createContext)(void* collComm, ncclGinConfig_v14_t* config, void** ginCtx,
                                ncclNetDeviceHandle_v11_t** devHandle);
  // Collective memory registration
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle,
                           void** ginHandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                 uint64_t mrFlags, void** mhandle, void** ginHandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  // Close and free collective comm objects
  ncclResult_t (*destroyContext)(void* ginCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Progress function. Will be called if devHandle.needsProxyProgress=1.
  ncclResult_t (*ginProgress)(void* ginCtx);

  // Query the last error for the GIN support. Particularly important when ginProgress is not used, to report errors.
  ncclResult_t (*queryLastError)(void* ginCtx, bool* hasError);

  // Finalize the GIN support
  ncclResult_t (*finalize)(void* ctx);
} ncclGin_v14_t;
#endif // end include guard
