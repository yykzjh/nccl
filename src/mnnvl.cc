/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "mnnvl.h"
#include "transport.h"
#include <cuda.h>
#include "cudawrap.h"

// Determine if MNNVL support is available
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) {
  /*
  MNNVL 依赖 CUDA 版本支持 CUMEM VMM
  FABRIC handle 是 CUDA VMM(cuMem)独有的能力，只有 cuMemCreate/cuMemExportToShareableHandle 这套 VMM API 才能产出跨节点可共享的 FABRIC 句柄。
  传统 cudaMalloc + legacy IPC 做不到跨节点，没有 VMM 就谈不上 MNNVL，直接静默返回。
  */
  // MNNVL requires cuMem to be enabled
  if (!ncclCuMemEnable()) return ncclSuccess;

  // MNNVL also requires FABRIC handle support
  int cudaDev;
  int flag = 0;
  CUdevice currentDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  // 查询 CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED 属性判断 CUDA 是否支持 fabric handle 类型
  // Ignore error if CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not supported
  (void)CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
  if (!flag) return ncclSuccess;
  /*
  所有 rank 的 fabric 都必须注册完成。
  MNNVL 是跨节点协作，只要有一个 rank 的 fabric 没就绪(还在 IN_PROGRESS 或 NOT_SUPPORTED)，
  整个 comm 就不能用 MNNVL，这是集合操作的一致性要求。
  */
  // Check that all ranks have initialized the fabric fully
  for (int i = 0; i < comm->nRanks; i++) {
    if (comm->peerInfo[i].fabricInfo.state != NVML_GPU_FABRIC_STATE_COMPLETED) return ncclSuccess;
  }

  // clique / NVL 域计算 
  // Determine our MNNVL domain/clique and NVL domain size
  NCCLCHECK(ncclCalloc(&comm->clique.ranks, comm->nRanks));  // 初始化情况，所有 ranks 都与当前 rank 在一个 clique
  // 先初始化为当前 rank 的 cliqueId
  comm->clique.id = comm->peerInfo[comm->rank].fabricInfo.cliqueId;
  comm->nvlDomainSize = 0;
  for (int i = 0; i < comm->nRanks; i++) {
    nvmlGpuFabricInfoV_t* fabricInfo1 = &comm->peerInfo[comm->rank].fabricInfo;  // 固定获取当前 rank 的 fabricInfo
    nvmlGpuFabricInfoV_t* fabricInfo2 = &comm->peerInfo[i].fabricInfo;  // 遍历所有对端 rank fabricInfo
    // Check if the cluster UUID and cliqueId match
    // A zero UUID means we don't have MNNVL fabric info - disable MNNVL
    unsigned long uuid0 = 0;
    unsigned long uuid1 = 0;
    /*
    clusterUuid 是 16 字节，没法直接和 0 比。这里把它当两个 unsigned long(各 8 字节)读出来，| 后判 == 0，
    等价于"128 位 UUID 是否全零"。全零意味着该 rank 没有 fabric 信息。
    */
    memcpy(&uuid0, fabricInfo2->clusterUuid, sizeof(uuid0));
    memcpy(&uuid1, fabricInfo2->clusterUuid + sizeof(uuid0), sizeof(uuid1));
    if ((uuid0 | uuid1) == 0) return ncclSuccess;
    // 比较两端 rank 是否是相同 NVL 域
    // Check if same NVL domain (clusterUuid match)
    if (memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) {
      comm->nvlDomainSize++;
      // 且比较两端 rank 是否是相同 clique
      // Also check if same clique (cliqueId match)
      if (fabricInfo1->cliqueId == fabricInfo2->cliqueId) {
        if (i == comm->rank) {
          comm->cliqueRank = comm->clique.size;  // 如果 i 就是当前 rank，将此时 clique 大小作为当前 rank 在 clique 中的位次
        }
        comm->clique.ranks[comm->clique.size++] = i;  // 将 i 记录到当前 clique 中
      }
    }
  }

  /*
  clique 和 NVL 域里都只有当前 rank，没有跨节点 NVLink 可言，静默返回。
  但 ncclCommSplit 后可能 clique.size==1 但 nvlDomainSize>1)，
  这种 cross-clique 场景仍要启用 MNNVL，所以用 &&(两者都 ≤1 才退)。
  */
  // ncclCommSplit: clique.size may be 1 while nvlDomainSize > 1; still enable MNNVL.
  if (comm->clique.size <= 1 && comm->nvlDomainSize <= 1) return ncclSuccess;

  /*
  IMEX (Internode Memory Exchange)，NVIDIA 让 GPU 显存跨节点经 NVLink 共享的服务/守护进程。
  /dev/nvidia-caps-imex-channels 是 IMEX 通道设备文件，nvidia-imex-ctl -N 是它的管理工具。
  */
  // Check that FABRIC handles can be exported & imported by IMEX
  {
    void* ptr = NULL;
    CUmemGenericAllocationHandle handle;
    ncclCuDesc cuDesc;
    CUresult err;

    /*
    分配一块 FABRIC 句柄兼容显存(显式传 CU_MEM_HANDLE_TYPE_FABRIC，CUDA_IPC_MIN 是 IPC 分配最小粒度)，
    cuMemExportToShareableHandle 导出成 ncclCuDesc(即 64 字节 FABRIC 句柄)，再 cuMemImportFromShareableHandle 把它导回。
    本地 export+import 能跑通，就证明 FABRIC 句柄机制 + IMEX 守护进程是健康的。
    */
    // Allocate FABRIC handle compatible memory
    ncclResult_t ret =
      ncclCuMemAlloc(&ptr, &handle, CU_MEM_HANDLE_TYPE_FABRIC, CUDA_IPC_MIN, comm->memManager, ncclMemOffload);
    if (ret != ncclSuccess) {
      // Return an error if this is a MNNVL capable system but FABRIC handles are not supported
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX channel configuration "
           "(/dev/nvidia-caps-imex-channels). Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
           comm->clique.size);
      return ncclSystemError;
    }
    // FABRIC 兼容显存句柄导出为 64 字节 FABRIC 句柄 cuDesc
    err = CUPFN(cuMemExportToShareableHandle(&cuDesc, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    // 导出或者导入异常
    if (err != CUDA_SUCCESS ||
        (err = CUPFN(cuMemImportFromShareableHandle(&handle, &cuDesc, CU_MEM_HANDLE_TYPE_FABRIC))) != CUDA_SUCCESS) {
      const char* errStr;
      (void)pfn_cuGetErrorString(err, &errStr);
      NCCLCHECK(ncclCuMemFree(ptr, comm->memManager));
      // Return an error if this is a MNNVL capable system but it's not working
      WARN("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX configuration "
           "(nvidia-imex-ctl -N). Set NCCL_MNNVL_ENABLE=0 to ignore this issue.",
           comm->clique.size);
      return ncclSystemError;
    }
    NCCLCHECK(ncclCuMemFree(ptr, comm->memManager));

    // 全局句柄类型切到 FABRIC，之后所有 VMM 分配都用 FABRIC 句柄
    // Force the CUMEM handle type to be FABRIC for MNNVL
    ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
    comm->MNNVL = 1;  // 启用 MNNVL
    INFO(NCCL_INIT, "MNNVL %d cliqueId %x cliqueSize %d cliqueRank %d nvlDomainSize %d", comm->MNNVL, comm->clique.id,
         comm->clique.size, comm->cliqueRank, comm->nvlDomainSize);
  }
  return ncclSuccess;
}
