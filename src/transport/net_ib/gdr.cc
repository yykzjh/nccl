/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

// Detect whether GDR can work on a given NIC with the current CUDA device
// Returns :
// ncclSuccess : GDR works
// ncclSystemError : no module or module loaded but not supported by GPU

/*
int access(const char *pathname, int mode); 是一个 POSIX 标准系统调用（声明在 <unistd.h>），
作用是检查当前进程对某个文件是否具有某种访问权限，或者文件是否存在。它不读写文件内容，只做一次权限/存在性探测。
  pathname：要检查的文件路径。
  mode：想检查的访问类型，用几个预定义常量表示，可以用 | 组合：
    mode常量	含义
    F_OK	   文件是否存在（只测存在性，不关心权限）
    R_OK	   是否可读
    W_OK	   是否可写
    X_OK	   是否可执行
  返回值：
    成功（检查的条件成立）返回 0。
    失败（不成立，或出错）返回 -1，并设置 errno（比如 ENOENT 表示文件不存在、EACCES 表示无权限）。
*/
#define KNL_MODULE_LOADED(a) ((access(a, F_OK) == -1) ? 0 : 1)
static int ncclIbGdrModuleLoaded = 0; // 1 = true, 0 = false

static void ibGdrSupportInitOnce() {
  // 检查 nv_peer_mem module 是否被加载了
  // Check for the nv_peer_mem module being loaded
  ncclIbGdrModuleLoaded = KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem/version") ||
                          KNL_MODULE_LOADED("/sys/kernel/mm/memory_peers/nv_mem_nc/version") ||
                          KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

// 判断 NET 是否支持 GDR
// Returns ncclSuccess if any of the peermem modules are loaded.
ncclResult_t ncclIbGdrSupport() {
  static std::once_flag once;
  std::call_once(once, ibGdrSupportInitOnce);
  if (!ncclIbGdrModuleLoaded) return ncclSystemError;
  return ncclSuccess;
}

// 检查 nvidia_peermem module 是否加载
static int ncclIbPeerMemModuleLoaded = 0; // 1 = true, 0 = false
static void ibPeerMemSupportInitOnce() {
  ncclIbPeerMemModuleLoaded = KNL_MODULE_LOADED("/sys/module/nvidia_peermem/version");
}

// 判断 NET 是否支持 PeerMem
// Returns ncclSuccess if nvidia_peermem module is loaded. Does not check legacy implementations of nv_peer_mem
// (e.g. nv_mem, nv_mem_nc)
ncclResult_t ncclIbPeerMemSupport() {
  static std::once_flag once;
  std::call_once(once, ibPeerMemSupportInitOnce);
  if (!ncclIbPeerMemModuleLoaded) return ncclSystemError;
  return ncclSuccess;
}

// 多个线程可能同时对不同设备调 ncclIbDmaBufSupport，每线程各存一份，互不干扰
static thread_local int ibDmaSupportInitDev; // which device to init, must be thread local
// 用一次假调用探测内核+网卡到底支不支持 DMA-BUF 注册
static void ibDmaBufSupportInitOnce() {
  ncclResult_t res;
  int dev_fail = 0;

  // 取其中一个合并/虚拟设备（多轨聚合，vProps 里 ndevs + devs[8] 把多块物理网卡聚成一个逻辑设备）
  // This is a physical device, not a virtual one, so select from ibDevs
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  // 通过合并设备中第一个设备的索引在所有设备表中取到该设备对象
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  struct ibv_pd* pd;
  struct ibv_context* ctx = ibDev->context;
  /*
  reg_dmabuf_mr 需要一个 protection domain 才能尝试，所以临时 alloc_pd 一个，探测完立刻 dealloc_pd。
  成功路径和 dev_fail 路径都在 dealloc 之后才分叉，PD 不会泄漏。
  */
  NCCLCHECKGOTO(wrap_ibv_alloc_pd(&pd, ctx), res, failure);  // 临时分配一个 PD
  /*
  故意传一个非法的 fd = -1 去调 ibv_reg_dmabuf_mr，这个调用注定失败——但失败的方式恰恰暴露了信息：
    如果内核/rdma-core 根本不支持 DMA-BUF 注册 → 返回 EOPNOTSUPP（操作不支持）或 EPROTONOSUPPORT（协议不支持）。
    如果支持 DMA-BUF、只是我传的 fd 无效 → 返回 EBADF（坏文件描述符）。
  */
  // Test kernel DMA-BUF support with a dummy call (fd=-1)
  (void)wrap_direct_ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/, 0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
  // ibv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF otherwise)
  dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  NCCLCHECKGOTO(wrap_ibv_dealloc_pd(pd), res, failure);  // 释放 PD
  // stop the search and goto failure
  if (dev_fail) goto failure;
  ibDev->dmaBufSupported = 1;
  return;
failure:
  ibDev->dmaBufSupported = -1;
  return;
}

// 检查指定 GPU 设备是否支持 DMA-Buf
// Detect whether DMA-BUF support is present in the kernel
// Returns :
// ncclSuccess : DMA-BUF support is available
// ncclSystemError : DMA-BUF is not supported by the kernel
ncclResult_t ncclIbDmaBufSupport(int dev) {
  static std::once_flag onces[MAX_IB_DEVS];  // 每个设备一个 once_flag
  // init the device only once
  ibDmaSupportInitDev = dev;  // 记下"这次要探测哪个设备"
  /*
  探测逻辑，保证每个物理设备一生只探测一次，结果缓存在 ibDev->dmaBufSupported
  std::call_once 的回调不接受参数，所以"这次探测哪个 dev"没法直接传进去，只能通过 ibDmaSupportInitDev 这个变量中转
  */
  std::call_once(onces[dev], ibDmaBufSupportInitOnce);
  ncclIbMergedDev* mergedDev = ncclIbMergedDevs + ibDmaSupportInitDev;
  ncclIbDev* ibDev = ncclIbDevs + mergedDev->vProps.devs[0];
  int dmaBufSupported = ibDev->dmaBufSupported;
  if (dmaBufSupported == 1) return ncclSuccess;
  return ncclSystemError;
}
