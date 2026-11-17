/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset,
                                        int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  // 页大小缓存
  static thread_local uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  // 页对齐
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);  // 操作 mr 缓存，加锁
  // 默认 mr 缓存按照 addr 排序，依次遍历
  for (int slot = 0; /*true*/; slot++) {
    // 就是没匹配到
    if (slot == cache->population || addr < cache->slots[slot].addr) {
      // didn't find in cache
      // 缓存容量不够了，扩张 mr 缓存数组
      if (cache->population == cache->capacity) {
        // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      // Deregister / register
      struct ibv_mr* mr;
      // 基础权限：本地写 + 远端写/读/原子，RDMA 双向操作全开。
      unsigned int flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
      /*
        relaxedOrdering：全局开启 ncclIbRelaxedOrderingEnabled（由 ncclIbRelaxedOrderingCapable 探测 + NCCL_IB_PCI_RELAXED_ORDERING 决定）
      且本次注册没有强制严格序（mrFlags 不含 NCCL_NET_MR_FLAG_FORCE_SO）时才成立。成立则加 IBV_ACCESS_RELAXED_ORDERING。
        Relaxed Ordering 是 PCIe 特性：允许事务乱序完成，避免队头阻塞、提升吞吐（GDR 场景下正确性不依赖严格序时可用）。
      RMA 的 signal buffer 需要严格序，故带 FORCE_SO 标志排除它。
      */
      bool relaxedOrdering = ncclIbRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
      if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;
      // 按 fd 与 dataDirect 四选一分流注册
      if (fd != -1) {
        /* DMA-BUF support */
        if (!ncclIbDevs[base->ibDevN].capsProvider.mlx5.dataDirect) {
          // 标准 DMA-BUF 注册
          NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages * pageSize, addr, fd, flags));
        } else {
          /*
          mlx5 厂商专用，走 data-direct 直通路径
          MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT 告诉 mlx5 驱动：这个 MR 用 data-direct 直通引擎。
          */
          NCCLCHECK(wrap_mlx5dv_reg_dmabuf_mr(&mr, base->pd, offset, pages * pageSize, addr, fd, flags,
                                              MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT));
        }
      } else {
        if (relaxedOrdering) {
          /*
          经典注册，但用 IBVERBS_1.8 扩展 API
          普通 ibv_reg_mr 这个老 API 无法表达 IBV_ACCESS_RELAXED_ORDERING，ibv_reg_mr_iova2 是 IBVERBS_1.8 引入的扩展版，
          既能单独指定 IOVA、又支持 relaxed ordering 标志。这里 iova 传的还是 addr（和注册基址同值），换 API 纯粹是为了拿到能接受该标志的接口。
          ncclIbRelaxedOrderingCapable 也正是靠 dummy 调 wrap_ibv_reg_mr_iova2(NULL,...) 来探测这个扩展 API 在不在。
          而 DMA-BUF 那两个 API 本身就直接吃 flags，relaxed ordering 标志加在 flags 里即可，不需要另换 API。

          同一块内存，CPU 看、设备（网卡）看、物理上是三套不同的编号：
            VA（虚拟地址），CPU / 进程使用它，对应 addr；
            PA（物理地址），内存条 / 显存颗粒本身使用它，代码里看不到，被翻译层藏着；
            IOVA（I/O 虚拟地址），网卡（NIC）做 DMA 时使用它，对应 iova 参数。
          CPU 访问链：CPU 发出 VA → MMU（CPU 的内存管理单元）把 VA 翻译成 PA → 读写内存。
          设备访问链：网卡发出 IOVA → IOMMU / 网卡自己的地址翻译表把 IOVA 翻译成 PA → 读写同一块内存。
          */
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages * pageSize, addr, flags));
        } else {
          NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages * pageSize, flags));
        }
      }
      TRACE(NCCL_INIT | NCCL_NET, "regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d", (unsigned long)addr,
            (long long)pages * pageSize, mr->rkey, mr->lkey, fd);
      /*
      若插入点不在末尾，先 memmove 把 [slot, population) 的条目整体右移一格腾位，再填入新条目，
      把新 MR 当句柄返回。TRACE 打出 rkey/lkey，是排查注册问题的日志锚点。
      */
      if (slot != cache->population) {
        memmove(cache->slots + slot + 1, cache->slots + slot, (cache->population - slot) * sizeof(struct ncclIbMr));
      }
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      return ncclSuccess;
    // 单向区间包含，复用
    } else if ((addr >= cache->slots[slot].addr) &&
               ((addr - cache->slots[slot].addr) / pageSize + pages) <= cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

/*
把"一次逻辑注册"展开成"融合设备里每个物理子设备各注册一次"，再把产出的多个 ibv_mr 打包进一个 wrapper 句柄返回。
*/
/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  assert(size > 0);
  /*
  comm 是 sendComm 或 recvComm，二者都以 struct ncclIbNetCommBase base 打头，所以能安全上转。
  这个 base 携带两个关键信息：vProps（融合设备的子设备列表）和 isSend（区分收发）。
  */
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  // 每个物理子设备一个 MR
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)malloc(sizeof(struct ncclIbMrHandle));
  /*
  融合设备由多个物理网卡组成（多轨聚合），每个物理子设备有自己独立的 PD，而 MR 是挂在 PD 上的，
  所以同一块 buffer 必须在每个子设备上各注册一次，得到 ndevs 个 ibv_mr。
    单卡（ndevs==1）：循环一轮，wrapper 里只有 mrs[0] 有效，wrapper 仅是薄壳。绝大多数部署是这种。
    融合卡（ndevs>1，NIC Fusion/多轨）：数据会拆分到多块物理卡并行传输，每块卡用自己的 QP/PD/MR 访问同一 buffer，
  故需 ndevs 个 MR。wrapper 把它们捆成"一次注册"的假象，对上层透明。
  */
  for (int i = 0; i < base->vProps.ndevs; i++) {
    /*
    取第 i 个子设备的 comm base。因为 ncclIbSendComm 和 ncclIbRecvComm 各自把 ncclIbNetCommDevBase base 
    包在不同结构（ncclIbSendCommDev/ncclIbRecvCommDev）里的 devs、虽然命名相同但偏移不同，这个 helper 靠 base->isSend 分派。
    */
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    // 在该物理设备上注册，把产出的 ibv_mr* 写进 mrs[i]。底层会用这个设备的 pd 和 mrCache
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(devComm, data, size, type, offset, fd, mrFlags, mhandleWrapper->mrs + i),
                  ret, fail);
  }
  /*
  成功时返回给调用方的不是单个 MR，而是整个 wrapper。核心层把它当不透明句柄存着，
  日后 ncclIbDeregMr 再转回 ncclIbMrHandle* 遍历 mrs[i] 逐个反注册、最后 free(wrapper)
  */
  *mhandle = (void*)mhandleWrapper;
exit:
  return ret;
fail:
  free(mhandleWrapper);
  goto exit;
}

ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return ncclIbRegMrDmaBufInternal(comm, data, size, type, offset, fd, 0ULL, mhandle);  // 传入真实 fd
}

ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclIbRegMrDmaBufInternal(comm, data, size, type, 0ULL, -1, 0, mhandle);  // fd = -1, offset = 0
}

ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  for (int i = 0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECK(wrap_ibv_dereg_mr(mhandle));
      }
      return ncclSuccess;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  return ncclInternalError;
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}
