/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_COMMON_H_
#define NET_IB_COMMON_H_

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "profiler/net_ib.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>
#define ENABLE_TIMER 0
#include "timer.h"

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"
#include "wqe_lat_mon.h"

#define MAXSUFFIXSIZE 16
#define MAXNAMESIZE (64 + MAXSUFFIXSIZE)
extern char ncclIbIfName[MAX_IF_NAME_SIZE + 1];
extern union ncclSocketAddress ncclIbIfAddr;

enum ncclIbRequestMatchingScheme {
  BY_INDEX = 0,
  BY_ID = 1,
};

// 缓存里的一个条目（一块已注册区间的描述）
struct ncclIbMr {
  // 这块已注册区间的起始地址，向下页对齐（data & -pageSize）。用页对齐地址做 key，因为 ibv_reg_mr 本就按页粒度 pin
  uintptr_t addr;
  // 区间跨多少页（向上取整覆盖 data+size）。和 addr 一起界定区间的 [addr, addr+pages*pageSize) 范围
  size_t pages;
  // 引用计数，当前有多少个"逻辑注册"复用着这一条 MR。命中一次 +1，反注册一次 -1，归零才真正 ibv_dereg_mr
  int refs;
  // 真正的 verbs MR 句柄（内含 lkey/rkey）。缓存命中时直接把它当 *mhandle 返回给调用方，省掉整次注册
  ibv_mr* mr;
};

// 装条目的容器（挂在每块物理设备上）
struct ncclIbMrCache {
  // 动态增长的条目数组，按 addr 升序排列（注册时有序插入）。这是缓存主体
  struct ncclIbMr* slots;
  /*
  数组已分配容量。满了就几何扩容：capacity < 32 ? 32 : 2*capacity（ncclRealloc）
  已用条目数（≤ capacity）。遍历、判空的边界
  */
  int capacity, population;
};

extern int ncclNMergedIbDevs;
/*
MAXNAMESIZE * NCCL_IB_MAX_DEVS_PER_NIC：给每个子设备名留 MAXNAMESIZE 的空间 × 最多子设备数。
+ NCCL_IB_MAX_DEVS_PER_NIC：再为每个子设备之间的 + 分隔符各留一个字符。
*/
#define NCCL_IB_MAX_DEVS_PER_NIC NCCL_NET_MAX_DEVS_PER_NIC
#define MAX_MERGED_DEV_NAME (MAXNAMESIZE * NCCL_IB_MAX_DEVS_PER_NIC) + NCCL_IB_MAX_DEVS_PER_NIC
// IB 插件里表达"网卡融合（NIC Fusion / 多轨聚合）"的结构，把多块物理网卡在逻辑上捏成一块"虚拟大网卡"，让一次连接能同时用上多块卡的带宽
struct alignas(64) ncclIbMergedDev {  // 缓存行对齐
  /*
  这个虚拟设备聚合了哪些物理设备，核心就两个字段成员：
    ndevs：聚合了几块物理卡
    devs[8]：各物理卡在 ncclIbDevs 里的索引数组
  */
  ncclNetVDeviceProps_t vProps;
  // 聚合带宽
  int speed;
  /*
  轨道 ID（rail-optimized 拓扑），标识这块网卡属于轨道优化网络的哪一条"轨道"。
  rail-optimized 是数据中心常见布线，把所有节点的"第 k 张网卡"接到同一个交换机平面，
  形成一条 rail。同 rail 内通信跳数少、更快。
  融合的子设备若分属不同 rail，合并设备的 railId 就置为 UNDEF。
  因为它跨了多条轨道，无法再用单一 rail 描述。这个字段影响跨 NIC 策略（crossNic：0 表示不许跨且要求进出 NIC 同 railId）。
  */
  int16_t railId;
  /*
  平面 ID（多平面网络），标识网卡属于多平面网络的哪个平面（plane），会写进拓扑 XML 的 plane 属性。
    若只聚合一块卡，直接沿用物理设备的 planeId。
    若聚合多块卡，则打开"虚拟位" NCCL_IB_PLANE_VIRT_BIT，再用位或把每个子设备的 planeIdx 编进去，
  每一位代表"这个合并设备覆盖了该平面"。
  */
  int16_t planeId;
  // 合并设备的名字，把子设备名拼接起来的字符串，比如两块卡融合会叫 mlx5_0+mlx5_1。构建时"每加一个子设备就把 名字 + '+' + 新名字 追加进去"。
  char devName[MAX_MERGED_DEV_NAME]; // Up to NCCL_IB_MAX_DEVS_PER_NIC * name size, and a character for each '+'
};

struct ncclIbStats {
  int fatalErrorCount;
};

enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MAX = 2,
};

struct ncclIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

extern int ncclNIbDevs;

enum ncclIbRailPolicy {
  NCCL_IB_RAIL_POLICY_NONE,
  NCCL_IB_RAIL_POLICY_CX9_FLIP, // Enable CX9 rail flipped layout
  NCCL_IB_RAIL_POLICY_CX9_ALT, // Enable CX9 rail alternate layout
  NCCL_IB_RAIL_POLICY_CX9_BLOCK, // Enable CX9 rail blocked layout
};

// ncclIbDev 是 IB 插件对一块物理网卡（的一个 port）的完整建模，它是真正持有 verbs 运行时对象、能力参数和缓存的地方。
struct alignas(64) ncclIbDev {
  // 每设备一把锁，保护下面的 pdRefs/pd 和 mrCache 这些会被多线程并发改的状态。
  std::mutex mutex;
  // 在 ibverbs 设备列表里的索引（wrap_ibv_get_device_list 返回的第 d 个），构建时 ncclIbDevs[...].device = d
  int device;
  /*
  取自 devAttr.sys_image_guid，芯片级唯一标识。一块芯片有多个 PCI function（物理/虚拟）时，
  靠它区分"是不是同一颗芯片"。会上报为 props->guid 写进拓扑 XML
  */
  uint64_t guid;
  uint32_t vendorId;
  uint32_t vendorPartId;
  char fwVer[64];
  // 这块设备上的 IB port 号
  uint8_t portNum;
  /*
  链路层类型，取自 portAttr.link_layer，IBV_LINK_LAYER_INFINIBAND（IB）或 IBV_LINK_LAYER_ETHERNET（RoCE）。
  宏 NCCL_IB_LLSTR 把它渲染成日志里的 "IB"/"RoCE"。
  */
  uint8_t link;
  // 端口速率，上报为 props->speed，拓扑层换算成网卡带宽用于建图调优
  int speed;
  uint64_t currSpeed;
  // wrap_ibv_open_device 打开的设备上下文，这块卡上所有 verbs 操作的句柄根。
  ibv_context* context;
  /*
  延迟分配、引用计数共享的 Protection Domain。这块卡上第一条连接才真正 alloc_pd，
  之后同设备的所有连接共用同一个 PD（pdRefs 计数）。PD 是注册 MR、建 QP 的容器，共享它能省资源。
  */
  int pdRefs;
  ibv_pd* pd;
  // 设备名（如 mlx5_0）。MAXNAMESIZE = 64 + MAXSUFFIXSIZE(16) = 80。data-direct 设备会加 _dma 后缀
  char devName[MAXNAMESIZE];
  /*
  网卡的 PCI 拓扑路径。pciPath 是对外用的指针（可能指向 fullPciPath，data-direct 设备则单独 ncclCalloc 一份）。
  拓扑层靠它判断"这块 NIC 挂在哪个 PCIe switch / NUMA 下"，是 GPU-NIC 亲和性计算的关键。
  */
  char fullPciPath[PATH_MAX];
  char* pciPath;
  /*
  物理端口偏移，getProperties 里 props->port = ibDev->portNum + ibDev->realPort，两者相加才是对外的最终 port。
  现代网卡的 verbs 端口号 portNum 几乎恒为 1，哪怕一块物理卡有多个端口/多个 VF，每个端口都被暴露成独立的 PCI function / ibv 设备，各自的 portNum 都是 1。
  计算逻辑：遍历所有索引比我小的设备，凡是和我属于同一块物理卡的（PCI 路径匹配），计数 +1。得到的就是"我在同卡兄弟里的 0 基序号"。
  */
  int realPort;
  // 取自 devAttr.max_qp，这块卡能开的最大 QP 数。上报为 props->maxComms，进而被拓扑当作 maxChannels（每 NIC 可服务的连接上限）
  int maxQp;
  // 网卡延迟（props->latency，IB 这里默认 0 未设）
  float latency;
  /*
  每设备的内存注册缓存，把昂贵的 ibv_reg_mr/dereg_mr 变成查表。slots 按地址升序，命中条件是"新区间 ⊆ 已有区间"就 refs++ 复用，
  否则新建条目插入（不合并）。构建时初始化为 capacity=0, population=0。它正是 props->regIsGlobal=1 的底层支撑，
  因为有这个设备级 fast cache，注册句柄（lkey/rkey）能跨 QP、跨 comm 复用，所以 IB 插件的 regMr 是"全局"的（socket 插件没这层缓存故 regIsGlobal=0）。
  这也是 user buffer registration 零拷贝优化的前提。
  */
  struct ncclIbMrCache mrCache;
  /*
  ADAPTIVE_ROUTING（自适应路由）开关。默认"IB 链路置 1、RoCE 置 0"，可被 NCCL_IB_ADAPTIVE_ROUTING 覆盖。
  AR 让交换机为数据包动态选路、缓解拥塞。
  */
  int ar; // ADAPTIVE_ROUTING
  /*
  乱序接收队列大小，注释明确 "valid only when ar=1"。来自 mlx5dv 的 ooo_recv_wrs_caps.max_rc。
  AR 会导致乱序到达，需要 OOO RQ 承接。connect.cc 里若强开 OOO RQ 但 ar==0 会 WARN（AR 是 OOO RQ 的前提）
  */
  uint32_t oooRqSize;  // valid only when ar=1
  // 缓存整份 port 属性（link_layer、各种 speed、gid、mtu 等），避免反复查询
  struct ibv_port_attr portAttr;
  // 目前只有 fatalErrorCount，记录这块设备的致命错误计数，供 resiliency（故障恢复）逻辑判断。
  struct ncclIbStats stats;
  // DMA-BUF 探测的结果缓存（1 支持 / -1 不支持），存在物理设备上供反复查询
  int dmaBufSupported;
  /*
  通过 NCCL_IB_HCA 指定的，例如："mlx5_0:1:0:0,mlx5_1:1:1:0,mlx5_2:1:0:1,mlx5_3:1:1:1"
  */
  // 来自用户配置 userIfs[userIfId].rail，否则 -1（NCCL_NET_ID_UNDEF）。轨道优化网络里标识这块卡属于哪条 rail
  int16_t railId;
  // 来自用户配置 userIfs[userIfId].plane，否则 -1。多平面网络里的平面 ID。
  int16_t planeId;
  /*
  由 ncclIbGetPlaneIndex 把原始 planeId 归一化成的紧凑小整数索引（把稀疏的 planeId 映射到 0,1,2...）。
  它正是合并设备里 mDev->planeId |= (0x1 << dev->planeIdx) 那个位或编码的位移量，所以必须先归一化成小整数，才能当 bit 位用。
  */
  int16_t planeIdx;
  /*
  这块卡是否支持 mlx5 Direct Verbs（wrap_mlx5dv_is_supported → IB_PROVIDER_MLX5）。
  只有 MLX5 provider 才能用 OOO RQ、data-direct 这些厂商扩展特性。
  */
  enum ncclIbProvider ibProvider;
  // 按 provider 存放各自的专有能力（当前只有 mlx5 分支，未来可扩展别的厂商）。这是"用 union 承载 provider 特定能力"的可扩展设计。
  union {
    struct {
      /*
      标记这是不是一个 data-direct 设备。data-direct 是 mlx5 的一种模式，物理 NIC 会被暴露两次，一次经 C2C+PCIe 链路，
      一次经 data-direct 链路，后者名字带 _dma、dataDirect=1。它影响两处：
        注册走厂商专用路径：wrap_mlx5dv_reg_dmabuf_mr(..., MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT)。
        getProperties 里 if (ibDev->capsProvider.mlx5.dataDirect) props->forceFlush = 1（收后需强制 flush）。
      forceFlush 表示"接收后需要显式 flush（iflush）才能保证写进 CUDA 内存的数据对 GPU 可见"。
      data-direct 的直通写入路径与常规路径的可见性/排序语义不同，需要一次显式 flush 来确保 GPU 侧读到的是最新数据，所以这类设备强制打开它。
      
      data-direct 是 NVIDIA/Mellanox 网卡（ConnectX-7 / BlueField-3 及更新）上的一条"网卡直通 GPU 显存"的专用 PCIe 数据路径，
      让网卡的 DMA 绕过 CPU root complex 和 C2C 跳，直接读写 GPU 显存，降低延迟、节省 C2C 带宽。
      */
      int dataDirect;
    } mlx5;
  } capsProvider;
  struct ncclIbGidInfo gidInfo;
};

/*
  物理设备 ncclIbDev：一块真实的 IB/RoCE 网卡（或它的一个 port），最多 32 个，ncclNIbDevs 计数
  合并设备 ncclIbMergedDev：一个逻辑虚拟网卡，内部引用一个或多个物理设备。上层（NCCL 核心）看到的"网卡"其实是这一层的对象，
getProperties/connect 都以合并设备的索引 dev 为准。
为什么要合并？现代 GPU 服务器常给一块 GPU 配多张网卡（多轨 rail-optimized 网络）。把通往同一目的地的多张卡融合成一个逻辑设备，
就能在一条通信连接里并行用满多张卡的带宽（NIC Fusion），而不是每张卡各建各的。MAX_IB_VDEVS = 256 给足了组合空间。
*/
#define MAX_IB_DEVS 32
#define MAX_IB_VDEVS MAX_IB_DEVS * 8  // 256
extern struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];  // 合并/虚拟设备
extern struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];  // 物理设备
extern int ncclIbRelaxedOrderingEnabled;
extern uint64_t ncclIbSpeedChangeCounter;
extern int64_t ncclParamIbEventBasedLb();
extern int64_t ncclParamIbEventBasedLbRemote();

#define NCCL_IB_LLSTR(ll) \
  (((ll) == IBV_LINK_LAYER_INFINIBAND) ? "IB" : (((ll) == IBV_LINK_LAYER_ETHERNET) ? "RoCE" : "UNSPECIFIED"))

struct alignas(32) ncclIbRemoteSpeedBuf {
  volatile uint64_t counter;
  uint16_t speedGbps[NCCL_IB_MAX_DEVS_PER_NIC];
};
static_assert(sizeof(ncclIbRemoteSpeedBuf) == 32);

// Per-Dev connection metadata
struct ncclIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;

  // For RoCE and IB Rounter
  union ibv_gid gid;

  // The key used for remote access to the addr exchanged by the peers
  // in ncclIbConnectionMetadata::addr
  // This member is populated differently on the sender and on the receiver
  // side.
  // The sender side populates this member with the RKey obtained after it
  // registered the CTS FIFO (on the specific device).
  // The receiver side populates this member with the RKey obtained after it
  // registered the completion records structure (on the specific device).
  uint32_t rkey;

  // remote dev info
  union ibv_gid remoteGid;

  uint64_t currSpeed;
  uint32_t remSpeedBufRkey;
};

#define MAX_QPS_PER_REQ 8
struct ncclProfilerInfo {
  void* qpEventHandles[MAX_QPS_PER_REQ];
  int qpIndex[MAX_QPS_PER_REQ];
  int nEventHandles;
  ncclProfilerNetIbDescr_v1_t data;
  void* pHandle;
};

#define NCCL_NET_IB_MAX_RECVS 8

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
#define NCCL_NET_IB_REQ_GIN_IPUT 4
#define NCCL_NET_IB_REQ_GIN_IGET 5
extern const char* ncclIbReqTypeStr[];

// Maximal number of QPs a communicator can have for data transfers
#define NCCL_IB_MAX_QPS 128

// Tracks data transfers between sender and receiver. A multi-recv/send uses a
// single record.
struct ncclIbRequestCompletionRecord {
  // This array communicates data transfer sizes from the sender to the
  // receiver. The sender writes the size of each completed data transfer to
  // this array. The receiver reads these sizes before reporting completion of
  // the corresponding receive request to the user.
  int sizes[NCCL_NET_IB_MAX_RECVS];
  // The receiver fills this array to signal the completion of a data transfer.
  // The sender can then read this array to see the receiver's status. If the
  // sender detects an error or device failure, it reads this array to
  // determine if the receiver considered the transfer complete. This prevents
  // the sender from retransmitting data if the failure was only visible on the
  // sender's side. Based on the array's contents, the sender decides if, how,
  // and on which QPs/devices to replay the transfer.
  bool completions[NCCL_IB_MAX_QPS];
};

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  struct ncclSocket* sock;
  // Array of counters. Each element in the array is populated with the expected
  // number of completion events that the request is expecting to be generated
  // on the device corresponding to the index of the element. After the request
  // is initialized and posted, for every completion event generated by a
  // device, the corresponding counter is decremented. When the counter reaches
  // zero it means that the request was fully completed on that device.
  int events[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to the per-device base structures to make it easier to
  // poll the device's CQ when the request is tested for progress.
  // The pointers are initialized only for the devices that the request expects
  // to receive completions from.
  struct ncclIbNetCommDevBase* devBases[NCCL_IB_MAX_DEVS_PER_NIC];
#ifdef NCCL_ENABLE_NET_PROFILING
  struct ncclProfilerInfo pInfo[NCCL_NET_IB_MAX_RECVS];
#endif
  uint64_t id;
  int nreqs;
  union {
    struct {
      int size;
      void* data;
      uint32_t lkeys[NCCL_IB_MAX_DEVS_PER_NIC];
      // Tracks whether data was transmitted on a QP for this request.
      bool sentData[NCCL_IB_MAX_QPS];
      // Per-device LB weights used for chunk computation.
      uint8_t weights[NCCL_IB_MAX_DEVS_PER_NIC];
    } send;
    struct {
      struct ncclIbRequestCompletionRecord* cmplsRecords;
      // Aggregates the size of a send request when sender does not write to the
      // completion records array.
      int aggSize;
    } recv;
    struct {
      int rank;
    } iput;
    struct {
      int rank;
    } iget;
  };
  void* rmaProxyCtx;
};

struct ncclIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd* pd;
  struct ibv_cq* cq;
  uint64_t pad[2];
  struct ncclIbGidInfo gidInfo;
  // Resolved once at device init and reused by every QP (like the GID index above).
  int pkeyIndex;
};

// Snapshot the device-wide GID info into a comm's per-device base under a mutex.
static inline void ncclIbGidInfoSnapshot(struct ncclIbNetCommDevBase* base, struct ncclIbDev* ibDev) {
  std::lock_guard<std::mutex> lock(ibDev->mutex);
  base->gidInfo = ibDev->gidInfo;
}

struct ncclIbSendFifo {
  uint64_t addr;
  uint64_t size;
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint64_t idx;
};

struct ncclIbQpInitAttr {
  ibv_qp_state state;
  int pkeyIndex;
  uint8_t portNum;
  int qpAccessFlags;
};

struct ncclIbQpRtrAttr {
  enum ibv_mtu mtu;
  uint8_t linkLayer;
  uint8_t tc;
  int sl;

  uint32_t remoteQpNum;
  uint32_t remoteLid;
  union ibv_gid remoteGid;

  uint8_t localIbPort;
  uint8_t localPortFlags;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

struct ncclIbQpRtsAttr {
  int timeout;
  int retryCnt;
};

struct ncclIbQp {
  struct ibv_qp* qp;
  // The index of the device on which this QP was created on.
  int devIndex;

  // The ECE (enhanced connection establishment) used on this QP.
  // Note: This is the reduced ECE exchanged between the sender and receiver.
  struct ibv_ece ece;
  int eceSupported;

  // Stores the attributes used to configure the QP to allow QP restore after
  // failure.
  struct ncclIbQpInitAttr initAttr;
  struct ncclIbQpRtrAttr rtrAttr;
  struct ncclIbQpRtsAttr rtsAttr;

  // The index of the device on the remote side to which this QP is connected
  // to.
  int remDevIdx;
  struct ncclIbWqeLatMon latMon;
};

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256,
              "request id are encoded in wr_id and we need up to 8 requests ids per completion");

// Structure to describe the completion records on the sender side.
struct ncclIbRemCompletionsRecords {
  // A "shadow" structure of the receiver's completion records in which the
  // sender tracks the completion records locally on its side. Sender uses this
  // memory to place the records it writes/reads to/from the receiver's
  // completion records.
  int elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  // Address in memory of the completion records structure on the receiver side.
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the sender chooses the
  // RKey (depending on the device being used) when it accesses the receiver's
  // completion records structure.
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
};

// A per-dev struct for netIbSendComm
struct alignas(8) ncclIbSendCommDev {
  struct ncclIbNetCommDevBase base;
  struct ibv_mr* ctsFifoMr;
  struct ibv_mr* putSignalScratchpadMr;
  struct ibv_mr* cmplsRecordsMr;
  struct ibv_sge sge;
};

// Wrapper to track an MR per-device, if needed
struct ncclIbMrHandle {
  ibv_mr* mrs[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Forward declaration
struct ncclIbResiliency;

struct alignas(32) ncclIbNetCommBase {
  ncclNetVDeviceProps_t vProps;
  bool isSend;
  struct ncclIbRequest reqs[NET_IB_MAX_REQUESTS];
  struct ncclIbQp qps[NCCL_IB_MAX_QPS];
  // Array of pointers to the "actual" QPs that are used for data transfers.
  // The pointers point to QPs in the ncclIbNetCommBase::qps[] array.
  struct ncclIbQp* activeQps[NCCL_IB_MAX_QPS];
  uint64_t fifoHead;
  int nqps;
  int splitDataOnQps;
  struct ncclSocket sock;
  int ready;
  // Track necessary remDevInfo here
  int nRemDevs;
  bool remOooRq;
  bool localOooRq;
  int recvMatchingScheme;
  int nDataQps;
  struct ncclIbDevInfo remDevs[NCCL_IB_MAX_DEVS_PER_NIC];
  // statistics about the comm
  struct ncclIbStats stats;
  struct ncclIbResiliency* resiliency;
  uint64_t speedChangeCounter;
  uint64_t totalSpeed;
  uint8_t weights[NCCL_IB_MAX_DEVS_PER_NIC];
  uint64_t devSpeeds[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Compute per-device LB weights (1-100); weight is never 0 since it is not considered a speed update but rather a port down.
static inline void ncclIbComputeLbWeights(struct ncclIbNetCommBase* base) {
  int ndevs = base->vProps.ndevs;
  // totalSpeed can not be 0: devices with inactive ports (speed 0) are
  // skipped at init, and speed-to-zero events are skipped in
  // ncclIbUpdateDeviceSpeed (port-failover handles those).
  uint8_t totalWeight = 0;
  for (int d = 0; d < ndevs; d++) {
    base->weights[d] = ncclParamIbEventBasedLb() ? (base->devSpeeds[d] * 100 / base->totalSpeed) : (100 / ndevs);
    totalWeight += base->weights[d];
  }
  if (totalWeight < 100) {
    base->weights[ndevs - 1] += (100 - totalWeight);
  }
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex);

static inline void ncclIbComputeDevSpeeds(struct ncclIbNetCommBase* base) {
  uint64_t totalSpeed = 0;
  for (int d = 0; d < base->vProps.ndevs; d++) {
    int ibDevN = ncclIbGetNetCommDevBase(base, d)->ibDevN;
    int remDevIdx = base->qps[d].remDevIdx;
    uint64_t localSpeed = COMPILER_ATOMIC_LOAD(&ncclIbDevs[ibDevN].currSpeed, std::memory_order_relaxed);
    uint64_t remoteSpeed = (base->remDevs[remDevIdx].currSpeed > 0) ? base->remDevs[remDevIdx].currSpeed : localSpeed;
    base->devSpeeds[d] = std::min(localSpeed, remoteSpeed);
    totalSpeed += base->devSpeeds[d];
  }
  base->totalSpeed = totalSpeed;
}

// qpIndex is the index relative to a device.
// For example, if a device has 2 QPs, qpIndex can be 0 or 1.
static inline ncclResult_t ncclIbCommBaseGetQpByIndex(struct ncclIbNetCommBase* commBase, int devIndex, int qpIndex,
                                                      ncclIbQp** qp) {
  if (devIndex < 0 || devIndex >= commBase->vProps.ndevs) {
    WARN("NET/IB: Invalid device index %d, expected [0, %d)", devIndex, commBase->vProps.ndevs);
    return ncclInternalError;
  }
  *qp = commBase->activeQps[commBase->vProps.ndevs * qpIndex + devIndex];
  return ncclSuccess;
}

// Each request is transferred over all devices, and depending on the
// "splitDataOnQps" configuration parameter, a request may be transferred over
// a single QP per device or on all QPs of each device.
static inline ncclResult_t ncclIbCommBaseGetNqpsPerRequest(struct ncclIbNetCommBase* baseComm, int* nQps) {
  if (nQps == NULL) {
    WARN("NET/IB: nQps output parameter is NULL");
    return ncclInternalError;
  }
  if (baseComm->nDataQps == -1) {
    WARN("NET/IB: nDataQps is not initialized");
    return ncclInternalError;
  }
  if (baseComm->nqps == -1) {
    WARN("NET/IB: nqps is not initialized");
    return ncclInternalError;
  }
  *nQps = (baseComm->splitDataOnQps == 1) ? baseComm->nqps : baseComm->nDataQps;
  return ncclSuccess;
}

// The function selects the QP to be used for the request. The QP selected
// based on the request ID and also based on the provided QP index. A request
// can be posted on multiple QPs. For example, if a request is posted on 4
// QPs, this function should be called 4 times, each time with a different
// qpIndex, ranging from 0 to 3.
// The function outputs the selected QP in the outQp argument and populates the
// outQpIndex argument with the index of the selected QP. Note that the
// outQpIndex is the index of the QP in the base::qps[] array.
static inline ncclResult_t ncclIbCommBaseGetQpForRequest(struct ncclIbNetCommBase* baseComm, const uint64_t id,
                                                         const uint8_t qpIndex, ncclIbQp** outQp, int* outQpIndex) {
  int nQps = 0;
  NCCLCHECK(ncclIbCommBaseGetNqpsPerRequest(baseComm, &nQps));
  *outQpIndex = (id * nQps + qpIndex) % baseComm->nqps;
  *outQp = baseComm->activeQps[*outQpIndex];
  if (*outQp == NULL) {
    WARN("NET/IB: QP is NULL for request id %lu, QP index %d", id, *outQpIndex);
    return ncclInternalError;
  }
  return ncclSuccess;
}

// Get a QP object from a QP number. If not NULL, qpIndex will also return the
// index of the QP in the ncclIbNetCommBase::qps[] array.
static inline ncclResult_t ncclIbCommBaseGetQpByQpNum(struct ncclIbNetCommBase* commBase, int devIndex, uint32_t qpNum,
                                                      ncclIbQp** qp, int* qpIndex) {
  if (devIndex < 0 || devIndex >= commBase->vProps.ndevs) {
    WARN("NET/IB: Invalid device index %d, expected [0, %d)", devIndex, commBase->vProps.ndevs);
    return ncclInternalError;
  }
  if (qp == NULL) {
    WARN("NET/IB: QP output pointer is NULL");
    return ncclInternalError;
  }
  TRACE(NCCL_NET, "NET/IB: %s: Looking for QP num %u on devIndex %d among %d QPs", __func__, qpNum, devIndex,
        commBase->nqps / commBase->vProps.ndevs);
  for (int qpIndexInDev = 0; qpIndexInDev < (commBase->nqps / commBase->vProps.ndevs); qpIndexInDev++) {
    *qp = &(commBase->qps[commBase->vProps.ndevs * qpIndexInDev + devIndex]);
    if ((*qp)->qp->qp_num == qpNum) {
      if (qpIndex != NULL) {
        *qpIndex = *qp - commBase->qps;
      }
      return ncclSuccess;
    }
  }
  *qp = NULL;
  return ncclInternalError;
}

static inline ncclResult_t ncclIbPostRecvWorkRequest(struct ibv_qp* qp, struct ibv_recv_wr* wr) {
  struct ibv_recv_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_recv(qp, wr, &bad_wr));
  return ncclSuccess;
}

struct ncclIbSendComm {
  struct ncclIbNetCommBase base;
  // Start with CTS FIFO and ibv structs as they have alignment restrictions

  // CTS FIFO from which the sender reads the Clear-to-Send (CTS) messages that
  // are written by the receiver (The receiver side writes into it upon
  // issuing a (multi-)receive request). Each row in the 2D array corresponds
  // to a single CTS message but can describe multiple recv-requests issued
  // on the receiver side.
  struct ncclIbSendFifo ctsFifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
  struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];
  // Each dev correlates to a mergedIbDev
  struct ncclIbSendCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the send requests for faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array based on the "slot" they
  // are associated with.
  struct ncclIbRequest* sendReqs[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];

  // Counter per "slot" on how many send request were called for a multi-recv
  int sendReqsCnt[NET_IB_MAX_REQUESTS];
  struct ncclIbRemCompletionsRecords remCmplsRecords;
  int ar; // Use adaptive routing when all merged devices have it enabled
  uint64_t putSignalScratchpad;

  struct {
    uint64_t posted;
    uint64_t completed;
  } ginSeq;

  struct ncclIbRemoteSpeedBuf remoteSpeedBuf;
  struct ibv_mr* remoteSpeedMr;
  uint64_t remoteSpeedCounter;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0,
              "ncclIbNetCommBase size must be 32-byte multiple to ensure ctsFifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, ctsFifo) % 32) == 0, "ncclIbSendComm ctsFifo must be 32-byte aligned");
static_assert((sizeof(struct ncclIbSendFifo) % 32) == 0, "ncclIbSendFifo element size must be 32-byte multiples");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

struct ncclIbGpuFlush {
  struct ibv_mr* hostMr;
  struct ibv_sge sge;
  struct ncclIbQp qp;
};

// This structure describes the FIFO which the receiver uses when it sends CTS
// messages to the sender.
struct ncclIbRemCtsFifo {
  // A "shadow" structure of the sender's CTS FIFO in which the receiver tracks
  // the CTS FIFO locally on its side. Receiver uses this memory to place the
  // CTS messages and populates the RDMA message "gather address" with the
  // memory of the CTS message that is sent.
  struct ncclIbSendFifo elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;
  // Array of RKeys (one RKey per device) from which the receiver chooses the
  // RKey (depending on the device being used) when it posts a CTS to the
  // sender
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
};

struct alignas(16) ncclIbRecvCommDev {
  struct ncclIbNetCommDevBase base;
  struct ncclIbGpuFlush gpuFlush;
  // MR that is obtained after registering the "shadow" CTS FIFO on the
  // receiver's side. The LKey of this MR allows RDMA operations on the receiver
  // side to gather CTS messages (formatted by the receiver) and write them to
  // the sender's CTS FIFO.
  struct ibv_mr* ctsFifoMr;
  // MR that is obtained after registering the completion records on the
  // receiver side. The RKey of this MR is provided to the sender side, to allow
  // the sender side to access receiver's completion records using RDMA
  // operations.
  struct ibv_mr* cmplsRecordsMr;
  // SGE to avoid allocation of SGE structures on the stack when receiver
  // posts RDMA operations. The SGE is populated by the address of the memory
  // in which the CTS message formatted on the receiver is placed.
  struct ibv_sge sge;
  struct ibv_mr* speedUpdateMr;
};

#define NCCL_IB_RECV_WR_ID_DUMMY UINT64_MAX
#define NCCL_IB_SPEED_UPDATE_WR_ID (UINT64_MAX - 1)

struct ncclIbRecvComm {
  struct ncclIbNetCommBase base;
  struct ncclIbRecvCommDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  // Array of pointers to store the recv requests to allow faster access. The
  // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
  // array. The requests are inserted to this array using a hash (modulo) on
  // their ID.
  struct ncclIbRequest* recvReqs[NET_IB_MAX_REQUESTS];
  // Structure to hold all the related structures regarding the CTS FIFO
  // structure.
  struct ncclIbRemCtsFifo remCtsFifo;
  // Structure to hold all the completion records of all the outstanding
  // receive requests on the receiver side.
  struct ncclIbRequestCompletionRecord cmplsRecords[NET_IB_MAX_REQUESTS];
  int gpuFlushHostMem;
  int flushEnabled;
  int flushQpSl;
  int flushQpTc;
  bool flushQpsCreated;
  bool prepostReceiveWorkRequests;
  // To avoid allocation and memset on the data-path a single structure is used
  // and only the wr_id is updated before posting a receive work request.
  struct ibv_recv_wr ibRecvWorkRequest;

  uint64_t remSpeedBufAddr;
  struct ncclIbRemoteSpeedBuf speedUpdateBuf;
  uint16_t lastSentSpeeds[NCCL_IB_MAX_DEVS_PER_NIC];
  uint64_t lastSentCounter;
  bool postedSpeedUpdate;
  struct {
    uint64_t posted;
    uint64_t completed;
  } ginSeq;
};
static_assert((offsetof(struct ncclIbRecvComm, remCtsFifo) % 32) == 0,
              "ncclIbRecvComm ctsFifo must be 32-byte aligned");

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend);
ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm);
ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm);

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage* stage;
};

static inline void ncclIbCheckSpeedChanges(struct ncclIbSendComm* sendComm, struct ncclIbNetCommBase* base) {
  bool speedChanged = false;
  // Local speed change detection
  if (base->speedChangeCounter != COMPILER_ATOMIC_LOAD(&ncclIbSpeedChangeCounter, std::memory_order_acquire)) {
    base->speedChangeCounter = COMPILER_ATOMIC_LOAD(&ncclIbSpeedChangeCounter, std::memory_order_acquire);
    speedChanged = true;
  }
  // Remote speed change detection
  if (sendComm->remoteSpeedCounter !=
      COMPILER_ATOMIC_LOAD(&sendComm->remoteSpeedBuf.counter, std::memory_order_acquire)) {
    sendComm->remoteSpeedCounter = COMPILER_ATOMIC_LOAD(&sendComm->remoteSpeedBuf.counter, std::memory_order_acquire);
    for (int i = 0; i < base->nRemDevs; i++) {
      base->remDevs[i].currSpeed = (uint64_t)sendComm->remoteSpeedBuf.speedGbps[i] * 1000;
    }
    speedChanged = true;
  }
  if (speedChanged) {
    ncclIbComputeDevSpeeds(base);
    ncclIbComputeLbWeights(base);
  }
}

static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  COMPILER_ATOMIC_STORE(&stat->fatalErrorCount, 0, std::memory_order_relaxed);
  return ncclSuccess;
}
static void ncclIbStatsFatalError(struct ncclIbStats* stat) {
  COMPILER_ATOMIC_FETCH_ADD(&stat->fatalErrorCount, 1, std::memory_order_relaxed);
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}
ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName);

extern ncclProfilerCallback_t ncclProfilerFunction;

extern std::thread ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args);

ncclResult_t ncclIbGdrSupport();
ncclResult_t ncclIbPeerMemSupport();
ncclResult_t ncclIbDmaBufSupport(int dev);

void ncclIbAddEvent(struct ncclIbRequest* req, int devIndex);
ncclResult_t ncclIbGetGidIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                               int* gidIndex);
ncclResult_t ncclIbGetPkeyIndex(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                int* pkeyIndex);
ncclResult_t ncclIbGidInfoQuery(struct ibv_context* context, uint8_t portNum, struct ibv_port_attr* portAttr,
                                struct ncclIbGidInfo* gidInfo);
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req);
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r);

ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mrFlags, void** mhandle);

int ncclIbGetTrafficClass(void* ctx);
void ncclIbSetTrafficClass(void* ctx, int trafficClass);

// CX9 rail policy helper. Returns NCCL_NET_ID_UNDEF when nDevs is not a positive multiple of eight.
int ncclIbCx9RailId(enum ncclIbRailPolicy policy, int devId, int nDevs);

// Net IB plugin entry functions.
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction,
                        ncclProfilerCallback_t profFunction);
ncclResult_t ncclIbDevices(int* ndev);
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbListen(void* ctx, int dev, void* opaqueHandle, void** listenComm);
ncclResult_t ncclIbConnectImpl(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                               ncclNetDeviceHandle_t** sendDevComm, int nQpsPerDev, int envTrafficClass);
ncclResult_t ncclIbConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm,
                           ncclNetDeviceHandle_t** sendDevComm);
ncclResult_t ncclIbAcceptImpl(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm, int nQpsPerDev);
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm);
ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle);
ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle);
ncclResult_t ncclIbIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle,
                         void** request);
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                         void** request);
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t ncclIbCreateFlushQp(struct ncclIbRecvComm* comm);
ncclResult_t ncclIbTest(void* request, int* done, int* sizes);
ncclResult_t ncclIbCloseSend(void* sendComm);
ncclResult_t ncclIbCloseRecv(void* recvComm);
ncclResult_t ncclIbCloseListen(void* listenComm);
ncclResult_t ncclIbMakeVDevice(int* d, ncclNetVDeviceProps_t* props);
ncclResult_t ncclIbFinalizeDevices(void);
ncclResult_t ncclIbFinalize(void* ctx);
ncclResult_t ncclIbSetNetAttr(void* ctx, ncclNetAttr_t* netAttr);

static inline void printIbWcStatusHint(int status) {
  switch (status) {
  case IBV_WC_LOC_PROT_ERR:
    INFO(NCCL_NET,
         "HINT: In many cases this error indicates that ACS is enabled, which breaks the GPU Direct RDMA protocol.");
    INFO(NCCL_NET, "HINT: To confirm, set NCCL_NET_GDR_LEVEL=0; if that resolves it, "
                   "disable ACS following your vendor documentation.");
    return;
  case IBV_WC_WR_FLUSH_ERR:
    INFO(NCCL_NET, "HINT: In many cases this error indicates that NICs on the same node cannot talk to each other.");
    INFO(NCCL_NET, "HINT: To confirm, use a lower level tool like ib_write_bw to communicate across NICs on the same "
                   "node.");
    return;
  case IBV_WC_RETRY_EXC_ERR:
    INFO(NCCL_NET, "HINT: In many cases this error indicates that NCCL_IB_TIMEOUT is set too short (the default value "
                   "is 20, which is ~30 seconds before timing out).");
    INFO(NCCL_NET, "HINT: To confirm, increase NCCL_IB_TIMEOUT (see "
                   "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-timeout).");
    return;
  default:
    break;
  }
}

// GID Format
// global:  |              64b  - subnet-prefix                |                 64b - EUI                          |
// raw   :  | 10b fixed | 22b 0 | 16b FLID | 16b subnet-prefix |                 64b - EUI                          |
static uint16_t ncclIbExtractLocalSubnetPrefix(uint64_t subnet_prefix) {
  return (be64toh(subnet_prefix) & 0xffff);
}

static int ncclIbExtractFlid(union ibv_gid* gid) {
  return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
}

#endif
