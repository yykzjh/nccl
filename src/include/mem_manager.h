/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MEM_MANAGER_H_
#define NCCL_MEM_MANAGER_H_

#include "nccl.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <mutex>

#ifdef __cplusplus
extern "C" {
#endif

/*
补定义的原因是让编译期版本与运行期能力解耦，即便用 CUDA 12.0 编译，只要运行时驱动支持 MNNVL，
NCCL 照样能用 FABRIC。垫片补的是“编译期的声明缺口”，真正的能力判定交给运行时。这正是它比“直接要求用户升级 CUDA 工具链”更友好的地方。
*/
#if CUDART_VERSION < 12030  // 官方头文件里还没有 FABRIC 这些符号，NCCL 自己补定义
// MNNVL: FABRIC handle support lifted from CUDA 12.3
// 传给 cuDeviceGetAttribute 的属性号(值 128)，用于运行时查询“本设备是否支持 FABRIC 句柄”，mnnvl.cc 靠它探测 MNNVL 能力
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ((CUdevice_attribute)128)
/*
句柄类型枚举值：
  CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR，一个文件描述符(fd)，只能同节点内传递(经 Unix domain socket / SCM_RIGHTS)
  CU_MEM_HANDLE_TYPE_FABRIC，一段 64 字节不透明数据(CUmemFabricHandle)，可跨节点：直接当普通消息广播给别的机器上的 rank
这是 CUmemAllocationHandleType 里代表 FABRIC 的枚举值(0x8)，出现在三处驱动调用：
  cuMemCreate(通过 prop.requestedHandleTypes 申请 FABRIC 兼容内存)；
  cuMemExportToShareableHandle(导出)；
  cuMemImportFromShareableHandle(导入)。
*/
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
// 句柄 blob 大小，CU_IPC_HANDLE_SIZE 是个更老的符号，即使在旧 CUDA 头里也可能已存在，加 #ifndef 防止重复定义
#ifndef CU_IPC_HANDLE_SIZE
#define CU_IPC_HANDLE_SIZE 64
#endif
/*
句柄结构体（一个 64 字节的不透明 blob） + ABI 版本化 typedef，
命名严格照搬 CUDA 的 _st(struct tag)→ _v1(ABI 版本)→ 无后缀(公开别名) 三层模式，
目的是当哪天用真 CUDA ≥ 12.3 编译时，NCCL 自定义的这个类型与官方类型二进制布局完全一致，
代码里所有 CUmemFabricHandle 的使用点无需改动即可无缝切换。
*/
typedef struct CUmemFabricHandle_st {
  unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

struct ncclComm;

// Initial capacity for exported peers array
#define NCCL_MEM_EXPORT_PEERS_INIT 8

// Memory Type for NCCL allocations
typedef enum {
  ncclMemPersist = 0,  // Persistent memory - track stats only, never release/offload
  ncclMemScratch = 1,  // Free without saving
  ncclMemOffload = 2   // Copy to CPU before free, restore on resume
} ncclMemType_t;

// Memory entry state
typedef enum {
  ncclDynMemStateActive = 0,  // Memory is allocated and usable
  ncclDynMemStateReleased = 1   // Memory has been released
} ncclDynMemState_t;

/*
本地自有显存的描述符，当一块显存是本 rank 自己 cuMemCreate 出来的（entry->isImportedFromPeer == false），用这个描述符。
它回答两个问题：这块显存的可共享句柄是什么，以及它借给了谁。
*/
// Local owned memory descriptor
typedef struct ncclDynMemLocalDesc {
  /*
  CUDA VMM 的物理内存要给别的进程/rank 用，必须先导出成一个"可共享句柄"。有两种形态：
    POSIX FD：一个文件描述符，只能在同一节点内传递。
    FABRIC：一段 64 字节的不透明数据（CUmemFabricHandle），能跨节点通过普通消息直接广播。
  */
  // Shareable handle for P2P exports
  // TODO: Remove the 'fd' field - POSIX FD handles are converted on-demand via proxy
  // (ncclProxyClientGetFdBlocking), so we no longer export them upfront. Only FABRIC
  // handles need upfront export since they can be shared directly via messaging.
  union {
    int fd;            // For POSIX_FILE_DESCRIPTOR (unused)
    CUmemFabricHandle fabricHandle;  // For FABRIC
  } shareableHandle;
  // 表示这个 union 内容是否有效的哨兵，resume 里 FABRIC 重新导出成功后才置 true，导出失败或还没导出时是 false，读取前必须先查它。
  bool shareableHandleValid;
  /*
  动态数组（经典的 size / capacity / 指针 三件套），记录"我这块 buffer 借给了哪些 peer rank"。
    为什么必须记：suspend 会释放物理页，resume 时重新 cuMemCreate 会得到一个全新的物理 handle。
  之前把这块 buffer 映射进自己地址空间的那些 peer，手里的旧 handle 全部失效，必须拿新 handle 重新导入一次。
  exportedPeerRanks 就是 resume 时"该通知谁重新导入"的名单。
    增长方式：ncclDynMemMarkExportToPeer 里，容量为 0 时首次分配 NCCL_MEM_EXPORT_PEERS_INIT（8），此后每次翻倍（*2）。
  这是摊还 O(1) 的标准倍增策略，避免每加一个 peer 就 realloc。
  */
  // Peer tracking for P2P exports
  int numExportedPeers;
  int exportedPeersCapacity;
  int* exportedPeerRanks;
} ncclDynMemLocalDesc;

/*
从 peer 导入显存的描述符，当一块显存是别的 rank 的、本 rank 只是 P2P 映射进来用（isImportedFromPeer == true）。
用途集中在 resume 的重导入阶段：owner 恢复自己的 buffer 后，会广播一批新 handle 信息（下面的 ncclDynMemP2pHandleInfo）。
本 rank 拿着 (ownerRank, ownerPtr, size) 这个三元组去广播结果里匹配出属于自己这块导入 buffer 的那条新 handle，然后重新导入。
没有这三元组就无法在一堆广播信息里认出"哪条是我要的"。注意 ownerPtr 是 owner 那边的地址，不是本地地址，它纯粹当匹配用的键，本地不会去解引用它。
*/
// Imported from peer memory descriptor
typedef struct ncclDynMemImportDesc {
  int ownerRank;  // 这块显存的真正主人是哪个 rank
  int ownerDev;  // 主人的 CUDA 设备号
  void* ownerPtr;  // 在主人地址空间里的虚拟地址
} ncclDynMemImportDesc;

// 这是链表 ncclMemManager::entries 的节点，把上面两个描述符组合成一条完整记录。
// Individual tracked memory entry (only track scratch and offload allocations)
typedef struct ncclDynMemEntry {
  /*
  GPU 虚拟地址(VA，suspend/resume 全程不变)，ptr 是 cuMemAddressReserve 保留的虚拟地址，
  挂起时只 cuMemUnmap+cuMemRelease 而不 cuMemAddressFree，
  所以虚拟地址一直被这块占着，恢复后能映射回同一个 ptr——上层代码持有的指针因此永远有效。
  */
  void* ptr;
  size_t size;
  /*
  物理显存句柄(suspend 时释放，resume 时重建)，handle 指向物理页，挂起时释放并被显式置 0，
  恢复时 cuMemCreate 出新值回填。代码里多处 entry->handle = 0; 就是在标记"物理页现在没了"。
  */
  CUmemGenericAllocationHandle handle;
  // POSIX fd / FABRIC(决定如何跨进程/跨节点共享)
  CUmemAllocationHandleType handleType;
  /*
  memType 决定挂起策略：scratch 直接丢、offload 先拷 host 再丢、persist 根本不进这个链表（前一轮讲过）。
  所以链表里的 entry 只可能是 scratch 或 offload。
  */
  ncclMemType_t memType;
  /*
  state 是单块显存的挂起标志，和 manager 级的 released 区分开：released 是"整个 manager 挂起了没"，state 是"这一块的物理页当前在不在"。
  suspend 把 entry 置 Released，resume 重建后置回 Active。两遍遍历时靠它跳过已处理的项。
  */
  ncclDynMemState_t state;
  // esume 重建物理页时 prop.location.id 要用它，且共享 manager 场景下不同 entry 理论上可能属于不同设备。
  int cudaDev;

  // 仅 offload 用：卸载到 host 的数据副本
  void* cpuBackup;

  /*
    本地拥有（isImportedFromPeer == false）用 desc.local 里面的 exportedPeerRanks 动态数组记录"我这块 buffer 借给了哪些 peer"。
  为什么要记？因为 resume 时本地 buffer 会拿到全新的物理 handle，之前映射了它的 peer 必须重新导入，exportedPeerRanks 就是 resume 时通知谁的名单。
  它的增长用经典倍增：ncclDynMemMarkExportToPeer 里从 NCCL_MEM_EXPORT_PEERS_INIT(8) 开始 *2 扩容。
    从 peer 导入（isImportedFromPeer == true）用 desc.imported 记 ownerRank / ownerDev / ownerPtr，即"这块是谁的、在他那边的虚拟地址是多少"。
  resume 第 4 步用这三元组去匹配 owner 广播来的新 handle。
  */
  // Ownership type and type-specific data
  bool isImportedFromPeer;  // true if this is a peer-imported buffer
  union {
    ncclDynMemLocalDesc local;
    ncclDynMemImportDesc imported;
  } desc;

  // 链表指针(侵入式)
  struct ncclDynMemEntry* next;
} ncclDynMemEntry;

/*
跨 rank 交换的线上格式，没有 handle 之外的指针成员（ptr 只当整数键用、不解引用），
整体是 POD，能直接 memcpy 进发送缓冲、用 bootstrapSend/Recv 收发，
这是它区别于进程内 ncclDynMemEntry（含链表指针、动态数组）的根本原因，能上网线的必须是自包含的定长数据。
*/
// P2P Handle Exchange Structure
typedef struct ncclDynMemP2pHandleInfo {
  // 前三个字段是进程内的台账，发到网线上给别的 rank 的消息格式，用在 resume 的 handle 交换阶段。
  void* ptr;
  int ownerRank;
  int ownerDev;
  size_t size;
  int handleType;
  union {
    /*
    跨进程传输时 FD 没有意义，所以线上格式改传 handleData（cuMem handle 的原始位），接收方再凭它走 proxy 现场换成本地 fd。
    FABRIC 两边都能直接用，所以两个 union 里都保留 fabricHandle。
    */
    uint64_t handleData;  // POSIX FD 路径，装 cuMem handle，供对端转 FD
    CUmemFabricHandle fabricHandle;  // // FABRIC 路径，装可直接导入的 fabric 句柄
  };
} ncclDynMemP2pHandleInfo;

/*
ncclMemManager 是挂在每个 ncclComm 上的一张 GPU 显存台账（comm->memManager）。它的存在有两个目的：
  1. 统计。随时回答"这个 communicator 占了多少 GPU 显存、其中多少是可释放的"。
  2. 挂起/恢复（suspend/resume）。把这个 communicator 的动态显存临时吐还给系统、之后再原地恢复（对应 ncclCommSuspend / ncclCommResume）。
这是让训练任务能被抢占、迁移、或给别的进程临时腾显存的底层机制。
  关键设计前提是它只跟踪 CUDA VMM（Virtual Memory Management，即 cuMemCreate/cuMemMap 那套）分配的显存，
因为只有 VMM 能做到"释放物理页但保留虚拟地址"，这是原地恢复的硬性条件。
*/
// Memory manager attached to ncclComm
typedef struct ncclMemManager {
  // 单向链表头，串起所有被跟踪的分配。只挂 scratch 和 offload 两类，persist 类不进链表
  ncclDynMemEntry* entries;
  // 链表长度
  int numEntries;
  std::mutex lock;
  // 挂起状态位。0 = 活跃，1 = 已挂起。Suspend 成功后置 1，Resume 成功后置 0。
  int released;
  // 生命周期哨兵。Init 末尾用 release 语义置 1，Destroy 到 refCount 0 时置 0。
  // 所有对外函数入口都先用 acquire 语义读它，目的是避免加锁已经被析构的 mutex。
  int initialized;
  // 跨 communicator 共享计数
  int refCount;

  /*
  这些都是 GPU 显存字节数。Imported 后缀 = 这块显存的物理页属于别的 rank，本 rank 只是 P2P 映射进来的，
  跟自己 cuMemCreate 出来的要分开记账（否则多机多卡会重复计算总量）。它们全部用 COMPILER_ATOMIC_ADD_FETCH / SUB_FETCH 以 relaxed 序更新，
  因为统计量之间没有相互依赖，只要每个自身原子即可，不需要顺序保证。
  类型	              挂起时怎么处理	                                      是否进 entries 链表
  ncclMemPersist	   永不释放，只记统计数字	                                  ❌ 不进
  ncclMemScratch	   直接释放物理页，不保存内容（恢复后是脏数据，反正是临时缓冲）	  ✅ 进
  ncclMemOffload	   先 cudaMemcpy 拷到 host 备份，释放物理页，恢复时再拷回	    ✅ 进
  */
  size_t totalPersist;
  size_t totalPersistImported;
  size_t totalScratch;
  size_t totalScratchImported;
  size_t totalOffload;
  size_t totalOffloadImported;
  // offload 类挂起时，当前卸载到 host 的数据占用多少 CPU 内存
  size_t cpuBackupUsage;

  /*
  创建时从 comm->cudaDev 拷来的 CUDA 设备号。存一份的原因是 manager 的部分操作发生时，不一定还能方便地拿到 comm，
  就地存一个设备号最省事。每个 entry 里也各自再存一份 entry->cudaDev。
  */
  int commCudaDev;
} ncclMemManager;

struct ncclMemManagerTask {
  struct ncclMemManagerTask* next;
  struct ncclComm* comm;
};

// Initialize memory manager
ncclResult_t ncclMemManagerInit(struct ncclComm* comm);

// Destroy memory manager and free all resources
ncclResult_t ncclMemManagerDestroy(struct ncclComm* comm);

// Track a new allocation
ncclResult_t ncclMemTrack(struct ncclMemManager* manager, void* ptr, size_t size, CUmemGenericAllocationHandle handle,
                          CUmemAllocationHandleType handleType, ncclMemType_t memType);

// Track imported allocation from peer
ncclResult_t ncclMemTrackImportFromPeer(struct ncclMemManager* manager, void* ptr, size_t size,
                                        CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType,
                                        ncclMemType_t memType, int ownerRank, int ownerDev, void* ownerPtr);

// Untrack allocation
ncclResult_t ncclMemUntrack(struct ncclMemManager* manager, void* ptr, size_t size);

// Add peer info for buffers in the linked list entries (only for dynamic memory: scratch/offload)
ncclResult_t ncclDynMemMarkExportToPeer(struct ncclMemManager* manager, void* ptr, int peerRank);

ncclResult_t ncclCommMemSuspend(struct ncclComm* comm);
ncclResult_t ncclCommMemResume(struct ncclComm* comm);

#ifdef __cplusplus
}
#endif

#endif /* NCCL_MEM_MANAGER_H_ */
