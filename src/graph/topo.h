/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"
#include "xml.h"
#include "net.h"
#include "os.h"
#include "nvmlwrap.h"
#include <map>

#define LOC_BW 5000.0
#define MLOPART_LOC_BW 2618.0
#define SM60_NVLINK_BW 18.0
#define SM70_NVLINK_BW 20.0
#define SM80_NVLINK_BW 20.0
#define SM90_NVLINK_BW 20.6
#define SM86_NVLINK_BW 12.0
#define SM100_NVLINK_BW 40.1
#define RUBIN_NVLINK_BW 31.2
#define PCI_BW 12.0           // PCI Gen3 x16
#define AMD_ZEN12_BW 16.0
#define AMD_ZEN34_BW 24.0
#define AMD_ZEN5_BW 32.0
#define BDW_QPI_BW 6.0
#define SKL_QPI_BW 10.0
#define SRP_QPI_BW 22.0
#define ERP_QPI_BW 40.0
#define ZPI_BW 6.0
#define YONGFENG_ZPI_BW 9.0
#define P9_BW 32.0
#define ARM_BW 6.0
#define NET_BW 12.0           // 100Gbit

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth.
// Intel 的 Root Complex 把 GPU P2P 流量切成 64 字节的 TLP，头部开销让有效带宽降到 5/6，所以记账时要按 1.2 倍扣
#define INTEL_P2P_OVERHEAD(bw) (bw * 6 / 5)

// 节点类型数，即 GPU(0)/PCI(1)/NVS(2)/CPU(3，实为 NUMA 域)/NIC(4)/NET(5)/GIN(6)/RMA(7)/DEV(8)/CXB(9，C2C Cross-Bridge)。
#define NCCL_TOPO_NODE_TYPES 10
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
#define GIN 6
#define RMA 7
#define DEV 8
#define CXB 9 // C2C Cross-Bridge: shared C2C bus node for GPUs split with mlopart
extern const char* topoNodeTypeStr[];

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_NVL 1
// Skipping 2 for PATH_NVB
#define LINK_C2C 3
#define LINK_PCI 4
// Skipping 5 for PATH_PXB
// Skipping 6 for PATH_PXN
// Skipping 7 for PATH_P2C
// Skipping 8 for PATH_PHB
#define LINK_SYS 9
#define LINK_NET 10
extern const char* topoLinkTypeStr[];

extern int64_t ncclParamPxnC2c();

struct ncclTopoNode;
// 图中的边
struct ncclTopoLink {
  int type;  // 链路类型（LINK_NVL/LINK_PCI/LINK_NET…）
  float bw;  // 带宽
  struct ncclTopoNode* remNode;  // 指向对端顶点
};

// 每个顶点的边数上限
// Allows for up to 32 NICs per node on GB200-NVL72
#define NCCL_TOPO_MAX_LINKS 576
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES * NCCL_TOPO_NODE_TYPES)

struct ncclTopoLinkList {
  struct ncclTopoLink** list;
  int count;     // Number of links stored in list.
  int capacity;  // Number of entries allocated for list.
  float bw;
  int type;
};

#define NCCL_TOPO_UNDEF (-1)

#define NCCL_TOPO_ID_LOCAL_ID_MASK 0x00ffffffffffffff
#define NCCL_TOPO_ID_SYSTEM_ID(id) (id >> 56)
#define NCCL_TOPO_ID_LOCAL_ID(id) (id & NCCL_TOPO_ID_LOCAL_ID_MASK)
// Bits 63:56 are reserved for systemId; encode the 8-bit NUMA ID in bits 55:48.
#define NCCL_TOPO_LOCAL_NIC_ID(numaid, busid) ((((int64_t)(numaid) & 0xff) << 48) + (busid))
#define NCCL_TOPO_ID(systemid, localid) (((int64_t)systemid << 56) + (localid & NCCL_TOPO_ID_LOCAL_ID_MASK))  // 拼接 host 标识和 host 内局部标识
#define NCCL_TOPO_GPU_LOCAL_RANK_SHIFT 40
#define NCCL_TOPO_GPU_LOCAL_ID(busId, localRankOnDev) \
  ((((uint64_t)(localRankOnDev)) << 40) | ((busId) & ((((uint64_t)1) << 40) - 1)))
#define NCCL_TOPO_MLOPART_MASK (0x3) // lower 2 bits: bit[0]=enabled, bit[1]=partition index
#define NCCL_TOPO_MLOPART_DEV_MAX (2) // max DEV nodes per physical GPU (one per uGPU partition)
#define NCCL_TOPO_MLOPART(mloPart) ((((int64_t)(mloPart) << 1) | 0x1) & NCCL_TOPO_MLOPART_MASK)
#define NCCL_TOPO_MLOPART_BUSID(busId, mloPart) \
  ((mloPart) != NCCL_TOPO_UNDEF ? ((busId) | NCCL_TOPO_MLOPART(mloPart)) : (busId))

/*
拓扑图中的一个顶点，它承载四部分信息：
  1. 身份：type（是哪种节点）+ id（全局唯一标识）；
  2. 类型专属数据：一个 union，按 type 五选一（gpu/dev/net/cpu/pci）；
  3. 连接关系：links[]（邻接边表）+ paths[]（预计算路径缓存）；
  4. 搜索状态：used（DFS 位标记）。
*/
struct ncclTopoNode {
  // 节点类型，取值就是 10 个类型。它是这个 union 的 tag，决定该读 union 的哪个成员（典型的 tagged union）。
  int type;
  /*
  全局唯一标识，高 8 位编码 systemId（属于哪个 host）、低 56 位是 host 内 local id。
  复合编码 NCCL_TOPO_ID(systemId, localId)，把"第几台 host"和"host 内本地 id"（busId/numaId/nicId）拼成一个 int64_t。
  GPU 的 local id 用 NCCL_TOPO_GPU_LOCAL_ID(busId, localRankOnDev) 打包，NIC 用 NCCL_TOPO_LOCAL_NIC_ID(numaid, busid)。
  */
  int64_t id;
  // Type specific data
  union {
    /*
    分离 GPU 与 DEV 是为了 MLOPart（把一个物理 GPU 拆成多个逻辑分区参与通信）的建模，
    DEV 顶点代表物理 GPU 设备，GPU 顶点代表逻辑分区。一个 DEV 下可挂 nGpus 个 GPU 分区，
    每个 GPU 分区的 parent 指回 DEV，mloPart 记自己是第几个分区，NCCL_TOPO_MLOPART_DEV_MAX=2，
    即每物理 GPU 最多 2 个分区。非 MLOPart 时 mloPart=NCCL_TOPO_UNDEF。
    */
    struct {
      int dev; // NVML 物理 GPU 号
      int rank;  // 这个 GPU 对应的通信域 rank（不是本 comm 用的 GPU 则为 -1）
      int cudaCompCap;  // CUDA compute capability（如 90=Hopper）
      int gdrSupport;  // 是否支持 GPUDirect RDMA
      int mloPart; // MLOPart 分区索引，NCCL_TOPO_UNDEF 表示非分区
      struct ncclTopoNode* parent; // 指向父 DEV 顶点
    } gpu;  // 逻辑 GPU 参与者
    struct {
      uint64_t device;  // PCI device id（vendor+device+subsystem 组合，同 pci.device）
      int dev; // NVML dev number
      int cudaCompCap;
      int nGpus; // 挂在这个 DEV 下的 GPU 分区数
    } dev;  // 物理 GPU 设备
    struct {
      // 网络插件里的设备号
      int dev; // Plugin dev number
      uint64_t vendor; // PCI vendor ID
      uint64_t device; // PCI device ID
      uint64_t pciId;  // 网卡 PCI id
      uint64_t asic;  // 网卡 ASIC 型号
      int port;  // 端口号
      float bw;  // 带宽
      float latency;  // 延迟
      int gdrSupport;  // GDR 支持
      int collSupport;  // CollNet（网内集合，如 SHARP）支持
      int maxChannels;  // 这个网卡支持的最大通道数
      // 这个网卡最本地的 GPU 下标，-1 表示无。搜索网卡通道时优先从这个本地 GPU 起试，也是 PXN（PCI+NVLink）判定的依据
      int localGpu;
      int railId;  // 多轨标识
      int planeId;  // 多平面标识
    } net;  // 网络端点（三类网络节点共用这个成员，因为 ncclTopoFillNet 统一处理 net/gin/rma/collnet）
    struct {
      int arch;  // CPU 架构
      int vendor;  // CPU 厂商
      int model;  // CPU 型号
      /*
      这个 NUMA 域拥有的 CPU 核心集合。ncclAffinity 是个跨平台 CPU 集类型，
      Linux 底层是 cpu_set_t，取出靠近 GPU 的 CPU 的 affinity，与传入 mask 求交得 finalMask，
      用于把进程/proxy 线程绑到 GPU 所在 NUMA 的核上（NCCL_IGNORE_CPU_AFFINITY 可忽略）。
      */
      ncclAffinity affinity;
    } cpu;  // 一个 NUMA 域
    struct {
      uint64_t device; // PCI device id（vendor/device/subsystem 组合），用于识别这个 PCIe 设备是什么
    } pci;  // PCIe 锚点
  };
  int nlinks;  // 这个顶点的邻接边条数
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];  // 每个顶点的边数上限 576
  /*
  预计算路径缓存。paths[t] 指向从本顶点到所有 type==t 顶点的路径列表，每个元素描述一条路径，
  经过的 link 序列 + 跳数 count + 瓶颈带宽 bw + 路径质量 type。把顶点间最短路径预先算好缓存，避免建图/搜索时反复重算。
  */
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];  // 预计算路径缓存，节点类型数 10
  /*
  拓扑搜索时的 DFS 回溯位标记。判断某 GPU 是否已被当前搜索路径占用，
  进递归前 gpu->used ^= flag 置位、递归返回后再 gpu->used ^= flag 清除。
  用 uint64_t 而非 bool，是因为不同 pattern/channel 的搜索用不同的 flag 位，64 个 bit 能让多个搜索上下文共存、互不干扰。
  */
  uint64_t used;
};

// 同类型节点的集合
struct ncclTopoNodeSet {
  int count;  // 有效节点数量
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];  // 同类节点集最多 640 个顶点
};

struct ncclTopoNetRailKeyList {
  std::map<uint64_t, int> keys;
};

/*
代表一个通信域能看到的全部物理拓扑，所有 GPU、NIC、CPU/NUMA、PCIe switch 等，以及它们之间的连接，它本质是一张图。它可以跨多个 host，
普通单机时只含本机拓扑，MNNVL（如 GB200 NVL72）时多个节点的拓扑被融合进同一个 ncclTopoSystem，靠 hostHashes[]/nHosts/systemId 区分"哪个节点"。
*/
struct ncclTopoSystem {
  int systemId;  // 本机认领到的 host 下标
  uint64_t hostHashes[NCCL_TOPO_MAX_NODES];  // 所有 host 的 hash 表(≤640)
  int nHosts;  // host 台数
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];  // 按类型分为 10 个桶存放图的顶点
  // 建图（ncclTopoCompute）阶段的输入参数，不是拓扑探测阶段填
  /*
  单 channel（per-SM）能达到的最大带宽上界，实际是所有 DEV 中，到指定类型目标（由 inter 决定是 NET 还是 DEV）的最宽单条路径的带宽，
  取全局最大。因为一个 channel 走一条路径，它构成「单 channel 带宽」的上界。一个 channel 在设备端通常由一个（或一小撮）SM 驱动。
  */
  float maxBw;
  /*
  所有 DEV 中，单个 DEV（即单块 GPU）本地直接邻边容量的最大值。NVLink 边带宽累加，PCIe 取单条，
  两者取较大者（因为这两类路径互斥，不能同时用）。构成「从同一块 GPU 出发的所有 channel 带宽之和」的上界。
  */
  float totalBw;
  int inter;  // 是否跨节点，0=所有 rank 都在本 NVLink 域内，1=有 rank 在别处需走网络
  // mirrors the comm's values
  bool cuMemGdrSupport;  // global cuMem GDR support
  int minDriverVersion;  // min CUDA driver version across ranks
  // NET selection cache of the graph search. Obtained by ncclTopoCompute for its duration, NULL otherwise.
  struct ncclTopoSelectNetsCache* selectNetsCache;
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
// Removing a node invalidates computed paths. Callers must remove any paths before calling this
// function and recompute them before using the topology for path-dependent operations.
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);
void ncclTopoRemovePaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId, int* intermediateRank);
ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min);
ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max);
ncclResult_t ncclTopoSplitNvLink(struct ncclTopoSystem* system, int* splitNvLink);

enum {
  NCCL_NET_MERGE_POLICY_ALL = 0,
  NCCL_NET_MERGE_POLICY_RAIL = 1
};

/*
NCCL 有四类异构网络后端，gin（GPU 发起网络）、rma、collnet（SHARP 网内集合卸载）、net（通用点对点网络）。
它们的插件对象类型各不相同（ncclGin_t/ncclRma_t/ncclCollNet_t/ncclNet_t），但"枚举设备 → 按需融合成 vNic → 填 <nic> 节点"的流程是同一套。
这个结构把四类后端的能力标志 + comm 配置 + 计数簿记函数 + API vtable 抹平成一个统一画像，让同一个 ncclTopoProcessNet 能处理任意一类。
*/
struct ncclTopoNetInfo {
  /*
  类型标志，四个互斥布尔，恰有一个为 1，标记当前导入的是哪类后端。
  下游据此给 <nic> 节点写对应标记（如 coll=1 带 collnet 属性），四类填充时各自把自己的位置 1、其余清 0。
  */
  bool coll;
  bool gin;
  bool rma;
  bool net;
  // communicator-specific information
  /*
  插件库下标。三类下标空间不同，gin 取 comm->ginPluginIndex、rma 取 comm->rmaPluginIndex、collnet 与 net 都取 comm->netPluginIndex。
  collnet/net 共用下标，是因为它们本就是同一插件库的两套接口，只读不同的计数字段。它被传给 getDevCount/setVirtDevCount 用来定位是哪块插件库的设备表。
  */
  int netPluginIndex;
  /*
  单个 vNic 最多能聚合几个物理 NIC。只有 collnet/net 填，按 comm->ncclNetVer >= 12 选 NCCL_NET_MAX_DEVS_PER_NIC（v12 = 8）或 _V11（= 4）。
  一旦 ndevs > maxDevsPerNic 就 WARN "Tried to merge too many NICs" 报错，自动合并循环在 ndevs == maxDevsPerNic 时 break。gin/rma 不填（无融合）。
  */
  int maxDevsPerNic;
  // 取自 comm->dmaBufSupport，四类都填
  bool dmaBufSupport;
  // NIC fusion，只 collnet/net 填
  /*
  合并的拓扑距离阈值（PATH_* 枚举，越小代表两设备越近）。从 NCCL_NET_MERGE_LEVEL 解析（kvConvertToInt + nicPathKvList），
  默认 PATH_PORT。消费在 ncclTopoAutoMerge：两物理 NIC 间 paths[i*nPhysDevs+j] <= mergeLevel 才允许并进同一 vNic。
  */
  int mergeLevel;
  /*
  NCCL_NET_MERGE_POLICY_ALL(0) 或 _RAIL(1)，从 NCCL_NET_MERGE_POLICY 解析，默认 ALL。
  消费在 ncclTopoAutoMerge：ALL 只要 path 够近就合并；RAIL 还额外要求两 NIC 的 railId 相同（用于 rail-optimized 网络只合并同轨网卡）。
  */
  int mergePolicy;
  // 来自 NCCL_NET_FORCE_MERGE 环境变量，用户显式指定哪些 NIC 合并成组。消费在 ncclTopoForceMerge 解析该字符串，绕过自动合并直接按用户指定分组。
  const char* forceMerge;
  // dev count tracking functions (not part of ncclNet)
  /*
  四类都绑（ncclGinGetDevCount/ncclRmaGetDevCount/ncclCollNetGetDevCount/ncclNetGetDevCount）。
  返回该插件库的物理 NIC 数与虚拟 NIC 数。ncclTopoProcessNet 开头第一件事就是调它拿 nPhysicalNics/nVirtualNics。
  虚拟数用哨兵 NCCL_UNDEF_DEV_COUNT(-1) 表示"还没算过"。
  */
  ncclResult_t (*getDevCount)(int, int*, int*);
  // 只 collnet/net 绑，gin/rma 留 NULL。融合出 vNic 后把虚拟设备数缓存回插件的进程级全局状态（实现"每进程只融合一次"），同进程下一个 comm 直接读缓存、不再重复融合。
  ncclResult_t (*setVirtDevCount)(int, int);
  // ncclNet API functions
  // 插件名（gin->name / rma->name / comm->ncclCollNet->name / comm->ncclNet->name），用于日志与 <nic> 标记。
  const char* name;
  // 按设备号取网卡能力（pciPath/guid/port/speed/maxComms/ptrSupport 等），后续据此填 <nic> 节点属性。
  ncclResult_t (*getProperties)(int, ncclNetProperties_t*);
  /*
  把多个物理 NIC 融合成一个虚拟设备的插件回调。gin/rma 是 NULL、collnet/net 是真实回调。
  gin/rma 因 NULL 走"只用物理设备、整个融合分支短路"，collnet/net 才进融合分支。
  */
  ncclResult_t (*makeVDevice)(int*, ncclNetVDeviceProps_t*);
  // 取设备总数（物理 + 虚拟）。融合后 ncclTopoProcessNet 用 devices(&nDevs) 拿增长后的合并数，反推 nVirtualNics = nDevs - nPhysicalNics。
  ncclResult_t (*devices)(int*);

  // system-wide list of unique rail keys
  struct ncclTopoNetRailKeyList* railKeyList;
};

ncclResult_t ncclTopoProcessNet(ncclXml* xml, const char* dumpXmlFile, struct ncclTopoNetInfo* net);
ncclResult_t ncclTopoGetXmlCpuArch(ncclXml* xml, int* cpuArch);
ncclResult_t ncclTopoGetFusionEnv(int* mergeLevel, const char** forceMerge, int cudaCompCap, int cpuArch);

// Owned subset of network properties that remains valid after the topology lock is released.
struct ncclTopoNetPropertiesSnapshot {
  bool propertiesValid;
  ncclNetVDeviceProps_t vProps;
  int speed;
  int port;
  ncclNetDeviceType netDeviceType;
  char name[PATH_MAX];
  char pciPath[PATH_MAX];
};

// Caller keeps comm and its network plugin alive. When requireCommInitialized is true, the plugin is queried only
// after communicator initialization succeeds. The caller owns *snapshots.
ncclResult_t ncclTopoGetNetPropertiesSnapshot(struct ncclComm* comm, bool requireCommInitialized,
                                              struct ncclTopoNetPropertiesSnapshot** snapshots, int* nSnapshots);

#define NCCL_TOPO_XML_MAX_NODES 256  // 能装下一整台机器的拓扑的 nodes 数量上限
#define NCCL_GRAPH_XML_MAX_NODES 65536
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, uint64_t localHostHash);
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode* xmlGraphs, struct ncclTopoSystem* system,
                                     struct ncclTopoGraph* graph, int* nChannels);
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system,
                                      struct ncclXml* xml);

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax);
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw);

// 根据 ID 找到指定顶点类型的顶点在类型数组中的索引
static ncclResult_t ncclTopoIdToIndex(struct ncclTopoSystem* system, int type, int64_t id, int* index) {
  *index = -1;
  for (int i = 0; i < system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *index = i;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

// 在拓扑图中查找指定 rank 的 GPU 顶点的索引
static ncclResult_t ncclTopoRankToIndex(struct ncclTopoSystem* system, int rank, int* index, bool showWarn) {
  *index = -1;
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].gpu.rank == rank) {
      *index = i;
      return ncclSuccess;
    }
  }
  if (showWarn) WARN("ncclTopoRankToIndex could not find rank %d", rank);
  return ncclInternalError;
}

// 找到指定 DEV 顶点
static ncclResult_t ncclTopoDevToRank(struct ncclTopoSystem* system, int systemId, int dev, bool warn, int* rank) {
  *rank = -1;
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    // Only consider GPUs on the given node
    if (NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id) != systemId) continue;  // 确保同节点
    if (system->nodes[GPU].nodes[i].gpu.dev == dev) {  // 相同 CUDA device 号
      *rank = system->nodes[GPU].nodes[i].gpu.rank; // 返回对应的 rank
      return ncclSuccess;
    }
  }
  if (warn) WARN("ncclTopoDevToRank could not find rank for nvml dev %d in systemId %d", dev, systemId);
  return ncclInternalError;
}

extern struct kvDict nicPathKvList[];

static ncclResult_t ncclTopoIdToNetDev(struct ncclTopoSystem* system, int64_t id, int* netDev) {
  *netDev = -1;
  for (int i = 0; i < system->nodes[NET].count; i++) {
    if (system->nodes[NET].nodes[i].id == id) {
      *netDev = system->nodes[NET].nodes[i].net.dev;
      return ncclSuccess;
    }
  }
  WARN("Could not find NET with id %lx", id);
  return ncclInternalError;
}

// Returns NVLink bw in GB/s
static float ncclTopoNVLinkBw(int cudaCompCap) {
  return RUBIN_AND_LATER(cudaCompCap) ? RUBIN_NVLINK_BW :
         cudaCompCap >= 100           ? SM100_NVLINK_BW :
         cudaCompCap >= 90            ? SM90_NVLINK_BW :
         cudaCompCap == 86            ? SM86_NVLINK_BW :
         cudaCompCap >= 80            ? SM80_NVLINK_BW :
         cudaCompCap >= 70            ? SM70_NVLINK_BW :
         cudaCompCap >= 60            ? SM60_NVLINK_BW :
                                        SM80_NVLINK_BW;
}

// 判断 val 是否是以 2 为底的幂
static bool isPow2(int val) {
  return (val & (val - 1)) == 0;
}

// 在 pow2 = 2^n 的前提下，把 val 的低 n 位整体反转
static int mirrorBits(int val, int pow2) {
  int mirror = 0;
  for (int b = 1, mb = (pow2 >> 1); b < pow2; b <<= 1, mb >>= 1) {
    if (val & b) mirror |= mb;
  }
  return mirror;
}
#endif
