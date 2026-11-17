/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "device.h"
#include "os.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include "gdrwrap.h"

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile = NULL);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm);
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks);
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink);
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected);
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm);

// Query topology
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId,
                               int peerRank, int64_t* id, int* dev, int* proxyRank);
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p,
                              int* read, int* intermediateRank, int* cudaP2p, int* isCrossClique = nullptr);
ncclResult_t ncclTopoCheckMNNVL(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2,
                                int* ret);
enum ncclTopoGdrMode {
  ncclTopoGdrModeDisable = 0,  // GDR 不可用，数据须经主机内存中转
  ncclTopoGdrModeDefault = 1,  // 正常 GDR，网卡直接 DMA 读写 GPU 显存
  ncclTopoGdrModePci = 2,  // 走 PCIe 映射（C2C 系统专用，需要额外的 flush），C2C 系统的 PCIe NIC 与 C2C 同步路径之间没有顺序保证，所以必须走 Pci 模式并配合显式 flush
  ncclTopoGdrModeNum = 3
};
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int rank, int64_t netId, int read,
                              enum ncclTopoGdrMode* gdrMode);

enum ncclTopoFlushType {
  ncclTopoFlushNone = 0,   // no flush needed
  ncclTopoFlushAlways = 1, // flush always needed
  ncclTopoFlushC2c = 2     // PCIe NIC and C2C sync path are unordered, flush is needed.
};
static inline uint32_t ncclGdcPinFlag(enum ncclTopoFlushType flush) {
  if (flush == ncclTopoFlushC2c && ncclGdrPinV2Available()) return GDR_PIN_FLAG_FORCE_PCIE;
  return GDR_PIN_FLAG_DEFAULT;
}
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank,
                               enum ncclTopoFlushType* flush);
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw);
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net);
int ncclPxnDisable(struct ncclComm* comm);
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks);
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu);

ncclResult_t ncclGetUserP2pLevel(int* level);

// Find CPU affinity
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity);

#define NCCL_TOPO_CPU_ARCH_UNDEF 0
#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_ARCH_ARM 3
#define NCCL_TOPO_CPU_ARCH_MIXED 4
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define NCCL_TOPO_CPU_VENDOR_MIXED 4
#define NCCL_TOPO_CPU_MODEL_INTEL_BDW 1
#define NCCL_TOPO_CPU_MODEL_INTEL_SKL 2
#define NCCL_TOPO_CPU_MODEL_INTEL_SRP 3
#define NCCL_TOPO_CPU_MODEL_INTEL_ERP 4
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN12 1
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN34 2
#define NCCL_TOPO_CPU_MODEL_AMD_ZEN5 3
#define NCCL_TOPO_CPU_MODEL_YONGFENG 1
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* comm, int* localGinDevs, int* localGinCount);
ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm* comm, int* localRmaDevs, int* localRmaCount);
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex);
ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw);

enum netDevsPolicy {
  NETDEVS_POLICY_AUTO = 0x0,  // 默认，无数量概念
  NETDEVS_POLICY_ALL = 0x1,  // 用全部本地网卡，无数量概念
  NETDEVS_POLICY_MAX = 0x2,  // 设置最大上限，例如 MAX:8
  NETDEVS_POLICY_UNDEF = 0xffffffff
};
ncclResult_t ncclTopoGetNetDevsPolicy(enum netDevsPolicy* policy, int* policyNum);

/*
拓扑系统(ncclTopoSystem)里同类节点数量的容量上限。
注释的意思是这个值要能装下最多 576 个 GPU(例如 NVL144 这类机架级 NVLink 域)，
640 = 576 + 64，多出的 64 是留给内部操作的余量(headroom)。
*/
// Allows for up to 576 GPUs (e.g., NVLD144) with headroom for internal operations
#define NCCL_TOPO_MAX_NODES 640
ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                              int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType);
ncclResult_t ncclTopoGetDevNodes(struct ncclTopoSystem* system, int64_t baseId, struct ncclTopoNode** nodes,
                                 int* nNodes);

// Local (myself)
#define PATH_LOC 0

// Connection traversing NVLink
#define PATH_NVL 1

// Connection through NVLink using an intermediate GPU
#define PATH_NVB 2

// Connection through C2C
#define PATH_C2C 3

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 4

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
#define PATH_PXB 5

// Connection between a GPU and a NIC using the C2C connection to the CPU and the PCIe connection to the NIC
#define PATH_P2C 6

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network
// send/recv operations.
#define PATH_PXN 7

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 8

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
#define PATH_SYS 9

// Connection through the network
#define PATH_NET 10

// New type of path which should precede PATH_PIX
#define PATH_PORT PATH_NVL

// Disconnected
#define PATH_DIS 11
extern const char* topoPathTypeStr[];

// Init search. Needs to be done before calling ncclTopoCompute
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

#define NCCL_TOPO_PATTERN_BALANCED_TREE \
  1   // Spread NIC traffic between two GPUs (Tree parent + one child on first
                                            // GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE \
  2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree
                                            // children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct

/*
ncclTopoGraph 是一次拓扑搜索(ncclTopoCompute)的输入约束 + 输出结果的载体。
每个算法(ring/tree/collnet/nvls...)对应一个图对象。搜索前 NCCL 填入"我要什么形态的拓扑"(输入字段)，
搜索后算法填回"我找到的最优拓扑长这样"(输出字段)。
*/
struct ncclTopoGraph {
  // 输入/输出
  /*
  (ring:0, tree:1, collnet:2, nvls:3, collnetDirect:4)是图自己的编号空间，
  用于日志/打印/dump。它和 NCCL_ALGO_*不是一套编号，别混用。
  */
  int id;
  /*
  取 NCCL_TOPO_PATTERN_*,决定搜索算法走哪条分支:
  值	宏	              含义
  1	  BALANCED_TREE	   树:NIC 流量摊到两 GPU(父+一子在 GPU0、另一子在 GPU1)
  2	  SPLIT_TREE	     树:父在 GPU0、两子在 GPU1
  3	  TREE	           树:所有 NIC 流量集中同一 GPU(搜索时要求"树对称"，net->id 必须等于 startNet->id)
  4	  RING	           环
  5	  NVLS	           NVLS+SHARP 与 NVLS+Tree
  6	  COLLNET_DIRECT	 CollNet 直连
  */
  int pattern;
  /*
  跨 NIC 策略，控制一个 channel 能否"从一张网卡进、另一张网卡出"。
    0：不许跨，且要求进/出 NIC 同 railId(rail-local，走同一条轨道)
    2：特殊模式(环且 channel 数为奇数时，要求首尾 NIC 一致
    其它(如 1)：允许跨 NIC
  */
  int crossNic;
  /*
  是否启用 CollNet，标记该图是否用 CollNet(如 SHARP 网内聚合)，影响搜索时是否把 net 节点当聚合根处理。
  */
  int collNet;
  /*
  channel 数上下界，搜索的硬约束。minChannels 不达标直接不采纳，maxChannels 限制搜索不超过的上限(受 MAXCHANNELS=64 封顶)。
  */
  int minChannels;
  int maxChannels;

  // 输出(搜索填回)
  /*
  实际搜出的 channel 数，NCCL 把数据流拆到多个并行 channel 上跑以提升带宽利用率。
  ncclTopoCompareGraphs 的择优逻辑：先尽量让 ring 和 tree 的 nChannels 一致，
  再比 nChannels * bwIntra(总带宽)，最后比 nHops。
  */
  int nChannels;
  int nCtasPerChannel;
  // 节点内 GPU 间链路带宽(GB/s)，取决于走的是 NVLink 还是 PCIe
  float bwIntra;
  // 跨节点带宽，受 NIC 速率限制
  float bwInter;
  // 跨节点路径的延迟估计，供 tuner 算成本、选算法/协议时用
  float latencyInter;
  /*
  路径"质量"的枚举，取值见下表(数值越小越优)：
  值	    宏	     含义
  0	    PATH_LOC	自己
  1	    PATH_NVL	直连 NVLink
  2	    PATH_NVB	经中间 GPU 的 NVLink
  3	    PATH_C2C	C2C(如 Grace-Hopper)
  4	    PATH_PIX	至多过一个 PCIe bridge
  5	    PATH_PXB	过多个 PCIe bridge(不过 Host Bridge)
  6	    PATH_P2C	GPU↔NIC 经 C2C 到 CPU 再 PCIe 到 NIC
  7	    PATH_PXN	GPU↔NIC 经中间 GPU(rail-local 聚合收发)
  8	    PATH_PHB	过 PCIe + Host Bridge(通常 CPU)
  9	    PATH_SYS	过 PCIe + NUMA 间 SMP 互连(QPI/UPI)
  10	  PATH_NET	经网络
  11	  PATH_DIS	不连通
  下面两个值是后续决定 GDR 是否可用、是否 PXN、用哪种 transport 的关键依据。
  */
  // 节点内 GPU 互连的最小路径类型，全 NVLink 就是 PATH_NVL
  int typeIntra;
  // GPU 到 NIC 的最小路径类型，理想是 GPU 与 NIC 同 PCIe switch 下(PIX)
  int typeInter;
  /*
  各 channel 是否同构
  标记搜出的多个 channel 是否结构完全一致(相同的 GPU 序列与 NIC 分配)。
  若为真，下游 ncclTopoPreset/Postset 可只按 channel 0 的接法复制到所有 channel，
  省去逐 channel 重算。若为假，则每个 channel 的连接要分别处理。
  */
  int sameChannels;
  /*
  总跳数
  搜索路径的累计跳数，在 ncclTopoFollowPath 里 graph->nHops += mult * path->count 累加。
  它是择优的最后一道 tie-breaker，带宽相同时，nHops 更小者胜。
  */
  int nHops;
  /*
  每 channel 节点内 GPU 序列
    声明上界：MAXCHANNELS(64) * NCCL_TOPO_MAX_NODES(640)(可支持到 576 GPU)。
    实际索引：graph->intra[graph->nChannels * ngpus + step]，其中 ngpus = system->nodes[GPU].count(本节点的 GPU 数，远小于 640)。
    布局：按 channel 分片，stride = ngpus。第 c 个 channel 的切片 intra[c*ngpus .. c*ngpus+ngpus-1] 存的是该 channel 上节点内 GPU 的 rank 顺序序列:
      intra[c*ngpus + 0] = 起始 GPU(ncclTopoSearchTryGpu 里的 g);
      intra[c*ngpus + step](step 从 1)= 其余 GPU 按搜索顺序的 gpu.rank。
    用途：ncclTopoReplayGetGpu 就靠 intra[(nChannels-1)*ngpus + step + 1] 回放上一个 channel 的 GPU 顺序，
  ncclTopoPreset 据此为每个 channel 建 ring/tree 的节点内连接。
  */
  int intra[MAXCHANNELS * NCCL_TOPO_MAX_NODES];
  /*
  每 channel 的进/出 NIC
    声明上界：MAXCHANNELS * 2，stride = 2(每 channel 两个 int64_t)。
    实际索：graph->inter[channel * 2] 与 graph->inter[channel * 2 + 1]。
    布局：每个 channel 存一对 netId(int64_t，网卡的全局唯一 id):
      inter[c*2] = 该 channel 的入向 NIC(getNetPaths 取 inter[nChannels*2] 作为当前在建 channel 的 net);
      inter[c*2+1] = 该 channel 的出向 NIC(BALANCED_TREE 校验时比对 inter[nChannels*2+1])。
    为何是 netId(int64_t)而非小整数：net id 是拓扑系统里网卡节点的全局标识，跨节点统一。
  crossNic 策略正是靠比较进/出 netId(及其 railId)来判定是否允许跨 NIC。
  */
  int64_t inter[MAXCHANNELS * 2];
};

ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);

struct ncclTopoRanks {
  int crossNicRing;
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
  int nvlsHeads[MAXCHANNELS];
  int nvlsHeadNum;
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
                             struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs,
                             struct ncclComm* parent);

ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm);
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs);
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes,
                                 int numPipeOps, float* time);

#endif
