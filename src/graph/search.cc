/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "core.h"
#include "graph.h"
#include "topo.h"
#include "transport.h"
#include "xml.h"
#include <math.h>

/*
网络拓扑允不允许 NIC 之间跨 rail 互通。默认 NCCL_CROSS_NIC=2。
  0：入口和出口网卡必须同 railId，railId 未定义时退化为「同 asic 且同 port」
  1：完全不检查，任意两张网卡都能做一进一出
  2：只对奇数号通道检查，必须接回上一个通道的出口网卡，偶数号通道不检查
*/
NCCL_PARAM(CrossNic, "CROSS_NIC", 2);

// 从 DEV 的路径矩阵里取最宽的一条，如果跨节点就计算 DEV 到 NET 的最大带宽，否则计算 DEV 到 DEV 的最大带宽
// Initialize system->maxBw. This is the per-channel (i.e. per-SM)
// max bw.
static float getMaxBw(struct ncclTopoSystem* system, struct ncclTopoNode* dev, int type) {
  float maxBw = 0.0;
  for (int i = 0; i < system->nodes[type].count; i++) {  // 遍历 DEV 到所有指定类型顶点的 paths
    struct ncclTopoLinkList* path = dev->paths[type] + i;
    float bw = path->bw;
    if (path->count == 0) continue;
    maxBw = std::max(maxBw, bw);  // 统计最大带宽
  }
  return maxBw;
}

static float getTotalBw(struct ncclTopoSystem* system, struct ncclTopoNode* dev) {
  float nvlinkBw = 0.0, pciBw = 0.0;
  for (int l = 0; l < dev->nlinks; l++) {  // 遍历 DEV 的直连 links
    struct ncclTopoLink* link = dev->links + l;
    if (link->type == LINK_NVL) nvlinkBw += link->bw;  // NVLink 累加
    if (link->type == LINK_PCI) pciBw = link->bw;  // PCIe 记录最后一条
  }
  return std::max(pciBw, nvlinkBw);  // 返回累加的 NVLink 带宽和 PCIe 带宽的较大值
}

// 计算每个 DEV 顶点到 NET/DEV 路径带宽上限，以及 NVLink 和 PCIe 外连带宽上限
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  system->maxBw = 0.0;
  system->totalBw = 0.0;
  int inter = system->inter;
  if (inter == 0 && system->nodes[GPU].count == 1) {  // 如果不跨 LSA 并且 GPU 个数为 1
    // 带宽上限都设为最大值
    system->maxBw = LOC_BW;
    system->totalBw = LOC_BW;
    return ncclSuccess;
  }
  for (int d = 0; d < system->nodes[DEV].count; d++) {
    struct ncclTopoNode* dev = system->nodes[DEV].nodes + d;
    // 计算所有 DEV 到所有 NET/DEV 路径的最大带宽，如果跨节点就是 DEV->NET，不跨节点是 DEV->DEV
    system->maxBw = std::max(system->maxBw, getMaxBw(system, dev, inter ? NET : DEV));
    // 区分 DEV 外连的两种类型的链路 NVLink 和 PCIe，获取 NVLink 总带宽和 PCIe 带宽的较大值
    system->totalBw = std::max(system->totalBw, getTotalBw(system, dev));
  }
  return ncclSuccess;
}

// 从拓扑图中选一个 CPU 顶点获取 CPU 架构和厂商记录在 comm 中
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) {
  // We assume there is at least one CPU and that the CPUs have the same
  // architecture and vendor.
  const struct ncclTopoNodeSet* cpus = &comm->topo->nodes[CPU];
  comm->cpuArch = cpus->nodes[0].cpu.arch;
  comm->cpuVendor = cpus->nodes[0].cpu.vendor;
  return ncclSuccess;
}

// 输入 node1 和 node2，找到 node2 到 node1 的反向 link
static ncclResult_t findRevLink(struct ncclTopoNode* node1, struct ncclTopoNode* node2, int type,
                                struct ncclTopoLink** revLink) {
  for (int l = 0; l < node2->nlinks; l++) {
    struct ncclTopoLink* link = node2->links + l;
    if (link->remNode == node1 && link->type == type) {
      *revLink = link;
      return ncclSuccess;
    }
  }
  WARN("Could not find rev link for %d/%ld -> %d/%ld", node1->type, node1->id, node2->type, node2->id);
  return ncclInternalError;
}

// 量化到 0.001 GB/s
// This is unfortunately needed since manipulating floats often results in rounding errors.
#define SUB_ROUND(a, b) (a = roundf((a - b) * 1000) / 1000)

// 沿一条已算好的路径，逐跳占用或归还带宽的执行器。它自己不做任何路径质量判断，只管带宽账。
static ncclResult_t followPath(struct ncclTopoLinkList* path, struct ncclTopoNode* start, int maxSteps, float bw,
                               int* steps) {
  /*
  Intel 的 Root Complex 把 GPU 之间的 P2P 流量切成 64 字节的小 TLP，头部开销占比上升，
  有效载荷效率降到 5/6。所以记账时按 1.2 倍扣，才能让搜索结果反映真实可用带宽。
  */
  float pciBw = bw;
  for (int step = 0; step < path->count; step++) {  // 遍历 path 边
    struct ncclTopoNode* node = path->list[step]->remNode;
    if (node->type == CPU) {  // 如果跳到的节点是 CPU
      // Account for P2P inefficiency through Intel CPU RC
      if (path->type == PATH_PHB && start->type == GPU && node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
          node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {  // 路径过 CPU（PATH_PHB）、起点是 GPU、x86 架构、Intel 厂商
        pciBw = INTEL_P2P_OVERHEAD(bw);
      }
    }
  }

  struct ncclTopoNode* node = start;
  for (int step = 0; step < maxSteps; step++) {
    struct ncclTopoLink* link = path->list[step];
    struct ncclTopoLink* revLink = NULL;
    float fwBw = link->type == LINK_PCI ? pciBw : bw;  // 如果是 pci，link 带宽采用更新后的 pci 带宽
    float revBw = 0;
    // 落点是 DEV 且 cudaCompCap < 80 且 start->type != GPU，把惩罚限定在「从网卡进来」的路径上（NET → GPU 的首跳）。GPU → GPU 的普通环内跳转不受影响
    if (link->remNode->type == DEV && link->remNode->dev.cudaCompCap < 80 && start->type != GPU) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw / 8;  // Ampere 之前的 GPU 做中转时，反向链路有额外开销
    }
    // 落点是 CPU 且 POWER 架构且边是 LINK_NVL
    if (link->remNode->type == CPU && link->remNode->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER && link->type == LINK_NVL) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw;  // POWER9 的 NVLink 直连，反向同样占满
    }
    // Coverity thinks that revLink could be NULL below.  However, we access it only if revBw is non-0, and the
    // logic of the code is that revBw can become non-0 only if revLink is non-NULL (see the "if" statement right
    // above).
    // coverity[var_deref_op]
    if (link->bw < fwBw || (revBw && revLink->bw < revBw)) {  // 如果正向边的剩余额度不够，或被挤占的反向边的剩余额度不够
      *steps = step;  // 精确表达已扣条数，为回滚提供边界
      return ncclSuccess;
    }
    SUB_ROUND(link->bw, fwBw);  // link->bw 减去 fwBw
    if (revBw) SUB_ROUND(revLink->bw, revBw);  // 反向边也减去被挤占的带宽
    node = link->remNode;  // 下一跳
  }
  *steps = maxSteps;
  return ncclSuccess;
}

// 尝试找到从指定的 tpye1 类型 index1 顶点到 tpye2 类型 index2 顶点的 path，在这个过程中会改变路径剩余带宽（如果找到符合条件的 path）
// Try to go from node type1/index1 to no type2/index2. mult indicates whether we are counting the
// bandwidth (1) or undoing (-1).
static ncclResult_t  ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1,
                                       int index1, int type2, int index2, float mult, struct ncclTopoNode** node) {
  // First handle easy cases
  *node = system->nodes[type2].nodes + index2;
  if (type1 == -1) return ncclSuccess;  // type1=-1 说明是节点内通信且第一个是 GPU，不需要从网卡进，直接返回第一个 GPU 顶点
  struct ncclTopoNode* node1 = system->nodes[type1].nodes + index1;  // 顶点 1
  struct ncclTopoNode* node2 = system->nodes[type2].nodes + index2;  // 顶点 2

  if (node1->paths[type2] == nullptr || node2->paths[type1] == nullptr) {
    WARN("No path computed to go from %s/%d to %s/%d", topoNodeTypeStr[type1], index1, topoNodeTypeStr[type2], index2);
    return ncclInternalError;
  }

  // 顶点 1 到顶点 2 的  path
  struct ncclTopoLinkList* path = node1->paths[type2] + index2;
  // 顶点 2 到顶点 1 的反向 path
  struct ncclTopoLinkList* revPath = node2->paths[type1] + index1;

  // Now check link type
  *node = NULL;  // 设置默认值
  /*
  只有两端都是计算或交换顶点才算节点内，实际出现的组合：
    type1 → type2	        intra	        用哪套额度
    NET → GPU	            false	        bwInter / typeInter
    GPU → NET	            false	        bwInter / typeInter
    GPU → GPU           	true	        bwIntra / typeIntra
    NVS → GPU	            true	        bwIntra / typeIntra
    GPU → NVS	            true	        bwIntra / typeIntra
    -1 → GPU	            早退了	       无
  */
  int intra = (type1 == GPU || type1 == NVS) && (type2 == GPU || type2 == NVS);
  float bw = intra ? graph->bwIntra : graph->bwInter;
  int type = intra ? graph->typeIntra : graph->typeInter;

  if (path->type >= PATH_DIS) return ncclSuccess;  // 不可达
  if (mult == 1 && (path->type > type)) return ncclSuccess;  // type 是允许的最差路径质量，比它还差直接返回
  /*
  反向质量，只对树家族生效。树的同一条链路要承载上行和下行两份流量，
  所以两个方向的路径质量都必须达标。环只沿一个方向转，反向路径长什么样无所谓。
  */
  if (mult == 1 &&
      (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE || graph->pattern == NCCL_TOPO_PATTERN_TREE ||
       graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) &&
      (revPath->type > type))
    return ncclSuccess;

  bw *= mult;

  // Check there is enough bandwidth on paths.
  int step = 0;
  NCCLCHECK(followPath(path, node1, path->count, bw, &step));  // 判断当前路径是否能容纳 bw 带宽，或者在第几步不够
  if (step < path->count) goto rewind;  // path 不够容纳 bw

  // Enough bandwidth : return destination node.
  graph->nHops += mult * path->count;  // 带宽足够累加跳数
  *node = system->nodes[type2].nodes + index2;  // 返回目标顶点
  return ncclSuccess;

rewind:
  // Not enough bandwidth : rewind and exit.
  NCCLCHECK(followPath(path, node1, step, -bw, &step));  // 归还带宽
  return ncclSuccess;
}

/*
这块 GPU 所挂的 DEV 设备，与它上游那个 PCIe 顶点之间这条链路的带宽。
它不是 GPU 到网卡的带宽，也不是 GPU 到 GPU 的带宽，而是 GPU 的 PCIe 插槽侧带宽
*/
static int gpuPciBw(struct ncclTopoNode* gpu) {
  struct ncclTopoNode* dev = gpu->gpu.parent;
  for (int l = 0; l < dev->nlinks; l++) {
    struct ncclTopoLink* gpuLink = dev->links + l;
    if (gpuLink->type != LINK_PCI) continue;
    struct ncclTopoNode* pci = gpuLink->remNode;
    for (int l = 0; l < pci->nlinks; l++) {
      struct ncclTopoLink* pciLink = pci->links + l;
      if (pciLink->remNode != dev) continue;
      return std::min(gpuLink->bw, pciLink->bw);
    }
  }
  return -1;
}

/* Choose the order in which we try next GPUs. This is critical for the search
   to quickly converge to the best solution even if it eventually times out. */
struct ncclGpuScore {
  int g;             // Retain the index
  int startIndex;    // Least important
  int intraNhops;
  int intraBw;
  int interNhops;
  int interPciBw;
  int interBw;    // Most important
};

static int cmpScore(const void* g1, const void* g2) {
  struct ncclGpuScore* s1 = (struct ncclGpuScore*)g1;
  struct ncclGpuScore* s2 = (struct ncclGpuScore*)g2;
  int d;
  if ((d = (s2->interBw - s1->interBw))) return d;  // 入网卡到 GPU 的带宽大优先
  if ((d = (s2->interPciBw - s1->interPciBw))) return d;  // GPU 侧 PCIe 带宽大优先
  if ((d = (s1->interNhops - s2->interNhops))) return d;  // 入网卡到 GPU 路径跳数小优先
  if ((d = (s2->intraBw - s1->intraBw))) return d;  // 前一个 GPU 到当前 GPU 路径带宽大优先
  if ((d = (s1->intraNhops - s2->intraNhops))) return d;  // 前一个 GPU 到当前 GPU 路径跳数小优先
  return s1->startIndex - s2->startIndex;  // GPU 索引小优先
}

// 只比 intraBw 与 intraNhops 两项，是否都相等
static int cmpIntraScores(struct ncclGpuScore* scores, int count) {
  int intraBw = scores[0].intraBw;
  int intraNhops = scores[0].intraNhops;
  for (int i = 1; i < count; i++) {
    if (scores[i].intraBw != intraBw || scores[i].intraNhops != intraNhops) return 1;
  }
  return 0;
}

// 根据 rank 查找 GPU 索引
static ncclResult_t getGpuIndex(struct ncclTopoSystem* system, int rank, int* index) {
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      *index = g;
      return ncclSuccess;
    }
  }
  WARN("Could not find gpu rank %d", rank);
  return ncclInternalError;
}

// 获取指定 ID 的网卡索引
static ncclResult_t getNetIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  for (int n = 0; n < system->nodes[NET].count; n++) {
    if (system->nodes[NET].nodes[n].id == id) {
      *index = n;
      return ncclSuccess;
    }
  }
  WARN("Could not find net id %lx", id);
  return ncclInternalError;
}

// 获取当前 channel 入网卡到所有 GPU 的路径表
static ncclResult_t getNetPaths(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                struct ncclTopoLinkList** netPaths) {
  int64_t netId = graph->inter[graph->nChannels * 2];  // 当前 channel 入网卡
  int n;
  NCCLCHECK(getNetIndex(system, netId, &n));  // 获取入网卡索引
  *netPaths = system->nodes[NET].nodes[n].paths[GPU];  // 获取入网卡到所有 GPU 的路径表
  return ncclSuccess;
}

// NVSwitches prefer when we talk to a limited set of peers: only the previous and next GPUs are candidates,
// in an order fixed by the pattern. Scoring and sorting all GPUs would only be discarded.
// NVSwitch 拓扑直接构造相邻 GPU 候选；旧版排序后再截断的推导保存在 2.30.7 历史注释中。
static ncclResult_t ncclTopoSearchNextGpuSortNvl(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int index,
                                                 int ngpus, struct ncclTopoLinkList* paths, uint64_t flag, int* next,
                                                 int* countPtr) {
  int prevGpu = (index - 1 + ngpus) % ngpus;
  int nextGpu = (index + 1) % ngpus;
  int candidates[2];
  int nCandidates = 0;
  if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
    candidates[nCandidates++] = prevGpu;
    candidates[nCandidates++] = nextGpu;
  } else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
    candidates[nCandidates++] = nextGpu;
    candidates[nCandidates++] = prevGpu;
  } else {
    candidates[nCandidates++] = nextGpu;
  }
  if (nCandidates == 2 && prevGpu == nextGpu) nCandidates = 1;
  int count = 0;
  for (int i = 0; i < nCandidates; i++) {
    int g = candidates[i];
    if (paths[g].count == 0) continue; // There is no path to that GPU
    if (system->nodes[GPU].nodes[g].used & flag) continue;
    next[count++] = g;
  }
  *countPtr = count;
  return ncclSuccess;
}

// 给定当前 GPU，产出「下一跳候选列表」next[] 与个数 *countPtr，按优先级排序。调用方按顺序逐个 TryGpu 递归
ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                       struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  const uint64_t flag = 1ULL << (graph->nChannels);  // 获取当前 channel 的 used flag
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoLinkList* paths = gpu->paths[GPU];  // 当前 GPU 到所有 GPU 的路径表
  // 指针减法把顶点转成数组索引
  int index = gpu - system->nodes[GPU].nodes;

  if (system->nodes[NVS].count) {
    return ncclTopoSearchNextGpuSortNvl(system, graph, index, ngpus, paths, flag, next, countPtr);
  }

  struct ncclTopoLinkList* netPaths = NULL;
  if (sortNet) NCCLCHECK(getNetPaths(system, graph, &netPaths));  // 获取当前 channel 入网卡到所有 GPU 的路径表

  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
  memset(scores, 0, ngpus * sizeof(struct ncclGpuScore));
  int count = 0;  // 候选计数
  for (int i = 1; i < ngpus; i++) {
    int g = (index + i) % ngpus;  // 从「自己的下一个」开始绕一圈
    // 排除从当前 GPU 不可达的 GPU
    if (paths[g].count == 0) continue; // There is no path to that GPU
    if (system->nodes[GPU].nodes[g].used & flag) continue;  // 排除已经到过的 GPU
    scores[count].g = g;  // 记录索引
    scores[count].startIndex = i;  // 记录旋转距离
    scores[count].intraNhops = paths[g].count;  // 当前 GPU 到 GPU g 的路径跳数
    scores[count].intraBw = paths[g].bw;  // 但前 GPU 到 GPU g 的路径带宽
    if (netPaths) {  // 如果考虑网卡
      scores[count].interNhops = netPaths[g].count;  // 入网卡到 GPU g 的跳数
      scores[count].interPciBw = gpuPciBw(system->nodes[GPU].nodes + g);  // 获取 GPU 插槽侧 PCIe 带宽
      scores[count].interBw = netPaths[g].bw;  // 入网卡到 GPU g 的带宽
    }
    count++;
  }

  // 按照上述优先级条件排序候选 GPU
  // Sort GPUs
  qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

  // 回连网卡还远时，只比 intraBw 与 intraNhops 两项，是否都相等
  // Check if all have the same intra-node score in which case we go reverse for sortNet = -1
  if (sortNet == -1 && cmpIntraScores(scores, count) == 0) {
    for (int i = 0; i < count; i++) next[i] = scores[count - 1 - i].g;  // 给出反序
  } else {
    for (int i = 0; i < count; i++) next[i] = scores[i].g;  // 正序
  }

  *countPtr = count;  // 先写入全量个数
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                               struct ncclTopoGraph* saveGraph, int* time);

// Try to keep all searchs within one second
#define NCCL_SEARCH_GLOBAL_TIMEOUT (1ULL << 19)
#define NCCL_SEARCH_TIMEOUT (1 << 14)
#define NCCL_SEARCH_TIMEOUT_TREE (1 << 14)
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1 << 8)

#define FORCED_ORDER_PCI 1
#define FORCED_ORDER_REPLAY 2

// 获取上一个 channel 指定 step 的 GPU 索引，replay
ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  *g = -1;
  if (graph->nChannels == 0) return ncclInternalError;  // 需要已有上一条 channel
  int ngpus = system->nodes[GPU].count;
  int nextRank = graph->intra[(graph->nChannels - 1) * ngpus + step + 1];  // 获取上一个 channel 对应 step 的 rank
  for (int i = 0; i < ngpus; i++) {
    if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
      *g = i;  // 返回 rank 对应 GPU 索引
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                  struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet,
                                  int backToFirstRank, int forcedOrder, int* time);

// 给「递归搜索下一个 GPU」这件事套上两层可逆的资源占用：带宽和访问标记。进去时占，出来时还，中间夹着完整的递归。
ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                  struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank,
                                  int forcedOrder, int* time, int type, int index, int g) {
  const uint64_t flag = 1ULL << (graph->nChannels);
  struct ncclTopoNode* gpu;
  // 尝试找到从指定的 tpye 类型 index 顶点到 GPU g 顶点的 path，在这个过程中会改变路径剩余带宽（如果找到符合条件的 path）
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  if (gpu) {
    gpu->used ^= flag;  // 标记该 GPU 已经被当前 channel 用过了，flag = 1ULL << (graph->nChannels)
    // 判断当前 gpu 的下一步决策
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag;  // 回溯当前 gpu 上的 used flag
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));  // 回溯占用的路径带宽
  }
  return ncclSuccess;
}

/*
COLLNET_DIRECT 的节点内不是链，是以 head g 为中心的星形。mul 把 head 的 bwIntra 预算平摊到 ngpus-1 条辐条，
两个方向各合计一份 bwIntra。注意两轮循环只扣与 g 相连的路径，非 head GPU 之间的两两路径不建模。这与星形语义一致，
数据是「所有 GPU 汇聚到 head，head 上网，结果回来再由 head 分发」。两端都是 GPU，所以 intra = true，用的是 bwIntra / typeIntra。
*/
ncclResult_t ncclTopoSearchTryCollnetDirect(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                            struct ncclTopoGraph* saveGraph, int g, int ngpus, int* time) {
  int fwdg = 0;
  int bwdg = 0;
  struct ncclTopoNode* gpu = NULL;
  float mul = 1.0 / (float)(system->nodes[GPU].count - 1);  // 预期带宽按照 n-1 个 GPU 均分
  do {
    /*
    尝试 GPU g 到所有其他 GPU 的路径，如果满足带宽要求会扣除已用带宽。
    fwdg == g 那一轮是空操作，因为自己到自己的路径 basePath->count = 0;
    */
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, mul, &gpu));
  } while (gpu && ++fwdg < system->nodes[GPU].count);

  if (gpu != NULL) {  // 全部路径都成功
    // 反向带宽再做一遍
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, mul, &gpu));
    } while (gpu && ++bwdg < system->nodes[GPU].count);
    if (gpu != NULL) {  // 两个方向都成功了
      // Both directions worked. Now we already have head, so pop the all other intra ranks.
      int step = 1;
      // 把除了 head 之外的 rank 都加到 graph->intra，从索引为 1 开始
      for (int index = 0; index < ngpus; ++index) {
        if (index != g) {
          graph->intra[graph->nChannels * ngpus + step] = system->nodes[GPU].nodes[index].gpu.rank;
          step++;
        }
      }
      // NULL + ngpus 直跳完成分支
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    // 归还反向带宽
    while (bwdg) {
      bwdg--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, -mul, &gpu));
    }
  }
  // 归还正向带宽
  while (fwdg) {
    fwdg--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, -mul, &gpu));
  }
  return ncclSuccess;
}

/*
NVLS 的节点内部分不是 GPU 链，而是「所有 GPU ↔ 单个 NVS 顶点」的双向全互连。这个函数一次性把 2 × ngpus 条边全部记账，然后直接跳到通道完成。
*/
ncclResult_t ncclTopoSearchTryNvls(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                   struct ncclTopoGraph* saveGraph, int g, int ngpus, int* time) {
  struct ncclTopoNode* nvs;
  struct ncclTopoNode* gpu;
  int d0 = 0; // See if there is enough bandwidth for NVS->GPU traffic
  do {
    /*
    先找 NVS 0（NVS 顶点全系统只创建一个）节点到 GPU d0 的路径带宽是否满足要求。
    g 是本通道的 head GPU，它除了参与 NVLS 规整，还要负责与网卡的网络中继，所以 NVS 链路承担双份流量。
    */
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? 2 : 1, &gpu));
    d0++;
  } while (gpu && d0 < system->nodes[GPU].count);
  if (gpu == NULL) {  // 如果有任意 NVS 到 GPU 的路径不满足要求都不行
    d0--;
  } else {
    int d1 = 0; // See if there is enough bandwidth for GPU->NVS traffic
    // 检查 GPU 到 NVS 的流量带宽是否充足
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? 2 : 1, &nvs));
      d1++;
    } while (nvs && d1 < system->nodes[GPU].count);
    if (nvs == NULL) {  // 如果有任意 GPU 到 NVS 的路径不满足要求都不行
      d1--;
    } else {
      // step = ngpus 让 RecGpu 直接命中完成分支
      // Both directions worked. Move on to the next path.
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    while (d1) {  // 归还 GPU→NVS
      d1--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? -2 : -1, &nvs));
    }
  }
  while (d0) {  // 归还 NVS→GPU
    d0--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? -2 : -1, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                   struct ncclTopoGraph* refGraph, int* copy) {
  // 1. Try to get the same nChannels between Rings and Trees
  if (graph->nChannels < graph->minChannels) return ncclSuccess;  // 要达到最小 channels 数

  const bool evenReference = refGraph->nChannels > 0 && !(refGraph->nChannels & 1);
  const bool evenReferenceIsBetter = refGraph->nChannels * refGraph->bwIntra >= graph->nChannels * graph->bwIntra;

  // Favor an even number of channels when aggregate bandwidth is equal or better.
  if (graph->pattern != NCCL_TOPO_PATTERN_NVLS && evenReference && (graph->nChannels & 1) &&
      graph->nChannels < system->nodes[NET].count && evenReferenceIsBetter)
    return ncclSuccess;

  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // NVLS 通道对应于从 NVLS 读取数据的 GPU。因此，通道越多越好。
    // NVLS channels correspond to GPUs pulling from NVLS. So the more the better.
    if (graph->nChannels > refGraph->nChannels && graph->nChannels <= system->nodes[GPU].count) *copy = 1;
    // 所有 channels 的 inter 带宽总和越大越好
    if (graph->nChannels * graph->bwInter > refGraph->nChannels * refGraph->bwInter) *copy = 1;
    return ncclSuccess;
  }
  // 所有 channels 的 intra 带宽总和越大越好
  // 2. Try to get better bandwidth
  if (graph->nChannels * graph->bwIntra > refGraph->nChannels * refGraph->bwIntra) {
    *copy = 1;
    return ncclSuccess;
  }
  // 如果 intra 带宽总和更小，直接不用看
  if (graph->nChannels * graph->bwIntra < refGraph->nChannels * refGraph->bwIntra) return ncclSuccess;

  // 如果 intra 带宽总和不变，pattern、crossNic 一样的情况下，nHops 越小越好
  // 3. Less hops
  if (graph->pattern == refGraph->pattern && graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) {
    *copy = 1;
  }
  return ncclSuccess;
}

// 获取所有 GPU（gpu=-1）或指定 gpu 对应的所有网卡列表，先按照 GPU 顺序排列，即外层为 channel，每个 GPU 对应的网卡跳着排列
// Add the preferred NICs ordered by GPU first
static ncclResult_t ncclTopoPrefNetsGpuFirst(struct ncclTopoSystem* system, int gpu, int nets[NCCL_TOPO_MAX_NODES],
                                             int* netCount) {
  const int nGpus = (gpu == -1) ? system->nodes[GPU].count : 1;  // gpu == -1 时遍历所有 GPU，否则只遍历 1 个
  int gpuCount = nGpus;
  int gpuIds[NCCL_TOPO_MAX_NODES] = {gpu};  // gpu == -1 时先把 gpuIds[0] 设成 -1、其余设成 0。只对局部场景（nGpus == 1，gpuIds[0] = gpu）有意义
  int firstNets[NCCL_TOPO_MAX_NODES];
  if (gpu == -1)
    for (int g = 0; g < nGpus; g++) gpuIds[g] = g;

  for (int c = 0; c < MAXCHANNELS; c++) {  // 遍历所有 channel
    for (int g = 0; g < nGpus; g++) {  // 遍历所有 GPU
      if (gpuIds[g] == -1) continue;  // 遇到 -1 跳过
      int localNet;
      int64_t netId;
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + gpuIds[g];  // 找到 GPU 顶点
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));  // 找到 GPU rank + channel 对应的那张网卡 ID
      NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &localNet));  // NET ID 转为数组索引
      // store the first net found for each GPU in case of duplicates
      if (c == 0) firstNets[g] = localNet;  // channel 为 0 时逐 GPU 记录对应的第一张网卡
      // if the NET has already been returned for channel 0, that GPU is done
      if (c > 0 && firstNets[g] == localNet) {  // 如果当前 GPU + channel 回到第一张网卡了
        gpuIds[g] = -1;  // 当前 GPU 标记为不需要继续了
        gpuCount--;
        continue;
      }
      // only add it to the list if it doesn't already exist
      int found = 0;
      while (found < (*netCount) && nets[found] != localNet) found++;  // 线性去重遍历 nets
      if (found == (*netCount)) nets[(*netCount)++] = localNet;  // 如果遍历到尾部还没找到说明可以作新 NET 加到 nets
    }
    if (gpuCount == 0) break;  // 没有需要处理的 GPU，提前退出
  }
  return ncclSuccess;
}

// 获取所有 GPU（gpu=-1）或指定 gpu 对应的所有网卡列表，先按照 channel 顺序排列，即外层为 GPU，内层先排列一个 GPU 对应的所有网卡
// Add the preferred NICs ordered by channels first
static ncclResult_t ncclTopoPrefNetsChannelFirst(struct ncclTopoSystem* system, int gpu, int nets[NCCL_TOPO_MAX_NODES],
                                                 int* netCount) {
  for (int g = 0; g < system->nodes[GPU].count; g++) {  // 外层遍历 GPU
    if (gpu != -1 && gpu != g) continue;
    int localNetCount = 0, localNets[MAXCHANNELS];
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    for (int c = 0; c < MAXCHANNELS; c++) {  // 内层遍历 channel
      int64_t netId;
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));  // 获取当前 GPU rank + channel 对应的网卡 ID
      NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, localNets + localNetCount));
      if (localNetCount > 0 && localNets[localNetCount] == localNets[0]) break;  // 当前 GPU 轮转一圈（netsPerGpu+1）
      localNetCount++;
    }
    // 将网卡去重添加到 nets
    // Append NICs to list
    for (int i = 0; i < localNetCount; i++) {
      int n = localNets[i];
      int found = 0;
      while (found < (*netCount) && nets[found] != n) found++;
      if (found == (*netCount)) nets[(*netCount)++] = n;
    }
  }
  return ncclSuccess;
}

/*
函数产出一份按优先级排好的候选网卡清单，规模服从用户的 NCCL_NETDEVS_POLICY 策略。
gpu 传 -1 时面向所有 GPU 构建、用于通道起点选入口网卡，传具体索引时只面向该 GPU、用于通道末尾折返出口网卡。
清单分两段拼成，先收集各 GPU 的偏好网卡，再在策略允许时把其余满足 typeInter 且未收录的网卡追加在后。
*/
// Build a sorted list of the NETs to try, the list will follow the NETDEVS_POLICY set by the user.
//
// The value of "gpu" can be set to -1 to build a list suitable for all GPUs (for example for the search start).
// The value of "gpu" can be set to the desired index when trying to get back to the NIC.
//
// The list is built the following way:
// 1. First gather the preferred NETs for each of the GPU(s), based on the NETDEVS_POLICY and the connection.
// 2. If the NETDEV_policy allows it, add all the other NETs satisfying typeInter but not already in
//    the list of preferred NETs.
//
// The result is cached for the duration of a search (see ncclTopoSelectNets), so it must depend only on the
// precomputed paths and static node attributes, never on state the search mutates (link and NET bandwidth,
// used flags).
/*
NCCL_MNNVL_SCATTER_NETS_ENABLE=1(默认开启)，在 MNNVL 系统里把网卡分散到不同 GPU 旁边，避免所有通道都挤在同一块网卡上
*/
NCCL_PARAM(ScatterEnable, "MNNVL_SCATTER_NETS_ENABLE", 1);
// 获取全部 GPU 对应的按照一定顺序排列的网卡
static ncclResult_t ncclTopoSelectNetsCompute(struct ncclTopoSystem* system, int typeInter, int gpu, int* nets,
                                              int* netCountRet) {
  ncclResult_t ret = ncclSuccess;
  int netCount = 0;

  // First add the preferred NETs.
  if (system->nHosts > 1 && ncclParamScatterEnable() && ncclParamCrossNic() != 0) {  // MNNVL 多个 hosts 的拓扑被融合进了一份拓扑图中，并且把网卡分散到不同 GPU 旁边
    // For MNNVL systems, we sort the devices by GPU first, then by channel.
    // This logic cannot guarantee no cross-rail connectivity with different GPU count per NVLD.
    // When cross-rail connectivity must be avoided (CROSS_NIC=0), we use the ChannelFirst one.
    // 获取所有 GPU 或指定 GPU 对应的所有网卡列表，先按 GPU 顺序，channel 主序
    NCCLCHECK(ncclTopoPrefNetsGpuFirst(system, gpu, nets, &netCount));
  } else {  // 普通跨节点 IB 集群每个进程只看到自己 host 的拓扑
    // 获取所有 GPU 或指定 GPU 对应的所有网卡列表，先按 channel 顺序，GPU 主序
    // For other systems, we sort the devices by channel first, then by GPU
    NCCLCHECK(ncclTopoPrefNetsChannelFirst(system, gpu, nets, &netCount));
  }

  // Get the maximum of network devices allowed, depending on the policy.
  // If the policy is not MAX, then allow all devices.
  int maxDevCount = 0;  // 允许的最大网卡数
  enum netDevsPolicy netDevsPolicy;
  NCCLCHECK(ncclTopoGetNetDevsPolicy(&netDevsPolicy, &maxDevCount));  // 获取 GPU 和网卡的映射策略
  if (gpu == -1) maxDevCount *= system->nodes[GPU].count;
  if (netDevsPolicy != NETDEVS_POLICY_MAX) maxDevCount = NCCL_TOPO_MAX_NODES;  // 只有 MAX 策略才真正限量，AUTO 和 ALL 一律放开
  if (netCount >= maxDevCount) goto exit;

  // 把那些「路径质量达标、但带宽竞争不过别人」的网卡捞回来
  // Then add others satisfying typeInter
  for (int t = 0; t <= typeInter; t++) {  // 遍历所有 GPU->NET 的路径类型
    for (int g = 0; g < system->nodes[GPU].count; g++) {
      // do not consider this GPU is it's not the GPU we asked for
      if (gpu != -1 && gpu != g) continue;  // 跳过不关注的 gpu
      int localNetCount = 0, localNets[MAXCHANNELS];
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
      struct ncclTopoLinkList* paths = gpu->paths[NET];  // 当前 GPU 到所有 NET 的 paths
      for (int n = 0; n < system->nodes[NET].count && n < MAXCHANNELS; n++) {  // 遍历所有 NET，并且当前 GPU 到每个 NET 都对应一个 channel
        if (paths[n].type == t) localNets[localNetCount++] = n;  // 当前 GPU->NET 路径类型刚好是当前遍历的
      }
      // 如果还有些网卡没在 nets 中，就添加到 nets 中
      // Append NICs to list
      for (int i = 0; i < localNetCount; i++) {
        int n = localNets[i];
        int found = 0;
        while (found < netCount && nets[found] != n) found++;
        if (found == netCount) nets[netCount++] = n;
        if (netCount >= maxDevCount) goto exit;  // 网卡上限
      }
    }
  }

exit:
  *netCountRet = netCount;
  return ret;
}

// The search queries the same (typeInter, gpu) selection at every step of every channel of every pass, so
// ncclTopoCompute installs a cache in the system and each selection is computed once.
struct ncclTopoSelectNetsCache {
  int count; // -1: not computed yet
  int* nets;
};
// One entry per (typeInter + 1, gpu + 1), covering the typeInter == -1 and gpu == -1 queries.
#define NCCL_TOPO_SELECT_NETS_CACHE_SIZE(ngpus) ((PATH_DIS + 2) * ((ngpus) + 1))

static ncclResult_t ncclTopoSelectNetsCacheCreate(struct ncclTopoSystem* system) {
  if (system->selectNetsCache != NULL) {
    WARN("Graph search NET selection cache is already installed");
    return ncclInternalError;
  }
  int nEntries = NCCL_TOPO_SELECT_NETS_CACHE_SIZE(system->nodes[GPU].count);
  NCCLCHECK(ncclCalloc(&system->selectNetsCache, nEntries));
  for (int i = 0; i < nEntries; i++) system->selectNetsCache[i].count = -1;
  return ncclSuccess;
}

static void ncclTopoSelectNetsCacheDestroy(struct ncclTopoSystem* system) {
  struct ncclTopoSelectNetsCache* cache = system->selectNetsCache;
  if (cache == NULL) return;
  int nEntries = NCCL_TOPO_SELECT_NETS_CACHE_SIZE(system->nodes[GPU].count);
  for (int i = 0; i < nEntries; i++) free(cache[i].nets);
  free(cache);
  system->selectNetsCache = NULL;
}

// Return the NETs to try for (typeInter, gpu), see ncclTopoSelectNetsCompute. The list belongs to the cache and
// stays valid until ncclTopoCompute returns.
static ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system, int typeInter, int gpu, const int** nets,
                                       int* netCount) {
  struct ncclTopoSelectNetsCache* entry = system->selectNetsCache;
  if (entry == NULL) {
    WARN("Graph search NET selection used outside of ncclTopoCompute");
    return ncclInternalError;
  }
  entry += (typeInter + 1) * (system->nodes[GPU].count + 1) + (gpu + 1);
  if (entry->count < 0) {
    // ncclTopoSelectNetsCompute writes at most one entry per NET.
    NCCLCHECK(ncclCalloc(&entry->nets, std::max(system->nodes[NET].count, 1)));
    NCCLCHECK(ncclTopoSelectNetsCompute(system, typeInter, gpu, entry->nets, &entry->count));
  }
  *nets = entry->nets;
  *netCount = entry->count;
  return ncclSuccess;
}

// NCCL_MNNVL_RAIL_PER_HOST=0，
NCCL_PARAM(MnnvlRailPerHost, "MNNVL_RAIL_PER_HOST", -1);

static bool ncclTopoMnnvlRailPerHost(struct ncclTopoSystem* system) {
  int railPerHost = ncclParamMnnvlRailPerHost();
  if (railPerHost == -1) {
    railPerHost = 1;
    for (int g = 0; g < system->nodes[GPU].count && railPerHost == 1; g++) {
      railPerHost &= RUBIN_AND_LATER(system->nodes[GPU].nodes[g].gpu.cudaCompCap) ? 1 : 0;
    }
  }
  return railPerHost;
}

// // 判断当前 NET n 是否可以作为出网卡
static bool ncclTopoSearchCheckNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                   struct ncclTopoNode* startNet, int n, int step) {
  struct ncclTopoNode* net = system->nodes[NET].nodes + n;
  // Always forbid connections between different networking planes
  // 始终禁止不同网络平面之间的连接（如果这两个平面均已定义）。
  if (net->net.planeId != startNet->net.planeId) return false;

  // Tree 算法传入和传出网卡要是相同的
  if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) return false; // Trees are symmetric
  // Ring、crossNic=2、当前 channel 是奇数，当前传出网卡必须等于上一个 channel 的入网卡
  if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2) {
    if (graph->nChannels & 1 && net->id != graph->inter[(graph->nChannels - 1) * 2]) return false;
  // 定义了平面，两张网卡必须同平面
  } else if (graph->crossNic == 0) {  // Ring、crossNic=0，必须同轨
    // Different hosts have the same rails.
    // If the user did not ask for MNNVL rail per host, we need to ensure that the system IDs are matching as well.
    // 在 MNNVL 融合拓扑里，当网卡没有显式 railId 时，用 PCI 插槽位置而不是芯片 GUID 来判定「不同 host 上的两张网卡是否属于同一条 rail」
    if (net->net.railId != startNet->net.railId) return false;
    if ((net->net.railId & NCCL_TOPO_UNDEF_BIT) && !ncclTopoMnnvlRailPerHost(system) &&
        NCCL_TOPO_ID_SYSTEM_ID(net->id) != NCCL_TOPO_ID_SYSTEM_ID(startNet->id))
      return false;  // 否则采用 asic，跨 hosts 全局唯一
  }
  // BALANCED_TREE 的第二个根（step 1 的 GPU）必须与第一个根（step 0 的 GPU）走同一张出口网卡
  if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE && step != 0 &&
      net->id != graph->inter[graph->nChannels * 2 + 1]) {
    return false;
  }
  return true;
}

// 当前在 GPU gpu 上，已经走完 step 个 GPU，决定下一步去哪。整条通道就是它自己递归自己堆出来的
ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                  struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet,
                                  int backToFirstRank, int forcedOrder, int* time) {
  if ((*time) <= 0) return ncclSuccess;  // 搜索次数到上限了，退出
  (*time)--;  // 按照 step 减少一次搜索次数

  ncclResult_t ret = ncclSuccess;
  const int* nets = NULL;
  int* next = NULL;
  int ngpus = system->nodes[GPU].count;
  int g;
  // 完成一个 channel 的阶段，可以判断一次是否是最优 graph
  if (step == ngpus) {
    // Determine whether we found a better solution or not
    int copy = 0;
    graph->nChannels++;  // 通道数才 + 1
    NCCLCHECKGOTO(ncclTopoCompareGraphs(system, graph, saveGraph, &copy), ret, exit);  // 比较当前 graph 是否比最优 graph 更优
    if (copy) {  // 新的方案更好，将临时 graph 拷贝到最优的 saveGraph
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1;  // 如果 channels 到达最大值，搜索直接结束
    }
    if (graph->nChannels < graph->maxChannels) {  // 如果当前 channels 数没达到上限，外层递归
      NCCLCHECKGOTO(ncclTopoSearchRec(system, graph, saveGraph, time), ret, exit);
    }
    graph->nChannels--;  // 比较结束，回溯 channels 数
    ret = ncclSuccess;
    goto exit;
  }
  // 如果当前阶段不是完整 channel
  graph->intra[graph->nChannels * ngpus + step] = gpu->gpu.rank;  // 先记录当前 GPU rank
  g = gpu - system->nodes[GPU].nodes;  // 获取当前 GPU 顶点索引
  // 如果当前要连到网卡
  if (step == backToNet) {
    // first get back to NIC
    if (system->inter) {
      // 获取当前 channel 入网卡的索引
      int startNetIndex;
      NCCLCHECKGOTO(getNetIndex(system, graph->inter[graph->nChannels * 2], &startNetIndex), ret, exit);
      struct ncclTopoNode* startNet = system->nodes[NET].nodes + startNetIndex;
      int netCount;
      NCCLCHECKGOTO(ncclTopoSelectNets(system, graph->typeInter, g, &nets, &netCount), ret, exit);  // 获取当前 GPU 对应的所有网卡列表
      for (int i = 0; i < netCount; i++) {
        int n = nets[i];
        if (!ncclTopoSearchCheckNet(system, graph, startNet, n, step)) continue;  // 判断当前 NET n 是否可以作为出网卡
        // Balanced Tree : count half of the bandwidth on first two GPUs
        int nextBackToNet = -1;
        float bwInterSave = graph->bwInter;
        if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {  // BALANCED_TREE 会在前两个 GPUs 上均分带宽
          // Count half of the bandwidth on each of the first two GPUs
          if (step == 0) nextBackToNet = 1;  // 下一个出流量网卡在第二个 GPU
          graph->bwInter /= 2;  // 临时预期 inter 带宽减半，用于后续查 path
        }

        // 尝试找到从 GPU g 顶点到 NET n 顶点的 path，在这个过程中会改变路径剩余带宽（如果找到符合条件的 path）
        struct ncclTopoNode* net;
        NCCLCHECKGOTO(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net), ret, exit);
        graph->bwInter = bwInterSave;
        if (net) {  // 如果找到的 path 带宽没问题
          // 继续往下推进机内 GPU 链路，其实是重调用当前函数，不过将 nextBackToNet 改为 -1 (或 1)，下次不会再进入当前查找出网卡的分支
          graph->inter[graph->nChannels * 2 + 1] = net->id;  // 记录该出网卡
          NCCLCHECKGOTO(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank,
                                             forcedOrder, time),
                        ret, exit);

          // BALANCED_TREE，临时预期 inter 带宽减半，用于后续回溯 path 带宽
          // 将 GPU g 到 NET n 之前扣除的带宽回溯加回来
          if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) graph->bwInter /= 2;
          NCCLCHECKGOTO(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net), ret, exit);
          graph->bwInter = bwInterSave;
        }
      }
    }
  // 一次性完成 NVLS
  } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    NCCLCHECKGOTO(ncclTopoSearchTryNvls(system, graph, saveGraph, g, ngpus, time), ret, exit);
  // 一次性完成 collNet
  } else if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
    NCCLCHECKGOTO(ncclTopoSearchTryCollnetDirect(system, graph, saveGraph, g, ngpus, time), ret, exit);
  } else if (step < system->nodes[GPU].count - 1) {  // 普通的机内 GPU
    // Go to next GPU
    int nextGpu = -1;
    int count;
    // 物理相邻的 GPU 往往就是一条不错的环，而每步只试 1 个候选比试 ngpus-1 个便宜得多，尝试 PCI 顺序上的下一块 GPU
    if (forcedOrder == FORCED_ORDER_PCI) {
      // Try the PCI order
      nextGpu = step + 1;
      count = 1;
    } else if (forcedOrder == FORCED_ORDER_REPLAY) {  // 复用上一通道的顺序
      // Try last channel order
      NCCLCHECKGOTO(ncclTopoReplayGetGpu(system, graph, step, &nextGpu), ret, exit);
      count = 1;
    } else {
      /*
      给定当前 GPU，产出「下一跳候选列表」next[] 与个数 *countPtr，按优先级排序。
      backToNet=-1 不跨机，整条通道不碰网卡，传 0。backToNet == step + 1，下一跳回连网卡，1.
      其他，离回连网卡还有两步以上，-1。
      */
      // Normal search
      NCCLCHECKGOTO(ncclCallocQuiet(&next, ngpus), ret, exit);
      NCCLCHECKGOTO(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count,
                                              backToNet == -1       ? 0 :
                                              backToNet == step + 1 ? 1 :
                                                                      -1),
                    ret, exit);
    }
    for (int i = 0; i < count; i++) {  // 搜索所有可能的下一个 GPU 
      int nextIndex = next ? next[i] : nextGpu;
      NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, step + 1, backToNet, backToFirstRank, forcedOrder,
                                         time, GPU, g, nextIndex),
                    ret, exit);
    }
  } else if (step == backToFirstRank) {  // 当前 GPU 要回连到第一个 GPU
    // Find first GPU and loop back to it
    int p;
    NCCLCHECKGOTO(getGpuIndex(system, graph->intra[graph->nChannels * ngpus], &p), ret, exit);  // 获取当前 channel intra 第一个 rank 找到 GPU 索引
    struct ncclTopoNode* firstGpu;
    NCCLCHECKGOTO(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu), ret, exit);  // 尝试采用当前 GPU g 到首 GPU p 的路径
    // step+1，backToFirstRank 改为 -1，再调用当前函数，理论上会走到 channel 结束的分支
    if (firstGpu) {  // 带宽满足要求并扣减成功
      NCCLCHECKGOTO(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step + 1, backToNet, -1, forcedOrder,
                                         time),
                    // 回溯 GPU g 到首 GPU p 的路径带宽
                    ret, exit);
      NCCLCHECKGOTO(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu), ret, exit);
    }
  } else {
    // Next path
    NCCLCHECKGOTO(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time), ret, exit);
  }

exit:
  free(next);
  return ret;
}

ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                                  struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  ncclResult_t ret = ncclSuccess;
  const int* nets = NULL;
  int* localGpus = NULL;
  const int bw = graph->bwInter;  // 取整
  int netCount;
  int graphFound = 0;
  NCCLCHECKGOTO(ncclCallocQuiet(&localGpus, system->nodes[GPU].count), ret, exit);

  NCCLCHECKGOTO(ncclTopoSelectNets(system, graph->typeInter, -1, &nets, &netCount), ret, exit);  // 获取全部 GPU 对应的按照一定顺序排列的网卡
  for (int i = 0; i < netCount; i++) {  // 遍历所有网卡
    if ((graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) &&
        graphFound) {  // NVLS 和 collNet 这两个 pattern 只需要找到一个 head（网卡 + GPU 组合），提前退出
      break;
    }
    // nets 中的网卡是排了序的，质量高的网卡在前，nChannels 作为偏移确保下一个 channel 进来能换一个网卡
    int n = nets[(graph->nChannels + i) % netCount];
    struct ncclTopoNode* net = system->nodes[NET].nodes + n;
    if (graph->collNet && net->net.collSupport == 0) continue;  // 这张图要求网内集合卸载（SHARP），跳过不支持 collNet 的网卡
    if (net->net.bw < bw) continue;  // 如果网卡剩余带宽小于预期带宽，跳过该网卡
    /*
    RING、crossNic == 2、当前是奇数通道、当前网卡（入口）要等于上一个 channel 的出口。这里只限制了入口网卡这半。
    网卡是全双工的。a→b 用掉 a 的发送与 b 的接收，b→a 用上 b 的发送与 a 的接收，两条通道合起来吃满这对网卡的双向带宽
    */
    if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2 && (graph->nChannels & 1) &&
        net->id != graph->inter[(graph->nChannels - 1) * 2 + 1])
      continue;

    graph->inter[graph->nChannels * 2] = net->id;  // 新 channel inter 入口网卡为当前网卡
    graph->latencyInter = net->net.latency;  // 填写 inter 时延，循环里每块被试的网卡都会覆盖它，最终留下的是最后一块通过守卫的网卡的延迟

    // 把这块物理端口在所有拓扑视图里的剩余带宽同步扣减
    for (int i = 0; i < system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw -= bw;
      }
    }

    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
      // NVLS search only tries to find NIC:GPU combinations to compute the heads.
      if (graph->nChannels < netCount) {  // NVLS 的通道数对应从 NVLS 拉数据的 GPU 数，每个通道需要一个不同的 head。这个守卫确保 head 数不超过候选网卡数
        int gpu = net->net.localGpu;
        if (gpu != -1) {
          int duplicate = 0;
          // check whether there is duplicate head when one GPU connects with multiple NICs
          for (int gc = 0; gc < graph->nChannels; gc++) {
            /*
            一个 GPU 旁边挂两块网卡时，两块网卡会给出同一个 localGpu，如果不去重就会为同一个 GPU 建两条 NVLS 通道，浪费。
            存在某个通道的第一个 gpu rank 等于当前 NET 亲和的代表 gpu rank，记录重复
            */
            if (graph->intra[gc * system->nodes[GPU].count] == system->nodes[GPU].nodes[gpu].gpu.rank) {
              duplicate = 1;
              break;
            }
          }
          // 从开头的 NET n 到 gpu 开始递归搜索通道
          if (!duplicate) {  // 没有重复
            NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n,
                                               gpu),
                          ret, exit);
            graphFound = 1;
          }
        }
      }
    } else {  // RING / TREE / BALANCED_TREE / COLLNET_CHAIN
      if (graph->nChannels > 0 && graph->sameChannels == 1) {  // 已有 channel 并且要求同构，重放上一个 channel
        // Try to replay the last channel
        int g;
        // 固定重放上一个 channel 的 gpu 开始递归搜索通道
        NCCLCHECKGOTO(ncclTopoReplayGetGpu(system, graph, -1, &g), ret, exit);  // 获取上一个 channel，intra 第一个 gpu
        NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY,
                                           time, NET, n, g),
                      ret, exit);
      } else {
        if (graph->nChannels == 0 && system->nodes[NVS].count == 0) {
          // 始终先尝试 PCI 顺序以建立参考基准，但不要计算超时时间，也不要让它运行太久
          // Always try the PCI order first to set a reference, but don't count in the timeout nor let it run for long
          int t = 1 << 10;  // 独立局部预算
          NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI,
                                             &t, NET, n, 0),
                        ret, exit);
          /*
          >0 候选试完，提前结束。
          =0 1024 步耗尽（RecGpu 入口 (*time) <= 0 拦住，不再自减，所以停在 0 不会变负）
          =-1 完成分支达到 maxChannels，显式写入的完美信号
          */
          if (t == -1) *time = -1;
        }

        // NET n 到亲和的 GPU 开始执行
        // Then try the most local GPUs
        int localGpu = net->net.localGpu;
        if (localGpu != -1) {
          NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n,
                                             localGpu),
                        ret, exit);
        }
        int localGpuCount, pathType;
        NCCLCHECKGOTO(ncclTopoGetLocal(system, NET, n, GPU, localGpus, &localGpuCount, &pathType), ret, exit);  // 获取与 NET n 亲和的一组 GPU
        // if no GPUs are connected, skip this net
        if (pathType == PATH_DIS) continue;  // 跳过没有 GPU 相连的情况
        // 遍历当前网卡 NET n 所有亲和的 GPU g
        for (int g = 0; g < localGpuCount; ++g) {
          // We already tried this one  // 跳过上述已经搜索过的
          if (localGpus[g] == localGpu) continue; // We already tried this one
          NCCLCHECKGOTO(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n,
                                             localGpus[g]),
                        ret, exit);
        }
      }
    }

    // 把扣除的网卡带宽加回去
    for (int i = 0; i < system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw += bw;
      }
    }
  }

exit:
  free(localGpus);
  return ret;
}

// 首先第一个 GPU 固定连接输入 NET，然后根据算法判断输出 NET 应该连到那个 GPU，并且判断是否需要 Loop 以及那个 GPU 回连到第一个 GPU
/* Search Patterns
 *
 *     Intra-node
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 * (=Split Tree Loop)
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 * (=Split Tree)
 *
 *     Inter-node
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 * Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                              `--> NET n (or m if crossNic)
 * Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                                       `--> NET n (or m if crossNic)
 * Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                                       `--> NET n (or m if crossNic)
 */
ncclResult_t ncclTopoSearchParams(struct ncclTopoSystem* system, int pattern, int* backToNet, int* backToFirstRank) {
  if (system->inter) {
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count - 1;  // Ring 算法最后一个 GPU 顶点连到 NET
    else if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) *backToNet = 1;  // Split Tree 算法第二个 GPU 连到 NET
    else *backToNet = 0;  // 其他算法都是第一个 GPU 顶点连到 NET
    *backToFirstRank = -1;  // 跨 LSA 没有回环
  } else {
    *backToNet = -1;  // 单 LSA 不用连到 NET
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToFirstRank = system->nodes[GPU].count - 1;  // Ring 算法最后一个 GPU 连回第一个 GPU
    else *backToFirstRank = -1;  // 其他算法没有回环
  }
  return ncclSuccess;
}

// Compute the highest NVLink load on any GPU for the graph->minChannels channels of this search, in units of the
// channel speed. The intent of this function is to return the load of the most loaded GPU accurately; this
// enables us to skip searches for speeds that cannot be satisfied by the topology. If it is too high, valid
// speeds may be skipped. If it is too low, we do more work than necessary.
static ncclResult_t ncclTopoGraphMaxGpuLoad(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, float* load) {
  int ndevs = system->nodes[DEV].count;
  int minChannels = graph->minChannels;
  if (ndevs <= 1) {
    *load = minChannels;
    return ncclSuccess;
  }
  switch (graph->pattern) {
  case NCCL_TOPO_PATTERN_RING:
  case NCCL_TOPO_PATTERN_TREE:
  case NCCL_TOPO_PATTERN_BALANCED_TREE:
  case NCCL_TOPO_PATTERN_SPLIT_TREE:
    // Chains (ncclTopoSearchRecGpu): one per channel per GPU, except the last GPU of a channel when the chain does
    // not loop back. Some GPU is last in at most minChannels / ndevs channels: minChannels - minChannels / ndevs.
    {
      int backToNet, backToFirstRank;
      NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
      *load = backToFirstRank != -1 ? minChannels : minChannels - minChannels / ndevs;
      break;
    }
  case NCCL_TOPO_PATTERN_NVLS:
    // NVLS (ncclTopoSearchTryNvls): one per channel per GPU, two for the head: minChannels + 1.
    *load = minChannels + 1;
    break;
  case NCCL_TOPO_PATTERN_COLLNET_DIRECT:
    // COLLNET_DIRECT (ncclTopoSearchTryCollnetDirect): 1 for the head, 1 / (ndevs - 1) for each peer:
    // 1 + (minChannels - 1) / (ndevs - 1).
    *load = 1.0f + (minChannels - 1) * 1.0f / (ndevs - 1);
    break;
  default:
    // With load 0, speed * load > totalBw is never true, so no speed is skipped. The result is still the same,
    // but a slower, more exhaustive search is done.
    INFO(NCCL_GRAPH, "No speed pre-filter bound for graph pattern %d; trying every speed", graph->pattern);
    *load = 0;
    break;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph,
                               struct ncclTopoGraph* saveGraph, int* time) {
  /*
  backToNet：在哪个 step 上，当前 GPU 要走一条 GPU → NET 的边（把数据吐回网卡）
  backToFirstRank：在哪个 step 上，当前 GPU 要走一条 GPU → 第一个 GPU 的边（闭环）
  两个参数都用 -1 表示「不需要」
  */
  int backToNet, backToFirstRank;
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));  // 根据算法获取出口和回连 GPU
  if (system->inter) {  // 跨 LSA，channel 从 NET 开始
    // 从 NET 开始执行 channel 搜索
    // Start from NET
    NCCLCHECK(ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time));
  } else {  // 单 LSA 内
    // Intra-node only.
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {  // NVLS 每个 channel 依次将所有 GPU 作为 head
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1,
                                     graph->nChannels));
      return ncclSuccess;
    } else if (graph->nChannels == 0) {  // 第一个 channel 以 GPU 0 作为开头
      // Try PCI order first
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time,
                                     -1, -1, 0));
    } else {  // 其他情况尝试重放之前的 channel
      // Also try to replay previous channel
      int g;
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time,
                                     -1, -1, g));
    }
    if (graph->sameChannels == 0 || graph->nChannels == 0) {  // 如果不要求同构，可以搜索以所有 GPU 作为开头
      // Finally, try all other possibilities unless we are forced to use the same channels
      for (int g = 0; g < system->nodes[GPU].count; g++) {
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
      }
    }
  }
  return ncclSuccess;
}

/************************************/
/* User defined graph from XML file */
/************************************/

struct kvDict kvDictLinkType[] = {{"LOC", PATH_LOC}, {"NVL", PATH_NVL}, {"NVB", PATH_NVB}, {"PIX", PATH_PIX},
                                  {"PXB", PATH_PXB}, {"P2C", PATH_P2C}, {"PXN", PATH_PXN}, {"PHB", PATH_PHB},
                                  {"SYS", PATH_SYS}, {NULL, 0}};

// 把一条 <channel> 里 <net> 和 <gpu> 的书写顺序翻译成两个数组的下标序
ncclResult_t ncclTopoGetChannelFromXml(struct ncclXmlNode* xmlChannel, int c, struct ncclTopoSystem* system,
                                       struct ncclTopoGraph* graph) {
  // 计算当前 channel 在数组的偏移
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter + 2 * c;
  int* intra = graph->intra + ngpus * c;
  int n = 0, g = 0;
  for (int s = 0; s < xmlChannel->nSubs; s++) {  // 遍历 <channel> 下所有节点，可能是 <gpu>/<net>
    struct ncclXmlNode* sub = xmlChannel->subs[s];
    int64_t dev;
    const char* str;
    NCCLCHECK(xmlGetAttrStr(sub, "dev", &str));
    dev = strtol(str, NULL, 16);
    if (strcmp(sub->name, "net") == 0) {  // 如果是 <net> 编号记录到 inter 数组
      inter[n++] = dev;
    } else if (strcmp(sub->name, "gpu") == 0) {  // 如果是 <gpu> 编号记录到 intra 数组
      int rank = -1;
      int rankIndex = -1;
      NCCLCHECK(xmlGetAttrIndex(sub, "rank", &rankIndex));
      if (rankIndex != -1) {
        rank = strtol(sub->attrs[rankIndex].value, NULL, 0);
      } else {
        // 如果没有 rank 属性，查拓扑图找到 DEV 上的 dev 编号相同的 rank
        for (int g = 0; g < ngpus; g++) {
          int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].gpu.parent->id);
          if (NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[g].gpu.dev) == dev) {
            rank = system->nodes[GPU].nodes[g].gpu.rank;
          }
        }
        if (rank == -1) {
          WARN("XML Import Channel : dev %ld not found.", dev);
          return ncclSystemError;
        }
      }
      intra[g++] = rank;  // 将 rank 编号记录到 intra 数组
    }
  }
  /*
  以下三种写法产出 inter 和 intra 数组结果相同：inter = {A, B}、intra = {0, 1, 2}
    <channel> <net A/> <gpu 0/> <gpu 1/> <gpu 2/> <net B/> </channel>
    <channel> <net A/> <net B/> <gpu 0/> <gpu 1/> <gpu 2/> </channel>
    <channel> <gpu 0/> <gpu 1/> <gpu 2/> <net A/> <net B/> </channel>
  */
  return ncclSuccess;
}

ncclResult_t ncclTopoGetGraphFromXmlSub(struct ncclXmlNode* xmlGraph, struct ncclTopoSystem* system,
                                        struct ncclTopoGraph* graph, int* nChannels) {
  int id;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "id", &id));
  if (graph->id != id) return ncclSuccess;  // 过滤指定图类型的 ID

  // 获取图上的各个属性
  int crossNic;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "crossnic", &crossNic));
  if (ncclParamCrossNic() == 0 && crossNic == 1) return ncclSuccess;
  graph->crossNic = crossNic;

  NCCLCHECK(xmlGetAttrInt(xmlGraph, "pattern", &graph->pattern));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "nchannels", &graph->nChannels));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedintra", &graph->bwIntra));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedinter", &graph->bwInter));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlGraph, "latencyinter", &str));
  if (!str) INFO(NCCL_GRAPH, "latencyinter not found in graph, using 0.0");
  graph->latencyInter = str ? strtof(str, NULL) : 0.0;
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeintra", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeIntra, kvDictLinkType));
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeinter", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeInter, kvDictLinkType));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "samechannels", &graph->sameChannels));
  for (int s = 0; s < xmlGraph->nSubs; s++) {  // 遍历所有 channels
    NCCLCHECK(ncclTopoGetChannelFromXml(xmlGraph->subs[s], s, system, graph));
  }
  *nChannels = xmlGraph->nSubs;
  return ncclSuccess;
}

// 用 graph->id 从文件里挑出对应的那个 <graph> 节点，把属性灌进 graph
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode* xmlGraphs, struct ncclTopoSystem* system,
                                     struct ncclTopoGraph* graph, int* nChannels) {
  for (int s = 0; s < xmlGraphs->nSubs; s++) {  // 遍历所有类型的图节点
    NCCLCHECK(ncclTopoGetGraphFromXmlSub(xmlGraphs->subs[s], system, graph, nChannels));
  }
  return ncclSuccess;
}

/* And the reverse : graph->xml */
ncclResult_t ncclTopoGetXmlFromChannel(struct ncclTopoGraph* graph, int c, struct ncclTopoSystem* system,
                                       struct ncclXml* xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlChannel;
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter + 2 * c;
  int* intra = graph->intra + ngpus * c;
  NCCLCHECK(xmlAddNode(xml, parent, "channel", &xmlChannel));
  struct ncclXmlNode* node;
  if (system->inter) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[0]));
  }
  for (int g = 0; g < ngpus; g++) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "gpu", &node));
    int64_t dev = -1;
    for (int i = 0; i < ngpus; i++) {
      if (system->nodes[GPU].nodes[i].gpu.rank == intra[g]) {
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id);
        dev = NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[i].gpu.dev);
      }
    }
    if (dev == -1) {
      WARN("XML Export Channel : rank %d not found.", intra[g]);
      return ncclInternalError;
    }
    NCCLCHECK(xmlSetAttrLong(node, "dev", dev));
    NCCLCHECK(xmlSetAttrInt(node, "rank", intra[g]));
    if (graph->id == 3) break; // NVLS graphs only use the first GPU
  }
  if (system->inter) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[1]));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromGraph(struct ncclTopoGraph* graph, struct ncclTopoSystem* system, struct ncclXml* xml,
                                     struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlGraph;
  NCCLCHECK(xmlAddNode(xml, parent, "graph", &xmlGraph));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "id", graph->id));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "pattern", graph->pattern));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "crossnic", graph->crossNic));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "nchannels", graph->nChannels));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedintra", graph->bwIntra));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedinter", graph->bwInter));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "latencyinter", graph->latencyInter));
  const char* str = NULL;
  NCCLCHECK(kvConvertToStr(graph->typeIntra, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeintra", str));
  NCCLCHECK(kvConvertToStr(graph->typeInter, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeinter", str));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "samechannels", graph->sameChannels));
  for (int c = 0; c < graph->nChannels; c++) {
    NCCLCHECK(ncclTopoGetXmlFromChannel(graph, c, system, xml, xmlGraph));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system,
                                      struct ncclXml* xml) {
  xml->maxIndex = 0;
  struct ncclXmlNode* xmlGraphs;
  NCCLCHECK(xmlAddNode(xml, NULL, "graphs", &xmlGraphs));
  NCCLCHECK(xmlSetAttrInt(xmlGraphs, "version", NCCL_GRAPH_XML_VERSION));
  for (int g = 0; g < ngraphs; g++) {
    NCCLCHECK(ncclTopoGetXmlFromGraph(graphs[g], system, xml, xmlGraphs));
  }
  return ncclSuccess;
}

// 把已搜出的通道原样复制到后面的槽位，然后把标称带宽除以复制倍数。总带宽 nChannels × bw 保持不变，但并行度翻倍
ncclResult_t  ncclTopoDupChannels(struct ncclTopoGraph* graph, int ccMin, int ngpus) {
  if (graph->nChannels == 0) return ncclSuccess;
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) return ncclSuccess;
  if (graph->bwIntra < 25.0) return ncclSuccess;
  if (ccMin > 80 && graph->bwIntra < 50.0 && graph->nChannels > 4) return ncclSuccess;

  int dupChannels = std::min(graph->nChannels * 2, graph->maxChannels);  // 通道数翻倍
  // 通道复制
  memcpy(graph->intra + graph->nChannels * ngpus, graph->intra, (dupChannels - graph->nChannels) * ngpus * sizeof(int));
  memcpy(graph->inter + graph->nChannels * 2, graph->inter, (dupChannels - graph->nChannels) * 2 * sizeof(int64_t));
  // 带宽按比例缩小
  graph->bwIntra /= DIVUP(dupChannels, graph->nChannels);
  graph->bwInter /= DIVUP(dupChannels, graph->nChannels);
  graph->nChannels = dupChannels;
  return ncclSuccess;
}


/*
inter 数组的最高档 = 当代主流网卡的线速
  数组	                  [0]	      对应网卡	      换算
  speedArrayInter	        48.0	    400 Gb/s	    50 GB/s × ~0.96
  sm90SpeedArrayInter	    48.0	    400 Gb/s	    同上
  sm100SpeedArrayInter	  96.0	    800 Gb/s	    100 GB/s × ~0.96

intra 数组的最高档 = 单通道能吃到的 NVLink 份额，单条 NVLink 的单向有效带宽，约为理论值的 80%：
  代际	                NVLink 版本	    理论单向/link	        NCCL 取值	      折扣
  Pascal (sm60)	       1.0	           20 GB/s	           18.0	          0.90
  Volta (sm70)	       2.0	           25 GB/s	           20.0	          0.80
  Ampere (sm80)	       3.0	           25 GB/s	           20.0	          0.80
  Hopper (sm90)	       4.0	           25 GB/s	           20.6	          0.82
  Blackwell (sm100)	   5.0	           50 GB/s	           40.1	          0.80

而 intra 数组的最高档：
  数组	                  [0]	          与聚合带宽的比例
  speedArrayIntra	        40.0	        40 / 240（A100 12 links × 20） ≈ 1/6
  sm90SpeedArrayIntra	    60.0	        60 / 370.8 ≈ 1/6
  sm100SpeedArrayIntra	  90.0	        90 / 721.8 ≈ 1/8
规律很清楚：intra 最高档 ≈ NVLink 聚合带宽的 1/6 ~ 1/8。因为多条 channels 会划分带宽。
*/
float speedArrayIntra[] = {40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0};
float speedArrayInter[] = {48.0, 30.0, 28.0, 24.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0,
                           7.0,  6.0,  5.0,  4.0,  3.0,  2.4,  1.2,  0.24, 0.12};
#define NSPEEDSINTRA (sizeof(speedArrayIntra) / sizeof(float))
#define NSPEEDSINTER (sizeof(speedArrayInter) / sizeof(float))

float sm90SpeedArrayIntra[] = {60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0};
float sm90SpeedArrayInter[] = {48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5,
                               15.0, 12.0, 6.0,  3.0,  2.4,  1.2,  0.24, 0.12};
#define NSPEEDSINTRA_SM90 (sizeof(sm90SpeedArrayIntra) / sizeof(float))
#define NSPEEDSINTER_SM90 (sizeof(sm90SpeedArrayInter) / sizeof(float))

float sm100SpeedArrayIntra[] = {90.0, 80.0, 70.0, 60.0, 50.0, 45.0, 40.0, 30.0, 24.0, 20.0, 19.0, 18.0};
float sm100SpeedArrayInter[] = {96.0, 90.1, 86.0, 80.0, 48.0, 45.1, 42.0, 40.0, 30.0, 24.0, 22.0,
                                20.0, 17.5, 15.0, 12.0, 6.0,  3.0,  2.4,  1.2,  0.24, 0.12};
#define NSPEEDSINTRA_SM100 (sizeof(sm100SpeedArrayIntra) / sizeof(float))
#define NSPEEDSINTER_SM100 (sizeof(sm100SpeedArrayInter) / sizeof(float))

// clang-format off
float rubinSpeedArrayIntra[] = {/*4x*/280.8, /*6x*/187.2, /*8x*/140.4, /*12x*/93.6, /*16x*/70.2,
                                /*24x*/46.8,  /*32x*/35.1, /*48x*/23.4,  /*64x=*/17.55};
float rubinSpeedArrayInter[] = {
  /*4x*/ 280.8,    /*6x*/ 187.2,  /*8x*/ 140.4,  96.0,          /*12x*/ 93.6,
  /*15x*/ 74.6,    /*16x*/ 70.2,  48.0,         /*24x*/ 46.8,
  /*30x*/ 37.4,    /*31x*/ 36.1,  /*32x*/ 35.1,
  /*rest*/ 32.8, 24.6, /*48x*/23.4, 19.7, 18.1, /*64x*/17.55, 16.4, 14.4, 12.3, 9.9, 4.9, 2.5, 2.0, 1.0, 0.2, 0.1
};
// clang-format on
#define NSPEEDSINTRA_RUBIN (sizeof(rubinSpeedArrayIntra) / sizeof(float))
#define NSPEEDSINTER_RUBIN (sizeof(rubinSpeedArrayInter) / sizeof(float))

// 根据是否跨节点和最低计算能力，选择对应的 intra/inter 带宽候选表。
static void ncclTopoGetSpeedArray(int inter, int ccMin, int* nSpeeds, float** speedArray) {
  if (RUBIN_AND_LATER(ccMin)) {
    *nSpeeds = inter ? NSPEEDSINTER_RUBIN : NSPEEDSINTRA_RUBIN;
    *speedArray = inter ? rubinSpeedArrayInter : rubinSpeedArrayIntra;
  } else if (ccMin >= 100) {
    *nSpeeds = inter ? NSPEEDSINTER_SM100 : NSPEEDSINTRA_SM100;
    *speedArray = inter ? sm100SpeedArrayInter : sm100SpeedArrayIntra;
  } else if (ccMin >= 90) {
    *nSpeeds = inter ? NSPEEDSINTER_SM90 : NSPEEDSINTRA_SM90;
    *speedArray = inter ? sm90SpeedArrayInter : sm90SpeedArrayIntra;
  } else {
    *nSpeeds = inter ? NSPEEDSINTER : NSPEEDSINTRA;
    *speedArray = inter ? speedArrayInter : speedArrayIntra;
  }
}

// 读环境变量 NCCL_CROSS_NIC，告诉调用方"这套配置到底允不允许跨 NIC 通信"。
ncclResult_t ncclTopoCheckCrossNicSupport(bool* supported) {
  /*
  NCCL_CROSS_NIC 3 个值的含义：
    0：不许跨 NIC，要求一条 channel 的进/出 NIC 同 railId；
    2（默认）：特殊模式，环且 channel 数为奇数时，要求首尾 NIC 一致；
    1/其他：允许跨 NIC
  */
  *supported = (ncclParamCrossNic() != 0);
  return ncclSuccess;
}

/*
给定一张已由调用者设好 pattern、minChannels、maxChannels 的空图，它先按 GPU 架构和是否跨节点选定一张带宽候选表，
再从表里挑出「物理上可能达到的最快档位」，然后以最保守的配置进入搜索循环。要求所有 channel 结构一致、不允许跨 NIC、
路径质量取系统最优。只要搜不出解，就依次放宽一级，依次变成各通道可异构、退化成更简单的树、放宽节点内路径质量、放宽节点间路径质量、
允许跨网卡、降低带宽要求。一旦搜到「走满所有 GPU」的完美解，或是搜出的总带宽已经足以覆盖系统总容量，就停止。拿到这个解之后，
它先调用 ncclTopoDupChannels 把通道数翻倍，以填满带宽，再以翻倍后的图为新起点，反向尝试把带宽逐档提回去。
这里的倍增规则因 pattern 而异，RING 同时提升两个带宽，NVLS 只提升 bwInter，树只提升 bwIntra。最终把选出的 GPU 顺序、
出入网卡、通道数、带宽和路径质量写回 graph->intra[]、inter[]、nChannels、bw*、type* 这些字段。如果连最宽松的配置都搜不到，
并且当前不是 CollNet 或 NVLS，就返回一个「GPU 按编号顺序排列、单通道、带宽标 0.1」的退化图作为保底结果，让上层 tuner 几乎不会选择它，但至少还能跑通。
*/
ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  ncclResult_t ret = ncclSuccess;
  struct ncclTopoGraph* tmpGraph = NULL;
  int ccMin;
  int nspeeds = 0;
  float* speedArray = NULL;
  int pass = 1;
  int speedIndex = 0;
  float maxBw;
  float totalBw;
  float maxGpuLoad;
  int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
  int time;
  NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));  // 获取所有 DEV 顶点 CC 的最小值和最大值

  int ngpus = system->nodes[GPU].count;
  // 当前 NET 有多个，并且满足 3 中模式时可以考虑跨网卡
  int crossNic = (system->nodes[NET].count > 1) &&
                     (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
                      graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ?
                   ncclParamCrossNic() :
                   0;
  // 2 这种情况先考虑严格不跨网卡，后续降级再放宽
  graph->crossNic = crossNic == 1 ? 1 : 0;
  graph->nCtasPerChannel = 1;
  graph->bwIntra = graph->bwInter = 0;
  graph->latencyInter = 0;
  // Intra 空间的最小值是 LOC（GPU 自己到自己），Inter 空间的最小合法值就是 PIX（GPU 与网卡在同一个 PCIe 交换芯片下）
  int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
  int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;
  if (ngpus > 1) {
    NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));  // 计算所有 GPU 到所有 GPU 顶点的最小路径类型
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));  // 计算所有 GPU 到所有 GPU 顶点的最大路径类型
  }
  if (system->inter) {  // 如果要跨 LSA
    NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));  // 计算所有 GPU 到所有 NET 顶点的最小路径类型
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));  // 计算所有 GPU 到所有 NET 顶点的最大路径类型
    maxTypeIntra = maxTypeInter;
  }
  // Ampere 依赖于 BALANCED_TREE，该树有时需要通过 SYS 或 PHB 返回。
  if (ccMin < 90) maxTypeInter = PATH_SYS;

  graph->typeIntra = minTypeIntra;
  graph->typeInter = minTypeInter;
  graph->nChannels = 0;
  /*
  决定所有 channel 是否必须使用完全相同的 GPU 结构，强制搜索策略相同，第一个 channel 搜出什么结构，后续 channel 就沿着同样的起点重走一遍 DFS。
  因为 used 标记在每个 channel 开始前会被重置，同样的起点 + 同样的贪心规则，就会得到同样的结构。除 NVLS 以外全部「必须同构」，NVLS 则允许异构。
  
  后续每次降级都会分别跑同构、异构，这里是根据可能加载的 graph 确定它是同构还是异构。
  */
  int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;
  graph->sameChannels = trySameChannels;

  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(system, &cpuArch, &cpuVendor, &cpuModel));  // 获取 CPU 架构、厂商和代际

  /*
  从指定的 Graph File 中获取解析 XML，示例结构如下：
    <graphs version="1" name="...">
      <graph id="0" pattern="4" crossnic="0" nchannels="2"
            speedintra="370.000000" speedinter="24.000000" latencyinter="1.500000"
            typeintra="NVL" typeinter="PIX" samechannels="1">
        <channel>
          <net dev="0x1a00000000000005"/>
          <gpu dev="0x1a00000000000001" rank="0"/>
          <gpu dev="0x1a00000000000002" rank="1"/>
          <gpu dev="0x1a00000000000003" rank="2"/>
          <gpu dev="0x1a00000000000004" rank="3"/>
          <net dev="0x1a00000000000009"/>
        </channel>
        <channel>
          ... 同上结构 ...
        </channel>
      </graph>

      <graph id="1" pattern="1" ...>...</graph>   <!-- tree     -->
      <graph id="2" pattern="3" ...>...</graph>   <!-- collnet  -->
      <graph id="3" pattern="5" ...>...</graph>   <!-- nvls     -->
      <graph id="4" pattern="6" ...>...</graph>   <!-- collNetDirect -->
    </graphs>
  */
  const char* str = ncclGetEnv("NCCL_GRAPH_FILE");
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_FILE set by environment to %s", str);
    struct ncclXml* xml;
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));  // 从文件解析
    int nChannels;
    NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));  // 将 xml 解析转换为 graph 对象
    INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);
    free(xml);
    if (graph->nChannels > 0) return ncclSuccess;  // 如果指定了 NCCL_GRAPH_FILE，会跳过后续搜索流程
  }

  // NVLS 需要有 NVSwitch 并且 CC >= 90
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && (system->nodes[NVS].count == 0 || ccMin < 90)) return ncclSuccess;
  // NVLS and COLLNET_DIRECT search must have ngpus heads at most.
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    graph->maxChannels = std::min(NCCL_MAX_NVLS_ARITY, system->nodes[GPU].count);  // 重置最大 channels = min(32, ngpus)
  }
  if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
    graph->maxChannels = std::min(NCCL_MAX_DIRECT_ARITY + 1, system->nodes[GPU].count);  // 重置最大 channels = min(8, ngpus)
  }

  // 如果只有一个 GPU，模式不是 RING 就必须是 TREE
  if (ngpus == 1 && graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;

  // 如果不跨 LSA 并且模式是 NVLS，最小 channels 数 = 最大 channels 数
  if (system->inter == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // Force intra-node NVLS algorithm to pull evenly from all GPUs.
    graph->minChannels = graph->maxChannels;
  }

  int splitNvLink;
  NCCLCHECK(ncclTopoSplitNvLink(system, &splitNvLink));  // 判断是否有两个 NVLink 域
  if (graph->pattern == NCCL_TOPO_PATTERN_RING && splitNvLink) {
    /*
    如果当前是 split NVLink 拓扑，中间采用更慢的 link 连接（通常是 QPI），一般 TREE 会比 Ring 性能更好。
    但通常 Tree 的收益需要 ≥ 2 个通道才能兑现，Tree 的通道数被 Ring 锁死，所以要回过头来抬 Ring	的通道数。
    */
    // We have two sockets with NVLink and a slower link in between (typically QPI).
    // Tree is likely going to work better but it needs at least 2 channels.
    // Since Tree needs to have the same number of channels as Ring, also force Ring to use 2 channels.
    if (graph->maxChannels >= 2 && graph->minChannels == 1) graph->minChannels = 2;
  }

  NCCLCHECKGOTO(ncclCalloc(&tmpGraph, 1), ret, exit);
  memcpy(tmpGraph, graph, sizeof(struct ncclTopoGraph));
  NCCLCHECKGOTO(ncclTopoSelectNetsCacheCreate(system), ret, exit);

  ncclTopoGetSpeedArray(system->inter, ccMin, &nspeeds, &speedArray);

  // First try crossnic, then decrease bw and finally increase bwIntra.
  maxBw = system->maxBw;  // 所有 DEV 到所有 DEV/NET 的最大路径带宽
  // 对于非 ring 的 pattern，把 totalBw 放大 ndevs / (ndevs - 1) 倍，唯一的用途是决定搜索的起始带宽档位
  totalBw = system->totalBw;  // 所有 DEV 对外总和带宽，划分 NVLink 和 PCIe 带宽对比

  // Skip speeds at which the most loaded GPU exceeds its NVLink bandwidth (totalBw). See ncclTopoGraphMaxGpuLoad.
  NCCLCHECKGOTO(ncclTopoGraphMaxGpuLoad(system, graph, &maxGpuLoad), ret, exit);
  /*
  从 speedArray 中预估的 channel 带宽不能超过路径最大带宽，并且所有 channels 总带宽（按照最小 channels 数算）也不能超过单个 DEV 的出口带宽总和，
  注意这里
  */
  while ((speedArray[speedIndex] > maxBw || speedArray[speedIndex] * maxGpuLoad > totalBw) &&
         speedIndex < nspeeds - 1) {
    speedIndex++;
  }
  /*
  它的单位不是秒，也不是毫秒，而是 DFS 递归步数，ncclTopoSearchRecGpu 每被调用一次就消耗 1。
  作者实测出 524288 步 DFS 大约耗时 1 秒，于是用它作为整个 ncclTopoCompute 的时间预算上限，
  它的作用是防止降级阶梯无限展开。ncclTopoCompute 的 pass1 有七个降级维度（sameChannels、
  pattern、typeIntra、typeInter、crossNic、speedIndex），如果每一档都跑满预算，初始化可能耗时几十秒。
  */
  tmpGraph->bwIntra = tmpGraph->bwInter = speedArray[speedIndex];  // 机内和跨机带宽都先估计为一个不超过上限的较大值

// 同构搜索只需要重放上一条通道的 GPU 序列，搜索空间比异构小几个数量级 256 步，否则 16384 步
search:  // 524288
  time = tmpGraph->sameChannels                      ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
         tmpGraph->pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE :
                                                       NCCL_SEARCH_TIMEOUT;
  tmpGraph->nChannels = 0;  // 清零通道计数
  globalTimeout -= time;  // 预扣全局预算

  NCCLCHECKGOTO(ncclTopoSearchRec(system, tmpGraph, graph, &time), ret, exit);  // 执行一轮搜索，最优的算法图更新维护为 graph
#if 0
  printf("Id %d Pattern %d, crossNic %d, Bw %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s\n", tmpGraph->id, tmpGraph->pattern, tmpGraph->crossNic, tmpGraph->bwInter, tmpGraph->bwIntra, tmpGraph->typeInter, tmpGraph->typeIntra, tmpGraph->minChannels, tmpGraph->maxChannels, tmpGraph->sameChannels, graph->nChannels, graph->bwInter, graph->bwIntra, time == 0 ? "TIMEOUT" : time == -1 ? "PERFECT" : "");
  for (int c=0; c<graph->nChannels; c++) {
    printf("%2d : ", c);
    for (int g=0; g<ngpus; g++) {
      printf("%d ", graph->intra[c*ngpus+g]);
    }
    printf("[%lx %lx]", graph->inter[c*2+0], graph->inter[c*2+1]);
    printf("\n");
  }
#endif
  // Optimal solution, stop here
  if (time == -1) goto done;  // 搜满 maxChannels
  if (graph->nChannels * graph->bwInter >= system->totalBw) goto done;  // 总带宽已吃满物理上限

  // pass 1 降级阶梯，目标是「先找到任何可行解」。pass 2 的方向相反，是在已有解上抬高带宽。
  if (pass == 1) {
    // First pass, we don't have a solution yet ; try other options

    // Try having different channels (except when going through AMD CPUs)
    if (tmpGraph->sameChannels == 1 && !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD &&
                                         tmpGraph->typeIntra == PATH_SYS)) {  // 允许异构
      tmpGraph->sameChannels = 0;  // 降级到异构，重新搜索
      goto search;
    }
    tmpGraph->sameChannels = trySameChannels;

    if (time != -1) globalTimeout += time;  // 记账，超时则推出
    else globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
    if (globalTimeout < 0 && graph->nChannels) goto done;

    // Try a simpler tree
    if (ccMin >= 90 && tmpGraph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {  // BALANCED_TREE 降级为 TREE
      tmpGraph->pattern = NCCL_TOPO_PATTERN_TREE;
      goto search;
    }
    tmpGraph->pattern = graph->pattern;

    int maxIntra = system->inter ? tmpGraph->typeInter : maxTypeIntra;  // 跨节点时 intra 段的质量上限跟着 inter 段走
    // 当前 intra 路径类型要求比 maxIntra 严格，并且之前的搜索没找到满足条件的 channels 或者当前 intra 路径类型要求比最优的 graph 严格
    if (tmpGraph->typeIntra < maxIntra && (graph->nChannels == 0 || tmpGraph->typeIntra < graph->typeIntra)) {
      tmpGraph->typeIntra += 1;  // 路径质量要求降一档
      if (tmpGraph->typeIntra < PATH_DIS) goto search;
    }
    tmpGraph->typeIntra = minTypeIntra;

    // 尝试降级 inter 路径质量要求
    if (system->inter && tmpGraph->typeInter < maxTypeInter &&
        (graph->nChannels == 0 || tmpGraph->typeInter < graph->typeInter || tmpGraph->typeInter < PATH_PXN)) {
      // PXN 档位被特别允许探到，即使已有更优解也不剪枝。体现对 PXN（借中间 GPU 上网）的偏好
      tmpGraph->typeInter += 1;
      if (tmpGraph->typeInter < PATH_DIS) goto search;
    }
    tmpGraph->typeInter = minTypeInter;

    // env 为 2 且 pattern 是 RING/BALANCED_TREE
    if (crossNic == 2 && tmpGraph->crossNic == 0 &&
        (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE)) {
      // 之前 crossNic=2 被强制改为 0，不能跨网卡。现在改为允许跨网卡
      // Try again with crossNic if permitted
      tmpGraph->crossNic = 2;
      goto search;
    }
    tmpGraph->crossNic = crossNic == 1 ? 1 : 0;

    // 未到 speed 数组末尾 且 降幅不超过约一半
    // Decrease bw until we find a solution
    if ((speedIndex < nspeeds - 1) && (graph->nChannels == 0 || (speedArray[speedIndex + 1] / graph->bwInter > .49))) {
      tmpGraph->bwInter = tmpGraph->bwIntra = speedArray[++speedIndex];  // 预期带宽降一档
      goto search;
    }
    speedIndex = 0;
    while (speedArray[speedIndex] > maxBw && speedIndex < nspeeds - 1) speedIndex++;
    tmpGraph->bwIntra = tmpGraph->bwInter = speedArray[speedIndex];
  }

done:
  // We have a solution. Start from that solution and move to pass 2.
  if (pass == 1) {
    // 先构造一个「高并行度、低单通道带宽」的等价解，再尝试在保住通道数的前提下把带宽抬回去。抬得动就白赚并行度，抬不动也至少不亏
    time = -1;  // time = -1 是「pass 2 尚未搜索过」的哨兵
    NCCLCHECKGOTO(ncclTopoDupChannels(graph, ccMin, ngpus), ret, exit);
    memcpy(tmpGraph, graph, sizeof(*tmpGraph));
    speedIndex = 0;
    while (speedArray[speedIndex] > graph->bwInter && speedIndex < nspeeds - 1) speedIndex++;
    tmpGraph->bwIntra = tmpGraph->bwInter = speedArray[speedIndex];
    tmpGraph->minChannels = graph->nChannels;
    pass = 2;
  }

  if (pass == 2) {
    // See if we can increase bw
    if (time != 0 && speedIndex > 0) {
      if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
        // Ring bwIntra 与 bwInter 一起抬带宽
        // increase bw for Ring
        tmpGraph->bwIntra = tmpGraph->bwInter = speedArray[--speedIndex];
        goto search;  // 搜索
      } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && tmpGraph->bwInter == graph->bwInter &&
                 tmpGraph->bwInter < tmpGraph->bwIntra * 2) {
        tmpGraph->minChannels = tmpGraph->maxChannels = graph->nChannels;
        tmpGraph->bwInter = speedArray[--speedIndex];  // NVLS 只抬 bwInter
        goto search;
      } else if (tmpGraph->bwIntra == graph->bwIntra && tmpGraph->bwIntra < tmpGraph->bwInter * 2) {
        // increase bwIntra for trees (2 nodes or collnet)
        tmpGraph->bwIntra = speedArray[--speedIndex];  // 树家族 / CollNet，只抬 bwIntra
        goto search;
      }
    }
    time = -1;
    memcpy(tmpGraph, graph, sizeof(*tmpGraph));
  }

  // 搜完一条 channels 都没有，不是 collNet 图、不是 NVLS 图（它们是可选没有图无所谓），其他图（Ring 和 Tree）要进行兜底
  if (graph->nChannels == 0 && graph->collNet == 0 && graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
    const int* nets;
    int netCount;

    INFO(NCCL_GRAPH, "Could not find a path for pattern %d, falling back to simple order", graph->pattern);
    // 直接按 GPU 索引顺序（0,1,2,…，通常即 PCI 顺序）排成一条链。只填了第 0 通道的 ngpus 个槽
    for (int i = 0; i < ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
    graph->bwIntra = 0.1;  // 声明一个极小的带宽（0.1 GB/s）
    // 给链首链尾各挑一块网卡
    graph->typeIntra = PATH_SYS;  // 声明最差的路径质量（跨 socket / 跨系统，PATH_SYS 是有效路径里数值最大、质量最差的）

    NCCLCHECKGOTO(ncclTopoSelectNets(system, /*typeInter =*/-1, /*gpu =*/0, &nets, &netCount), ret, exit);
    graph->inter[0] = (netCount > 0 ? system->nodes[NET].nodes[nets[0]].id : -1);
    NCCLCHECKGOTO(ncclTopoSelectNets(system, /*typeInter =*/-1, /*gpu =*/ngpus - 1, &nets, &netCount), ret, exit);
    graph->inter[1] = (netCount > 0 ? system->nodes[NET].nodes[nets[0]].id : -1);
    if (graph->inter[0] != -1 && graph->inter[1] != -1) {
      // 链首链尾能挑出网卡，inter 带宽设为 0.1 GB/s，路径质量也最差
      graph->bwInter = 0.1;
      graph->typeInter = PATH_SYS;
    } else {
      // 如果链首链尾有一端挑不出网卡，不可跨节点
      graph->inter[0] = graph->inter[1] = -1;
      graph->bwInter = 0;
      graph->typeInter = PATH_DIS;
    }
    graph->nChannels = 1;  // 兜底只造一条通道，不求多
  }
  ret = ncclSuccess;
exit:
  ncclTopoSelectNetsCacheDestroy(system);
  free(tmpGraph);
  return ret;
}

// Max chars per GPU entry: " GPU/xxxxxxxxxxxxxxxx-xxxxxxxxxxxxxxxx" (~40 chars)
#define CHARS_PER_GPU_ENTRY 48

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  INFO(NCCL_GRAPH, "Pattern %d, crossNic %d, nChannels %d, bw %f/%f, type %s/%s, sameChannels %d", graph->pattern,
       graph->crossNic, graph->nChannels, graph->bwIntra, graph->bwInter, topoPathTypeStr[graph->typeIntra],
       topoPathTypeStr[graph->typeInter], graph->sameChannels);
  int ngpus = system->nodes[GPU].count;

  char* line = (char*)malloc(ngpus * CHARS_PER_GPU_ENTRY);
  for (int c = 0; c < graph->nChannels; c++) {
    sprintf(line, "%2d :", c);
    int offset = strlen(line);
    if (system->inter) {
      sprintf(line + offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2 * c]),
              NCCL_TOPO_ID_LOCAL_ID(graph->inter[2 * c]));
      offset = strlen(line);
    }
    for (int i = 0; i < ngpus; i++) {
      int g;
      ncclTopoRankToIndex(system, graph->intra[ngpus * c + i], &g, true);
      int64_t topoId = system->nodes[GPU].nodes[g].id;
      sprintf(line + offset, " %s/%lx-%lx", topoNodeTypeStr[GPU], NCCL_TOPO_ID_SYSTEM_ID(topoId),
              NCCL_TOPO_ID_LOCAL_ID(topoId));
      offset = strlen(line);
      if (graph->id == 3) break; // NVLS graphs only use the first GPU
    }
    if (system->inter) {
      sprintf(line + offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2 * c + 1]),
              NCCL_TOPO_ID_LOCAL_ID(graph->inter[2 * c + 1]));
      offset = strlen(line);
    }
    INFO(NCCL_GRAPH, "%s", line);
  }
  free(line);
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) {
  ncclResult_t ret = ncclSuccess;
  const char* str = ncclGetEnv("NCCL_GRAPH_DUMP_FILE");
  struct ncclXml* xml = NULL;
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_DUMP_FILE set by environment to %s", str);
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECKGOTO(ncclTopoGetXmlFromGraphs(ngraphs, graphs, system, xml), ret, fail);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(str, xml), ret, fail);
  }
exit:
  free(xml);
  return ret;
fail:
  goto exit;
}

#include "comm.h"
// NVLS channels aren't compute channels. Find which NIC corresponds to our rank being the head
ncclResult_t getNvlsNetDev(struct ncclComm* comm, struct ncclTopoGraph* graph, int channelId, int64_t* netId) {
  ncclResult_t ret = ncclSuccess;
  int localRanks = comm->topo->nodes[GPU].count;
  int netNum = 0;
  int64_t net[MAXCHANNELS];

  for (int c = 0; c < graph->nChannels; c++) {
    if (graph->intra[c * localRanks] == comm->rank) {
      net[netNum++] = graph->inter[c * 2];
    }
  }
  if (netNum) {
    *netId = net[channelId % netNum];
  } else {
    ret = ncclInternalError;
    goto fail;
  }

exit:
  return ret;
fail:
  WARN("Could not find NIC for rank %d in NVLS graph", comm->rank);
  goto exit;
}

// 0: don't use PXN for P2P, 1: use PXN if needed, 2: use PXN as much as possible to maximize aggregation
NCCL_PARAM(P2pPxnLevel, "P2P_PXN_LEVEL", 2);

ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId,
                               int peerRank, int64_t* id, int* dev, int* proxyRank) {
  int64_t netId = -1;
  int netDev = -1;
  if (graph) {
    // Honor the net device in the graph
    int channel = channelId % graph->nChannels;
    int ngpus = comm->topo->nodes[GPU].count;
    int index = graph->intra[channel * ngpus] == rank ? 0 : 1;
    if (graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
      netId = graph->inter[channel * 2 + index];
    } else {
      NCCLCHECK(getNvlsNetDev(comm, graph, channelId, &netId));
    }
    NCCLCHECK(ncclTopoIdToNetDev(comm->topo, netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, netId, proxyRank));
  } else if (peerRank == -1) {
    return ncclInternalError;
  } else {
    // Start with our local NIC and local Rank
    NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, channelId, &netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    *proxyRank = rank;

    int pxnLevel = ncclPxnDisable(comm) == 1 ? 0 : ncclParamP2pPxnLevel();
    // See whether we can use the remote rank preferred device.
    if (ncclParamCrossNic() == 0 || (pxnLevel != 0)) {
      // Find local NIC number close to local nvmlDev
      int nvmlDev = comm->peerInfo[peerRank].nvmlDev;
      int localRank;
      if (ncclTopoDevToRank(comm->topo, comm->topo->systemId, nvmlDev, /*warn=*/false, &localRank) != ncclSuccess) {
        return ncclSuccess;
      }
      NCCLCHECK(ncclTopoGetLocalNet(comm->topo, localRank, channelId, &netId, &netDev));

      // Check that device exists on our node
      if (ncclParamCrossNic() == 0) {
        if (dev) *dev = netDev;
        if (id) *id = netId;
      }
      if (pxnLevel == 1) {
        int g, n;
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g, /*showWarn=*/true));
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes + g;
        // No need to check for GDR here, PATH_PXN is only set if GDR is enabled between the peer GPU and the NET.
        if (gpu->paths[NET][n].type <= PATH_PXN) {
          if (dev) *dev = netDev;
          if (id) *id = netId;
          NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, netId, proxyRank));
        }
      } else if (pxnLevel == 2) {
        // Check which local GPU corresponds to that NIC and see if we can use PXN.
        int n, g1, g2;
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g1, /*showWarn=*/true));
        NCCLCHECK(ncclTopoGetLocalGpu(comm->topo, netId, &g2));
        if (g2 != -1) {
          struct ncclTopoNode* peerGpu = comm->topo->nodes[GPU].nodes + g2;
          int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;
          enum ncclTopoGdrMode gdrMode;
          NCCLCHECK(ncclTopoCheckGdr(comm->topo, peerGpu->gpu.rank, netId, 0, &gdrMode));
          if (peerGpu->paths[GPU][g1].type <= PATH_NVL && peerGpu->paths[NET][n].type <= pxnType &&
              (gdrMode != ncclTopoGdrModeDisable)) {
            *proxyRank = peerGpu->gpu.rank;
            if (dev) *dev = netDev;
            if (id) *id = netId;
            return ncclSuccess;
          }
        }
      }
    }
  }
  return ncclSuccess;
}
