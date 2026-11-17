/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "net.h"
#include "channel.h"
#include "transport.h"
#include "device.h"

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

// 对于指定的顶点，获取该顶点上指定类型 path 槽上对应指定 ID 的顶点的 path
static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int64_t id,
                            struct ncclTopoLinkList** path) {
  for (int i = 0; i < system->nodes[t].count; i++) {
    if (system->nodes[t].nodes[i].id == id) {
      *path = node->paths[t] + i;
      return ncclSuccess;
    }
  }
  WARN("Could not find node of type %d id %lx", t, id);
  return ncclInternalError;
}

NCCL_PARAM(NvbDisable, "NVB_DISABLE", 0);

static ncclResult_t ncclTopoPathReserve(struct ncclTopoLinkList* path, int required) {
  if (required <= path->capacity) return ncclSuccess;
  if (required > NCCL_TOPO_MAX_HOPS) {
    WARN("Path has %d hops, maximum is %d", required, NCCL_TOPO_MAX_HOPS);
    return ncclInternalError;
  }

  int capacity = std::max(required + 2, path->capacity * 2);
  capacity = std::min(capacity, NCCL_TOPO_MAX_HOPS);
  NCCLCHECK(ncclReallocQuiet(&path->list, path->capacity, capacity));
  path->capacity = capacity;
  return ncclSuccess;
}

/*
NCCL 路径缓存的填充器，它从 baseNode 出发做层序遍历，为全图每一个顶点 X 写上路径 X，baseNode，注意方向不是 baseNode → X，而是 X → baseNode。
ncclTopoComputePaths 里那 7 个循环，每个循环对该类的每个节点各调一次本函数，总调用次数 = 七类节点数之和，每次调用负责为全图所有人补上一格。
*/
static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
  /*
  BFS 遍历时 baseNode 顶点的 paths 对应类型的槽可能已经在之前分配内存了，
  注意这里只分配了 baseNode 顶点上的对应类型的槽，其他顶点上该类型的槽在 BFS 遍历到再分配。
  */
  if (baseNode->paths[baseNode->type] == NULL) {
    // 长度等于该类型的顶点数，动态分配内存
    NCCLCHECK(ncclCalloc(baseNode->paths + baseNode->type, system->nodes[baseNode->type].count));
    /*
    PATH_DIS 是未到达标记，不能用 count == 0 代替。因为 count == 0 已经被自己到自己占用了。
    这是四层状态（NULL / PATH_DIS / PATH_LOC 自己 / 有效）能区分开的关键。
    */
    for (int i = 0; i < system->nodes[baseNode->type].count; i++) baseNode->paths[baseNode->type][i].type = PATH_DIS;
  }

  // breadth-first search to set all paths to that node in the system
  struct ncclTopoNodeList nodeList;
  struct ncclTopoNodeList nextNodeList = {{0}, 0};
  nodeList.count = 1;
  nodeList.list[0] = baseNode;
  struct ncclTopoLinkList* basePath;
  NCCLCHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));  // 获取 baseNode 上对应自身的 path
  basePath->count = 0;
  basePath->bw = LOC_BW;  // 哨兵，后面每跳都会取最小值
  basePath->type = PATH_LOC;

  while (nodeList.count) {  // BFS 直到堆栈中没有可遍历的顶点为止
    nextNodeList.count = 0;  // nextNodeList 清空
    for (int n = 0; n < nodeList.count; n++) {  // 遍历当前层级的顶点
      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path;
      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));  // 获取当前顶点上对应 baseNode 的 path
      for (int l = 0; l < node->nlinks; l++) {  // 遍历当前顶点所有 links
        struct ncclTopoLink* link = node->links + l;
        struct ncclTopoNode* remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {  // 如果当前顶点的对端顶点的指定类型 path 槽为空，说明是新走到的顶点
          NCCLCHECK(ncclCalloc(remNode->paths + baseNode->type, system->nodes[baseNode->type].count));  // 对端顶点动态分配指定类型的 path 槽
          for (int i = 0; i < system->nodes[baseNode->type].count; i++)  // 初始化
            remNode->paths[baseNode->type][i].type = PATH_DIS;
        }
        struct ncclTopoLinkList* remPath;
        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));  // 获取对端顶点上对应 baseNode 的 path
        float bw = std::min(path->bw, link->bw);  // 更新 path 带宽为最小值

        /*
        DEV 顶点代表实际物理 gpu 或者 mlopart 分区，路径每穿过一个 DEV，物理含义就是数据必须落在那块物理 GPU 上，代价极高。
        继续扩展需要满足如下条件之一：
          1. 当前 node 不是 DEV（只关注 DEV 顶点）；
          2. 当前 baseNode 是 GPU 顶点，node 是该 GPU 顶点对应的 DEV 顶点；
          3. 当前 link 是 DEV -> GPU 的 LINK_LOC；
          4. 当前 link 是 DEV -> DEV 的 LINK_NVL 并且启用 NVB，并且路径长度不超过 2；
        */
        // Only allow path to go through a DEV if either
        // - the remNode is a GPU and the link type is PATH_LOC, or
        // - NVB is enabled and remNode is a DEV and link type is NVLink and the path isn't too long for NVB;
        // else, discard the path.
        int pathMaxLength = (baseNode->type == GPU) ? 2 : 1;
        ncclTopoNode* baseDevNode = (baseNode->type == GPU) ? baseNode->gpu.parent : baseNode;
        if (node != baseDevNode && node->type == DEV && (link->type != LINK_LOC || remNode->type != GPU) &&
            (ncclParamNvbDisable() || link->type != LINK_NVL || remNode->type != DEV || path->count > pathMaxLength)) {
          continue;
        }

        /*
        NET 顶点是网络端点，NIC -> LINK_NET -> NET 那条边表达的是端口归属，不是 PCIe 距离。
        若直接用 LINK_NET(10)，所有经过 NIC 的路径都会被标成 PATH_NET(10)，但 PATH_NET 的语义是跨网络可达，
        那是为 P2P 和 SHM 都不通专设的标记，用在这里会让同一台机器上 GPU 到本地 IB 卡被误判成跨网络。
        */
        // Start with path type = link type. PATH and LINK types are supposed to match.
        // Don't consider LINK_NET as we only care about the NIC->GPU path.
        int newType = link->type == LINK_NET ? LINK_LOC : link->type;  // 除了 LINK_NET 转成 LINK_LOC，其他 link 类型保持不变
        // Differentiate between one and multiple PCI switches
        if (node->type == PCI && remNode->type == PCI) newType = PATH_PXB;  // 两端都是 PCI 顶点，一律 PXB
        // Consider a path going through the CPU as PATH_PHB
        if (link->type == LINK_PCI && (node->type == CPU || link->remNode->type == CPU)) newType = PATH_PHB;  // 如果有一端是 CPU 顶点，跨 PCIe Host Bridge，PHB
        /*
        确定是 NVB：
          node->type == DEV：当前站在物理 GPU 上
          path->type == PATH_NVL：已有的那段是 NVLink（不是 LOC 也不是 PCI）
          newType == PATH_NVL：这一跳也是 NVLink
          path->count == pathMaxLength：长度正好到顶
        */
        // Set 1 hop NVLink as NVB.
        if (node->type == DEV && path->type == PATH_NVL && newType == PATH_NVL && path->count == pathMaxLength)
          newType = PATH_NVB;  // 
        newType = std::max(path->type, newType);  // 整条路径的好坏取决于最差的那一跳，越大越差，PATH_LOC=0 不会增加距离

        /*
        如果当前顶点到 baseNode 的路径距离比当前要到的远端节点更近，或者带宽更高，或者跳数更少，覆盖 remNode 的 path。
        用严格不等号，是为了让相等时保持原值。如果 remNode 未访问，则表示为 PATH_DIS，newType < remPath->type 天然成立。
        */
        // Update if better path type, OR same type with higher bw, OR same type/bw with strickly fewer hops.
        // Note: path->count +1 to account for the existing path + current candidate, see remPath->count update.
        if (newType < remPath->type || (newType == remPath->type && remPath->bw < bw) ||
            (newType == remPath->type && remPath->bw == bw && remPath->count > (path->count + 1))) {
          // Find reverse link
          struct ncclTopoLink* reverseLink = nullptr;
          for (int l = 0; l < remNode->nlinks; l++) {
            // 找 remNode 指向当前 node 的反向边，作为 remPath 的第一条 link
            if (remNode->links[l].remNode == node && remNode->links[l].type == link->type) {
              reverseLink = remNode->links + l;
              break;
            }
          }
          if (reverseLink == nullptr) {
            WARN("Failed to find reverse path from remNode %d/%lx nlinks %d to node %d/%lx", remNode->type, remNode->id,
                 remNode->nlinks, node->type, node->id);
            return ncclInternalError;
          }
          NCCLCHECK(ncclTopoPathReserve(remPath, path->count + 1));
          remPath->list[0] = reverseLink;
          // Copy the rest of the path
          // 将当前 node 到 x 的 path 中的 links 依次拷贝到 remPath 中
          for (int i = 0; i < path->count; i++) remPath->list[i + 1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->bw = bw;
          remPath->type = newType;

          // 判断当前 nextNodeList 中是否已经有 remNode 了，去重，否则将 remNode 加到 nextNodeList 中
          // Add to the list for the next iteration if not already in the list
          int i;
          for (i = 0; i < nextNodeList.count; i++) {
            if (nextNodeList.list[i] == remNode) break;
          }
          if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
        }
      }
    }
    // 遍历完一层 nodes，将记录的延伸到的新的一层 nodes 从 nextNodeList 拷贝到 nodeList
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  const int linesize = 1024;
  char line[linesize];
#ifdef ENABLE_TRACE
  INFO(NCCL_GRAPH, "Paths from %s/%lx-%lx :", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
       NCCL_TOPO_ID_LOCAL_ID(node->id));
#else
  snprintf(line, linesize, "%s/%lx-%lx :", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
           NCCL_TOPO_ID_LOCAL_ID(node->id));
  int offset = strlen(line);
#endif
  for (int t = 0; t < NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;
    for (int n = 0; n < system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i = 0; i < node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        snprintf(line + offset, linesize - offset, "--%s(%g)->%s/%lx-%lx", topoLinkTypeStr[link->type], link->bw,
                 topoNodeTypeStr[remNode->type], NCCL_TOPO_ID_SYSTEM_ID(remNode->id),
                 NCCL_TOPO_ID_LOCAL_ID(remNode->id));
        offset = strlen(line);
      }
      INFO(NCCL_GRAPH, "%s (%f)", line, node->paths[t][n].bw);
#else
      snprintf(line + offset, linesize - offset, "%s/%lx-%lx (%d/%.1f/%s) ", topoNodeTypeStr[t],
               NCCL_TOPO_ID_SYSTEM_ID(system->nodes[t].nodes[n].id),
               NCCL_TOPO_ID_LOCAL_ID(system->nodes[t].nodes[n].id), node->paths[t][n].count, node->paths[t][n].bw,
               topoPathTypeStr[node->paths[t][n].type]);
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system) {
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes + i);
  }
  for (int i = 0; i < system->nodes[NET].count; i++) {
    printNodePaths(system, system->nodes[NET].nodes + i);
  }
  for (int i = 0; i < system->nodes[GIN].count; i++) {
    printNodePaths(system, system->nodes[GIN].nodes + i);
  }
  for (int i = 0; i < system->nodes[RMA].count; i++) {
    printNodePaths(system, system->nodes[RMA].nodes + i);
  }
  return ncclSuccess;
}

// 输入 GPU 顶点下标，输出与之最近的 CPU 顶点下标，它是所有 P2P 不可用就改道经 CPU 逻辑的落点
ncclResult_t ncclGetLocalCpu(struct ncclTopoSystem* system, int gpu, int* cpu) {
  int localCpus[NCCL_TOPO_MAX_NODES], count;
  // 查询与当前 GPU 顶点的 path 带宽最大，路径类型最近的所有 CPU 顶点
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, CPU, localCpus, &count, NULL));
  if (count == 0) {
    WARN("Error : could not find CPU close to GPU %d", gpu);
    return ncclInternalError;
  }
  struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[gpu].paths[CPU];
  *cpu = localCpus[0];
  // 获取所有带宽和距离都相同的 paths 中跳数最小的 CPU 顶点索引
  for (int i = 1; i < count; i++) {
    if (paths[localCpus[i]].count < paths[*cpu].count) *cpu = localCpus[i];
  }
  return ncclSuccess;
}

static int mergePathType(int type0, int type1) {
  int max = std::max(type0, type1);
  int min = std::min(type0, type1);
  if (max == PATH_PHB && min == PATH_C2C) return PATH_P2C;  // 唯一特例，PATH_PHB + PATH_C2C → PATH_P2C
  else return max;  // 正常取最差质量的路径类型
}

// 将顶点 1 和 顶点 2 之间的 path 用采用顶点 x 中转的方式覆盖
static ncclResult_t addInterStep(struct ncclTopoSystem* system, int tx, int ix, int t1, int i1, int t2, int i2) {
  struct ncclTopoNode* cpuNode = system->nodes[tx].nodes + ix;  // 中转顶点
  struct ncclTopoNode* srcNode = system->nodes[t1].nodes + i1;  // 源顶点
  struct ncclTopoLinkList* path = srcNode->paths[t2] + i2;
  int count = srcNode->paths[tx][ix].count + cpuNode->paths[t2][i2].count;
  NCCLCHECK(ncclTopoPathReserve(path, count));

  int l = 0;
  // Node 1 -> CPU
  // 将源顶点到中转顶点的 path 拷贝到源顶点到目标顶点 path 的前半部分
  for (int i = 0; i < srcNode->paths[tx][ix].count; i++) path->list[l++] = srcNode->paths[tx][ix].list[i];
  // CPU -> Node 2
  // 将中转顶点到目标顶点的 path 拷贝到源顶点到目标顶点 path 的后半部分
  for (int i = 0; i < cpuNode->paths[t2][i2].count; i++) path->list[l++] = cpuNode->paths[t2][i2].list[i];

  // Update path characteristics
  // 合并两段 path 路径类型
  path->count = l;  // path 中的 links 长度
  path->type = mergePathType(srcNode->paths[tx][ix].type, cpuNode->paths[t2][i2].type);
  // 路径类型在这里不只是物理拓扑的描述，还承担策略标签的职责。靠这个标签跳过 NVS 顶点、定位第一个 DEV
  if (tx == GPU) path->type = PATH_PXN;
  path->bw = std::min(srcNode->paths[tx][ix].bw, cpuNode->paths[t2][i2].bw);  // path 带宽取最小值
  return ncclSuccess;
}

// 释放所有类型顶点上的 paths 路径信息内存
// Remove/free all paths
void ncclTopoRemovePaths(struct ncclTopoSystem* system) {
  for (int t1 = 0; t1 < NCCL_TOPO_NODE_TYPES; t1++) {
    for (int n = 0; n < system->nodes[t1].count; n++) {
      struct ncclTopoNode* node = system->nodes[t1].nodes + n;
      for (int t2 = 0; t2 < NCCL_TOPO_NODE_TYPES; t2++) {
        if (node->paths[t2]) {
          for (int p = 0; p < system->nodes[t2].count; p++) free(node->paths[t2][p].list);
          free(node->paths[t2]);
          node->paths[t2] = nullptr;
        }
      }
    }
  }
}

/*
旧数字  新常量	      效果
0	     PATH_LOC(0)	禁用
1      PATH_PIX(4)	同 PCIe switch
2	     PATH_PXB(5)	多 PCIe bridge
3      PATH_PHB(8)	过 CPU
4      PATH_SYS(9)	跨 NUMA
5      PATH_SYS(9)	钳位
*/
static const int levelsOldToNew[] = {PATH_LOC, PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS, PATH_SYS};
ncclResult_t ncclGetLevel(int* level, const char* disableEnv, const char* levelEnv) {
  if (*level == -1) { // 未初始化才执行
    int l = -1;
    if (disableEnv) {  // 先看 NCCL_P2P_DISABLE
      const char* str = ncclGetEnv(disableEnv);
      if (str) {
        int disable = strtol(str, NULL, 0);
        if (disable == 1) l = PATH_LOC;  // NCCL_P2P_DISABLE=1 → 层级锁死为 PATH_LOC(0)，等效于关闭跨 GPU 的 P2P
        if (l >= 0) INFO(NCCL_ALL, "%s set by environment to %d", disableEnv, disable);
      }
    }
    if (l == -1) {  // NCCL_P2P_DISABLE 没生效才看 NCCL_P2P_LEVEL
      const char* str = ncclGetEnv(levelEnv);
      if (str) {
        /*
        字符串形式(现代)：NCCL_P2P_LEVEL=<名字>，名字来自 topoPathTypeStr[]，循环上界 i <= PATH_SYS，
        故可识别 LOC/NVL/NVB/C2C/PIX/PXB/P2C/PXN/PHB/SYS(再往后的 NET/DIS 不在此循环范围)。
        */
        for (int i = 0; i <= PATH_SYS; i++) {  // 9
          if (strcmp(str, topoPathTypeStr[i]) == 0) {
            l = i;
            break;
          }
        }
        // 老式数字形式(兼容旧版)：NCCL_P2P_LEVEL=<0..5>，经 levelsOldToNew[] 映射，越界钳到 5
        // Old style numbering
        // levelsOldToNew to is an array with each index corresponding to the
        // "old level" int, and each value mapping to the correct value defined in topo.h
        // maxOldLevel is a quick check to handle out of bounds (based on the length of levelsOldToNew)
        if (l == -1 && str[0] >= '0' && str[0] <= '9') {
          int oldLevel = strtol(str, NULL, 0);
          const int maxOldLevel = sizeof(levelsOldToNew) / sizeof(int) - 1;  // 5
          if (oldLevel > maxOldLevel) oldLevel = maxOldLevel;  // 上界限定到 5
          l = levelsOldToNew[oldLevel];  // 经 levelsOldToNew[] 映射
        }
        if (l >= 0) INFO(NCCL_ALL, "%s set by environment to %s", levelEnv, topoPathTypeStr[l]);
      }
    }
    *level = l >= 0 ? l : -2;
  }
  return ncclSuccess;
}

/*
取值	      checkNvml	  日志	              *p2p
0（默认）    做校验	      打 WARN 或 INFO	    可能被打回 0，PATH_NVB 及以内直接报错
1	          做校验	    完全静默	            仍然会被打回 0
2	          不做校验	   无	                 保持 1，彻底忽略
*/
NCCL_PARAM(IgnoreDisabledP2p, "IGNORE_DISABLED_P2P", 0);

/*
这个 static 变量是整个 P2P_LEVEL 机制的核心，有三种状态:
值	  含义	                                    ncclGetUserP2pLevel的行为
-1	  未初始化(还没解析过环境变量)	                触发一次 ncclGetLevel 解析(惰性、只一次)
-2	  已初始化但用户没设(两个环境变量都没给有效值)	  不动 *level，调用方保留自己的默认值
>= 0	用户显式设了某个层级(是个 PATH_* 常量)	      把它写进 *level，覆盖调用方的值
*/
static int ncclTopoUserP2pLevel = -1; // Initially "uninitialized".  When initialized but unset, changes to -2.

// Gets the user-provided value of NCCL_P2P_LEVEL/NCCL_P2P_DISABLE.  If the user did not provide any, the value
// of the "level" argument is left unchanged.
ncclResult_t ncclGetUserP2pLevel(int* level) {
  // 如果还没初始化，从用户提供的 NCCL_P2P_DISABLE 和 NCCL_P2P_LEVEL 这两个环境变量获取 P2P_LEVEL 的值
  if (ncclTopoUserP2pLevel == -1) NCCLCHECK(ncclGetLevel(&ncclTopoUserP2pLevel, "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL"));
  // 如果已有缓存的 P2P_LEVEL 的值，写入 *level
  if (ncclTopoUserP2pLevel != -2) *level = ncclTopoUserP2pLevel;
  // 否则，如果两个环境变量都没给有效值，不动 *level，调用方保留自己的默认值
  return ncclSuccess;
}

/*
判断两个 GPU 之间是否支持 p2p（p2p_level、NVML 硬件限制且可 NVB 中转），以及是否支持 CUDA P2P 直接 P2P 连接

参数	                回答的问题	                        判定依据	                          必填
*p2p	               该不该用 P2P	                       拓扑距离 ≤ p2pLevel（+ NVML 实测）	  ✅ 必填
*cudaP2p	           能不能用 CUDA P2P	                 NVML 设备对状态 / 同 host 假设	      可传 NULL
*read	               是否支持 P2P Read	                 仅 Ampere + NVLink	                可传 NULL
*intermediateRank	   若经中间 GPU 中转，那个 rank 是多少	  路径形状 = count == 4	              可传 NULL

p2p 和 cudaP2p 是正交的两个维度，四种组合都合法：
            cudaP2p = 0	                               cudaP2p = 1
  p2p = 0	  CUDA 不支持 + 距离太远 → 彻底不能用	           CUDA 支持但用户禁用 / 距离超阈值 → 主动不用
  p2p = 1	  距离够近但 NVML 报 P2P 坏了 → 硬件故障态	     正常可用
*/
// Tests two ranks for CUDA P2P connectivity.
// *cudaP2p returns 1 if CUDA P2P between the ranks is supported.
// *p2p returns 1 only if the distance between the ranks is no greater than NCCL_P2P_LEVEL.
// *isCrossClique returns 1 when P2P is classified using MNNVL fabric metadata.
// The connection may go through an intermediate rank.
ncclResult_t ncclTopoCheckP2p(struct ncclComm* comm, struct ncclTopoSystem* system, int rank1, int rank2, int* p2p,
                              int* read, int* intermediateRank, int* cudaP2p, int* isCrossClique) {
  int mnnvl = 0;
  struct ncclPeerInfo* info1 = NULL;
  struct ncclPeerInfo* info2 = NULL;
  *p2p = 0;
  if (read) *read = 0;
  if (intermediateRank) *intermediateRank = -1;
  if (cudaP2p) *cudaP2p = 0;
  if (isCrossClique) *isCrossClique = 0;

  // Rule out different nodes / isolated containers
  if (comm) {
    info1 = comm->peerInfo + rank1;
    info2 = comm->peerInfo + rank2;
    if (info1->hostHash != info2->hostHash) {  // 不属于同一节点
      if (comm->MNNVL) {  // 前序探测确定可以启用 MNNVL
        NCCLCHECK(ncclTopoCheckMNNVL(comm, info1, info2, &mnnvl));  // 检查两个 ranks 是否位于相同 fabric cluster 和 clique
        // 强制开启 P2P 和 CUDA P2P
        if (mnnvl < 0) {  // -1：相同 cluster 不同 clique 但允许跨 clique
          // Cross-clique connectivity comes from communicator-wide fabric metadata because the peer is absent from
          // this rank's topology.
          int p2pLevel = PATH_NVL;
          NCCLCHECK(ncclGetUserP2pLevel(&p2pLevel));
          *p2p = p2pLevel >= PATH_NVL;
          if (cudaP2p) *cudaP2p = 1;
          if (isCrossClique) *isCrossClique = 1;
          return ncclSuccess;
        }
        if (!mnnvl) return ncclSuccess;  // 不开启 P2P 和 CUDA P2P
      } else {  // 前序 MNNVL 条件都不满足
        return ncclSuccess;
      }
    } else if (info1->shmDev != info2->shmDev) {
      /*
      hostHash 相同只说明是同一台物理机，但两个容器可能有独立的 /dev/shm。
      shmDev 是这个 shm 实例的设备号或标识，不同就意味着这两组 rank 之间连共享内存都不能用，P2P 更谈不上。
      */
      return ncclSuccess;
    }
  }

  // 在拓扑图中查找指定 rank1 和 rank2 的 GPU 顶点的索引
  // Get GPUs from topology
  int g1, g2;
  NCCLCHECK(ncclTopoRankToIndex(system, rank1, &g1, /*showWarn=*/true));
  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes + g1;
  if (ncclTopoRankToIndex(system, rank2, &g2, /*showWarn=*/false) == ncclInternalError) {
    // GPU not found, we can't use p2p.
    return ncclSuccess;
  }

  int intermediateIndex = -1;
  // Set intermediate GPU rank, if routing through an intermediate GPU.
  struct ncclTopoLinkList* path = gpu1->paths[GPU] + g2;  // 找到 gpu1 到 gpu2 的 path
  if (path->count == 4) {  // 如果 path 中的 links 数刚好是 4，索引位 1 的 link 指向 intermediate DEV 顶点
    // Intermediate goes through DEV, not GPU.
    // path is GPU1 - DEV1 - DEV2 - DEV3 - GPU2, so the intermediate DEV is located at path->list[1]->remNode
    struct ncclTopoNode* intermediateNode = path->list[1]->remNode;
    if (intermediateNode->type == DEV) {
      int interRank;
      NCCLCHECK(ncclTopoDevToRank(system, NCCL_TOPO_ID_SYSTEM_ID(intermediateNode->id), intermediateNode->dev.dev,
                                  /*warn=*/true, &interRank));  // 找到与中间 DEV 顶点对应的 GPU rank
      NCCLCHECK(ncclTopoRankToIndex(system, interRank, &intermediateIndex, true));  // 找到该 rank 在 GPU 顶点数组中的索引
      if (intermediateRank) *intermediateRank = interRank;
    }
  }

  // By default don't use P2P across CPU Host Bridges and further apart
  int p2pLevel = PATH_PXB;  // 默认使用 P2P 不跨越 CPU Host Bridges 及更远的距离

  int arch, vendor, model;
  NCCLCHECK(ncclTopoCpuType(system, &arch, &vendor, &model));  // 获取 CPU 的架构、厂商、微架构
  // Allow P2P between pairs of GPU devices on AMD systems
  if ((arch == NCCL_TOPO_CPU_ARCH_X86 && vendor == NCCL_TOPO_CPU_VENDOR_AMD) && system->nodes[DEV].count <= 2)  // 总的物理 GPU 不超过 2 个
  // 对于 AMD x86 平台的 AMD Infinity Fabric 的链路足以支撑 GPU Direct P2P，数据不需要落到 CPU 缓存
    p2pLevel = PATH_SYS;

  // User override
  NCCLCHECK(ncclGetUserP2pLevel(&p2pLevel));  // 用户指定的 P2P LEVEL 覆盖阈值

  // Compute the PCI distance and compare with the p2pLevel.
  if (path->type <= p2pLevel) *p2p = 1;  // 两个 GPU 的路径距离小于阈值，则可以用 p2

  /*
  使用父指针比较来处理每个 GPU 多个 rank 的情况，当多个 rank 共享一个设备时，
  不同的拓扑索引（g1 != g2）可能指向同一物理 GPU，对于同一设备，跳过 NVML P2P 验证。
  */
  // Use parent pointer comparison to handle multi-rank-per-GPU case:
  // Different topology indices (g1 != g2) may point to the same physical GPU
  // when multiple ranks share one device. Skip NVML P2P validation for same device.
  bool checkNvml =
    (ncclParamIgnoreDisabledP2p() != 2 &&  // 0 或 1 要做校验
     system->nodes[GPU].nodes[g1].gpu.parent != system->nodes[GPU].nodes[g2].gpu.parent &&  // 两个 rank 不是共享相同物理 GPU（或分区）
     (comm == NULL || (info1->hostHash == comm->peerInfo[comm->rank].hostHash && info1->hostHash == info2->hostHash)));  // 三个 rank 必须都在本机
  if (*p2p == 1) {  // 如果已经关闭 P2P，则不需要再校验
    if (checkNvml) {  // 满足上述的校验条件
      int indexes[3] = {-1, -1, -1};
      int verticeN = 0;
      NCCLCHECK(ncclNvmlEnsureInitialized());  // 确保 NVML 初始化

      // 无中间 GPU 和有中间 GPU 两种情况，获取 GPU 的 NVML 索引
      indexes[verticeN++] = system->nodes[GPU].nodes[g1].gpu.dev;
      if (intermediateIndex != -1) indexes[verticeN++] = system->nodes[GPU].nodes[intermediateIndex].gpu.dev;
      indexes[verticeN++] = system->nodes[GPU].nodes[g2].gpu.dev;

      for (int i = 1; i < verticeN; i++) {  // 采用 NVML 分段检验各条物理直连是否支持 P2P
        nvmlGpuP2PStatus_t status;
        // 分别查询两个 GPU Device 之间是否支持 P2P 读/写能力
        status = ncclNvmlDevicePairs[indexes[i - 1]][indexes[i - 0]].p2pStatusRead;
        bool good = status == NVML_P2P_STATUS_OK;
        status = ncclNvmlDevicePairs[indexes[i - 1]][indexes[i - 0]].p2pStatusWrite;
        good &= status == NVML_P2P_STATUS_OK;
        if (!good) {  // 不支持 P2P 
          if (!ncclParamIgnoreDisabledP2p()) {  // NCCL_IGNORE_DISABLED_P2P 为 0，日志报错
            if (path->type <= PATH_NVB) {  // NVLink 直连却不通，硬件故障
              WARN("P2P is disabled between NVLINK connected GPUs %d and %d. This should not be the case given their "
                   "connectivity, and is probably due to a hardware issue. If you still want to proceed, you can set "
                   "NCCL_IGNORE_DISABLED_P2P=1.",
                   indexes[i - 1], indexes[i - 0]);
              return ncclUnhandledCudaError;
            } else if (path->type < PATH_SYS) {  // PCIe 路径不通，可容忍
              INFO(NCCL_INIT,
                   "P2P is disabled between connected GPUs %d and %d. You can repress this message with "
                   "NCCL_IGNORE_DISABLED_P2P=1.",
                   indexes[i - 1], indexes[i - 0]);
            }
          }
          *p2p = 0;
        }
      }
    }
  }

  /*
  两个 GPU 之间的路径是 NVLink，并且 CC 为 80（Ampere）启用 P2P Read 模式。
  Hopper 和 Blackwell 都不满足。 这是个刻意的保守选择，更高架构有 TMA / DSMEM / NVLink 5.0 等新机制，
  read 模式的收益模型变了，NCCL 不默认启用（但用户可以用环境变量强开）。
  */
  if (path->type == PATH_NVL) {
    struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes + g2;
    // Enable P2P Read for Ampere/NVLink only
    if (read && (gpu1->gpu.cudaCompCap == gpu2->gpu.cudaCompCap) && (gpu1->gpu.cudaCompCap == 80)) *read = 1;
  }

  /*
  p2p：这条通信路径能不能走 P2P（含中转）
  cudaP2p：这两块设备之间有没有直连 CUDA P2P 能力
  */
  if (cudaP2p) {
    if (checkNvml) {  // 直接读本机 NVML 的 N×N 设备对状态矩阵，和上一段的 *p2p 校验用的是同一张表
      int n1, n2;
      n1 = system->nodes[GPU].nodes[g1].gpu.dev;
      n2 = system->nodes[GPU].nodes[g2].gpu.dev;
      *cudaP2p = (ncclNvmlDevicePairs[n1][n2].p2pStatusRead == NVML_P2P_STATUS_OK &&
                  ncclNvmlDevicePairs[n1][n2].p2pStatusWrite == NVML_P2P_STATUS_OK);
    } else {  // 如果各队列通过 MNNVL 连接，或者位于同一主机上，乐观假设存在点对点（P2P）连接。
      // We assume P2P connectivity in case the ranks are connected using MNNVL or are on the same host.
      *cudaP2p = (mnnvl || comm == NULL || info1->hostHash == info2->hostHash);
    }
  }

  return ncclSuccess;
}

// 检查两个 ranks 是否在相同的 fabric cluster 或 clique
// MNNVL: Check whether peers are in the same fabric cluster and clique
ncclResult_t ncclTopoCheckMNNVL(struct ncclComm* comm, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2,
                                int* ret) {
  *ret = 0;

  nvmlGpuFabricInfoV_t* fabricInfo1 = &info1->fabricInfo;
  nvmlGpuFabricInfoV_t* fabricInfo2 = &info2->fabricInfo;
  // A zero UUID means we don't have MNNVL fabric info
  unsigned long uuid0 = 0;
  unsigned long uuid1 = 0;
  memcpy(&uuid0, fabricInfo2->clusterUuid, sizeof(uuid0));
  memcpy(&uuid1, fabricInfo2->clusterUuid + sizeof(uuid0), sizeof(uuid1));
  if ((uuid0 | uuid1) == 0) return ncclSuccess;  // 0：无 fabric info / UUID 不匹配 / 跨 clique 未开开关
  // 两个 ranks 位于相同 fabric cluster 和 clique，或者相同 fabric cluster 不同 clique 但 启用跨 clique
  // Same UUID required. Within same UUID: either same clique OR cross-clique enabled
  if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) &&
      (comm->p2pCrossClique || fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
    TRACE(NCCL_NET, "MNNVL rank %d matching peer %d 0x%lx UUID %lx.%lx cliqueId 0x%x/0x%x crossClique %d", info1->rank,
          info2->rank, info2->busId, uuid0, uuid1, fabricInfo1->cliqueId, fabricInfo2->cliqueId, comm->p2pCrossClique);
    // Return -1 to distinguish cross-clique peers (different clique but same UUID).
    /*
    -1：跨 clique，且 NCCL_MNNVL_CROSS_CLIQUE=1
    1：同 clique 的 MNNVL
    */
    *ret = (comm->p2pCrossClique && fabricInfo1->cliqueId != fabricInfo2->cliqueId) ? -1 : 1;
  }
  return ncclSuccess;
}

// NCCL_NET_GDR_READ=-2(默认) 走 pre-Ampere 检查，-1 同 -2，0 直接禁用发送方向 GDR，> 0	跳过 pre-Ampere 检查，强制启用。
NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
int ncclTopoUserGdrLevel = -1;
const char* ncclTopoGdrModeStr[ncclTopoGdrModeNum] = {"Disabled", "Default", "PCI"};

// NCCL_NET_GDR_C2C=1 指定 GDR_LEVEL = PATH_P2C(6)
// On C2C platforms use GDRDMA on NICs which are connected to the CPUs
// NCCL_NET_GDR_MLOPART=0 默认禁用 MLOPart 分区 GPU 的 GDR
NCCL_PARAM(NetGdrC2c, "NET_GDR_C2C", 1);

// 检查指定 NET 与对应 GPU rank 之间是否支持 GDR
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* system, int rank, int64_t netId, int read,
                              enum ncclTopoGdrMode* gdrMode) {
  *gdrMode = ncclTopoGdrModeDisable;  // 默认禁用

  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));  // 获取指定 NET 顶点
  struct ncclTopoNode* net = system->nodes[NET].nodes + n;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));  // 获取指定 rank 的 GPU 顶点
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
#ifdef ENABLE_TRACE
  char gpuNetMsg[1024] = "";
  snprintf(gpuNetMsg, sizeof(gpuNetMsg), "GPU/%ld-%ld (rank %d) - NET/%ld-%ld (", NCCL_TOPO_ID_SYSTEM_ID(gpu->id),
           NCCL_TOPO_ID_LOCAL_ID(gpu->id), rank, NCCL_TOPO_ID_SYSTEM_ID(net->id), NCCL_TOPO_ID_LOCAL_ID(net->id));
#endif

  // 确认 NIC 和 GPU 均支持 GDR
  if (net->net.gdrSupport == 0) return ncclSuccess;
  if (gpu->gpu.gdrSupport == 0) return ncclSuccess;

  // mlopart 分区并且禁用
  if (read) {  // read 指的是「网卡读 GPU 显存」，也就是发送方向（发送时网卡从 GPU 显存读数据）
    // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();  // 是否启用 GDR Read
    if (gdrReadParam == 0) return ncclSuccess;  // 0 直接禁用
    // Disable GDR Reads pre-Ampere when we have other PCI flows
    if (gdrReadParam < 0 && gpu->gpu.cudaCompCap < 80) {  // 小于 0，pre-Ampere 检查
      /*
      Ampere（compute capability 80）之前，GPU 的 PCIe BAR1 窗口远小于显存容量，无法把整个显存映射出去。
      当既没有 NVLink、又要走 PCIe 时，读方向的 GDR 会因为 BAR1 太小而严重退化甚至失败。
      于是逻辑是：pre-Ampere + 无 NVLink → 保守关闭。
      */
      int nvlink = 0;
      // 因为无法知道是否存在其他 communicator 在争抢同一块 GPU 的 PCIe 带宽，单卡场景下保守地保持本地处理。
      // Since we don't know whether there are other communicators,
      // it's better to keep things local if we have a single GPU.
      if (system->nodes[GPU].count == 1) nvlink = 1;
      for (int i = 0; i < system->nodes[GPU].count; i++) {
        if (i == g) continue;
        // 这里的逻辑是如果 GPU 之间有 NVLink 就不会走 PCIe，所以 GPU 到网卡的 PCIe GDR Read 可以被独占。
        if (gpu->paths[GPU][i].type == PATH_NVL) {  // 遍历 gpu->paths[GPU][i] 找 PATH_NVL，判定有没有任何一块 GPU 与我 NVLink 直连
          nvlink = 1;
          break;
        }
      }
      if (!nvlink) return ncclSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = PATH_PXB;
  if (ncclParamNetGdrC2c()) {
    // MloPart over C2C requires cuMem GDR support with CUDA 13.4+ (must be available communicator wide)
    bool gdrSupport = system->cuMemGdrSupport && system->minDriverVersion >= 13040;
    if (gpu->gpu.mloPart == NCCL_TOPO_UNDEF || gdrSupport) netGdrLevel = PATH_P2C;  // 根据是否启用 P2C GDR 指定 GDR_LEVEL 默认值为 PATH_P2C 还是 PATH_PXB
  }
  NCCLCHECK(ncclGetLevel(&ncclTopoUserGdrLevel, NULL, "NCCL_NET_GDR_LEVEL"));  // 从环境变量 NCCL_NET_GDR_LEVEL 获取用户指定的 GDR_LEVEL
  if (ncclTopoUserGdrLevel != -2) netGdrLevel = ncclTopoUserGdrLevel;  // 用户指定覆盖默认
  int distance = gpu->paths[NET][n].type;
  if (distance == PATH_PXN) {  // 识别 PXN 路径
    // 把「距离」换成真正执行收发的那块 relay GPU 到网卡的距离，因为 PXN 下 DMA 是由 relay 发起的，判定依据自然也应是 relay 的通路。
    // In case of PXN, use the intermediate GPU distance instead
    int proxyRank;
    NCCLCHECK(ncclTopoGetIntermediateRank(system, gpu->gpu.rank, netId, &proxyRank));  // 找到中间 rank
    NCCLCHECK(ncclTopoRankToIndex(system, proxyRank, &g, /*showWarn=*/true));  // 找到中间 rank 对应的 GPU 顶点索引
    gpu = system->nodes[GPU].nodes + g;
    distance = gpu->paths[NET][n].type;  // 获取中间 GPU 顶点到 NET 顶点的路径类型
#ifdef ENABLE_TRACE
    snprintf(gpuNetMsg + strlen(gpuNetMsg), sizeof(gpuNetMsg) - strlen(gpuNetMsg), " using PXN via GPU/%ld-%ld, ",
             NCCL_TOPO_ID_SYSTEM_ID(gpu->id), NCCL_TOPO_ID_LOCAL_ID(gpu->id));
#endif
  }

  if (distance > netGdrLevel) {  // 如果路径类型大于 GDR_LEVEL 阈值，则不可用 GDR
#ifdef ENABLE_TRACE
    snprintf(gpuNetMsg + strlen(gpuNetMsg), sizeof(gpuNetMsg) - strlen(gpuNetMsg), "distance %d > %d)", distance,
             netGdrLevel);
    TRACE(NCCL_GRAPH | NCCL_NET, "GPU Direct RDMA Disabled for %s", gpuNetMsg);
#endif
    return ncclSuccess;
  }

  // 剩下都是可用 GDR 的路径类型，但要区分 GPU 与 NIC 间是非 P2C 的 C2C 系统
  // Force PCIe mapping if path goes through PCI on a C2C system
  int c;
  NCCLCHECK(ncclGetLocalCpu(system, g, &c));  // 获取与 GPU g 最近的 CPU 顶点下标
  // GPU→CPU 是 PATH_C2C 且 距离不是 PATH_P2C（C2C 系统但走的是非 C2C 通路），需要 PCIe 映射 + flush，输出 ncclTopoGdrModePci
  if (gpu->paths[CPU][c].type == PATH_C2C && distance != PATH_P2C) *gdrMode = ncclTopoGdrModePci;
  else *gdrMode = ncclTopoGdrModeDefault;  // 其他常规 GDR

#ifdef ENABLE_TRACE
  snprintf(gpuNetMsg + strlen(gpuNetMsg), sizeof(gpuNetMsg) - strlen(gpuNetMsg), "distance %d <= %d, read %d, mode %s)",
           distance, netGdrLevel, read, ncclTopoGdrModeStr[*gdrMode]);
  TRACE(NCCL_GRAPH | NCCL_NET, "GPU Direct RDMA Enabled for %s", gpuNetMsg);
#endif
  return ncclSuccess;
}

// Set to 0 to disable the flush on Hopper when using GDR
NCCL_PARAM(NetForceFlush, "NET_FORCE_FLUSH", 0);

// Based on the system topology, determine whether an explicit iflush is needed on the GDR recv path.
ncclResult_t ncclTopoNeedFlush(struct ncclComm* comm, int64_t netId, int netDev, int rank,
                               enum ncclTopoFlushType* flush) {
  *flush = ncclTopoFlushAlways;
  ncclNetProperties_t props;
  NCCLCHECK(comm->ncclNet->getProperties(netDev, &props));
  if (props.forceFlush == 1 || ncclParamNetForceFlush()) {
    TRACE(NCCL_NET, "NET/%s/%d: flush type = Always (forced)", comm->ncclNet->name, netDev);
    return ncclSuccess;
  }
  int g;
  struct ncclTopoSystem* system = comm->topo;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
  // Flush is required on Ampere and earlier
  if (gpu->gpu.cudaCompCap >= 90) {
    *flush = ncclTopoFlushNone;
    // DataDirect NIC require a flush operation because control path is using C2C and data path is using PCIe.
    int c, n;
    NCCLCHECK(ncclGetLocalCpu(system, g, &c));
    NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
    if (gpu->paths[NET][n].type <= PATH_PXB && gpu->paths[CPU][c].type == PATH_C2C) {
      *flush = ncclTopoFlushC2c;
    }
  }
  return ncclSuccess;
}

NCCL_PARAM(NetDisableIntra, "NET_DISABLE_INTRA", 0);

// Check whether going through the network would be faster than going through P2P/SHM.
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int rank1, int rank2, int* net) {
  if (ncclParamNetDisableIntra() == 1) {
    *net = 0;
    return ncclSuccess;
  }
  *net = 1;
  // First check the current GPU-to-GPU speed.
  int g1, g2;
  if (ncclTopoRankToIndex(system, rank1, &g1, /*showWarn=*/false) != ncclSuccess ||
      ncclTopoRankToIndex(system, rank2, &g2, /*showWarn=*/false) != ncclSuccess) {
    return ncclSuccess;
  }

  struct ncclTopoNode* gpu1 = system->nodes[GPU].nodes + g1;
  struct ncclTopoNode* gpu2 = system->nodes[GPU].nodes + g2;
  float speed = gpu1->paths[GPU][g2].bw;

  // Now check the speed each GPU can access the network through PXB or better
  float netSpeed1 = 0, netSpeed2 = 0;
  for (int n = 0; n < system->nodes[NET].count; n++) {
    struct ncclTopoLinkList* path = gpu1->paths[NET] + n;
    if (path->type <= PATH_PXB && path->bw > netSpeed1) netSpeed1 = path->bw;
    path = gpu2->paths[NET] + n;
    if (path->type <= PATH_PXB && path->bw > netSpeed2) netSpeed2 = path->bw;
  }

  if (netSpeed1 > speed && netSpeed2 > speed) return ncclSuccess;
  *net = 0;
  return ncclSuccess;
}

// 找到指定 rank 和 NET 顶点 ID 之间的中间 rank
ncclResult_t ncclTopoGetIntermediateRank(struct ncclTopoSystem* system, int rank, int64_t netId,
                                         int* intermediateRank) {
  // Get GPU and NET
  int n, g;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &n));
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g, /*showWarn=*/true));
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
  struct ncclTopoLinkList* path = gpu->paths[NET] + n;
  if (path->type == PATH_PXN) {
    // 找到中间 DEV 顶点
    // PXN path follows GPU-DEV-NVS-..., start from the first NVS node and find the first DEV in the path
    int i = 1;
    while (i < path->count && path->list[i]->remNode->type == NVS) i++;
    struct ncclTopoNode* node = path->list[i]->remNode;

    // 找到该 DEV 顶点的任意 GPU 顶点，获取中间 rank
    // Select the first GPU on the device found to be the PXN intermediate rank
    if (node->type == DEV) {
      for (int i = 0; i < node->nlinks; i++) {
        if (node->links[i].remNode->type == GPU) {
          node = node->links[i].remNode;
          break;
        }
      }
    }
    if (node->type != GPU) {
      WARN("Could not find intermediate GPU between GPU rank %d and NIC %lx", rank, netId);
      return ncclInternalError;
    }
    NCCLCHECK(ncclTopoDevToRank(system, NCCL_TOPO_ID_SYSTEM_ID(node->id), node->gpu.dev, /*warn=*/true,
                                intermediateRank));
  } else {
    *intermediateRank = rank;
  }
  return ncclSuccess;
}

// NCCL_PXN_DISABLE 是否禁用 PXN
NCCL_PARAM(PxnDisable, "PXN_DISABLE", 0);

/*
  remote proxy：PXN 的语义是 GPU g 要发数据给远端的 rank R，但用的是兄弟 GPU localGpuIndex 旁边那块网卡。
而 NCCL 的网络发送是由 proxy 执行的。所以「用别人的网卡」在实现上就是「让 proxy 替另一个 rank 收发」，这就是 remote proxy。
  v5 插件引入了非阻塞的 connect/accept，proxy 可以「发起连接 → 返回 → 处理别的请求 → 回来轮询」，才能安全地做多路代理。
*/
// Net v4 plugins don't have non-blocking connect/accept. We can't therefore use
// remote proxies without risking deadlocks
int ncclPxnDisable(struct ncclComm* comm) {
#if defined(NCCL_OS_LINUX)
  static int pxnDisable = -1;
  if (pxnDisable == -1) {
    if (comm && comm->ncclNetVer == 4) {
      INFO(NCCL_INIT, "PXN Disabled as plugin is v4");
      pxnDisable = 1;
    } else {
      pxnDisable = ncclParamPxnDisable();
    }
  }
  return pxnDisable;
#else
  return 1;
#endif
}

ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks) {
  struct ncclTopoSystem* system = comm->topo;
  *nranks = 0;
  *intermediateRanks = NULL;
  if (system->inter == 0) return ncclSuccess;

  int nr = 0;
  int* ranks = NULL;
  for (int rank = 0; rank < comm->nRanks; rank++) {
    int64_t netId;
    int proxyRank;
    NCCLCHECK(ncclTopoGetNetDev(comm, comm->rank, NULL, 0, rank, &netId, NULL, &proxyRank));
    if (proxyRank == comm->rank) continue;
    enum ncclTopoGdrMode useGdr;
    NCCLCHECK(ncclTopoCheckGdr(comm->topo, comm->rank, netId, 1, &useGdr));
    if (useGdr == ncclTopoGdrModeDisable) continue;
    int found = 0;
    for (int r = 0; r < nr; r++) {
      if (ranks[r] == proxyRank) found = 1;
    }
    if (!found) {
      NCCLCHECK(ncclRealloc(&ranks, nr, nr + 1));
      ranks[nr++] = proxyRank;
    }
  }
  *nranks = nr;
  *intermediateRanks = ranks;
  return ncclSuccess;
}

// NCCL_PXN_C2C=1 PXN 中转的 GPU 到网卡的路径类型优于 P2C 就可以作为 PXN 中转
NCCL_PARAM(PxnC2c, "PXN_C2C", 1);

/*
几何路径计算 + 业务可用性改写，先用 BFS 算出图上所有顶点对的几何最短路，
然后按四条业务规则（P2P 是否允许、SHM 是否可用、C2C 平台特例、GDR 是否可用）直接改写这些路径缓存，
让下游（建图搜索、带宽估算）读到的是实际能走的路而不是物理上最近的路。
*/
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) {
  // Precompute paths between GPUs/NICs.

  // 释放所有类型顶点上的 paths 路径信息内存
  // Remove everything in case we're re-computing
  ncclTopoRemovePaths(system);

  // 下面分别遍历 CPU/DEV/GPU/NET/GIN/RMA/NVS 等类型的顶点，建立所有图中所有顶点到这些目标顶点的 path

  // Set direct paths to CPUs. We need them in many cases.
  for (int c = 0; c < system->nodes[CPU].count; c++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes + c, system));
  }

  // Set direct paths to DEVs, needed in the graph search.
  for (int d = 0; d < system->nodes[DEV].count; d++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[DEV].nodes + d, system));
  }

  // Set direct paths to GPUs.
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes + g, system));
  }

  // Set direct paths to NICs.
  for (int n = 0; n < system->nodes[NET].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes + n, system));
  }

  // Set direct paths to GIN devices.
  for (int n = 0; n < system->nodes[GIN].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GIN].nodes + n, system));
  }

  // Set direct paths to RMA devices.
  for (int n = 0; n < system->nodes[RMA].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[RMA].nodes + n, system));
  }

  // Set direct paths to NVSwitches.
  for (int n = 0; n < system->nodes[NVS].count; n++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[NVS].nodes + n, system));
  }

  /*
  ncclTopoComputePaths 的路径改写分四段：
    顺序	  段	        做什么
    A	     GPU↔GPU	   P2P 不通 → 改道经 CPU
    B	     GPU↔GPU	   P2P 和 SHM 都不通 → 标 PATH_NET
    C	     GPU↔NET	   C2C + PHB → 降级成 PATH_P2C
    D	     GPU↔NET	   PXN 建链 + GDR 不可用改道经 CPU
  */

  // 针对 GPUs 之间存在不想或不能使用 GPU P2P 的情况进行更新
  // Update path for GPUs when we don't want to / can't use GPU Direct P2P
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    for (int p = 0; p < system->nodes[GPU].count; p++) {  // 对于每个 GPU p 到 GPU g
      int p2p;
      // 判断两个 GPU 之间是否支持 p2p（p2p_level、NVML 硬件限制且可 NVB 中转）
      NCCLCHECK(ncclTopoCheckP2p(comm, system, system->nodes[GPU].nodes[p].gpu.rank,
                                 system->nodes[GPU].nodes[g].gpu.rank, &p2p, NULL, NULL, NULL));
      if (p2p == 0) {
        // 将所有流量都通过 CPU 处理
        int cpu;
        NCCLCHECK(ncclGetLocalCpu(system, g, &cpu));  // 获取离 GPU g 带宽最大、路径类型最差、跳数最少的 CPU 顶点索引
        NCCLCHECK(addInterStep(system, CPU, cpu, GPU, p, GPU, g));  // 用 CPU 中转的 path 覆盖原本的 GPU p 到 GPU g 的 path
      }
    }

    if (comm == NULL) continue;
    // Remove GPUs we can't (or don't want to) communicate with through P2P or SHM
    struct ncclPeerInfo* dstInfo = comm->peerInfo + system->nodes[GPU].nodes[g].gpu.rank;  // 获取目标 GPU rank 信息
    for (int p = 0; p < system->nodes[GPU].count; p++) {
      if (p == g) continue;
      struct ncclPeerInfo* srcInfo = comm->peerInfo + system->nodes[GPU].nodes[p].gpu.rank;  // 获取源 GPU rank 信息
      int p2p;
      NCCLCHECK(ncclTransports[TRANSPORT_P2P]->canConnect(&p2p, comm, NULL, srcInfo, dstInfo));  // 判断 P2P 行不行
      if (p2p == 0) {  // P2P 不行再判断 SHM 行不行
        int shm;
        NCCLCHECK(ncclTransports[TRANSPORT_SHM]->canConnect(&shm, comm, NULL, srcInfo, dstInfo));
        if (shm == 0) {
          // P2P 和 SHM 都不行，标记这两个 GPU peer 的 p -> g 方向路径类型为 PATH_NET
          // Mark this peer as inaccessible. We'll trim it later.
          system->nodes[GPU].nodes[p].paths[GPU][g].type = PATH_NET;
        }
      }
    }
  }
  /*
  修正 BFS 计算出的 GPU↔NET 路径类型，当 GPU 通过 C2C 连到 CPU、网卡通过 PCIe 也连到同一个 CPU 时，
  朴素 BFS 会把这条路算成 PATH_PHB(8)（普通 PCIe 过 host bridge），完全丢失了 C2C 高带宽的信息。
  这段把它改成 PATH_P2C(6)，一个专门为「C2C + PCIe 组合」定义的中间级别。
  */
  // update the GPU -> NIC path in the case of C2C + PHB
  // P2C is only set when the NET is the closest to the GPU. Otherwise PXN connections should be preferred
  for (int g = 0; g < system->nodes[GPU].count; g++) {  // 遍历所有 GPU 顶点
    struct ncclTopoNode* gpuNode = system->nodes[GPU].nodes + g;
    int c = 1, localNetCount = 0, localNet[NCCL_TOPO_MAX_NODES];
    NCCLCHECK(ncclGetLocalCpu(system, g, &c));  // 获取与该 GPU 亲和的 CPU 顶点索引
    if (c == -1) continue;
    // 获取所有与该 GPU 最近的 NET 顶点
    NCCLCHECK(ncclTopoGetLocal(system, GPU, g, NET, localNet, &localNetCount, /*pathType=*/NULL));
    for (int l = 0; l < localNetCount; l++) {
      int n = localNet[l];
      struct ncclTopoNode* netNode = system->nodes[NET].nodes + n;
      // 如果将 GPU 顶点到亲和的 CPU 顶点的路径类型与 NET 顶点到 CPU 顶点的路径类型融合后为 P2C 路径类型
      if (mergePathType(gpuNode->paths[CPU][c].type, netNode->paths[CPU][c].type) == PATH_P2C) {
        // 升级 GPU 顶点和 NET 顶点相互的路径类型，原本如果是 PATH_PXB(5) ，就不用动
        gpuNode->paths[NET][n].type = std::min(PATH_P2C, gpuNode->paths[NET][n].type);
        netNode->paths[GPU][g].type = std::min(PATH_P2C, netNode->paths[GPU][g].type);
      }
    }
  }

  // Update paths for NICs (no GPU Direct, PXN, ...)
  for (int n = 0; n < system->nodes[NET].count; n++) {  // 遍历所有 NIC 顶点
    struct ncclTopoNode* netNode = system->nodes[NET].nodes + n;

    for (int g = 0; g < system->nodes[GPU].count; g++) {  // 遍历所有 GPU 顶点
      // Check whether we can access the NIC through another NVLink-connected GPU (PXN)
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
      if (ncclPxnDisable(comm) != 1) {  // 不禁用 PXN，可以先把能用 PXN 的路径改为 PXN，因为 PATH_PXB 优于 PATH_PHB
        int localGpuIndex;
        NCCLCHECK(ncclTopoGetLocalGpu(system, netNode->id, &localGpuIndex));  // 查找采用指定 NET 顶点的 GPU 顶点索引
        if (localGpuIndex != g && localGpuIndex != -1) {  // 查找到的 GPU 顶点索引有效且不是当前遍历的 GPU 顶点
          // PXN = PCI + NVLink.
          struct ncclTopoNode* peerNode = system->nodes[GPU].nodes + localGpuIndex;
          enum ncclTopoGdrMode gdrMode;
          NCCLCHECK(ncclTopoCheckGdr(system, peerNode->gpu.rank, netNode->id, 1, &gdrMode));  // 检查指定 NET 与对应 GPU rank 之间是否支持 GDR
          // Only use PXN for NIC n if remote GPU p ...
          int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;  // 是否 PXN 中转的 GPU 到网卡的路径类型优于 P2C 就可以作为 PXN 中转
          if (/* (1) is connected to the NIC with PxN type and GDR is enabled*/
              peerNode->paths[NET][n].type <= pxnType && (gdrMode != ncclTopoGdrModeDisable) &&  // relay 到这块网卡必须够近（默认 ≤ PATH_P2C(6)）且支持 GDR
              /* and (2) is connected to us through NVLink */
              peerNode->paths[GPU][g].type <= PATH_NVL &&  // relay 与本 GPU 之间必须是 NVLink
              /* and (3) is on the same node as us */
              NCCL_TOPO_ID_SYSTEM_ID(peerNode->id) == NCCL_TOPO_ID_SYSTEM_ID(gpu->id) &&  // 两个 GPU 在同节点
              /* and (4) has either higher bw to that NIC or avoid going through the CPU (path.type is > PATH_PXN)*/
              (peerNode->paths[NET][n].bw > gpu->paths[NET][n].bw || gpu->paths[NET][n].type > PATH_PXN)) {  // 躲不开 CPU 换成 PXN 后带宽或者路径类型有收益
            // We can use that GPU as relay to communicate with that NIC.
            // Only enabling it in the GPU->NIC direction for now to favor
            // receiving locally and sending remotely (consistent with net.cc)
            // 用 GPU localGpuIndex 作为 GPU g 与 NET n 通信的中继，覆盖路径。本 GPU 从网卡收数据时走原路径（就近），发数据时绕道 relay 的网卡出去。
            NCCLCHECK(addInterStep(system, GPU, localGpuIndex, GPU, g, NET, n));
          }
        }
      }
      if (gpu->paths[NET][n].type < PATH_PHB) {  // 只看还没经过 CPU 的路径
        // Update path when we dont want to / can't use GPU Direct RDMA.
        enum ncclTopoGdrMode gdr;
        /*
        检查接收方向是否支持 GDR，注意内部是会看到 PXN 的，PATH_PXN(7) > PATH_P2C(6)，
        直接判定 GDR 不可用，所以内部会把 PXN 路径的路径类型换成 relay GPU 到网卡的距离来判定。
        */
        NCCLCHECK(ncclTopoCheckGdr(system, system->nodes[GPU].nodes[g].gpu.rank, netNode->id, 0, &gdr));
        if (gdr == 0) {
          // 不能使用 GDR，需要将流量从 CPU 转向 GPU
          // We cannot use GPU Direct RDMA, divert all traffic through the CPU local to the GPU
          int localCpu;
          NCCLCHECK(ncclGetLocalCpu(system, g, &localCpu));  // 找到 GPU g 最近的 CPU 顶点索引
          // GPU 与 NET 的双向路径都要经过 CPU
          NCCLCHECK(addInterStep(system, CPU, localCpu, NET, n, GPU, g));
          NCCLCHECK(addInterStep(system, CPU, localCpu, GPU, g, NET, n));
        }
      }
    }
  }

  // 所有 NET 网卡都记录一个会采用它的 GPU 顶点索引
  // Pre-compute NET local gpus to accelerate search
  for (int n = 0; n < system->nodes[NET].count; n++) {
    struct ncclTopoNode* net = system->nodes[NET].nodes + n;
    NCCLCHECK(ncclTopoGetLocalGpu(system, net->id, &net->net.localGpu));
  }
  return ncclSuccess;
}

/*
  1. 用路径表划出「本 rank 所在的本地可达域」
  2. 把不在这个域里的 GPU 顶点从拓扑图里物理删除
  3. 顺带算出 system->inter（整个通信是不是跨节点）
*/
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  int* domains;
  int64_t* ids = NULL;
  int myDomain = 0;
  int ngpus = system->nodes[GPU].count;
  NCCLCHECK(ncclCalloc(&domains, system->nodes[GPU].count));
  NCCLCHECKGOTO(ncclCalloc(&ids, system->nodes[GPU].count), ret, fail);
  /*
  为全部 GPU 划分连通域，找出哪些 GPU 彼此之间是本地可达的，把属于一伙的编上同一个域号
  */
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    domains[g] = g;
    ids[g] = gpu->id;
    for (int p = 0; p < g; p++) {
      if (gpu->paths[GPU][p].type < PATH_NET) {
        domains[g] = std::min(domains[g], domains[p]);  // 类似并查集，直接选择连通域中最小 GPU 索引作为代表
      }
    }
    if (gpu->gpu.rank == comm->rank) myDomain = domains[g];  // 找到当前 rank 的 GPU 所处连通域的代表 GPU 索引
  }

  ncclTopoRemovePaths(system);
  for (int i = 0; i < ngpus; i++) {
    if (domains[i] == myDomain) continue;  // 同一个域的 GPU 不用处理
    struct ncclTopoNode* gpu = NULL;
    int g;
    for (g = 0; g < system->nodes[GPU].count /* This one varies over the loops */; g++) {
      gpu = system->nodes[GPU].nodes + g;
      if (gpu->id == ids[i]) break;  // 比较 GPU id，因为删除左移会让 GPU 数组索引发生变化
      else gpu = NULL;
    }
    if (gpu == NULL) {
      WARN("Could not find id %lx", ids[i]);
      ret = ncclInternalError;
      goto fail;
    }
    // 删除不属于当前连通域的 GPU 顶点，包括直连它的 links，并且左移 GPU 数组
    NCCLCHECKGOTO(ncclTopoRemoveNode(system, GPU, g), ret, fail);
  }
  // 如果连通 GPU 顶点数等于总 ranks 数，不用跨节点
  system->inter = system->nodes[GPU].count == comm->nRanks ? 0 : 1;
exit:
  free(domains);
  free(ids);
  return ret;
fail:
  goto exit;
}

void ncclTopoFree(struct ncclTopoSystem* system) {
  ncclTopoRemovePaths(system);
  free(system);
}

NCCL_PARAM(P2pPerChannelNetBw, "P2P_PER_CHANNEL_NET_BW", /*GB/s*/ 14);

static ncclResult_t ncclTopoGetNchannels(struct ncclComm* comm, int g /*local gpu index*/, int peerRank,
                                         int* nChannels) {
  int peer;
  struct ncclTopoSystem* system = comm->topo;
  struct ncclTopoLinkList* path = NULL;
  if (ncclTopoRankToIndex(system, peerRank, &peer, /*showWarn=*/false) == ncclSuccess) {
    // Same rank
    if (g == peer) {
      *nChannels = -1;
      return ncclSuccess;
    }
    // Local rank
    path = system->nodes[GPU].nodes[peer].paths[GPU] + g;
    if (path->type == PATH_NVL || path->type == PATH_NVB) {
      // NVLink-based connection (NVB uses NVLink)
      float nvlBw = ncclTopoNVLinkBw(system->nodes[GPU].nodes[g].gpu.cudaCompCap);
      *nChannels = 2 * std::max(1, (int)(path->bw / nvlBw));
    } else {
      // PCIe connection
      *nChannels = 2;
    }
  } else {
    // Remote rank, use network
    int nNetChannels = comm->config.nChannelsPerNetPeer;
    if (nNetChannels == NCCL_CONFIG_UNDEF_INT) {
      float netBw = 0.0;
      int netCount = 0;
      NCCLCHECK(ncclTopoGetLocalNetCountByBw(system, g, &netCount, &netBw));
      // We use at least 1 channel per NIC, and more if needed to meet the bw requirement.
      nNetChannels = 2;
      if (netCount > 0) nNetChannels = std::max(netCount, divUp((int)netBw, (int)ncclParamP2pPerChannelNetBw()));
    }
    *nChannels = nNetChannels;
  }
  return ncclSuccess;
}

NCCL_PARAM(MinP2pNChannels, "MIN_P2P_NCHANNELS", 1);
NCCL_PARAM(MaxP2pNChannels, "MAX_P2P_NCHANNELS", MAXCHANNELS);
extern int64_t ncclParamWorkArgsBytes();

ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm) {
  int g = 0;
  while (comm->topo->nodes[GPU].nodes[g].gpu.rank != comm->rank) g++;
  if (g == comm->topo->nodes[GPU].count) return ncclInternalError;

  int minChannels = MAXCHANNELS;
  for (int r = 0; r < comm->nRanks; r++) {
    int nChannels;
    NCCLCHECK(ncclTopoGetNchannels(comm, g, r, &nChannels));
    if (nChannels >= 0) minChannels = std::min(minChannels, nChannels);
  }
  comm->p2pnChannelsPerPeer = minChannels;
  comm->p2pMaxPeers = (comm->config.maxP2pPeers == NCCL_CONFIG_UNDEF_INT) ? comm->nRanks : comm->config.maxP2pPeers;
  return ncclSuccess;
}

ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) {
  if (comm->sharedRes->owner != comm) {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels =
      std::min(std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels()), comm->sharedRes->tpP2pNChannels);
  } else {
    comm->p2pnChannels = std::min(comm->nChannels, (int)ncclParamMaxP2pNChannels());
    comm->p2pnChannels = std::max(comm->p2pnChannels, (int)ncclParamMinP2pNChannels());
  }

  // Make nChannelsPerPeer and nChannels powers of 2. This is relied on when mapping p2p peers to channels.
  comm->p2pnChannelsPerPeer = pow2Up(comm->p2pnChannelsPerPeer);
  comm->p2pnChannels = pow2Up(comm->p2pnChannels);
  comm->p2pnChannels = std::min(comm->p2pnChannels, pow2Down(ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes())));
  // Rounding up may exceed maxCTAs, which bounds the grid of every kernel; round down to honor it.
  comm->p2pnChannels = std::min(comm->p2pnChannels, pow2Down(comm->config.maxCTAs));

  if (comm->nNodes > 1 && comm->config.nChannelsPerNetPeer == NCCL_CONFIG_UNDEF_INT) {
    // In the case of >1 NVLD (and the user didn't set nChannelsPerNetPeer), the network is the botteneck.
    // Reduce the number of channels per host to avoid going above p2pnChannels to fit all the peers
    // within a single round.
    INFO(NCCL_INIT, "Tuning P2P operations with maxP2pPeers = %d", comm->p2pMaxPeers);
    while (comm->p2pnChannelsPerPeer * divUp(comm->p2pMaxPeers, NCCL_MAX_DEV_WORK_P2P_PER_BATCH) > comm->p2pnChannels &&
           comm->p2pnChannelsPerPeer > 1) {
      comm->p2pnChannelsPerPeer /= 2;
    }
  } else {
    comm->p2pnChannelsPerPeer = std::min(comm->p2pnChannels, comm->p2pnChannelsPerPeer);
  }

  // Init channels that weren't used so far
  for (int c = comm->nChannels; c < comm->p2pnChannels; c++) NCCLCHECK(initChannel(comm, c));

  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks) {
  int ngpus = system->nodes[GPU].count;
  NCCLCHECK(ncclCalloc(ranks, ngpus));
  int nvbGpus = 0;
  for (int g = 0; g < ngpus; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    if (gpu->gpu.rank != rank) continue;
    for (int p = 0; p < ngpus; p++) {
      if (gpu->paths[GPU][p].type == PATH_NVB) {
        (*ranks)[nvbGpus++] = system->nodes[GPU].nodes[p].gpu.rank;
      }
    }
  }
  *nranks = nvbGpus;
  return ncclSuccess;
}

// 计算所有 GPU 到指定类型所有顶点的最小路径类型
ncclResult_t ncclTopoGetGpuMinPath(struct ncclTopoSystem* system, int type, int* min) {
  int minPath = PATH_SYS;
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];  // 获取当前 GPU 到指定类型的 paths
    if (paths == NULL) continue;
    for (int j = 0; j < system->nodes[type].count; j++) {  // 遍历指定类型的各个 path
      if (type == GPU && i == j) continue;  // 如果是 GPU->GPU，忽略自己到自己
      minPath = std::min(minPath, paths[j].type);  // 维护 path 的最小类型
    }
  }
  *min = minPath;
  return ncclSuccess;
}

// 计算所有 GPU 到指定类型所有顶点的最大路径类型
ncclResult_t ncclTopoGetGpuMaxPath(struct ncclTopoSystem* system, int type, int* max) {
  int maxPath = PATH_LOC;
  for (int i = 0; i < system->nodes[GPU].count; i++) {
    struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[i].paths[type];
    if (paths == NULL) continue;
    for (int j = 0; j < system->nodes[type].count; j++) {
      if (type == GPU && i == j) continue;
      maxPath = std::max(maxPath, paths[j].type);
    }
  }
  *max = maxPath;
  return ncclSuccess;
}

// Check whether the system is all GPUs directly or indirectly connected to each other
// through NVLink and C2C.
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink) {
  int maxPath;
  NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxPath));
  *allNvLink = maxPath >= PATH_PIX ? 0 : 1;
  return ncclSuccess;
}

// Check whether the system is all GPUs connected directly to each other through NVLink/NVSwitch.
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* directNvlink) {
  int maxPath;
  NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxPath));
  *directNvlink = maxPath <= PATH_NVL;
  return ncclSuccess;
}

/*
判断是不是一台双路服务器，每颗 CPU 各挂 4 块 GPU，左右两半内部都是 NVLink 全连接，但两半之间只有 QPI/UPI（走 PATH_SYS）的情况，拓扑如下：
        CPU0 (socket 0)                    CPU1 (socket 1)
            |                                   |
      [NVSwitch A]                        [NVSwitch B]
      /    |    |    \                    /    |    |    \
    G0    G1   G2   G3                  G4   G5   G6   G7
    └────┴────┴────┘                    └────┴────┴────┘
          PATH_NVL                            PATH_NVL
                ←──────── PATH_SYS ────────→
*/
// Check whether we are in a split NVLink situation, with two NVLink domains, not
// connected through NVLink (e.g. QPI).
ncclResult_t ncclTopoSplitNvLink(struct ncclTopoSystem* system, int* splitNvLink) {
  ncclResult_t res = ncclSuccess;
  int nvlDomains = 0;
  int *nvlDomain = NULL, *nvlDomainCount = NULL;
  // Compute NVLink domains
  NCCLCHECKGOTO(ncclCalloc(&nvlDomain, system->nodes[GPU].count), res, exit);
  // 类似并查集合并 GPUs
  for (int g = 0; g < system->nodes[GPU].count; g++) nvlDomain[g] = g;
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    int domain = nvlDomain[g];
    for (int p = g + 1; p < system->nodes[GPU].count; p++) {
      if (gpu->paths[GPU][p].type == PATH_NVL) {
        nvlDomain[p] = domain;
      }
    }
  }
  // 在每个域的代表索引处累加
  // Compute number of GPUs per NVLink domain.
  NCCLCHECKGOTO(ncclCalloc(&nvlDomainCount, system->nodes[GPU].count), res, exit);
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    nvlDomainCount[nvlDomain[g]]++;
  }
  // 统计有多少个域
  // Count the number of NVLink domains
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    if (nvlDomainCount[g] > 1) nvlDomains++;
  }
  *splitNvLink = nvlDomains == 2 ? 1 : 0;

exit:
  free(nvlDomain);
  free(nvlDomainCount);
  return res;
}
