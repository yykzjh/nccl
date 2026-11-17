/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "nccl.h"
#include "nvmlwrap.h"
#include "coll_net.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin.h"
#endif
#include "rma.h"
#include "transport.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "cpuset.h"
#include "bootstrap.h"
#include <mutex>
#include <float.h>

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

/*
常量	     值	  字符串	含义
PATH_LOC	 0	 LOC	  自己(等效禁用跨卡 P2P)
PATH_NVL	 1	 NVL	  NVLink 直连
PATH_NVB	 2	 NVB	  经中间 GPU 的 NVLink
PATH_C2C	 3	 C2C	  Chip-to-Chip
PATH_PIX	 4	 PIX	  单个 PCIe bridge
PATH_PXB	 5	 PXB	  多个 PCIe bridge(不过 host bridge)
PATH_P2C	 6	 P2C	  GPU↔NIC 经 C2C+PCIe
PATH_PXN	 7	 PXN	  PXN
PATH_PHB	 8	 PHB	  PCIe + host bridge(通常 CPU)
PATH_SYS	 9	 SYS	  PCIe + NUMA 间 SMP 互连(QPI/UPI)
*/
const char* topoNodeTypeStr[] = {"GPU", "PCI", "NVS", "CPU", "NIC", "NET", "GIN", "RMA", "DEV", "CXB"};
const char* topoLinkTypeStr[] = {"LOC", "NVL", "", "C2C", "PCI", "", "", "", "", "SYS", "NET"};
const char* topoPathTypeStr[] = {"LOC", "NVL", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS"};

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

// 从 node 上溯到 CPU 顶点，from 顶点是找到 node 的前一个顶点
static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu, struct ncclTopoNode* from) {
  *cpu = NULL;
  if (node->type == CPU) {  // 找到 CPU 顶点，返回
    *cpu = node;
    return ncclSuccess;
  }
  for (int l = 0; l < node->nlinks; l++) {  // 遍历顶点的所有 links
    // link 类型要是 PCI，link 的对端要是 PCI 或 CPU 顶点，而且对端顶点不能是之前的顶点。
    // Go up the PCI tree to find the CPU. Follow only PCI switches.
    if (node->links[l].type == LINK_PCI && node->links[l].remNode != from &&
        (node->links[l].remNode->type == PCI || node->links[l].remNode->type == CPU)) {
      // 递归对端顶点
      NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu, node));
    }
    if (*cpu != NULL) return ncclSuccess;
  }
  return ncclSuccess;
}

int interCpuBw = 0;
int cpuPciBw = 0;

// 按 CPU 代际查表估跨 NUMA 带宽
static ncclResult_t ncclTopoGetInterCpuBw(struct ncclTopoNode* cpu, float* bw) {
  *bw = LOC_BW;  // 初值 5000，最高档
  /*
  对应的硬件事实与取值：
    平台	        互连技术	        cpu.model 分档	                                常量	                     值（GB/s）
    Intel Xeon	 UPI / QPI	      Emerald Rapids（familyId==6 && modelId>=0xCF）	ERP_QPI_BW	              40.0
                                  Sapphire Rapids（>=0x8F）	                      SRP_QPI_BW	              22.0
                                  Skylake-SP（>=0x55）	                          SKL_QPI_BW	              10.0
                                  Broadwell 及更早（else）	                       BDW_QPI_BW	               6.0
    AMD EPYC	   Infinity Fabric	不分代	                                        AMD_BW	                  16.0
    IBM POWER9	 X-Bus / NVLink	  —	                                             P9_BW	                   32.0
    ARM	         —	              —	                                             ARM_BW	                   6.0
    兆芯	       ZPI	            永丰 / 其他	                                     YONGFENG_ZPI_BW / ZPI_BW	 9.0 / 6.0
  */
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
    *bw = P9_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
    *bw = ARM_BW;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    *bw = cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_ERP ? ERP_QPI_BW :
          cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SRP ? SRP_QPI_BW :
          cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SKL ? SKL_QPI_BW :
                                                            BDW_QPI_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
    *bw = AMD_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    *bw = cpu->cpu.model == NCCL_TOPO_CPU_MODEL_YONGFENG ? YONGFENG_ZPI_BW : ZPI_BW;
  }
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

// 在指定类型的顶点桶中获取指定 ID 的顶点
ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  for (int i = 0; i < system->nodes[type].count; i++) {  // 在指定类型桶中遍历顶点
    if (system->nodes[type].nodes[i].id == id) {  // 匹配 id
      *node = system->nodes[type].nodes + i;  // 返回匹配的顶点
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

// 初始化指定类型的 topo 图顶点
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {  // 保存指定类型顶点的桶满了 > 640
    WARN("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  struct ncclTopoNode* n = system->nodes[type].nodes + system->nodes[type].count;  // 取空顶点槽位
  system->nodes[type].count++;
  // 初始化顶点类型和 ID
  n->type = type;
  n->id = id;
  // 根据顶点类型分别初始化顶点属性
  if (type == GPU) {
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
    n->gpu.mloPart = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    n->cpu.arch = NCCL_TOPO_UNDEF;
    n->cpu.vendor = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
  } else if (type == NET) {
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.bw = 0.0;
    n->net.latency = 0.0;
    n->net.railId = NCCL_TOPO_UNDEF;
    n->net.planeId = NCCL_TOPO_UNDEF;
  } else if (type == DEV) {
    n->dev.dev = NCCL_TOPO_UNDEF;
    n->dev.cudaCompCap = NCCL_TOPO_UNDEF;
  }
  *node = n;  // 返回该顶点指针
  return ncclSuccess;
}

// 删除指定顶点类型和索引的顶点
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  struct ncclTopoNode* delNode = system->nodes[type].nodes + index; // 找到该顶点
  for (int t = 0; t < NCCL_TOPO_NODE_TYPES; t++) {  // 遍历所有顶点类型
    free(delNode->paths[t]);  // 释放删除顶点当前类型的路径缓存，PCI 类型 paths 还是 NULL，删除 GPU 时已经算好
    for (int n = 0; n < system->nodes[t].count; n++) {  // 遍历当前类型的所有顶点
      struct ncclTopoNode* node = system->nodes[t].nodes + n;
      if (node == delNode) continue;
      for (int l = 0; l < node->nlinks; l++) {
        while (l < node->nlinks && node->links[l].remNode == delNode) {  // 删除指向删除顶点的边
          memmove(node->links + l, node->links + l + 1, (node->nlinks - l - 1) * sizeof(struct ncclTopoLink));
          node->nlinks--;
        }
        // 因为后续会删除 delNode 并将它同类型顶点数组中右边所有顶点左移一位，提前修改在它右边顶点的指针
        if (l < node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  if (type == DEV) {  // 如果删除的事 DEV 顶点
    for (int n = 0; n < system->nodes[GPU].count; n++) {  // 遍历所有 GPU 顶点
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + n;
      if (gpu->gpu.parent == delNode) {
        gpu->gpu.parent = NULL;  // GPU 顶点指向的该 DEV 顶点改为 NULL
      } else if (gpu->gpu.parent > delNode) {
        gpu->gpu.parent--;  // 所有指向的 DEV 顶点地址都左移一位
      }
    }
  }
  // 删除该顶点
  memmove(delNode, delNode + 1, (system->nodes[type].count - index - 1) * sizeof(struct ncclTopoNode));
  system->nodes[type].count--;
  return ncclSuccess;
}

/*
每个顶点 ncclTopoNode 有一个 links[NCCL_TOPO_MAX_LINKS] 边表（576 槽）和 nlinks 计数。
当前函数就是往 node 的边表里加一条"从 node 指向 remNode、类型为 type、带宽为 bw"的边。
*/
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float bw) {
  // Aggregate links into higher bw for NVLink
  struct ncclTopoLink* link;
  // 遍历所有已有 links，找到匹配的、可能已经有的 link 槽
  for (link = node->links; link - node->links != NCCL_TOPO_MAX_LINKS && link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link - node->links == NCCL_TOPO_MAX_LINKS) {  // 满了
    WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
    return ncclInternalError;
  }
  if (link->remNode == NULL) node->nlinks++;  // 只有落在空槽（落点 b，真正的新边）时才递增 nlinks
  /*
  填边 + 带宽聚合。
    写 type 和 remNode：对新槽是从 NULL 填成 remNode，对既有边是原值重写，无害
    link->bw += bw：新槽 bw 初始是 calloc 的 0，0 += bw = bw；既有边，累加
  同一对顶点之间的多条平行链路会被合并成一条胖边，带宽相加，而不是存成多条边。最典型的就是 NVLink，两块 GPU 之间往往有多条物理 NVLink，全部叠加成一条高带宽边。
  */
  link->type = type;
  link->remNode = remNode;
  link->bw += bw;

  // 插入排序保证 links 按照带宽降序排列
  // Sort links in BW descending order
  struct ncclTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));  // 拷贝到临时 link
  while (link != node->links) {
    if ((link - 1)->bw >= linkSave.bw) break;  // 某条 link 带宽大于等于当前 link，停止
    memcpy(link, link - 1, sizeof(struct ncclTopoLink));  // 带宽小于当前 link 带宽的 link 往右移一格
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));  // 把暂存 link 边放进腾出的位置
  return ncclSuccess;
}

/*
id 为 [vendor<<48][device<<32][subsys_vendor<<16][subsys_device]。拆开看：
代次	 vendor	 device	 subsys_vendor	subsys_device
Gen4	0x1000	0xc010	0x1000	        0xa000（固定）
Gen5	0x1000	0xc030	0x1000	        level × 0x1000（0x0000 / 0x1000）

Gen5 用 subsys_device 的 bit12 显式编码自己在第几级虚拟 switch，父 switch 是 level 0（0x0000）、
子 switch 是 level 1（0x1000），所以 getBcmGen 必须带 level 参数才能分别匹配。
*/
// BCM Gen4 Switches present themselves as a two-level hierarchical switch
// even though they're supposed to sustain full BW across all ports.
// Flatten the switch as this extra level can break the search and make
// NCCL take wrong topology decisions.
int getBcmGen(uint64_t id, int level) {
  if ((id & 0xfffffffffffff000) == 0x1000c0101000a000) return 4;
  if ((id & 0xfffffffffffff000) == (0x1000c03010000000 | level * 0x1000)) return 5;
  return 0;
}

/*
Broadcom Switch 有两层，pcilink 连的是端口，不是设备，target 指向的是对端的 virtual switch 顶点，GPU／NIC 挂在那一层之下。
  硬件事实：BCM PEX switch 本应在所有端口之间维持全带宽（一颗芯片就是一个大交换平面）。
  软件呈现：但它对 PCIe 枚举暴露成 parent switch → sub switch → 设备 的两级结构。
  后果：这个多出来的一级会破坏搜索，让 NCCL 做出错误的拓扑决策。

物理上一颗 switch，软件上两级虚拟 switch，而 NCCL 认为这层虚拟层级是多余的，所以在建完图后把它压掉。
  1. 遍历 PCI 顶点，getBcmGen(pciSwitch->pci.device, 0) 认出 BCM 父 switch;
  2. 在它的链接里挑出子 switch，条件是 sub->type == PCI && getBcmGen(sub->pci.device, 1) == gen，
必须是同代、且自己是 level 1（Gen4 靠 device id 完全相同配对,Gen5 靠 subsys_device 区分）;
  3. 把子 switch 的所有链接搬到父 switch 上（memcpy(pciSwitch->links + pciSwitch->nlinks, sub->links + l, ...)），同时把对端的反向链接改指父 switch;
  4. ncclTopoRemoveNode(system, PCI, index) 删掉子 switch 顶点;
  5. pciSwitch->pci.device |= 0xffff 把低 16 位打成 0xffff，让它不再匹配任何代次，防止被重复压平;
  6. s = -1 重启循环，因为 nodes[PCI] 数组已经变了。
*/
ncclResult_t ncclTopoFlattenBcmSwitches(struct ncclTopoSystem* system) {
  ncclResult_t ret = ncclSuccess;
  for (int s = 0; s < system->nodes[PCI].count; s++) {  // 遍历所有 PCI 顶点
    struct ncclTopoNode* pciSwitch = system->nodes[PCI].nodes + s;
    int gen = getBcmGen(pciSwitch->pci.device, 0);  // 判断是否是 Gen4 或者 Gen5 父 switch
    // Flatten Gen4 PEX switches in base mode
    if (gen) {
      // Find sub switches with the same device ID.
      int64_t* subSwIds;
      NCCLCHECK(ncclCalloc(&subSwIds, pciSwitch->nlinks));
      int subs = 0;
      for (int l = 0; l < pciSwitch->nlinks; l++) {  // 遍历该 switch 所有边
        struct ncclTopoNode* sub = pciSwitch->links[l].remNode;  // 对端 sub switch
        // Only fuse sub switches with the same device ID.
        if (sub->type != PCI || getBcmGen(sub->pci.device, 1) != gen) continue;  // 必须是同代 sub switch
        // Save sub switch for later
        subSwIds[subs++] = sub->id;  // 保存 sub switch ID
        // Remove link to that sub switch
        memmove(pciSwitch->links + l, pciSwitch->links + l + 1,
                (pciSwitch->nlinks - l - 1) * (sizeof(struct ncclTopoLink)));  // 去掉 parent switch 和 sub switch 之间的边
        pciSwitch->nlinks--;
        // Don't increase l for the next iteration as we just shifted all links by one.
        l--;  // 左移了，索引补偿
      }

      // 把 sub switch 的邻居给 parent switch
      for (int s = 0; s < subs; s++) {
        // Find sub switch (system->nodes[PCI].nodes is changing every time we remove a node)
        int index;
        NCCLCHECKGOTO(ncclTopoIdToIndex(system, PCI, subSwIds[s], &index), ret, fail);  // 找到该 sub switch PCI 顶点索引
        struct ncclTopoNode* sub = system->nodes[PCI].nodes + index;
        // Connect all sub PCI devices to the parent switch
        for (int l = 0; l < sub->nlinks; l++) {
          struct ncclTopoNode* remNode = sub->links[l].remNode;
          if (remNode == pciSwitch) continue;  // 跳过指向 parent switch 的边
          // Add link from parent PCI switch -> PCI device
          if (pciSwitch->nlinks == NCCL_TOPO_MAX_LINKS) {
            WARN("Error : too many Topo links (max %d)", NCCL_TOPO_MAX_LINKS);
            ret = ncclInternalError;
            goto fail;
          }
          // sub switch 其他 links 都托付给 parent switch
          memcpy(pciSwitch->links + pciSwitch->nlinks, sub->links + l, sizeof(struct ncclTopoLink));
          pciSwitch->nlinks++;
          // 这些被托付的设备原本连向 sub switch 的边也改为连向 parent switch
          // Update link from PCI device -> parent PCI switch
          for (int rl = 0; rl < remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        NCCLCHECKGOTO(ncclTopoRemoveNode(system, PCI, index), ret, fail);  // 删除 sub switch PCI 顶点
      }
      // Set subdevice to 0xffff to make sure we don't merge this switch again.
      pciSwitch->pci.device |= 0xffff;  // 让它不再被识别为 BCM switch
      free(subSwIds);
      // Restart, as system->nodes[PCI].nodes has changed.
      s = -1;  // 删除一个重新从头遍历所有 PCI 顶点
      continue;
    fail:
      free(subSwIds);
      return ret;
    }
  }
  return ret;
}

/*
跨 NUMA 域的 CPU 之间补上最后一条长跳，并把它标成 LINK_SYS。
因为 XML 里每个 <cpu> 是平级的独立顶层节点、彼此没有任何边，不补这一步，
不同 NUMA 下的设备在图上就是两个互不连通的子图，路径搜索算不出跨 NUMA 的路径。
*/
ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // And connect all CPU nodes together
  for (int n = 0; n < system->nodes[CPU].count; n++) {   // 遍历所有 CPU 顶点
    struct ncclTopoNode* cpu1 = system->nodes[CPU].nodes + n;
    for (int p = 0; p < system->nodes[CPU].count; p++) {  // 遍历两两 CPU 顶点对
      struct ncclTopoNode* cpu2 = system->nodes[CPU].nodes + p;
      // 跳过自己和不同 host 上的 CPU 顶点对
      if (n == p || (NCCL_TOPO_ID_SYSTEM_ID(cpu1->id) != NCCL_TOPO_ID_SYSTEM_ID(cpu2->id))) continue;
      float bw;
      // 按 CPU 代际查表估跨 NUMA 带宽
      NCCLCHECK(ncclTopoGetInterCpuBw(cpu1, &bw));
      // 建立两个 CPU 顶点之间的 link，链路类型为 LINK_SYS，带宽为跨 NUMA 带宽
      NCCLCHECK(ncclTopoConnectNodes(cpu1, cpu2, LINK_SYS, bw));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    char mloStr[128] = "";
    snprintf(mloStr, sizeof(mloStr), " [MLOPart %d]", node->gpu.mloPart);
    sprintf(line + offset, "%s/%lx-%lx (%d)%s", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->gpu.rank, (node->gpu.mloPart == -1) ? "" : mloStr);
  } else if (node->type == CPU) {
    sprintf(line + offset, "%s/%lx-%lx (%d/%d/%d)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->cpu.arch, node->cpu.vendor, node->cpu.model);
  } else if (node->type == PCI) {
    sprintf(line + offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->pci.device);
  } else if (node->type == DEV) {
    sprintf(line + offset, "%s/%lx-%lx (%lx)", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id), node->dev.device);
  } else {
    sprintf(line + offset, "%s/%lx-%lx", topoNodeTypeStr[node->type], NCCL_TOPO_ID_SYSTEM_ID(node->id),
            NCCL_TOPO_ID_LOCAL_ID(node->id));
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i = 0; i < offset; i++) line[i] = ' ';

  for (int l = 0; l < node->nlinks; l++) {
    struct ncclTopoLink* link = node->links + l;
    if (link->type == LINK_LOC) {
      sprintf(line + offset, "+ %s[%2.1f] - %s/%lx-%lx", topoLinkTypeStr[link->type], link->bw,
              topoNodeTypeStr[link->remNode->type], NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id),
              NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
      INFO(NCCL_GRAPH, "%s", line);
    } else if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line + offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->bw);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line + nextOffset, "%s/%lx-%lx (%d/%lx/%d/%f)", topoNodeTypeStr[link->remNode->type],
                  NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id),
                  link->remNode->net.collSupport, link->remNode->net.asic, link->remNode->net.port,
                  link->remNode->net.bw);
        } else {
          sprintf(line + nextOffset, "%s/%lx-%lx", topoNodeTypeStr[link->remNode->type],
                  NCCL_TOPO_ID_SYSTEM_ID(link->remNode->id), NCCL_TOPO_ID_LOCAL_ID(link->remNode->id));
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

/*
纯日志输出，需要打开 GRAPH 子系统：NCCL_DEBUG_SUBSYS=GRAPH。
如下原始结构：
  CPU(NUMA 0) ── PCIe root ── PCIe switch ─┬─ NVSwitch ─ DEV0 ─ GPU0
                                            │            DEV1 ─ GPU1
                                            │            DEV2 ─ GPU2
                                            │            DEV3 ─ GPU3
                                            └─ mlx5_0 (IB)
对应打印的拓扑图示例为：
  node-01:12345:12345 [0] NCCL INFO === System : maxBw 25.2 totalBw 25.2 ===
  node-01:12345:12345 [0] NCCL INFO  CPU/0-0 (11/0/0) 
  node-01:12345:12345 [0] NCCL INFO  + PCI[12.0] - PCI/0-1 (15b8) 
  node-01:12345:12345 [0] NCCL INFO   + PCI[12.0] - PCI/0-2 (15b3) 
  node-01:12345:12345 [0] NCCL INFO    + PCI[150.0] - NVS/0-3 (a2) 
  node-01:12345:12345 [0] NCCL INFO     + NVL[25.2] - DEV/0-4 (248) 
  node-01:12345:12345 [0] NCCL INFO      + LOC[25.2] - GPU/0-0 (0) 
  node-01:12345:12345 [0] NCCL INFO     + NVL[25.2] - DEV/0-5 (249) 
  node-01:12345:12345 [0] NCCL INFO      + LOC[25.2] - GPU/0-1 (1) 
  node-01:12345:12345 [0] NCCL INFO    + PCI[12.0] - DEV/0-6 (24a) 
  node-01:12345:12345 [0] NCCL INFO     + LOC[25.2] - GPU/0-2 (2) 
  node-01:12345:12345 [0] NCCL INFO    + PCI[12.0] - NET/0-8 (0/0x101b/1/25.0) 
  node-01:12345:12345 [0] NCCL INFO ==========================================
  node-01:12345:12345 [0] NCCL INFO GPU/0-0 :GPU/0-0 (0/0.0/LOC) GPU/0-1 (2/25.2/NVL) GPU/0-2 (1/12.0/PIX) ...
*/
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxBw %2.1f totalBw %2.1f ===", s->maxBw, s->totalBw);
  char line[1024];
  // 从每个 CPU 顶点出发，DFS 展开整张图
  for (int n = 0; n < s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes + n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  NCCLCHECK(ncclTopoPrintPaths(s));
  return ncclSuccess;
}

// 把 PCIe 树里向上走的那条边挪到边表的最后，让后续遍历天然形成"先下行、后上行"的顺序
static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // Shift all links to have upLink as last link
  if (upNode) {
    // 找到当前顶点的父连边
    int l = 0;
    while (node->links[l].remNode != upNode) l++;
    // 先拷贝一份父连边
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links + l, sizeof(struct ncclTopoLink));
    // 后面的有效边依次左移填补
    while (node->links[l + 1].remNode) {
      memcpy(node->links + l, node->links + l + 1, sizeof(struct ncclTopoLink));
      l++;
    }
    // 父连边放到最后一个有效槽
    memcpy(node->links + l, &upLink, sizeof(struct ncclTopoLink));
  }

  // 递归下钻 PCIe 子树
  for (int l = 0; l < node->nlinks; l++) {
    struct ncclTopoLink* link = node->links + l;
    // 只沿纵向 PCIe 树递归，防回溯
    if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

/*
links 的期望顺序如下：
  1. NVLink 在前，因为 ncclTopoConnectNodes 的带宽降序插入排序，NVLink 带宽最高（18~40.1），自然排最前。
  2. PCI down，就是 ncclTopoSort 保留下来的部分，保持原有的带宽降序。
  3. PCI up，ncclTopoSort 第一段强制挪到最后。
  4. SYS 在后	CPU 节点根处 upNode == NULL，整段排序跳过，SYS 边保持建边时的位置。
*/
// We want the graph to be organized to ease/accelerate traversal :
// 1. NVLinks (already the case)
// 2. PCI down
// 3. PCI up
// 4. SYS (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n = 0; n < system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes + n, NULL));
  return ncclSuccess;
}

// Minimum network BW of a single device accessible by rank.
// Note: This function does not sum up the bw over multiple NICs if many are accessible.
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw) {
  int g = 0;
  while (g < system->nodes[GPU].count && system->nodes[GPU].nodes[g].gpu.rank != rank) g++;
  if (g == system->nodes[GPU].count) return ncclInternalError;

  int64_t firstNetId = 0;
  float minBw = FLT_MAX;
  for (int c = 0; c < MAXCHANNELS; c++) {
    int net;
    int64_t netId;
    NCCLCHECK(ncclTopoGetLocalNet(system, rank, c, &netId, NULL));
    NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &net));
    if (c == 0) firstNetId = netId;
    else if (netId == firstNetId) break;

    minBw = std::min(minBw, system->nodes[GPU].nodes[g].paths[NET][net].bw);
  }
  // if no net is found, return 0 as a minimum bw
  *bw = (minBw < FLT_MAX) ? minBw : 0.0;
  return ncclSuccess;
}

// 把一个 <net> XML 节点落成一个 NET 顶点（type=NET），填全 net 成员的所有属性，再用 LINK_NET 双向连到父 NIC 顶点。
ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));  // 获取 dev 属性整型，NET 插件维护的 net 设备 ID

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, NET, netId));  // 创建 NET 空顶点
  net->net.dev = dev;
  const char* str;
  /*
  asic 通过获取 guid 属性字符串并按十六进制转为整型，如果没有该属性采用 NET ID 代替。
  ASIC（Application-Specific Integrated Circuit，专用集成电路），的 guid 取自 devAttr.sys_image_guid，
  是 IB 规范里的 "System Image GUID"，本就标识整个 HCA 设备镜像（整颗芯片），不是端口级的（端口级的是 port_guid/GID）。
  */
  // if not guid is present use the net->id unique id instead, which will be unique within the node/NVLD
  NCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  net->net.asic = (str) ? strtoull(str, NULL, 16) : netId;

  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);  //  获取 speed 属性整型，默认为 0
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;  // Mb/s 变为 GB/s，兜底为 12.5 GB/s
  ncclResult_t ret;
  NOWARN(ret = xmlGetAttrFloat(xmlNet, "latency", &net->net.latency), NCCL_GRAPH);  // 获取 latency 属性浮点数
  if (ret != ncclSuccess) net->net.latency = 0;
  // 依次获取 <net> 节点属性并赋值给 net 顶点
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "port", &net->net.port, 0), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "gdr", &net->net.gdrSupport, 0), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "coll", &net->net.collSupport, 0), NCCL_GRAPH);
  int railId, planeId;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "rail", &railId, NCCL_TOPO_UNDEF), NCCL_GRAPH);
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "plane", &planeId, NCCL_TOPO_UNDEF), NCCL_GRAPH);
  net->net.railId = railId;
  net->net.planeId = planeId;

  // build the PCI id using the parent PCI link
  uint64_t hacc[2] = {1, 1};
  const char* busId = NULL;
  struct ncclXmlNode* parent = xmlNet->parent;  // <nic> 节点
  while (parent != NULL && strcmp(parent->name, "pci") != 0) parent = parent->parent;  // 向上找到网卡的 <pci> 为止
  if (parent) NCCLCHECK(xmlGetAttr(parent, "busid", &busId));  // 获取网卡的 busId
  // 对网卡的 busId hash 获得 pciId，fallback 采用 guid
  // If we fail to find the PCIe path, we use the GUID instead.
  if (busId) eatHash(hacc, busId, strlen(busId));
  else eatHash(hacc, &net->net.asic);
  net->net.pciId = digestHash(hacc);

  // 添加 nic 和 net 顶点之间的双向 links
  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

// 把一个 <net gin=1> XML 节点落成一个 GIN 顶点（type=GIN），父 NIC 顶点，但只填 dev 和 bw 两个字段，省掉了一系列属性。
ncclResult_t ncclTopoAddGin(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, GIN, netId));  // 创建 gin 空顶点
  net->net.dev = dev;

  // 计算带宽 bw
  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;

  // 建立 nic 和 gin 顶点双向 links
  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

// 把一个 <net rma=1> XML 节点落成一个 RMA 顶点（type=RMA）。RMA = Remote Memory Access（远程/单边内存访问），也是 GIN 兜底后端的底座（GIN-over-RMA）。
ncclResult_t ncclTopoAddRma(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, RMA, netId));
  net->net.dev = dev;

  int mbps;
  NCCLCHECKNOWARN(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0), NCCL_GRAPH);
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.bw = mbps / 8000.0;

  // 建立 nic 和 rma 顶点双向 links
  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.bw));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.bw));
  return ncclSuccess;
}

// 将 <nic> 下的 <net> 节点都转换为 net/gin/rma 图顶点，并都建立与 nic 顶点的双向 links
ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic,
                            int systemId) {
  for (int s = 0; s < xmlNic->nSubs; s++) {  // 遍历 <nic> 下的子节点
    struct ncclXmlNode* xmlNet = xmlNic->subs[s];
    if (strcmp(xmlNet->name, "net") != 0) continue;  // 只处理 <net> 节点
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));  // 获取 <net> 节点的 dev 属性索引，没有该属性就跳过该节点
    // This means that the "dev" attribute wasn't set on this net xml node. That means it should not be added to
    // the system topology graph
    if (index == -1) continue;

    // 向后兼容性：没有“net”属性的 net 是网络设备，没有“gin”属性的 net 则不是 gin 设备
    // Backward compatibility: net withouh "net" attr is a net dev, net without a "gin" is not a gin dev
    int net = 0, gin = 0, rma = 0;
    NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "net", &net, 1));
    NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "gin", &gin, 0));
    NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "rma", &rma, 0));
    // 分别添加 NET、GIN、RMA 类型的顶点，迁移 dev/bw/asic/pciId/port/gdr/maxconn/coll/rail/plane 属性值并与 nic 顶点连接双向 links
    if (net) NCCLCHECK(ncclTopoAddNet(xmlNet, system, nic, systemId));
    if (gin) NCCLCHECK(ncclTopoAddGin(xmlNet, system, nic, systemId));
    if (rma) NCCLCHECK(ncclTopoAddRma(xmlNet, system, nic, systemId));
  }
  return ncclSuccess;
}

// 为一个新创建的 GPU 顶点填写属性
ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cudaCompCap));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &gpu->gpu.mloPart, NCCL_TOPO_UNDEF));
  // Do not go any further, nvlinks will be added in a second pass
  return ncclSuccess;
}

// 从一个 <pci> XML 节点上读出 vendor、device、subsystem_vendor、subsystem_device 四个属性，各占 16 位，拼成一个 uint64_t
static ncclResult_t ncclTopoGetIntDevice(struct ncclXmlNode* xmlPci, uint64_t* device) {
  uint64_t ret = 0;
  const char* str;
  /*
  一个前提是这 4 个属性都不超过 16 位，示例：
  <pci busid="0000:3b:00.0" class="0x030200" vendor="0x10de" device="0x2330"
    subsystem_vendor="0x10de" subsystem_device="0x1234"/>
  最终 *device = 0x10de233010de1234。
  */
  NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
  if (str) ret += strtol(str, NULL, 0) << 48;  // 自动识别进制
  NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
  if (str) ret += strtol(str, NULL, 0) << 32;
  NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
  if (str) ret += strtol(str, NULL, 0) << 16;
  NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
  if (str) ret += strtol(str, NULL, 0);
  *device = ret;
  return ncclSuccess;
}

#define PCI_BRIDGE_DEVICE_CLASS "0x060400"

struct kvDict kvDictPciClass[] = {{PCI_BRIDGE_DEVICE_CLASS, PCI},
                                  {"0x080100", /*CX8 data direct*/ PCI},
                                  {PCI_NVSWITCH_CLASS, NVS},
                                  {"0x068001", CPU},
                                  {"0x03", GPU},
                                  {"0x02", NIC},
                                  {NULL, PCI /* Default fallback value */}};
struct kvDict kvDictPciGen[] = {{"2.5 GT/s", 15},
                                {"5 GT/s", 30},
                                {"8 GT/s", 60},
                                {"16 GT/s", 120},
                                {"32 GT/s", 240}, /* Kernel 5.6 and earlier */
                                {"2.5 GT/s PCIe", 15},
                                {"5.0 GT/s PCIe", 30},
                                {"8.0 GT/s PCIe", 60},
                                {"16.0 GT/s PCIe", 120},
                                {"32.0 GT/s PCIe", 240},
                                {"64.0 GT/s PCIe", 480},
                                {NULL, 60 /* Default fallback */}}; // x100 Mbps per lane

// 获取属于同一物理 GPU 但不同 mlopart 分区的 DEV 图顶点数量
// Return all DEV nodes whose id matches baseId with mlopart bits masked out.
ncclResult_t ncclTopoGetDevNodes(struct ncclTopoSystem* system, int64_t baseId, struct ncclTopoNode** nodes,
                                 int* nNodes) {
  *nNodes = 0;
  int64_t maskedBase = baseId & ~(int64_t)NCCL_TOPO_MLOPART_MASK;  // 后两位标识 mlopart 分区号变为 0
  for (int d = 0; d < system->nodes[DEV].count; d++) {  // 遍历所有 DEV 顶点
    struct ncclTopoNode* dev = &system->nodes[DEV].nodes[d];
    if ((dev->id & ~(int64_t)NCCL_TOPO_MLOPART_MASK) == maskedBase) {  // 判断与当前 DEV 属于相同物理 GPU
      if (*nNodes == NCCL_TOPO_MLOPART_DEV_MAX) {  // 相同物理卡的 DEV，即 mlopart 分区数不能大于 2
        WARN("ncclTopoGetDevNodes : too many DEV nodes for busId %lx (max %d)", baseId, NCCL_TOPO_MLOPART_DEV_MAX);
        return ncclInternalError;
      }
      if (nodes) nodes[(*nNodes)] = dev;  // 返回给数组
      (*nNodes)++;
    }
  }
  return ncclSuccess;
}

/*
校验 busId & NCCL_TOPO_MLOPART_MASK(0x3) 必须为 0，否则 报错，因为低 2 位要被征用来编码分区号，
若原始 busId 这两位非 0 就会冲突（GPU 的 PCI function 通常是 0，低 2 位天然空闲，所以正常满足）。
*/
static ncclResult_t ncclTopoCheckMloPartBusId(int64_t busId) {
  // check that the bits used for mlopart information are free and always 0 to avoid collision.
  if (busId & NCCL_TOPO_MLOPART_MASK) {
    WARN(
      "BusId 0x%lx has non-zero bits in MLOPart mask 0x%llx, cannot encode MLOPart partition index without collision",
      busId, (long long)NCCL_TOPO_MLOPART_MASK);
    return ncclInternalError;
  }
  return ncclSuccess;
}

// 默认启用 mlopart 分区
NCCL_PARAM(TopoSplitMlopart, "TOPO_SPLIT_MLOPART", 1);

/*
把一个 <gpu> XML 节点落成拓扑顶点，建立 DEV（物理设备/分区）+ GPU（逻辑参与者）两类顶点，并连上三种边。
它是 ncclTopoNode 里 gpu.parent/dev.nGpus/mloPart 字段、以及 ncclTopoCreateNode 工厂的真正落地处，
也是 MLOPart"一个物理 GPU 拆成多个逻辑分区"建模的核心。调用者是 ncclTopoAddPci，遍历 <pci> 子树时对每个 <gpu> 调一次。
*/
static ncclResult_t ncclTopoAddGpuSub(struct ncclXmlNode* xmlPci, struct ncclXmlNode* xmlGpu,
                                      struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId,
                                      int64_t busId, float bw) {
  int mloPart = 0, sm = 0;
  int64_t devBusId = busId;
  NCCLCHECK(ncclTopoCheckMloPartBusId(busId));  // 检验 GPU busId 低两位为 0 是符合预期的
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &mloPart, NCCL_TOPO_UNDEF));  // 获取 <gpu> 的 mlopart 属性整型
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &sm));  // 获取 sm 属性整型
  if (mloPart != NCCL_TOPO_UNDEF && ncclParamTopoSplitMlopart()) {  // 如果是分区号不为 -1 并且启用 mlopart 分区
    if (mloPart >= NCCL_TOPO_MLOPART_DEV_MAX) {  // 分区号不能大于等于 2
      WARN("MLOPart index %d out of range (max %d)", mloPart, NCCL_TOPO_MLOPART_DEV_MAX - 1);
      return ncclInternalError;
    }
    devBusId = NCCL_TOPO_MLOPART_BUSID(busId, mloPart);  // 分区设备的 busId 结合 0 或 1
  }

  struct ncclTopoNode* gpudeviceNode = NULL;
  NCCLCHECK(ncclTopoGetNode(system, &gpudeviceNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));  // 尝试获取指定 DEV 设备 id 的顶点
  if (gpudeviceNode == NULL) {  // 如果没有该 id 的 DEV 顶点，则创建
    NCCLCHECK(ncclTopoCreateNode(system, &gpudeviceNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));  // 创建 DEV 空顶点
    NCCLCHECK(ncclTopoGetIntDevice(xmlPci, &gpudeviceNode->dev.device));  // 获取 PCI device id（vendor+device+subsystem 组合）
    NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpudeviceNode->dev.cudaCompCap));  // 拷贝计算能力 CC
    NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpudeviceNode->dev.dev));  // 拷贝 GPU 设备索引
    // DEV 顶点双向指向 PCI 顶点，链路类型为 PCI，带宽为外部计算的 link_width * link_speed
    NCCLCHECK(ncclTopoConnectNodes(gpudeviceNode, parent, LINK_PCI, bw));
    NCCLCHECK(ncclTopoConnectNodes(parent, gpudeviceNode, LINK_PCI, bw));
    // 与同物理卡的其它分区 DEV 互联
    struct ncclTopoNode* sibDevs[NCCL_TOPO_MLOPART_DEV_MAX];
    int nSibDevs = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, gpudeviceNode->id, sibDevs, &nSibDevs));  // 获取与当前 DEV 属于相同物理 GPU 的所有 DEV 顶点
    for (int s = 0; s < nSibDevs; s++) {
      if (sibDevs[s] == gpudeviceNode) continue;  // 遍历到自己
      // 当前 DEV 顶点与其他同属相同物理 GPU 的 DEV 顶点双向建联，链路类型为 LOC，带宽为 2618 GB/s
      NCCLCHECK(ncclTopoConnectNodes(gpudeviceNode, sibDevs[s], LINK_LOC, MLOPART_LOC_BW));
      NCCLCHECK(ncclTopoConnectNodes(sibDevs[s], gpudeviceNode, LINK_LOC, MLOPART_LOC_BW));
    }
  }

  struct ncclTopoNode* gpuNode = NULL;
  NCCLCHECK(ncclTopoCreateNode(system, &gpuNode, GPU,
                               NCCL_TOPO_ID(systemId, NCCL_TOPO_GPU_LOCAL_ID(devBusId, gpudeviceNode->dev.nGpus))));  // 创建一个 GPU 空顶点
  NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, gpuNode));  // 填写 GPU 顶点属性：rank/sm/dev/gdr/mlopart
  gpuNode->gpu.parent = gpudeviceNode;  // parent 指向 DEV
  // GPU 顶点与对应的 DEV 顶点双向建联，链路类型为 LOC，带宽为 5000 GB/s
  NCCLCHECK(ncclTopoConnectNodes(gpudeviceNode, gpuNode, LINK_LOC, LOC_BW));
  NCCLCHECK(ncclTopoConnectNodes(gpuNode, gpudeviceNode, LINK_LOC, LOC_BW));
  gpudeviceNode->dev.nGpus++;  // DEV 顶点计数 +1，表示有不同 rank 使用相同的 DEV
  return ncclSuccess;
}

/*
处理一个 <pci> XML 节点的递归函数，先读三样东西（class→顶点类型、busid→int64、link_width/speed→带宽），
再按设备种类把这条 PCI 落成 GPU / NIC / PCI switch 三态之一的顶点，递归建子树，最后用 LINK_PCI 把新顶点连到 parent。
它由 ncclTopoAddCpu 对每个 <pci> 子节点发起，是整棵 PCIe 树的建造主力。
*/
ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent,
                            int systemId, int numaId) {
  const char* str;

  /*
  获取 pci 的 class 属性，并映射到顶点类型：
  class 值	                              type	  含义
  0x060400(PCI_BRIDGE_DEVICE_CLASS)	      PCI	    PCIe bridge/switch
  0x080100	                              PCI	    ConnectX-8 data direct
  PCI_NVSWITCH_CLASS	                    NVS	    NVSwitch
  0x068001	                              CPU	    特殊 bridge
  0x03(前缀)	                             GPU	   display/3D controller
  0x02(前缀)	                             NIC	   network controller
  NULL(兜底)	                             PCI	   识别不了当普通 PCI
  */
  int type;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
  NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

  // 获取 budId 属性字符串转换为整型
  int64_t busId;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  NCCLCHECK(busIdToInt64(str, &busId));

  /*
  link_width × link_speed -> 带宽 bw，把速度字符串换成档位（x100 Mbps per lane）：
  ink_speed	        档位 speed	   PCIe 代
  2.5 GT/s	        15	          Gen1
  5 GT/s	          30	          Gen2
  8 GT/s	          60	          Gen3
  16 GT/s	          120	          Gen4
  32 GT/s	          240	          Gen5
  64.0 GT/s PCIe	  480	          Gen6
  兜底	             60	           (当 Gen3)
  */
  float bw;
  {
    int width, speed;
    NCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));  // 获取 link_width 属性整型
    NCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));  // 获取 link_speed 属性字符串
    // Manage cases where speed was not indicated in /sys
    if (width == 0) width = 16;
    /*
    以 GT/s 为单位的 lane 速率通过折合 Gen1/2(8b/10b 编码效率 0.8) 和 Gen3+(128b/130b 编码效率 ≈0.985，NCCL 中当作 1) 的 编码效率，
    然后再含约 0.75 的协议开销折扣，最后得到以 100 Mbps/lane 为单位的数值。但实际上并不等于准确的物理数值，可以当作速率档位。
    */
    NCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)
    /*
    speed 单位是 100Mbps/lane = 0.1 Gbps/lane，width×speed 得到总的 0.1Gbps 数，×0.1 转 Gbps，再 ÷8 转 GB/s，×0.1÷8 = /80。
    举例 Gen4 x16：16×120/80 = 24 GB/s；Gen3 x8：8×60/80 = 6 GB/s。这是 NCCL 内部的相对带宽档（成比例即可，不追求物理精确值）。
    */
    bw = width * speed / 80.0;
  }

  // 为 <pci> 下的 <gpu> 构建图顶点和连线 links
  for (int g = 0; g < xmlPci->nSubs; ++g) {  // 遍历 <pci> 下的子节点
    if (strcmp(xmlPci->subs[g]->name, "gpu") != 0) continue;  // 仅处理 <gpu>
    struct ncclXmlNode* xmlGpu = xmlPci->subs[g];
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));  // 获取 <gpu> 下 rank 属性的索引
    if (index == -1) return ncclSuccess;
    /*
    搜索或创建 <gpu> 对应的 DEV 顶点，与 parent PCI 顶点建立双向 links，DEV 顶点间建立双向 links。
    再创建 <gpu> 对应的 GPU 顶点，parent 指向 DEV 顶点，并且与 DEV 顶点建立双向 links。
    */
    NCCLCHECK(ncclTopoAddGpuSub(xmlPci, xmlGpu, system, parent, systemId, busId, bw));
  }

  struct ncclTopoNode* node = NULL;
  struct ncclXmlNode* xmlNic = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));  // 获取当前 <pci> 下的 <nic> 节点
  if (xmlNic != NULL) {  // 为 <pci> 下的 <nic> 下的 <net> 构建图顶点和连线 links
    type = NIC;
    /*
    忽略 sub device ID，把多端口网卡并成一个 PCI 设备。localNicId 基于 busId，配合 ncclTopoGetNode 的 get-or-create，
    让同一物理网卡复用一个 NIC 顶点；多个端口在 ncclTopoAddNic 里作为该顶点下的多个 <net> 子顶点区分。
    ncclTopoAddNic 遍历 <net>，按 net/gin/rma 属性分别建 NET/GIN/RMA 顶点。
    */
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    struct ncclTopoNode* nicNode = NULL;
    // 计算网卡 ID
    int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, busId);
    int64_t id = NCCL_TOPO_ID(systemId, localNicId);
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, id));  // 在 system 中查找该 ID 的网卡顶点
    if (nicNode == NULL) {  // system 中没找到该 ID 的网卡顶点
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, id));  // 在 system 中创建该 ID 的网卡空顶点
      node = nicNode; // Connect it to parent later on
    }
    /*
    为 nic 下的所有 <net> 节点创建对应的 net/gin/rma 顶点，
    传递 dev/bw/asic/pciId/port/gdr/maxconn/coll/rail/plane 等属性，
    建立 net/gin/rma 顶点到 nic 顶点的双向 links。
    */
    NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId));
  } else if (type == PCI) {  // 识别到当前 <pci> 节点是 PCIe bridge/switch
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));  // 创建 pci 空顶点
    // 通过 vendor、device、subsystem_vendor、subsystem_device 四个属性构成 device
    NCCLCHECK(ncclTopoGetIntDevice(xmlPci, &node->pci.device));

    for (int s = 0; s < xmlPci->nSubs; s++) {  // 遍历 <pci> 下的子节点
      struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
      if (strcmp(xmlSubPci->name, "pcilink") != 0) {  // 跳过所有 <pcilink> 节点，剩下 <pci>
        // PCI links will be added later
        // 递归调 ncclTopoAddPci 建子树，处理嵌套的 <pci>、<gpu>、<nic>
        NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
      }
    }
  }

  /*
  ncclTopoAddPci 函数会传入 xmlPci 当前需要处理的 <pci> 节点，以及 parent 该 <pci> 节点更上一级的顶点。
  如果 <pci> 节点对应的是一个 gpu，则会直接创建 DEV 顶点并双向链接到 parent 顶点。
  如果 <pci> 节点下是网卡 <nic> 和 <pci>，则会创建 nic 和 pci 顶点，需要单独与 parent 建立双向 links。
  */
  if (node) {
    NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, bw));
    NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, bw));
  }
  return ncclSuccess;
}

struct kvDict kvDictCpuArch[] = {
  {"x86_64", NCCL_TOPO_CPU_ARCH_X86}, {"arm64", NCCL_TOPO_CPU_ARCH_ARM}, {"ppc64", NCCL_TOPO_CPU_ARCH_POWER}, {NULL, 0}
};
struct kvDict kvDictCpuVendor[] = {{"GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL},
                                   {"AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD},
                                   {"CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN},
                                   {"  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN},
                                   {NULL, 0}};

// 获取当前 CPU 节点所处 host 的索引
ncclResult_t ncclGetSystemId(struct ncclTopoSystem* system, struct ncclXmlNode* xmlCpu, int* systemIdPtr) {
  const char* hostHashStr;
  NCCLCHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));  // 获取当前 host hash
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;  // 字符串按十六进制转整型
  int systemId;
  for (systemId = 0; systemId < system->nHosts; systemId++) {  // 遍历匹配 host hash 的 host 索引
    if (system->hostHashes[systemId] == hostHash) break;
  }
  if (systemId == system->nHosts) system->hostHashes[system->nHosts++] = hostHash;  // 没匹配到就保存当前 host 的 host hash 并返回新增索引
  *systemIdPtr = systemId;
  return ncclSuccess;
}

/*
处理一个 <cpu> 子树，CPU 顶点其实是一个 NUMA 域，一台机器有几个 NUMA 就有几个 <cpu>。
函数做四件事：建 CPU 顶点、填 affinity/arch/vendor/model、递归建其下 <pci> 树、直挂 <nic>。
*/
ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
  int numaId;
  NCCLCHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));  // 获取当前 CPU 对应的 numa node id 属性
  int systemId;
  // 获取当前 <cpu> 所处 host 的索引
  NCCLCHECK(ncclGetSystemId(system, xmlCpu, &systemId));
  struct ncclTopoNode* cpu;
  // 初始化一个 CPU 类型的空的图顶点
  NCCLCHECK(ncclTopoCreateNode(system, &cpu, CPU, NCCL_TOPO_ID(systemId, numaId)));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));  // 获取亲和的 cpu mask 属性
  if (str != NULL) {
    // 亲和的 cpu mask 字符串转换成 glibc 的 cpu_set_t 数据类型
    NCCLCHECK(ncclStrToCpuset(str, &cpu->cpu.affinity));
  }

  NCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));  // 获取 CPU 架构属性
  // 将 CPU 架构字符串映射为整型
  NCCLCHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {  // 如果 CPU 架构是 x86_64
    NCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));  // 获取 CPU 厂商属性
    NCCLCHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));  // 将 CPU 厂商字符串转换为整型
    /*
    给 Intel 和兆芯的 x86 CPU 识别"微架构代际"：把 CPUID 的 familyId/modelId 号翻译成一个粗粒度的档位枚举存进 cpu->cpu.model，
    供下游 ncclTopoGetInterCpuBw 估算跨 NUMA 互连带宽（Intel 各代 UPI/QPI 速率差好几倍，必须分档）。

    只有 Intel/兆芯分 model，其他的带宽是单一常量，无需分档：
      AMD：统一 AMD_BW=16.0（不按 Zen 代际细分）
      ARM：ARM_BW=6.0
      IBM Power（即 ppc64）：P9_BW=32.0
    */
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {  // 如果 x86_64 CPU 的厂商为英特尔
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      /*
      familyId==6 是 Intel Xeon 服务器 CPU 的经典 family（P6 微架构以来都在 family 6），
      真正区分代际的是 modelId，用"从大到小 >=` 阈值"归成 4 档。中间代际被就近归档：
      Cascade Lake/Cooper Lake（model 0x55）、Ice Lake-SP（model 0x6A）都落在 [0x55, 0x8F) 区间，
      统一归 SKL 档。Broadwell-EP（0x4F）及更早落到 else 的 BDW 档，这是粗粒度归并。
        条件	                cpu.model 枚举	   对应 Intel 代号	          带宽常量	    值(GB/s)
        fam6 && model>=0xCF	 INTEL_ERP	       Emerald Rapids (2023)	   ERP_QPI_BW	  40.0
        fam6 && model>=0x8F	 INTEL_SRP	       Sapphire Rapids (2021)	   SRP_QPI_BW	  22.0
        fam6 && model>=0x55	 INTEL_SKL	       Skylake-SP (2017)	       SKL_QPI_BW	  10.0
        else	               INTEL_BDW	       Broadwell 及更早	          BDW_QPI_BW	 6.0
      */
      cpu->cpu.model = (familyId == 6 && modelId >= 0xCF) ? NCCL_TOPO_CPU_MODEL_INTEL_ERP :
                       (familyId == 6 && modelId >= 0x8F) ? NCCL_TOPO_CPU_MODEL_INTEL_SRP :
                       (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_MODEL_INTEL_SKL :
                                                            NCCL_TOPO_CPU_MODEL_INTEL_BDW;
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {  // 如果 x86_64 CPU 的厂商为兆芯
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      /*
      兆芯（国产 x86，VIA 授权）只认一个特例，family 7 && model 0x5B(91) -> YONGFENG（永峰代际，带宽 YONGFENG_ZPI_BW=9.0）。
      其余兆芯保持 model 默认值，走 ZPI_BW=6.0。注意兆芯这里不设 else，不匹配永峰就保持初始化的默认（NCCL_TOPO_UNDEF），下游按"非永峰"处理。
      */
      if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_MODEL_YONGFENG;
    }
  }
  for (int s = 0; s < xmlCpu->nSubs; s++) {  // 遍历 <cpu> 节点的子节点
    struct ncclXmlNode* node = xmlCpu->subs[s];
    // 添加 <cpu> 下的 <pci> 子树到 topo 图
    if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu, systemId, numaId));
    if (strcmp(node->name, "nic") == 0) {
      // 非 PCI 网卡，USB 网卡、虚拟/软件网卡、板载网卡等没有 PCIe 位置（没有 busId）的设备。它们无法挂进 PCIe 树，只能直接挂到 CPU（NUMA 域）下。
      struct ncclTopoNode* nic = NULL;
      int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, 0);  // busId 显式传 0
      int64_t id = NCCL_TOPO_ID(systemId, localNicId);
      NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, id));  // 查询有没有同 id 的顶点
      if (nic == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, id));  // 创建 nic 空顶点
        // 建立 nic 与 cpu 顶点的双向 links
        NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_BW));
        NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_BW));
      }
      // 将 <nic> 下的子树转换为 nic 顶点关联的子图
      NCCLCHECK(ncclTopoAddNic(node, system, nic, systemId));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetGpuDevNode(struct ncclXmlNode* xmlGpu, const char* busId, int systemId,
                                          struct ncclTopoSystem* system, struct ncclTopoNode** devNode) {
  int mloPart, sm;
  int64_t rawBusId;
  // 复现建图顶点阶段 ncclTopoAddGpuSub 的 devBusId 编码
  NCCLCHECK(busIdToInt64(busId, &rawBusId));
  NCCLCHECK(ncclTopoCheckMloPartBusId(rawBusId));
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &mloPart, NCCL_TOPO_UNDEF));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &sm));
  int64_t devBusId =
    (mloPart != NCCL_TOPO_UNDEF && ncclParamTopoSplitMlopart()) ? NCCL_TOPO_MLOPART_BUSID(rawBusId, mloPart) : rawBusId;
  NCCLCHECK(ncclTopoGetNode(system, devNode, DEV, NCCL_TOPO_ID(systemId, devBusId)));  // 获取对应 ID 的 DEV 顶点
  return ncclSuccess;
}

// 判断当前这个 <gpu> XML 节点，是不是共享同一个 DEV 顶点的那一组 gpu 里、XML 顺序上的第一个，确保同一个物理 DEV 的 NVLink/C2C 拓扑只被建一次。
// Return true only for the first GPU sibling (in XML order) that shares the same DEV node.
static ncclResult_t ncclTopoXmlIsPrimaryGpuForDev(struct ncclXmlNode* xmlGpu, bool* isPrimary) {
  *isPrimary = true;  // 假设当前是第一个
  int myMloPart = NCCL_TOPO_UNDEF;
  NCCLCHECK(xmlGetAttrIntDefault(xmlGpu, "mlopart", &myMloPart, NCCL_TOPO_UNDEF));  // 获取 mlopart 属性整型（缺失为 UNDEF）
  for (int s = 0; s < xmlGpu->parent->nSubs; s++) {  // 遍历同父 <pci> 下的所有子节点
    struct ncclXmlNode* sib = xmlGpu->parent->subs[s];
    if (strcmp(sib->name, "gpu") != 0) continue;  // 只关注 <gpu> 节点
    if (sib == xmlGpu) break;  // 遍历到自己，说明前面没有同 DEV 兄弟，保持 true 退出循环
    int sibMloPart = NCCL_TOPO_UNDEF;
    NCCLCHECK(xmlGetAttrIntDefault(sib, "mlopart", &sibMloPart, NCCL_TOPO_UNDEF));  // 获取当前 <gpu> 的 mlopart 属性整型
    // 同 DEV 判断，没开启 MLOPart 或者 mlopart 分区索引相等
    if (sibMloPart == myMloPart || !ncclParamTopoSplitMlopart()) {
      *isPrimary = false;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

/*
递归遍历整棵 XML，把每个 <nvlink> 标签翻成图里的 LINK_NVL 边。它的视角是从 <gpu> 出发，
每条 <nvlink> 挂在某个 <gpu> 下，描述本端 GPU 经 count 条 NVLink 连到 tclass 类型的对端。
*/
ncclResult_t ncclTopoAddNvLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                                int systemId) {
  if (strcmp(node->name, "nvlink") == 0) {  // 当前节点如果是 nvlink
    bool isPrimary;
    /*
    判断当前 <gpu> 节点是不是同 DEV 顶点（mlopart 分区）下的第一个 <gpu>。
    一个物理卡（DEV 顶点）可能被同一 <pci> 下的多个 <gpu> 兄弟引用（MLOPart 分区，或多 rank 共享同一物理卡）。
    但 NVLink 是物理卡级属性，只能建一次，否则重复建边会让 ncclTopoConnectNodes 的 bw +=` 把带宽算翻倍。
    */
    NCCLCHECK(ncclTopoXmlIsPrimaryGpuForDev(node->parent, &isPrimary));
    if (!isPrimary) return ncclSuccess;  // 避免重复建立 NVLlink 边
    struct ncclTopoNode* devNode = NULL;
    NCCLCHECK(ncclTopoGetGpuDevNode(node->parent, parentBusId, systemId, system, &devNode));  // 获取该 <gpu> 对应的 DEV 图顶点
    if (devNode == NULL) {
      WARN("Add NVLink error : could not find DEV for GPU %s", parentBusId);
      return ncclInternalError;
    }
    int localDevsCount = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, devNode->id, NULL, &localDevsCount));  // 获取属于同一物理 GPU 但不同 mlopart 分区的 DEV 图顶点数量
    int count, targetType;
    const char* targetClass;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));  // <nvlink> 节点的 count 属性整型代表几条 NVLink
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));  // <nvlink> 节点的 tclass 属性字符串代表对端设备类型
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));  // 对端设备类型编码转换为类型编号
    float nvlBw = ncclTopoNVLinkBw(devNode->dev.cudaCompCap);  // 根据 CC 返回 NVLink 单 link 带宽

    if (targetType == GPU) {  // 对端是 GPU 通过 NVLink 直连
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));  // 获取对端 busId，不带 mlopart
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      int remDevsCount = 0;
      struct ncclTopoNode* remDevs[NCCL_TOPO_MLOPART_DEV_MAX];
      NCCLCHECK(ncclTopoGetDevNodes(system, NCCL_TOPO_ID(systemId, busId), remDevs, &remDevsCount));  // 获取对端分区数
      // Bandwidth is split between the different devices both at source and destination.
      for (int j = 0; j < remDevsCount; j++) {
        // 当前 DEV 顶点连向对端所有 DEV 顶点，link 带宽要按照 localDevsCount * remDevsCount 条 links 划分
        NCCLCHECK(ncclTopoConnectNodes(devNode, remDevs[j], LINK_NVL, count * nvlBw / remDevsCount / localDevsCount));
      }
    } else {
      struct ncclTopoNode* remote = NULL;
      if (targetType == CPU) {  // 对端是 CPU
        // POWER9 NVLink C2C，上溯找本端 CPU 顶点
        NCCLCHECK(findLocalCpu(devNode, &remote, NULL));
      } else {
        if (system->nodes[NVS].count == 0) {  // 对端是 NVSwitch (NVS)
          NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));  // 创建一个 NVS 空顶点
        } else {
          remote = system->nodes[NVS].nodes;  // 找到 NVS 类型的节点数组的第一个 NVS 顶点
        }
      }
      // DEV 顶点与远端 CPU 或 NVS 顶点建立双向 links，链路类型为 NVL，链路带宽为总带宽按 localDevsCount 划分
      if (remote) {
        NCCLCHECK(ncclTopoConnectNodes(devNode, remote, LINK_NVL, count * nvlBw / localDevsCount));
        NCCLCHECK(ncclTopoConnectNodes(remote, devNode, LINK_NVL, count * nvlBw / localDevsCount));
      }
    }
  } else {  // 非 <nvlink> 节点
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));  // 遇 <cpu> 更新并获取 systemId
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));  // 读当前节点的 busid（软读，可能 NULL）
    for (int s = 0; s < node->nSubs; s++) {  // 递归所有子节点
      // busId 做三态继承，例如 <gpu> 节点没有 budid，会继承最近的那个 <pci> 祖先的 busid
      NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

// 把 <pcilink> 翻成 LINK_LOC 边，建模 Broadcom（BCM）PCIe Switch 下游设备之间的 P2P 直通。
ncclResult_t ncclTopoAddPciLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                                 int systemId) {
  if (strcmp(node->name, "pcilink") == 0) {
    // parentBusId 来自递归下传，<pcilink> 是 <pci busid=X> 的子节点，自身无 busid，于是继承 X，正好是所属 switch 的 busId。
    struct ncclTopoNode* pci = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &pci, PCI, pBusId));  // 获取 switch 对应的 PCI 顶点
    if (pci == NULL) {
      WARN("Add PCI Link error : could not find PCI SW %lx", pBusId);
      return ncclInternalError;
    }
    struct ncclTopoNode* remote = NULL;
    const char* target;
    NCCLCHECK(xmlGetAttrStr(node, "target", &target));  // 获取 pcilink 的对端设备 busId
    int64_t busId;
    NCCLCHECK(busIdToInt64(target, &busId));
    NCCLCHECK(ncclTopoGetNode(system, &remote, PCI, NCCL_TOPO_ID(systemId, busId)));  // 获取对端设备 PCI 顶点
    /*
      BCM 驱动列出的 peers 是该 switch 下所有可 P2P 直通的设备，其中可能有些设备根本不在本次拓扑 XML 里（未被探测、被过滤、或属另一 host），
    为这些 peer 建不出边是正常情况，不该让整个拓扑构建失败。
      1. 是 LINK_LOC 而不是 LINK_PCI，LINK_PCI 是真实 PCIe 链路，带宽受 link_width × link_speed / 80 约束，语义是经 switch 的正常转发。
    LINK_LOC + LOC_BW=5000.0 是零距离/本地的建模哨兵，带宽刻意取到远超任何真实链路，让它在路径瓶颈计算（沿途取 min）中永不成为短板，
    在 ncclTopoConnectNodes 的降序插入里排最前。
      2. 单向边就够，生成侧是放射状的。一个 switch 下挂 A、B 两块 GPU 时会生成两条 pcilink，<pcilink target=A/> 和 <pcilink target=B/>，
    建出 switch→A、switch→B 两条单向边，构成 switch 到所有 peer 的星形。
      3. 只有 Broadcom（vendor 0x1000）PCIe Switch 具备这种下游设备间 P2P 直通，标准 PCIe 拓扑探测看不到（标准 PCIe 里下游设备通信必须经 
    switch 上行到 Host Bridge 再下来）。BCM 驱动通过 /sys/kernel/pci_switch_link/virtual_switch_links/<busId> 这个专有 sysfs 目录导出该 
    switch 下所有可直通的 peer busId。

    LINK_LOC 不起作用，主要是 LOC_BW=5000 GB/s，对带宽不设限。注意这里的 remote 一定也是一个 virtual switch / bridge 的 PCI 顶点，不可能是 GPU 或 NIC
    */
    if (remote) NCCLCHECK(ncclTopoConnectNodes(pci, remote, LINK_LOC, LOC_BW));
  } else {
    // 其他节点往下递归遍历
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddPciLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

// 把 <c2c> 标签翻成图里的 LINK_C2C 边，建模 Grace Hopper（GH200）超级芯片里 CPU（Grace）和 GPU（Hopper）的 NVLink-C2C 片间缓存一致互连
ncclResult_t ncclTopoAddC2c(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId,
                            int systemId) {
  if (strcmp(node->name, "c2c") == 0) {
    bool isPrimary;
    NCCLCHECK(ncclTopoXmlIsPrimaryGpuForDev(node->parent, &isPrimary));  // 确定当前 <gpu> 是不是 DEV 顶点下的第一个
    if (!isPrimary) return ncclSuccess;  // 不是第一个，去重直接返回
    struct ncclTopoNode* devNode = NULL;
    NCCLCHECK(ncclTopoGetGpuDevNode(node->parent, parentBusId, systemId, system, &devNode));  // 获取对应 DEV 顶点
    if (devNode == NULL) {
      WARN("Add C2c error : could not find DEV for GPU %s", parentBusId);
      return ncclInternalError;
    }
    int nSibDevs = 0;
    NCCLCHECK(ncclTopoGetDevNodes(system, devNode->id, NULL, &nSibDevs));  // 获取同物理 GPU 下有多少 DEV 顶点，即 mlopart 分区数

    int linkCount = 0, bw = 0;
    NCCLCHECK(xmlGetAttrInt(node, "count", &linkCount));  // 获取 C2C links 数
    NCCLCHECK(xmlGetAttrInt(node, "bw", &bw));  // 获取单 link 带宽
    // NVML_FI_DEV_C2C_LINK_GET_MAX_BW 的单位是 MB/s，这里换算为单位为 GB/s 的总带宽
    float c2cBw = (bw * linkCount) / 1000.0;
    struct ncclTopoNode* cpu = NULL;
    // XML 里没有 target，C2C 的对端永远是本端 CPU，无需指定，这与 <nvlink> 必带 target/tclass 根本不同。
    NCCLCHECK(findLocalCpu(devNode, &cpu, NULL));
    if (cpu == NULL) return ncclSuccess;  // 找不到本地 CPU 就当没这个特性，C2C 是可选增强，不影响基本拓扑

    if (nSibDevs > 1) {
      // 多 mlopart 分区创建一个 CXB（C2C bridge）顶点，CXB 桥汇聚与 CPU 顶点建联
      // Use a C2C bridge node to guarantee the total bw of the C2C link is shared between the devices.
      // Note: pBusId is the dev busId; we have checked that the 2 last bits are 0 in ncclTopoAddGpuSub
      int64_t xc2cId = devNode->id & ~(int64_t)NCCL_TOPO_MLOPART_MASK;  // 抹掉低 2 位分区位，得到物理卡基准 id
      struct ncclTopoNode* xc2cNode = NULL;
      NCCLCHECK(ncclTopoGetNode(system, &xc2cNode, CXB, xc2cId));  // 获取 CXB 节点，可能由另一个 gpu 下的 <c2c> 创建了
      if (xc2cNode == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &xc2cNode, CXB, xc2cId));  // 创建一个 CXB 空顶点
        // 建立 CXB 顶点与 CPU 顶点的双向 links，类型为 C2C，带宽为 C2C 总带宽
        NCCLCHECK(ncclTopoConnectNodes(xc2cNode, cpu, LINK_C2C, c2cBw));
        NCCLCHECK(ncclTopoConnectNodes(cpu, xc2cNode, LINK_C2C, c2cBw));
      }
      // 建立当前 DEV 顶点与 CXB 顶点的双向 links，类型为 C2C，带宽为 C2C 总带宽
      NCCLCHECK(ncclTopoConnectNodes(devNode, xc2cNode, LINK_C2C, c2cBw));
      NCCLCHECK(ncclTopoConnectNodes(xc2cNode, devNode, LINK_C2C, c2cBw));
    } else {  // 如果一个物理 GPU 只有一个 DEV 顶点，即没有 mlopart 分区
      // 直接建立当前 DEV 顶点与 CPU 顶点的双向 links，类型为 C2C，带宽为 C2C 总带宽
      NCCLCHECK(ncclTopoConnectNodes(devNode, cpu, LINK_C2C, c2cBw));
      NCCLCHECK(ncclTopoConnectNodes(cpu, devNode, LINK_C2C, c2cBw));
    }
  } else {
    // <cpu> 节点获取 systemId
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    // 获取 busid
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s = 0; s < node->nSubs; s++) {
      // 递归遍历
      NCCLCHECK(ncclTopoAddC2c(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

// 将 xml 拓扑转换为拓扑图，输入一棵 XML 树，输出一个 ncclTopoSystem 图。
ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem,
                                      const uint64_t localHostHash) {
  NCCLCHECK(ncclCalloc(topoSystem, 1));  // 分配 ncclTopoSystem 的内存
  struct ncclTopoSystem* system = *topoSystem;
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "system", &topNode));  // 找到 topo xml 中的 <system> 根节点
  if (topNode == NULL) {
    WARN("ncclTopoGetSystemFromXml: system node not found in XML");
    return ncclInternalError;
  }
  // 遍历添加 <system> 下的所有 <cpu> 节点，结束后 system->hostHashes 收集了 XML 里出现的所有节点 host
  for (int s = 0; s < topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "cpu") == 0) NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
  }

  // 确定本机是这份融合拓扑图里的第几台 host
  int systemId = 0;
  while (systemId < system->nHosts && system->hostHashes[systemId] != localHostHash) systemId++;
  system->systemId = systemId;
  if (systemId == system->nHosts) {
    WARN("localHostHash = 0x%lx not found in the list of system hostHashes", localHostHash);
    return ncclInvalidArgument;
  }

  // 把 <gpu> 下 <nvlink> 节点转换成 topo 图的 LINK_NVL 边，DEV 与 DEV/CPU/NVS 建联
  NCCLCHECK(ncclTopoAddNvLinks(topNode, *topoSystem, NULL, 0));
  // 把 <c2c> 节点转换成 topo 图的 LINK_C2C 边，DEV 与 CPU 建联
  NCCLCHECK(ncclTopoAddC2c(topNode, *topoSystem, NULL, 0));
  // 把 <pcilink> 节点转换成成 LINK_LOC 边，建模 Broadcom（BCM）PCIe Switch 下游虚拟 switch 之间的 P2P 直通。
  NCCLCHECK(ncclTopoAddPciLinks(topNode, *topoSystem, NULL, 0));

  // Gen4/Gen5，删除 Broadcom PCI Switch 的第二层虚拟 switch
  NCCLCHECK(ncclTopoFlattenBcmSwitches(*topoSystem));
  // 建立 CPU 顶点两两之间的跨 NUMA 链路 links 和带宽
  NCCLCHECK(ncclTopoConnectCpus(*topoSystem));
  // 把 PCIe 树里向上走的那条边挪到边表的最后，让后续遍历天然形成"先下行、后上行"的顺序
  NCCLCHECK(ncclTopoSortSystem(*topoSystem));

  return ncclSuccess;
}

// 哪个 rank 负责将 topo xml dump 到文件
NCCL_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

// Only set values if not already set
static ncclResult_t xmlInitAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));  // 获取指定节点某个指定属性索引
  if (index == -1) {  // 如果没有该属性，就创建该属性，并设置指定的属性值；有该属性不会修改属性值
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  }
  return ncclSuccess;
}

static ncclResult_t xmlInitAttrUint64(struct ncclXmlNode* node, const char* attrName, const uint64_t value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%f", value);
  }
  return ncclSuccess;
}

// 读一下 /sys/kernel/pci_switch_link/refresh_switch_toplogy 文件中的一个整数
ncclResult_t ncclTopoRefreshBcmP2pLinks(void) {
#ifdef NCCL_OS_LINUX
  // refresh the switch topology by reading the link below
  FILE* fp = fopen("/sys/kernel/pci_switch_link/refresh_switch_toplogy", "r");
  if (fp != NULL) {
    int tmp;
    size_t r = fread(&tmp, sizeof(tmp), 1, fp);
    if (r != 1) INFO(NCCL_GRAPH, "Failed to read refresh_switch_toplogy");
    fclose(fp);
  }
#endif
  return ncclSuccess;
}

// This is just checking for direct descendence
int ncclTopoCheckPix(ncclXmlNode* common, ncclXmlNode** nodes, int nNodes) {
  const char* tempBusId;
  // If the common parent isn't a pci switch, then this isn't PIX
  NCCLCHECK(xmlGetAttrStr(common, "busid", &tempBusId));
  if (tempBusId == NULL) return 0;  // common 必须有 busid（是 pci switch）
  TRACE(NCCL_GRAPH, "Checking pix for busid=%s", tempBusId);

  /*
  判断是否每个节点的 <net>→<nic>→<pci> 链上那个 pci 的 parent 必须正好等于 common，
  即所有 NIC 都直接挂在同一个一级 switch 下才是 PIX（一跳）。若隔了多级 switch，node->parent != common → PXB（多跳）。
  */
  // All the nodes must have a "nic" which is a parent, and then a pci node (busid) which must be a child of the
  // "common"
  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* node = nodes[i];
    if (strcmp(node->name, "net") == 0) {
      node = node->parent;
      if (node == NULL) return 0;
      if (strcmp(node->name, "nic") == 0) {
        node = node->parent;
        if (node == NULL) return 0;
        // All nodes must descend from the same first level pci switch
        if (strcmp(node->name, "pci") == 0) {
          TRACE(NCCL_GRAPH, "Comparing parent of node=%p to common=%p", node->parent, common);
          if (node->parent != common) return 0;
        }
      }
    }
  }

  return 1;
}

// 一个简单的定长数组栈实现
#define NCCL_TOPO_XML_DEPTH_MAX 256
typedef struct xmlNodeStack {
  ncclXmlNode* elems[NCCL_TOPO_XML_DEPTH_MAX];
  int tail;

  ncclXmlNode* top() {
    if (!empty()) {
      return elems[tail - 1];
    } else {
      return NULL;
    }
  }

  ncclXmlNode* pop() {
    ncclXmlNode* node = top();
    if (node) {
      tail--;
    }
    return node;
  }

  void push(ncclXmlNode* node) {
    if (tail < NCCL_TOPO_XML_DEPTH_MAX) {
      elems[tail++] = node;
    }
  }

  bool empty() {
    return tail == 0;
  }

} xmlNodeStack;

/*
沿 parent 上移到第一个 pci 祖先，若一路撞到 system 或 NULL 都没遇到 pci，就保持原值不动。
这一步专为 ncclTopoGetVNicParent 服务，它需要一个 pci 节点来承载/新建 vNic 锚点。
*/
ncclResult_t ncclFindFirstPciParent(ncclXmlNode** parent) {
  ncclXmlNode* newParent = *parent;
  while (strcmp(newParent->name, "pci") != 0) {
    newParent = newParent->parent;
    if (newParent == nullptr) return ncclSuccess;  /
    if (strcmp(newParent->name, "system") == 0) return ncclSuccess;
  }
  *parent = newParent;
  return ncclSuccess;
}

/*
本质是一个最近公共祖先（LCA）算法：给一组 XML 节点（通常是物理 NIC 的 <net> 节点），它求它们的公共祖先，
并根据"公共祖先在 PCIe 树里的深度"折算出一个拓扑距离（PATH_* 分级），出参 *path 是距离、*parent 是那个公共祖先。

关键 trick：#define PATH_PORT PATH_NVL（值 1），NIC 融合语境用不到 NVLink，于是把值 1 借给 PATH_PORT（语义="同一块物理 NIC 的不同端口"），
让它成为仅次于 PATH_LOC 的最近距离。nicPathKvList 里 "PORT"→PATH_PORT，NCCL_NET_MERGE_LEVEL 默认解析成它。
*/
// 1. Find the common parent xmlNode between the given set of nodes
ncclResult_t ncclTopoGetPath(ncclXmlNode** nodes, int nNodes, int* path, ncclXmlNode** parent) {
  // Track a stack of parents per-net node being merged
  xmlNodeStack* parents;
  NCCLCHECK(ncclCalloc(&parents, nNodes));
  // Find the common parent
  ncclXmlNode* common = NULL;

  if (nNodes == 1) {  // 当节点只有一个时，公共祖先就是当前节点，自己到自己距离 0
    common = nodes[0];
    *path = PATH_LOC;
    goto out;
  }

  /*
  对每个节点，从自己开始沿 parent 一路 push 到栈里，遇到 system 压入后停。于是每个栈栈底是自己、栈顶是 system 根。
  xmlNodeStack 是定长数组栈（NCCL_TOPO_XML_DEPTH_MAX=256，防深树溢出，push 满则丢弃）。
  */
  for (int i = 0; i < nNodes; i++) {
    ncclXmlNode* temp;
    temp = nodes[i];
    while (temp) {
      parents[i].push(temp);
      temp = strcmp(temp->name, "system") == 0 ? NULL : temp->parent;
    }
  }

  /*
  同步弹栈找 LCA，只要所有栈顶指向同一节点，它就是公共祖先，一起 pop 上移、common 不断刷新为更"深"的公共祖先，
  直到某层栈顶分叉（不再全等），说明找到了最近公共祖先
  */
  common = NULL;
  int c;
  c = 1;
  while (c && !parents[0].empty()) {
    ncclXmlNode* temp = parents[0].top();
    // 遍历所有路径的栈顶节点，判断是否都相同
    for (int i = 1; i < nNodes; i++) {
      if (!parents[i].empty()) {
        c &= (temp == parents[i].top());
      } else {
        c = 0;
        break;
      }
    }

    // 如果当前所有路径的栈顶都还相同，更新更深的公共祖先
    if (c) {
      common = temp;
      if (common == NULL) TRACE(NCCL_GRAPH, "COMMON IS NULL");
      for (int i = 0; i < nNodes; i++) {  // 所有路径都弹出一层公共祖先
        parents[i].pop();
      }
      // Check multi-port while we still have the mismatched parents
      // For multi-port to be true, all parents (peers) must have the busId attribute with all but the last character
      // matching
    } else {  // 走到这里说明目前所有路径顶层节点不再是公共祖先
      int multiPort = 1;
      const char* tempBusId;

      /*
      栈顶分叉时，检查各栈当前栈顶（分叉的那批兄弟节点）的 busId 是否"除末位字符外全同"。
      PCI 地址是 DDDD:BB:DD.F，末位就是 function 号 F。同一块双端口卡的两个口通常是 ...00.0 和 ...00.1，
      device 位相同、只有 function 位不同，于是 busId 除末位全同，判为 multiPort，距离 PATH_PORT。
      */
      NCCLCHECK(xmlGetAttr(temp, "busid", &tempBusId));
      if (tempBusId) {
        for (int i = 1; i < nNodes; i++) {
          if (!parents[i].empty()) {
            const char* busId;
            NCCLCHECK(xmlGetAttr(parents[i].top(), "busid", &busId));
            if (busId) {
              if (strlen(busId) != strlen(tempBusId)) {
                multiPort = 0;  // busId 长度不合法
                break;
              }
              if (strncmp(busId, tempBusId, strlen(busId) - 1) != 0) {
                multiPort = 0;  // 出现非 multiPort
                break;
              }
            } else {
              multiPort = 0;  // 没有获取到 busId
              break;
            }
          }
        }
      } else {
        multiPort = 0;  // 用于对比的第一条路径的顶部节点没有获取到 busId
      }

      if (multiPort) {  // 传入节点之间只有 function 的差异，它们的路径距离为 PATH_PORT
        *path = PATH_PORT;
        goto out;
      }
    }
  }

  if (common == NULL) {
    *path = PATH_DIS;  // 理论上 system 兜底，几乎不发生
  } else if (strcmp(common->name, "system") == 0) {
    *path = PATH_SYS;  // 只在根汇合→跨 NUMA
  } else if (strcmp(common->name, "cpu") == 0) {
    *path = PATH_PHB;  // 在 CPU/Host Bridge 汇合
  } else if (strcmp(common->name, "nic") == 0) {
    *path = PATH_PORT;  // 同一个 nic
  } else if (strcmp(common->name, "net") == 0) {
    *path = PATH_PORT;  // // 同一个 net
  } else if (ncclTopoCheckPix(common, nodes, nNodes)) {
    *path = PATH_PIX;  // 同一个一级 pci switch 直下
  } else {
    *path = PATH_PXB;
  }

out:
  // 把 common（可能是 nic/net 这种非 pci 节点）沿 parent 上移到第一个 pci 祖先。若一路撞到 system 或 NULL 都没遇到 pci，就保持原值不动。
  ncclFindFirstPciParent(&common);
  *parent = common;
  free(parents);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeUniqueBusId(struct ncclXml* xml, char* busId, struct ncclXmlNode** pciNode,
                                     struct ncclXmlNode* parent) {
  int i = 0;
  int64_t rBusId;
  NCCLCHECK(busIdToInt64(busId, &rBusId));
  // Try to find an unused busid - NCCL expects leaf busid to be unique
  while (i < 100) {
    rBusId++;
    TRACE(NCCL_GRAPH, "Trying to make new busId %lx", rBusId);
    int64ToBusId(rBusId, busId);
    struct ncclXmlNode* temp = NULL;
    NCCLCHECK(xmlFindTagKv(xml, "pci", &temp, "busid", busId));
    if (temp == NULL) {
      NCCLCHECK(xmlAddNode(xml, parent, "pci", pciNode));
      NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
      TRACE(NCCL_GRAPH, "Made new busId %lx", rBusId);
      return ncclSuccess;
    }
    TRACE(NCCL_GRAPH, "Conflicting busId %lx", rBusId);
    i++;
  }

  WARN("TOPO/NET : Couldn't generate unique busId after %d tries", i);
  return ncclInternalError;
}

// 在 (*parent) 下添加一个具有唯一 busId 的新 PCI 节点，并覆盖 (*parent) 的值，使其指向该新节点
// Add a new PCI node with a unique busId under (*parent) and overwrite the value of (*parent) to point to the new node
ncclResult_t ncclTopoMakePciParent(struct ncclXml* xml, struct ncclXmlNode** parent, struct ncclXmlNode* physNetNode) {
  struct ncclXmlNode* newBusId = NULL;
  struct ncclXmlNode* pci = physNetNode->parent;
  if (pci) {
    pci = pci->parent;
    if (pci) {
      if (strcmp(pci->name, "pci") == 0) {
        char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        memset(busId, 0, sizeof(busId));
        const char* originalBusId;
        // Seed busId with the current NIC 0's busId to make discovering a unique hash quicker
        NCCLCHECK(xmlGetAttrStr(pci, "busid", &originalBusId));
        snprintf(busId, sizeof(busId), "%s", originalBusId);
        NCCLCHECK(ncclTopoMakeUniqueBusId(xml, busId, &newBusId, *parent));
        for (int i = 0; i < pci->nAttrs; i++) {
          NCCLCHECK(xmlSetAttr(newBusId, pci->attrs[i].key, pci->attrs[i].value));
        }
        NCCLCHECK(xmlSetAttr(newBusId, "busid", busId));
        *parent = newBusId;
      }
    }
  }

  if (newBusId == NULL) {
    const char* name;
    NCCLCHECK(xmlGetAttr(physNetNode, "name", &name));
    WARN("TOPO/NET : Can't find busId of child 0 %s", name);
    return ncclInternalError;
  }

  return ncclSuccess;
}

// 将选择的一组物理网卡融合成一个虚拟设备
ncclResult_t ncclTopoMakeVnic(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, ncclNetVDeviceProps_t* vProps,
                              struct ncclXmlNode** physNetNodes) {
  if (vProps->ndevs > netInfo->maxDevsPerNic) {
    WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps->ndevs, netInfo->maxDevsPerNic);
    return ncclInternalError;
  }

  // Don't make vNics of size 1
  if (vProps->ndevs == 1) {  // 分组内只有一个物理网卡，不用合并
    TRACE(NCCL_GRAPH, "TOPO/NET : Skipping vNic of size 1");
    return ncclSuccess;
  }

  // Trigger the merge, then get the new device's properties
  int vDevIndex = 0;
  ncclResult_t ret;
  // 把多个物理 NIC 融合成一个虚拟设备的插件回调
  NOWARN(ret = netInfo->makeVDevice(&vDevIndex, vProps), NCCL_GRAPH | NCCL_INIT | NCCL_NET);
  if (ret != ncclSuccess) {
    INFO(NCCL_GRAPH | NCCL_INIT | NCCL_NET,
         "TOPO/NET : Tried merging multiple devices together and failed. vProps={ndevs=%d, devs=[%d %d %d %d]}. Set "
         "NCCL_NET_MERGE_LEVEL=LOC to disable NIC fusion.",
         vProps->ndevs, vProps->devs[0], vProps->devs[1], vProps->devs[2], vProps->devs[3]);
    return ret;
  }

  INFO(NCCL_GRAPH, "TOPO/NET : Made vNic %d", vDevIndex);
  return ncclSuccess;
}

ncclResult_t ncclTopoForceMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs,
                                ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs) {
  ncclResult_t ret = ncclSuccess;
  const char* str = netInfo->forceMerge;
  INFO(NCCL_ENV | NCCL_NET, "TOPO/NET : Force-fusing NICs using NCCL_NET_FORCE_MERGE=%s", str);
  char* ncStr;
  NCCLCHECK(ncclCalloc(&ncStr, strlen(str) + 1));
  strcpy(ncStr, str);
  char* semi_token;
  char* semi = strtok_r(ncStr, ";", &semi_token);  // 分割按 ; 分隔的网卡分组
  while (semi) {
    TRACE(NCCL_NET, "Fusing %s", semi);
    struct netIf userIfs[NCCL_NET_MAX_DEVS_PER_NIC];
    // 把形如 "eth0:1,eth1:2" 或 "mlx5_0:1:0:0" 的字符串解析成 struct netIf 数组
    int nUserIfs = parseStringList(semi, userIfs, NCCL_NET_MAX_DEVS_PER_NIC);
    if (nUserIfs == 0) {
      INFO(NCCL_NET,
           "NET/IB : Invalid NCCL_NET_FORCE_MERGE specified %s. Couldn't parse substring %s. Please provide a "
           "semicolon-delimited list of comma-delimited NIC groups.",
           ncStr, semi);
      continue;
    }

    ncclNetVDeviceProps_t vProps = {0};
    // 匹配所有网卡名和端口的物理网卡
    for (int d = 0; d < nPhysDevs; d++) {
      if (matchIfList(propsList[d].name, propsList[d].port, userIfs, nUserIfs, 1)) {
        vProps.devs[vProps.ndevs++] = d;
      }
    }

    if (vProps.ndevs != nUserIfs) {  // 用户指定的当前分组网卡没有全部匹配，报错
      WARN("TOPO/NET : Only matched %d devices, %d requested from %s", vProps.ndevs, nUserIfs, semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    if (vProps.ndevs > netInfo->maxDevsPerNic) {  // 要合并的分组大小大于最大设备数，报错
      WARN("Specified fused NIC %s which has too many devices (%d). Max %d", semi, vProps.ndevs,
           netInfo->maxDevsPerNic);
      ret = ncclInvalidUsage;
      goto fail;
    }

    // 真正调插件 makeVDevice 建 vNic。只有建成功后才标 placedDevs，因为这些标记会被 AutoMerge 读取跳过，若中途失败却已标记，设备就"两头落空"了
    ret = ncclTopoMakeVnic(xml, netInfo, &vProps, physNetNodes);
    if (ret == ncclSuccess) {
      // Only set that a device is "placed" after successfully making a vNic (it's possible to exit before this)
      for (int i = 0; i < vProps.ndevs; i++) {
        placedDevs[vProps.devs[i]] = 1;  // 标记已经被融合到某个 vNIC 的物理网卡
      }
    } else {
      WARN("TOPO/NET : Could not force merge NICs %s. Please specify a valid NCCL_NET_FORCE_MERGE string.", semi);
      ret = ncclInvalidUsage;
      goto fail;
    }

    semi = strtok_r(NULL, ";", &semi_token);  // 获取下一个待融合网卡的分组
  }

exit:
  free(ncStr);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTopoAutoMerge(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int* placedDevs,
                               ncclNetProperties_t* propsList, struct ncclXmlNode** physNetNodes, int nPhysDevs) {
  // Compute the path type between each device
  int* paths = NULL;
  ncclResult_t res = ncclSuccess;
  ncclCalloc(&paths, nPhysDevs * nPhysDevs);  // 所有网卡两两之间的距离矩阵
  TRACE(NCCL_GRAPH, "Allocated %d paths", nPhysDevs * nPhysDevs);
  for (int i = 0; i < nPhysDevs; i++) {
    for (int j = 0; j < nPhysDevs; j++) {
      struct ncclXmlNode* nodes[2];
      nodes[0] = physNetNodes[i];
      nodes[1] = physNetNodes[j];
      struct ncclXmlNode* parent;
      // 计算一对物理网卡之间的最近公共祖先和距离
      NCCLCHECKGOTO(ncclTopoGetPath(nodes, 2, &paths[i * nPhysDevs + j], &parent), res, out);
    }
  }

  // 根据 mergeLevel 标准，将所有剩余的物理设备放入一个虚拟设备中
  // Place all remaining physical devices into a virtual device given the mergeLevel criteria
  for (int i = 0; i < nPhysDevs; i++) {  // 遍历所有物理网卡
    // Select the first unplaced device "i" as the root
    if (placedDevs[i] == 0) {  // 当前物理网卡还没被放到某个 vNIC 中
      // Init a new vDevice
      // 先单独为当前物理网卡构建初始化一个 vNIC
      ncclNetVDeviceProps_t vProps;
      vProps = {0};
      vProps.devs[vProps.ndevs++] = i;
      placedDevs[i] = 1;
      TRACE(NCCL_GRAPH, "Placed dev %d", i);

      /*
      再次遍历所有物理网卡，选择其中没有被合并、与当前网卡的距离小于 mergeLevel、并且不是当前网卡的其他网卡。
      此外，mergePolicy 没有要求（也就是 NCCL_NET_MERGE_POLICY_ALL），或者 mergePolicy 有要求（也就是 NCCL_NET_MERGE_POLICY_RAIL）并且
      网卡 j 与当前网卡 i 同轨。满足上述条件的情况下将网卡 j 合并到当前网卡 i 的 vNIC 中。
      */
      // Select each unplaced device "j" which is at most "mergeLevel" distance from "i", but not equal to "i"
      // (Don't merge the same device with itself)
      for (int j = 0; j < nPhysDevs; j++) {
        if ((paths[i * nPhysDevs + j] <= netInfo->mergeLevel) && (placedDevs[j] == 0 && j != i) &&
            (netInfo->mergePolicy != NCCL_NET_MERGE_POLICY_RAIL ||
             (propsList[i].railId != NCCL_NET_ID_UNDEF && propsList[i].railId == propsList[j].railId))) {
          vProps.devs[vProps.ndevs++] = j;
          placedDevs[j] = 1;
          TRACE(NCCL_GRAPH, "Placed dev %d path=%d", j, paths[i * nPhysDevs + j]);
        }
        if (vProps.ndevs == netInfo->maxDevsPerNic) break;  // 如果达到合并网卡的数量上限，提前结束
      }

      if (vProps.ndevs > netInfo->maxDevsPerNic) {
        WARN("TOPO/NET : Tried to merge too many NICs. %d > %d", vProps.ndevs, netInfo->maxDevsPerNic);
        return ncclInternalError;
      }

      // // 将选择的一组物理网卡融合成一个虚拟设备
      ncclResult_t ret = ncclTopoMakeVnic(xml, netInfo, &vProps, physNetNodes);

      /*
      失败回滚，ncclTopoMakeVnic 失败（如插件拒绝合并）时，把这组全标回未放置、把 i 到各成员的距离改成 PATH_DIS（永久拉远，下次不再尝试同组）、
      i=0 重启整个循环。靠"渐进拉远距离"保证每张卡最终都有归属、不死循环。
      */
      // Merging failed.
      // Mark all as unplaced and increase their distance to disconnected (PATH_DIS)
      // Set i to 0 to restart the automatic merging process and ensure all are placed
      if (ret != ncclSuccess) {
        INFO(NCCL_GRAPH | NCCL_INIT | NCCL_NET,
             "Marking physical devices as unplaced, increasing distance and restarting search.");
        placedDevs[i] = 0;
        TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, keeping distance -> self as PATH_LOC", i);
        for (int k = 1; k < vProps.ndevs; k++) {
          int dev = vProps.devs[k];
          placedDevs[dev] = 0;
          paths[i * nPhysDevs + dev] = PATH_DIS;
          paths[dev * nPhysDevs + i] = PATH_DIS;
          TRACE(NCCL_GRAPH, "Setting dev %d as unplaced, setting distance -> %d as PATH_DIS", dev, i);
        }
        i = 0;
      }
    }
  }

out:
  free(paths);
  return res;
}

// NIC 语境相关的路径类型
// clang-format off
struct kvDict nicPathKvList[] = {
  { "LOC",  PATH_LOC },
  { "PORT", PATH_PORT },
  { "PIX",  PATH_PIX },
  { "PXB",  PATH_PXB },
  { "P2C",  PATH_P2C },
  { "PXN",  PATH_PXN },
  { "PHB",  PATH_PHB },
  { "SYS",  PATH_SYS },
  { NULL, 0 }
};
// clang-format on

ncclResult_t ncclTopoFindLinkWidthRec(ncclXmlNode* node, ncclXmlNode** physNetNodes, int ndevs, int* foundPhysNet,
                                      int* linkWidth) {
  int myLinkWidth = 0;
  // 如果是 pci 节点，获取当前节点的 link_width 属性值
  if (strcmp(node->name, "pci") == 0) {
    NCCLCHECK(xmlGetAttrInt(node, "link_width", &myLinkWidth));
#ifdef ENABLE_TRACE
    const char *busidAttr, *linkAttr;
    NCCLCHECK(xmlGetAttrStr(node, "busid", &busidAttr));
    NCCLCHECK(xmlGetAttr(node, "link_width", &linkAttr));
    TRACE(NCCL_GRAPH, "Found link_width (%s)=%d for busid=%s", linkAttr, myLinkWidth, busidAttr);
#endif
  }

  *foundPhysNet = 0;
  // Detect if a physical child is found. This information will be propagated up the stack.
  int devId = 0;
  while (devId < ndevs && !(*foundPhysNet)) *foundPhysNet = (node == physNetNodes[devId++]);

  int totalChildLinkWidth = 0;
  for (int i = 0; i < node->nSubs; i++) {
    ncclXmlNode* child = node->subs[i];
    int found = 0;
    int tempLinkWidth = 0;
    NCCLCHECK(ncclTopoFindLinkWidthRec(child, physNetNodes, ndevs, &found, &tempLinkWidth));
    if (found) {
      *foundPhysNet = 1;
      totalChildLinkWidth += tempLinkWidth;
    }
  }

  if (*foundPhysNet == 0) {
    // No child NICs were found, do not accrue any detected link_width
    *linkWidth = 0;
    TRACE(NCCL_GRAPH, "Did not find child net device. Returning link_width=%d totalChildLinkWidth=%d", *linkWidth,
          totalChildLinkWidth);
  } else if (totalChildLinkWidth == 0) {
    // If A child NIC was found but no link_width was detected among children, assign the link_width to mine (I am
    // the first pci node right above the physNetNode).
    *linkWidth = myLinkWidth;
    TRACE(NCCL_GRAPH, "Found child net device for %s. Returning link_width=%d totalChildLinkWidth=%d", node->name,
          *linkWidth, totalChildLinkWidth);
  } else {
    // Standard recursive accrual of link_width. The link_width is either the bottleneck of this PCI node's width or
    // the sum of its children's width.
    *linkWidth = myLinkWidth > 0 ? std::min(myLinkWidth, totalChildLinkWidth) : totalChildLinkWidth;
    TRACE(NCCL_GRAPH, "Found child net device for %s. Returning link_width=%d totalChildLinkWidth=%d", node->name,
          *linkWidth, totalChildLinkWidth);
  }

  return ncclSuccess;
}

// DFS over nodes under common parent
// Exclude link widths of non-physNetNodes chains
ncclResult_t ncclTopoFindLinkWidth(ncclXmlNode* parent, ncclXmlNode** physNetNodes, int ndevs, int* linkWidth) {
  *linkWidth = 0;
  // 遍历所有子节点
  for (int i = 0; i < parent->nSubs; i++) {
    ncclXmlNode* child = parent->subs[i];
    int foundPhysNet = 0;
    int childLinkWidth = 0;
    // 获取当前子节点下是否有物理网卡，并且聚合带宽是多少
    NCCLCHECK(ncclTopoFindLinkWidthRec(child, physNetNodes, ndevs, &foundPhysNet, &childLinkWidth));
    if (foundPhysNet) {
      *linkWidth += childLinkWidth;
    }
  }

  return ncclSuccess;
}

/*
一张物理 NIC 对应一个确定的 PCIe 设备（有唯一 busId），ncclTopoFillNet 能把它精确挂进 PCIe 树的某个 <pci> 下。
但一张虚拟 NIC（vNic） 代表的是好几张物理卡的聚合，它没有单一的 PCIe 位置。要让后续的图搜索能像对待普通网卡一样算它的路径/带宽，就必须：
  1. 找到这几张物理卡在拓扑树里的最近公共祖先；
  2. 判断这个聚合在拓扑上是否合法（跨 NUMA 就拒绝）；
  3. 必要时构造一个合成 PCI 父节点给 vNic 挂载；
  4. 把聚合后的总链路宽度盖到共享的 PCIe 脊柱上。
*/
ncclResult_t ncclTopoGetVNicParent(struct ncclXml* xml, ncclResult_t (*getProperties)(int, ncclNetProperties_t*),
                                   ncclNetVDeviceProps_t* vProps, ncclXmlNode** parent) {
  // 初始化的两个组都是 NCCL_NET_MAX_DEVS_PER_NIC（8 或 4，按插件版本）的定长栈数组，融合的设备数不会超过单 vNic 上限
  ncclNetProperties_t props[NCCL_NET_MAX_DEVS_PER_NIC];
  ncclXmlNode* physNetNodes[NCCL_NET_MAX_DEVS_PER_NIC];
  /*
  对每个物理设备，先获取属性，再用属性里的 name 通过 xmlFindTagKv 在 XML 里按名字回捞对应的 <net> 物理节点
  （这些节点在更早的 ncclTopoPopulateNics(virtual=false) 阶段已建好），存进 physNetNodes[]。
  */
  for (int i = 0; i < vProps->ndevs; i++) {
    NCCLCHECK(getProperties(vProps->devs[i], props + i));
    struct ncclXmlNode* physNetNode;
    NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name));
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Re-found physical ncclNet node %d %s", i, props[i].name);
  }

  int path = PATH_LOC;
  // 给一组节点，求它们在 XML 树里的最近公共祖先（LCA），并给出 path 类型
  NCCLCHECK(ncclTopoGetPath(physNetNodes, vProps->ndevs, &path, parent));
  int aggregateWidth = 0;
  /*
  只有当公共父落在 PCIe 脊柱上（同一 switch 的 PIX、多 switch 的 PXB、经 host bridge 的 PHB）时，几张卡的链路宽度才可叠加。
  ncclTopoFindLinkWidth 对 parent 的每条子链做 DFS，只累加"链下确实含被融物理 NIC"的那些子链的 link_width，排除不含这些 NIC 的旁支。
  结果 aggregateWidth = 融合后 vNic 的总 PCIe 宽度。PATH_PORT（多端口，宽度本就是单卡的）和 PATH_SYS（马上会被拒）都不进这个分支。
  */
  if (path == PATH_PHB || path == PATH_PXB || path == PATH_PIX) {
    NCCLCHECK(ncclTopoFindLinkWidth(*parent, physNetNodes, vProps->ndevs, &aggregateWidth));
  }

  // 必要时构造合成 PCI 父
  // If the common parent is a PCI switch or the CPU, we must reparent the new NIC under a made up pci device with a
  // unique busid
  // This adds a pci node between the physNetParent and the fused NIC.
  struct ncclXmlNode* physNetParent = *parent;  // 记住原始公共父，第五步当终止标记
  if (*parent) {
    if (strcmp((*parent)->name, "pci") == 0) {
      // Compare PCI class here to avoid NCCL WARN when the "class" attribute doesn't exist
      const char* c;
      NCCLCHECK(xmlGetAttrStr(*parent, "class", &c));
      if (c && strcmp(c, PCI_BRIDGE_DEVICE_CLASS) == 0) {  // "0x060400"
        /*
        pci 且 class 是 PCI bridge（switch）：不能把 vNic 直接挂在 switch 下（switch 下还挂着别的设备），
        需在 switch 与 vNic 之间插入一个专用合成 pci 节点当父。若 class 不是 bridge（本身就是 endpoint pci），则不 reparent，直接用。
        */
        NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
      }
    } else if (strcmp((*parent)->name, "cpu") == 0) {
      // 对应 PATH_PHB，vNic 是 PCI 设备概念，不能直接挂 cpu，同样调 ncclTopoMakePciParent 在 cpu 下造一个合成 pci。
      // If the common parent is a CPU, we must reparent the new NIC under a made up pci device with a unique busid
      NCCLCHECK(ncclTopoMakePciParent(xml, parent, physNetNodes[0]));
    } else if (strcmp((*parent)->name, "system") == 0) {
      // 对应 PATH_SYS，跨 NUMA，直接 WARN + return ncclInvalidArgument，不支持跨 NUMA 域融合网卡
      WARN("Fusing NET devices from different NUMA domains is not supported.");
      return ncclInvalidArgument;
    }
  }

  // Update the speed for all the pci nodes between the parent and the physNetparent
  if (aggregateWidth > 0) {
    struct ncclXmlNode* node = *parent;
    while (node) {
      TRACE(NCCL_GRAPH, "Set link_width to %d for vNIC parent %s", aggregateWidth, (*parent)->name);
      if (strcmp(node->name, "pci") == 0) NCCLCHECK(xmlSetAttrInt(node, "link_width", aggregateWidth));
      if (node == physNetParent) break;
      node = node->parent;
    }
  }

  TRACE(NCCL_GRAPH, "Selected parent %s with path %d", (*parent)->name, path);
  return ncclSuccess;
}

ncclResult_t ncclTopoMakeVNics(struct ncclXml* xml, struct ncclTopoNetInfo* netInfo, int physicalDevs) {
  int* placedDevs = NULL;
  struct ncclXmlNode** physNetNodes = NULL;
  ncclNetProperties_t* props = NULL;
  ncclResult_t res = ncclSuccess;
  if (physicalDevs == 0) return ncclSuccess;

  NCCLCHECK(ncclCalloc(&physNetNodes, physicalDevs));
  NCCLCHECK(ncclCalloc(&placedDevs, physicalDevs));
  NCCLCHECK(ncclCalloc(&props, physicalDevs));
  // 遍历所有物理设备
  for (int i = 0; i < physicalDevs; i++) {
    NCCLCHECKGOTO(netInfo->getProperties(i, props + i), res, out);  // 取当前物理设备的属性
    struct ncclXmlNode* physNetNode;
    NCCLCHECKGOTO(xmlFindTagKv(xml, "net", &physNetNode, "name", props[i].name), res, out);  // 获取当前物理设备对应的 <net> 节点
    physNetNodes[i] = physNetNode;
    TRACE(NCCL_GRAPH, "Found physical ncclNet node %d %s", i, props[i].name);
  }

  if (netInfo->forceMerge) {  // NCCL_NET_FORCE_MERGE 用户强制分组，先做
    NCCLCHECKGOTO(ncclTopoForceMerge(xml, netInfo, placedDevs, props, physNetNodes, physicalDevs), res, out);
  }
  // 自动融合（按拓扑距离贪心）
  NCCLCHECKGOTO(ncclTopoAutoMerge(xml, netInfo, placedDevs, props, physNetNodes, physicalDevs), res, out);

out:
  free(physNetNodes);
  free(props);
  if (placedDevs) free(placedDevs);
  return res;
}

/*
把一批网卡设备（物理或虚拟）逐个填成 XML 里的 <net> 节点，并打上各类属性标记，同一块物理网卡的 <net> 节点，
可能被不同后端类型（net、gin、coll…）多次访问，叠加能力属性标签。
*/
static ncclResult_t ncclTopoPopulateNics(ncclXml* xml, int startIndex, int endIndex, struct ncclTopoNetInfo* netInfo,
                                         int virtualNics) {
  // 遍历所有设备
  for (int n = startIndex; n < endIndex; n++) {
    ncclNetProperties_t props;
    /*
    获取设备的属性，包括 name、pciPath、guid、speed、port、latency、maxComms、ptrSupport、railId、planeId，
    以及 vNic 的 vProps（融合了哪几块物理卡）。
    */
    NCCLCHECK(netInfo->getProperties(n, &props));
    struct ncclXmlNode* netNode = NULL;
    struct ncclXmlNode* parent = NULL;
    // 仅虚拟网卡走这段。一块 vNic 是多块物理网卡融合成的逻辑设备，它在 PCIe 树里该挂哪儿？答案是挂在这些物理卡的最近公共祖先上。
    if (virtualNics) {
      // 多线程防护：多线程场景下，若该 net 已被别的线程创建（xmlFindTagKv 命中），就不再重新发现父节点。
      struct ncclXmlNode* net = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &net, "name", props.name));
      // 遍历 props.vProps（被融合的物理设备列表）算出这个共享父节点，存进 parent。
      // In the event of multithreaded use case, we need to re-discover the shared parent of the given devices for
      // this vNIC
      // Only run this if the net doesn't exist locally - this may alter the XML state
      if (net == NULL) NCCLCHECK(ncclTopoGetVNicParent(xml, netInfo->getProperties, &props.vProps, &parent));
    }

    /*
    ncclTopoFillNet 与 GPU 侧的 ncclTopoFillGpu 对称，按 pciPath 向上建 <pci> 祖先链，并定位到 <cpu>，
    在其下 find-or-create <nic>/<net> 节点，出参 netNode。若传入了 parent（vNic 情形），强制挂到那个共享父节点下。
    */
    NCCLCHECK(ncclTopoFillNet(xml, "net", props.pciPath, props.name, &netNode, parent));

    const char* colAttr;
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));  // 获取 <net> 节点上的 coll 属性，用于输出日志

    // <net> 节点打上 keep=1 属性，标记这个节点要在后续阶段 ncclTopoTrimXml 中存活（裁剪只删不含 keep=1 的分支）
    NCCLCHECK(xmlSetAttrInt(netNode, "keep", 1));
    int dev;
    /*
    先用 xmlGetAttrIntDefault 读回 dev 属性旧值（缺省 -1）。若 NCCL_TOPO_FILE 预置过一个不同的 dev 索引，
    就 INFO 提示"正在把设备索引从旧值改成新值"，然后强制写成本地下标 n，因为设备索引是本进程的权威值，必须以本地枚举为准。
    */
    xmlGetAttrIntDefault(netNode, "dev", &dev, -1);
    if (dev != -1 && dev != n) {
      INFO(NCCL_GRAPH, "TOPO/NET : Changing %s dev index from %d to %d", netInfo->name, dev, n);
    }
    NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));  // 无论有没有 dev 属性最后都会设置值为 n 的 dev 属性
    /*
    只有属性不存在时才填写：延迟、速率、端口、rail、plane、guid、最大连接数等能力属性
    */
    NCCLCHECK(xmlInitAttrInt(netNode, "latency", props.latency));
    NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
    NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
    if (props.railId != NCCL_NET_ID_UNDEF) NCCLCHECK(xmlInitAttrInt(netNode, "rail", props.railId));
    if (props.planeId != NCCL_NET_ID_UNDEF) NCCLCHECK(xmlInitAttrInt(netNode, "plane", props.planeId));
    NCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
    NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
    /*
    GDR 支持有两条获取路径，满足任一即可：
      1. 网卡直接支持 CUDA 指针（NCCL_PTR_CUDA）；
      2. 全局 dmabuf 支持（netInfo->dmaBufSupport）且 网卡支持 DMABUF（NCCL_PTR_DMABUF）
    */
    bool gdrSupport =
      (props.ptrSupport & NCCL_PTR_CUDA) || (netInfo->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF));
    INFO(NCCL_NET, "NET/%s : GPU Direct RDMA %s for HCA %d '%s'", netInfo->name, gdrSupport ? "Enabled" : "Disabled", n,
         props.name);
    NCCLCHECK(xmlInitAttrInt(netNode, "gdr", gdrSupport));  // 同样尊重文件预设

    // GIN and COLL plugins must set "net=0"; it's absence is interpretted as "net=1" in ncclTopoAddNic.
    int isNet = 0;
    const char* netAttr = NULL;
    NCCLCHECK(xmlGetAttr(netNode, "net", &netAttr));  // 先读出该节点已有的 net 值
    if (netAttr) isNet = strtol(netAttr, NULL, 0);
    /*
    与"本轮后端是否 net 类型"做或，这样做的意义是一块物理网卡可能既被 net 后端用、又被 gin/coll 后端用，
    多轮访问中 net 标志一旦置 1 就不会被后续轮次清掉（靠 || isNet 保住旧值）。
    */
    NCCLCHECK(xmlSetAttrInt(netNode, "net", netInfo->net || isNet));
    /*
    coll/gin/rma：只有当本轮后端正是该类型时（netInfo->coll 等为真）才用 xmlInitAttrInt 打上对应标记（缺失才写）。
    net 即使是 0 也会填写属性 <net net=0>
    */
    // Only set coll or gin if it's not 0
    if (netInfo->coll) NCCLCHECK(xmlInitAttrInt(netNode, "coll", netInfo->coll));
    if (netInfo->gin) NCCLCHECK(xmlInitAttrInt(netNode, "gin", netInfo->gin));
    if (netInfo->rma) NCCLCHECK(xmlInitAttrInt(netNode, "rma", netInfo->rma));

    // 最终一块网卡的这几个标志，就表达出"它服务于哪几类后端"，是后续建图时区分通用网络 / GPU 发起网络 / 网内集合卸载的依据
    const char *keepAttr, *ginAttr, *rmaAttr;
    NCCLCHECK(xmlGetAttr(netNode, "net", &netAttr));
    NCCLCHECK(xmlGetAttr(netNode, "gin", &ginAttr));
    NCCLCHECK(xmlGetAttr(netNode, "rma", &rmaAttr));
    NCCLCHECK(xmlGetAttr(netNode, "coll", &colAttr));
    NCCLCHECK(xmlGetAttr(netNode, "keep", &keepAttr));
    INFO(
      NCCL_GRAPH,
      "ncclTopoPopulateNics : Filled %s in topo with pciPath=%s net=%s gin=%s rma=%s keep=%s coll=%s rail=%d plane=%d",
      props.name, props.pciPath, netAttr, ginAttr, rmaAttr, keepAttr, colAttr, props.railId, props.planeId);
  }

  return ncclSuccess;
}

static ncclResult_t ncclTopoUpdateVNics(ncclXml* xml, struct ncclTopoNetInfo* net, int nPhysicalNics,
                                        int nVirtualNics) {
  for (int n = nPhysicalNics; n < nPhysicalNics + nVirtualNics; n++) { // 遍历每个虚拟网卡
    ncclNetProperties_t vProps;
    NCCLCHECK(net->getProperties(n, &vProps));
    for (int i = 0; i < vProps.vProps.ndevs; i++) {  // 遍历它融合的每个物理子设备
      ncclNetProperties_t physProps;
      NCCLCHECK(net->getProperties(vProps.vProps.devs[i], &physProps));  // 获取到合并中的物理网卡属性
      struct ncclXmlNode* physNetNode = NULL;
      NCCLCHECK(xmlFindTagKv(xml, "net", &physNetNode, "name", physProps.name));  // 找到该物理网卡 <net> 节点
      if (physNetNode) {
        NCCLCHECK(xmlSetAttrInt(physNetNode, net->net ? "net" : (net->gin ? "gin" : "coll"), 0));  // 只清理自己的插件标志
        // net is always present (see ncclTopoPopulateNics).
        int net = 0, gin = 0, coll = 0;
        NCCLCHECK(xmlGetAttrInt(physNetNode, "net", &net));
        NCCLCHECK(xmlGetAttrIntDefault(physNetNode, "gin", &gin, 0));
        NCCLCHECK(xmlGetAttrIntDefault(physNetNode, "coll", &coll, 0));
        // 当没有插件使用物理网卡时，设置属性 keep=0，后续会裁掉 <net> 节点
        // Set "keep = 0" only if no plugin is using the physical device
        if (net == 0 && gin == 0 && coll == 0) NCCLCHECK(xmlSetAttrInt(physNetNode, "keep", 0));
      }
    }
  }
  return ncclSuccess;
}

// Calls to network plugin APIs should be protected. This function should be called inside a per-process lock.
ncclResult_t ncclTopoProcessNet(ncclXml* xml, const char* dumpXmlFile, struct ncclTopoNetInfo* net) {
  /*
  门控 usePhysicalDevices 的两个触发条件，决定了"只填物理、不融合"：
    dumpXmlFile 非空：NCCL_TOPO_DUMP_FILE 模式要导出真实物理拓扑给人看或喂回 NCCL_TOPO_FILE，融合会污染它，故跳过；
    net->makeVDevice == NULL：gin/rma 后端根本没有融合能力（它们的 makeVDevice 是 NULL），只有 collnet/net 有真回调。
  */
  bool usePhysicalDevices = (dumpXmlFile || net->makeVDevice == NULL);
  int nPhysicalNics, nVirtualNics;
  // 此处获取的 nPhysicalNics 等于 ndevs，nVirtualNics 应该是 NCCL_UNDEF_DEV_COUNT(-1)
  NCCLCHECK(net->getDevCount(net->netPluginIndex, &nPhysicalNics, &nVirtualNics));
  /*
  把所有物理网卡设备翻译成 <net> 节点以及构建拓扑树中的完整路径，大致如下所示：
    <system version="1">
      <!-- 上游：本 NUMA 的 CPU / root complex -->
      <cpu numaid="0" affinity="0-31" arch="x86_64" vendor="GenuineIntel" host_hash="0x5f3a...">

        <!-- 上游：PCIe switch（class 0x060400 = PCI bridge），由 ncclTopoGetXmlFromSys 向上冒泡建出 -->
        <pci busid="0000:3a:00.0" class="0x060400" vendor="0x10b5" device="0xc010"
            link_speed="16.0 GT/s" link_width="16">

          <!-- 下游：双端口网卡端点（class 0x02 = 网络设备）-->
          <!-- 两个端口 function .0/.1 被 ncclIbGetPciPath 末位归零，合并成这一个 busid -->
          <pci busid="0000:3b:00.0" class="0x02" vendor="0x15b3" device="0x101b"
              link_speed="16.0 GT/s" link_width="16">

            <!-- 一个 nic 容器，双端口共享 -->
            <nic>
              <!-- 端口 1 -->
              <net name="mlx5_0" dev="0" port="1" speed="200000" gdr="1" net="1"
                  guid="0x946dae0300..." maxconn="262144"/>
              <!-- 端口 2：同芯片故 guid 相同，靠 name/dev/port 区分 -->
              <net name="mlx5_1" dev="1" port="2" speed="200000" gdr="1" net="1"
                  guid="0x946dae0300..." maxconn="262144"/>
            </nic>

          </pci>
        </pci>
      </cpu>
    </system>
  */
  // List the physical devices in the topo and set keep = 1
  NCCLCHECK(ncclTopoPopulateNics(xml, 0, nPhysicalNics, net, /*virtual=*/false));
  if (!usePhysicalDevices) {  // collnet/net 且非 dump
    // 融合虚拟设备只在进程全局做一次，nVirtualNics 是从 NET 插件的进程级全局字段，NCCL_UNDEF_DEV_COUNT = -1 是哨兵
    // Virtual devices are only created once per network
    if (nVirtualNics == NCCL_UNDEF_DEV_COUNT) {
      NCCLCHECK(ncclTopoMakeVNics(xml, net, nPhysicalNics));
      // Update the number of virtual devices both locally and in the state tracking the plugin.
      // Note: 0 is a valid number of virtual devices
      int nDevs;
      NCCLCHECK(net->devices(&nDevs));  // 取融合后"物理+虚拟"总设备数
      nVirtualNics = nDevs - nPhysicalNics;  // 反推虚拟数
      NCCLCHECK(net->setVirtDevCount(net->netPluginIndex, nVirtualNics));  // 写回进程级台账 netVirtDevs
    }
    // populate the virtual devices if any
    if (nVirtualNics > 0) {  // 如果有虚拟网卡
      // 需要把被取代的物理网卡降级 keep=0
      // Note: ncclTopoMakeVnic doesn't create a vNic if ndevs = 1; no special case needed
      NCCLCHECK(ncclTopoUpdateVNics(xml, net, nPhysicalNics, nVirtualNics));
      // 把虚拟网卡填进 XML
      // Populate the virtual devices and set keep = 1
      NCCLCHECK(ncclTopoPopulateNics(xml, nPhysicalNics, nPhysicalNics + nVirtualNics, net, /*virtual=*/true));
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoGetFusionEnv(int* mergeLevel, const char** forceMerge) {
  /*
  直接把 NCCL_NET_FORCE_MERGE 的原始字符串（可能 NULL）透传给出参。
  真正的语法解析（; 分组、组内 ,`、name:port:rail:plane）发生在ncclTopoForceMerge里。
  */
  if (forceMerge) *forceMerge = ncclGetEnv("NCCL_NET_FORCE_MERGE");
  const char* mergeLevelEnv = ncclGetEnv("NCCL_NET_MERGE_LEVEL");  // 合并网卡的距离级别，合并近于这个距离的物理网卡
  if (mergeLevelEnv) {
    kvConvertToInt(mergeLevelEnv, mergeLevel, nicPathKvList);  // 字符串转化为 PATH_* 整数
  } else {
    *mergeLevel = PATH_PORT;  // 默认网卡合并距离为 port
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetMergePolicy(int* mergePolicy) {
  *mergePolicy = NCCL_NET_MERGE_POLICY_ALL;  // 默认 ALL(0)
  // NCCL_NET_MERGE_POLICY=RAIL/rail/ALL/all
  const char* env = ncclGetEnv("NCCL_NET_MERGE_POLICY");
  if (env) {
    if (strcasecmp(env, "RAIL") == 0) {  // 大小写不敏感
      *mergePolicy = NCCL_NET_MERGE_POLICY_RAIL;  // RAIL(1)
      INFO(NCCL_ENV, "NCCL_NET_MERGE_POLICY set by environment to RAIL");
    } else if (strcasecmp(env, "ALL") != 0) {  // 既非 RAIL 也非 ALL，报错
      WARN("NCCL_NET_MERGE_POLICY: unknown value '%s', defaulting to ALL", env);
    }
  }
  return ncclSuccess;
}

static std::mutex netMutex;

/*
每个进程只探测"自己管的那块 GPU + 本地 NIC"，建成 XML 中间表示，
再和"本地组"里的其它 rank 做 XML all-gather + 融合，拼出一份统一的大拓扑 XML，
最后转成 ncclTopoSystem。 其中"本地组"的定义随 comm->MNNVL 而变。
*/
ncclResult_t  ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) {
  ncclResult_t ret = ncclSuccess;
  struct ncclXml* xml;  // 拓扑的 XML 中间表示
  char* mem = NULL;  // all-gather 大缓冲(nLocalRanks 份 XML)
  int* localRanks = NULL;  // "本地组"成员 rank 列表
  struct ncclXml* rankXml;  // 指向 mem 里"自己那一格"
  int localRank = -1, nLocalRanks = 0;
  struct ncclTopoNetInfo netInfo = {0};  // 网络插件导入的复用描述符
  // 初始化 xml 占用的内存，单份 XML 的节点上限 256
  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));
  // 获取指定的已有 topo 文件
  const char* xmlTopoFile = ncclGetEnv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    // 读取 topo 文件并解析为 xml 对象
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1), ret, fail);
  } else {
    // Try default XML topology location
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0), ret, fail);
  }
  // XML 文件里的 host_hash 不打算保留(文件可能来自别的机器或是通用模板)，所以遍历每个 <cpu> 节点，用 getHostHash()(本机真实哈希)覆盖
  // Fixup the cpu's host_hashes.
  struct ncclXmlNode* node;
  // Update every cpu node's host_hash attribute since those are not
  // intended to be preserved from the XML files that have been read.
  NCCLCHECKGOTO(xmlFindTag(xml, "cpu", &node), ret, fail);  // 遍历所有 nodes，找到并返回指定 name 的 node
  while (node != nullptr) {  // 遍历修改所有 cpu node
    NCCLCHECKGOTO(xmlSetAttrLong(node, "host_hash", getHostHash()), ret, fail);  // 将 cpu node 的 host_hash 属性值改为本机 host hash
    NCCLCHECKGOTO(xmlFindNextTag(xml, "cpu", node, &node), ret, fail);  // 从上一个 cpu node 开始找到下一个 cpu node
  }
  // 如果文件没载入任何节点，手动建一个带版本号的 <system> 根。后面的融合(ncclTopoFuseXml 找 "system" 根)依赖它存在。
  if (xml->maxIndex == 0) {
    // Create top tag
    struct ncclXmlNode* top;
    NCCLCHECKGOTO(xmlAddNode(xml, NULL, "system", &top), ret, fail);  // 单独创建一个 system node，没有父节点
    NCCLCHECKGOTO(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION), ret, fail);  // 设置改 system node 的唯一属性 version=1
  }

  /*
  Linux 下 fopen("/sys/kernel/pci_switch_link/refresh_switch_toplogy", "r")，
  触发内核重新扫描 Broadcom PCI switch 的链路拓扑。某些平台(用 BCM PCIe switch 的)必须先戳一下这个 sysfs 钩子，
  switch 下游的 P2P 链路才会更新到可探测状态。平台特定的预处理。
  */
  NCCLCHECKGOTO(ncclTopoRefreshBcmP2pLinks(), ret, fail);

  // 只探测本进程管的 GPU，其它 GPU 通过 XML 融合获得。这就是分布式拓扑发现的分工，没人独自扫全机，各扫各的再合并。
  // Detect only the GPU managed by this process.  We'll get any others through XML fusion.
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  NCCLCHECKGOTO(int64ToBusId(comm->peerInfo[comm->rank].busId, busId), ret, fail);
  // 探测这块 GPU 对应的 <pci> 节点完整信息（回溯向上对应的 <cpu>/<pci> 链路），以及其下的 <gpu> 节点完整信息，包括 <nvlink>、<c2c>
  NCCLCHECKGOTO(ncclTopoFillGpu(xml, busId, &node), ret, fail);
  /*
  Set vs Init 的区别是这里的深入点:
    xmlSetAttrInt(总是覆盖/新增)用于 keep/rank/mlopart，这些是本进程的权威值，必须写成真实值。
    xmlInitAttrInt("Only set values if not already set"，xmlGetAttrIndex 返回 -1 才写)用于 gdr，
  因为 gdr 可能已在载入的 XML 文件里被显式配置过，要尊重文件预设，不覆盖。
    keep=1 是给 ncclTopoTrimXml 用的"保留标记"。
  */
  if (node) {
    NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 1), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(node, "rank", comm->rank), ret, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(node, "gdr", comm->peerInfo[comm->rank].gdrSupport), ret, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(node, "mlopart", comm->peerInfo[comm->rank].mloPart), ret, fail);
  }

  /*
  四段的顺序是刻意的：gin → rma → collnet → net。注先导入 gin 和 collnet，让它们取得优先权。原因在于：
    这几族插件可能描述重叠的物理 NIC（同一张网卡既能被 net 枚举、也能被 collnet 枚举）；
    物理 NIC 在 XML 里按 busid find-or-create，填充走 xmlSetAttrIfUnset 这类"缺失才填、已设不覆盖"的幂等惯用法；
    所以第一个填充某张物理 NIC 的插件"认领"了它，后来的插件看到属性已存在便不再覆盖。
  把 gin/collnet 排在 plain net 前面，就是让它们在共享 NIC 节点上优先于普通 net 插件定调。
  */
  // Auto-detect NICs if needed, net/gin/collnet share the same xml/graph nodes.
  // Start with gin, then with collnet so that they precedence.
  {
    /*
    网络插件（尤其 net/collnet 的设备枚举与虚拟设备计数）内部维护进程级共享状态，插件是 dlopen 一次、全进程共用的。
    同一进程里多个 communicator 可能并发跑 ncclTopoGetSystem，若不加锁，
    两个 comm 同时调 getDevCount/makeVDevice/setVirtDevCount 会撕裂插件的设备表。所以用一把全局互斥锁把"导入网络插件"整段串行化。
    */
    std::lock_guard<std::mutex> lock(netMutex);
    INFO(NCCL_GRAPH, "TOPO/NET : Importing network plugins to topology");
    /*
    四段结构相似，但有几处实质性不对称，正好对应"哪些插件支持虚拟 NIC 融合"：
      字段	           gin	              rma	               collnet	                                      net
      守卫条件	        if (gin)	         if (rma)	          if (collNetSupport(comm))	                    无条件
      makeVDevice	     NULL	              NULL	             comm->ncclCollNet->makeVDevice	               comm->ncclNet->makeVDevice
      setVirtDevCount	不绑	              不绑	              ncclCollNetSetVirtDevCount	                  ncclNetSetVirtDevCount
      maxDevsPerNic	  不设	              不设	              按 ncclNetVer>=12 选	                         按 ncclNetVer>=12 选
      融合环境	       不调	               不调	               ncclTopoGetFusionEnv + ncclTopoGetMergePolicy  同 collnet
      getDevCount	    ncclGinGetDevCount ncclRmaGetDevCount	 ncclCollNetGetDevCount	                       ncclNetGetDevCount
    
      gin / rma 是"纯物理、不融合"：makeVDevice = NULL，也不绑 setVirtDevCount、不读融合环境。这直接决定了下游 usePhysicalDevices 为真，
    它们的 NIC 永远以物理形态进 XML。GIN（GPU 发起的网络）和 RMA 是点对点/设备发起原语，"把多张物理网卡并成一张虚拟网卡"对它们不适用，插件本身也没有 makeVDevice API。
      collnet / net 支持虚拟 NIC 融合：绑了真的 makeVDevice、setVirtDevCount，设了 maxDevsPerNic，还读融合策略。net 段无条件执行，
    是主力（普通 IB/RoCE/以太网都走它）；collnet 段受 collNetSupport(comm) 门控（只有底层网络支持集合卸载，如 SHARP，才导入）。
      maxDevsPerNic 的版本分支：(comm->ncclNetVer >= 12) ? NCCL_NET_MAX_DEVS_PER_NIC : NCCL_NET_MAX_DEVS_PER_NIC_V11，
    net 插件 API v12 改了"单虚拟 NIC 最多并几张物理 NIC"的上限，这里按协商出的插件版本选对应常量。
    */
    ncclGin_t* gin = comm->sharedRes->ginState.ncclGin;
    if (gin) {
      netInfo.net = 0;
      netInfo.coll = 0;
      netInfo.gin = 1;  // 只有 gin 插件类型置 1
      netInfo.rma = 0;
      netInfo.netPluginIndex = comm->ginPluginIndex;
      netInfo.dmaBufSupport = comm->dmaBufSupport;
      netInfo.getDevCount = ncclGinGetDevCount;
      netInfo.name = gin->name;
      netInfo.getProperties = gin->getProperties;
      netInfo.makeVDevice = NULL;
      netInfo.devices = gin->devices;
      NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
    }
    ncclRma_t* rma = comm->rmaState.rmaProxyState.ncclRma;
    if (rma) {
      netInfo.net = 0;
      netInfo.coll = 0;
      netInfo.gin = 0;
      netInfo.rma = 1;
      netInfo.netPluginIndex = comm->rmaPluginIndex;
      netInfo.dmaBufSupport = comm->dmaBufSupport;
      netInfo.getDevCount = ncclRmaGetDevCount;
      netInfo.name = rma->name;
      netInfo.getProperties = rma->getProperties;
      netInfo.makeVDevice = NULL;
      netInfo.devices = rma->devices;
      NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
    }
    if (collNetSupport(comm)) {
      netInfo.net = 0;
      netInfo.coll = 1;
      netInfo.gin = 0;
      netInfo.rma = 0;
      netInfo.netPluginIndex = comm->netPluginIndex;
      netInfo.maxDevsPerNic = (comm->ncclNetVer >= 12) ? NCCL_NET_MAX_DEVS_PER_NIC : NCCL_NET_MAX_DEVS_PER_NIC_V11;
      netInfo.dmaBufSupport = comm->dmaBufSupport;
      netInfo.getDevCount = ncclCollNetGetDevCount;
      netInfo.setVirtDevCount = ncclCollNetSetVirtDevCount;
      netInfo.name = comm->ncclCollNet->name;
      netInfo.getProperties = comm->ncclCollNet->getProperties;
      netInfo.makeVDevice = comm->ncclCollNet->makeVDevice;
      netInfo.devices = comm->ncclCollNet->devices;
      // 获取合并物理网卡为虚拟网卡相关环境变量
      NCCLCHECK(ncclTopoGetFusionEnv(&netInfo.mergeLevel, &netInfo.forceMerge));
      NCCLCHECK(ncclTopoGetMergePolicy(&netInfo.mergePolicy));
      NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
    }

    netInfo.net = 1;
    netInfo.coll = 0;
    netInfo.gin = 0;
    netInfo.rma = 0;
    netInfo.netPluginIndex = comm->netPluginIndex;
    netInfo.maxDevsPerNic = (comm->ncclNetVer >= 12) ? NCCL_NET_MAX_DEVS_PER_NIC : NCCL_NET_MAX_DEVS_PER_NIC_V11;
    netInfo.dmaBufSupport = comm->dmaBufSupport;
    netInfo.getDevCount = ncclNetGetDevCount;
    netInfo.setVirtDevCount = ncclNetSetVirtDevCount;
    netInfo.name = comm->ncclNet->name;
    netInfo.getProperties = comm->ncclNet->getProperties;
    netInfo.makeVDevice = comm->ncclNet->makeVDevice;
    netInfo.devices = comm->ncclNet->devices;
    // 获取合并物理网卡为虚拟网卡相关环境变量
    NCCLCHECK(ncclTopoGetFusionEnv(&netInfo.mergeLevel, &netInfo.forceMerge));
    NCCLCHECK(ncclTopoGetMergePolicy(&netInfo.mergePolicy));
    NCCLCHECKGOTO(ncclTopoProcessNet(xml, dumpXmlFile, &netInfo), ret, fail);
  }

  // 对 xml 树进行剪纸，删除其下没有 keep=1 属性的所有子树分支
  // Remove XML branches which don't have a node with keep="1" (typically when importing a topology)
  NCCLCHECKGOTO(ncclTopoTrimXml(xml), ret, fail);

  // XML topo fusion.
  if (comm->MNNVL) {
    // 哪些 rank 参与把各自探测的局部拓扑拼成一张统一大图，这些 rank 是本地组（local group），即整个跨节点 NVLink clique
    // MNNVL clique support
    nLocalRanks = comm->clique.size;
    localRank = comm->cliqueRank;
    localRanks = comm->clique.ranks;
  } else {
    // 同节点内的 ranks 构造相同的一个数组，表示哪些全局 rank 与自己是同节点
    // Intra-node fusion.  Much of the comm is not initialized yet at this point so we need to do our own calculations.
    NCCLCHECKGOTO(ncclCalloc(&localRanks, comm->nRanks), ret, fail);
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank) localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
  // 分配 nLocalRanks 个 ranks 的拓扑 xml 占用的内存
  NCCLCHECKGOTO(ncclCalloc(&mem, nLocalRanks * xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  // 计算当前 rank 在 mem 中对应的内存地址，并将准备好的拓扑 xml 拷贝到该地址
  rankXml = (struct ncclXml*)(mem + xmlMemSize(NCCL_TOPO_XML_MAX_NODES) * localRank);
  memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
  // 序列化所有指针（parent/sub）的绝对地址为相对偏移
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), ret, fail);
  // 在本地组内（节点内 / MNNVL 域内）对 mem 进行 AllGather
  // nLocalRanks can't actually be 0, or we wouldn't be running at all...
  // coverity[divide_by_zero]
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem,
                                            xmlMemSize(NCCL_TOPO_XML_MAX_NODES)),
                ret, fail);
  if (comm->MNNVL) {
    // 确保合并来自不同节点的 topo 时有足够空间，先释放当前节点的 topo xml 空间，再针对 nodes 大小申请足够本地组合并的内存大小上限的 topo xml
    // Ensure that we have enough room when fusing topos from multiple nodes.
    free(xml);
    xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(&xml, nLocalRanks * NCCL_TOPO_XML_MAX_NODES), ret, fail);
  } else {
    // 节点内场景不需要扩展 topo xml，复用之前当前 rank 的 topo xml 的内存，直接索引清零
    // In the intra-node case there's no need to enlarge the topo xml.
    xml->maxIndex = 0;
  }
  for (int i = 0; i < nLocalRanks; i++) {  // 遍历所有本地组的 topo xml
    struct ncclXml* peerXml = (struct ncclXml*)(mem + xmlMemSize(NCCL_TOPO_XML_MAX_NODES) * i);
    // 将所有本地组 rank 的 topo xml 反序列化，所有相对偏移地址改回绝对地址指针
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);
    // 依次将本地组内所有 rank 的 topo xml 合并到 xml 中，取并集树
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
  }

  // 根据 NCCL_TOPO_DUMP_FILE_RANK 判断当前 rank 是否负责将 topo xml dump 到文件
  if (dumpXmlFile && comm->rank == ncclParamTopoDumpFileRank()) {
    INFO(NCCL_ENV, "NCCL_TOPO_DUMP_FILE set by environment to %s", dumpXmlFile);
    // 负责 rank 将本地组合并后的 topo xml dump 到指定文件
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(dumpXmlFile, xml), ret, fail);
  }

  /*
  把前面融合好的 XML 树 xml 转换成内存里的 ncclTopoSystem，即 comm->topo，后续建图/路径判定/算法选择的物理拓扑基础，
  只有构建模式才落这个内存结构，dump 模式那次调用 system=NULL，只写文件不建 system
  */
  // Only update our topo tracking structure if we aren't dumping (separate steps)
  if (dumpXmlFile == NULL) NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, getHostHash()), ret, fail);

exit:
  if (!comm->MNNVL && localRanks) free(localRanks);
  if (mem) free(mem);
  free(xml);
  return ret;
fail:
  goto exit;
}

/*
查询与指定顶点的 path 带宽最大，路径类型质量最高的所有指定类型的顶点。对 GPU→NET 就是与这块 GPU 最亲和的网卡，
对 NET→GPU 就是与这块网卡最亲和的 GPU。一层可能包含多个顶点，这正是返回集合而非单值的根本原因。
  type / index：源顶点类型 + 在该类型桶 nodes[type] 里的下标
  resultType：目标顶点类型
  locals[]：输出命中的目标顶点下标
  localCount：输出命中个数
  pathType：可选输出最优路径的 PATH_* 类型
*/
ncclResult_t ncclTopoGetLocal(struct ncclTopoSystem* system, int type, int index, int resultType,
                              int locals[NCCL_TOPO_MAX_NODES], int* localCount, int* pathType) {
  int minType = PATH_DIS;
  float maxBw = 0;
  int count = 0;
  struct ncclTopoLinkList* paths = system->nodes[type].nodes[index].paths[resultType];  // 获取源顶点到所有指定类型顶点的 paths
  if (paths == NULL) {
    *localCount = 0;
    return ncclSuccess;
  }
  for (int i = 0; i < system->nodes[resultType].count; i++) {  // 遍历所有指定类型顶点
    // path 带宽超过最大值或 path 路径类型差于最差类型
    if (paths[i].bw > maxBw || (paths[i].bw == maxBw && paths[i].type < minType)) {
      maxBw = paths[i].bw;  // 更新最大带宽
      minType = paths[i].type;  // 更新最优路径类型
      if (pathType) *pathType = minType;
      count = 0;  // 计数从 0 开始
    }
    if (paths[i].bw == maxBw && paths[i].type == minType) {  // 出现一个带宽和距离都相同的 path
      if (count == NCCL_TOPO_MAX_NODES) {
        WARN("Error : ran out of room to store found nodes in ncclTopoGetLocal."
             " Filled %d of type %d, starting from index %d of type %d.",
             NCCL_TOPO_MAX_NODES, resultType, index, type);
        return ncclInternalError;
      }
      locals[count++] = i;
    }
  }
  *localCount = count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int* count, float* bw) {
  // Assuming BW to CPU reflects the GPU bandwidth via P2P or C2C.
  // Caveat, this could be wrong if there is a PCIe switch, and a narrower link to the CPU.
  int c;
  NCCLCHECK(ncclGetLocalCpu(system, gpu, &c));
  float gpuBw = system->nodes[GPU].nodes[gpu].paths[CPU][c].bw;
  int rank = system->nodes[GPU].nodes[gpu].gpu.rank;

  int netCountByBw = 0;
  float totalNetBw = 0;
  int64_t firstNetId = 0;
  for (int c = 0; c < MAXCHANNELS; c++) {
    int net;
    int64_t netId;
    NCCLCHECK(ncclTopoGetLocalNet(system, rank, c, &netId, NULL));
    NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &net));
    if (c == 0) firstNetId = netId;
    else if (firstNetId == netId) break;

    totalNetBw += system->nodes[GPU].nodes[gpu].paths[NET][net].bw;
    netCountByBw++;
    if (totalNetBw >= gpuBw) break;
  }
  *count = netCountByBw;
  *bw = totalNetBw;
  return ncclSuccess;
}

static int netDevsPolicyNum = -1;
static enum netDevsPolicy netDevsPolicy = NETDEVS_POLICY_UNDEF;
// 从 NCCL_NETDEVS_POLICY 获取 netDevsPolicy 枚举类型的值 AUTO/ALL/MAX，默认采用 AUTO
static void getNetDevsPolicyOnce() {
  const char* envStr = ncclGetEnv("NCCL_NETDEVS_POLICY");
  if (envStr) {
    if (strcasecmp(envStr, "AUTO") == 0) {
      netDevsPolicy = NETDEVS_POLICY_AUTO;
    } else if (strcasecmp(envStr, "ALL") == 0) {
      netDevsPolicy = NETDEVS_POLICY_ALL;
    } else if (strncasecmp(envStr, "MAX:", strlen("MAX:")) == 0) {
      int envNum = atoi(envStr + strlen("MAX:"));  // MAX: 后面跟了一个整型值
      if (envNum > 0) {
        netDevsPolicy = NETDEVS_POLICY_MAX;
        netDevsPolicyNum = envNum;
      }
    }
    if (netDevsPolicy == NETDEVS_POLICY_UNDEF) {
      INFO(NCCL_ENV, "Unable to recognize NCCL_NETDEVS_POLICY=%s, using NCCL_NETDEVS_POLICY_AUTO instead.", envStr);
    } else {
      INFO(NCCL_ENV, "NCCL_NETDEVS_POLICY set by environment to %s", envStr);
    }
  }
  if (netDevsPolicy == NETDEVS_POLICY_UNDEF) netDevsPolicy = NETDEVS_POLICY_AUTO;
}

// 获取每个 GPU 对应网卡的策略
ncclResult_t ncclTopoGetNetDevsPolicy(enum netDevsPolicy* policy, int* policyNum) {
  static std::once_flag onceFlag;
  std::call_once(onceFlag, getNetDevsPolicyOnce);  // 阻塞执行一次，从环境变量获取 netDevsPolicy
  if (netDevsPolicy == NETDEVS_POLICY_MAX && netDevsPolicyNum <= 0) {  // MAX 必须配套一个正数
    WARN("Invalid number of network devices = %d for policy MAX", netDevsPolicyNum);
    return ncclInternalError;
  }
  if (policy) *policy = netDevsPolicy;  // 输出 netDevsPolicy
  if (policyNum && netDevsPolicyNum >= 0) *policyNum = netDevsPolicyNum;  // 输出 netDevsPolicyNum
  return ncclSuccess;
}

/*
给定 rank 和 channel 号，这个 rank 该用哪块本地网卡。一个 GPU 附近可能等距等带宽地挂着多块网卡，
多个 GPU 又共享同一批网卡，还要保证所有进程算出来的答案一致。这个函数就是这套约束的解法。
  type: NET 或 GIN
  rank: 本 rank
  channelId: channel 序号，决定在可用网卡中轮转到第几块
  id: 选中网卡的拓扑顶点 id（NCCL_TOPO_ID(systemId, localId)），供拓扑查询与跨 rank 交换
  dev: 选中网卡的 net.dev，网络插件层设备号
*/
ncclResult_t ncclTopoGetLocalNetType(struct ncclTopoSystem* system, int type, int rank, int channelId, int64_t* id,
                                     int* dev) {
  int gpu;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpu, /*showWarn=*/true));  // 查询 rank 对应的 GPU 顶点索引

  int localNets[NCCL_TOPO_MAX_NODES];
  int localNetCount;
  NCCLCHECK(ncclTopoGetLocal(system, GPU, gpu, type, localNets, &localNetCount, NULL));  // 找出该 GPU 的并列最优网卡候选集
  if (localNetCount == 0) {
    WARN("Could not find any local path from gpu %d to net.", gpu);
    return ncclInternalError;
  }

  int localGpuCount = 0, netsPerGpu = 0, policyCount = 0;
  int localGpus[NCCL_TOPO_MAX_NODES];
  enum netDevsPolicy policy;
  NCCLCHECK(ncclTopoGetNetDevsPolicy(&policy, &policyCount));  // 获取每个 GPU 对应网卡的策略
  // 以本地第一张网卡为代表，获取它最优的本地 GPU 顶点列表
  NCCLCHECK(ncclTopoGetLocal(system, type, localNets[0], GPU, localGpus, &localGpuCount, NULL));
  if (policy == NETDEVS_POLICY_AUTO) {  // AUTO 策略：本地 GPU 均分本地网卡
    netsPerGpu = DIVUP(localNetCount, localGpuCount);
  } else if (policy == NETDEVS_POLICY_ALL) {  // ALL 策略：所有 GPU 共用所有本地网卡
    netsPerGpu = localNetCount;
  } else if (policy == NETDEVS_POLICY_MAX) {  // MAX 策略：取 MAX 值和本地网卡之间的较小值
    netsPerGpu = std::min(policyCount, localNetCount);
  } else {
    WARN("Unknown netDevs policy");
    return ncclInternalError;
  }

  /*
  选择初始网络是为了避免冲突，并使所有 GPU 遵循类似的模式。
  localGpuCount 个 GPU 共享 localNetCount 个 NET 设备，每个 GPU 使用 netsPerGpu 个 NET 设备。
  */
  // Starting net is chosen to avoid collision and follow a similar pattern for all GPUs.
  // localGpuCount GPUs share localNetCount NET devs; each GPU using netsPerGpu NET devs.
  int net = system->nodes[GPU].nodes[gpu].gpu.dev % localGpuCount;  // 用 GPU 的 NVML 设备索引做散列种子，能能均匀铺满 [0, localGpuCount)
  /*
  在 localNetCount = localGpuCount × netsPerGpu 且 localNetCount 是 2 的幂的情况下，localGpuCount 和 netsPerGpu 也都是 2 的幂，
  此时 net 在 [0, localGpuCount) 范围内，翻转后一定是 netsPerGpu 的倍数，再作为基偏移可以尽可能地将网卡不重复的分配到 GPU。
  */
  if (isPow2(localNetCount)) net = mirrorBits(net, localNetCount);
  net += channelId % (netsPerGpu);  // channelId 在 对应的 GPU 对应的 那组网卡之间轮转
  if (id) *id = system->nodes[type].nodes[localNets[net % localNetCount]].id;  // 输出网卡顶点 ID
  if (dev) *dev = system->nodes[type].nodes[localNets[net % localNetCount]].net.dev;  // 输出网卡插件层设备号
  return ncclSuccess;
}

// 获取当前 rank + channelId 对应的 NET 顶点 ID 和设备索引号
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, NET, rank, channelId, id, dev);
}
ncclResult_t ncclTopoGetLocalGinDev(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, GIN, rank, channelId, id, dev);
}

ncclResult_t ncclTopoGetLocalGinDevs(struct ncclComm* comm, int* localGinDevs, int* localGinCount) {
  for (int c = 0; c < NCCL_TOPO_MAX_NODES; c++) {
    NCCLCHECK(ncclTopoGetLocalGinDev(comm->topo, comm->rank, c, NULL, localGinDevs + c));
    if (c > 0 && localGinDevs[c] == localGinDevs[0]) {
      *localGinCount = c;
      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalRmaDev(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  return ncclTopoGetLocalNetType(system, RMA, rank, channelId, id, dev);
}

ncclResult_t ncclTopoGetLocalRmaDevs(struct ncclComm* comm, int* localRmaDevs, int* localRmaCount) {
  for (int c = 0; c < NCCL_TOPO_MAX_NODES; c++) {
    NCCLCHECK(ncclTopoGetLocalRmaDev(comm->topo, comm->rank, c, NULL, localRmaDevs + c));
    if (c > 0 && localRmaDevs[c] == localRmaDevs[0]) {
      *localRmaCount = c;
      break;
    }
  }
  return ncclSuccess;
}

// 查找采用指定 NET 的 GPU 索引
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex) {
  ncclResult_t ret = ncclSuccess;
  int netIndex;
  NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &netIndex));  // 获取指定 NET 顶点的索引

  int localGpus[NCCL_TOPO_MAX_NODES];
  int localGpuCount;
  NCCLCHECK(ncclTopoGetLocal(system, NET, netIndex, GPU, localGpus, &localGpuCount, NULL));  // 获取指定 NET 顶点最近的 GPU 顶点列表

  int foundGpu = -1;
  for (int c = 0; c < MAXCHANNELS; c++) {
    for (int lg = 0; lg < localGpuCount; lg++) {  // 每个 channel 都遍历所有最近的 GPU 顶点
      int g = localGpus[lg];
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
      int64_t id;
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &id, NULL));  // 获取当前 gpu rank + channel 对应的网卡 ID
      if (netId == id) {  // 找到采用指定 NET 的 GPU
        foundGpu = g;
        goto exit;
      }
    }
  }
exit:
  *gpuIndex = foundGpu;
  return ret;
}

/****************************/
/* External query functions */
/****************************/

// 从第一个代表 CPU 顶点获取 CPU 架构、厂商和代际
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model) {
  *arch = system->nodes[CPU].nodes[0].cpu.arch;
  *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
  *model = system->nodes[CPU].nodes[0].cpu.model;
  return ncclSuccess;
}

/*
这个参数名叫「忽略 CPU 亲和性」，但它忽略的对象是「job 或外部环境给进程设置的绑核」（也就是 mask），
并非「CPU 亲和性这个优化本身」。忽略之后，NCCL 用的是更严格的 GPU 本地 NUMA 掩码。

取值	                              代码行为	                      直观效果
NCCL_IGNORE_CPU_AFFINITY=0（默认）	finalMask = mask & cpuMask	   尊重你的绑核，只在其中挑 GPU 本地的那部分
NCCL_IGNORE_CPU_AFFINITY=1	       finalMask = cpuMask	          无视你的绑核，直接绑到整个 GPU 本地 NUMA
*/
NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity) {
  struct ncclTopoNode *cpu = NULL, *gpu = NULL;
  int gpuIndex, cpuIndex;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpuIndex, /*showWarn=*/true));  // 获取当前 rank 对应的 GPU 顶点索引
  NCCLCHECK(ncclGetLocalCpu(system, gpuIndex, &cpuIndex));  // 找到当前 GPU 顶点对应的 CPU 顶点索引
  gpu = system->nodes[GPU].nodes + gpuIndex;
  cpu = system->nodes[CPU].nodes + cpuIndex;

  // 尊重用户已有的绑核，只在用户允许的范围内进一步收缩到 GPU 本地 NUMA
  // Query the CPU affinity set we were provided
  ncclAffinity mask;
  NCCLCHECK(ncclOsGetAffinity(&mask));  // 查询 CPU 位图

  // Get the affinity of the CPU close to our GPU.
  ncclAffinity cpuMask = cpu->cpu.affinity;  // 获取亲和当前 GPU rank 的 CPU 上的亲和 CPU 核位图

  // 
  // Get the final affinity
  ncclAffinity finalMask;
  if (ncclParamIgnoreCpuAffinity()) {  // 忽略绑核
    // Ignore the CPU affinity set and use the GPU one instead
    finalMask = cpuMask;  // 最终采用的 CPU mask 等于当前 CPU 对应的所有亲和 CPU 位图
  } else {
    // Use a subset of the GPU affinity set
    finalMask = ncclOsCpuAnd(mask, cpuMask);  // 取限制的 CPU 位图与当前 CPU 亲和的所有核的交集
  }

  memcpy(affinity, &finalMask, sizeof(ncclAffinity));  // 拷贝到输出结果

  /*
  示例 1：默认模式，用户绑核横跨两个 NUMA，numactl --physcpubind=0-7,16-23 ./app
  NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 0 is 0-7. (GPU affinity = 0-15 ; CPU affinity = 0-7,16-23).

  示例 2：强制模式，无视用户绑核，NCCL_IGNORE_CPU_AFFINITY=1 ./app
  NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 0 is 0-15. (GPU affinity = 0-15).

  示例 3：交集为空（用户绑在别的 NUMA 上），taskset -c 16-23 ./app
  NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 0 is empty, ignoring. (GPU affinity = 0-15 ; CPU affinity = 16-23).

  示例 4：用户给了全机核，NCCL 自行收窄，taskset -c 0-31 ./app
  NCCL INFO ncclTopoGetCpuAffinity: Affinity for GPU 0 is 0-15. (GPU affinity = 0-15 ; CPU affinity = 0-31).
  */
  // display the final affinity
  char msg[1024] = "";
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "Affinity for GPU %d is ", gpu->gpu.dev);
  if (ncclOsCpuCount(finalMask)) {
    (void)ncclCpusetToRangeStr(&finalMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  } else {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "empty, ignoring");
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ". (GPU affinity = ");
  (void)ncclCpusetToRangeStr(&cpuMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  if (!ncclParamIgnoreCpuAffinity()) {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ; CPU affinity = ");
    (void)ncclCpusetToRangeStr(&mask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ").");
  INFO(NCCL_INIT, "%s: %s", __func__, msg);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[GPU].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NET].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NVS].count;
  return ncclSuccess;
}

// 获取所有 DEV CC 的最小值和最大值
ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax) {
  if (system->nodes[DEV].count == 0) return ncclInternalError;
  int min, max;
  min = max = system->nodes[DEV].nodes[0].dev.cudaCompCap;  // 以第一个 DEV 顶点作为代表获取 CC
  for (int g = 1; g < system->nodes[DEV].count; g++) {  // 计算所有 DEV CC 的最小值和最大值
    min = std::min(min, system->nodes[DEV].nodes[g].dev.cudaCompCap);
    max = std::max(max, system->nodes[DEV].nodes[g].dev.cudaCompCap);
  }
  if (ccMin) *ccMin = min;
  if (ccMax) *ccMax = max;
  return ncclSuccess;
}
