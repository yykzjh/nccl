/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef XML_H_
#define XML_H_

#include "nccl.h"
#include "debug.h"
#include "checks.h"
#include "alloc.h"
#include <stdlib.h>
#include <cinttypes>

// PCI device class for NVSwitch (used to identify remote NVLink targets)
#define PCI_NVSWITCH_CLASS "0x068000"

// A few constraints to make the implementation easy
#define MAX_STR_LEN 255
#define MAX_ATTR_COUNT 16
#define MAX_SUBS 640

#define NODE_TYPE_NONE 0
#define NODE_TYPE_OPEN 1
#define NODE_TYPE_CLOSE 2
#define NODE_TYPE_SINGLE 3

// 一个 XML 元素（一个树节点），所有节点顺序存进容器的 nodes[]，靠内部指针串成树
struct ncclXmlNode {
  /*
  元素/标签名。MAX_STR_LEN=255，+1 是给字符串结尾的 \0。
  取值就是拓扑文法里那些标签：system/cpu/pci/gpu/nvlink/nic/net/c2c/pcilink。
  xmlLoadSub 靠它匹配 handler，消费侧也靠它识别节点种类。
  */
  char name[MAX_STR_LEN + 1];
  /*
  每项 key[256] + value[256] 的属性表，每个属性是一个 key="value" 对，键和值都以字符串存。
  这是 XML 节点携带全部数据的地方：system 的 version/name、gpu 的 busid/rank/keep、pci 的 vendor/device/busid、
  net 的 dev/guid/mbps 等等，全是属性。数组大小 MAX_ATTR_COUNT + 1 = 17。
  这第 17 个"额外槽"，它是个 sink（吸收槽），被不断覆盖，只为把多余属性从输入流里消费掉。
  */
  struct {
    char key[MAX_STR_LEN + 1];
    char value[MAX_STR_LEN + 1];
  } attrs[MAX_ATTR_COUNT + 1]; // Need an extra one to consume extra params
  /*
  实际保留的属性个数，被 xmlGetNode 钳在 MAX_ATTR_COUNT（≤16）。查属性的 xmlGetAttrIndex 遍历 attrs[a].key 找匹配，
  xmlGetAttr/xmlGetAttrInt/xmlGetAttrLong 再据此取值/转数值。
  */
  int nAttrs;
  /*
  这个 token 是哪类标签，四态：
  值	                  含义	              xmlLoadSub怎么用
  NODE_TYPE_NONE(0)	    无 token / EOF	    顶层→解析结束；子层→"unterminated" 报错
  NODE_TYPE_OPEN(1)	    <foo ...> 开标签	   正常分派、递归下降
  NODE_TYPE_CLOSE(2)	  </foo> 闭标签	       校验与 head 同名后出栈返回
  NODE_TYPE_SINGLE(3)	  <foo .../> 自闭合	   短路：无子节点，直接返回
  */
  int type;
  /*
  指向父节点的指针（根 system 的 parent 为 NULL）。在 xmlLoadSub 里由 node->parent = head 建立。
  用途是向上回溯，例如求一个 net 的 PCIe 路径就是顺着 parent 往上找到名为 pci 的祖先。
  */
  struct ncclXmlNode* parent;
  /*
  子节点指针数组 + 子节点个数（MAX_SUBS=640）。在 xmlLoadSub 里由 head->subs[head->nSubs++] = node 建立。
  用途是向下遍历子树，融合/转图/dump 都是递归走 subs[]。
  */
  struct ncclXmlNode* subs[MAX_SUBS];  // 640
  int nSubs;
};

// 整个文档的容器（头部 + 那块 nodes[] 扁平数组）
struct ncclXml {
  /*
  已用节点数（高水位 / 下一个空闲槽下标）。解析器 node = xml->nodes + xml->maxIndex; ... maxIndex++ 逐个占用槽.
  ncclTopoGetSystem 用 maxIndex == 0 判"空树"，ncclTopoConvertXml/FuseXml 用 for (n = 0; n < maxIndex; n++) 当遍历界。
  */
  int maxIndex;
  /*
  容量（分配了多少个槽）。由 xmlAlloc 写入（如 NCCL_TOPO_XML_MAX_NODES=256），解析器用 if (maxIndex == maxNodes) 做溢出守卫。
  注意 maxIndex（已用）与 maxNodes（容量）分工，别混。
  */
  int maxNodes;
  /*
  struct hack 尾部占位数组。声明 [1]，实际按 maxNodes 分配（xmlMemSize/xmlAlloc）。它就是承载全部节点的扁平存储。
  256 不是通用上限，只是"单 rank 局部 XML"的默认值。
  */
  struct ncclXmlNode nodes[1];
};

/* File functions */
#define NCCL_TOPO_XML_VERSION 1
ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml, int warn);
ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml);
#define NCCL_GRAPH_XML_VERSION 1
ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml);

/* Auto-detect functions */
ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode);
ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* tagName, const char* pciPath, const char* netName,
                             struct ncclXmlNode** netNode, struct ncclXmlNode* forceParent = NULL);

/* Remove unneeded parts */
ncclResult_t ncclTopoTrimXml(struct ncclXml* xml);

/* Fuse multiple system XMLs into one, skipping duplicate entries */
ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src);
/* Relocate pointers in XML to (de-)serialize the structure */
ncclResult_t ncclTopoConvertXml(struct ncclXml* xml, uintptr_t base, int exp);

/**************/
/* XML Struct */
/* Functions  */
/**************/

// 计算 ncclXml 头部 + maxNodes 个 ncclXmlNode 占用的内存大小
static size_t xmlMemSize(int maxNodes) {
  /*
  offsetof(struct ncclXml, nodes) = nodes 字段在结构体里的字节偏移，也就是它前面那个"头部"的大小。
  ncclXml 头部是 int maxIndex, maxNodes 两个 int = 8 字节，又因 ncclXmlNode 含指针、需 8 字节对齐，
  而 8 恰好已对齐，所以 offsetof(nodes) == 8。用 offsetof 而非硬编码 8，是为了让编译器自动算入任何对齐填充，改结构也不会算错。
  */
  return offsetof(struct ncclXml, nodes) + sizeof(struct ncclXmlNode) * maxNodes;
}

// 一次性 calloc 出整块内存，把首地址当成 struct ncclXml* 返回，并记下容量。
static ncclResult_t xmlAlloc(struct ncclXml** xml, int maxNodes) {
  char* mem;
  // 申请指定字节数的内存大小
  NCCLCHECK(ncclCalloc(&mem, xmlMemSize(maxNodes)));
  *xml = (struct ncclXml*)mem;
  (*xml)->maxNodes = maxNodes;
  return ncclSuccess;
}

// 获取指定属性名在属性列表中的下标索引
static ncclResult_t xmlGetAttrIndex(struct ncclXmlNode* node, const char* attrName, int* index) {
  *index = -1;
  const int nAttrs = node->nAttrs;
  for (int a = 0; a < nAttrs; a++) {
    if (strncmp(node->attrs[a].key, attrName, MAX_STR_LEN) == 0) {
      *index = a;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

// 获取指定属性名在属性列表中的属性值
static ncclResult_t xmlGetAttr(struct ncclXmlNode* node, const char* attrName, const char** value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));  // 获取指定属性名在属性列表中的下标索引
  *value = index == -1 ? NULL : node->attrs[index].value;  // 再根据索引获取对应位置的属性值
  return ncclSuccess;
}

// 获取指定属性名在属性列表中的属性值字符串
static ncclResult_t xmlGetAttrStr(struct ncclXmlNode* node, const char* attrName, const char** value) {
  NCCLCHECK(xmlGetAttr(node, attrName, value));  // 获取指定属性名在属性列表中的属性值
  if (*value == NULL) {
    WARN("Attribute %s of node %s not found", attrName, node->name);
    return ncclInternalError;
  }
  return ncclSuccess;
}
static ncclResult_t xmlGetAttrInt(struct ncclXmlNode* node, const char* attrName, int* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));  // 获取指定属性名在属性列表中的属性值字符串
  *value = strtol(str, NULL, 0);  // 属性值字符串转成整型
  return ncclSuccess;
}

// 获取指定节点上的某个整型属性，如果没有就返回指定的默认值
static ncclResult_t xmlGetAttrIntDefault(struct ncclXmlNode* node, const char* attrName, int* value, int defaultValue) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, attrName, &str));
  *value = str ? strtol(str, NULL, 0) : defaultValue;
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrUint64(struct ncclXmlNode* node, const char* attrName, uint64_t* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtoull(str, NULL, 0);
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrUint64Default(struct ncclXmlNode* node, const char* attrName, uint64_t* value,
                                            uint64_t defaultValue) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, attrName, &str));
  *value = str ? strtoull(str, NULL, 0) : defaultValue;
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrLong(struct ncclXmlNode* node, const char* attrName, int64_t* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtol(str, NULL, 0);
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrFloat(struct ncclXmlNode* node, const char* attrName, float* value) {
  const char* str;
  NCCLCHECK(xmlGetAttrStr(node, attrName, &str));
  *value = strtof(str, NULL);
  return ncclSuccess;
}

static ncclResult_t xmlGetAttrFloatDefault(struct ncclXmlNode* node, const char* attrName, float* value,
                                           float defaultValue) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, attrName, &str));
  *value = str ? strtof(str, NULL) : defaultValue;
  return ncclSuccess;
}

static ncclResult_t xmlFindTag(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node) {
  *node = NULL;
  for (int i = 0; i < xml->maxIndex; i++) {
    struct ncclXmlNode* n = xml->nodes + i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlFindNextTag(struct ncclXml* xml, const char* tagName, struct ncclXmlNode* prev,
                                   struct ncclXmlNode** node) {
  *node = NULL;
  for (int i = prev - xml->nodes + 1; i < xml->maxIndex; i++) {
    struct ncclXmlNode* n = xml->nodes + i;
    if (strcmp(n->name, tagName) == 0) {
      *node = n;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlFindTagKv(struct ncclXml* xml, const char* tagName, struct ncclXmlNode** node,
                                 const char* attrName, const char* attrValue) {
  *node = NULL;
  // 全 xml 查找
  for (int i = 0; i < xml->maxIndex; i++) {
    struct ncclXmlNode* n = xml->nodes + i;
    if (strcmp(n->name, tagName) == 0) {  // 找到指定的 tagName 节点
      const char* value;
      NCCLCHECK(xmlGetAttr(n, attrName, &value));  // 获取指定属性名在属性列表中的属性值
      if (value && strcmp(value, attrValue) == 0) {  // 比较指定属性的属性值也是匹配的
        *node = n;
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}

// 在指定节点的子节点层级查找是否有满足 name、type、属性全都匹配目标节点的节点
static ncclResult_t xmlFindNode(struct ncclXmlNode* parentNode, struct ncclXmlNode* searchNode,
                                struct ncclXmlNode** node) {
  *node = NULL;
  // 仅在当前 parentNode 的子层级搜索满足要求的 node
  // Search for the node at the current level only.
  for (int i = 0; i < parentNode->nSubs; i++) {
    struct ncclXmlNode* n = parentNode->subs[i];
    // 匹配节点 name、type、属性个数都相同
    if (strcmp(n->name, searchNode->name) == 0 && n->type == searchNode->type && n->nAttrs == searchNode->nAttrs) {
      int a;
      // 还需要满足节点属性都匹配相同
      // Ensure that all the attributes are the same.
      for (a = 0; a < searchNode->nAttrs; a++) {
        const char* val;
        NCCLCHECK(xmlGetAttr(n, searchNode->attrs[a].key, &val));
        if (!val || strcmp(val, searchNode->attrs[a].value)) break;
      }
      if (a == searchNode->nAttrs) {  // 说明属性全都匹配
        *node = n;  // 返回找到的节点
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlSetAttr(struct ncclXmlNode* node, const char* attrName, const char* value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

static ncclResult_t xmlPrintNodeRecursive(struct ncclXmlNode* node, const char* name) {
  while (node) {
    char line[1024 * 8];
    int cursor = 0;
    snprintf(line, sizeof(line), "<name=%s", node->name);
    for (int i = 0; i < node->nAttrs; i++) {
      cursor = strlen(line);
      snprintf(line + cursor, sizeof(line) - cursor, " %s=%s", node->attrs[i].key, node->attrs[i].value);
    }
    cursor = strlen(line);
    snprintf(line + cursor, sizeof(line) - cursor, ">");
    INFO(NCCL_GRAPH, "%s", line);
    node = node->parent;
  }
  return ncclSuccess;
}

static ncclResult_t xmlSetAttrIfUnset(struct ncclXmlNode* node, const char* attrName, const char* value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));  // 获取指定属性的下标索引
  if (index != -1) return ncclSuccess;
  // 如果没有指定属性，则在属性列表末尾加上该属性
  index = node->nAttrs++;
  strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
  node->attrs[index].key[MAX_STR_LEN] = '\0';
  strncpy(node->attrs[index].value, value, MAX_STR_LEN);
  node->attrs[index].value[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

// 对于指定节点、指定属性名和属性值，如果指定节点没有该属性，则创建该属性，最终设为指定属性值
static ncclResult_t xmlSetAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));  // 获取指定节点某个指定属性索引
  if (index == -1) {  // 如果没有该属性，就创建该属性
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  // 修改属性值为指定的属性值
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  return ncclSuccess;
}

static ncclResult_t xmlSetAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%g", value);
  return ncclSuccess;
}

static ncclResult_t xmlSetAttrLong(struct ncclXmlNode* node, const char* attrName, const int64_t value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    node->attrs[index].key[MAX_STR_LEN] = '\0';
  }
  snprintf(node->attrs[index].value, MAX_STR_LEN, "%#" PRIx64, (uint64_t)value);
  return ncclSuccess;
}

static ncclResult_t xmlUnsetAttr(struct ncclXmlNode* node, const char* attrName) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) return ncclSuccess;
  for (int i = index + 1; i < node->nAttrs; i++) {
    strcpy(node->attrs[i - 1].key, node->attrs[i].key);
    strcpy(node->attrs[i - 1].value, node->attrs[i].value);
  }
  node->nAttrs--;
  return ncclSuccess;
}

static ncclResult_t xmlGetSub(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub) {
  *sub = NULL;
  for (int s = 0; s < node->nSubs; s++) {
    if (strcmp(node->subs[s]->name, subName) == 0) {
      *sub = node->subs[s];
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

// 获取指定 node 下满足指定 node 名、属性名、属性值的子 node
static ncclResult_t xmlGetSubKv(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub,
                                const char* attrName, const char* attrValue) {
  *sub = NULL;
  // 遍历指定 node 的所有子 nodes
  for (int s = 0; s < node->nSubs; s++) {
    struct ncclXmlNode* subNode = node->subs[s];
    if (strcmp(subNode->name, subName) == 0) {  // 对比 node 名
      const char* value;
      NCCLCHECK(xmlGetAttr(subNode, attrName, &value));  // 获取指定属性名的属性值
      if (value && strcmp(value, attrValue) == 0) {  // 对比属性值
        *sub = node->subs[s];
        return ncclSuccess;
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t xmlGetSubKvInt(struct ncclXmlNode* node, const char* subName, struct ncclXmlNode** sub,
                                   const char* attrName, const int attrValue) {
  char strValue[10];
  snprintf(strValue, 10, "%d", attrValue);
  NCCLCHECK(xmlGetSubKv(node, subName, sub, attrName, strValue));
  return ncclSuccess;
}

// 新增一个指定 name 的 node，没有属性，并与指定的父节点建立关联
static ncclResult_t xmlAddNode(struct ncclXml* xml, struct ncclXmlNode* parent, const char* subName,
                               struct ncclXmlNode** sub) {
  // 如果没有剩余 node slot 就报错
  if (xml->maxIndex == xml->maxNodes) {
    WARN("Error : too many XML nodes (max %d)", xml->maxNodes);
    return ncclInternalError;
  }
  // 新增一个 node，初始化
  struct ncclXmlNode* s = xml->nodes + xml->maxIndex++;
  s->nSubs = 0;
  s->nAttrs = 0;
  *sub = s;
  s->parent = parent;
  if (parent) {
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : too many XML subnodes (max %d)", MAX_SUBS);
      return ncclInternalError;
    }
    parent->subs[parent->nSubs++] = s;
  }
  strncpy(s->name, subName, MAX_STR_LEN);
  s->name[MAX_STR_LEN] = '\0';
  return ncclSuccess;
}

// 删除 xml 中指定节点
static ncclResult_t xmlRemoveNode(struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  struct ncclXmlNode* parent = node->parent;  // 记录 parent 节点
  if (parent == NULL) return ncclSuccess;
  int shift = 0;
  // 遍历 parent 的所有子节点，删除当前节点，之后所有节点往前移一格
  for (int s = 0; s < parent->nSubs; s++) {
    if (parent->subs[s] == node) shift = 1;
    else if (shift) parent->subs[s - 1] = parent->subs[s];
  }
  parent->nSubs--;
  return ncclSuccess;
}

// 将 src 子树拷贝到 dst 中，并且关联到 dst 中的 parent
static ncclResult_t xmlAddTree(struct ncclXml* dst, struct ncclXmlNode* parent, struct ncclXmlNode* srcNode) {
  if (dst->maxIndex == dst->maxNodes) {  // dst 满了，报错
    WARN("Error : too many XML nodes (max %d)", dst->maxNodes);
    return ncclInternalError;
  }
  struct ncclXmlNode* dstNode = dst->nodes + dst->maxIndex++;  // 找到 node 空的起始槽位
  *dstNode = *srcNode;  // 节点深拷贝
  dstNode->parent = parent;
  if (parent) {
    if (parent->nSubs == MAX_SUBS) {  // 如果有 parent 但 parent 的子节点槽位满了，报错
      WARN("Error : too many XML subnodes (max %d)", MAX_SUBS);
      return ncclInternalError;
    }
    parent->subs[parent->nSubs++] = dstNode;  // parent 关联 dst 节点
  }
  dstNode->nSubs = 0;  // dst 节点的子节点初始化为 0，因为要后续关联拷贝到 dst 的子节点
  // 递归地将 src 中的子树拷贝到 dst 中
  // Recursively copy the subtree(s)
  for (int i = 0; i < srcNode->nSubs; i++) NCCLCHECK(xmlAddTree(dst, dstNode, srcNode->subs[i]));
  return ncclSuccess;
}

// Dictionary for STR -> INT conversions. No dictionary size information,
// there needs to be a last element with str == NULL.

inline void printMissingTopoDictValueHint() {
  INFO(NCCL_GRAPH,
       "HINT: In many cases this issue indicates missing or faulty information in the provided topology file.");
  INFO(NCCL_GRAPH, "HINT: To confirm, set NCCL_TOPO_DUMP_FILE=topo.xml to produce the topology NCCL has detected and "
                   "compare to the one provided.");
}

struct kvDict {
  const char* str;
  int value;
};

// 通过映射表将字符串转换为对应的整型
static ncclResult_t kvConvertToInt(const char* str, int* value, struct kvDict* dict) {
  struct kvDict* d = dict;
  while (d->str) {
    if (strncmp(str, d->str, strlen(d->str)) == 0) {
      *value = d->value;
      return ncclSuccess;
    }
    d++;
  }
  INFO(NCCL_GRAPH, "KV Convert to int : could not find value of '%s' in dictionary, falling back to %d", str, d->value);
  printMissingTopoDictValueHint();
  *value = d->value;
  return ncclSuccess;
}

static ncclResult_t kvConvertToStr(int value, const char** str, struct kvDict* dict) {
  struct kvDict* d = dict;
  while (d->str) {
    if (value == d->value) {
      *str = d->str;
      return ncclSuccess;
    }
    d++;
  }
  WARN("KV Convert to str : could not find value %d in dictionary", value);
  printMissingTopoDictValueHint();
  return ncclInternalError;
}

#endif
