/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <ctype.h>
#include <float.h>
#include "core.h"
#include "nvmlwrap.h"
#include "xml.h"
#include "os.h"
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// Arbitrarily large number for constructing virtual topology string
#define NCCL_MAX_XML_DEPTH 1024

/*******************/
/* XML File Parser */
/*******************/

typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclXml*, struct ncclXmlNode*);

// 元素名 -> 处理函数的绑定
struct xmlHandler {
  const char* name;
  xmlHandlerFunc_t func;
};

#if defined(NCCL_OS_LINUX)

/*
单字节读取器，读一个字节，读到 EOF（fread 返回 0）就硬报 "Unexpected EOF"。
这是流中途耗尽的报错，而 xmlGetNode 开头那个 fread==0 是标签之间正常到文件尾，
走的是软返回，两者语义不同。
*/
ncclResult_t xmlGetChar(FILE* file, char* c) {
  /*
  size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
    ptr：接收读取数据的缓冲区指针；
    size：每个元素的大小；
    nmemb：要读取的元素个数
    stream：文件流指针
  */
  if (fread(c, 1, 1, file) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

/*
从文件流中读取 XML 属性 = 号后面的值，支持两种格式：
  1. 引号包裹的字符串
  2. 裸整数
读完后通过last出参告知调用方值之后的下一个字符是什么。整体流程为：
  调用方（xmlGetToken）遇到 '=' 后调用此函数
          │
          ▼
    xmlGetChar 读出第一个字符 c
          │
          ├── c 是引号（" 或 '）──► 进入引号模式
          │
          └── c 不是引号 ──────────► 进入裸整数模式（需 INT_OK 宏开启）
*/
ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(xmlGetChar(file, &c));  // 读取一个字符判断是不是引号
  if (c != '"' && c != '\'') {
// 裸整数模式（#if INT_OK）
#if INT_OK
    int o = 0;
    // 逐字符循环读下一个字符串，直到达到 255 长度，或者不是数字字符
    do {
      value[o] = c;
      if (o == MAX_STR_LEN - 1) {
        value[o] = '\0';
        WARN("Error : value %s too long (max %d)", value, MAX_STR_LEN);
        return ncclInternalError;
      }
      o++;
      NCCLCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0'; // 收尾 \0 终止符
    // 把那个非数字字符吐回给调用方（它可能是 >、空格等，对后续状态机有意义）
    *last = c;
    return ncclSuccess;
#else
    // 如果 INT_OK 未开启，则直接报错 "Expected (double) quote."
    WARN("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  // 引号模式（主路径）
  int o = 0;
  char quote = c;  // 记住是 " 还是 '，以便匹配闭合引号
  // 每轮循环先 xmlGetChar 读新字符，再写入 value[o]，直到配对引号
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    value[o] = c;
    if (o == MAX_STR_LEN - 1) {
      value[o] = '\0';
      WARN("Error : value %s too long (max %d)", value, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != quote);
  value[o - 1] = '\0';  // 尾部用 \0 覆盖写入的闭合引号
  NCCLCHECK(xmlGetChar(file, last));
  return ncclSuccess;
}

/*
NCCL XML 词法分析器的核心 token 读取函数，职责是从文件中读取一个 name（标签名或属性名），
如果紧跟 = 则继续读取对应的 value。
  file：XML 文件句柄
  name：输出缓冲区，存放 token 名称
  value：出缓冲区，存放属性值。传 NULL 表示调用方只期望读名称，不接受赋值
  last：告诉调用方 token 结束的那个分隔符字符（如 ' '、'>'、'/'），供上层状态机决定后续走向
*/
ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));  // 逐字符循环：通过 xmlGetChar 从文件一次读一个字符 c
    if (c == '=') { // 遇到 = 走 value 分支
      ptr[o] = '\0';  // 属性名 name 收尾
      if (value == NULL) {  // 如果预期没有属性值，遇到 = 则报错
        WARN("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return xmlGetValue(file, value, last);  // 读取属性值
    }
    // 往下都是属性名 name 字符累积
    ptr[o] = c;
    if (o == MAX_STR_LEN - 1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  // c 为以下任一分隔符时退出循环：' '（空格）、'>'（标签结束）、'/'（自闭合标记）、'\n'、'\r'
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o - 1] = '\0';  // 可能是只有属性名 name 的情况也需要收尾
  *last = c; // 把分隔符吐回给调用方，让上层状态机知道当前停在了哪种语法边界上
  return ncclSuccess;
}

/*
对一个长度为 3 的字符数组执行"左移一位，尾部追加新字符"操作，
等价于一个 固定容量为 3 的 FIFO 队列。每次调用后，s 中保留的是最近读入的 3 个字符。
*/
// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) \
  do { \
    s[0] = s[1]; \
    s[1] = s[2]; \
    s[2] = c; \
  } while (0)

/*
实现了 XML 注释（<!-- ... -->）的跳过逻辑，核心思想是用一个 3 字符滑动窗口 在文件流中单向扫描，直到窗口内容恰好等于 "-->" 为止。
  start：被 xmlGetToken 误读进 node->name 的注释正文残余（即 name+3，去掉了前缀 "!--"）
  next：xmlGetToken 结束时吐出的那个分隔符字符
xmlGetToken 的分隔符集合是 ' '、'>'、'/'、'\n'、'\r'，不包含 ! 和 -。所以当解析器遇到 <!-- some comment --> 时，
xmlGetToken 会把 !-- some comment --> 中直到第一个分隔符（可能是 > 或空格）之前的所有字符都塞进 name 缓冲区。
上层 xmlGetNode 通过 strncmp(name, "!--", 3) == 0 判定这是注释后，将 name+3（已读入的注释体片段）和 c（终止字符）传给 xmlSkipComment，
让它从已经被"吃掉"的位置继续扫描。
*/
ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";  // 初始化滑动窗口

  /*
  把 start（被误读的注释片段）和 next（终止字符）依次灌入窗口。> 本身就是 xmlGetToken 的分隔符，它不会出现在 name 内部。
  next 如果是 >，注入后窗口可能刚好变成 "-->" ，这时主循环直接跳过。
  */
  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i = 0; i < strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // 扫描到 "-->" 为止，如果之前是空格分割的 (<!-- some comment -->)，还需要往后
  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      WARN("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

/*
NCCL 自研 XML 解析器的词法器（lexer），从 FILE* 读出下一个 XML 标签，
填好一个 ncclXmlNode 的 type/name/attrs/nAttrs 四个字段。它不建父子关系、不懂文法，
那是调用方 xmlLoadSub（语法器/递归下降引擎）的活。二者分工：
  xmlGetNode = 词法层：把字节流切成一个个"标签 token"，产出 type 四态。
  xmlLoadSub = 语法层：用 type 四态驱动建树（挂 subs[]/parent、匹配 handler 表递归）。

*/
ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;  // 默认置为 None
  char c = ' ';  // c 先种一个 ' ' 保证进循环读第一个真实字符
  /*
  跳过空格/换行/回车，不跳 \t（制表符）。这是个小局限，
  标签间若夹 tab 会让下一步 c != '<' 报错（NCCL 自己生成的 XML 不用 tab，故无碍）。
  */
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;  // 到文件尾时软返回 ncclSuccess 且 type 保持 NONE。
  }
  if (c != '<') {  // 跳过空白后，下一个有意义字符必须是 < 开头
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // 读元素名进 node->name，c 更新为名字之后的分隔符（空格/>///换行）
  // Read XML element name
  NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // 注释为 <!--... ，由于分隔符不含 !/-，会让 name 以 "!--" 开头
  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    /*
    把误读进 name 的注释正文残余（name+3）和终止字符 c 先注入一个 3 字符滑动窗口 end[4]（SHIFT_APPEND 宏左移追加），
    再 fread 扫到 strcmp(end,"-->")==0。这样无论 --> 是否已被读进 name（如 <!--x--> 无空格时 name 可能已含 -->）都能正确定位。
    */
    NCCLCHECK(xmlSkipComment(file, node->name + 3, c));
    return xmlGetNode(file, node);
  }

  // 如果是闭合标签 </foo>，读取到的 name 会是空的。判断 name 是否为空并且分割字符是否是 / 可以确定是闭合标签
  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;  // 设置闭合标签类型
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));  // 闭合标签名当作无属性值的属性 token 读取
    if (c != '>') {  // 闭合标签没有属性值，必定直接结束 >
      WARN("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  // 下面正常处理打开标签
  node->type = NODE_TYPE_OPEN;

  // 遍历获取属性
  // Get Attributes
  int a = 0;
  while (c == ' ') {  // 后续如果是空格分割符说明有其他属性
    // 获取一个属性 token，包括属性名和属性值
    NCCLCHECK(xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {  // 多余属性在第 17 个额外的溢出吸收 slot 不断覆盖
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  // 如果是自闭合标签 <foo/>
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;  // 修改标签类型
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(file, str, NULL, &c));  // 再调一次 xmlGetToken 把游标推进到 >
  }
  if (c != '>') {  // 最后剩下的合法情况必定为 > 符号，如果不是就报错
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[],
                        int nHandlers) {
  /*
  若当前元素是 <foo/> 这种自闭合（NODE_TYPE_SINGLE），它没有子节点，直接返回。
  NODE_TYPE 四态：NONE=0/OPEN=1/CLOSE=2/SINGLE=3。
  */
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    // 容量超限了
    if (xml->maxIndex == xml->maxNodes) {
      WARN("Error : XML parser is limited to %d nodes", xml->maxNodes);
      return ncclInternalError;
    }
    // 获取尾部待填写的 node slot
    struct ncclXmlNode* node = xml->nodes + xml->maxIndex;
    memset(node, 0, sizeof(struct ncclXmlNode));  // 清零
    // 词法分析：读下一个标签进 node（填 type/name/attrs）
    NCCLCHECK(xmlGetNode(file, node));
    // NODE_TYPE_NONE 表示"没有更多 token"
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        // 在子元素里遇到它 = 父元素没闭合就到文件尾（"unterminated" 错误）
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // 在顶层遇到它 = 正常结束
        // All done
        return ncclSuccess;
      }
    }
    /*
    读到 </xxx>：若名字和当前 head 对不上就是标签错配（"XML Mismatch"）。
    对上则返回，这一 return 就是递归下降的"出栈"，回到父元素的 xmlLoadSub 循环继续。
    */
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    // 只有 NODE_TYPE_OPEN 才往下递归
    int found = 0;
    // 匹配所有 handlers
    for (int h = 0; h < nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {  // 如果匹配到
        if (head) {  // 如果有父节点
          if (head->nSubs == MAX_SUBS) {
            // 父节点的字节点插槽满了
            WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
            return ncclInternalError;
          }
          // 父节点指向当前节点
          head->subs[head->nSubs++] = node;
        }
        node->parent = head;  // 当前节点也指向父节点
        node->nSubs = 0;
        xml->maxIndex++; // 说明只有打开标签占用节点，闭合标签不占节点
        /*
        调用该元素对应的 handler 函数递归解析它的子元素
        每个 ncclTopoXmlLoadXxx 都声明"我这个元素允许哪些子元素"，这些 handler 数组连起来就是拓扑 XML 的文法（产生式）：
        元素	                    允许的子元素（handler 表）
        system（根）	             cpu
        cpu	                      pci、nic
        pci	                      pci（可自嵌套）、gpu、nic、pcilink
        gpu	                      nvlink、c2c
        nic	                      net
        nvlink/c2c/pcilink/net	  无（叶子：xmlLoadSub(..., NULL, 0) 只消费子树）
        */
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    // 如果没找到解析的 handler，优雅跳过整棵子树
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      // 用【0 个 handler】递归，把这个元素及其子树整个消费掉
      NCCLCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

#endif

/**************/
/* XML Writer */
/**************/

/*
把 XML 树里 parent/subs[] 这些本进程绝对地址指针，在发送前转成相对 nodes 起点的字节偏移（序列化 exp=1）
接收后用新地址空间的 base 加回偏移还原成真实指针（反序列化 exp=0）
*/
// exp == 1 -- serialize; exp == 0 -- deserialize
ncclResult_t ncclTopoConvertXml(struct ncclXml* xml, uintptr_t base, int exp) {
  for (int n = 0; n < xml->maxIndex; n++) {  // 遍历所有已用槽
    struct ncclXmlNode* node = &xml->nodes[n];

    /*
    所有非空的 parent 指针计算相对于 base 的相对地址，+1 是为了将指向 base 的指针与 NULL 区分开
      序列化：偏移 = 指针 - base + 1
      反序列化：指针 = base - 1 + 偏移
    */
    // For "parent", we shift the base by 1 so that we can distinguish actual
    // NULL pointers from pointers pointing to the first node.
    if (node->parent) {
      node->parent =
        (struct ncclXmlNode*)(exp ? ((uintptr_t)node->parent - base + 1) : (base - 1 + (uintptr_t)node->parent));
    }

    // 处理当前节点的所有子节点指针，没有 +1
    for (int s = 0; s < node->nSubs; s++) {
      node->subs[s] =
        (struct ncclXmlNode*)(exp ? ((uintptr_t)node->subs[s] - base) : (base + (uintptr_t)node->subs[s]));
    }
  }
  return ncclSuccess;
}

// 递归输出每个节点到 dump 文件，注意缩进和自闭合叶节点
ncclResult_t ncclTopoDumpXmlRec(int indent, FILE* file, struct ncclXmlNode* node) {
  for (int i = 0; i < indent; i++) fprintf(file, " ");  // 输出指定空格数量的缩进
  fprintf(file, "<%s", node->name);  // 输出开标签 + 标签名，例如 <system

  for (int a = 0; a < node->nAttrs; a++) {  // 逐个属性输出 key="value"，例如 <system version="1"
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {  // 没有子节点，说明是叶子，自闭合，例如 <net name="mlx5_0" speed="200" .../>
    fprintf(file, "/>\n");
  } else {  // 有子节点，输出闭合开标签 + 递归子节点 + 闭标签
    fprintf(file, ">\n");  // <system version="1">
    for (int s = 0; s < node->nSubs; s++) {  // 递归遍历所有子节点，主要是缩进 +2
      NCCLCHECK(ncclTopoDumpXmlRec(indent + 2, file, node->subs[s]));
    }
    for (int i = 0; i < indent; i++) fprintf(file, " ");  // 闭标签缩进
    fprintf(file, "</%s>\n", node->name);  // 输出闭标签
    /*
    最后输出类似如下：
    <system version="1">
      ...
    </system>
    */
  }
  return ncclSuccess;
}

// 融合后 XML 拓扑树的落盘器，把内存里的 ncclXml 树序列化成标准 XML 文本写进文件
ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlTopoFile, "w");  // 打开 dump 文件
  if (file == NULL) {
    INFO(NCCL_GRAPH | NCCL_ENV, "Unable to open %s, not dumping topology.", xmlTopoFile);
    return ncclSuccess;
  }
  // 从 system 根递归打印，缩进从 0 起
  NCCLCHECK(ncclTopoDumpXmlRec(0, file, xml->nodes));
  fclose(file);
  return ncclSuccess;
}

// 递归合并两个 xml 子树的并集。匹配指定两个节点层级是否有匹配的子节点，如果匹配到相同节点则继续递归，否则就将 src 子节点子树拷贝合并到 dst 节点中
static ncclResult_t xmlTopoFuseXmlRecursive(struct ncclXml* dst, struct ncclXmlNode* dstParent,
                                            struct ncclXmlNode* srcParent) {
  for (int i = 0; i < srcParent->nSubs; i++) {  // 从 src 当前节点的子节点开始遍历
    struct ncclXmlNode* srcNode = srcParent->subs[i];  // 直接取得 src 的子节点
    struct ncclXmlNode* dstNode;
    NCCLCHECK(xmlFindNode(dstParent, srcNode, &dstNode));  // 在 dst 的子节点中查找是否有 name、type、属性全都匹配 srcNode 的节点
    if (dstNode == NULL) {
      NCCLCHECK(xmlAddTree(dst, dstParent, srcNode)); // 没匹配到，就把 src 的当前子节点拷贝到 dst
    } else {
      NCCLCHECK(xmlTopoFuseXmlRecursive(dst, dstNode, srcNode));  // 匹配成功，则往下递归对比匹配成功的两节点层级
    }
  }
  return ncclSuccess;
}

/*
两棵拓扑 XML 树的并集去重器，它把 src 树合并进 dst 树，规则是身份相同的节点只保留 dst 里那一份、只递归融合子层，
dst 里不存在的节点整棵子树拷进来。它自己只是找 system 根，分派的薄壳，真正干活的是三个函数，
xmlTopoFuseXmlRecursive（递归下钻）、xmlFindNode（身份匹配）、xmlAddTree（整棵拷贝）。
*/
ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
  struct ncclXmlNode* topNodeDst;
  NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));  // 先找到 dst 的根节点 <system>

  if (topNodeDst == NULL) {  // dst 是空树(还没有 system)
    xmlAddTree(dst, NULL, src->nodes);  // 把 src 整棵树直接拷贝到 dst
    return ncclSuccess;
  }

  struct ncclXmlNode* topNodeSrc;
  NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));  // 也找到 src 的根节点 <system>

  // 此时 dst 和 src 都位于同层级的根节点 <system>，同层匹配则下钻，不匹配则整棵挂
  NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

  return ncclSuccess;
}

#if NCCL_OS_LINUX

/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // 已经到递归尾部，要么 head 是自闭合，要么读取到闭合标签、要么没有 handler 处理直接返回
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPciLink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // 已经到递归尾部，要么 head 是自闭合，要么读取到闭合标签、要么没有 handler 处理直接返回
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // 已经到递归尾部，要么 head 是自闭合，要么读取到闭合标签、要么没有 handler 处理直接返回
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}
ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // gpu 没有额外需要处理的属性，其下子标签有：<nvlink>、<c2c>
  struct xmlHandler handlers[] = {{"nvlink", ncclTopoXmlLoadNvlink}, {"c2c", ncclTopoXmlLoadC2c}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // 已经到递归尾部，要么 head 是自闭合，要么读取到闭合标签、要么没有 handler 处理直接返回
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // nic 没有额外需要处理的属性，其下子标签为：<net>
  struct xmlHandler handlers[] = {{"net", ncclTopoXmlLoadNet}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // pci 没有额外需要处理的属性，其下子标签有：<pci>(嵌套)、<gpu>、<nic>、<pcilink>
  struct xmlHandler handlers[] = {{"pci", ncclTopoXmlLoadPci},
                                  {"gpu", ncclTopoXmlLoadGpu},
                                  {"nic", ncclTopoXmlLoadNic},
                                  {"pcilink", ncclTopoXmlLoadPciLink}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 4));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // cpu 没有太多额外要处理的，其下的子标签有 <pci> 和 <nic>
  struct xmlHandler handlers[] = {{"pci", ncclTopoXmlLoadPci}, {"nic", ncclTopoXmlLoadNic}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));  // 从 <system> 标签的 version 属性中获取整型版本 
  if (version != NCCL_TOPO_XML_VERSION) {  // version 为 1
    WARN("XML Topology has wrong version %d, %d needed", version, NCCL_TOPO_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));  // 从 <system> 标签获取拓扑的 name 属性值
  if (name != NULL) INFO(NCCL_GRAPH, "Loading topology %s", name);
  else INFO(NCCL_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = {{"cpu", ncclTopoXmlLoadCpu}};
  // <system> 标签下的子标签是 <cpu>
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

#endif // NCCL_OS_LINUX (XML file parsers)

ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml, int warn) {
// 文件拓扑加载只在 Linux 实现，Windows 分支是空操作。
#if NCCL_OS_LINUX
  FILE* file = fopen(xmlTopoFile, "r");
  /*
  用户显式设 NCCL_TOPO_FILE，调用方传 warn=1，打不开时 INFO 报一句（带 strerror(errno) 说明原因），但不失败；
  走默认路径 /var/run/nvidia-topologyd/virtualTopology.xml，传 warn=0，打不开就静默返回（因为它本就是可选的，多数机器上没有）。
  */
  if (file == NULL) {
    if (warn) {
      INFO(NCCL_GRAPH | NCCL_ENV, "Could not open XML topology file %s : %s", xmlTopoFile, strerror(errno));
    }
    return ncclSuccess;  // 打不开也返回成功
  }
  INFO(NCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  // 顶层只认 <system> 这一个根元素，匹配到就交给 ncclTopoXmlLoadSystem 处理。这是整棵解析树的入口产生式。
  struct xmlHandler handlers[] = {{"system", ncclTopoXmlLoadSystem}};
  // 把扁平数组的"已用节点数"清零（maxNodes 容量不变），确保从空开始填，不受之前内容干扰
  xml->maxIndex = 0;
  // 启动递归下降解析，head=NULL 表示"顶层、没有父节点"，1 是 handler 数量。
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
#elif NCCL_OS_WINDOWS
  (void)xmlTopoFile;
  (void)xml;
  (void)warn;
#endif
  return ncclSuccess;
}

/**********************/
/* XML creation       */
/* from autodetection */
/**********************/

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

ncclResult_t ncclTopoSetAttrFromSys(struct ncclXmlNode* pciNode, const char* path, const char* fileName,
                                    const char* attrName) {
  char strValue[MAX_STR_LEN];
  // 从 {path}/{fileName} 文件中读取值到 strValue
  NCCLCHECK(ncclOsTopoGetStrFromSys(path, fileName, strValue, MAX_STR_LEN));
  // 读到了有效值，将值设置到 pci node 的指定属性
  if (strValue[0] != '\0') NCCLCHECK(xmlSetAttr(pciNode, attrName, strValue));
  TRACE(NCCL_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName, strValue);
  return ncclSuccess;
}

// 从 NVML 获取 GPU 的 PCI 身份信息，格式化成十六进制字符串后写入 XML <pci> 节点的属性。它与 ncclTopoSetAttrFromSys（从 sysfs 文件读取）是一对互补函数。
ncclResult_t ncclTopoSetAttrFromNvml(struct ncclXmlNode* pciNode, nvmlDevice_t device, const char* attrName) {
  nvmlPciInfo_t pciInfo;
  ncclResult_t ret = ncclNvmlDeviceGetPciInfo(device, &pciInfo);  // 从 nvml 中获取指定设备的 info
  if (ret != ncclSuccess) return ret;

  // 按属性名拆包并格式化
  char strValue[MAX_STR_LEN];
  strValue[0] = '\0';
  if (strcmp(attrName, "vendor") == 0) {
    snprintf(strValue, MAX_STR_LEN, "0x%x", pciInfo.pciDeviceId & 0xFFFF);
  } else if (strcmp(attrName, "device") == 0) {
    snprintf(strValue, MAX_STR_LEN, "0x%x", (pciInfo.pciDeviceId >> 16) & 0xFFFF);
  } else if (strcmp(attrName, "subsystem_vendor") == 0) {
    snprintf(strValue, MAX_STR_LEN, "0x%x", pciInfo.pciSubSystemId & 0xFFFF);
  } else if (strcmp(attrName, "subsystem_device") == 0) {
    snprintf(strValue, MAX_STR_LEN, "0x%x", (pciInfo.pciSubSystemId >> 16) & 0xFFFF);
  }
  if (strValue[0] == '\0') return ncclInternalError;

  NCCLCHECK(xmlSetAttr(pciNode, attrName, strValue));  // 设置 pci node 上述 4 个指定属性值
  TRACE(NCCL_GRAPH, "Read from NVML %s=%s", attrName, strValue);
  return ncclSuccess;
}

/*
<cpu> 节点的属性填充器，当向上冒泡找到或新建了一个 <cpu> 节点后，递归链会调用此函数把该 CPU 的 NUMA 亲和性、架构、厂商、型号等信息填入 XML。
它与 ncclTopoGetXmlFromSys（填 <pci> 节点）是对称的两条递归终止路径之一。整体设计遵循同一套惯用法：xmlGetAttrIndex == -1（缺失才填），
已有值不覆盖，尊重 NCCL_TOPO_FILE 预设。
*/
ncclResult_t ncclTopoGetXmlFromCpu(struct ncclXmlNode* cpuNode, struct ncclXml* xml) {
  int index;
  // 从 cpu node 获取 NUMA affinity(CPU 亲和性)属性，没有再填
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char* numaId;
    NCCLCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    // numaId 不能缺失，因为 numaId 是前面冒泡时从 sysfs numa_node 读出并设置的
    if (numaId == NULL) {
      WARN("GetXmlFromCpu : could not find CPU numa ID.");
      return ncclInternalError;
    }
    // Set affinity using OS-specific implementation
    unsigned int nodeNumber = (unsigned int)strtoul(numaId, NULL, 0);
    char affinityStr[MAX_STR_LEN];
    // 获取当前 numaId 对应的逻辑 CPU Mask 字符串，例如：00000000,0000ffff,ffff0000,00000000
    NCCLCHECK(ncclOsGetNumaNodeAffinity(nodeNumber, affinityStr, sizeof(affinityStr)));
    NCCLCHECK(xmlSetAttr(cpuNode, "affinity", affinityStr));  // 设置 affinity 属性
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));  // 尝试获取 CPU arch 架构属性
  if (index == -1) {
    /*
    纯编译期决定，用预定义宏检测当前编译目标架构，写入 arch 属性。不是运行时探测，因为一个进程必然跑在确定架构上。支持的三种架构：
      宏	                              写入值	    对应平台
      __PPC__	                          "ppc64"	   IBM POWER
      __aarch64__	                      "arm64"	   ARM 64 位（如 Grace、Ampere）
      __x86_64__ / _M_X64 / _M_AMD64	  "x86_64"	 Intel/AMD x86-64
    */
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#else
    WARN("getXmlFromCpu: Unknown CPU architecture");
    return ncclInternalError;
#endif
  }

  // 获取 CPU 厂商（仅 x86）
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

#if NCCL_OS_LINUX
    unsigned unused;
    /*
    利用 x86 CPUID 指令（leaf 0）获取 CPU 厂商标识字符串，执行 CPUID 指令、EAX=0 时，处理器返回：
      EAX = 支持的最大 leaf 编号（此处不用，存入 unused）
      EBX、EDX、ECX = 12 字节厂商字符串（按 EBX → EDX → ECX 的字节序排列）
    __cpuid 是编译器提供的内建函数（intrinsic），直接映射到 x86 的 CPUID 汇编指令。但 GCC/Clang 和 MSVC 的签名不同，NCCL 代码里两种都出现了。
    x86 的 CPUID 指令行为是：
      输入：EAX 寄存器放"leaf 编号"（查询类别），部分 leaf 还需要 ECX 作为子叶（sub-leaf）
      输出：结果分别写入 EAX、EBX、ECX、EDX 四个寄存器
    Linux（GCC/Clang）版本 5 个参数，全部是引用/输出：
      参数位置	  含义	            方向
      第 1 个	   leaf 编号（输入）	 → CPU
      第 2 个	   输出 EAX 的值	    CPU →
      第 3 个	   输出 EBX 的值	    CPU →
      第 4 个	   输出 ECX 的值	    CPU →
      第 5 个	   输出 EDX 的值	    CPU →
    leaf 0 的 EAX 返回"支持的最大 leaf 编号"，NCCL 不需要它所以存入 unused；EBX/ECX/EDX 返回厂商字符串的 12 字节。
    常见 leaf 速查：
      Leaf (EAX 输入)	   返回内容
      0	                EAX=最大 leaf；EBX+EDX+ECX = 12 字节厂商字符串
      1	                EAX=Family/Model/Stepping 位域；EDX/ECX=特性标志位
      2	                缓存/TLB 描述符
      4	                确定性缓存参数（需 sub-leaf）
      0x80000002–4	    处理器品牌字符串（48 字节）

    注意寄存器顺序不是 EBX/ECX/EDX，而是 EBX/EDX/ECX——这是 Intel 手册定义的"反直觉"排列。代码用 union 精确映射这个顺序：
      union 内存布局（小端）:
        字节: [0  1  2  3 | 4  5  6  7 | 8  9  10 11]
        字段: [   ebx     |    edx     |    ecx     ]
        对应: [  "Genu"   |   "ineI"   |   "ntel"   ]  ← Intel 示例
    常见结果：
      厂商	  CPUID 返回字符串
      Intel	 "GenuineIntel"
      AMD	   "AuthenticAMD"
    */
    __cpuid(0, unused, cpuid0.ebx, cpuid0.ecx, cpuid0.edx);
#elif NCCL_OS_WINDOWS
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    cpuid0.ebx = cpuInfo[1];
    cpuid0.edx = cpuInfo[3];
    cpuid0.ecx = cpuInfo[2];
#endif
    // 拷贝到 13 字节缓冲区加 \0 截断
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    NCCLCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  // 设置 CPU Family/Model ID（仅 x86）
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    /*
    :4、:2、:8 这些是 C/C++ 的位域（bit-field）声明语法，表示该字段只占结构体中的指定位数，而非完整的 unsigned（通常 32 位）。
    通过 cpuid1.val = ... 可以一次性写入整个 32 位，通过 cpuid1.familyId、cpuid1.extModelId 等可以按字段名读取其中几个 bit，无需手动做移位和掩码。
    */
    union {
      struct {
        unsigned steppingId:4;
        unsigned modelId:4;
        unsigned familyId:4;
        unsigned processorType:2;
        unsigned resv0:2;
        unsigned extModelId:4;
        unsigned extFamilyId:8;
        unsigned resv1:4;
      };
      uint32_t val;
    } cpuid1;
#if NCCL_OS_LINUX
    unsigned unused;
    // leaf 1 的 EAX 包含 Family/Model/Stepping 位域信息，其余三个寄存器此处不关心。
    __cpuid(1, cpuid1.val, unused, unused, unused);
#elif NCCL_OS_WINDOWS
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    cpuid1.val = cpuInfo[0];  // EAX contains the processor info
#endif
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);  // 4位基础 + 8位扩展
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);  // 4位基础 + 4位扩展
    /*
    常见结果示例：
      CPU	                    DisplayFamily	  DisplayModel	备注
      Intel Sapphire Rapids	  6 (0x06)	      143 (0x8F)	  familyId=6, extFamilyId=0
      AMD Genoa (Zen 4)	      25 (0x19)	      17 (0x11)	    familyId=0xF, extFamilyId=0xA → 0xF + 0xA0 = 0xAF?
    为什么 NCCL 需要 familyid/modelid？
      下游拓扑算法（尤其是路径计算 ncclTopoComputePaths）用 CPU family/model 来判断某些微架构特性，
      例如特定 Intel CPU 的 IIO 栈结构会影响 PCIe P2P 路径是否真正走直连，还是被迫绕行 UPI。
    */
    NCCLCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));  // 设置 familyid 属性
    NCCLCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));  // 设置 modelid 属性
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclTopoGetPciNode(struct ncclXml* xml, const char* busId, struct ncclXmlNode** pciNode) {
  // 全 nodes 遍历查找 <pci> node，该 node 有 busid 属性且值等于 busId
  NCCLCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  // 如果没找到指定 node
  if (*pciNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, NULL, "pci", pciNode));  // 增加一个 pci nod，先悬空
    NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));  // 设置 gpu 的 busId 为 busid 属性
  }
  return ncclSuccess;
}

// Check whether a string is in BDF format or not.
// BDF (Bus-Device-Function) is "BBBB:BB:DD.F" where B, D and F are hex digits.
// There can be trailing chars.
int isHex(char c) {
  return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'));
}

// 检查该目录名是否为 BDF 格式（DDDD:BB:DD.F，12 字符全十六进制+分隔符）
int checkBDFFormat(char* bdf) {
  if (strlen(bdf) != 12) return 0;
  if ((bdf[4] != ':') || (bdf[7] != ':') || (bdf[10] != '.')) return 0;
  if ((isHex(bdf[0]) == 0) || (isHex(bdf[1]) == 0) || (isHex(bdf[2]) == 0) || (isHex(bdf[3]) == 0) ||
      (isHex(bdf[5]) == 0) || (isHex(bdf[6]) == 0) || (isHex(bdf[8]) == 0) || (isHex(bdf[9]) == 0) ||
      (isHex(bdf[11]) == 0))
    return 0;
  return 1;
}

ncclResult_t ncclTopoGetXmlFromSys(struct ncclXmlNode* pciNode, struct ncclXml* xml) {
  ncclResult_t ret = ncclSuccess;
  const char* vendor = NULL;
  struct ncclXmlNode* parent = NULL;
  char* peers = NULL;

  // Fill info, then parent
  const char* busId;
  NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));  // 从 pci node 属性中获取 GPU busId
  nvmlDevice_t device;
  bool nvmlDeviceFound = false;

#if NCCL_OS_LINUX
  char* path = NULL;
  const char* deviceClass = NULL;
  // 获取 busId 的 pci 路径，例如：/sys/devices/pci0000:00/0000:00:1c.0/0000:3b:00.0
  NOWARN(ncclOsGetPciPath(busId, &path), NCCL_GRAPH);
  // 从 /sys/devices/pci0000:00/0000:00:1c.0/0000:3b:00.0/class 文件中获取值并设置为 pci node 的 class 属性
  if (path) NCCLCHECKGOTO(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"), ret, exit);

  // 获取指定属性名在属性列表中的属性值字符串
  NCCLCHECKGOTO(xmlGetAttr(pciNode, "class", &deviceClass), ret, exit);
  /*
  判断当前这个 PCI 设备"是不是 GPU（或身份未知）"，若是则尝试向 NVML 索取该设备的句柄，作为后续填充属性的首选数据源。

  PCI 类码（Class Code）是 PCI/PCIe 规范定义的一套标准化设备分类编号。每个 PCI 设备都必须上报一个类码，用来告诉系统"我是哪一类设备"，
  是显卡、网卡、存储控制器，还是桥接器。操作系统和 NCCL 就是靠它来识别设备用途的。
  PCI 类码是一个 24 位（3 字节） 的字段，在 sysfs 的 class 文件里通常写成 0x 开头的 6 位十六进制，格式为：
  0x  BB   SS   PP
      │    │    └── Programming Interface（编程接口，8 位）
      │    └─────── Subclass（子类，8 位）
      └──────────── Base Class（基类，8 位）★ 最高字节，决定"大类"
  以 GPU 常见的 0x030000 为例：
    字节	                  值	  含义
    Base Class（基类）	    0x03	 Display controller（显示控制器）
    Subclass（子类）	      0x00	 VGA compatible controller（VGA 兼容控制器）
    Programming Interface	 0x00	  VGA controller
  再看一个 NVIDIA GPU 常见的 0x030200：基类仍是 0x03（显示控制器），子类 0x02 表示 3D controller（无 VGA 输出的纯计算卡，如数据中心 GPU）。

  0x03 指的是基类（最高字节）。PCI 规范给每个基类分配了固定编号，0x03 这个编号被官方定义为 "Display controller"（显示控制器）。常见的基类编号有：
    Base Class	  设备大类
    0x01	        Mass Storage Controller（存储控制器，如 SATA/NVMe）
    0x02	        Network Controller（网络控制器，如网卡）
    0x03	        Display Controller（显示控制器，即 GPU）
    0x04	        Multimedia Controller（多媒体）
    0x06	        Bridge Device（桥接器，如 PCIe 交换/根端口）
    0x08	        Base System Peripherals
  */
  if (deviceClass == NULL || deviceClass[0] == '\0' || strncmp(deviceClass, "0x03", 4) == 0) {
    ncclResult_t nvmlRet;
    // 通过 busId 尝试向 NVML 获取 GPU 句柄
    NOWARN(nvmlRet = ncclNvmlDeviceGetHandleByPciBusId(busId, &device), NCCL_GRAPH);
    if (nvmlRet == ncclSuccess) nvmlDeviceFound = true;
  }

#elif NCCL_OS_WINDOWS
  char* parentBusId = NULL;
  char deviceClass[MAX_STR_LEN];
  deviceClass[0] = '\0';
  bool isGpuDevice = false;
  if (ncclOsGetPciDeviceClassByBusId(busId, deviceClass, sizeof(deviceClass)) == ncclSuccess &&
      deviceClass[0] != '\0') {
    NCCLCHECK(xmlSetAttr(pciNode, "class", deviceClass));
    TRACE(NCCL_GRAPH, "Read from Windows SetupDi class=%s", deviceClass);
    isGpuDevice = (strncmp(deviceClass, "0x03", 4) == 0);
  } else {
    TRACE(NCCL_INIT, "ncclTopoGetXmlFromSys: Could not get device class for %s", busId);
  }
  if (isGpuDevice) {
    ncclResult_t winNvmlRet;
    NOWARN(winNvmlRet = ncclNvmlDeviceGetHandleByPciBusId(busId, &device), NCCL_GRAPH);
    if (winNvmlRet == ncclSuccess) nvmlDeviceFound = true;
  }
#endif

  /*
  PCI 配置空间是每个 PCI/PCIe 设备必须实现的一块标准化寄存器区域，操作系统通过它来发现设备身份、配置资源、确定能力。
  规范把它定义为一个按字节编址、固定布局的寄存器表，每个字段在哪个偏移、占几个字节都是 PCI-SIG 组织硬性规定的，
  所有厂商的设备必须遵守，这样通用驱动/工具才能"盲读"。

  一、配置空间全景（前 64 字节 = 标准头部）
    PCI 规范定义了 Type 0 头部（普通设备）和 Type 1 头部（桥），前 64 字节布局如下（Type 0）：
      偏移	      大小	    字段名	                说明
      0x00	     2 字节	   Vendor ID	            厂商编号，由 PCI-SIG 统一分配
      0x02	     2 字节	   Device ID	            设备编号，由厂商自行定义
      0x04	     2 字节	   Command	              设备控制寄存器（I/O 使能、Memory 使能等）
      0x06	     2 字节	   Status	                设备状态
      0x08	     1 字节	   Revision ID	          硬件版本
      0x09	     3 字节	   Class Code	            基类 + 子类 + 编程接口（即前面说的 0x030000）
      0x0C	     1 字节	   Cache Line Size	
      0x0D	     1 字节	   Latency Timer	
      0x0E	     1 字节	   Header Type	          0=普通设备，1=桥
      0x0F	     1 字节	   BIST	自检
      0x10–0x24	 24 字节	 BAR 0–5	              Base Address Registers（地址空间映射）
      0x28	     4 字节	   CardBus CIS Pointer	
      0x2C	     2 字节	   Subsystem Vendor ID	  子系统/板卡厂商
      0x2E	     2 字节	   Subsystem              ID	子系统/板卡型号
      0x30	     4 字节	   Expansion ROM Base	
      0x34	     1 字节	   Capabilities Pointer	  指向能力链表首项
      0x3C	     1 字节	   Interrupt Line	
      0x3D	     1 字节	   Interrupt Pin	
  二、四个关键字段的语义
  0x00 — Vendor ID（16 位）
    这是 PCI-SIG 统一分配给各芯片/设备制造商的全球唯一编号。常见值：
      Vendor  ID	厂商
      0x10DE	NVIDIA
      0x8086	Intel
      0x1022	AMD
      0x15B3	Mellanox（现 NVIDIA Networking）
      0x1000	Broadcom / LSI（PCIe 交换芯片）
    操作系统枚举 PCI 总线时，读到 Vendor ID == 0xFFFF 表示该槽位无设备。
  0x02 — Device ID（16 位）
    由厂商自定义，标识具体产品型号。例如 NVIDIA 的 0x2330 可能是某一代 A100，0x26B5 可能是 H100。
  Vendor ID + Device ID 组合起来就是设备的"型号身份证"，驱动靠这对值匹配。
  0x2C — Subsystem Vendor ID（16 位）
    为什么需要"子系统"？因为同一颗芯片可能被不同板卡厂商使用。例如一块 Broadcom PCIe 交换芯片，可能出现在 Dell 服务器主板上，
  也可能出现在 Supermicro 主板上。Subsystem Vendor ID 标识的是板卡/系统级的制造商，允许驱动针对特定 OEM 做定制行为。
  0x2E — Subsystem ID（16 位）
    与 Subsystem Vendor ID 配对使用，标识具体板卡型号。比如同样是 NVIDIA A100 GPU 芯片，"SXM4 模组"和"PCIe 插卡版"可能有不同的 Subsystem ID，
  因为散热方案、供电设计不同，某些驱动行为需区分。

  设置 vendor/device/subsystem_vendor/subsystem_device 这 4 个身份属性，NVML 优先，fallback sysfs 兜底
  */
  int index;
  NCCLCHECKGOTONOWARN(xmlGetAttrIndex(pciNode, "vendor", &index), ret, exit, NCCL_GRAPH);
  if (index == -1) {
    if (nvmlDeviceFound) NOWARN(ncclTopoSetAttrFromNvml(pciNode, device, "vendor"), NCCL_GRAPH);
#if NCCL_OS_LINUX
    else if (path) NOWARN(ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor"), NCCL_GRAPH);
#endif
  }
  NCCLCHECKGOTONOWARN(xmlGetAttrIndex(pciNode, "device", &index), ret, exit, NCCL_GRAPH);
  if (index == -1) {
    if (nvmlDeviceFound) NOWARN(ncclTopoSetAttrFromNvml(pciNode, device, "device"), NCCL_GRAPH);
#if NCCL_OS_LINUX
    else if (path) NOWARN(ncclTopoSetAttrFromSys(pciNode, path, "device", "device"), NCCL_GRAPH);
#endif
  }
  NCCLCHECKGOTONOWARN(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index), ret, exit, NCCL_GRAPH);
  if (index == -1) {
    if (nvmlDeviceFound) NOWARN(ncclTopoSetAttrFromNvml(pciNode, device, "subsystem_vendor"), NCCL_GRAPH);
#if NCCL_OS_LINUX
    else if (path) NOWARN(ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor"), NCCL_GRAPH);
#endif
  }
  NCCLCHECKGOTONOWARN(xmlGetAttrIndex(pciNode, "subsystem_device", &index), ret, exit, NCCL_GRAPH);
  if (index == -1) {
    if (nvmlDeviceFound) NOWARN(ncclTopoSetAttrFromNvml(pciNode, device, "subsystem_device"), NCCL_GRAPH);
#if NCCL_OS_LINUX
    else if (path) NOWARN(ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device"), NCCL_GRAPH);
#endif
  }
  // 尝试获取 link_speed 属性索引
  NCCLCHECKGOTO(xmlGetAttrIndex(pciNode, "link_speed", &index), ret, exit);
  if (index == -1) {  // 没有 link_speed 属性
    if (nvmlDeviceFound) {
      unsigned int linkGen = 0;
      // 通过 NVML 获取 linkGen (PCIe 代际 1-6)
      if (ncclNvmlDeviceGetCurrPcieLinkGeneration(device, &linkGen) == ncclSuccess && linkGen > 0) {
        // 映射到 speeds[]（Gen1=2.5、Gen2=5、Gen3=8、Gen4=16、Gen5=32、Gen6=64 GT/s）
        const char* speeds[] = {
          "", "2.5 GT/s PCIe", "5.0 GT/s PCIe", "8.0 GT/s PCIe", "16.0 GT/s PCIe", "32.0 GT/s PCIe", "64.0 GT/s PCIe"
        };
        if (linkGen <= 6) {  // 如果 PCIe 代际 <= 6，合法，设置 pci node 的 link_speed 属性
          NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", speeds[linkGen]), ret, exit);
        }
      } else {
        // 查询失败或 linkGen==0，设置默认 "16.0 GT/s"
        NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", "16.0 GT/s"), ret, exit);
      }
    }
#if NCCL_OS_LINUX
    /*
    fallback 到 sysfs 路径，同时读设备自身 max_link_speed 和上游端口 ../max_link_speed，sscanf 出浮点，取较小者对应的字符串。
    因为链路实际速率由两端较弱一方决定（Gen5 设备插 Gen4 槽跑 Gen4），FLT_MAX 是初值哨兵，某端读不到（空串 sscanf 失败）就保留 FLT_MAX、自然被另一端压过。
    */
    else if (path) {
      char deviceSpeedStr[MAX_STR_LEN];
      float deviceSpeed = FLT_MAX;
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr, sizeof(deviceSpeedStr)), ret, exit);
      sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
      char portSpeedStr[MAX_STR_LEN];
      float portSpeed = FLT_MAX;
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr, sizeof(portSpeedStr)), ret, exit);
      /*
      当 deviceSpeedStr（或 portSpeedStr）为空串 "" 时，sscanf 立即发现输入已耗尽（首字符就是 \0），无法匹配 %f，
      返回 EOF（即 -1），且不会修改 deviceSpeed 的值，它保持原样不动。
      */
      sscanf(portSpeedStr, "%f GT/s", &portSpeed);
      NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr), ret,
                    exit);
    }
#endif
    else {
#if NCCL_OS_LINUX
      NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", ""), ret, exit);  // 最终兜底给 pci node 设置 link_speed 属性，值为空
#elif NCCL_OS_WINDOWS
      NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_speed", "16.0 GT/s"), ret, exit);
#endif
    }
  }
  // 与 link_speed 处理流程类似，不过属性值类型从字符串改为整型
  // Set link width from NVML (shared), sysfs fallback (Linux), or defaults
  NCCLCHECKGOTO(xmlGetAttrIndex(pciNode, "link_width", &index), ret, exit);
  if (index == -1) {
    if (nvmlDeviceFound) {
      unsigned int linkWidth = 0;
      if (ncclNvmlDeviceGetCurrPcieLinkWidth(device, &linkWidth) == ncclSuccess && linkWidth > 0) {
        NCCLCHECKGOTO(xmlSetAttrInt(pciNode, "link_width", linkWidth), ret, exit);
      } else {
        NCCLCHECKGOTO(xmlSetAttrInt(pciNode, "link_width", 16), ret, exit);
      }
    }
#if NCCL_OS_LINUX
    else if (path) {
      char strValue[MAX_STR_LEN];
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "max_link_width", strValue, MAX_STR_LEN), ret, exit);
      int deviceWidth = strtol(strValue, NULL, 0);
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "../max_link_width", strValue, MAX_STR_LEN), ret, exit);
      int portWidth = strtol(strValue, NULL, 0);
      NCCLCHECKGOTO(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth, portWidth)), ret, exit);
    }
#endif
    else {
#if NCCL_OS_LINUX
      NCCLCHECKGOTO(xmlSetAttr(pciNode, "link_width", ""), ret, exit);
#elif NCCL_OS_WINDOWS
      NCCLCHECKGOTO(xmlSetAttrInt(pciNode, "link_width", 16), ret, exit);
#endif
    }
  }

  /*
    特殊处理 Broadcom，当发现某个 <pci> 节点的 vendor 是 Broadcom 时，就去查询该 Switch 芯片内部维护的"虚拟交换链接表"，
  即哪些 endpoint 被该 Switch 直连，可以互相做 P2P。
    具体实现走的是 /sys/kernel/pci_switch_link/virtual_switch_links/<busId> 这个特殊的 sysfs 目录（由 Broadcom 的驱动程序导出），
  里面列出了该 Switch 下所有可 P2P 直通的 peer 设备的 busId。

  查到 peers 后，NCCL 会在该 Broadcom Switch 对应的 <pci> 节点下建 <pcilink> 子节点：
    <pci busid="0000:00:1c.0" class="0x060400" vendor="0x1000" ...>
      <pcilink target="0000:3b:00.0"/>
      <pcilink target="0000:3c:00.0"/>
      <pci busid="0000:3b:00.0" class="0x030200" vendor="0x10de" ...>
        <gpu dev="0" sm="90" .../>
      </pci>
      ...
    </pci>
  */
  NCCLCHECKGOTO(xmlGetAttr(pciNode, "vendor", &vendor), ret, exit);
  if (vendor != NULL && strcmp(vendor, "0x1000") == 0) {
    // BCM switch, look for P2P connections
    int nlinks;
    // 扫 /sys/kernel/pci_switch_link/virtual_switch_links/<busId> 目录，每个"格式合法且 ncclOsGetPciPath 能解析"的条目当作一个 P2P peer
    NCCLCHECKGOTO(ncclOsGetBcmLinks(busId, &nlinks, &peers), ret, exit);
    for (int l = 0; l < nlinks; l++) {
      char* target = peers + l * BUSID_SIZE;
      struct ncclXmlNode* linkNode;
      // 获取当前 pci node 下是否有 pcilink 子 node，并且属性 targe={target}
      NCCLCHECKGOTO(xmlGetSubKv(pciNode, "pcilink", &linkNode, "target", target), ret, exit);
      if (linkNode == NULL) {  // 如果没有这样的 pcilink，添加到当前 pci node 下，并指定  target 属性
        NCCLCHECKGOTO(xmlAddNode(xml, pciNode, "pcilink", &linkNode), ret, exit);
        NCCLCHECKGOTO(xmlSetAttr(linkNode, "target", target), ret, exit);
      }
    }
    free(peers);
    peers = NULL;
  }

  /*
  当这个 pci 节点还没有父（新建的、悬空的）才向上冒泡

  先理解 sysfs 路径与 PCIe 物理层级的映射，以一台有 PCIe Switch 的服务器为例，sysfs 路径直接镜像硬件拓扑：
    /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:02:00.0/0000:3b:00.0
                │              │               │               │              │
                │              │               │               │              └─ endpoint（GPU/NIC）
                │              │               │               └─ Switch 下行端口
                │              │               └─ Switch 上行端口
                │              └─ Root Port（根端口/桥）
                └─ CPU Root Complex（根复合体）
  而一个简单系统（GPU 直连 Root Port，无 Switch）：
    /sys/devices/pci0000:00/0000:00:1c.0/0000:3b:00.0
                │               │              │
                │               │              └─ endpoint
                │               └─ Root Port
                └─ CPU Root Complex
  */
  parent = pciNode->parent;
  if (parent == NULL) {
#ifdef NCCL_OS_LINUX
    if (path) {
      // 从当前 pci 路径下的 numa_node 文件中获取信息
      // Save that for later in case next step is a CPU
      char numaIdStr[MAX_STR_LEN];
      NCCLCHECKGOTO(ncclOsTopoGetStrFromSys(path, "numa_node", numaIdStr, MAX_STR_LEN), ret, exit);

      // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
      // switch, or stop if we reach a CPU root complex.
      int slashCount = 0;
      int parentOffset;
      // 往上依次遍历 pci 路径字符
      for (parentOffset = strlen(path) - 1; parentOffset > 0; parentOffset--) {
        if (path[parentOffset] == '/') { // 每遇到一个 /
          slashCount++;
          path[parentOffset] = '\0';  // 截断
          int start = parentOffset - 1;
          // 找到截断后末尾的 busId
          while (start > 0 && path[start] != '/') start--;
          // 检查该目录名是否为 BDF 格式（DDDD:BB:DD.F，12 字符全十六进制+分隔符）
          // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
          if (checkBDFFormat(path + start + 1) == 0) {  // 目录名不是 BDF（如 pci0000:00）→ CPU Root Complex
            /*
            在 <system> 下按 numaid 查找已有的 <cpu> 节点（find-or-create 模式）
            新建时填入 host_hash（用于跨 rank XML 融合时识别同一主机）和 numaid（之前已从 sysfs numa_node 文件读出）
            parent 设置完毕，后续 break 退出循环
            */
            // This a CPU root complex. Create a CPU tag and stop there.
            struct ncclXmlNode* topNode;
            NCCLCHECKGOTO(xmlFindTag(xml, "system", &topNode), ret, exit);  // 找到 system node
            NCCLCHECKGOTO(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr), ret, exit);  // 尝试获取 system 下的 cpu node 并且匹配 numaid 属性
            if (parent == NULL) {  // 如果满足要求的 cpu node 不存在
              // 在 system node 下增加 cpu node，并且设置 host_hash 和 numaid 属性
              NCCLCHECKGOTO(xmlAddNode(xml, topNode, "cpu", &parent), ret, exit);
              NCCLCHECKGOTO(xmlSetAttrLong(parent, "host_hash", getHostHash()), ret, exit);
              NCCLCHECKGOTO(xmlSetAttr(parent, "numaid", numaIdStr), ret, exit);
            }
          } else if (slashCount == 2) {  // 目录名是 BDF 且 slashCount == 2
            /*
            爬了两级 / 后，发现祖父目录仍是一个 BDF 格式的 PCI 设备，说明它是上层 PCI Switch（通常是 Switch 的上行端口）。此时：
              用截断后路径的最末段作为 busId，在 XML 中 find-or-create 一个 <pci busid="..."> 节点
              新建时 parent=NULL（悬空），后续递归调用 ncclTopoGetXmlFromSys(parent) 时会继续向上冒泡，直到找到 CPU
            
            跳过直接父级是因为 sysfs 中 endpoint 与上行端口之间往往隔着一个下行端口（或 Root Port），这层中间桥对 NCCL 的拓扑图没有独立建模价值。
            NCCL 只关心"设备挂在哪个 Switch/CPU 下"，不关心经过了几个中间端口。跳过后，Switch 的上行端口才成为 XML 树中有意义的 <pci> 节点。
            */
            // Continue on the upper PCI switch
            for (int i = strlen(path) - 1; i > 0; i--) {
              if (path[i] == '/') {  // 此时找到 root port 和 PCI switch 上行端口之间的 /
                NCCLCHECKGOTO(xmlFindTagKv(xml, "pci", &parent, "busid", path + i + 1), ret, exit);  // 尝试查找 PCI switch 上行端口的 pci node
                if (parent == NULL) {  // 如果没找到
                  // 增加一个悬空的 PCI switch 上行端口的 pci node
                  NCCLCHECKGOTO(xmlAddNode(xml, NULL, "pci", &parent), ret, exit);
                  NCCLCHECKGOTO(xmlSetAttr(parent, "busid", path + i + 1), ret, exit);
                }
                break;
              }
            }
          }
        }
        if (parent) break;
      }
      free(path);
      path = NULL;
    }
#elif NCCL_OS_WINDOWS
    INFO(NCCL_INIT, "ncclTopoGetXmlFromSys: Windows - creating parent node");
    if (nvmlDeviceFound) {
      char numaIdStr[MAX_STR_LEN] = "0";
      INFO(NCCL_INIT, "ncclTopoGetXmlFromSys: Using NUMA node %s (Windows default)", numaIdStr);

      // Get PCI device parent using Windows Setup API
      ncclResult_t result = ncclOsGetPciDeviceParent(device, &parentBusId);

      if (result == ncclSuccess && parentBusId != NULL) {
        // Check if parent is a valid PCI device (has BDF format) or CPU root complex
        if (checkBDFFormat(parentBusId) == 1) {
          // Continue on the upper PCI switch
          NCCLCHECKGOTO(xmlFindTagKv(xml, "pci", &parent, "busid", parentBusId), ret, exit);
          if (parent == NULL) {
            NCCLCHECKGOTO(xmlAddNode(xml, NULL, "pci", &parent), ret, exit);
            NCCLCHECKGOTO(xmlSetAttr(parent, "busid", parentBusId), ret, exit);
          }
        } else {
          // This is a CPU root complex. Create a CPU tag and stop there.
          struct ncclXmlNode* topNode;
          NCCLCHECKGOTO(xmlFindTag(xml, "system", &topNode), ret, exit);
          NCCLCHECKGOTO(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr), ret, exit);
          if (parent == NULL) {
            NCCLCHECKGOTO(xmlAddNode(xml, topNode, "cpu", &parent), ret, exit);
            NCCLCHECKGOTO(xmlSetAttrLong(parent, "host_hash", getHostHash()), ret, exit);
            NCCLCHECKGOTO(xmlSetAttr(parent, "numaid", numaIdStr), ret, exit);
          }
        }
      } else {
        // Failed to get parent - default to CPU root complex
        INFO(NCCL_GRAPH, "ncclTopoGetXmlFromSys: could not get PCI parent for %s, defaulting to CPU root complex",
             busId);
        struct ncclXmlNode* topNode;
        NCCLCHECKGOTO(xmlFindTag(xml, "system", &topNode), ret, exit);
        NCCLCHECKGOTO(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr), ret, exit);
        if (parent == NULL) {
          NCCLCHECKGOTO(xmlAddNode(xml, topNode, "cpu", &parent), ret, exit);
          NCCLCHECKGOTO(xmlSetAttrLong(parent, "host_hash", getHostHash()), ret, exit);
          NCCLCHECKGOTO(xmlSetAttr(parent, "numaid", numaIdStr), ret, exit);
        } else {
          INFO(NCCL_INIT, "ncclTopoGetXmlFromSys: CPU node already exists (default)");
        }
      }
    }
#endif
    else {
      // 兜底，find-or-create <cpu numaid="-1">
      // No information on /sys, attach GPU to unknown CPU
      NCCLCHECKGOTO(xmlFindTagKv(xml, "cpu", &parent, "numaid", "-1"), ret, exit);
      if (parent == NULL) {
        // 在 system node 下创建一个 cpu node，并设置 host_hash 和 numaid=-1 属性
        struct ncclXmlNode* topNode;
        NCCLCHECKGOTO(xmlFindTag(xml, "system", &topNode), ret, exit);
        NCCLCHECKGOTO(xmlAddNode(xml, topNode, "cpu", &parent), ret, exit);
        NCCLCHECKGOTO(xmlSetAttrLong(parent, "host_hash", getHostHash()), ret, exit);
        NCCLCHECKGOTO(xmlSetAttr(parent, "numaid", "-1"), ret, exit);
        /*
        给 cpu node 设置如下属性：
          属性名	    数据来源	                                          平台限制	  示例值	                  用途
          affinity	 sysfs /sys/devices/system/node/node<N>/cpumap	    全平台	   "00000000,0000ffff"	    NUMA 感知的线程绑核与内存分配
          arch	     编译期宏（__x86_64__ / __aarch64__ / __PPC__）	      全平台	   "x86_64"	                区分指令集架构
          vendor	   CPUID leaf 0（EBX+EDX+ECX）	                      仅 x86	   "GenuineIntel"	          CPU 厂商识别
          familyid	 CPUID leaf 1 EAX 位域（familyId + extFamilyId<<4）	 仅 x86	    6（Intel）/ 25（AMD Zen）	微架构代际判定
          modelid	   CPUID leaf 1 EAX 位域（modelId + extModelId<<4）	   仅 x86	    143（Sapphire Rapids）	 具体型号识别
        */
        NCCLCHECKGOTO(ncclTopoGetXmlFromCpu(parent, xml), ret, exit);
      }
    }
    // gpu 的 pci node 的 parent 指向 cpu node 或者 PCI Switch 的上行端口 pci node
    pciNode->parent = parent;
    // Keep PCI sub devices ordered by PCI Bus ID (Issue #820)
    // Coverity complains about dereferenced parent being NULL
    // but this can never happen.
    // coverity[var_deref_op]
    /*
    把 pciNode 插入 parent->subs[] 时按 busId 升序找插入点（第一个 busid > newBusId 的位置），右移腾位后插入。
    保证同一父下的 PCI 子设备按 busid 有序（融合/遍历稳定）。带 MAX_SUBS 守卫。
    */
    int subIndex = parent->nSubs;
    const char* newBusId;
    NCCLCHECKGOTO(xmlGetAttrStr(pciNode, "busid", &newBusId), ret, exit);
    for (int s = 0; s < parent->nSubs; s++) {
      const char* busId;
      NCCLCHECKGOTO(xmlGetAttr(parent->subs[s], "busid", &busId), ret, exit);
      if (busId != NULL && strcmp(newBusId, busId) < 0) {
        subIndex = s;
        break;
      }
    }
    if (parent->nSubs == MAX_SUBS) {
      WARN("Error : XML parser is limited to %d subnodes", MAX_SUBS);
      ret = ncclInternalError;
      goto exit;
    }
    for (int s = parent->nSubs; s > subIndex; s--) parent->subs[s] = parent->subs[s - 1];
    parent->subs[subIndex] = pciNode;
    parent->nSubs++;
  }
  if (strcmp(parent->name, "pci") == 0) {
    // 如果 gpu 的 pci node 的 parent 指向 PCI Switch 的上行端口 pci node，递归向上
    NCCLCHECKGOTO(ncclTopoGetXmlFromSys(parent, xml), ret, exit);
  } else if (strcmp(parent->name, "cpu") == 0) {
    // 如果 gpu 的 pci node 的 parent 指向 cpu node，终点尝试设置 cpu node 的属性
    NCCLCHECKGOTO(ncclTopoGetXmlFromCpu(parent, xml), ret, exit);
  }
exit:
  free(peers);
#if NCCL_OS_LINUX
  free(path);
#elif NCCL_OS_WINDOWS
  free(parentBusId);
#endif
  return ret;
}

/*
本函数在 gpu pci node 下，get-or-create <gpu>，然后填 GPU 才有的东西：设备序号、算力等级、NVLink 互连图、C2C 片间互连、
每条 NVLink 对端的设备类型。全程沿用"缺失才填"，尊重 NCCL_TOPO_FILE 预设。
*/
ncclResult_t ncclTopoGetXmlFromGpu(struct ncclXmlNode* pciNode, nvmlDevice_t nvmlDev, struct ncclXml* xml,
                                   struct ncclXmlNode** gpuNodeRet) {
  struct ncclXmlNode* gpuNode = NULL;
  NCCLCHECK(xmlGetSub(pciNode, "gpu", &gpuNode));  // get-or-create <gpu> 节点
  if (gpuNode == NULL) NCCLCHECK(xmlAddNode(xml, pciNode, "gpu", &gpuNode));  // 如果 gpu pci 节点下没有 gpu 节点，则创建

  int index = -1;

  // dev 属性与提前返回哨兵
  int dev = -1;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "dev", &index));  // 尝试获取 gpu 节点的 dev 属性
  if (index == -1) {
    // 如果没有 dev 属性，则调 ncclNvmlDeviceGetIndex 求设备序号
    NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&dev));
    NCCLCHECK(xmlSetAttrInt(gpuNode, "dev", dev));
  }
  NCCLCHECK(xmlGetAttrInt(gpuNode, "dev", &dev));  // 在读取 dev 属性值，即 GPU 的 NVML 索引，通常等同 CUDA 设备序号
  /*
  ncclNvmlDeviceGetIndex 成功必给 >=0、失败则已在 NCCLCHECK 处 error 上抛，
  所以能走到这个 -1 守卫的唯一可能是 NCCL_TOPO_FILE 里预置了 <gpu dev="-1">，
  语义是"这个 PCI 槽位不是本进程可用的 CUDA GPU"。回 NULL 后，
  ncclTopoFillGpu 的调用方就会跳过 keep/rank/gdr 的设置。
  */
  if (dev == -1) {
    *gpuNodeRet = NULL;
    return ncclSuccess;
  }

  // sm（compute capability 算力等级）
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "sm", &index));
  if (index == -1) {
    int cudaMajor, cudaMinor;
    if (nvmlDev == NULL) {
      cudaDeviceProp devProp;
      CUDACHECK(cudaGetDeviceProperties(&devProp, dev));
      cudaMajor = devProp.major;
      cudaMinor = devProp.minor;
    } else {
      NCCLCHECK(ncclNvmlDeviceGetCudaComputeCapability(nvmlDev, &cudaMajor, &cudaMinor));
    }
    NCCLCHECK(xmlSetAttrInt(gpuNode, "sm", cudaMajor * 10 + cudaMinor));  // Hopper 9.0→90、Ampere 8.0/8.6→80/86、Ada 8.9→89、Blackwell 10.0→100
  }
  int sm;
  NCCLCHECK(xmlGetAttrInt(gpuNode, "sm", &sm));

  struct ncclXmlNode* nvlNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "nvlink", &nvlNode));  // 尝试获取 gpu 节点下的 nvlink 节点
  if (nvlNode == NULL) {  // 只有当 <gpu> 下还没有任何 <nvlink> 时才探测（文件预设了就整段跳过）。
    /*
    按架构给一个迭代上界（Pascal P100 为 4、Volta V100 为 6、Ampere A100 为 12、Hopper+ 为 18，pre-Pascal 为 0）。
    它只是循环次数上限，真实不存在/不可用的链路会被下面的三重过滤剪掉。每条 link 的三重过滤（for l < maxNvLinks）：
      1. P2P 能力：查询失败或 !canP2P，这条链路不能做点对点，可能未布线或对端不支持 P2P。
      2. 链路已激活：sm≥90 用 field-value 接口查 NVML_FI_DEV_NVLINK_GET_STATE（scopeId=l），
    否则 fallthrough 到 ncclNvmlDeviceGetNvLinkState。isActive != NVML_FEATURE_ENABLED，continue。
      3. 能拿到对端：ncclNvmlDeviceGetNvLinkRemotePciInfo(l, &remoteProc) 失败 → continue。
    */
    // 根据 sm 架构版本确定 NVLink 的 links 数
    // NVML NVLink detection
    int maxNvLinks = (sm < 60) ? 0 : (sm < 70) ? 4 : (sm < 80) ? 6 : (sm < 90) ? 12 : 18;

    if (maxNvLinks > 0 && nvmlDev == NULL) {
      INFO(NCCL_GRAPH, "No NVML device handle. Skipping nvlink detection.");
      maxNvLinks = 0;
    }

    for (int l = 0; l < maxNvLinks; ++l) {
      // Check whether we can use this NVLink for P2P
      unsigned canP2P;
      // 查询某个 NVML 设备上的某条 NVLink 链路是否支持 P2P
      if ((ncclNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) ||
          !canP2P) {
        continue;
      }

      // 第 2 重过滤：确认链路已激活
      // Make sure the Nvlink is up. The previous call should have trained the link.
      nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;
#if CUDART_VERSION >= 11080
      if (sm >= 90) {
        nvmlFieldValue_t fv;
        fv.fieldId = NVML_FI_DEV_NVLINK_GET_STATE;
        fv.scopeId = l;
        // sm >= 90（Hopper 及以后）且 CUDART ≥ 11.8，使用 ncclNvmlDeviceGetFieldValues API 查 NVML_FI_DEV_NVLINK_GET_STATE
        // fv.value will contain NV_FEATURE_ENABLED or NV_FEATURE_DISABLED
        if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS))
          isActive = (nvmlEnableState_t)fv.value.uiVal;
      } else /* FALLTHRU to GetNvLinkState if before SM90 */
#endif
      {
        // sm < 90 或 CUDART < 11.8，使用 ncclNvmlDeviceGetNvLinkState 传统专用接口
        (void)ncclNvmlDeviceGetNvLinkState(nvmlDev, l, &isActive);
      }
      if (isActive != NVML_FEATURE_ENABLED) continue;

      // 第 3 重过滤：查出对端是谁
      // Try to figure out what's on the other side of the NVLink
      // = {} 值初始化，把整个结构清零
      nvmlPciInfo_t remoteProc = {};
      // 查询这条 NVLink 对端设备的 PCI 信息填入 remoteProc，失败则 continue，拿不到对端就无法建模，跳过。
      if (ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;

      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case.
      // NVML may return empty or non-printable busId for non-visible
      // remote devices (e.g. NVSwitch on Windows). Use sentinel instead.
      char* p = remoteProc.busId;
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      /*
      两种非法情况都用哨兵 "fffffff:ffff:ff"，是一个不可能真实存在的 PCI 地址（domain 有 7 个 f，远超正常的 4 位），
      用它作占位符既能通过后续的字符串处理，又不会与任何真实设备 busId 撞车。

      DGX-2 / A100 / H100 这类系统，每块 GPU 的所有 NVLink 端口都接到 NVSwitch 芯片上，GPU 之间不直连：
                ┌──────────┐
        GPU0 ═══│ NVSwitch │═══ GPU1
        GPU2 ═══│  (交叉开关) │═══ GPU3
                └──────────┘
      此时 GetNvLinkRemotePciInfo(GPU0, l) 返回的对端是那颗 NVSwitch 芯片，不是 GPU1/2/3。
      NVSwitch 本身是一个暴露在 PCI 总线上的桥设备，class 是 0x068000（PCI_NVSWITCH_CLASS，属 Bridge 大类 0x06）。
      */
      if (p[0] == '\0') {  // busId 是空串，对端设备不可见（如 Windows 上的 NVSwitch，NVML 拿不到其 PCI 地址）
        strncpy(lowerId, "fffffff:ffff:ff", NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
      } else {  // 循环中遇到非可打印字符（!isprint），说明 busId 是垃圾数据，同样替换成哨兵
        for (int c = 0; c < NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
          if (p[c] && !isprint((unsigned char)p[c])) {
            strncpy(lowerId, "fffffff:ffff:ff", NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
            break;
          }
          lowerId[c] = tolower(p[c]);  // sysfs 里的 PCI 路径一律小写，而 NVML 返回的 busId 可能是大写
          if (p[c] == 0) break;  // 拷到源串结尾的 \0 就停，保证 lowerId 正确终止
        }
      }

      /*
      按 target（对端 busId）find-or-create <nvlink>：同一对端有多条 NVLink 时聚合成一个节点、count 累加。下游据此算 GPU↔对端的总 NVLink 带宽。
      两块 GPU 之间或者 GPU 与 NVSwitch 之间往往有多条 NVLink 并行。与其为每条链路建一个节点，不如用 count 表达"到这个 target 有几条链路"，
      既紧凑又直接反映了聚合带宽。最终 XML 形如：
        <gpu dev="0" sm="90" ...>
          <nvlink target="0000:3b:00.0" count="18"/>   <!-- 到对端的 18 条链路 -->
          <nvlink target="fffffff:ffff:ff" count="4"/> <!-- 身份未知的链路 -->
        </gpu>
      */
      NCCLCHECK(xmlGetSubKv(gpuNode, "nvlink", &nvlNode, "target", lowerId));
      if (nvlNode == NULL) {
        // 首次遇到该对端，新建 <nvlink target="lowerId" count="1">
        NCCLCHECK(xmlAddNode(xml, gpuNode, "nvlink", &nvlNode));
        NCCLCHECK(xmlSetAttr(nvlNode, "target", lowerId));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", 1));
      } else {
        // 已存在（同一对端有多条 NVLink 连过来）：读出 count 加 1 再写回
        int count;
        NCCLCHECK(xmlGetAttrInt(nvlNode, "count", &count));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count + 1));
      }
    }
  }
#if CUDART_VERSION >= 11080
  struct ncclXmlNode* c2cNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "c2c", &c2cNode));  // 查找 <gpu> 节点下是否存在 <c2c> 节点
  /*
  编译期 CUDART_VERSION >= 11080（CUDA 11.8+）+ 运行期 sm >= 90（Hopper+），
  因为 NVLink-C2C 是 Grace-Hopper（GH200）超级芯片才有的 CPU（Grace）↔GPU（Hopper）片间缓存一致性互连。
  */
  if (c2cNode == NULL) {
    if (sm >= 90) {
      int c2cLinksCount = 0;
      nvmlFieldValue_t fv;
      fv.fieldId = NVML_FI_DEV_C2C_LINK_COUNT;
      // 用 field-value NVML_FI_DEV_C2C_LINK_COUNT 获取 C2C 链路总数
      if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 1, &fv) == ncclSuccess) && (fv.nvmlReturn == NVML_SUCCESS)) {
        c2cLinksCount = fv.value.uiVal;
        // 对每条 link 批量查 GET_STATUS 和 GET_MAX_BW 这两个字段，status==1（up）且两字段都成功，记 bw、count++
        int bw = 0;
        int count = 0;
        for (int l = 0; l < c2cLinksCount; l++) {
          nvmlFieldValue_t fvs[2];
          fvs[0].fieldId = NVML_FI_DEV_C2C_LINK_GET_STATUS;
          fvs[0].scopeId = l;
          fvs[1].fieldId = NVML_FI_DEV_C2C_LINK_GET_MAX_BW;
          fvs[1].scopeId = l;
          if ((ncclNvmlDeviceGetFieldValues(nvmlDev, 2, fvs) == ncclSuccess) && (fvs[0].nvmlReturn == NVML_SUCCESS) &&
              (fvs[0].value.uiVal == 1) && (fvs[1].nvmlReturn == NVML_SUCCESS)) {
            bw = fvs[1].value.uiVal;
            count++;
          }
        }
        if (count > 0) {
          /*
          在 <gpu> 节点下创建 <c2c bw count> 节点。
          与 NVLink 的按对端分组不同，C2C 把所有 up 链路汇总成一个节点。bw 是最后一条 up 链路的带宽（各链路通常相同）、count 是 up 链路总数。
          下游对应 PATH_C2C/PATH_P2C 语义与 GDR_C2C/PXN_C2C 开关。
          */
          NCCLCHECK(xmlAddNode(xml, gpuNode, "c2c", &c2cNode));
          NCCLCHECK(xmlSetAttrInt(c2cNode, "bw", bw));
          NCCLCHECK(xmlSetAttrInt(c2cNode, "count", count));
        }
      }
    }
  }
#endif
  /*
  补全每条 NVLink 对端的 tclass，tclass 的意义是为了区分 NVLink 对端到底是另一块 GPU（0x03）还是 NVSwitch（0x068000）。
  这决定下游建图时这条 NVLink 是点对点直连、还是接入 NVSwitch 全互连域。
  */
  // Fill target classes
  // 遍历 <gpu> 节点下所有子节点
  for (int s = 0; s < gpuNode->nSubs; s++) {
    struct ncclXmlNode* sub = gpuNode->subs[s];
    if (strcmp(sub->name, "nvlink") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(sub, "tclass", &index));
    if (index == -1) {  // 只对 nvlink 且尚无 tclass 的补对端设备类型
      const char* busId;
      NCCLCHECK(xmlGetAttr(sub, "target", &busId));
      // target 是哨兵 "fffffff:ffff:ff"（对端 VM 内不可见），直接假定 NVSwitch，tclass = PCI_NVSWITCH_CLASS（"0x068000"）
      if (strcmp(busId, "fffffff:ffff:ff") == 0) {
        // Remote NVLink device is not visible inside this VM. Assume NVSwitch.
        INFO(NCCL_GRAPH, "NVLink target %s not visible, assuming NVSwitch", busId);
        NCCLCHECK(xmlSetAttr(sub, "tclass", PCI_NVSWITCH_CLASS));
      } else {
        /*
        否则 ncclOsGetPciDeviceClassByBusId(target, …) 查对端真实 class，这是跨平台函数，Linux 侧 ncclOsGetPciPath + 读 sysfs class 文件。
        查到非空，填真实 class；查不到，INFO 后同样假定 NVSwitch。
        */
        char deviceClass[MAX_STR_LEN];
        deviceClass[0] = '\0';
        // 从 busId 对应的 sysfs 路径下的 class 文件中读取 tclass
        if (ncclOsGetPciDeviceClassByBusId(busId, deviceClass, sizeof(deviceClass)) == ncclSuccess &&
            deviceClass[0] != '\0') {
          NCCLCHECK(xmlSetAttr(sub, "tclass", deviceClass));  // 设置 <nvlink tclass> 属性
          TRACE(NCCL_GRAPH, "Read NVLink target class: tclass=%s for busId=%s", deviceClass, busId);
        } else {
          INFO(NCCL_GRAPH, "Could not get device class for NVLink target %s, assuming NVSwitch", busId);
          NCCLCHECK(xmlSetAttr(sub, "tclass", PCI_NVSWITCH_CLASS));
        }
      }
    }
  }
  *gpuNodeRet = gpuNode;
  return ncclSuccess;
}

// 把一张 GPU 的完整 PCIe + NVLink/C2C 子树填进 XML
ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode) {
  struct ncclXmlNode* node;
  /*
  查找或者创建属性 busid 等于 gpu busId 的 pci node，先按 busid 全树查，命中就复用（拓扑文件里可能已有、或同一 GPU 被重复填），
  没有才建。新建时 parent=NULL（悬空），挂父节点的活延迟到后面的 ncclTopoGetXmlFromSys（它负责向上找桥/cpu 再 attach）。
  这是"find-or-create + 延迟挂父"模式。
  */
  NCCLCHECK(ncclTopoGetPciNode(xml, busId, &node));
  /*
    0x03 是 PCI class code 高位 = Display controller（GPU）。
    xmlSetAttrIfUnset 的语义是没这个属性才写，即尊重拓扑文件里可能已显式写的 class，不覆盖。
    下一步 ncclTopoGetXmlFromSys 用 strncmp(deviceClass,"0x03",4)==0 判断"这是不是 GPU"来决定要不要去 NVML 查句柄。
  */
  NCCLCHECK(xmlSetAttrIfUnset(node, "class", "0x03"));
  // 对于带 busId 的 gpu pci 节点，设置该 gpu pci node 的一系列属性，并向上关联到 cpu，并设置 cpu node 的一系列属性
  NCCLCHECK(ncclTopoGetXmlFromSys(node, xml));
  nvmlDevice_t nvmlDev;
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  /*
  前四步已经把 <pci> 节点建好、属性填满、锚进 PCIe 树，并拿到了这块卡的 NVML 句柄 nvmlDev。本函数接手这个 pci 锚点，
  在其下 get-or-create <gpu>，然后填 GPU 才有的东西：设备序号、算力等级、NVLink 互连图、C2C 片间互连、每条 NVLink 对端的设备类型。
  全程沿用"缺失才填"，尊重 NCCL_TOPO_FILE 预设。
  */
  NCCLCHECK(ncclTopoGetXmlFromGpu(node, nvmlDev, xml, gpuNode));
  return ncclSuccess;
}

/*
在 Linux sysfs 里，每个设备目录下都有一个 subsystem 符号链接，指向该设备所属的子系统目录。
这个链接解析后的路径末段（basename）就是子系统名字：
  设备	                  subsystem 链接解析后	basename
  PCI 设备（真网卡、GPU）	  /sys/bus/pci	      pci
  USB 网卡	              /sys/bus/usb	      usb
  虚拟网络设备	           /sys/class/net	      net

给定设备 sysfs 路径，解析它的 subsystem 链接，取出 basename 写进 subSys。
用于区分真 PCI 网卡（要精确挂进 PCIe 拓扑树）和虚拟/USB 网卡（没有 PCIe 位置，只能挂到第一个 CPU 下）
*/
// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
ncclResult_t ncclTopoGetSubsystem(const char* sysPath, char* subSys) {
#if NCCL_OS_LINUX
  char subSysPath[PATH_MAX];
  // 在传入的设备路径后拼上 /subsystem，得到那个符号链接的完整路径，比如 /sys/devices/pci0000:00/.../0000:3b:00.0/subsystem。
  snprintf(subSysPath, sizeof(subSysPath), "%s/subsystem", sysPath);
  /*
  realpath 的跨平台封装（Linux 直接转调 realpath）。第二个参数传 NULL 是 POSIX.1-2008 的用法，
  让 realpath 自己 malloc 一块缓冲区返回规范化的绝对路径，因此调用方负责 free。
  这一步把 subsystem 链接一路解析成真实目标，如 /sys/bus/pci。
  */
  char* path = ncclOsRealpath(subSysPath, NULL);
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    // 取 basename（从尾向前找最后一个 /）。以 path = "/sys/bus/pci" 为例，正确取出子系统名"pci"
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--);
    strcpy(subSys, path + offset + 1);
    free(path);
  }
#elif NCCL_OS_WINDOWS
  (void)sysPath;
  subSys[0] = '\0';
#endif
  return ncclSuccess;
}

/*
把一个由 netName 标识、pciPath 定位的网络设备，幂等地物化成 XML 拓扑树里的 <net> 节点，
并锚定到正确的父位置（物理网卡挂 <pci>、虚拟网卡挂 <cpu>、vNic 挂 forceParent）。
它只负责"建骨架 + 定位"，具体属性（dev/speed/guid/gdr…）由调用方 ncclTopoPopulateNics 在返回后补。
*/
ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* tagName, const char* pciPath, const char* netName,
                             struct ncclXmlNode** netNode, struct ncclXmlNode* forceParent) {
  /*
  全树按 name==netName 找同名 <net>，找到就直接返回（netNode 已指向它）。
  这是本函数最重要的性质：gin/rma/collnet/net 四类插件会反复认领同一块物理 NIC，
  幂等保证它们共享同一个 <net> 节点（只往上叠加 net/gin/rma/coll 标志），而不是各建一份。
  */
  NCCLCHECK(xmlFindTagKv(xml, tagName, netNode, "name", netName));
  if (*netNode != NULL) return ncclSuccess;

  struct ncclXmlNode* parent = NULL;
  if (forceParent) {
    /*
    forceParent 非空（虚拟 NIC），直接 parent = forceParent。vNic 是多个物理卡的聚合、没有单一 pciPath，
    其父由 ncclTopoGetVNicParent 事先算好（可能是造的假 pci 节点）传进来。
    */
    parent = forceParent;
  } else {
    const char* pciSysPath = pciPath;
    if (pciSysPath) {
      char subSystem[PATH_MAX];
      // 获取 pci 路径表示的子系统名字，例如真网卡"pci"，还有"usb"、"net"
      NCCLCHECK(ncclTopoGetSubsystem(pciSysPath, subSystem));
      // This is not a PCI device (virtual, usb, ...).
      if (strcmp(subSystem, "pci") != 0 && !forceParent) {  // 若不是真网卡（虚拟/USB 网卡）
        INFO(NCCL_NET | NCCL_GRAPH,
             "Topology detection: network path (name = %s) %s is not a PCI device (%s). Attaching to first CPU",
             netName, pciSysPath, subSystem);
        pciSysPath = NULL;
      }
    }

    if (pciSysPath) {  // 如果是真网卡
      // 从 pciPath 末段截出 busId（sysfs 惯例 .../0000:3b:00.0）
      int offset;
      for (offset = strlen(pciSysPath) - 1; pciSysPath[offset] != '/'; offset--);
      char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      strcpy(busId, pciSysPath + offset + 1);
      // find-or-create，新建的 pci 节点 parent=NULL 悬空，延迟挂父
      NCCLCHECK(ncclTopoGetPciNode(xml, busId, &parent));
      NCCLCHECK(xmlSetAttrIfUnset(parent, "class", "0x02"));  // 只在未设时打网络设备类标记
      NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));  // 从这个 pci 锚点向上补全整条 PCIe 路径并冒泡挂到 <cpu>
    } else {
      /*
      pciSysPath 为 NULL（虚拟网卡），xmlFindTag(xml,"cpu",&parent)，挂到第一个 <cpu>。
      虚拟 NIC/USB 网卡没有 PCI 位置，只能挂 CPU。
      */
      // Virtual NIC, no PCI device, attach to first CPU
      NCCLCHECK(xmlFindTag(xml, "cpu", &parent));
    }
  }

  if (parent == NULL) {  // 兜底建 CPU，若连 cpu 都还没建
    // No CPU node exists yet, create a default one
    struct ncclXmlNode* topNode;
    NCCLCHECK(xmlFindTag(xml, "system", &topNode));
    if (topNode) {
      NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
      NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
      NCCLCHECK(xmlSetAttr(parent, "numaid", "0"));
    }
  }
  if (parent == NULL) {
    WARN("ncclTopoFillNet: parent is NULL even after fallback for %s", netName);
    return ncclInternalError;
  }

  struct ncclXmlNode* nicNode = NULL;
  NCCLCHECK(xmlGetSub(parent, "nic", &nicNode)); // 找第一个 nic 子节点
  if (nicNode == NULL) {  // net pci 节点下没有 nic 节点就创建一个
    NCCLCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  // 在 <nic> 下创建 <net> 节点，同一个 parent（pci/cpu）下的多个 <net> 共享同一个 <nic> 容器
  // We know that this net does not exist yet (we searched for it at the
  // beginning of this function), so we can add it.
  NCCLCHECK(xmlAddNode(xml, nicNode, tagName, netNode));
  NCCLCHECK(xmlSetAttr(*netNode, "name", netName));
  return ncclSuccess;
}

/*
这是一个后序遍历（先递归处理所有子节点，再决定本节点去留）。每个节点通过出参 *keep 向父报告"我这棵子树里有没有需要保留的叶子"，
父节点据此决定是否剪掉自己。核心不变量：一个节点被保留，当且仅当它自己 keep=="1"，或它的子树里至少有一个 keep=="1" 的叶子。
*/
ncclResult_t ncclTopoTrimXmlRec(struct ncclXmlNode* node, int* keep) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, "keep", &str));
  if (str && strcmp(str, "1") == 0) {
    NCCLCHECK(xmlUnsetAttr(node, "keep"));  // 如果当前节点 keep=1，去掉该属性，并上传 keep=1
    *keep = 1;
  } else {
    // 由于递归剪枝，nSubs 可能减少，subs 可能被删除，所以拷贝一份备份，确保遍历不出问题
    // Copy nSubs and subs as they could change as we trim recursively.
    struct ncclXmlNode** subs = NULL;
    NCCLCHECK(ncclCalloc(&subs, MAX_SUBS));
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, node->nSubs * sizeof(struct ncclXmlNode*));
    *keep = 0;
    ncclResult_t subsRes = ncclSuccess;
    for (int s = 0; s < nSubs; s++) {  // 遍历所有子节点
      int k = 0;
      subsRes = ncclTopoTrimXmlRec(subs[s], &k);  // 递归处理子节点
      if (subsRes != ncclSuccess) {
        break;
      }
      *keep += k;  // 累加子树中有多少 keep
    }
    free(subs);
    NCCLCHECK(subsRes);
    // 子树和当前节点累加的 keep=0，并且节点类型为 pci/cpu/nic/net 其中之一，删除当前节点
    // Remove node if it has no children and no keep attribute
    if (*keep == 0 && // Trim PCI switches, CPUs with no used GPU/NIC under them, or pruned NICs
        (strcmp(node->name, "pci") == 0 || strcmp(node->name, "cpu") == 0 || strcmp(node->name, "nic") == 0 ||
         strcmp(node->name, "net") == 0)) {
#ifdef ENABLE_TRACE
      const char* name;
      const char* busid;
      NCCLCHECK(xmlGetAttr(node, "name", &name));
      NCCLCHECK(xmlGetAttr(node, "busid", &busid));
      TRACE(NCCL_GRAPH, "Removing node %s %s %s\n", node->name, name, busid);
#endif
      NCCLCHECK(xmlRemoveNode(node));  // 删除当前节点
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimXml(struct ncclXml* xml) {
  int keep = 0;
  NCCLCHECK(ncclTopoTrimXmlRec(xml->nodes, &keep));
  return ncclSuccess;
}

#if NCCL_OS_LINUX

/**************************************************/
/* Parser rules for the user-defined graph search */
/**************************************************/

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"net", ncclTopoXmlGraphLoadNet}, {"gpu", ncclTopoXmlGraphLoadGpu}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = {{"channel", ncclTopoXmlGraphLoadChannel}};
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_GRAPH_XML_VERSION) {
    WARN("XML Graph has wrong version %d, %d needed", version, NCCL_GRAPH_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading graphs for topology %s", name);
  else INFO(NCCL_GRAPH, "Loading graphs");

  struct xmlHandler handlers[] = {{"graph", ncclTopoXmlGraphLoadGraph}};
  NCCLCHECK(xmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlGraphFile, "r");
  if (file == NULL) {
    WARN("Could not open XML graph file %s : %s", xmlGraphFile, strerror(errno));
    return ncclSystemError;
  }
  struct xmlHandler handlers[] = {{"graphs", ncclTopoXmlGraphLoadGraphs}};
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

#elif NCCL_OS_WINDOWS

/* Stub implementations for Windows */

ncclResult_t xmlGetChar(FILE* file, char* c) {
  (void)file;
  (void)c;
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  (void)file;
  (void)value;
  (void)last;
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  (void)file;
  (void)name;
  (void)value;
  (void)last;
  return ncclSuccess;
}

ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  (void)file;
  (void)start;
  (void)next;
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  (void)file;
  (void)node;
  return ncclSuccess;
}

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[],
                        int nHandlers) {
  (void)file;
  (void)xml;
  (void)head;
  (void)handlers;
  (void)nHandlers;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPciLink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadC2c(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  (void)file;
  (void)xml;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  (void)file;
  (void)xmlGraph;
  (void)head;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  (void)xmlGraphFile;
  (void)xml;
  return ncclSuccess;
}

#endif
