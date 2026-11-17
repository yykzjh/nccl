/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "utils.h"
#include "core.h"
#include "os.h"

#include "nvmlwrap.h"

#include <stdlib.h>
#include <mutex>

// Get current Compute Capability
int ncclCudaCompCap() {
  int cudaDev;
  if (!CUDASUCCESS(cudaGetDevice(&cudaDev))) return 0;
  int ccMajor, ccMinor;
  if (!CUDASUCCESS(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev))) return 0;
  if (!CUDASUCCESS(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, cudaDev))) return 0;
  return ccMajor * 10 + ccMinor;
}

// 显式地按位切分重建字符串
ncclResult_t int64ToBusId(int64_t id, char* busId) {
  /*
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx",
        (id) >> 20,             // domain：bit 20 以上
        (id & 0xff000) >> 12,   // bus：bit 12–19（2 位 hex）
        (id & 0xff0) >> 4,      // device：bit 4–11（2 位 hex）
        (id & 0xf));            // function：bit 0–3（1 位 hex）
  */
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (unsigned long)((id) >> 20), (unsigned long)((id & 0xff000) >> 12),
          (unsigned long)((id & 0xff0) >> 4), (unsigned long)(id & 0xf));
  return ncclSuccess;
}

/*
把 : 和 . 全部删掉，把剩下的十六进制数字拼成一整个大 hex 串，一次性strtol成 int64。
它并不逐字段解析 domain/bus/device/function，而是把整串当一个数。
以 "0000:00:02.0" 为例，删分隔符，"000000020"，strtol(…,16) = 0x20。
*/
ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
  char hexStr[17];  // Longest possible int64 hex string + null terminator.
  int hexOffset = 0;
  for (int i = 0; hexOffset < sizeof(hexStr) - 1; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else {
      break;
    }
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  return ncclSuccess;
}

// Get an int64 from a PCI path. For example, sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will return 0x000002000.
ncclResult_t pciPathToInt64(char* path, int64_t* id) {
  char* str = path + strlen(path) - 1;
  // Remove trailing "/"
  if (*str == '/') str--;
  // Find next /
  while (*str != '/') str--;
  str++;
  NCCLCHECK(busIdToInt64(str, id));
  return ncclSuccess;
}

// 把一个逻辑 CUDA 设备号（cudaDev，就是 0、1、2… 这种序号）转换成该 GPU 的 PCI 总线地址的紧凑 int64 编码，写进 *busId
// Convert a logical cudaDev index to the NVML device minor number
ncclResult_t getBusId(int cudaDev, int64_t* busId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  /*
  预留尺寸的缓冲区，用初始化字面量顺手把缓冲区大小定下来。
  注意它写的是 "00000000:..."，8 个 0 的 domain 段，而典型格式只有 4 位 domain。
  */
  char busIdStr[] = "00000000:00:00.0";
  // 把 cudaDev 对应 GPU 的 BDF 以字符串形式（如 "0000:17:00.0"）填进 busIdStr。第二个参数是缓冲区容量，防止越界。
  CUDACHECK(cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), cudaDev));
  /*
  把字符串压缩成 int64，好处：
    1. 比较廉价：拓扑构建、设备去重时要大量比对"是不是同一块设备"，整数 ==` 远快于strcmp。
    2. 存储紧凑：comm->busId、peerInfo.busId 都是 int64 字段，一个 8 字节整数比字符串省空间、好在 rank 间用 bootstrap 传递。
    3. 可做键/可运算：pciPathToInt64 从 /sys 的 PCI 路径里抠出同样的整数，方便在拓扑树里做匹配、算父子关系。
  */
  NCCLCHECK(busIdToInt64(busIdStr, busId));
  return ncclSuccess;
}

ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return ncclSystemError;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
  hostname[i] = '\0';
  return ncclSuccess;
}

static uint64_t hostHashValue = 0;
/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the NCCL_HOSTID env var.
 */
#if defined(NCCL_OS_LINUX)

#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"

#elif defined(NCCL_OS_WINDOWS)

/* Get Windows MachineGuid - similar to boot_id on Linux */
static bool getWindowsMachineGuid(char* guid, size_t len) {
  HKEY hKey;
  LONG result = RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography", 0, KEY_READ, &hKey);
  if (result != ERROR_SUCCESS) {
    return false;
  }
  DWORD dataSize = (DWORD)len;
  DWORD dataType;
  result = RegQueryValueExA(hKey, "MachineGuid", NULL, &dataType, (LPBYTE)guid, &dataSize);
  RegCloseKey(hKey);
  if (result != ERROR_SUCCESS || dataType != REG_SZ) {
    return false;
  }
  return true;
}
#endif

static void getHostHashOnce() {
  char hostHash[1024];
  const char* hostId;

  // hostname 打底
  // Fall back is the full hostname if something fails
  (void)getHostName(hostHash, sizeof(hostHash), '\0');
  int offset = strlen(hostHash);

  // 优先 NCCL_HOSTID 覆盖 hostname
  if ((hostId = ncclGetEnv("NCCL_HOSTID")) != NULL) {
    INFO(NCCL_ENV, "NCCL_HOSTID set by environment to %s", hostId);
    strncpy(hostHash, hostId, sizeof(hostHash) - 1);
    hostHash[sizeof(hostHash) - 1] = '\0';
  } else {
  // 追加 boot_id(Linux)/ MachineGuid(Windows)
#if defined(NCCL_OS_LINUX)
    FILE* file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
      char* p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash + offset, p, sizeof(hostHash) - offset - 1);
        free(p);
      }
      fclose(file);
    }
#elif defined(NCCL_OS_WINDOWS)
    char machineGuid[256];
    if (getWindowsMachineGuid(machineGuid, sizeof(machineGuid))) {
      strncpy(hostHash + offset, machineGuid, sizeof(hostHash) - offset - 1);
    }
#endif
  }

  // Make sure the string is terminated
  hostHash[sizeof(hostHash) - 1] = '\0';

  TRACE(NCCL_INIT, "unique hostname '%s'", hostHash);

  // 哈希 + 缓存
  hostHashValue = getHash(hostHash, strlen(hostHash));
}

// 全局就计算一次 host hash
uint64_t getHostHash(void) {
  static std::once_flag once;
  std::call_once(once, getHostHashOnce);
  return hostHashValue;
}

uint64_t hashCombine(uint64_t baseHash, uint64_t value) {
  uint64_t hacc[2] = {1, 1};
  eatHash(hacc, &baseHash);
  eatHash(hacc, &value);
  return digestHash(hacc);
}

/*
拼出字符串 <pid> + readlink("/proc/self/ns/pid")，后者形如 pid:[4026531836]，即 PID namespace 的 inode
带 namespace 是因为裸 PID 跨容器不唯一(两个容器里都可能有 PID 7)，但 (PID, PID-ns-inode) 在同一物理节点上唯一。
Windows 无 namespace，PID 本就系统级唯一，故只用 PID。
*/
/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Linux: hash of $$ $(readlink /proc/self/ns/pid) (pid + PID namespace)
 * Windows: hash of PID only (no namespaces; PID is unique system-wide)
 */
uint64_t getPidHash(void) {
  char pname[1024];
  // Start off with our pid ($$)
  sprintf(pname, "%ld", (long)ncclOsGetPid());
  int plen = strlen(pname);
#if defined(NCCL_OS_LINUX)
  int len = readlink("/proc/self/ns/pid", pname + plen, sizeof(pname) - 1 - plen);
  if (len < 0) len = 0;
  plen += len;
#endif
  pname[plen] = '\0';
  TRACE(NCCL_INIT, "unique PID '%s'", pname);
  // getHash 是 NCCL 的 FNV 系字符串哈希,输出 uint64_t
  return getHash(pname, strlen(pname));
}

// 把形如 "eth0:1,eth1:2" 或 "mlx5_0:1:0:0" 的字符串解析成 struct netIf 数组
int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  char* str = strdup(string);
  if (!str) return 0;

  int ifNum = 0;
  char* savePtr = NULL;
  /*
  strtok_r 是线程安全的字符串分割函数，按逗号分割外层条目
    第一次调用：strtok_r(str, ",", &savePtr)
    后续调用：strtok_r(NULL, ",", &savePtr)
    返回每个逗号之间的子串
  */
  char* entry = strtok_r(str, ",", &savePtr);
  while (entry != NULL && ifNum < maxList) {
    char* c = entry;
    // cclOsStrSep 是跨平台封装，Linux 下就是 strsep，作用是按 :` 分割字符串，并移动指针到剩余部分
    char* tok = ncclOsStrSep(&c, ":");
    if (tok && tok[0] != '\0') {
      snprintf(ifList[ifNum].prefix, sizeof(ifList[ifNum].prefix), "%s", tok);
      // port, rail, and plane will default to -1 if absent or empty
      tok = ncclOsStrSep(&c, ":");
      ifList[ifNum].port = (tok && tok[0] != '\0') ? atoi(tok) : -1;
      tok = ncclOsStrSep(&c, ":");
      ifList[ifNum].rail = (tok && tok[0] != '\0') ? atoi(tok) : -1;
      tok = ncclOsStrSep(&c, ":");
      ifList[ifNum].plane = (tok && tok[0] != '\0') ? atoi(tok) : -1;
      ifNum++;
    }
    entry = strtok_r(NULL, ",", &savePtr);
  }
  free(str);
  return ifNum;
}

static bool matchIf(const char* string, const char* ref, bool matchExact) {
  // 如果精确匹配，匹配长度就是接口名+1（末尾的'\0'）；否则为前缀长度
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}

bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact, int* ifId) {
  // Make an exception for the case where no user list is defined
  if (ifId) *ifId = -1;
  if (listSize == 0) return true;

  for (int i = 0; i < listSize; i++) {
    // 接口名/接口名前缀和端口是否匹配
    if (matchIf(string, ifList[i].prefix, matchExact) && matchPort(port, ifList[i].port)) {
      if (ifId) *ifId = i;
      return true;
    }
  }
  return false;
}

thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance;

/*
当前 hunk 的剩余空间装不下这次请求，但注意这不等于对象大(可能 hunk 只剩 10 字节，对象只有 100 字节)，
也不等于对象小(可能是 200KB 的巨型请求)。spilled 的全部工作，就是在这两种情况之间做正确的路由。
*/
void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align) {
  // `me->hunks` points to the top of the stack non-empty hunks. Hunks above
  // this (reachable via `->above`) are empty.
  struct Hunk* top = me->topFrame.hunk;
  size_t mallocSize = 0;

  /*
  剩余空间还有 8KB 以上，却装不下这个对象 → 对象一定不小(粗略 > 8KB)
  为什么不换块？ 如果此时切换到别的 hunk,当前块剩余的 ≥8KB 就被浪费了，
  让一个大对象走外块(Unhunk + 独立 malloc),只消耗当前块里 16 字节的头，把整段剩余空间留给之后的小对象更划算。
  */
  // If we have lots of space left in hunk but that wasn't enough then we'll
  // allocate the object unhunked.
  if (me->topFrame.end - me->topFrame.bumper >= 8 << 10) goto unhunked;

  /*
  
  */
  // If we have another hunk (which must be empty) waiting above this one and
  // the object fits then use that.
  if (top && top->above) {
    // 取上方的一个空块
    struct Hunk* top1 = top->above;
    // 按“块基址 + 16 字节头 + 对齐”算出对象地址
    uintptr_t uobj = (reinterpret_cast<uintptr_t>(top1) + sizeof(struct Hunk) + align - 1) & -uintptr_t(align);
    // 装得下就整体搬迁
    if (uobj + size <= reinterpret_cast<uintptr_t>(top1) + top1->size) {
      me->topFrame.hunk = top1;
      me->topFrame.bumper = uobj + size;
      me->topFrame.end = reinterpret_cast<uintptr_t>(top1) + top1->size;
      return reinterpret_cast<void*>(uobj);
    }
  }

  // 再往下走两种情况：没有空块了（top->above == nullptr） or 下一块装不下对象

  { // If the next hunk we're going to allocate wouldn't be big enough but the
    // Unhunk proxy fits in the current hunk then go allocate as unhunked.
    // 预演新块尺寸：nextSize = 当前块大小 + 64KB(几何增长；对全新栈，top 是 size=0 的 stub,所以第一块恰为 64KB)
    size_t nextSize = (top ? top->size : 0) + (64 << 10);
    constexpr size_t maxAlign = 64;
    // 新块去掉头、再留出最坏 64 字节对齐垫料后，装不下对象
    if (nextSize < sizeof(struct Hunk) + maxAlign + size) {
      // 当前块里塞得下一个 16 字节的 Unhunk 代理头(alignof(Unhunk) = 8、sizeof(Unhunk) = 16,两个指针)，直接走外块
      uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk) - 1) & -uintptr_t(alignof(Unhunk));
      if (uproxy + sizeof(struct Unhunk) <= me->topFrame.end) goto unhunked;
    }

    /*
    这里分如下情况：
      1. 没有空块，预分配装得下对象；
      2. 没有空块，预分配装不下对象，当前块放不下 Unhunk 头；
      3. 有空块，装不下对象，当前块放不下 Unhunk 头；
    */

    // At this point we must need another hunk, either to fit the object
    // itself or its Unhunk proxy.
    mallocSize = nextSize;
    INFO_LOC(NCCL_ALLOC_HOST, "memory stack hunk malloc(%llu)", (unsigned long long)mallocSize);
    struct Hunk* top1 = (struct Hunk*)malloc(mallocSize);
    if (top1 == nullptr) goto malloc_exhausted;
    top1->size = nextSize;
    top1->above = nullptr;
    if (top) top->above = top1;
    top = top1;
    me->topFrame.hunk = top;
    me->topFrame.end = reinterpret_cast<uintptr_t>(top) + nextSize;
    me->topFrame.bumper = reinterpret_cast<uintptr_t>(top) + sizeof(struct Hunk);
  }

  // 新空块装得下对象 上述情况 1
  { // Try to fit object in the new top hunk.
    uintptr_t uobj = (me->topFrame.bumper + align - 1) & -uintptr_t(align);
    if (uobj + size <= me->topFrame.end) {
      me->topFrame.bumper = uobj + size;
      return reinterpret_cast<void*>(uobj);
    }
  }

  // 下面都用新空块装 Unhunk 头，对象存外块，上述情况 2 和 3
unhunked:
  { // We need to allocate the object out-of-band and put an Unhunk proxy in-band
    // to keep track of it.
    // 先存下 Unhunk 头
    uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk) - 1) & -uintptr_t(alignof(Unhunk));
    Unhunk* proxy = reinterpret_cast<Unhunk*>(uproxy);
    me->topFrame.bumper = uproxy + sizeof(Unhunk);
    // 将新 Unhunk 接在 me->topFrame.unhunks 链头
    proxy->next = me->topFrame.unhunks;
    me->topFrame.unhunks = proxy;
    // 分配对象内存
    mallocSize = size;
    proxy->obj = malloc(mallocSize);
    INFO_LOC(NCCL_ALLOC_HOST, "memory stack non-hunk malloc(%llu)", (unsigned long long)mallocSize);
    if (proxy->obj == nullptr) goto malloc_exhausted;
    return proxy->obj;
  }

malloc_exhausted:
  WARN("Unrecoverable error detected: malloc(size=%llu) returned null.", (unsigned long long)mallocSize);
  abort();
}

void ncclMemoryStackDestruct(struct ncclMemoryStack* me) {
  // Free unhunks first because both the frames and unhunk proxies lie within the hunks.
  struct ncclMemoryStack::Frame* f = &me->topFrame;
  while (f != nullptr) {
    struct ncclMemoryStack::Unhunk* u = f->unhunks;
    while (u != nullptr) {
      free(u->obj);
      u = u->next;
    }
    f = f->below;
  }
  // Free hunks
  struct ncclMemoryStack::Hunk* h = me->stub.above;
  while (h != nullptr) {
    struct ncclMemoryStack::Hunk* h1 = h->above;
    free(h);
    h = h1;
  }
}

/* return concatenated string representing each set bit */
ncclResult_t ncclBitsToString(uint32_t bits, uint32_t mask, const char* (*toStr)(int), char* buf, size_t bufLen,
                              const char* wildcard) {
  if (!buf || !bufLen) return ncclInvalidArgument;

  bits &= mask;

  // print wildcard value if all bits set
  if (wildcard && bits == mask) {
    snprintf(buf, bufLen, "%s", wildcard);
    return ncclSuccess;
  }

  // Add each set bit to string
  int pos = 0;
  for (int i = 0; bits; i++, bits >>= 1) {
    if (bits & 1) {
      if (pos > 0) pos += snprintf(buf + pos, bufLen - pos, "|");
      pos += snprintf(buf + pos, bufLen - pos, "%s", toStr(i));
    }
  }

  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////
// Hash function for pointer types (shared by address map implementations)
// Uses shadowpool's algorithm
uint64_t ncclHashPointer(int hbits, void* key) {
  uintptr_t h = reinterpret_cast<uintptr_t>(key);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13;
  return (uint64_t)h >> (64 - hbits);
}

////////////////////////////////////////////////////////////////////////////////
// Intrusive address map implementation (untyped core functions)

// Helper: Read key from object at given offset
// Key must be convertible to uintptr_t, so we read keySize bytes and zero-extend
static inline uintptr_t readKey(void* object, int keySize, int keyFieldOffset) {
  void* keyPtr = (char*)object + keyFieldOffset;
  uintptr_t result = 0;
  memcpy(&result, keyPtr, keySize);
  return result;
}

// Helper: Read next pointer from object at given offset
// Uses memcpy to avoid strict aliasing violations when actual type is T*, not void*
static inline void* readNextPtr(void* object, int nextFieldOffset) {
  void* nextPtr = (char*)object + nextFieldOffset;
  void* result = nullptr;
  memcpy(&result, nextPtr, sizeof(void*));
  return result;
}

// Helper: Write next pointer to object at given offset
// Uses memcpy to avoid strict aliasing violations when actual type is T*, not void*
static inline void writeNextPtr(void* object, int nextFieldOffset, void* value) {
  void* nextPtr = (char*)object + nextFieldOffset;
  memcpy(nextPtr, &value, sizeof(void*));
}

ncclResult_t ncclIntruAddressMapInsert_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                               int nextFieldOffset, uintptr_t key, void* object) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (object == nullptr) {
    WARN("Object pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  // Lazy initialization - create table on first insert
  if (map->hbits == 0) {
    const int initHbits = 4;
    const size_t tableEntries = (size_t)1 << initHbits;
    map->hbits = initHbits;
    map->table = (void**)calloc(tableEntries, sizeof(void*));
    if (map->table == nullptr) {
      map->hbits = 0; // Reset on failure
      WARN("Intrusive address map initialization failed: calloc(%zu entries) returned null", tableEntries);
      return ncclSystemError;
    }
  }

  int hbits = map->hbits;

  // Check for address map size increase before inserting. Maintain 2:1 object:bucket ratio.
  if (map->count + 1 > 2 << hbits) {
    int oldHbits = hbits;
    int oldSize = 1 << oldHbits;
    int newHbits = hbits + 1;
    int newSize = 1 << newHbits;

    // Allocate a new table (don't use realloc to avoid data corruption during rehashing)
    void** newTable = (void**)malloc(newSize * sizeof(void*));
    if (newTable == nullptr) {
      WARN("Intrusive address map resize failed: malloc(%d entries) returned null", newSize);
      return ncclSystemError;
    }

    // Initialize all new buckets to nullptr
    for (int i = 0; i < newSize; i++) {
      newTable[i] = nullptr;
    }

    // Rehash all existing entries from old table to new table
    for (int i = 0; i < oldSize; i++) {
      void* obj = map->table[i];

      while (obj) {
        void* next = readNextPtr(obj, nextFieldOffset);
        uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
        uint64_t b = ncclHashPointer(newHbits, (void*)objKey);
        writeNextPtr(obj, nextFieldOffset, newTable[b]);
        newTable[b] = obj;
        obj = next;
      }
    }

    // Free old table and update to new table
    free(map->table);
    map->table = newTable;
    map->hbits = newHbits;
  }

  // Insert object into appropriate bucket
  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* currentNext = readNextPtr(object, nextFieldOffset);

  // Check if next pointer is already non-NULL (object might already be in a list)
  if (currentNext != nullptr) {
    INFO(NCCL_INIT,
         "Intrusive map: inserting object %p with non-NULL next pointer %p (key=0x%lx). "
         "Object may already be in another list or this is intentional reuse.",
         object, currentNext, (unsigned long)key);
  }

  writeNextPtr(object, nextFieldOffset, map->table[b]);
  map->table[b] = object;
  map->count += 1;

  return ncclSuccess;
}

ncclResult_t ncclIntruAddressMapFind_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                             int nextFieldOffset, uintptr_t key, void** object) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (object == nullptr) {
    WARN("Output object pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  *object = nullptr;

  // Empty map is not an error - just means key not found
  if (map->hbits == 0) {
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* obj = map->table[b];

  while (obj) {
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    if (objKey == key) {
      *object = obj;
      return ncclSuccess;
    }
    obj = readNextPtr(obj, nextFieldOffset);
  }

  // Key not found is not an error - *object is already nullptr
  return ncclSuccess;
}

ncclResult_t ncclIntruAddressMapRemove_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                               int nextFieldOffset, uintptr_t key) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  // Removing from empty map is not an error - it's idempotent
  if (map->hbits == 0) {
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* prev = nullptr;
  void* obj = map->table[b];

  while (obj) {
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    if (objKey == key) {
      void* next = readNextPtr(obj, nextFieldOffset);

      // Update the previous pointer to skip the current object
      if (prev == nullptr) {
        // Removing from head of bucket
        map->table[b] = next;
      } else {
        // Removing from middle/end of list
        writeNextPtr(prev, nextFieldOffset, next);
      }

      map->count -= 1;

      // If this was the last entry, clean up the table (same pattern as non-intrusive map)
      if (map->count == 0) {
        free(map->table);
        map->hbits = 0;
        map->count = 0;
        map->table = nullptr;
      }

      return ncclSuccess;
    }
    prev = obj;
    obj = readNextPtr(obj, nextFieldOffset);
  }

  // Key not found is not an error - remove is idempotent
  return ncclSuccess;
}
