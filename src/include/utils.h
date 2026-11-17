/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_UTILS_H_
#define NCCL_UTILS_H_

#include "nccl.h"
#include "alloc.h"
#include "bitops.h"
#include "checks.h"
#include "compiler.h"
#include <stdint.h>
#include <time.h>
#include <algorithm>
#include <new>
#include <type_traits>
#include <mutex>
#include <condition_variable>
#include <random>
#include <chrono>

// On Windows, strtok_s is equivalent to POSIX strtok_r with same signature
#ifdef NCCL_OS_WINDOWS
#define strtok_r strtok_s
#endif

int ncclCudaCompCap();

// PCI Bus ID <-> int64 conversion functions
ncclResult_t int64ToBusId(int64_t id, char* busId);
ncclResult_t busIdToInt64(const char* busId, int64_t* id);
ncclResult_t pciPathToInt64(char* path, int64_t* id);

ncclResult_t getBusId(int cudaDev, int64_t* busId);

ncclResult_t getHostName(char* hostname, int maxlen, const char delim);
uint64_t getHostHash();
uint64_t getPidHash();
uint64_t hashCombine(uint64_t baseHash, uint64_t value);
ncclResult_t getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];  // 前缀，如网卡名 “eth0” 或 IP "192.168.1.1"
  int port;  // 端口
  int16_t rail;  // 轨道号（多轨网络用）
  int16_t plane;  // 平面号（多平面网络用）
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact, int* ifId = NULL);

static long log2i(long n) {
  return log2Down(n);
}

// Comparator function for qsort/bsearch to compare integers
static int compareInts(const void* a, const void* b) {
  int ia = *(const int*)a, ib = *(const int*)b;
  return (ia > ib) - (ia < ib);
}

inline uint64_t clockNano() {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

inline void clockRealtime(struct timespec* ts) {
  using namespace std::chrono;
  auto now = system_clock::now();
  auto secs = time_point_cast<seconds>(now);
  ts->tv_sec = static_cast<decltype(ts->tv_sec)>(secs.time_since_epoch().count());
  ts->tv_nsec = static_cast<decltype(ts->tv_nsec)>(duration_cast<nanoseconds>(now - secs).count());
}

/* get any bytes of random data from system RNG, return ncclSuccess (0) if it succeeds. */
inline ncclResult_t getRandomData(void* buffer, size_t bytes) {
  if (bytes > 0) {
    if (buffer == NULL) {
      WARN("getRandomData: buffer is NULL");
      return ncclSystemError;
    }
    try {
      // Linux 上通常后端是 /dev/urandom(或 RDRAND)，是非确定性真随机，不是 rand()/mt19937 那种可复现序列
      std::random_device rd;
      unsigned char* buf = static_cast<unsigned char*>(buffer);
      // 逐字节填。调 rd() 8 次，每次取低 8 bit(略浪费，但简单可移植)。
      for (size_t i = 0; i < bytes; ++i) {
        /*
        rd() 返回类型是 std::random_device::result_type，标准规定是某种无符号整数类型。
        在 libstdc++(gcc)上就是 unsigned int，即 32 位，取值范围 [0, 2³²-1](由 rd.min()=0、rd.max()=UINT_MAX 给出)。
        这里只取低 8 位。
        */
        buf[i] = static_cast<unsigned char>(rd());
      }
    } catch (const std::exception& e) {
      WARN("getRandomData: std::random_device failed: %s", e.what());
      return ncclSystemError;
    }
  }
  return ncclSuccess;
}

static inline int gcd(int a, int b) {
  // use the euclidian algorithm
  while (b != 0) {
    int temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

////////////////////////////////////////////////////////////////////////////////

template <typename Int>
inline void ncclAtomicRefCountIncrement(Int* refs) {
  COMPILER_ATOMIC_FETCH_ADD(refs, static_cast<Int>(1), std::memory_order_relaxed);
}

template <typename Int>
inline Int ncclAtomicRefCountDecrement(Int* refs) {
  return COMPILER_ATOMIC_SUB_FETCH(refs, static_cast<Int>(1), std::memory_order_acq_rel);
}

////////////////////////////////////////////////////////////////////////////////
/* ncclMemoryStack: Pools memory for fast LIFO ordered allocation. Note that
 * granularity of LIFO is not per object, instead frames containing many objects
 * are pushed and popped. Therefor deallocation is extremely cheap since its
 * done at the frame granularity.
 *
 * The initial state of the stack is with one frame, the "nil" frame, which
 * cannot be popped. Therefor objects allocated in the nil frame cannot be
 * deallocated sooner than stack destruction.
 */
struct ncclMemoryStack;

void ncclMemoryStackConstruct(struct ncclMemoryStack* me);
void ncclMemoryStackDestruct(struct ncclMemoryStack* me);
void ncclMemoryStackPush(struct ncclMemoryStack* me);
void ncclMemoryStackPop(struct ncclMemoryStack* me);
void* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t size, size_t align);
template <typename T>
T* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t n = 1);
template <typename Header, typename Element>
inline Header* ncclMemoryStackAllocInlineArray(struct ncclMemoryStack* me, size_t nElt);

////////////////////////////////////////////////////////////////////////////////
/* ncclMemoryPool: A free-list of same-sized allocations. It is an invalid for
 * a pool instance to ever hold objects whose type have differing
 * (sizeof(T), alignof(T)) pairs. The underlying memory is supplied by
 * a backing `ncclMemoryStack` passed during Alloc(). If memory
 * backing any currently held object is deallocated then it is an error to do
 * anything other than reconstruct it, after which it is a valid empty pool.
 */
struct ncclMemoryPool;

// Equivalent to zero-initialization
void ncclMemoryPoolConstruct(struct ncclMemoryPool* me);
template <typename T>
T* ncclMemoryPoolAlloc(struct ncclMemoryPool* me, struct ncclMemoryStack* backing);
template <typename T>
void ncclMemoryPoolFree(struct ncclMemoryPool* me, T* obj);
void ncclMemoryPoolTakeAll(struct ncclMemoryPool* me, struct ncclMemoryPool* from);

////////////////////////////////////////////////////////////////////////////////
/* ncclIntruQueue: A singly-linked list queue where the per-object next pointer
 * field is given via the `next` template argument.
 *
 * Example:
 *   struct Foo {
 *     struct Foo *next1, *next2; // can be a member of two lists at once
 *   };
 *   ncclIntruQueue<Foo, &Foo::next1> list1;
 *   ncclIntruQueue<Foo, &Foo::next2> list2;
 */
template <typename T, T* T::* next>
struct ncclIntruQueue;

template <typename T, T* T::* next>
void ncclIntruQueueConstruct(ncclIntruQueue<T, next>* me);
template <typename T, T* T::* next>
bool ncclIntruQueueEmpty(ncclIntruQueue<T, next>* me);
template <typename T, T* T::* next>
T* ncclIntruQueueHead(ncclIntruQueue<T, next>* me);
template <typename T, T* T::* next>
void ncclIntruQueueEnqueue(ncclIntruQueue<T, next>* me, T* x);
template <typename T, T* T::* next>
void ncclIntruQueueEnqueueFront(ncclIntruQueue<T, next>* me, T* x);
template <typename T, T* T::* next>
T* ncclIntruQueueDequeue(ncclIntruQueue<T, next>* me);
template <typename T, T* T::* next>
T* ncclIntruQueueTryDequeue(ncclIntruQueue<T, next>* me);
template <typename T, T* T::* next>
void ncclIntruQueueTransfer(ncclIntruQueue<T, next>* dst, ncclIntruQueue<T, next>* src);
template <typename T, T* T::* next>
inline T* ncclIntruQueueDelete(ncclIntruQueue<T, next>* me, T* x, bool (*cmp)(T*, T*));

////////////////////////////////////////////////////////////////////////////////
/* ncclThreadSignal: Couples a std::mutex and std::condition_variable together. The "mutex"
 * and "cond" fields are part of the public interface.
 */
struct ncclThreadSignal {
  std::mutex mutex;
  std::condition_variable cond;
};

// A convenience instance per-thread.
extern thread_local struct ncclThreadSignal ncclThreadSignalLocalInstance;

////////////////////////////////////////////////////////////////////////////////

template <typename T, T* T::* next>
struct ncclIntruQueueMpsc;

template <typename T, T* T::* next>
void ncclIntruQueueMpscConstruct(struct ncclIntruQueueMpsc<T, next>* me);
template <typename T, T* T::* next>
bool ncclIntruQueueMpscEmpty(struct ncclIntruQueueMpsc<T, next>* me);
// Enqueue element. Returns true if queue is not abandoned. Even if queue is
// abandoned the element enqueued, so the caller needs to make arrangements for
// the queue to be tended.
template <typename T, T* T::* next>
bool ncclIntruQueueMpscEnqueue(struct ncclIntruQueueMpsc<T, next>* me, T* x);
// Dequeue all elements at a glance. If there aren't any and `waitSome` is
// true then this call will wait until it can return a non empty list.
template <typename T, T* T::* next>
T* ncclIntruQueueMpscDequeueAll(struct ncclIntruQueueMpsc<T, next>* me, bool waitSome);
// Dequeue all elements and set queue to abandoned state.
template <typename T, T* T::* next>
T* ncclIntruQueueMpscAbandon(struct ncclIntruQueueMpsc<T, next>* me);

////////////////////////////////////////////////////////////////////////////////

// 一次规划周期内要批量分配大量小对象，生命周期整齐地整批结束，之后全部一次性“弹掉”，分配路径上不能有 malloc(锁、size 头、慢)，释放路径上不能逐个 free
struct ncclMemoryStack {
  // "大块"：一次 malloc 出来的大内存块(64KB 起步,几何增长)
  /*
  堆内存的真实来源。每个 hunk 就是 malloc(64KB + 前块大小) 的一整块，头部之后的空间用 bump 方式切给对象。
  多个 hunk 通过 above 串成链，topFrame.hunk 指向最高的非空块——它上方的块(经 ->above 可达)都是弹栈后遗留的空块，留作复用。
  */
  struct Hunk {
    struct Hunk* above; // reverse stack pointer
    size_t size; // size of this allocation (including this header struct)
  };
  // "外块"：放不进 hunk 的大对象的"登记条"
  /*
  当请求的对象太大(放不进 hunk 也不值得新开 hunk),就单独 malloc 给它本体，同时在 hunk 内只放一个 16 字节的代理头挂到当前帧的 unhunks 链上。
  这样 Pop 时能顺着链把每个外块对象逐个 free——hunk 内对象整块丢弃，外块对象单独回收。
  */
  struct Unhunk { // proxy header for objects allocated out-of-hunk
    struct Unhunk* next;
    void* obj;
  };
  // "快照"：某一作用域时刻的完整分配状态
  /*
  Frame 是 Push/Pop 的载体：进入新作用域前把当前状态拍快照，退出时恢复。快照本身也存在栈自己的 hunk 里
  */
  struct Frame {
    struct Hunk* hunk; // top of non-empty hunks
    uintptr_t bumper, end; // points into top hunk
    struct Unhunk* unhunks;
    struct Frame* below;
  };

  static void* allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align);
  static void* allocate(struct ncclMemoryStack* me, size_t size, size_t align);

  struct Hunk stub;  // 哨兵 hunk(永远空,让代码无需判空)
  struct Frame topFrame;  // 当前活跃帧
};

inline void ncclMemoryStackConstruct(struct ncclMemoryStack* me) {
  me->stub.above = nullptr;
  me->stub.size = 0;
  me->topFrame.hunk = &me->stub;
  me->topFrame.bumper = 0;
  me->topFrame.end = 0;
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = nullptr;
}

inline void* ncclMemoryStack::allocate(struct ncclMemoryStack* me, size_t size, size_t align) {
  /*
  把 bumper(当前分配水位)向上取整到 align 的倍数，得到候选对象地址 o，这个技巧只对 2 的幂对齐值成立(传入的都是 alignof(T),天然是 2 的幂)。
    align = 8 → 二进制 1000,align - 1 = 0111 是低位掩码，~(align-1) = ...11111000 把低 3 位清零；
    先 +7(111) 保证向上进位，再 & ~7 舍掉零头。
  */
  uintptr_t o = (me->topFrame.bumper + align - 1) & ~(uintptr_t(align) - 1);
  void* obj;
  if (COMPILER_EXPECT(o + size <= me->topFrame.end, true)) {
    // 分配 o ~ o+size 这块内存区域
    me->topFrame.bumper = o + size;
    obj = reinterpret_cast<void*>(o);
  } else {
    // 溢出路径。当前 hunk 塞不下，交给 spilled 处理(复用空块 / 新开 hunk / Unhunk 外块
    obj = allocateSpilled(me, size, align);
  }
  return obj;
}

inline void* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t size, size_t align) {
  void* obj = ncclMemoryStack::allocate(me, size, align);
  memset(obj, 0, size);
  return obj;
}

template <typename T>
inline T* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t n) {
  void* obj = ncclMemoryStack::allocate(me, n * sizeof(T), alignof(T));
  memset(obj, 0, n * sizeof(T));
  return (T*)obj;
}

template <typename Header, typename Element>
inline Header* ncclMemoryStackAllocInlineArray(struct ncclMemoryStack* me, size_t nElt) {
  size_t size = sizeof(Header);
  size = (size + alignof(Element) - 1) & -alignof(Element);
  size += nElt * sizeof(Element);
  size_t align = alignof(Header) < alignof(Element) ? alignof(Element) : alignof(Header);
  void* obj = ncclMemoryStack::allocate(me, size, align);
  memset(obj, 0, size);
  return (Header*)obj;
}

inline void ncclMemoryStackPush(struct ncclMemoryStack* me) {
  using Frame = ncclMemoryStack::Frame;
  Frame tmp = me->topFrame;
  Frame* snapshot = (Frame*)ncclMemoryStack::allocate(me, sizeof(Frame), alignof(Frame));
  *snapshot = tmp; // C++ struct assignment
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = snapshot;
}

inline void ncclMemoryStackPop(struct ncclMemoryStack* me) {
  ncclMemoryStack::Unhunk* un = me->topFrame.unhunks;
  while (un != nullptr) {
    free(un->obj);
    un = un->next;
  }
  me->topFrame = *me->topFrame.below; // C++ struct assignment
}

////////////////////////////////////////////////////////////////////////////////

/*
为什么需要"池"这一层？栈（ncclMemoryStack）的特性是只能整体弹出，不能单独释放某个对象。而 NCCL 有一类对象的模式是：
  数量巨大：每次集合通信、每个 channel 都要一个 ncclProxyOp；每个 group 一个 ncclKernelPlan。
  生命周期短且交错：plan 执行完就该回收，但下一个 plan 可能还在跑，无法用"整层弹出"的方式管。
  大小固定：同一种 task 结构体尺寸完全一样。
对这类对象，malloc/free 太慢（NCCL 的 enqueue 路径是延迟敏感的热路径），栈又不支持单个回收。
free list 池正是这个场景的标准答案：同尺寸对象反复分配释放，回收的内存原地挂回链表，下次直接摘走用。
*/

struct ncclMemoryPool {
  /*
  当一个对象被 free 回池里时，它已经"死了"，内容不再有意义，于是把它的头 8 字节拿来存 next 指针，
  把所有空闲对象串成一条链。这样 free list 不需要任何额外的元数据内存。
  */
  struct Cell {
    Cell* next;
  };
  struct Cell* head;
  struct Cell* tail; // meaningful only when head != nullptr
};

// 构造一个池等价于把它清零
inline void ncclMemoryPoolConstruct(struct ncclMemoryPool* me) {
  me->head = nullptr;
}

template <typename T>
inline T* ncclMemoryPoolAlloc(struct ncclMemoryPool* me, struct ncclMemoryStack* backing) {
  using Cell = ncclMemoryPool::Cell;
  Cell* cell;
  if (COMPILER_EXPECT(me->head != nullptr, true)) {  // 大概率链表不为空
    /*
    摘出链表头
    格子大小是第一次分配时按当时的 T 定下的，若后续用不同尺寸的 T 混用同一个池，复用旧格子时就可能装不下。
    这个约束纯靠调用方自律，没有任何编译期或运行期检查。所以 NCCL 才为每种对象开一个独立命名的池
    （memPool_ncclTaskColl、memPool_ncclProxyOp…），用命名约定代替类型系统来保证安全
    */
    cell = me->head;
    me->head = cell->next;
  } else {
    /*
    Free 时要把对象当 Cell 用、往它头上写 next 指针（8 字节）。
    如果 T 本身小于 8 字节，写 next 就会越界破坏相邻内存。取 max 保证每个格子至少能装下一个指针。
    */
    size_t cellSize = std::max(sizeof(Cell), sizeof(T));
    // cellAlign 取最大值保证两种用法的对齐要求都满足
    size_t cellAlign = std::max(alignof(Cell), alignof(T));
    /*
    外层的 ncclMemoryStackAlloc 会自动 memset(obj, 0, size)，而这里末尾统一有一次 memset(cell, 0, sizeof(T))，
    用外层版本会清零两遍。这是避免冗余工作的细节优化。
    */
    cell = (Cell*)ncclMemoryStack::allocate(backing, cellSize, cellAlign);
  }
  memset(cell, 0, sizeof(T));
  return reinterpret_cast<T*>(cell);
}

template <typename T>
inline void ncclMemoryPoolFree(struct ncclMemoryPool* me, T* obj) {
  /*
  内存完全不归还给栈，更不归还给 OS——只是挂回 free list。这意味着：
    池占用的内存量 = 历史峰值并发对象数 × 对象大小，之后稳定不变，永不下降。
    直到 comm 销毁、memPermanent 整体 ncclMemoryStackDestruct 时才真正还给 OS。
  */
  using Cell = ncclMemoryPool::Cell;
  Cell* cell = reinterpret_cast<Cell*>(obj);  // 把对象重新解释成链表节点
  cell->next = me->head;
  if (me->head == nullptr) me->tail = cell;  // 第一个入链者成为 tail
  me->head = cell;
}

inline void ncclMemoryPoolTakeAll(struct ncclMemoryPool* me, struct ncclMemoryPool* from) {
  if (from->head != nullptr) {
    from->tail->next = me->head;
    if (me->head == nullptr) me->tail = from->tail;
    me->head = from->head;
    from->head = nullptr;
  }
}

////////////////////////////////////////////////////////////////////////////////

/*
侵入式队列(intrusive queue) 的容器头
  typename T 是类型参数，表示队列元素的结构体类型
  T* T::* next 是非类型参数，表示一个“指向 T 中某个成员”的指针，该成员类型必须是 T*。
T::* 表示指向 T 中某个成员的指针，T* 表示该成员类型必须是 T* 
*/
template <typename T, T* T::* next>
struct ncclIntruQueue {
  T *head, *tail;
};

template <typename T, T* T::* next>
inline void ncclIntruQueueConstruct(ncclIntruQueue<T, next>* me) {
  me->head = nullptr;
  me->tail = nullptr;
}

template <typename T, T* T::* next>
inline bool ncclIntruQueueEmpty(ncclIntruQueue<T, next>* me) {
  return me->head == nullptr;
}

template <typename T, T* T::* next>
inline T* ncclIntruQueueHead(ncclIntruQueue<T, next>* me) {
  return me->head;
}

template <typename T, T* T::* next>
inline T* ncclIntruQueueTail(ncclIntruQueue<T, next>* me) {
  return me->tail;
}

template <typename T, T* T::* next>
inline void ncclIntruQueueEnqueue(ncclIntruQueue<T, next>* me, T* x) {
  x->*next = nullptr;
  // ->*next 表示取模板中指定的 next 这个成员指针的值，可能成员名并不是 next，由模板参数指定
  (me->head ? me->tail->*next : me->head) = x;
  me->tail = x;
}

template <typename T, T* T::* next>
inline void ncclIntruQueueEnqueueFront(ncclIntruQueue<T, next>* me, T* x) {
  if (me->head == nullptr) me->tail = x;
  x->*next = me->head;
  me->head = x;
}

template <typename T, T* T::* next>
inline T* ncclIntruQueueDequeue(ncclIntruQueue<T, next>* me) {
  T* ans = me->head;
  me->head = ans->*next;
  if (me->head == nullptr) me->tail = nullptr;
  return ans;
}

template <typename T, T* T::* next>
inline T* ncclIntruQueueDelete(ncclIntruQueue<T, next>* me, T* x, bool (*cmp)(T*, T*)) {
  T* prev = nullptr;
  T* cur = me->head;
  bool found = false;

  while (cur) {
    if (cmp(cur, x)) {
      found = true;
      break;
    }
    prev = cur;
    cur = cur->*next;
  }

  if (found) {
    if (prev == nullptr) me->head = cur->*next;
    else prev->*next = cur->*next;
    if (cur == me->tail) me->tail = prev;
  }
  return cur;
}

template <typename T, T* T::* next>
inline T* ncclIntruQueueTryDequeue(ncclIntruQueue<T, next>* me) {
  T* ans = me->head;
  if (ans != nullptr) {
    me->head = ans->*next;
    if (me->head == nullptr) me->tail = nullptr;
  }
  return ans;
}

template <typename T, T* T::* next>
void ncclIntruQueueTransfer(ncclIntruQueue<T, next>* dst, ncclIntruQueue<T, next>* src) {
  (dst->tail ? dst->tail->next : dst->head) = src->head;
  if (src->tail) dst->tail = src->tail;
  src->head = nullptr;
  src->tail = nullptr;
}

////////////////////////////////////////////////////////////////////////////////

/*
普通版 ncclIntruQueue，Enqueue/Dequeue 都是普通指针操作，没有任何原子操作或锁。这意味着它只能在单线程内（或调用方自己保证不并发）使用。

ncclIntruQueueMpsc，多生产者、单消费者、无锁。无锁（lock-free）队列，专为"多个线程同时往里塞、只有一个线程往外取"的场景设计。几个精妙点：
  1. tail 是 uintptr_t 而非指针，因为它同时兼任"状态标志"。它有三个魔术值：
    0x0：队列空
    0x1：队列空，且消费者正在睡眠等待（需要被唤醒）
    0x2：队列已被 abandon（弃用）
    其他值：指向真正的尾节点（把指针 reinterpret 成整数存）
    ncclIntruQueueMpscEmpty 判空就是 tail <= 0x2。
  2. 入队用原子交换（COMPILER_ATOMIC_EXCHANGE）。多个生产者线程同时入队时，靠这一条原子交换把新元素"抢"到 tail 位置，
  返回旧 tail。谁抢到旧 tail，谁负责把旧尾节点的 next 指向自己。这是无锁 MPSC 队列的经典手法，全程不加锁。
  3. 消费者休眠/唤醒机制（waiting 字段）：消费者取空时不会一直空转烧 CPU，先自旋 10 微秒（spin for first 10us），
  还没数据就把 tail 用 CAS 从 0x0 改成 0x1（标记"我要睡了"）然后 cond.wait 睡下。生产者入队时若发现旧 tail 是 0x1，
  就知道消费者在睡，负责 notify_all 把它叫醒。这就是 waiting/ncclThreadSignal 那套东西的用途。
*/
template <typename T, T* T::* next>
struct ncclIntruQueueMpsc {
  T* head;
  uintptr_t tail;  // 注意是整数，不是指针
  struct ncclThreadSignal* waiting;
};

template <typename T, T* T::* next>
void ncclIntruQueueMpscConstruct(struct ncclIntruQueueMpsc<T, next>* me) {
  me->head = nullptr;
  me->tail = 0x0;
  me->waiting = nullptr;
}

template <typename T, T* T::* next>
bool ncclIntruQueueMpscEmpty(struct ncclIntruQueueMpsc<T, next>* me) {
  return COMPILER_ATOMIC_LOAD(&me->tail, std::memory_order_relaxed) <= 0x2;
}

template <typename T, T* T::* next>
bool ncclIntruQueueMpscEnqueue(ncclIntruQueueMpsc<T, next>* me, T* x) {
  COMPILER_ATOMIC_STORE(&(x->*next), static_cast<T*>(nullptr), std::memory_order_relaxed);
  uintptr_t utail = COMPILER_ATOMIC_EXCHANGE(&me->tail, reinterpret_cast<uintptr_t>(x), std::memory_order_acq_rel);
  T* prev = reinterpret_cast<T*>(utail);
  T** prevNext = utail <= 0x2 ? &me->head : &(prev->*next);
  COMPILER_ATOMIC_STORE(prevNext, x, std::memory_order_relaxed);
  if (utail == 0x1) {
    // waiting
    std::atomic_thread_fence(std::memory_order_acquire); // to see me->waiting
    // This lock/unlock is essential to ensure we don't race ahead of the consumer
    // and signal the cond before they begin waiting on it.
    struct ncclThreadSignal* waiting = me->waiting;
    {
      std::unique_lock<std::mutex> lock(waiting->mutex);
    }
    waiting->cond.notify_all();
  }
  return utail != 0x2; // not abandoned
}

template <typename T, T* T::* next>
T* ncclIntruQueueMpscDequeueAll(ncclIntruQueueMpsc<T, next>* me, bool waitSome) {
  T* head = COMPILER_ATOMIC_LOAD(&me->head, std::memory_order_relaxed);
  if (head == nullptr) {
    if (!waitSome) return nullptr;
    uint64_t t0 = clockNano();
    bool sleeping = false;
    do {
      if (clockNano() - t0 >= 10 * 1000) {
        // spin for first 10us
        struct ncclThreadSignal* waitSignal = &ncclThreadSignalLocalInstance;
        std::unique_lock<std::mutex> lock(waitSignal->mutex);
        uintptr_t expected = sleeping ? 0x1 : 0x0;
        uintptr_t desired = 0x1;
        me->waiting = waitSignal; // release done by successful compare exchange
        if (COMPILER_ATOMIC_COMPARE_EXCHANGE(&me->tail, &expected, desired, std::memory_order_release,
                                             std::memory_order_relaxed)) {
          sleeping = true;
          waitSignal->cond.wait(lock);
        }
      }
      head = COMPILER_ATOMIC_LOAD(&me->head, std::memory_order_relaxed);
    } while (head == nullptr);
  }

  COMPILER_ATOMIC_STORE(&me->head, static_cast<T*>(nullptr), std::memory_order_relaxed);
  uintptr_t utail = COMPILER_ATOMIC_EXCHANGE(&me->tail, uintptr_t(0), std::memory_order_acq_rel);
  T* tail = utail <= 0x2 ? nullptr : reinterpret_cast<T*>(utail);
  T* x = head;
  while (x != tail) {
    T* x1;
    int spins = 0;
    while (true) {
      x1 = COMPILER_ATOMIC_LOAD(&(x->*next), std::memory_order_relaxed);
      if (x1 != nullptr) break;
      if (++spins == 1024) {
        spins = 1024 - 1;
        std::this_thread::yield();
      }
    }
    x = x1;
  }
  return head;
}

template <typename T, T* T::* next>
T* ncclIntruQueueMpscAbandon(ncclIntruQueueMpsc<T, next>* me) {
  uintptr_t expected = 0x0;
  if (COMPILER_ATOMIC_COMPARE_EXCHANGE(&me->tail, &expected, /*desired=*/uintptr_t(0x2), std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
    return nullptr;
  } else {
    int spins = 0;
    T* head;
    while (true) {
      head = COMPILER_ATOMIC_LOAD(&me->head, std::memory_order_relaxed);
      if (head != nullptr) break;
      if (++spins == 1024) {
        spins = 1024 - 1;
        std::this_thread::yield();
      }
    }
    COMPILER_ATOMIC_STORE(&me->head, static_cast<T*>(nullptr), std::memory_order_relaxed);
    uintptr_t utail = COMPILER_ATOMIC_EXCHANGE(&me->tail, uintptr_t(0x2), std::memory_order_acq_rel);
    T* tail = utail <= 0x2 ? nullptr : reinterpret_cast<T*>(utail);
    T* x = head;
    while (x != tail) {
      T* x1;
      spins = 0;
      while (true) {
        x1 = COMPILER_ATOMIC_LOAD(&(x->*next), std::memory_order_relaxed);
        if (x1 != nullptr) break;
        if (++spins == 1024) {
          spins = 1024 - 1;
          std::this_thread::yield();
        }
      }
      x = x1;
    }
    return head;
  }
}

ncclResult_t ncclBitsToString(uint32_t bits, uint32_t mask, const char* (*toStr)(int), char* buf, size_t bufLen,
                              const char* wildcard);

////////////////////////////////////////////////////////////////////////////////
// Hash function for pointer types (shared by address map implementations)
uint64_t ncclHashPointer(int hbits, void* key);

////////////////////////////////////////////////////////////////////////////////
// Intrusive address map implementation (avoids per-entry allocations)

/*
 * ncclIntruAddressMap Usage Contract
 * ===================================
 *
 * OVERVIEW:
 *   - Intrusive map that stores next-pointers directly in user objects
 *   - Avoids separate malloc per entry (only allocates bucket table)
 *   - Uses C++ templates for type safety with type-erased implementation
 *
 * CONSTRUCTION:
 *   - POD type with automatic zero-initialization for static/global instances
 *   - Example (global): static ncclIntruAddressMap<Obj, void*, &Obj::key, &Obj::next> globalMap;
 *   - Example (local): ncclIntruAddressMap<Obj, void*, &Obj::key, &Obj::next> localMap = {};
 *   - Zero-initialization works: ncclIntruAddressMap<...> map = {};
 *
 * DESTRUCTION:
 *   - NO explicit destructor function is provided
 *   - The map automatically cleans up internal memory when the last entry is removed
 *   - When count reaches 0, the internal table is freed and the map returns to
 *     zero-initialized state, making destruction trivial (no-op)
 *
 * USER RESPONSIBILITY:
 *   - Caller MUST remove all inserted objects before abandoning the map
 *   - Failure to remove all objects will leak memory (the internal bucket table)
 *   - Objects must outlive their presence in the map
 *   - Key and next-pointer fields are modified by the map
 *   - Delete/free objects separately after removing from map
 *
 * THREAD SAFETY:
 *   - This data structure is NOT thread-safe
 *   - Caller must provide external synchronization
 *
 * USAGE VALIDATION:
 *   Compile-time checks (via static_assert):
 *   - Key type size must be <= sizeof(uintptr_t)
 *
 *   Runtime checks (returns ncclInvalidUsage with WARN):
 *   - Map pointer must not be NULL
 *   - Object pointer must not be NULL (for Insert/Find operations)
 *   - Key size must be valid (0 < keySize <= sizeof(uintptr_t))
 */

// Untyped internal structure
struct ncclIntruAddressMap_untyped {
  int hbits;  // log2 of table size
  int count;  // number of entries
  void** table;
};

// Typed wrapper (uses composition for C compatibility)
template <typename Obj, typename Key, Key Obj::* keyField, Obj* Obj::* nextField>
struct ncclIntruAddressMap {
  // Compile-time checks for valid usage
  static_assert(sizeof(Key) <= sizeof(uintptr_t),
                "ncclIntruAddressMap: Key type size must be <= sizeof(uintptr_t). "
                "Keys larger than a pointer cannot be safely converted to uintptr_t.");

  ncclIntruAddressMap_untyped base;
};

// Destructor (optional - only needed if entries remain in map)
// Note: Map auto-cleans when last entry is removed, so this is only needed
// if abandoning a non-empty map to avoid leaking the bucket table.
template <typename Obj, typename Key, Key Obj::* keyField, Obj* Obj::* nextField>
static inline void ncclIntruAddressMapDestruct(struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map) {
  if (map->base.table != nullptr) {
    free(map->base.table);
    map->base.table = nullptr;
  }
  map->base.hbits = 0;
  map->base.count = 0;
}

// Internal untyped function prototypes
ncclResult_t ncclIntruAddressMapInsert_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                               int nextFieldOffset, uintptr_t key, void* object);

ncclResult_t ncclIntruAddressMapFind_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                             int nextFieldOffset, uintptr_t key, void** object);

ncclResult_t ncclIntruAddressMapRemove_untyped(struct ncclIntruAddressMap_untyped* map, int keySize, int keyFieldOffset,
                                               int nextFieldOffset, uintptr_t key);

// Typed template implementations (type-erasing wrappers)
template <typename Obj, typename Key, Key Obj::* keyField, Obj* Obj::* nextField>
static inline ncclResult_t ncclIntruAddressMapInsert(struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
                                                     Key key, Obj* object) {
  Obj dummy;
  // Using offsetof macro would be better except it won't work with non-C types,
  // like those that involve inheritance.
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  return ncclIntruAddressMapInsert_untyped(&map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
                                           reinterpret_cast<uintptr_t>(key), object);
}

template <typename Obj, typename Key, Key Obj::* keyField, Obj* Obj::* nextField>
static inline ncclResult_t ncclIntruAddressMapFind(struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
                                                   Key key, Obj** object) {
  Obj dummy;
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  void* tmp;
  ncclResult_t ret = ncclIntruAddressMapFind_untyped(&map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
                                                     reinterpret_cast<uintptr_t>(key), &tmp);
  *object = (Obj*)tmp;
  return ret;
}

template <typename Obj, typename Key, Key Obj::* keyField, Obj* Obj::* nextField>
static inline ncclResult_t ncclIntruAddressMapRemove(struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
                                                     Key key) {
  Obj dummy;
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  return ncclIntruAddressMapRemove_untyped(&map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
                                           reinterpret_cast<uintptr_t>(key));
}

inline ncclResult_t ncclThreadJoin(std::thread& thread) {
  try {
    thread.join();
    return ncclSuccess;
  } catch (const std::exception& e) {
    WARN("Thread join failed: %s", e.what());
    return ncclSystemError;
  }
}

// Convert NCCL numeric version to x.yy.zz string
static inline const char* ncclVersionToString(int version, char* buf, size_t bufSize) {
  snprintf(buf, bufSize, "%d.%d.%d", version / 10000, (version % 10000) / 100, version % 100);
  return buf;
}

#endif
