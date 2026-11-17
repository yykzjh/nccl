/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_STRONGSTREAM_H_
#define NCCL_STRONGSTREAM_H_

#include "nccl.h"
#include "checks.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <mutex>

// ncclCudaContext: wraps a CUDA context with per-context state.
struct ncclCudaContext;

// Get a ncclCudaContext to track the currently active CUDA context.
ncclResult_t ncclCudaContextTrack(struct ncclCudaContext** out, int launchOrderImplicit, uint64_t commHash);
// Drop reference.
void ncclCudaContextDrop(struct ncclCudaContext* cxt);

/* ncclCudaGraph: Wraps a cudaGraph_t so that we can support pre-graph CUDA runtimes
 * easily.
 */
struct ncclCudaGraph {
#if CUDART_VERSION >= 11030
  cudaStream_t origin;
  cudaGraph_t graph;
  unsigned long long graphId;
  int graphUsageMode;
#endif
};

inline struct ncclCudaGraph ncclCudaGraphNone(int graphUsageMode) {
  struct ncclCudaGraph tmp;
#if CUDART_VERSION >= 11030
  tmp.origin = nullptr;
  tmp.graph = nullptr;
  tmp.graphId = ULLONG_MAX;
  tmp.graphUsageMode = graphUsageMode;
#endif
  return tmp;
}

inline bool ncclCudaGraphValid(struct ncclCudaGraph graph) {
#if CUDART_VERSION >= 11030
  return graph.graphId != ULLONG_MAX;
#else
  return false;
#endif
}

inline bool ncclCudaGraphSame(struct ncclCudaGraph a, struct ncclCudaGraph b) {
#if CUDART_VERSION >= 11030
  return a.graphId == b.graphId;
#else
  return true;
#endif
}

ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph* graph, cudaStream_t stream, int graphUsageMode);
ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, cudaHostFn_t fn, void* arg);

/* ncclStrongStream: An abstraction over CUDA streams that do not lose their
 * identity while being captured. Regular streams have the deficiency that the
 * captured form of a stream in one graph launch has no relation to the
 * uncaptured stream or to the captured form in other graph launches. This makes
 * streams unfit for the use of serializing access to a persistent resource.
 * Strong streams have been introduced to address this need.
 *
 * All updates to a strong stream must be enclosed by a Acquire/Release pair.
 *
 * Acquire retrieves a "work" stream (cudaStream_t) which may be used to add
 * work.
 *
 * Release publishes the work streams work into the strong stream. The Release
 * must be issued by the same thread that did the Acquire.
 */
struct ncclStrongStream;

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss);
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss);

// Acquire the strong stream. Upon return `*workStream` will be usable to add work.
// `concurrent` indicates if other threads may be using the strong stream.
ncclResult_t ncclStrongStreamAcquire(struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent,
                                     cudaStream_t* workStream);

// Get the workStream for an already acquired strong stream.
// `concurrent` indicates if other threads may be using the strong stream.
ncclResult_t ncclStrongStreamAcquiredWorkStream(struct ncclCudaGraph graph, struct ncclStrongStream* ss,
                                                bool concurrent, cudaStream_t* workStream);

// Release of the strong stream.
// `concurrent` indicates if other threads may be using the strong stream.
ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent);
ncclResult_t ncclCudaGraphRecordEvent(struct ncclCudaGraph graph, cudaEvent_t event, cudaStream_t stream);

ncclResult_t ncclStreamWaitStream(cudaStream_t a, cudaStream_t b, cudaEvent_t scratchEvent);

// Like cudaStreamWaitEvent except `e` must be strictly ahead of everything in `s`.
ncclResult_t ncclStreamAdvanceToEvent(struct ncclCudaGraph g, cudaStream_t s, cudaEvent_t e);

// Synchrnoization does not need the strong stream to be acquired.
ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss);

////////////////////////////////////////////////////////////////////////////////

/*
普通 CUDA stream 有个缺陷：当它被 CUDA Graph capture 时，
它在某次 graph launch 里的"captured 形态"与未 capture 的原始 stream、
以及与其他 graph launch 里的 captured 形态没有任何关联。这就让普通 stream 无法胜任"串行化访问一个持久资源"的任务
*/

struct ncclStrongStreamCapture; // internal to ncclStrongStream

/*
ncclStrongStream 就是为了给 stream 一个在 capture 前后都不丢失的稳定身份。它的契约是:
  所有对它的更新必须包在一对 ncclStrongStreamAcquire / ncclStrongStreamRelease 之间。
  Acquire 返回一个真正可提交工作的 cudaStream_t(称为 work stream)。
  Release 把 work stream 上的工作"发布"回这条 strong stream,且必须由 Acquire 的同一线程调用。
*/
struct ncclStrongStream {
  // The stream to use for non-captured work.
  /*
  非 capture 场景下使用的真实 CUDA 流。在 ncclStrongStreamConstruct 里用 
  cudaStreamCreateWithFlags(..., cudaStreamNonBlocking) 创建。
  当不在 graph 中(graph.graphId == ULLONG_MAX)时,Acquire 直接把它作为 work stream 返回。
  */
  cudaStream_t liveStream;
  /*
  记录 liveStream 被哪个线程 Acquire(用 thread_local 变量地址 localThreadId() 作线程标识)。
  Release 时校验 liveAcquiredBy != localThreadId(),不匹配且 NCCL_LAUNCH_RACE_FATAL 开启
  就报 "host threads racing to launch NCCL on same device"——这就是"Release 必须由 Acquire 同线程发起"契约的强制执行。
  */
  void* liveAcquiredBy;
#if CUDART_VERSION >= 11030
  // This stream ever appeared in a graph capture.
  /*
  标记这条流是否曾经出现在某次 graph capture 中,一旦 capture 就置 true。
  用于 mixing 模式判断是否需要在 live 与 capture 形态之间用 serialEvent 补一条顺序边。
  */
  bool everCaptured;
  // 保护 captureHead 链表。仅当 concurrent=true(可能有其他线程并发使用同一条 strong stream)时,Acquire/Release 才加锁。
  std::mutex mutex;
  /*
  一个链表头，跟踪当前活跃的多个 graph capture。
  因为同一条 strong stream 可能被多个不同 graph 同时 capture,
  所以为每个 graphId 维护一个独立的 capture 节点(含该 graph 内代表本流的 captureStream),
  这样每个 graph 都能找到"属于自己"的那条流形态,身份不丢。Acquire 按 graphId 查表:命中就复用,不命中就新建(并回收已失活的节点)。
  */
  struct ncclStrongStreamCapture* captureHead;
  // The event used to establish order between graphs and streams. During acquire
  // this event is waited on, during release it is recorded to.
  /*
  整个机制的核心。Acquire 时等待它,Release 时记录到它。通过"Release 记录 / 下次 Acquire 等待"这一对操作，
  把分散在不同 graph launch、以及 live 与 capture 形态之间的工作串成一条全局有序链。
  */
  cudaEvent_t serialEvent;
#endif
};

struct ncclCudaContext {
  // 台账单链表指针
  struct ncclCudaContext* next;
  /*
  键：这个节点对应哪个 CUDA context
  CUDA context 之于 GPU，约等于进程之于 CPU。它是 CUDA driver 层的一个对象，容纳了某个进程在某块 GPU 上的全部状态，
  显存分配、stream、加载的 module/kernel、地址空间映射等。关键事实：
    cuCtxGetCurrent(&hcontext) 取的是当前 CPU 线程此刻绑定的那个 context，context 是"每线程当前值"的概念。
    实践中最常见的是 primary context：每块设备在每个进程里有一个主上下文，cudaSetDevice 后 runtime API 用的就是它。
  同一进程里对同一设备的多个 comm，通常共享同一个 CUcontext。
    但并非绝对唯一，一个进程理论上可对同一设备建多个 context。所以 NCCL 选择按 context（而非按 device）做台账的键，是更精确的口径。
  */
  CUcontext hcontext;
  // 有多少个 comm 正引用它
  int refCount;
  bool launchOrderImplicitEverEnabled;
  bool launchOrderImplicitEverDisabled;
  // 关键：该 context 的全局启动顺序流
  struct ncclStrongStream launchOrder;
};

#endif
