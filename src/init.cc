/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "coll_net.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin.h"
#endif
#include "rma.h"
#include "enqueue.h"
#include "graph.h"
#include "graph/topo.h"
#include "argcheck.h"
#include "tuner.h"
#include "ras.h"
#include "compiler.h"
#include "profiler.h"
#include "mnnvl.h"
#include <sys/stat.h>
#include "param.h"
#include "nvtx_payload_schemas.h"
#include "utils.h"
#include <mutex>
#include "ce_coll.h"
#include "nvtx.h"
#include "os.h"
#include "env.h"
#include "rma/rma.h"

#define STR2(v) #v
#define STR(v) STR2(v)

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = {"Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce"};
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {"Tree",     "Ring", "CollNetDirect", "CollNetChain", "NVLS",
                                                "NVLSTree", "PAT"};
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = {"LL", "LL128", "Simple"};

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(RuntimeConnect, "RUNTIME_CONNECT", 1);
NCCL_PARAM(WinEnable, "WIN_ENABLE", 1);
NCCL_PARAM(CollnetEnable, "COLLNET_ENABLE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NumRmaCtx, "NUM_RMA_CTX", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MaxP2pPeers, "P2P_MAX_PEERS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(SetCpuStackSize, "SET_CPU_STACK_SIZE", 1);
// 控制是否允许多个 rank 共用同一块 GPU 或同一个 MLOPart 分区
NCCL_PARAM(MultiRankGpuEnable, "MULTI_RANK_GPU_ENABLE", 0);

extern int64_t ncclParamSingleProcMemRegEnable();

static bool ctaPolicyIsValid(int ctaPolicy) {
  int availCtaPolicies[3] = {NCCL_CTA_POLICY_DEFAULT, NCCL_CTA_POLICY_EFFICIENCY, NCCL_CTA_POLICY_ZERO};
  int maxPolicy = 0;
  for (int i = 0; i < 3; ++i) {
    maxPolicy |= availCtaPolicies[i];
  }
  return ctaPolicy >= 0 && ctaPolicy <= maxPolicy;
}

static int ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
static void getEnvCtaPolicyOnce() {
  const char* env = ncclGetEnv("NCCL_CTA_POLICY");
  if (env == NULL) return;

  // support the old CTA Policy assignment (only valid for 0-9)
  if (isdigit(env[0])) {
    switch (env[0]) {
    case '0':
      ctaPolicyEnv = NCCL_CTA_POLICY_DEFAULT;
      break;
    case '1':
      ctaPolicyEnv = NCCL_CTA_POLICY_EFFICIENCY;
      break;
    case '2':
      ctaPolicyEnv = NCCL_CTA_POLICY_ZERO;
      break;
    default:
      INFO(NCCL_ENV, "Unknown CTA policy; the legacy usage of NCCL_CTA_POLICY only supports the value of 0 (DEFAULT), "
                     "1 (EFFICIENCY), or 2 (ZERO). Using DEFAULT instead.");
    };
  } else {
    // newer way allows the user to combine the modes
    char* str = strdup(env);
    char* token = strtok(str, "|");
    while (token) {
      int tokenPolicy = NCCL_CONFIG_UNDEF_INT;
      if (strcasecmp(token, "DEFAULT") == 0) tokenPolicy = NCCL_CTA_POLICY_DEFAULT;
      else if (strcasecmp(token, "EFFICIENCY") == 0) tokenPolicy = NCCL_CTA_POLICY_EFFICIENCY;
      else if (strcasecmp(token, "ZERO") == 0) tokenPolicy = NCCL_CTA_POLICY_ZERO;
      else INFO(NCCL_ENV, "Unknown CTA policy %s passed as environment variable. Ignoring.", token);
      if (tokenPolicy != NCCL_CONFIG_UNDEF_INT) {
        if (ctaPolicyEnv == NCCL_CONFIG_UNDEF_INT) ctaPolicyEnv = tokenPolicy;
        else ctaPolicyEnv |= tokenPolicy;
      }
      token = strtok(NULL, "|");
    }
    if (ctaPolicyEnv == NCCL_CONFIG_UNDEF_INT) {
      INFO(NCCL_ENV, "No valid CTA policies found in NCCL_CTA_POLICY=%s.", env);
    } else {
      INFO(NCCL_ENV, "Parsed environment variable NCCL_CTA_POLICY=%s to %d", env, ctaPolicyEnv);
    }
    free(str);
  }
}

static ncclResult_t commReclaim(struct ncclAsyncJob* job_);

// GDRCOPY support: Off by default
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);
NCCL_PARAM(IgnoreNetMismatch, "IGNORE_NET_MISMATCH", 1);
NCCL_PARAM(IgnoreCollNetMismatch, "IGNORE_COLLNET_MISMATCH", 0);

// GDRCOPY support
gdr_t ncclGdrCopy = NULL;

ncclResult_t initGdrCopy() {
  // 根据 NCCL_GDRCOPY_ENABLE 环境变量确定是否初始化 GDRCOPY
  if (ncclParamGdrCopyEnable() == 1) {
    // 动态加载 GDRCopy 库，并加载要用到的 API 函数，调用 gdr_open 获取句柄
    ncclGdrCopy = ncclGdrInit();
  }
  return ncclSuccess;
}

static ncclResult_t initResult = ncclSuccess;
static std::once_flag initOnceFlag;

static void initOnceFunc() {
  // 初始化 pthread 线程栈大小至少为 8MiB
  NCCLCHECKGOTO(ncclOsInitialize(), initResult, exit);
  // 动态初始化 GDRCopy API，保存句柄
  initGdrCopy();
  // 根据用户指定 NCCL_COMM_ID，或者其他 hints 环境变量 NCCL_SOCKET_IFNAME、NCCL_SOCKET_FAMILY 来自动检测用于后续 bootstrap 的网口
  // Always initialize bootstrap network
  NCCLCHECKGOTO(bootstrapNetInit(), initResult, exit);
  // 向 NVTX 注册 ncclRedOp_t 枚举类型到 profiler 可读字符串的映射表
  initNvtxRegisteredEnums();
exit:;
}

static ncclResult_t ncclInit() {
  std::call_once(initOnceFlag, initOnceFunc);
  return initResult;
}

// 全局缓存初始化环境配置的结果
static ncclResult_t envInitResult = ncclSuccess;
// 只执行一次的标记
static std::once_flag envInitOnceFlag;

// 真正执行的初始化环境配置的函数
static void envInitOnceFunc() {
  NCCLCHECKGOTO(ncclEnvPluginInit(), envInitResult, exit);
exit:;
}

ncclResult_t ncclInitEnv() {
  /*
  std::call_once 是 C++11 <mutex> 提供的线程安全的一次性调用工具
  std::call_once 的规则是"若可调用对象抛异常，flag 不置位，下次调用会重新执行"。
  这里不抛，所以初始化失败也算"已完成"，后续每次调用都直接返回同一个缓存的失败码。
  对于"网卡探测失败"这种确定性错误，重试确实没意义。
  */
  std::call_once(envInitOnceFlag, envInitOnceFunc);
  return envInitResult;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  // 加载 NCCL 配置文件、初始化环境插件
  NCCLCHECK(ncclInitEnv());
  // 设置 pthread 线程栈大小、GDRCopy API，保存句柄、获取 bootstrap 网口、初始化 NVTX 枚举类型
  NCCLCHECK(ncclInit());
  // 确保分配好内存了
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  /*
  ncclUniqueId 是按照 1 节点对齐，大小为 128 字节；ncclBootstrapHandle 是按照 8 字节对齐，大小为 40 字节。
  强转后写 ->magic（一次 8 字节写）落在奇地址上：x86 上慢但能跑，ARM/RISC-V 上某些指令直接 SIGBUS，而且编译器一旦认为指针是对齐的就可能发射 movdqa 之类的对齐指令 —— 这是未定义行为，不是"性能问题"。
  正确解法是在栈上声明一个真正对齐的 struct ncclBootstrapHandle（编译器保证 8 对齐），填好之后用 memcpy 搬过去。memcpy 的语义是字节级拷贝，对源和目标都没有对齐要求，编译器会根据实际情况选指令。
  */
  struct ncclBootstrapHandle handle;  // 栈上未初始化
  /*
  调用者	                        第二参	        magic 来源
  ncclGetUniqueId（这里）	         NULL	         getRandomData() 随机
  ncclCommGetUniqueId（grow）	    comm	        hashCombine(comm->magic, comm->childCount+1)
  */
  // 指定了 NCCL_COMM_ID，解析后填写到 handle->addr，magic 固定
  // 没指定 NCCL_COMM_ID，采用初始化选择的网口地址填充 handle->addr，随机生成 magic
  NCCLCHECK(bootstrapGetUniqueId(&handle, NULL));
  // memcpy 只写前 40 字节，剩下 88 字节是填充。commHash hash 的是整个 128 字节，包括那 88 字节填充。memset 是跨进程 hash 一致性的必要条件。
  // ncclUniqueId and bootstrapHandle don't have the same size and alignment
  // reset to 0 to avoid undefined data
  memset(out, 0, sizeof(*out));
  // copy to avoid alignment mismatch
  memcpy(out, &handle, sizeof(handle));
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)getHash(out->internal, NCCL_UNIQUE_ID_BYTES));
  return ncclSuccess;
}

// Prevent compiler from optimizing out these operations
#if defined(_MSC_VER)
#define NCCL_NO_OPTIMIZE
#elif defined(__clang__)
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  // Important that this does not trash intraComm0.
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
  comm->startMagic = comm->endMagic = 0;
}

#undef NCCL_NO_OPTIMIZE

static ncclResult_t ncclDestructorFnFree(struct ncclDestructor* dtor) {
  free(dtor->obj);
  return ncclSuccess;
}
void ncclCommPushFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnFree;
  dtor->obj = obj;
  dtor->comm = comm;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaFree(dtor->obj, dtor->comm->memManager));
  return ncclSuccess;
}
void ncclCommPushCudaFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaFree;
  dtor->obj = obj;
  dtor->comm = comm;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaHostFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaHostFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaHostFree;
  dtor->obj = obj;
  dtor->comm = comm;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaGdrFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclGdrCudaFree(dtor->obj, dtor->comm->memManager));
  return ncclSuccess;
}
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaGdrFree;
  dtor->obj = handle;
  dtor->comm = comm;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t commFree(ncclComm_t comm) {
  int abort = 0;
  /* commFree() should not involve any sync among ranks. */
  if (comm == NULL) return ncclSuccess;

  NCCLCHECK(ncclCeFinalize(comm));
  NCCLCHECK(ncclRmaCeFinalize(comm));

  if (comm->symmetricSupport) {
    NCCLCHECK(ncclSymkFinalize(comm));
    NCCLCHECK(ncclDevrFinalize(comm));
  }
  NCCLCHECK(ncclRasCommFini(comm));

  /* in commReclaim, we have guaranteed only last rank which calls ncclCommDestroy() will
   * free all intra-process communicators; therefore, we only need to focus on local
   * resource cleanup in commFree(). */
  if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread.joinable()) {
    comm->proxyState->thread.join();
    if (comm->proxyState->threadUDS.joinable()) {
      // UDS support
      comm->proxyState->threadUDS.join();
    }
  }

  // Free any pending suspend/resume tasks
  while (!ncclIntruQueueEmpty(&comm->suspendTaskQueue)) {
    struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->suspendTaskQueue);
    free(task);
  }
  while (!ncclIntruQueueEmpty(&comm->resumeTaskQueue)) {
    struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->resumeTaskQueue);
    free(task);
  }
  // Destroy dynamic memory manager only after all proxy threads have been joined
  NCCLCHECK(ncclMemManagerDestroy(comm));

  if (comm->memPool) CUDACHECK(cudaMemPoolDestroy(comm->memPool));

  delete[] comm->userRedOps;

  free(comm->connectSend);
  free(comm->connectRecv);

  free(comm->peerInfo);
  if (comm->topo) ncclTopoFree(comm->topo);
  if (comm->nodeRanks) {
    for (int n = 0; n < comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
    free(comm->nodeRanks);
  }
  free(comm->rankToNode);
  free(comm->rankToLocalRank);
  free(comm->collNetHeads);
  free(comm->clique.ranks);

  if (comm->bootstrap) NCCLCHECK(bootstrapClose(comm->bootstrap));

  for (int channel = 0; channel < MAXCHANNELS; channel++) {
    NCCLCHECK(freeChannel(comm->channels + channel, comm->nRanks, 1, comm->localRanks, comm));
  }

  // GIN may use proxy. We need to finalize it before destroying the proxy.
  NCCLCHECK(ncclGinHostFinalize(comm));
  NCCLCHECK(ncclRmaProxyFinalize(comm));

  int sharedResRefCount = 0;
  if (comm->sharedRes) {
    sharedResRefCount = ncclAtomicRefCountDecrement(&comm->sharedRes->refCount);
    if (sharedResRefCount == 0) {
      for (int c = 0; c < MAXCHANNELS; c++) {
        if (comm->sharedRes->peers[c]) free(comm->sharedRes->peers[c]);
        if (comm->sharedRes->devPeers[c]) ncclCudaFree(comm->sharedRes->devPeers[c], comm->memManager);
      }
      free(comm->sharedRes->tpRankToLocalRank);
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->hostStream));
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->deviceStream));
      CUDACHECK(cudaEventDestroy(comm->sharedRes->launchEvent));
      CUDACHECK(cudaEventDestroy(comm->sharedRes->scratchEvent));
      NCCLCHECK(ncclProxyDestroy(comm));
      delete comm->sharedRes;
    }
  }

  if (comm->nvlsSupport) NCCLCHECK(ncclNvlsFree(comm));

  struct ncclDestructor* dtor = comm->destructorHead;
  while (dtor != nullptr) {
    NCCLCHECK(dtor->fn(dtor));
    dtor = dtor->next;
  }

  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  abort = *comm->abortFlag;
  if (ncclAtomicRefCountDecrement(comm->abortFlagRefCount) == 0) {
    free(comm->abortFlag);
    NCCLCHECK(ncclCudaHostFree((void*)comm->abortFlagDev));
    free(comm->abortFlagRefCount);
  }
  free((void*)comm->config.netName);

  free(comm->topParentRanks);
  free(comm->topParentLocalRanks);
  free(comm->gproxyConn);

  NCCLCHECK(ncclRegCleanup(comm));

  INFO(NCCL_DESTROY, "comm %p rank %d nranks %d cudaDev %d busId %lx - %s COMPLETE", comm, comm->rank, comm->nRanks,
       comm->cudaDev, comm->busId, abort ? "Abort" : "Destroy");

  commPoison(comm); // poison comm before free to avoid comm reuse.
  NCCLCHECK(ncclProfilerPluginFinalize(comm));
  if (sharedResRefCount == 0) {
    NCCLCHECK(ncclNetFinalize(comm));
    NCCLCHECK(ncclGinFinalize(comm));
    NCCLCHECK(ncclRmaFinalize(comm));
  }
  ncclCudaContextDrop(comm->context);
  free(comm);

  return ncclSuccess;
}

NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
#define NCCL_WORK_FIFO_BYTES_DEFAULT (1 << 20)
NCCL_PARAM(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);
NCCL_PARAM(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);
enum ncclLaunchMode ncclParamLaunchMode;

// 默认启用 DMA-BUF
NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 1);

// Detect DMA-BUF support
static ncclResult_t dmaBufSupported(struct ncclComm* comm) {
  // 没启用 DMA-BUF、NET 插件中没有 regMrDmaBuf 函数、CUDA 没有初始化
  if (ncclParamDmaBufEnable() == 0 || comm->ncclNet->regMrDmaBuf == NULL || ncclCudaLibraryInit() != ncclSuccess) {
    return ncclInternalError;
  }
// CUDA >= 11.7
#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion;
  CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));
  if (CUPFN(cuDeviceGet) == NULL || cudaDriverVersion < 11070) return ncclInternalError;
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));
  // 查询当前设备属性 CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED，是否支持 DMA-BUF
  // Query device to see if DMA-BUF support is available
  (void)CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0) return ncclInternalError;
  INFO(NCCL_INIT, "DMA-BUF is available on GPU device %d", comm->cudaDev);
  return ncclSuccess;
#endif
  return ncclInternalError;
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* comm must be ready, or error will be reported */
  ncclResult_t ret = ncclSuccess;
  if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
    ncclGroupJobAbort(comm->groupJob);
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret == ncclInProgress) {
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
      ret = ncclInvalidArgument;
      goto exit;
    }
    /* if ret is not ncclInProgress, we just keep it. */
  }

exit:
  return ret;
}

// 填好 rank/nRanks、创建本进程本 GPU 侧的全部资源(流、事件、内存池、插件、队列)，并处理 split 场景下的资源继承
static ncclResult_t commAlloc(struct ncclComm* comm, struct ncclComm* parent, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  // 初始化通信域生命期的栈式分配器(后续 devComm 上传、ringTasks 等从这里分配，销毁时一次性回收)
  ncclMemoryStackConstruct(&comm->memPermanent);
  // 初始化单次 group 调用作用域的分配器，每次 ncclGroupCommLeave 会执行 ncclMemoryStackPop(&comm->memScoped),整层弹掉，这就是 NCCL 集合操作“零碎片”的内存模型
  ncclMemoryStackConstruct(&comm->memScoped);
  // 延迟析构链表头(错误路径上按序回滚用)
  comm->destructorHead = nullptr;
  comm->rank = rank;
  comm->nRanks = ndev;

  /*
  设备上下文异常(驱动失配、设备被独占、ECC 错误)是初始化失败的头号原因，与其在深处的拓扑探测里炸出难懂的错，
  不如在第一步就用 cudaGetDevice 探出来。compCap 记录计算能力(如 900),后续算法/NVLS 支持判定大量使用。
  */
  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  CUDACHECK(cudaGetDevice(&comm->cudaDev));
  comm->compCap = ncclCudaCompCap();

  if (parent == NULL || !parent->shareResources) {
    struct ncclSharedResources* sharedRes;
    NEW_NOTHROW(sharedRes, ncclSharedResources);
    /* most of attributes are assigned later in initTransportsRank(). */
    // 归属标记与 rank→localRank 映射
    sharedRes->owner = comm;
    sharedRes->tpNRanks = comm->nRanks;
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm->nRanks));
    // 初始化设备侧工作流，devComm 上传、kernel 工作缓冲分配都走它
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->deviceStream));
    // 初始化主机侧控制流，proxy op 上传等
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->hostStream));
    // 初始化两个事件，均带 cudaEventDisableTiming，只做顺序同步不做计时，比计时事件更轻量
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->launchEvent, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->scratchEvent, cudaEventDisableTiming));
    comm->sharedRes = sharedRes;
    sharedRes->refCount = 1;
    // 初始化三大子系统：ncclNetInit(网络插件)、ncclRmaInit(单边内存访问)、ncclGinInit(GPU-initiated 通信扩展)
    NCCLCHECK(ncclNetInit(comm));
    NCCLCHECK(ncclRmaInit(comm));
    NCCLCHECK(ncclGinInit(comm));
  } else {
    comm->sharedRes = parent->sharedRes;
    ncclAtomicRefCountIncrement(&parent->sharedRes->refCount);
    NCCLCHECK(ncclNetInitFromParent(comm, parent));
    NCCLCHECK(ncclRmaInitFromParent(comm, parent));
    NCCLCHECK(ncclGinInitFromParent(comm, parent));
  }

  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);

  // 判断如果存在继承关系，子 comm 和父 comm 的 NET 要一样
  if (parent && parent->shareResources) {
    if (parent->ncclNet != comm->ncclNet) {
      WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s",
           parent->ncclNet->name, comm->ncclNet->name);
      return ncclInvalidUsage;
    }
  }

  // Initialize memory manager
  if (parent && parent->shareResources && parent->memManager) {
    // Share parent's memory manager
    comm->memManager = parent->memManager;
    ncclAtomicRefCountIncrement(&comm->memManager->refCount);
    INFO(NCCL_INIT, "MemManager: Shared from parent, refCount=%d", comm->memManager->refCount);
  } else {
    // 创建新的 memory manager
    // Create new memory manager
    NCCLCHECK(ncclMemManagerInit(comm));
  }

  // 进程全局保存当前 context 和对应的 stream 到全局链表
  NCCLCHECK(ncclCudaContextTrack(&comm->context));

  // 获取当前 GPU 的 PCIe BusId 并转换为 int64_t 格式
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  nvmlDevice_t nvmlDev;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];  // 16
  NCCLCHECK(int64ToBusId(comm->busId, busId));  // int64_t 转回字符串 BusId
  /*
  NVML = NVIDIA Management Library，是英伟达的管理/监控库（nvidia-smi 底层就是它）。它和 CUDA runtime 是两套东西：
    CUDA runtime：面向计算，cudaDev 是"我这个进程能看到的第几块卡"，会被 CUDA_VISIBLE_DEVICES 过滤和重排。
    NVML：面向管理，站在整机视角枚举所有 GPU（无视 CUDA_VISIBLE_DEVICES），编号在同一台机器上对所有进程一致。
  为什么 NCCL 需要 nvmlDev（cudaDev 为什么不够），多进程训练时，进程 A 和进程 B 可能各自设了不同的 CUDA_VISIBLE_DEVICES，
  于是两个进程都以为自己在用 cudaDev 0，但它们可能：
    + 落在同一块物理 GPU上（配置错误，应报警），或
    + 落在不同物理 GPU上（正常）。
  */
 // busId 字符串 → NVML 句柄
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  // NVML 句柄 → NVML 索引
  NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&comm->nvmlDev));

  TRACE(NCCL_INIT, "comm %p rank %d nranks %d cudaDev %d busId %lx compCap %d", comm, rank, ndev, comm->cudaDev,
        comm->busId, comm->compCap);

  /*
  检测是否支持 DMA-BUF
  DMA-BUF 是 Linux 内核的“跨设备缓冲区共享”框架。一个驱动(exporter，导出方)分配的缓冲区，可以通过一个文件描述符(fd)作为句柄，
  安全地共享给其它驱动/子系统(importer，导入方)，让后者能直接对这块内存做 DMA，而无需了解导出方的内部细节。
  它最初为图形栈(在 GPU 显示驱动、摄像头、视频编解码器之间共享 framebuffer)而生，如今被广泛用于 GPU↔网卡 的零拷贝传输。

  放到 NCCL 的 GPUDirect RDMA(GDR)场景：
  CUDA 驱动(exporter)                RDMA 网卡驱动(importer，如 mlx5)
  GPU 显存缓冲区  ──cuMemGetHandleForAddressRange──►  DMA-BUF fd  ──regMrDmaBuf──►  注册为 RDMA MR
                                                                                      │
                                                                    网卡直接 DMA 读写 GPU 显存
  
  对比老方案，为什么需要它：
    老方案是 nvidia-peermem/nv_peer_mem 内核模块。网卡驱动通过 NVIDIA 的私有 peer API“伸进”NVIDIA 驱动去映射显存。
  这是个 out-of-tree(非主线)内核模块,需单独安装、维护，脆弱。
    DMA-BUF 是现代、主线内核、厂商中立的标准做法。显存通过标准 fd 机制共享，网卡驱动导入 fd 即可，不再需要 peermem 内核模块，更健壮。

  GDR 是“目的/能力”，DMA-BUF 是“手段/实现方式”之一。二者不在同一概念层级，不是竞争或替代关系。
    GPUDirect RDMA(GDR) = 一种能力：网卡(NIC)直接对 GPU 显存做 RDMA 读写，零拷贝、不经过主机内存中转。
    DMA-BUF = 达成 GDR 的两种内核机制之一(另一种是 nv_peermem)。这是“用什么办法达成”。
  */
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;

  /*
  collNetSupportMatrix 是 “网内归约(collNet / SHARP)能力查找表，是一个[归约操作][数据类型] 的二维布尔矩阵，
  记录 SHARP 硬件支持哪些 (op, datatype) 组合的网内归约。它是 NCCL 决定“这次 AllReduce 能否走网内归约，
  还是回落到 Ring/Tree”的关键依据。

  collNet = collective network = 网内集合运算卸载，
  典型实现是 NVIDIA SHARP(Scalable Hierarchical Aggregation and Reduction Protocol)。
  它让交换机自己完成 sum/prod/max/min 归约，而不是让 GPU 之间来回搬运全部数据再各自计算，大幅减少网络流量和延迟。
  
  在 NCCL 里它是一套独立于点对点 ncclNet_t 的插件接口 ncclCollNet_t(comm->ncclCollNet)。
  collNetSupportMatrix 就是描述这套硬件“具体能算哪些 (操作, 类型)”的能力表。
  */
  memset(comm->collNetSupportMatrix, 0, sizeof(comm->collNetSupportMatrix));

  // 内存池链表头置为 nullptr
  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);

  /*
  把三个"链表 next 指针"字段全部预置成一个特殊地址值 0x1，
  这里的 0x1 不是一个合法指针，也不是 nullptr，而是一个刻意选的"魔术哨兵值"。
  NCCL 在一次 ncclGroupStart()/ncclGroupEnd() 之间，会把本线程内所有"参与了这次 group 的 comm"串成一条链表：
    groupNext[ncclGroupTaskTypeCollective] —— 参与集合通信任务的 comm 链
    groupNext[ncclGroupTaskTypeSymRegister] —— 参与对称内存注册任务的 comm 链
    preconnectNext —— 上面这条链的子集，专指"本次还需要预建连接（preconnect）"的 comm 链
  链表头在 group.cc 里是 thread_local 的全局变量（ncclGroupCommHead[] / ncclGroupCommPreconnectHead），
  next 指针则内嵌在每个 comm 自己身上。
  */
  for (int i = 0; i < ncclGroupTaskTypeNum; i++) {
    comm->groupNext[i] = reinterpret_cast<struct ncclComm*>(0x1);
  }
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);

  /*
  64 <= 64，成立，编译通过。
  connectSend/connectRecv 是位掩码，每个元素对应一个 rank，一个元素 bits 数为 64，可以表示最多 64 条 channels。
  connectSend[peer] / connectRecv[peer] 里的每个 uint64_t 被当成位图（bitmask）使用，第 i 个 bit 代表"与该 peer 在 channel i 上是否需要建立连接。
  */
  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend) * 8,
                "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv) * 8,
                "comm->connectRecv must have enough bits for all channels");
  // 给每个 rank 分配一个 uint64_t 的位图（bitmask）
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  /*
  把所有 channel 标记为"未初始化"
  comm->channels[] 是一个定长数组，长度 MAXCHANNELS（=64），一次性内嵌在 comm 里。
  但一个 comm 真正会用到多少 channel 是运行时才决定的（取决于拓扑、算法），所以这里先把全部 64 个槽位标记成"空"。
  */
  // Mark channels as non initialized.
  for (int c = 0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;
  /*
  为 broadcast 的环形排序预分配一个数组，它是一个长度为 nRanks 的指针数组，
  用于 broadcast 这类"有根（rooted）"集合通信在 ring 算法上做按环深度排序的临时工作区。
  几个要点：
    1. 从 memPermanent 分配；
    2. 元素类型 void*，实际使用时按具体类型强转；
    3. 这里只是"预留一次内存"，具体内容每次调度时重填。
  */
  // for bcast: init the ringTasks and min/max bcast peer
  comm->ringTasks = ncclMemoryStackAlloc<void*>(&comm->memPermanent, comm->nRanks);
  comm->planner.bcast_info.minBcastPeer = INT_MAX;
  comm->planner.bcast_info.maxBcastPeer = INT_MIN;

  // comm->topParentRanks 映射初始化为 rank_i = i
  if (comm->topParentRanks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; ++i) comm->topParentRanks[i] = i;
  }

  // 初始化多生产者单消费者的无锁队列
  ncclIntruQueueMpscConstruct(&comm->callbackQueue);
  // 初始化如下四个链表队列的头尾指针为 nullptr
  ncclIntruQueueConstruct(&comm->legacyRegCleanupQueue);
  ncclIntruQueueConstruct(&comm->ceInitTaskQueue);
  ncclIntruQueueConstruct(&comm->suspendTaskQueue);
  ncclIntruQueueConstruct(&comm->resumeTaskQueue);

  /*
  用来把用户给的 [addr, addr+size) 区间向外扩展到整页边界，起始地址向下取整到页首、结束地址向上取整到页尾。原因有两个：
    1. 硬件要求：网卡注册内存（pin 住物理页、建 MTT/MPT 映射表）是以页为最小单位的。
    哪怕你只想注册 1 个字节，底层也必须按整页 pin。所以区间必须页对齐。
    2. 缓存命中率：ncclRegCache 的命中判据是"区间包含"，新区间被已有条目完全覆盖才算命中、复用。
    把地址统一页对齐后，才能让"同一大块内存里的不同小切片"稳定命中同一个缓存条目，避免因几字节的偏差反复新建注册。
  */
  comm->regCache.pageSize = ncclOsGetPageSize();

  /*
  创建 CUDA 显存池，池句柄存进 comm->memPool，之后用它做 stream-ordered 的显存分配。
  制造一个局部作用域：
    把 props、releaseThreshold 这两个临时变量限制在这个块内，出了块就销毁，不污染外层函数的命名空间。
    让这段"创建+配置"逻辑在视觉和作用域上成为一个独立单元。
  */
  do {
    /*
    props = {} 先全部清零，再逐个填关键字段：
      allocType=cudaMemAllocationTypePinned，分配类型为"固定（pinned）"设备内存，这是显存池当前唯一支持的分配类型
      handleTypes=cudaMemHandleTypeNone，不需要可导出/共享的句柄，这个池的内存纯本地自用，不参与跨进程 IPC 共享。
      说明这个池分配出来的显存不打算共享给别的进程，只是本 comm 内部用的暂存/工作缓冲，所以不需要可导出句柄。
      location.type=cudaMemLocationTypeDevice，内存位于 GPU 设备上（而非主机）
      location.id=comm->cudaDev，具体是哪一块 GPU，绑定到本 comm 的 CUDA 设备号
    */
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = comm->cudaDev;
    CUDACHECK(cudaMemPoolCreate(&comm->memPool, &props));
    // UINT64_MAX，全 1，永远不要把显存还给系统。被用来分配 kernel 的工作 FIFO 缓冲区。comm 销毁时把整个池连同缓存的显存一次性还给系统。
    uint64_t releaseThreshold = ~uint64_t(0);
    CUDACHECK(cudaMemPoolSetAttribute(comm->memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
  } while (0);

  ncclIntruQueueConstruct(&comm->eventCallbackQueue);

  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;
  struct ncclKernelCommAndChannels tmpCommAndChans;
  struct ncclKernelCommAndChannels* devCommAndChans = NULL;
  struct ncclNvmlCCStatus ccStatus;
  bool ccEnable;
  cudaStream_t deviceStream;

  memset(&tmpCommAndChans, '\0', sizeof(tmpCommAndChans));
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                        /*concurrent=*/false, &deviceStream),
                ret, fail);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, deviceStream, comm->memManager), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks, deviceStream,
                                    comm->memManager),
                ret, fail);
  ncclCommPushCudaFree(comm, tmpCommAndChans.comm.rankToLocalRank);
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank, comm->nRanks,
                                    deviceStream),
                ret, fail);
  comm->devComm = &devCommAndChans->comm;
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.node = comm->node;
  tmpCommAndChans.comm.nNodes = comm->nNodes;
  tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;
  tmpCommAndChans.comm.isAllNvlink = comm->isAllNvlink;
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;
  tmpCommAndChans.comm.p2pCrossClique = comm->p2pCrossClique;
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  comm->workArgsBytes = std::min<size_t>(ncclParamWorkArgsBytes(), ncclMaxKernelArgsSize(comm->cudaArch));

  memset(&ccStatus, 0, sizeof(ccStatus));
  ccEnable = (ncclSuccess == ncclNvmlGetCCStatus(&ccStatus)) &&
             (ccStatus.CCEnabled || ccStatus.multiGpuProtectedPCIE || ccStatus.multiGpuNVLE);
  comm->ccEnable = ccEnable;
  if (ccEnable) {
    comm->workFifoBytes = 0;
  } else {
    comm->workFifoBytes = ncclParamWorkFifoBytes();
    if (0 != (comm->workFifoBytes & (comm->workFifoBytes - 1))) {
      WARN("NCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
      comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;
    }
    comm->workFifoBytes = std::min(comm->workFifoBytes, 1u << 30);
  }

  if (comm->rank == 0) {
    INFO(NCCL_INIT, "CC %s, workFifoBytes %d", ccEnable ? "On" : "Off", comm->workFifoBytes);
  }

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1 && comm->workFifoBytes > 0) {
    // The workFifoBuf lives in GDR mapped CUDA memory.
    NCCLCHECKGOTO(ncclGdrCudaCalloc(&comm->workFifoBuf, &comm->workFifoBufDev, comm->workFifoBytes,
                                    &comm->workFifoBufGdrHandle, comm->memManager),
                  ret, fail);
    ncclCommPushCudaGdrFree(comm, comm->workFifoBufGdrHandle);
  } else {
    // The workFifoBuf lives in cudaHost memory.
    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoBuf, comm->workFifoBytes), ret, fail);
    ncclCommPushCudaHostFree(comm, comm->workFifoBuf);
    comm->workFifoBufDev = comm->workFifoBuf;
  }

  comm->workFifoProduced = 0;
  comm->workFifoProducedLastRecorded = 0;
  comm->workFifoConsumed = 0;

  // Alloc profiler counters for the kernel
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workStarted, MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workCompleted, MAXCHANNELS), ret, fail);
  tmpCommAndChans.comm.workStarted = comm->profiler.workStarted;
  tmpCommAndChans.comm.workCompleted = comm->profiler.workCompleted;
  ncclCommPushCudaHostFree(comm, comm->profiler.workStarted);
  ncclCommPushCudaHostFree(comm, comm->profiler.workCompleted);

  if (comm->collNetDenseToUserRank != nullptr) {
    NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.collNetDenseToUserRank, nRanks, deviceStream,
                                      comm->memManager),
                  ret, fail);
    ncclCommPushCudaFree(comm, tmpCommAndChans.comm.collNetDenseToUserRank);
    NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.collNetDenseToUserRank, comm->collNetDenseToUserRank, nRanks,
                                      deviceStream),
                  ret, fail);
  }

  for (int c = 0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks,
                                        nRanks, deviceStream),
                    ret, fail);
    }
  }

  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, deviceStream), ret, fail);
exit:
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                    /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  return ret;
fail:
  goto exit;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define VERSION_STRING \
  "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX \
    "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
extern const char* ncclGetGitVersion(void);
static void showVersion() {
  if (ncclDebugLevel == NCCL_LOG_VERSION || ncclDebugLevel == NCCL_LOG_WARN) {
    // rank = 0 && level != None
    VERSION("%s", VERSION_STRING);
  } else {
    // level >= info
    INFO(NCCL_ALL, "%s", VERSION_STRING);
  }
  // NCCL git branch + commit
  INFO(NCCL_ALL, "%s", ncclGetGitVersion());
}

NCCL_PARAM(MNNVLUUID, "MNNVL_UUID", -1);
NCCL_PARAM(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);
NCCL_PARAM(MNNVLCrossClique, "MNNVL_CROSS_CLIQUE", 0);

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  // 直接拷贝填写基础信息
  cudaDeviceProp prop;
  info->rank = comm->rank;
  info->cudaDev = comm->cudaDev;
  info->nvmlDev = comm->nvmlDev;
  info->version = NCCL_VERSION_CODE;
  info->hostHash = getHostHash() + commHash;
  info->pidHash = getPidHash() + commHash;
  info->cuMemSupport = ncclCuMemEnable();
  CUDACHECK(cudaGetDeviceProperties(&prop, comm->cudaDev));
  // 全局显存按照 4G 向上取整
  info->totalGlobalMem = ROUNDUP(prop.totalGlobalMem, (1ULL << 32));
  /*
  MLOPart 是一种把一块物理 GPU 切成多个独立分区（partition）对外提供的技术，每个分区在 CUDA 看来就是一块独立的设备（代码里叫 uGPU，micro-GPU）。
  一块物理卡被硬件/固件隔离成几份，每份有自己的算力、显存和设备身份，多个负载能安全地共享同一块物理卡而互不干扰。
  "分区号"（mloPart / partition index）就是标识"我这个 rank 用的是这块物理卡上的第几个分区"，一个从 0 开始的小整数。

  设备名里若含 "MLOPart" 字样(如 ...MLOPart0)，取其后数字当分区号，否则 -1。
  */
  const char* mlopartStr = strstr(prop.name, "MLOPart");
  info->mloPart = mlopartStr ? atoi(mlopartStr + strlen("MLOPart")) : NCCL_TOPO_UNDEF;

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
#if defined(NCCL_OS_WINDOWS)
  // On Windows, shared memory uses file mapping objects, not /dev/shm
  // Set shmDev to 0 as it's not applicable on Windows
  info->shmDev = 0;
#elif defined(NCCL_OS_LINUX)
  // 声明一个 POSIX struct stat 缓冲，用来接收 stat() 返回的文件元数据
  struct stat statbuf;
  // 对路径 /dev/shm 调 stat(2) 系统调用，结果填进 statbuf。
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  // st_dev 是 dev_t 类型，编码了该文件所在文件系统的设备号(MAJOR:MINOR)。/dev/shm 通常是一个 tmpfs 挂载点，每个 tmpfs 实例都有自己唯一的 st_dev。
  info->shmDev = statbuf.st_dev;
#endif
  info->busId = comm->busId;
  CUCHECK(cuDeviceGetUuid((CUuuid*)&info->gpuUuid, (CUdevice)comm->cudaDev));  // 获取对应 GPU 设备的 uuid
  // 查询是否支持 GDR
  NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));
  info->comm = comm;
  info->cudaCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;

  /*
  要让 NCCL 使用 MNNVL，得先知道哪些 GPU 处在同一个 NVLink fabric 里，这正是这段探测代码的意义，问 NVML 要 
  clusterUuid（同域？）和 cliqueId（同子团？），后面 mnnvl.cc 才能拿这些字段跨 peer 比对，判定"我和你到底能不能走 MNNVL"。
  */
  // MNNVL support
  {
    // MNNVL: Request the fabric UUID and partition info
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];  // 16
    nvmlDevice_t nvmlDev;
    // 从 GPU busId 转字符串并从 nvml 获取设备句柄
    NCCLCHECK(int64ToBusId(info->busId, busId));
    NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
    // 预置默认状态 + 容错查询
    info->fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    (void)ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &info->fabricInfo);
    /*
    仅当"不是明确不支持"才继续，state 有 4 态：NOT_SUPPORTED(0) / NOT_STARTED(1) / IN_PROGRESS(2) / COMPLETED(3)。
    这里的判断是"只要不是 0，就进来"，即只要 fabric 概念存在（哪怕还在注册中），就执行下面的 UUID/cliqueId 加工与日志。
    真正"能不能用 MNNVL"的严格判定（必须 COMPLETED）留到消费侧 mnnvl.cc 再做，探测侧只负责"有信息就整理好"。
    */
    if (info->fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
      unsigned long uuid0 = 0;
      unsigned long uuid1 = 0;
      /*
      NCCL_MNNVL_UUID 默认 -1，即不干预，用 NVML 返回的真实 UUID。
      一旦用户设了别的值，就用这个值伪造 clusterUuid。这是测试/调试用的降级 UUID，
      因为所有设了相同 NCCL_MNNVL_UUID 的节点会被"人为归到同一个 cluster"。
      */
      if (ncclParamMNNVLUUID() != -1) {
        unsigned long temp_uuid0 = (unsigned long)ncclParamMNNVLUUID();
        unsigned long temp_uuid1 = (unsigned long)ncclParamMNNVLUUID();
        memcpy(info->fabricInfo.clusterUuid, &temp_uuid0, sizeof(temp_uuid0));
        memcpy(info->fabricInfo.clusterUuid + sizeof(temp_uuid0), &temp_uuid1, sizeof(temp_uuid1));
      }
      // 把 clusterUuid 前两个 unsigned long 拷进 uuid0/uuid1，纯粹是为了最后那条 INFO 日志能用 %lx.%lx 打印 UUID
      memcpy(&uuid0, info->fabricInfo.clusterUuid, sizeof(uuid0));
      memcpy(&uuid1, info->fabricInfo.clusterUuid + sizeof(uuid0), sizeof(uuid1));
      /*
      NCCL_MNNVL_CLIQUE_ID 默认 -1。这里有三种互斥取值，且判断顺序很关键（先判 -2）:
        -2 表示自动派生，查 nvmlPlatformInfo_t，拿机架序列号 chassisSerialNumber 做哈希当 cliqueId，把 NVL 域按机架切成一个个 clique；
        -1 表示不动，两个分支都不进，cliqueId 保持 NVML 返回的原值；
        其他值表示强制覆盖，直接把用户给的整数当 cliqueId。
      */
      if (ncclParamMNNVLCliqueId() == -2) {
        nvmlPlatformInfo_t platformInfo = {0};
        NCCLCHECK(ncclNvmlDeviceGetPlatformInfo(nvmlDev, &platformInfo));
        INFO(NCCL_INIT, "MNNVL rack serial %s slot %d tray %d hostId %d peerType %d moduleId %d",
             platformInfo.chassisSerialNumber, platformInfo.slotNumber, platformInfo.trayIndex, platformInfo.hostId,
             platformInfo.peerType, platformInfo.moduleId);
        /*
        在超大 NVLink 域里，一个 clusterUuid 可能横跨多个物理机架。用"同机架 = 同序列号 = 同哈希 = 同 cliqueId"，
        就能把 NVL 域按机架切成一个个 clique，让 NCCL 感知物理就近性（同机架的 GPU 归一组）。
        这是在 NVML 原生 cliqueId 不够用或需要按机架重分组时的一个派生策略。
        */
        // Use a hash of the Rack serial number to partition the NVLD clique
        info->fabricInfo.cliqueId = getHash(platformInfo.chassisSerialNumber, sizeof(platformInfo.chassisSerialNumber));
      } else if (ncclParamMNNVLCliqueId() != -1) {
        info->fabricInfo.cliqueId = ncclParamMNNVLCliqueId();
      }
      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x", info->busId,
           uuid0, uuid1, info->fabricInfo.cliqueId, info->fabricInfo.state, info->fabricInfo.healthMask);
    }
  }

  // 网络拓扑允不允许 NIC 之间跨 rail 互通，默认同 rail 通信
  NCCLCHECK(ncclTopoCheckCrossNicSupport(&info->crossNicSupport));
  // 查询 GPU VMM 申请的显存只不支持 GDR 注册 MR
  int cuMemGdrSupport;
  CUCHECK(cuDeviceGetAttribute(&cuMemGdrSupport, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                               comm->cudaDev));
  info->cuMemGdrSupport = (cuMemGdrSupport == 1);
  // GIN 支持的类型
  info->supportedGinType = comm->sharedRes->ginState.ginType;
  // RMA 插件是否可用
  info->rmaPluginAvailable = (comm->rmaState.rmaProxyState.ncclRma != nullptr);

  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Find our ring-distance from rank zero and reorganize ranks to start with rank.
  int ixZero = 0, ixRank = 0;
  for (int i = 0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank - ixZero + nranks) % nranks;
  for (int i = 0; i < nranks; i++) {
    ring->userRanks[i] = ringRanks[(i + ixRank) % nranks];
    ring->rankToIndex[ring->userRanks[i]] = i;
  }
  return ncclSuccess;
}

#define DEFAULT_LL_BUFFSIZE \
  (NCCL_LL_LINES_PER_THREAD * NCCL_LL_MAX_NTHREADS * NCCL_STEPS * sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD * NCCL_LL128_MAX_NTHREADS * NCCL_STEPS * sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB */

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int64_t envs[NCCL_NUM_PROTOCOLS] = {ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize()};
  int defaults[NCCL_NUM_PROTOCOLS] = {DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE};

  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  if (comm->nNodes > 1) {
    // GBx platforms only have 4 GPUs per host, which reduces the aggregation factor.
    // Increase the ratio by a factor of 2 to compensate.
    int ratio = (comm->cpuArch == NCCL_TOPO_CPU_ARCH_ARM && comm->minCompCap >= 100) ? 2 : 1;
    comm->p2pChunkSize = ratio * ncclParamP2pNetChunkSize();
  } else if (comm->isAllNvlink) {
    comm->p2pChunkSize = ncclParamP2pNvlChunkSize();
  } else {
    comm->p2pChunkSize = ncclParamP2pPciChunkSize();
  }

  // Make sure P2P chunksize is not larger than coll chunksize.
  if (comm->p2pChunkSize * NCCL_STEPS > comm->buffSizes[NCCL_PROTO_SIMPLE])
    comm->p2pChunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;

  if (comm->sharedRes->owner != comm) {
    /* make sure split comm p2pChunkSize won't exceed shared p2pChunkSize. */
    comm->p2pChunkSize = std::min(comm->p2pChunkSize, comm->sharedRes->tpP2pChunkSize);
  } else {
    comm->sharedRes->tpP2pChunkSize = comm->p2pChunkSize;
  }

  INFO(NCCL_INIT, "P2P Chunksize set to %d", comm->p2pChunkSize);
  return ncclSuccess;
}

// 指定哪个 rank 把拓扑图 dump 到文件
NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
// 节点数低于此阈值就禁用 CollNet
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
// 初始化时是否预建 NVB 路径的 p2p 连接
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 1);
// 是否为网络 p2p 连接额外分配 LL 协议缓冲
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

/*
值	      含义	      ncclMnnvlCheck何时被调
0	        禁用	      两条件都假 → 从不探测;也是"MNNVL 能用但坏了"时的逃生开关
1	       强制启用	     \|\| ...==1 分支 → 总是探测(无视 nNodes、无视 p2pLevel,单节点也探)
2(默认)	  自动	       仅当 nNodes>1(多节点)且 p2pLevel!=0(没禁 P2P)时才探测
*/
// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM(MNNVLEnable, "MNNVL_ENABLE", 2);

#define TIMER_INIT_TOTAL 0
#define TIMER_INIT_KERNELS 1
#define TIMER_INIT_BOOTSTRAP 2
#define TIMER_INIT_ALLGATHER 3
#define TIMER_INIT_TOPO 4
#define TIMER_INIT_GRAPHS 5
#define TIMER_INIT_CONNECT 6
#define TIMER_INIT_ALLOC 7
#define TIMERS_INIT_COUNT 8

extern int64_t ncclParamWinStride();

static ncclResult_t initNvlDomainInfo(struct ncclComm* comm) {
  // Initialize NVLink domain info
  comm->nvlDomainInfo.nNvlDomains = comm->nNodes;
  comm->nvlDomainInfo.minRanksPerNvlDomain = comm->minLocalRanks;
  comm->nvlDomainInfo.maxRanksPerNvlDomain = comm->maxLocalRanks;

  TRACE(NCCL_INIT, "NVLink domains: %d domains, min ranks per domain: %d, max ranks per domain: %d", comm->nNodes,
        comm->nvlDomainInfo.minRanksPerNvlDomain, comm->nvlDomainInfo.maxRanksPerNvlDomain);

  return ncclSuccess;
}

NCCL_PARAM(GroupSize, "P2P_SCHEDULE_GROUP_SIZE", NCCL_MAX_DEV_WORK_P2P_PER_BATCH);

static ncclResult_t ncclP2pSchedule(struct ncclComm* comm) {
  struct ncclNodeRanks* nodeRanks = comm->nodeRanks;
  // For MNNVL systems, we need to split the nodes into different groups to guarantee the proper PXN
  // aggregation factor.
  int groupSize = (comm->nNodes > 1) ? ncclParamGroupSize() : comm->maxLocalRanks;
  for (int node = 0; node < comm->nNodes; node++) {
    int localRanks = nodeRanks[node].localRanks;
    if (localRanks % groupSize != 0 || localRanks < groupSize) groupSize = gcd(groupSize, nodeRanks[node].localRanks);
  }
  comm->p2pSchedGroupSize = groupSize;

  int local = comm->localRank % groupSize; // local id inside my group
  int group = comm->localRank / groupSize; // id of my group, incremented when going over the previous nodes
  int nGroups = comm->nRanks / groupSize;
  int nGroupsPow2 = pow2Up(nGroups);

  int *groupToNode, *groupToLocal;
  NCCLCHECK(ncclCalloc(&groupToNode, nGroups));  // node hosting the group
  NCCLCHECK(ncclCalloc(&groupToLocal, nGroups)); // local offset of the group
  int groupCount = 0;
  for (int n = 0; n < comm->nNodes; ++n) {
    if (0 != comm->nodeRanks[n].localRanks % groupSize) {
      WARN("nLocals = %d should be a diviser of the number of ranks in node %d = %d", groupSize, n,
           comm->nodeRanks[n].localRanks);
      return ncclInternalError;
    }
    int nGroupsInNode = comm->nodeRanks[n].localRanks / groupSize;
    for (int g = 0; g < nGroupsInNode; ++g) {
      groupToLocal[groupCount] = g * groupSize;
      groupToNode[groupCount] = n;
      groupCount++;
    }
    if (n < comm->node) group += nGroupsInNode;
  }
  if (groupCount != nGroups) {
    WARN("Group creation failed: %d vs %d", groupCount, nGroups);
    return ncclInternalError;
  }
  INFO(NCCL_GRAPH, "%s: group size used is %d", __func__, groupSize);

  uint32_t groupRound = 0, groupDelta = 0;
  int round = 0;
  // When enumerating peer deltas we use the quadratic formula (x*x+x)/2 mod N.
  // Since that formula only produces valid permutations when N is a pow of 2,
  // we let N = pow2Up(n) and filter out results greater-eq to n.
  // Example sequence for 16 ranks: 0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
  do {
    if (groupDelta < nGroups) {
      // Filter nonsensical group deltas
      int sendGroup = (group + groupDelta) % nGroups;
      int recvGroup = (group - groupDelta + nGroups) % nGroups;
      int sendNode = groupToNode[sendGroup];
      int recvNode = groupToNode[recvGroup];
      for (int delta = 0; delta < groupSize; delta++) {
        int sendLocal = groupToLocal[sendGroup] + (local + delta) % groupSize;
        int recvLocal = groupToLocal[recvGroup] + (local - delta + groupSize) % groupSize;
        comm->p2pSchedule[round].sendRank = nodeRanks[sendNode].localRankToRank[sendLocal];
        comm->p2pSchedule[round].recvRank = nodeRanks[recvNode].localRankToRank[recvLocal];
        round += 1;
      }
    }
    groupRound += 1;
    groupDelta = (groupDelta + groupRound) & (nGroupsPow2 - 1); // Quadratic update
  } while (groupRound != nGroupsPow2);

  free(groupToNode);
  free(groupToLocal);

  if (round != comm->nRanks) {
    WARN("P2p schedule creation has bugs.");
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* parent,
                                       uint64_t timers[TIMERS_INIT_COUNT]) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;  // 本 rank
  int nranks = comm->nRanks;  // 总 rank 数
  int nNodes = 1;  // 节点数，先置 1，后续 allgather topoRanks 后重算真实值
  // 保存进入时的 CPU 亲和性(拓扑探测期可能临时改绑 NUMA，事后恢复)
  ncclAffinity affinitySave = {};
  // 取出 5 个不同的拓扑图指针(都指向 comm->graphs[] 这个存储数组的对应槽位)
  struct ncclTopoGraph* ringGraph = &comm->graphs[NCCL_ALGO_RING];
  struct ncclTopoGraph* treeGraph = &comm->graphs[NCCL_ALGO_TREE];
  struct ncclTopoGraph* collNetChainGraph = &comm->graphs[NCCL_ALGO_COLLNET_CHAIN];
  struct ncclTopoGraph* collNetDirectGraph = &comm->graphs[NCCL_ALGO_COLLNET_DIRECT];
  struct ncclTopoGraph* nvlsGraph = &comm->graphs[NCCL_ALGO_NVLS];
  /*
  建分派数组(按算法 id 索引)，需要注意的是 7 个算法对应 5 个图：
  算法 id	   宏	                         graphs[id] 指向	      是否独立图
  0         NCCL_ALGO_TREE	            treeGraph	            ✅ 独立
  1         NCCL_ALGO_RING	            ringGraph	            ✅ 独立
  2	        NCCL_ALGO_COLLNET_DIRECT	  collNetDirectGraph	  ✅ 独立
  3	        NCCL_ALGO_COLLNET_CHAIN	    collNetChainGraph	    ✅ 独立
  4	        NCCL_ALGO_NVLS	            nvlsGraph	            ✅ 独立
  5	        NCCL_ALGO_NVLS_TREE	        nvlsGraph(复用 4)	    ❌ 共享 NVLS
  6	        NCCL_ALGO_PAT	              treeGraph(复用 0)	    ❌ 共享 Tree
  有重复是因为拓扑搜索是按"需要几种不同的物理拓扑形态"来做的，不是按"有几个算法":
    NVLS_TREE 复用 nvlsGraph，NVLS_TREE = 节点内 NVLS 多播 + 节点间树。它的节点内多播拓扑与纯 NVLS 完全一样，
  差别只在跨节点部分，所以不必再独立搜一次 NVLS 图。
    PAT 复用 treeGraph，PAT(Parallel Aggregated Trees，并行聚合树，主打小消息)属于树家族，其拓扑搜索复用已搜好的树图。
  也就是说 comm->graphs[] 有 7 个存储槽位，但真正被独立搜索的只有 5 个图对象(tree/ring/collNetDirect/collNetChain/nvls)。
  graphs[] 这个分派数组的作用就是让"按算法 id 取图"时，NVLS_TREE/PAT 自动落到它们复用的那个已搜好的图上。
  这个数组随后被传给 ncclTopoPreset(comm, graphs, topoRanks) 等做 channel 预置。
  */
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = {treeGraph, ringGraph, collNetDirectGraph, collNetChainGraph,
                                                       nvlsGraph, nvlsGraph, treeGraph};
  
  /*
  ncclTopoGraph 的 8 个标量输出字段的子集，刻意丢掉了 intra[]/inter[] 这两个大数组
  (那是节点内私有的连接蓝图，跨 rank 对齐用不上，且太大不适合 allgather)。
  */
  struct graphInfo {
    int pattern;  // 拓扑形态(NCCL_TOPO_PATTERN_*)
    int nChannels;  // // 该算法搜出的 channel 数
    int sameChannels;  // 各 channel 是否同构
    float bwIntra;  // 节点内带宽
    float bwInter;  // 节点间带宽
    int typeIntra;  // 节点内链路类型(PATH_*)
    int typeInter;  // 节点间链路类型(PATH_*)
    int crossNic;  // 跨 NIC 策略
  };

  /*
  用于 AllGather 的完整 payload，allgather 之后，代码用两种不同的归约策略把"各 rank 各异"压成"全局一致"
    1. 能力位 → AND 交集(在 AllGather1 的 peerInfo 循环里)，某特性只有当所有 rank 都支持时才全局开启(交集)，isAllNvlink 同理(AND)。
    2. 计数/带宽 → min 取小 + max 检不匹配(在 AllGather3 后的循环里)。
      取 min：用"最小公分母"保守对齐，后续再跨 rank 取 min，确保所有 rank 用同样多的 channel。
      min≠max 即不匹配：rank 0 检查 if (minLocalNetCount != maxLocalNetCount) 直接报错，
    除非设 NCCL_IGNORE_NET_MISMATCH=1(CollNet 对应 NCCL_IGNORE_COLLNET_MISMATCH)，这是防止"各节点网卡数不一致导致 rail 错配"的硬校验。
  */
  struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];  // 7 个算法的拓扑摘要，归约时按算法取 min/max 对齐。
    // 本 rank 的连接序列蓝图，由 ncclTopoPreset 填，后续 ncclTopoPostset 用它把各节点的局部环/树跨节点拼成完整环/树。
    struct ncclTopoRanks topoRanks;
    /*
    用于探测混合 CPU，归约时若任一 rank 不同就置 NCCL_TOPO_CPU_ARCH_MIXED/VENDOR_MIXED，
    rank 0 打 INFO("CPUs with mixed architecture were detected")(历史上混合 CPU 会导致某些集合例程卡死，故预警)。
    */
    int cpuArch;
    int cpuVendor;
    int localRanks;  // 本节点 rank 数
    // p2p(send/recv)通信的 channel 与 peer 上限配置，需全员对齐。
    int p2pnChannelsPerPeer;
    int p2pMaxPeers;
    // 本地网卡最小带宽，归约后 供 tuner 估跨节点成本。
    float minNetBw;
    // 本地普通网卡 / CollNet 网卡的数量与带宽，这三者驱动下面的 min/max 归约与不匹配检测。
    int localNetDeviceCount;
    int localNetDeviceBw;
    int localCollNetCount;
    // 本 rank 是否所有路径都是 NVLink。归约时只要有一个 rank 不全 NVLink，全局就判定非全 NVLink(影响 NVLS 等特性)。
    int isAllNvlink;
  };

  /*
  保存 comm->nChannels 被跨 rank 取 min 之前的原值，若归约后 comm->nChannels < nChannelsOrig，
  要把 Preset() 期间复制到高位的 channel memcpy 搬回(comm->channels + nChannels + i ← nChannelsOrig + i)
  */
  int nChannelsOrig;
  // allGatherInfo 数组(nranks 份)，AllGather3 的收发缓冲
  struct allGatherInfo* allGather3Data = NULL;
  // 指向各 rank topoRanks 的指针数组，供 ncclTopoPostset 跨节点拼环/树
  struct ncclTopoRanks** allTopoRanks = NULL;
  // 按 node 索引：每节点的首 rank(ringRecv[0])、每节点的 tree pattern(可因 SM 架构不同而异)
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL;
  // ncclTopoPostset 输出的最终环顺序
  int* rings = NULL;
  // NVB(NVLink-Bridge)GPU peer 列表，NCCL_NVB_PRECONNECT(默认 1)开启时，对走 NVB 路径的 peer 预建 p2p 连接
  int* nvbPeers = NULL;
  // 与 proxy 服务线程通信的句柄，用于本地 net proxy 与 PXN 远端 proxy 的 ncclProxyMsgSharedInit
  struct ncclProxyConnector proxyConn;
  // PXN(PCIe×NVLink)中转 rank 列表，PXN 未禁用时对这些远端 rank 的 net proxy 预连接
  int* pxnPeers = NULL;
  // 把本地 rank 映射到顶层父 comm 的 local-rank 编号，供 split/grow 子 comm 继承父 comm 的本地编号，最终存入 comm->topParentLocalRanks
  int* topParentLocalRanks = NULL;
  int p2pLevel = -1;  // 用户 P2P 层级(ncclGetUserP2pLevel)，作为 MNNVL 检查的门控条件之一
  bool globalGinSupport = comm->sharedRes->ginState.ginType != NCCL_GIN_TYPE_NONE;  // 全员是否支持同一种 GIN 类型
  // 下面几个全局量是 AND 归约的标准起点，先假设"全都能"，遍历中一旦遇到不支持的就全局不支持
  bool globalCrossNicSupport = true;  // 全员是否都支持跨 NIC
  bool globalRmaPluginSupport = true;  // 全员是否都有 RMA 插件
  bool globalCuMemGdrSupport = true;  // 全员是否都支持 cuMem GDR
  bool isOneLsaTeams = false;  // 是否构成单一 LSA(Load/Store Accessible)

  int localNetDeviceCount = 0;  // 本 rank 的普通网卡数 
  int localNetDeviceBw = 0;  // 本 rank 的网卡带宽
  int localCollNetCount = 0;  // 本 rank 的 CollNet 网卡数
  // 跨 rank 网卡数的 min/max 累加器(min≠max 触发不匹配检测)
  int minLocalNetCount = INT_MAX;
  int maxLocalNetCount = 0;
  // 跨 rank CollNet 网卡数的 min/max 累加器(min≠max 触发不匹配检测)
  int minLocalCollNetCount = INT_MAX;
  int maxLocalCollNetCount = 0;

  timers[TIMER_INIT_ALLGATHER] = clockNano();
  // AllGather1 - begin
  /*
  分配 nranks + 1 个 ncclPeerInfo 并清零，额外一个 rank 代表 CollNet root
  CollNet（网内集合卸载，如 SHARP）在拓扑上需要一个"root"角色代表交换机侧的汇聚点，这个虚拟 root 不占真实 rank 号，就借用第 nranks 号槽位表示。
  */
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks + 1), ret, fail); // Extra rank to represent CollNet root
  // 填当前 rank 自己那一格信息
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo + rank, comm->commHash), ret, fail);
  // 所有 rank  AllGather 获取全局所有 peerInfo 信息
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);
  // release 语义将 comm->peerInfoValid 设为 true
  COMPILER_ATOMIC_STORE(&comm->peerInfoValid, true, std::memory_order_release);

  comm->cuMemSupport = 1;  // 循环前乐观初始化
  for (int i = 0; i < nranks; i++) {
    // NCCL 版本一致性校验
    if (comm->peerInfo[i].version != comm->peerInfo[rank].version) {
      WARN("Mismatched NCCL version detected : rank %d version %d rank %d version %d", i, comm->peerInfo[i].version,
           rank, comm->peerInfo[rank].version);
      ret = ncclInvalidUsage;
      goto fail;
    }
    /*
    不是精确节点数，比如 8 本地 + 8 远程的作业，从 rank 0 看 nNodes = 1 + 8 = 9，显然不等于 2。
    它在这里唯一的用途是紧接着 MNNVL 检查的 nNodes > 1 那个布尔门（"是不是多节点"）。
    真正的 comm->nNodes 在更后面用 topoRanks 的 nodesFirstRank 去重算出来。所以别把这个 nNodes 当节点数用。
    */
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) nNodes++;
    // 任一不支持 VMM，全局不支持
    if (!comm->peerInfo[i].cuMemSupport) comm->cuMemSupport = 0;
    // 遍历所有 rank 的 peerInfo，只要有任何一个 rank 用了 MLOPart 分区，就把全局标志 comm->hasMloPart 置 true。
    if (comm->peerInfo[i].mloPart != -1) comm->hasMloPart = true;
    // 两两配对检测，j < i 保证每对 rank 只比一次（不重复、不自比）
    for (int j = 0; j < i; j++) {
      /*
      nvmlDev（NVML 索引）不区分 MLOPart 分区：一块物理卡切成多个分区时，这些分区在 NVML 眼里可能报同一个设备索引。
      gpuUuid（CUDA UUID）区分分区：每个分区有各自不同的 UUID。
      */
      // NVML device is agnostic to MloPart being used. With MloPart, each partition has a different GPU UUID.
      // hasMultiRankNvml 表示多 rank 落在同一 NVML 设备（可能是同卡不同分区，也可能真同卡），不禁用，只标记，下游 nvls.cc 据此关掉 NVLS。
      comm->hasMultiRankNvml = (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash) &&
                               (comm->peerInfo[i].nvmlDev == comm->peerInfo[j].nvmlDev);
      // 如果没启用 MultiRankGpu，但是检测到有两个 rank 共用同一块 GPU 或同一个 MLOPart 分区，报错。
      if (!ncclParamMultiRankGpuEnable() && (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash) &&
          memcmp(&comm->peerInfo[i].gpuUuid, &comm->peerInfo[j].gpuUuid, sizeof(cudaUUID_t)) == 0) {
        WARN("Multiple Ranks are using the same GPU/Partition. Set NCCL_MULTI_RANK_GPU_ENABLE=1 to enable this "
             "configuration.");
        return ncclInvalidUsage;
      }
    }
    // AND 归约集合特性只有全员支持才能启用，一个掉队全体降级
    globalGinSupport &= (comm->peerInfo[i].supportedGinType == comm->sharedRes->ginState.ginType);  // 全员同一种 GIN 类型
    globalCrossNicSupport &= comm->peerInfo[i].crossNicSupport;  // 全员都允许跨 NIC
    globalRmaPluginSupport &= comm->peerInfo[i].rmaPluginAvailable;  // 全员都有 RMA 插件
    globalCuMemGdrSupport &= comm->peerInfo[i].cuMemGdrSupport;  // 全员都支持 cuMem GDR
  }
  // AllGather1 - end
  timers[TIMER_INIT_ALLGATHER] = clockNano() - timers[TIMER_INIT_ALLGATHER];

  // Check for MNNVL support
  // 获取 p2p level
  NCCLCHECKGOTO(ncclGetUserP2pLevel(&p2pLevel), ret, fail);
  /*
  MNNVL (Multi-Node NVLink) 是多节点 NVLink P2P，需要满足以下条件才会尝试探测：
    1. NCCL_MNNVL_ENABLE=1，强制探测；
    2. NCCL_MNNVL_ENABLE>1（自动模式），需要满足跨节点和启用 P2P 这两个条件。
  */
  if ((nNodes > 1 && ncclParamMNNVLEnable() != 0 && p2pLevel != 0) || ncclParamMNNVLEnable() == 1) {
    // 探测是否满足所有启用 MNNVL 的条件，最终设置 comm 全局开关 comm->MNNVL
    NCCLCHECKGOTO(ncclMnnvlCheck(comm), ret, fail);
  }

  /*
  在同一个进程里跑了多个 rank 时（单进程多卡，如 ncclCommInitAll，或一个进程持多张卡），
  把这些"同进程兄弟"识别出来、编好号、串成链、并建立一个进程内的轻量同步锚点。

  fillInfo 里 info->comm = comm 是每个 rank 把自己 comm 对象的裸指针写进了 peerInfo。
  随后 bootstrapAllGather 把全员 peerInfo（连同这 8 字节指针值）广播给所有人。
  于是当前 rank 手里的 comm->peerInfo[i].comm 是"rank i 当初填的、指向 rank i 自己 comm 对象的指针值"。
  这个指针在当前 rank 的地址空间里只有当 rank i 与当前 rank 同进程时才有效。
    同进程（hostHash 和 pidHash 都相同）：共享同一地址空间，peerInfo[i].comm 是可以直接解引用的活指针，能摸到兄弟 rank 的 comm 对象。
    不同进程：那是别人地址空间的值，在本地是野指针，绝不能解引用。
  */
  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;

    comm->nvlsRegSupport = 1;
    for (int i = 0; i < nranks; i++) {
      // 顺带完成全局计算能力规约
      comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[i].cudaCompCap);
      comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[i].cudaCompCap);
      // 确保 rank i 与当前 rank 同节点同进程
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) &&
          (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        // 第一个遇到的同进程 rank = 进程内"组长"的全局 rank 号
        if (intraProcRanks == 0) intraProcRank0 = i;
        // // 遇到自己时，记下"我在进程内排第几"
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;  // 同进程 rank 计数
        // intraNext 头插链表，只有组长建，且直接写进兄弟的内存
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;  // 兄弟的 next = 我当前的 next
          comm->intraNext = comm->peerInfo[i].comm;  // 我的 next = 兄弟（头插）
        }
      }

      if (comm->nvlsRegSupport) {
        /*
        内层 j>i 双循环检测的是全局是否存在任意一对 rank 同进程，一旦发现，立刻置 0 并 break。
        外层 if (comm->nvlsRegSupport) 守卫保证一旦置 0 就不再进内层扫描（提前终止，避免 O(n²) 白跑）。

        上面只关心当前 rank 是否别人同进程，而下面的循环检测关心"任何地方有没有两个 rank 同进程"，是个全局属性。
        所有 ranks 会得出一致的结论（要么全 0 要么全 1），这对后续需要全员一致的 NVLS 决策至关重要。
        */
        for (int j = i + 1; j < nranks; j++) {
          if (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash &&
              comm->peerInfo[i].pidHash == comm->peerInfo[j].pidHash) {
            comm->nvlsRegSupport = 0;
            break;
          }
        }
      }
    }

    /*
    条件	                                结果	            含义
    comm->MNNVL	                          强制 0	          MNNVL（跨节点 NVLink）下 buffer registration 不支持，一票否决
    NCCL_SINGLE_PROC_MEM_REG_ENABLE 非 0	强制 1	          用户显式断言"单进程多 rank 的内存注册没问题"，覆盖前面的检测
    否则	                                保留双循环检测结果	 有任一对同进程 → 0；否则 → 1

    为什么"同进程多 rank"要禁 NVLS buffer registration：
    NVLS（NVLink SHARP multicast）的用户 buffer 注册依赖"一个进程一个 rank"的假设来交换/去重注册句柄。
    一个进程里挤了多个 rank 时，注册路径无法区分彼此的 buffer 归属，于是保守地全 comm 关掉这个特性。
    */
    // Buffer Registration is not supported with MNNVL
    if (comm->MNNVL) comm->nvlsRegSupport = 0;  // MNNVL 下不支持 buffer registration
    else if (ncclParamSingleProcMemRegEnable()) comm->nvlsRegSupport = 1;  // 用户指定 NCCL_SINGLE_PROC_MEM_REG_ENABLE 强制打开，默认 0 关闭

    TRACE(NCCL_INIT, "pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d", rank,
          comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d "
           "intraProcRank0 %d",
           rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks,
           intraProcRank0);
      ret = ncclInternalError;
      goto fail;
    }
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    // 自检当前 rank 是组长，则组长 comm 要等于当前 rank 的 comm
    assert(intraProcRank == 0 ? comm == comm0 : true);
    comm->intraComm0 = comm0;  // 指向组长，即 comm->intraNext 表头
    comm->intraRank = intraProcRank;  // 当前 rank 在进程中的序号
    comm->intraRanks = intraProcRanks;  // 进程内 rank 总数
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while (0);  // 子作用域块

  timers[TIMER_INIT_TOPO] = clockNano();

  /*
  基于 NCCL_TOPO_FILE 指定的 topo xml 文件（如果有作为基线模板）探测本 rank GPU + 本机拓扑 XML，
  然后在 LSA 本地组范围内 AllGather 合并取并集，仅指定 rank 把融合结果 dump 到文件，仅非 dump 模式将 topo XML 转换为 topo 图。
  */
  // Dump XML if requested by user
  const char* dumpXmlFile;
  dumpXmlFile = ncclGetEnv("NCCL_TOPO_DUMP_FILE");
  if (dumpXmlFile) {
    NCCLCHECKGOTO(ncclTopoGetSystem(comm, NULL, dumpXmlFile), ret, fail);  // 仅 dump 本地组 topo XML
  }

  // Topo detection / System graph creation
  NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), ret, fail);  // topo XML 探测并转换为 topo 图保存到 comm->topo
  // Compute paths between GPUs and NICs
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);  // 计算所有顶点两两之间的 paths
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), ret, fail);  // 删除与当前 rank 的 GPU 不连通的 GPU
  // Recompute paths after trimming
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);  // 上一步删除不连通 GPU 会导致出现悬挂 link 指针，拓扑产生变化，需要重计算 paths
  // Init search
  NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);  // 计算每个 DEV 顶点到 NET/DEV 路径带宽上限，以及 NVLink 和 PCIe 外连带宽上限
  // Decide on comm's CPU architecture.
  NCCLCHECKGOTO(ncclTopoComputeCommCPU(comm), ret, fail);  // 从拓扑图中选一个 CPU 顶点获取 CPU 架构和厂商记录在 comm 中
  // Print final topology
  NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);  // 在日志打印拓扑图
  timers[TIMER_INIT_TOPO] = clockNano() - timers[TIMER_INIT_TOPO];

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, fail);  // 获取 rank 对应的 GPU 亲和的 CPU NUMA 在绑核限制后的 CPU 核位图
  if (ncclOsCpuCount(comm->cpuAffinity)) {  // CPU 核数不为 0
    NCCLCHECKGOTO(ncclOsGetAffinity(&affinitySave), ret, fail);  // 读出当前线程的绑定的 CPU 位图
    NCCLCHECKGOTO(ncclOsSetAffinity(comm->cpuAffinity), ret, fail);  // 设置新算出的 CPU 位图
  }

  // Determine local CollNet support
  if (!collNetSupport(comm)) {  // 根据是否有collNet 插件判断是否支持 collNet
    comm->config.collnetEnable = 0;
  }

  NCCLCHECK(ncclNvlsInit(comm));  // 初步判断是否支持 NVLS

  timers[TIMER_INIT_GRAPHS] = clockNano();
  // Get rings and trees
  memset(ringGraph, 0, sizeof(struct ncclTopoGraph));
  ringGraph->id = 0;  // 用于 bootstrap 的 tag
  ringGraph->pattern = NCCL_TOPO_PATTERN_RING;  // 4
  // ring 是唯一「有自主通道数选择权」的图，tree 和 collNetChain 的 min 与 max 都被锁死成 ringGraph->nChannels，被动对齐。
  ringGraph->minChannels = 1;
  ringGraph->maxChannels = MAXCHANNELS / 2;  // 因为 ncclTopoPreset 会无条件复制一份 channel
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);
  /*
  拓扑图输出到日志，示例格式如下：
    Search 0 : 8 channels
    Pattern 4, crossNic 0, nChannels 8, bw 370.000000/0.000000, type NVL/NET, sameChannels 1
    0 : GPU/0-0 GPU/0-1 GPU/0-2 GPU/0-3 GPU/0-4 GPU/0-5 GPU/0-6 GPU/0-7
    1 : GPU/0-1 GPU/0-2 GPU/0-3 GPU/0-4 GPU/0-5 GPU/0-6 GPU/0-7 GPU/0-0
    2 : GPU/0-2 GPU/0-3 ...
  */
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, ringGraph), ret, fail);

  memset(treeGraph, 0, sizeof(struct ncclTopoGraph));
  treeGraph->id = 1;
  treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;  // tree 默认是 BALANCED_TREE，入网卡连 step0，出网卡分别连 step 0 和 step 1
  // Tree 的 channel 数必须和 Ring 对齐
  treeGraph->minChannels = ringGraph->nChannels;
  treeGraph->maxChannels = ringGraph->nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, treeGraph), ret, fail);

  memset(collNetChainGraph, 0, sizeof(struct ncclTopoGraph));
  collNetChainGraph->id = 2;
  collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE;  // collNetChain 等于 TREE 的节点内链，出入网卡连同 step 0
  collNetChainGraph->collNet = 1;
  // Tree 的 channel 数必须和 Ring 对齐
  collNetChainGraph->minChannels = ringGraph->nChannels;
  collNetChainGraph->maxChannels = ringGraph->nChannels;

  memset(collNetDirectGraph, 0, sizeof(struct ncclTopoGraph));
  collNetDirectGraph->id = 4;
  collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT;  // COLLNET_DIRECT，n-1 个 gpu 连到 head gpu，出入网卡连 step 0
  collNetDirectGraph->collNet = 1;
  collNetDirectGraph->minChannels = 1;
  collNetDirectGraph->maxChannels = MAXCHANNELS;
  if (comm->config.collnetEnable) {  // collNet 满足有插件才搜索 collNetDirectGraph
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetDirectGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetDirectGraph), ret, fail);
  }

  memset(nvlsGraph, 0, sizeof(struct ncclTopoGraph));
  nvlsGraph->id = 3;
  nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS;  // NVLS 节点内所有 gpu 连 NVS，出入网卡连 step 0
  nvlsGraph->minChannels = 1;
  nvlsGraph->maxChannels = MAXCHANNELS;
  if (comm->nvlsSupport) {  // NVLS 满足启用条件才搜索 nvlsGraph
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, nvlsGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, nvlsGraph), ret, fail);
  }
  timers[TIMER_INIT_GRAPHS] = clockNano() - timers[TIMER_INIT_GRAPHS];

  // NCCL_ALLOC_P2P_NET_LL_BUFFERS=0，不为网络 p2p 连接额外分配 LL 协议缓冲
  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;
 
  if (comm->rank == ncclParamGraphDumpFileRank()) {  // 当前 rank dump graph 文件
    struct ncclTopoGraph* dumpGraphs[5] = {ringGraph, treeGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph};
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 5, dumpGraphs), ret, fail);
  }

  // Cap maxP2pPeers to nRanks
  if (comm->config.maxP2pPeers != NCCL_CONFIG_UNDEF_INT && comm->config.maxP2pPeers > comm->nRanks) {
    INFO(NCCL_INIT, "Max P2P Peers %d is too high, capping to communicator size %d", comm->config.maxP2pPeers,
         comm->nRanks);
    comm->config.maxP2pPeers = comm->nRanks;
  }
  // Compute nChannels per peer for p2p
  NCCLCHECKGOTO(ncclTopoComputeP2pChannelsPerPeer(comm), ret, fail);

  // Because timers[[TIMER_INIT_ALLGATHER] already contains the timing of the first allgather,
  // we temporarily store the start time of the subsequent one in an as-of-yet unused CONNECT timer.
  timers[TIMER_INIT_CONNECT] = clockNano();
  // AllGather3 - begin
  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);

  if (comm->ncclNet && comm->ncclNet->devices) {
    int gpu;
    float bw;
    NCCLCHECKGOTO(comm->ncclNet->devices(&localNetDeviceCount), ret, fail);
    NCCLCHECKGOTO(ncclTopoRankToIndex(comm->topo, comm->rank, &gpu, false), ret, fail);
    NCCLCHECKGOTO(ncclTopoGetLocalNetCountByBw(comm->topo, gpu, &localNetDeviceBw, &bw), ret, fail);
  }
  if (collNetSupport(comm)) {
    NCCLCHECKGOTO(collNetDevices(comm, &localCollNetCount), ret, fail);
  }
  INFO(NCCL_INIT, "Rank %d: %d Net devices", rank, localNetDeviceCount);
  INFO(NCCL_INIT, "Rank %d: %d CollNet devices", rank, localCollNetCount);

  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
    allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
    allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
    allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic;
  }

  allGather3Data[rank].cpuArch = comm->cpuArch;
  allGather3Data[rank].cpuVendor = comm->cpuVendor;
  allGather3Data[rank].p2pnChannelsPerPeer = comm->p2pnChannelsPerPeer;
  allGather3Data[rank].p2pMaxPeers = comm->p2pMaxPeers;

  allGather3Data[rank].localNetDeviceCount = localNetDeviceCount;
  allGather3Data[rank].localNetDeviceBw = localNetDeviceBw;
  allGather3Data[rank].localCollNetCount = localCollNetCount;
  NCCLCHECKGOTO(ncclTopoGetMinNetBw(comm->topo, comm->rank, &allGather3Data[rank].minNetBw), ret, fail);

  NCCLCHECK(ncclTopoPathAllNVLink(comm->topo, &comm->isAllNvlink));
  allGather3Data[rank].isAllNvlink = comm->isAllNvlink;

  comm->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  NCCLCHECKGOTO(ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks), ret, fail);

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);

  // Determine nNodes, firstRanks, ...
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks), ret, fail);
  comm->minNetCount = INT_MAX;

  for (int r = 0; r < nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    for (node = 0; node < comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
    }
    comm->rankToNode[r] = node;

    if (comm->cpuArch != allGather3Data[r].cpuArch && comm->cpuArch != NCCL_TOPO_CPU_ARCH_MIXED) {
      comm->cpuArch = NCCL_TOPO_CPU_ARCH_MIXED;
    }
    if (comm->cpuVendor != allGather3Data[r].cpuVendor && comm->cpuVendor != NCCL_TOPO_CPU_VENDOR_MIXED) {
      comm->cpuVendor = NCCL_TOPO_CPU_VENDOR_MIXED;
    }
    minLocalNetCount = std::min(minLocalNetCount, allGather3Data[r].localNetDeviceCount);
    maxLocalNetCount = std::max(maxLocalNetCount, allGather3Data[r].localNetDeviceCount);
    minLocalCollNetCount = std::min(minLocalCollNetCount, allGather3Data[r].localCollNetCount);
    maxLocalCollNetCount = std::max(maxLocalCollNetCount, allGather3Data[r].localCollNetCount);
    if (!allGather3Data[r].isAllNvlink) {
      comm->isAllNvlink = 0;
    }
    comm->minNetCount = std::min(comm->minNetCount, allGather3Data[r].localNetDeviceBw);
  }
  if (rank == 0) {
    INFO(NCCL_INIT, "Local Net device counts across ranks: min %d max %d", minLocalNetCount, maxLocalNetCount);
    INFO(NCCL_INIT, "Local CollNet device counts across ranks: min %d max %d", minLocalCollNetCount,
         maxLocalCollNetCount);

    // Check Net device count mismatch
    if (minLocalNetCount != maxLocalNetCount) {
      // Log mismatched ranks first
      for (int r = 0; r < nranks; r++) {
        if (allGather3Data[r].localNetDeviceCount < maxLocalNetCount) {
          INFO(NCCL_INIT, "Rank %d has %d local Net devices (max %d).", r, allGather3Data[r].localNetDeviceCount,
               maxLocalNetCount);
        }
      }
      // Then warn or error based on env var
      if (ncclParamIgnoreNetMismatch()) {
        INFO(NCCL_INIT,
             "Detected mixed local Net device counts across ranks (min %d, max %d). Ignoring due to "
             "NCCL_IGNORE_NET_MISMATCH.",
             minLocalNetCount, maxLocalNetCount);
      } else {
        WARN("Detected mixed local Net device counts across ranks (min %d, max %d). Set NCCL_IGNORE_NET_MISMATCH=1 to "
             "continue.",
             minLocalNetCount, maxLocalNetCount);
        ret = ncclSystemError;
      }
    }

    // Check CollNet device count mismatch
    if (minLocalCollNetCount != maxLocalCollNetCount) {
      // Log mismatched ranks first
      for (int r = 0; r < nranks; r++) {
        if (allGather3Data[r].localCollNetCount < maxLocalCollNetCount) {
          INFO(NCCL_INIT, "Rank %d has %d local CollNet devices (max %d).", r, allGather3Data[r].localCollNetCount,
               maxLocalCollNetCount);
        }
      }
      // Then warn or error based on env var
      if (ncclParamIgnoreCollNetMismatch()) {
        INFO(NCCL_INIT,
             "Detected mixed local CollNet device counts across ranks (min %d, max %d). Ignoring due to "
             "NCCL_IGNORE_COLLNET_MISMATCH.",
             minLocalCollNetCount, maxLocalCollNetCount);
      } else {
        WARN("Detected mixed local CollNet device counts across ranks (min %d, max %d). Set "
             "NCCL_IGNORE_COLLNET_MISMATCH=1 to continue.",
             minLocalCollNetCount, maxLocalCollNetCount);
        ret = ncclSystemError;
      }
    }

    // Abort on Net/CollNet mismatches
    if (ret != ncclSuccess) goto fail;
  }

  // Alert the user to the presence of mixed CPUs. In the past this has caused
  // locks in some collective routines. This may help debug issues in the future.
  if (rank == 0) {
    if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed architecture were detected.");
    }
    if (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed vendors were detected.");
    }
  }

  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail);
  for (int r = 0; r < comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  comm->minLocalRanks = INT_MAX;
  // Allocate ranks arrays for each node
  for (int n = 0; n < comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->minLocalRanks = std::min(comm->minLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r = 0; r < comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  NCCLCHECKGOTO(initNvlDomainInfo(comm), ret, fail);

  TRACE(NCCL_INIT, "hostHash[%d] %lx localRank %d localRanks %d localRank0 %d", rank, comm->peerInfo[rank].hostHash,
        comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash, comm->localRank, comm->localRanks,
         comm->localRankToRank[0]);
    ret = ncclInternalError;
    goto fail;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d nNodes %d localRanks %d localRank %d MNNVL %d", comm, rank, comm->nRanks,
       comm->nNodes, comm->localRanks, comm->localRank, comm->MNNVL);

  // Enable cross-clique P2P when MNNVL is active, parameter allows it, and there
  // are actually multiple cliques in the NVL domain (nvlDomainSize > clique size).
  comm->p2pCrossClique = comm->MNNVL && ncclParamMNNVLCrossClique() && comm->nvlDomainSize > comm->clique.size;
  if (comm->p2pCrossClique) {
    INFO(NCCL_INIT, "Cross-clique P2P enabled: nvlDomainSize=%d cliqueSize=%d", comm->nvlDomainSize, comm->clique.size);
  }

  nChannelsOrig = comm->nChannels;
  comm->minNetBw = allGather3Data[rank].minNetBw;
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), ret, fail);
  for (int i = 0; i < nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      graphs[a]->nChannels = std::min(allGather3Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
      graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
      graphs[a]->bwIntra = std::min(allGather3Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
      graphs[a]->bwInter = std::min(allGather3Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);
      graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
      graphs[a]->crossNic = std::max(allGather3Data[i].graphInfo[a].crossNic, graphs[a]->crossNic);
    }
    comm->maxTreePattern = std::max(comm->maxTreePattern, allGather3Data[i].graphInfo[NCCL_ALGO_TREE].pattern);
    comm->p2pnChannelsPerPeer = std::min(comm->p2pnChannelsPerPeer, allGather3Data[i].p2pnChannelsPerPeer);
    comm->p2pMaxPeers = std::max(comm->p2pMaxPeers, allGather3Data[i].p2pMaxPeers);
    comm->minNetBw = std::min(comm->minNetBw, allGather3Data[i].minNetBw);
  }
  if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0) comm->config.collnetEnable = 0;
  if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) comm->nvlsSupport = comm->nvlsChannels = 0;

  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclNvlsTuning(comm), ret, fail);
  }

  comm->nChannels = treeGraph->nChannels = ringGraph->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i = 0; i < comm->nChannels; i++) {
      memcpy(comm->channels + comm->nChannels + i, comm->channels + nChannelsOrig + i, sizeof(struct ncclChannel));
    }
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->config.collnetEnable == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet",
           comm->nNodes, collNetNodeThreshold);
      comm->config.collnetEnable = 0;
    }
  }
  comm->isOneRPN = (comm->maxLocalRanks == 1);

  NCCLCHECKGOTO(ncclCalloc(&rings, nranks * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, parent), ret,
                fail);
  // AllGather3 - end
  timers[TIMER_INIT_ALLGATHER] += clockNano() - timers[TIMER_INIT_CONNECT];

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[1024];
  line[0] = '\0';
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line + strlen(line), 1023 - strlen(line), " [%d] %d/%d/%d->%d->%d", c, tree->down[0], tree->down[1],
             tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %02d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank,
         comm->channels[c].ring.next);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  NCCLCHECKGOTO(computeBuffSizes(comm), ret, fail);
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, fail);

  /* until now, all info of comm should be known. We can initialize shared resources and
   * map localRanks to top parent local ranks. NOTE: this shareRes init must be put before
   * all proxy operations. */
  if (comm->sharedRes->owner == comm) {
    comm->sharedRes->tpNLocalRanks = comm->localRanks;
    comm->sharedRes->magic = comm->magic;
    comm->sharedRes->tpNChannels = comm->nChannels;
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), ret, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    int tpRank = comm->topParentRanks[comm->localRankToRank[i]];
    topParentLocalRanks[i] = comm->sharedRes->tpRankToLocalRank[tpRank];
  }
  comm->topParentLocalRanks = topParentLocalRanks;

  // Profiler plugin context has to be initialized before proxy thread
  NCCLCHECK(ncclProfilerPluginInit(comm));

  NCCLCHECKGOTO(ncclTransportCheckP2pType(comm, &comm->isAllDirectP2p, &comm->directMode, &comm->isAllCudaP2p), ret,
                fail);
  // Launch proxy service thread, after this, the proxy calls can be used.
  if (parent && parent->shareResources) {
    comm->proxyState = parent->sharedRes->proxyState;
    ncclAtomicRefCountIncrement(&parent->sharedRes->proxyState->refCount);
  } else {
    NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);
  }
  NCCLCHECKGOTO(ncclCalloc(&comm->gproxyConn, comm->nRanks), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano();
  // Build p2p schedule
  comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, comm->nRanks);
  comm->planner.peers = ncclMemoryStackAlloc<ncclKernelPlanner::Peer>(&comm->memPermanent, comm->nRanks);
  NCCLCHECK(ncclP2pSchedule(comm));
  // initialize non-zero fields.
  comm->planner.bcast_info.minBcastPeer = INT_MAX;
  comm->planner.bcast_info.maxBcastPeer = INT_MIN;

  if (comm->config.numRmaCtx > 0) {
    comm->planner.rmaTaskQueues = ncclMemoryStackAlloc<ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>>(
      &comm->memPermanent, comm->config.numRmaCtx);
    for (int i = 0; i < comm->config.numRmaCtx; i++) {
      ncclIntruQueueConstruct(&comm->planner.rmaTaskQueues[i]);
    }
  } else {
    comm->planner.rmaTaskQueues = NULL;
  }

  comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();
  if (comm->runtimeConn) {
    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings + c * nranks), ret, fail);
    }
    // Attempt to setup NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    // Check if we can setup CollNet
    if (comm->config.collnetEnable) ncclCollNetSetup(comm, parent, graphs);
  } else {
    for (int c = 0; c < comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings + c * nranks), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportRingConnect(comm), ret, fail);

    // Connect Trees
    NCCLCHECKGOTO(ncclTransportTreeConnect(comm), ret, fail);

    // Connect PAT only for communicators with 1 GPU per node
    if (comm->maxLocalRanks == 1) NCCLCHECKGOTO(ncclTransportPatConnect(comm), ret, fail);

    // Attempt to setup NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    NCCLCHECKGOTO(ncclNvlsBufferSetup(comm), ret, fail);

    // And NVLS trees if needed
    NCCLCHECKGOTO(ncclNvlsTreeConnect(comm), ret, fail);

    // Check if we can setup CollNet
    if (comm->config.collnetEnable) {
      ncclCollNetSetup(comm, parent, graphs);
      NCCLCHECKGOTO(ncclCollNetChainBufferSetup(comm), ret, fail);
      if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY + 1) {
        NCCLCHECKGOTO(ncclCollNetDirectBufferSetup(comm), ret, fail);
      }
    }

    // Connect to local net proxy
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), ret, fail);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int),
                                        NULL, 0),
                  ret, fail);

    // Then to remote ones when using PXN
    if (ncclPxnDisable(comm) == 0) {
      int nranks;
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
      for (int r = 0; r < nranks; r++) {
        NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int),
                                            NULL, 0),
                      ret, fail);
      }
    }

    if (ncclParamNvbPreconnect()) {
      // Connect p2p when using NVB path
      int nvbNpeers;
      NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail);
      for (int r = 0; r < nvbNpeers; r++) {
        int peer = nvbPeers[r];
        int sendRound = 0, recvRound = 0;
        while (comm->p2pSchedule[sendRound].sendRank != peer) sendRound++;
        while (comm->p2pSchedule[recvRound].recvRank != peer) recvRound++;
        uint8_t sendBase = ncclP2pChannelBaseForRound(comm, sendRound);
        uint8_t recvBase = ncclP2pChannelBaseForRound(comm, recvRound);
        for (int c = 0; c < comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, sendBase, c);
          if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
            comm->connectSend[peer] |= (1ULL << channelId);
          }
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, recvBase, c);
          if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
            comm->connectRecv[peer] |= (1ULL << channelId);
          }
        }
      }

      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
    }
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  NCCLCHECKGOTO(ncclTopoInitTunerConstants(comm), ret, fail);
  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), ret, fail);
  if (comm->tuner) {
    NCCLCHECK(comm->tuner->init(&comm->tunerContext, comm->commHash, comm->nRanks, comm->nNodes, ncclDebugLog,
                                &comm->nvlDomainInfo, &comm->tunerConstants));
  }
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);

  INFO(NCCL_INIT, "%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer",
       comm->nChannels, comm->nChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

  if (comm->intraRank == 0) {
    // Load ncclParamLaunchMode
    const char* str = ncclGetEnv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = COMPILER_ATOMIC_EXCHANGE(&ncclParamLaunchMode, mode, std::memory_order_relaxed);
    if (modeOld == ncclLaunchModeInvalid && str && str[0] != '\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s",
           mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  NCCLCHECKGOTO(ncclTopoPathAllDirectNVLink(comm->topo, &comm->isAllDirectNvlink), ret, fail);
  comm->globalGinSupport = NCCL_GIN_CONNECTION_NONE;
  if (globalGinSupport && globalCuMemGdrSupport && !comm->hasMloPart) {
    comm->globalGinSupport = globalCrossNicSupport ? NCCL_GIN_CONNECTION_FULL : NCCL_GIN_CONNECTION_RAIL;
  }
  comm->globalRmaProxySupport = globalRmaPluginSupport && globalCrossNicSupport && globalCuMemGdrSupport;
  isOneLsaTeams = ncclDevrIsOneLsaTeam(comm);
  comm->symmetricSupport = comm->isAllCudaP2p && ncclParamWinEnable() && ncclCuMemEnable() &&
                           (comm->globalGinSupport != NCCL_GIN_CONNECTION_NONE || isOneLsaTeams);
  comm->hostRmaSupport =
    comm->config.numRmaCtx > 0 && comm->symmetricSupport && (isOneLsaTeams || comm->globalRmaProxySupport);
  if (!comm->symmetricSupport) {
    INFO(NCCL_INIT,
         "Symmetric memory is not supported. cuMemEnable %d, "
         "globalGinSupport %d, cuMemGdrSupport %d",
         ncclCuMemEnable(), comm->globalGinSupport, globalCuMemGdrSupport);
  }

  comm->ceColl.baseUCSymReadyPtr = NULL;
  comm->ceColl.baseUCSymComplPtr = NULL;

  // Call devCommSetup before the last barrier, making sure we don't have a thread running in front and starting to
  // launch NCCL kernels before all cuda mem allocation is complete. That could cause a deadlock.
  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano() - timers[TIMER_INIT_CONNECT];
  /* Local intra-node barrier */
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                          comm->localRankToRank[0]),
                ret, fail);

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

exit:
  if (ncclOsCpuCount(comm->cpuAffinity)) ncclOsSetAffinity(affinitySave);
  /* If split resource is shared, we are not able to unlink the proxy ops pool here since the child comm can
   * attach the proxy ops pool of parent at any time; otherwise, unlink it here to make sure the pool will be
   * properly cleaned up. */
  if (comm->sharedRes->owner == comm && !comm->shareResources && ret == ncclSuccess && !ncclCuMemEnable()) {
    ncclProxyShmUnlink(comm);
  }
  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather3Data);
  free(rings);
  free(nvbPeers);
  free(pxnPeers);
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);
NCCL_PARAM(CGAClusterSize, "CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT);
// Match config max/minCTAs
NCCL_PARAM(MaxCTAs, "MAX_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MinCTAs, "MIN_CTAS", NCCL_CONFIG_UNDEF_INT);
#define NCCL_MAX_CGA_CLUSTER_SIZE 8

NCCL_PARAM(NChannelsPerNetPeer, "NCHANNELS_PER_NET_PEER", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlinkUtilCentricSchedEnable, "NVLINK_UTIL_CENTRIC_SCHED_ENABLE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(GraphMixingSupport, "GRAPH_MIXING_SUPPORT", NCCL_CONFIG_UNDEF_INT)

#define NCCL_COMMINIT_FUNCNAME_LEN 128
struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  struct ncclComm** newcomm;
  int cudaDev;
  // For ncclCommInitRank
  int nranks, myrank, nId;
  ncclUniqueId* commId;
  // for ncclCommSplit
  struct ncclComm* parent;
  int color, key;
  int childCount;
  // For Shrink
  int* excludeRanksList;
  int excludeRanksCount;
  // name of the function calling
  char funcName[NCCL_COMMINIT_FUNCNAME_LEN];
  // For grow operations
  bool isGrow;
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static void ncclCommFinalizeAsyncJobFree(void* _job) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*)_job;
  delete job;
}

NCCL_PARAM(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(CommShrinkShareResources, "COMM_SHRINK_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);

typedef struct {
  int key;
  int color;
} commSplitInfo;

/*
因为 color/key 是每个 rank 自己选的，谁都不知道别人选了什么，只能 AllGather 一轮。
拿到全局 info[] 后做插入排序：按 key 升序、key 相同则按父 rank 号升序（<=` 让相等时继续往后走，保持稳定性）。
所有 rank 跑同一份info[]、同一段确定性代码，所以算出的parentRanks[]逐字节相同——这是新 comm 的 rank 编号能全局一致的唯一保证。
*/
static ncclResult_t commGetSplitInfo(struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* nRanksRet,
                                     int* myRankRet, int* parentRanksRet) {
  int nRanks = 0, myRank = 0;
  ncclResult_t ret = ncclSuccess;

  commSplitInfo* info = NULL;
  NCCLCHECKGOTO(ncclCalloc(&info, parent->nRanks), ret, fail);

  // Compute nRanks, my rank and the ranks (of the original comm) before and after me
  info[parent->rank].color = color;
  info[parent->rank].key = key;
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, info, sizeof(commSplitInfo)), ret, fail);

  // Negative color does not create a new comm. Return now.
  if (color == NCCL_SPLIT_NOCOLOR) goto exit;

  memset(parentRanksRet, 0xff, sizeof(int) * parent->nRanks);
  for (int i = 0; i < parent->nRanks; i++) {
    if (info[i].color != color) continue;
    // Find where to insert this rank
    int insert = 0;
    while (insert < nRanks && info[parentRanksRet[insert]].key <= info[i].key) insert++;
    // Shift ranks by one after insert
    for (int r = nRanks; r > insert; r--) parentRanksRet[r] = parentRanksRet[r - 1];
    // Insert our rank
    parentRanksRet[insert] = i;
    nRanks++;
  }

  for (int i = 0; i < nRanks; i++) {
    if (parentRanksRet[i] == parent->rank) myRank = i;
  }

  *nRanksRet = nRanks;
  *myRankRet = myRank;

exit:
  free(info);
  return ret;
fail:
  goto exit;
}

/*
一次线性扫描，跳过被排除的 rank，剩下的按原序压紧。
零通信——因为 excludeRanksList 是所有 rank 都传入的同一份数据（调用方先 qsort 过，
we assume excludeRanksList is sorted 依赖的就是这个），所以每个 rank 本地算出的结果必然一致。
这是 shrink 比 split 便宜的根本原因，也是它在故障恢复场景（NCCL_SHRINK_ABORT，网络可能已经不通了）能用的原因。
*/
static ncclResult_t getParentRanks(int parentRanks, int parentRank, int* excludeRanksList, int excludeRanksCount,
                                   int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int count = 0, j = 0;
  for (int i = 0; i < parentRanks; i++) {
    // we assume excludeRanksList is sorted
    if (j < excludeRanksCount && excludeRanksList[j] == i) {
      j++;
      continue;
    }
    if (i == parentRank) *myRankRet = count;
    parentRanksRet[count++] = i;
  }
  *nRanksRet = parentRanks - excludeRanksCount;
  return ncclSuccess;
}

// 初始化主线程执行函数
static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  // 手动单继承，向下转型
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  int archMajor, archMinor;
  size_t maxLocalSizeBytes = 0;
  int cudaDev = job->cudaDev;
  int* parentRanks = NULL;
  int cudaArch;
  int maxSharedMem = 0;
  double sum_timers = 0;
  uint64_t timers[TIMERS_INIT_COUNT] = {0};
  unsigned long long commIdHash;

  timers[TIMER_INIT_TOTAL] = clockNano();
  // 当前可能是位于某个新线程中，新线程的 CUDA current device 是默认值（通常 0），所以需要重新制定当前 device
  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);
  /*
  获取 opt-in 后能拿到的每 block shared memory 上限，Hopper 上 opt-in 前默认每 block 48KB，opt-in 后可到 227KB。
  “Opt-in”的含义：主动申请/显式启用，如果你想让一个 kernel 使用超过“默认限额”的 shared memory，
  必须显式调用 API 向 CUDA runtime 报备，否则 runtime 只会按默认限额（Hopper 上 48 KB/block）给你。
  具体机制：三步 API
    1. 查询设备上限：cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    2. 为特定 kernel 申请(opt-in)：cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    3. 启动时再分配实际用量：kernel<<<grid, block, dynamicSmemBytes>>>(...); dynamicSmemBytes 必须 ≤ 步骤 2 里申请的量。
  为什么要 opt-in？
    1. 占用率(occupancy)的权衡。shared memory 是 SM 上的稀缺资源。一个 block 用得多，一个 SM 上能同时驻留的 block 数就少。
      默认 48 KB：一个 SM 可以塞多个 block。
      opt-in 到 227 KB：Hopper 上 SM 总共约 228 KB shared memory，一个 block 几乎占满 → 一个 SM 通常只能跑 1 个 block。
    2. SM 内部存储是 shared memory 和 L1 cache 共享的物理空间。设备需要知道“这个 kernel 打算把多大比例的 SM 存储划给 shared memory，
    多大比例留给 L1 cache”。opt-in 就是告诉它这个划分意图。
  Hopper 上的数值含义：
    默认 per block 48 KB 不申请就能用
    opt-in per block 227 KB 申请后单 block 上限
    SM 总量 ~228 KB 所以 227 KB/block 意味着一个 SM 基本只能跑一个 block
  注意：这个上限是 per block，不是 per kernel 常驻用量。kernel 里声明的静态 __shared__ + 启动时传入的动态 shared memory 之和不能超过 opt-in 后申请的量。
  */
  CUDACHECKGOTO(cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev), res, fail);
  // 获取当前 CUDA Arch 的计算能力 (Compute Capability, CC)
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), res, fail);
  // 数值格式对齐 __CUDA_ARCH__，例如：sm_90 → 900，sm_100 → 1000，sm_121 → 1210。留出的十位是给 minor 的，个位永远是 0
  cudaArch = 100 * archMajor + 10 * archMinor;

  timers[TIMER_INIT_KERNELS] = clockNano();
  /*
  遍历 ncclDevKernelList[] 和 ncclSymkKernelList[]，对每个 kernel 做三件事：
    1. 驱动版本过滤。把当前驱动跑不了的 kernel 从全局表里划掉，后续选路自然不会选到它。
    2. shared memory opt-in。用设备上限减去该 kernel 的静态 __shared__ 用量，剩下的全部报备为可用动态量。
    3. 栈需求统计。*maxStackSize = max(..., attr.localSizeBytes)，输出给 maxLocalSizeBytes。
  这里写的是进程级全局表，被每个 comm 各调一次，多线程会起 8 个 job 线程并发跑到这里，8 个线程同时写同一批全局槽位。
  这是 benign race（同 device 时写入同值），但在异构 GPU 的 ncclCommInitAll 下（不同 device 的 maxSharedMem 不同）就变成了真正的竞争。
  */
  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, maxSharedMem, &maxLocalSizeBytes));
  /*
  如果不预先把 cudaLimitStackSize 提到最大需求，CUDA 会在第一次 launch 需要大栈的 kernel 时重新配置 device memory，
  那个操作会隐式同步并可能与 NVSHMEM 之类的库冲突。所以宁可提前一次性抬高。
  但默认是关的，NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0)，因为抬高栈会吃掉本可用于其他用途的显存。
  作用域是整个 device、影响该 device 上所有后续 kernel，不只是 NCCL 的。
  */
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }
  timers[TIMER_INIT_KERNELS] = clockNano() - timers[TIMER_INIT_KERNELS];

  /*
  两个 bool 组合出四种情况，实际用到三种：
  parent	isGrow	走哪支	      对应 API	                           bootstrap
  ≠NULL	  false	  split 分支	 ncclCommSplit / ncclCommShrink	      bootstrapSplit
  ≠NULL	  true	  else 分支	   ncclCommGrow（已有 rank）	            bootstrapInit
  NULL	  false	  else 分支	   ncclCommInitRank(Config) / InitAll	  bootstrapInit
  NULL	  true	  else 分支	   ncclCommGrow（新加入的 rank）	        bootstrapInit
  分类维度是"用什么方式建立 bootstrap 网络"，不是"有没有父 comm"：
    split/shrink：新 comm 的成员是父 comm 成员的子集，父 comm 的 bootstrap 环还在，可以借道父环交换信息 → bootstrapSplit
    grow：新 comm 的成员是父 comm 的超集，新 rank 根本不在父环里，借不了 → 必须像冷启动一样用 ncclUniqueId 从头建环 → bootstrapInit
  所以 isGrow 这个字段存在的全部理由，就是把"有 parent 但不能借父环"这种情况从 split 分支拨出去。
  */
  if (job->parent && !job->isGrow) {
    // SPLIT/SHRINK: use bootstrapSplit
    // parentRanks[i] = 新 comm 里 rank i 在父 comm 里的 rank 号。长度按父 comm 的 nRanks 分配（上界），实际只用前 job->nranks 项
    // 这张表是 bootstrapSplit 的核心输入——它要靠它知道"我要和父环里的哪几个人组新环"
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);
    /*
    两种 rank 重编号算法
    excludeRanksCount != 0 是 shrink 的判据（只有 ncclCommShrink 会填 excludeRanksList）。两者的差别是要不要通信
    */
    if (job->excludeRanksCount) {  // SHRINK
      // 纯本地计算
      NCCLCHECKGOTO(getParentRanks(job->parent->nRanks, job->parent->rank, job->excludeRanksList,
                                   job->excludeRanksCount, &job->nranks, &job->myrank, parentRanks),
                    res, fail);
    } else {  // SPLIT
      // 必须通信
      NCCLCHECKGOTO(commGetSplitInfo(comm, job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks),
                    res, fail);
      // 这个 rank 不要新 comm，但它必须参加那次 AllGather，commGetSplitInfo 内部也有一道对应的早退，在 AllGather 之后、插入排序之前。
      // Negative color does not create a new comm object. We needed to take part in the allgather, but we're done now.
      if (job->color == NCCL_SPLIT_NOCOLOR) goto exit;
    }
    // eatHash/digestHash 是流式哈希：hacc[2] 是累加器，eatHash 逐块吃进去混淆，digestHash 最后收敛成一个 uint64_t。
    // child hash obtained from (parent hash, split count, color)
    uint64_t hacc[2] = {1, 1};
    eatHash(hacc, &job->parent->commHash);
    eatHash(hacc, &job->childCount);
    eatHash(hacc, &job->color);
    comm->commHash = digestHash(hacc);
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    comm->isGrow = false;
    INFO(NCCL_INIT,
         "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p childCount %d color %d key %d- Init "
         "START",
         job->funcName, comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent,
         job->childCount, job->color, job->key);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
    // debug info, no commId was used
    commIdHash = 0;
  } else {
    // GROW or NORMAL INIT: use bootstrapInit
    if (job->isGrow) {
      struct ncclBootstrapHandle* growHandle = (struct ncclBootstrapHandle*)job->commId;
      uint64_t baseMagic = growHandle ? growHandle->magic : hashCombine(job->parent->magic, job->parent->childCount);
      comm->commHash = commIdHash = hashCombine(baseMagic, job->nranks);
      INFO(NCCL_INIT, "Rank %d: Generated commHash 0x%lx from baseMagic 0x%lx and newNRanks %d", job->myrank,
           comm->commHash, baseMagic, job->nranks);
    } else {
      /*
      getHash 直接对 128 字节的 ncclUniqueId.internal 做哈希。只取 commId[0]，因为 commHash 的唯一要求是全局一致，所有 rank 必须算出同一个值。
      commHash 与 comm->magic 是两个不同的东西，都源于同一个 uniqueId：
        comm->commHash 的用途是 comm 的全局身份，group 内跨 comm 排序的键、日志/RAS 标识、内存注册表的键。
        comm->magic 的用途是线协议令牌，socket 握手时双方核对，防止不同 job 的进程串连
      */
      // obtain a unique hash using the first commId
      comm->commHash = commIdHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
    }
    timers[TIMER_INIT_ALLOC] = clockNano();
    /*
    1. 初始化 NCCL 内存栈 comm->memPermanent 和 comm->memScoped；
    2. 初始化共享资源：StrongStream、Events 等等；
    3. 初始化三大子系统插件：NET、RMA、GIN；
    4. 初始化 MemManager，记录采用 CUDA VMM 动态释放和申请物理内存，保留虚拟内存地址；
    5. 进程全局记录当前 context 作为 key 分配一个 StrongStream，用于跨 comm 的启动顺序串行化；
    6. 初始化 NVML，预存 P2P 读写能力探测，记录当前 CUDA 设备的全局索引编号；
    7. 探测是否支持 DMA-BUF（GPUDirect RDMA 零拷贝的手段之一）；
    8. 各种数据结构初始化（本次总结的重点区）
      对象内存池：ncclMemoryPoolConstruct 初始化 memPool_ncclKernelPlan、memPool_ncclProxyOp（host 端 free-list，只显式构造这两个）。
      group 链表哨兵：groupNext[]、preconnectNext 全置 0x1（区分"未入链 / 链中 / 链尾 nullptr"三态）。
      连接位掩码：两条 static_assert 保证 MAXCHANNELS(64) <= uint64_t 位宽，然后按 nRanks 分配 connectSend/connectRecv。
      channel 惰性初始化：全部 channels[c].id = -1 标记未初始化。
      bcast 调度状态：预分配 ringTasks（长 nRanks 的环深度排序工作区，挂 memPermanent），minBcastPeer=INT_MAX/maxBcastPeer=INT_MIN 空区间。
      topParentRanks：默认恒等映射 i→i。
      五个队列：callbackQueue 用 MPSC 无锁（跨线程收回调），其余四个（legacyRegCleanupQueue/ceInitTaskQueue/suspendTaskQueue/resumeTaskQueue）用普通侵入式队列。
      注册缓存页大小：regCache.pageSize = ncclOsGetPageSize()（内存注册页对齐基准）。
    9. 创建 CUDA 显存池
    */
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];

    comm->isGrow = job->isGrow;
    INFO(NCCL_INIT, "[Rank %d] %s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START",
         job->myrank, job->funcName, comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId,
         commIdHash);
    
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    /*
    1. 准备 bootstrapState 和 magic
    2. 创建 root 回连监听 socket 和 ring 监听 socket
    3. 错峰(stagger) 向 root 发送 ring 监听地址和回连监听地址
    4. 接收 root 回传的后继 ring rank 监听地址，close 回连监听 socket
    5. connect 后继 ring rank 监听地址，保存 send socket。从 ring 监听 socket 从 accept socket，保存到 recv socket
    6. 创建 proxy 监听 socket 和 p2p mesh 监听 socket，生成 UDS hash
    7. 初始化 RAS
      1. 创建 RAS 监听 socket 和固定端口的客户端监听 socket
      2. 通过 pipe 创建 fds 读写端口管道
      3. 启动 RAS 独立线程
    8. 全部 ranks AllGather（采用 Ring 环的 send/recv socket），把每个 rank 的 4 类信息(peerP2pAddresses mesh 地址、
    peerProxyAddresses、peerProxyAddressesUDS、rasRanks)交换给全员。至此每个 rank 都持有所有人的 mesh/proxy/UDS/RAS 地址。
    9. proxy 状态初始化
      1. 保存 proxy 监听 socket
      2. 指定 AF_UNIX 创建 UDS，并通过 rank + UDS hash 生成地址，最终采用抽象命名模式 bind UDS
    10. 经 self-pipe 把全员数组异步移交 RAS 线程
    */
    NCCLCHECKGOTO(bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm, job->parent), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  }
  comm->cudaArch = cudaArch;

  NCCLCHECKGOTO(initTransportsRank(comm, job->parent, timers), res, fail);

  // comm 初始化状态更新为 success
  // update communicator state
  comm->initState = ncclSuccess;
  timers[TIMER_INIT_TOTAL] = clockNano() - timers[TIMER_INIT_TOTAL];

  // 输出所有统计耗时相关信息
  // Trace this call for replay tool
  if (job->parent) {
    /* unlink child abort flag. */
    COMPILER_ATOMIC_STORE(&job->parent->childAbortFlag, static_cast<uint32_t*>(nullptr), std::memory_order_release);
    if (job->isGrow) {
      TRACE_CALL("ncclCommGrow(%p, %d, %p)", job->parent, comm->nRanks, comm);
    } else {
      TRACE_CALL("ncclCommSplit(%p, %d, %d, %p, %d, %d)", job->parent, job->color, job->key, comm, comm->rank,
                 comm->nRanks);
    }
    INFO(NCCL_INIT,
         "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p childCount %d color %d key %d - Init "
         "COMPLETE",
         job->funcName, comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent,
         job->childCount, job->color, job->key);
  } else {
    // the name for the replay tool is ncclCommInitRank for all the variations
    TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)", comm, comm->nRanks, commIdHash, comm->rank, comm->cudaDev);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init COMPLETE",
         job->funcName, comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  }
  sum_timers = 0.0;
  for (int it = 1; it < TIMERS_INIT_COUNT; ++it) sum_timers += (timers[it] / 1e9);
  INFO(NCCL_INIT | NCCL_PROFILE,
       "Init timings - %s: rank %d nranks %d total %.2f (kernels %.2f, alloc %.2f, bootstrap %.2f, allgathers %.2f, "
       "topo %.2f, graphs %.2f, "
       "connections %.2f, rest %.2f)",
       job->funcName, comm->rank, comm->nRanks, timers[TIMER_INIT_TOTAL] / 1e9, timers[TIMER_INIT_KERNELS] / 1e9,
       timers[TIMER_INIT_ALLOC] / 1e9, timers[TIMER_INIT_BOOTSTRAP] / 1e9, timers[TIMER_INIT_ALLGATHER] / 1e9,
       timers[TIMER_INIT_TOPO] / 1e9, timers[TIMER_INIT_GRAPHS] / 1e9, timers[TIMER_INIT_CONNECT] / 1e9,
       timers[TIMER_INIT_TOTAL] / 1e9 - sum_timers);
exit:
  if (job->newcomm) {
    /* assign it to user pointer. */
    COMPILER_ATOMIC_STORE(job->newcomm, comm, std::memory_order_release);
  }
  if (parentRanks) free(parentRanks);
  return res;
fail:
  comm->initState = res;
  goto exit;
}

#define NCCL_CONFIG_DEFAULT(config, field, undef, defvalue, fieldStr, format) \
  if (config->field == undef) { \
    config->field = defvalue; \
  } else { \
    INFO(NCCL_ENV, "Comm config " fieldStr " set to " format, config->field); \
  }

static ncclResult_t envConfigOverride(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  const char* tmpNetName = comm->config.netName;
  const char* envNetName;
  int blockingEnv;
  int cgaClusterSizeEnv;
  int minCTAsEnv;
  int maxCTAsEnv;
  int splitShareEnv;
  const char* collnetEnableEnv;
  int shrinkShareEnv;
  int nvlsCTAsEnv;
  int nChannelsPerNetPeerEnv;
  int nvlinkUtilCentricSchedEnableEnv;
  int graphMixingSupportEnv;
  int graphStreamOrderingEnv;
  int numRmaCtxEnv;
  int maxP2pPeersEnv;
  const char* checkModeEnv;

  /* override configuration with env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1) comm->config.blocking = blockingEnv;

  cgaClusterSizeEnv = ncclParamCGAClusterSize();
  if (0 <= cgaClusterSizeEnv && cgaClusterSizeEnv <= NCCL_MAX_CGA_CLUSTER_SIZE) {
    if (comm->config.cgaClusterSize != NCCL_CONFIG_UNDEF_INT) {
      INFO(NCCL_ENV, "Comm config cgaClusterSize reset to NCCL_MAX_CGA_CLUSTER_SIZE=%d", cgaClusterSizeEnv);
    }
    comm->config.cgaClusterSize = cgaClusterSizeEnv;
  } else if (cgaClusterSizeEnv > NCCL_MAX_CGA_CLUSTER_SIZE) {
    INFO(NCCL_ENV, "NCCL_CGA_CLUSTER_SIZE value %d is too big. Limiting value to %d.", cgaClusterSizeEnv,
         NCCL_MAX_CGA_CLUSTER_SIZE);
    comm->config.cgaClusterSize = NCCL_MAX_CGA_CLUSTER_SIZE;
  }

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (minCTAsEnv <= 0) {
      INFO(NCCL_ENV, "NCCL_MIN_CTAS %d is too low, leaving it set at %d", minCTAsEnv, comm->config.minCTAs);
    } else {
      if (comm->config.minCTAs != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config minCTAs reset to NCCL_MIN_CTAS=%d", minCTAsEnv);
      }
      comm->config.minCTAs = minCTAsEnv;
    }
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (maxCTAsEnv <= 0) {
      INFO(NCCL_ENV, "NCCL_MAX_CTAS %d is too low, leaving it set at %d", maxCTAsEnv, comm->config.maxCTAs);
    } else {
      if (comm->config.maxCTAs != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config maxCTAs reset to NCCL_MAX_CTAS=%d", maxCTAsEnv);
      }
      comm->config.maxCTAs = maxCTAsEnv;
    }
  }

  /* override configuration with env variable. */
  nChannelsPerNetPeerEnv = ncclParamNChannelsPerNetPeer();
  if (nChannelsPerNetPeerEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nChannelsPerNetPeerEnv <= 0) {
      INFO(NCCL_ENV, "NCCL_NCHANNELS_PER_NET_PEER %d is too low, leaving it set at %d", nChannelsPerNetPeerEnv,
           comm->config.nChannelsPerNetPeer);
    } else {
      if (comm->config.nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config nChannelsPerNetPeer reset to NCCL_NCHANNELS_PER_NET_PEER=%d",
             nChannelsPerNetPeerEnv);
      }
      comm->config.nChannelsPerNetPeer = nChannelsPerNetPeerEnv;
    }
  }

  nvlinkUtilCentricSchedEnableEnv = ncclParamNvlinkUtilCentricSchedEnable();
  if (nvlinkUtilCentricSchedEnableEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nvlinkUtilCentricSchedEnableEnv != 0 && nvlinkUtilCentricSchedEnableEnv != 1) {
      INFO(NCCL_ENV, "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE %d is not valid, leaving it set at %d",
           nvlinkUtilCentricSchedEnableEnv, comm->config.nvlinkCentricSched);
    } else {
      if (comm->config.nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config nvlinkCentricSched reset to NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=%d",
             nvlinkUtilCentricSchedEnableEnv);
      }
      comm->config.nvlinkCentricSched = nvlinkUtilCentricSchedEnableEnv;
    }
  }

  graphMixingSupportEnv = ncclParamGraphMixingSupport();
  if (graphMixingSupportEnv != NCCL_CONFIG_UNDEF_INT) {
    if (graphMixingSupportEnv != 0 && graphMixingSupportEnv != 1) {
      INFO(NCCL_ENV, "NCCL_GRAPH_MIXING_SUPPORT %d is not valid, leaving it set at %d", graphMixingSupportEnv,
           comm->config.graphUsageMode);
    } else {
      if (comm->config.graphUsageMode != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config graphUsageMode reset to %d by NCCL_GRAPH_MIXING_SUPPORT=%d",
             graphMixingSupportEnv == 1 ? 2 : 0, graphMixingSupportEnv);
      }
      comm->config.graphUsageMode = graphMixingSupportEnv == 1 ? 2 : 0;
    }
  }

  numRmaCtxEnv = ncclParamNumRmaCtx();
  if (numRmaCtxEnv != NCCL_CONFIG_UNDEF_INT) {
    if (numRmaCtxEnv < 0) {
      INFO(NCCL_ENV, "NCCL_NUM_RMA_CTX %d is too low, leaving it set at %d", numRmaCtxEnv, comm->config.numRmaCtx);
    } else {
      if (numRmaCtxEnv == 0) INFO(NCCL_ENV, "NCCL_NUM_RMA_CTX=0, RMA disabled for this communicator");
      comm->config.numRmaCtx = numRmaCtxEnv;
    }
  }

  maxP2pPeersEnv = ncclParamMaxP2pPeers();
  if (maxP2pPeersEnv != NCCL_CONFIG_UNDEF_INT) {
    if (maxP2pPeersEnv <= 0) {
      INFO(NCCL_ENV, "NCCL_MAX_P2P_PEERS %d is too low, leaving it set at %d", maxP2pPeersEnv,
           comm->config.maxP2pPeers);
    } else {
      if (comm->config.maxP2pPeers != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config maxP2pPeers reset to NCCL_MAX_P2P_PEERS=%d", maxP2pPeersEnv);
      }
      comm->config.maxP2pPeers = maxP2pPeersEnv;
    }
  }

  graphStreamOrderingEnv = ncclParamGraphStreamOrdering();
  if (graphStreamOrderingEnv != NCCL_CONFIG_UNDEF_INT) {
    if (graphStreamOrderingEnv != 0 && graphStreamOrderingEnv != 1) {
      INFO(NCCL_ENV, "NCCL_GRAPH_STREAM_ORDERING %d is not valid, leaving it set at %d", graphStreamOrderingEnv,
           comm->config.graphStreamOrdering);
    } else {
      if (comm->config.graphStreamOrdering != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config graphStreamOrdering reset to NCCL_GRAPH_STREAM_ORDERING=%d",
             graphStreamOrderingEnv);
      }
      comm->config.graphStreamOrdering = graphStreamOrderingEnv;
    }
  }

  envNetName = ncclGetEnv("NCCL_NET");
  if (envNetName) tmpNetName = envNetName;
  if (tmpNetName != NULL) {
    if (comm->config.netName != NCCL_CONFIG_UNDEF_PTR) {
      INFO(NCCL_ENV, "Comm config netName reset to NCCL_NET=%s", tmpNetName);
    }
    int netNameLen = strlen(tmpNetName) + 1;
    comm->config.netName = (char*)malloc(netNameLen);
    memcpy((void*)comm->config.netName, tmpNetName, netNameLen);
    INFO_LOC(NCCL_ALLOC_HOST, "netName buffer Size %d pointer %p", netNameLen, comm->config.netName);
  } else {
    comm->config.netName = NULL;
  }

  splitShareEnv = ncclParamCommSplitShareResources();
  if (splitShareEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.splitShare != NCCL_CONFIG_UNDEF_INT) {
      INFO(NCCL_ENV, "Comm config splitShare reset to NCCL_COMM_SPLIT_SHARE_RESOURCES=%d", splitShareEnv);
    }
    comm->config.splitShare = splitShareEnv;
  }
  shrinkShareEnv = ncclParamCommShrinkShareResources();
  if (shrinkShareEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.shrinkShare != NCCL_CONFIG_UNDEF_INT) {
      INFO(NCCL_ENV, "Comm config shrinkShare reset to NCCL_COMM_SHRINK_SHARE_RESOURCES=%d", shrinkShareEnv);
    }
    comm->config.shrinkShare = shrinkShareEnv;
  }

  // NCCL_COLLNET_ENABLE needs to be reloaded each time for comm init
  // since users might change the env on the fly to enable/disable collnet
  collnetEnableEnv = ncclGetEnv("NCCL_COLLNET_ENABLE");
  if (collnetEnableEnv != NULL) {
    int collnetEnableInt = (int)strtol(collnetEnableEnv, NULL, 0);
    if (collnetEnableInt != NCCL_CONFIG_UNDEF_INT) {
      if (comm->config.collnetEnable != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config collnetEnable reset to NCCL_COLLNET_ENABLE=%d", collnetEnableInt);
      }
      comm->config.collnetEnable = collnetEnableInt;
      INFO(NCCL_ENV, "NCCL_COLLNET_ENABLE set by environment to %d.", collnetEnableInt);
    }
  }

  static std::once_flag onceEnvCtaPolicy;
  std::call_once(onceEnvCtaPolicy, getEnvCtaPolicyOnce);
  if (ctaPolicyEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.CTAPolicy != NCCL_CONFIG_UNDEF_INT) {
      INFO(NCCL_ENV, "Comm config CTAPolicy reset to NCCL_CTA_POLICY=%d", ctaPolicyEnv);
    }
    comm->config.CTAPolicy = ctaPolicyEnv;
  }

  nvlsCTAsEnv = ncclParamNvlsChannels();
  if (nvlsCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT) {
      INFO(NCCL_ENV, "Comm config nvlsCTAs reset to NCCL_NVLS_NCHANNELS=%d", nvlsCTAsEnv);
    }
    comm->config.nvlsCTAs = nvlsCTAsEnv;
  }

  /* cap channels if needed */
  if (comm->config.minCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "minCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.minCTAs,
         MAXCHANNELS, MAXCHANNELS);
    comm->config.minCTAs = MAXCHANNELS;
  }

  if (comm->config.maxCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "maxCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.maxCTAs,
         MAXCHANNELS, MAXCHANNELS);
    comm->config.maxCTAs = MAXCHANNELS;
  }

  if (comm->config.minCTAs > comm->config.maxCTAs) {
    INFO(NCCL_ENV, "minCTAs %d is larger than maxCTAs %d, set both to %d", comm->config.minCTAs, comm->config.maxCTAs,
         comm->config.maxCTAs);
    comm->config.minCTAs = comm->config.maxCTAs;
  }

  if (comm->config.splitShare != 1 && comm->config.splitShare != 0) {
    INFO(NCCL_ENV, "splitShare %d is not a valid value 0/1, set it to 0", comm->config.splitShare);
    comm->config.splitShare = 0;
  }

  if (comm->config.collnetEnable != 1 && comm->config.collnetEnable != 0) {
    INFO(NCCL_ENV, "collnetEnable %d is not a valid value 0/1, set it to 0", comm->config.collnetEnable);
    comm->config.collnetEnable = 0;
  }

  if (!ctaPolicyIsValid(comm->config.CTAPolicy)) {
    INFO(NCCL_ENV, "CTAPolicy %d is not a valid value, set it to %d", comm->config.CTAPolicy, NCCL_CTA_POLICY_DEFAULT);
    comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
  }

  if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT && comm->config.nvlsCTAs <= 0) {
    INFO(NCCL_ENV, "nvlsCTAs %d is not a valid value, NCCL will decide the default value automatically",
         comm->config.nvlsCTAs);
    comm->config.nvlsCTAs = NCCL_CONFIG_UNDEF_INT;
  }

  // If POLICY_ZERO and POLICY_EFFICIENCY are set in CTAPolicy, unset POLICY_EFFICIENCY.
  if ((comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) && (comm->config.CTAPolicy & NCCL_CTA_POLICY_EFFICIENCY)) {
    WARN("Both NCCL_CTA_POLICY_ZERO and NCCL_CTA_POLICY_EFFICIENCY are set in CTAPolicy (%d). Unsetting "
         "POLICY_EFFICIENCY.",
         comm->config.CTAPolicy);
    comm->config.CTAPolicy &= ~NCCL_CTA_POLICY_EFFICIENCY;
  }

  // read non-config env settings
  comm->checkMode = ncclCheckModeDefault;
  if (ncclParamCheckPointers() == 1) {
    // @deprecated: use NCCL_CHECK_MODE instead
    comm->checkMode = ncclCheckModeDebugLocal;
  }

  checkModeEnv = ncclGetEnv("NCCL_CHECK_MODE");
  if (checkModeEnv) {
    INFO(NCCL_ENV, "NCCL_CHECK_MODE set by environment to %s", checkModeEnv);
    if (strcasecmp(checkModeEnv, "DEBUG_GLOBAL") == 0) {
      comm->checkMode = ncclCheckModeDebugGlobal;
    } else if (strcasecmp(checkModeEnv, "DEBUG_LOCAL") == 0) {
      comm->checkMode = ncclCheckModeDebugLocal;
    }
  }
  return ret;
}

static ncclResult_t copyCommConfig(ncclComm_t childComm, ncclComm_t parnet) {
  memcpy(&childComm->config, &parnet->config, sizeof(ncclConfig_t));
  NCCLCHECK(envConfigOverride(childComm));
  return ncclSuccess;
}

static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t* config) {
  ncclResult_t ret = ncclSuccess;
  /* config must not be NULL in this function */
  ncclConfig_t defaultConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t* internalConfigPtr;
  size_t realSize;

  internalConfig.magic = 0;
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != NCCL_API_MAGIC) {
      WARN("ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }

    /* check version. */
    if (internalConfigPtr->version < NCCL_VERSION(2, 14, 0)) {
      internalConfigPtr->blocking = defaultConfig.blocking;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 17, 0)) {
      internalConfigPtr->cgaClusterSize = defaultConfig.cgaClusterSize;
      internalConfigPtr->minCTAs = defaultConfig.minCTAs;
      internalConfigPtr->maxCTAs = defaultConfig.maxCTAs;
      internalConfigPtr->netName = defaultConfig.netName;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 25, 0)) {
      internalConfigPtr->trafficClass = defaultConfig.trafficClass;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 27, 0)) {
      internalConfigPtr->collnetEnable = defaultConfig.collnetEnable;
      internalConfigPtr->CTAPolicy = defaultConfig.CTAPolicy;
      internalConfigPtr->shrinkShare = defaultConfig.shrinkShare;
      internalConfigPtr->nvlsCTAs = defaultConfig.nvlsCTAs;
    }
    if (internalConfigPtr->version < NCCL_VERSION(2, 28, 0)) {
      internalConfigPtr->nChannelsPerNetPeer = defaultConfig.nChannelsPerNetPeer;
      internalConfigPtr->nvlinkCentricSched = defaultConfig.nvlinkCentricSched;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 29, 0)) {
      internalConfigPtr->graphUsageMode = defaultConfig.graphUsageMode;
      internalConfigPtr->numRmaCtx = defaultConfig.numRmaCtx;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 30, 0)) {
      internalConfigPtr->maxP2pPeers = defaultConfig.maxP2pPeers;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 30, 5)) {
      internalConfigPtr->graphStreamOrdering = defaultConfig.graphStreamOrdering;
    }
  }

  /* check input config attributes, -1 means user-undefined and we should use default value from NCCL. */
  if (internalConfigPtr->blocking != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->blocking != 0 &&
      internalConfigPtr->blocking != 1) {
    WARN("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->cgaClusterSize != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->cgaClusterSize < 0) {
    WARN("Invalid config cgaClusterSize attribute value %d", internalConfigPtr->cgaClusterSize);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if ((internalConfigPtr->minCTAs != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->minCTAs <= 0) ||
      (internalConfigPtr->maxCTAs != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->maxCTAs <= 0) ||
      (internalConfigPtr->minCTAs > internalConfigPtr->maxCTAs)) {
    WARN("Invalid config min/max channels attribute value %d/%d", internalConfigPtr->minCTAs,
         internalConfigPtr->maxCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->splitShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->splitShare != 0 &&
      internalConfigPtr->splitShare != 1) {
    WARN("Invalid config splitShare attribute value %d", internalConfigPtr->splitShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->collnetEnable != NCCL_CONFIG_UNDEF_INT &&
      (internalConfigPtr->collnetEnable < 0 || internalConfigPtr->collnetEnable > 1)) {
    WARN("Invalid config collnetEnable attribute value %d", internalConfigPtr->collnetEnable);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->CTAPolicy != NCCL_CONFIG_UNDEF_INT) {
    if (!ctaPolicyIsValid(internalConfigPtr->CTAPolicy)) {
      WARN("Invalid config policy attribute value %d", internalConfigPtr->CTAPolicy);
      ret = ncclInvalidArgument;
      goto fail;
    }
  }

  if (internalConfigPtr->shrinkShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->shrinkShare != 0 &&
      internalConfigPtr->shrinkShare != 1) {
    WARN("Invalid config shrinkShare attribute value %d", internalConfigPtr->shrinkShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlsCTAs != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlsCTAs <= 0) {
    WARN("Invalid config nvlsCTAs attribute value %d", internalConfigPtr->nvlsCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT &&
      (internalConfigPtr->nChannelsPerNetPeer <= 0 || internalConfigPtr->nChannelsPerNetPeer > MAXCHANNELS)) {
    WARN("Invalid config nChannelsPerNetPeer attribute value %d", internalConfigPtr->nChannelsPerNetPeer);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlinkCentricSched != 0 &&
      internalConfigPtr->nvlinkCentricSched != 1) {
    WARN("Invalid config nvlinkCentricSched attribute value %d", internalConfigPtr->nvlinkCentricSched);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->graphUsageMode != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->graphUsageMode != 0 &&
      internalConfigPtr->graphUsageMode != 1 && internalConfigPtr->graphUsageMode != 2) {
    WARN("Invalig config graphUsageMode attribute value %d", internalConfigPtr->graphUsageMode);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->numRmaCtx != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->numRmaCtx < 0) {
    WARN("Invalid config numRmaCtx attribute value %d", internalConfigPtr->numRmaCtx);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->maxP2pPeers != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->maxP2pPeers <= 0) {
    WARN("Invalid config maxP2pPeers attribute value %d", internalConfigPtr->maxP2pPeers);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->graphStreamOrdering != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->graphStreamOrdering != 0 &&
      internalConfigPtr->graphStreamOrdering != 1) {
    WARN("Invalid config graphStreamOrdering attribute value %d", internalConfigPtr->graphStreamOrdering);
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* default config value can be tuned on different platform. */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, NCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, cgaClusterSize, NCCL_CONFIG_UNDEF_INT, 4, "CGA cluster size", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, NCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, NCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, NCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, splitShare, NCCL_CONFIG_UNDEF_INT, 0, "Split share", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, trafficClass, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "Traffic class",
                      "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, commName, NCCL_CONFIG_UNDEF_PTR, NULL, "Comm name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, collnetEnable, NCCL_CONFIG_UNDEF_INT, 0, "Collnet enable", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, CTAPolicy, NCCL_CONFIG_UNDEF_INT, NCCL_CTA_POLICY_DEFAULT, "CTA policy flags",
                      "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, shrinkShare, NCCL_CONFIG_UNDEF_INT, 0, "shrinkShare", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlsCTAs, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "nvlsCTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nChannelsPerNetPeer, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT,
                      "nChannelsPerNetPeer", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlinkCentricSched, NCCL_CONFIG_UNDEF_INT, 0, "nvlinkCentricSched", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, graphUsageMode, NCCL_CONFIG_UNDEF_INT, 2, "graphUsageMode", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, numRmaCtx, NCCL_CONFIG_UNDEF_INT, 1, "numRmaCtx", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxP2pPeers, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "maxP2pPeers",
                      "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, graphStreamOrdering, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT,
                      "graphStreamOrdering", "%d");

  /* assign config to communicator */
  comm->config.blocking = internalConfigPtr->blocking;
  comm->config.cgaClusterSize = internalConfigPtr->cgaClusterSize;
  comm->config.minCTAs = internalConfigPtr->minCTAs;
  comm->config.maxCTAs = internalConfigPtr->maxCTAs;
  comm->config.netName = internalConfigPtr->netName;
  comm->config.splitShare = internalConfigPtr->splitShare;
  comm->config.trafficClass = internalConfigPtr->trafficClass;
  comm->config.commName = internalConfigPtr->commName;
  comm->config.collnetEnable = internalConfigPtr->collnetEnable;
  comm->config.CTAPolicy = internalConfigPtr->CTAPolicy;
  comm->config.shrinkShare = internalConfigPtr->shrinkShare;
  comm->config.nvlsCTAs = internalConfigPtr->nvlsCTAs;
  comm->config.nChannelsPerNetPeer = internalConfigPtr->nChannelsPerNetPeer;
  comm->config.nvlinkCentricSched = internalConfigPtr->nvlinkCentricSched;
  comm->config.graphUsageMode = internalConfigPtr->graphUsageMode;
  comm->config.numRmaCtx = internalConfigPtr->numRmaCtx;
  comm->config.maxP2pPeers = internalConfigPtr->maxP2pPeers;
  comm->config.graphStreamOrdering = internalConfigPtr->graphStreamOrdering;
  NCCLCHECKGOTO(envConfigOverride(comm), ret, fail);

  // Resolve to system default (serialize) if neither user config nor env var set it.
  if (comm->config.graphStreamOrdering == NCCL_CONFIG_UNDEF_INT) comm->config.graphStreamOrdering = 1;

  // Warn and fall back when graphStreamOrdering=0 is combined with graphUsageMode=2 (unsupported).
  if (comm->config.graphStreamOrdering == 0 && comm->config.graphUsageMode == 2) {
    WARN("graphStreamOrdering=0 with graphUsageMode=2 (graph mixing) is not supported; "
         "falling back to graphStreamOrdering=1 for this communicator");
    comm->config.graphStreamOrdering = 1;
  }

exit:
  return ret;
fail:
  goto exit;
}

// 
static void ncclCommInitJobFree(void* _job) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)_job;
  // 释放纯 char 数组构成的结构体 POD，采用 free
  free(job->commId);
  // 包含非 POD 数据结构，采用 delete
  delete job;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, int nId, ncclUniqueId* commId, int myrank,
                                        int cudaDev, ncclConfig_t* config, const char funcName[]) {
  // nId 表示传入了多少个 ncclUniqueId
  // 对于普通 ncclCommInitRank()，这里固定是 1；ncclCommInitRankScalable() 才可能大于 1，用于多 root bootstrap，缓解单一 root 的连接压力。
  if (nId <= 0 || nId > nranks) {
    WARN("improper usage of ncclCommInitRank: nId = %d, nranks=%d", nId, nranks);
    return ncclInvalidArgument; // 这个检查在任何资源分配之前，所以可以直接 return，不需要走 fail:
  }
  ncclResult_t res = ncclSuccess;
  const char* commIdEnv = NULL;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob* job = NULL;
  bool launchedJob = false;

  // 初始化栈大小、初始化 GDRCopy、获取用于 bootstrapt 的网口、注册 ncclRedOp_t 枚举类型到 NVTX
  // first call ncclInit, this will setup the environment
  NCCLCHECKGOTO(ncclInit(), res, fail);

  // 输出 NCCL 版本和 git branch+commit，详细日志所有 rank 用 info 级别日志输出，非详细日志 rank0 用 version 级别日志输出
  if (ncclDebugLevel > NCCL_LOG_WARN || (ncclDebugLevel != NCCL_LOG_NONE && myrank == 0)) {
    static std::once_flag once;
    std::call_once(once, showVersion);
  }
  // CUDA 采用惰性初始化（lazy initialization）策略，强制初始化 CUDA 运行时和当前设备的 context
  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), res, fail);

  // 断言 newcomm 和 config 指针不应该是 NULL
  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  NCCLCHECKGOTO(PtrCheck(config, "CommInitRank", "config"), res, fail);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  // ncclCalloc 组合 malloc+memset，分配 1 个 comm 类型的内存空间
  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), res, fail);
  // 分配 CUDA pinned + GPU 可映射的主机内存，并清 0。GPU kernel 执行 collective 操作时检查这个标志，如果发现被设置就提前退出。
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), res, fail);
  // 分配引用计数器，追踪有多少个 comm 共享这个 abort 机制（split/shrink 时会共享）
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), res, fail);
  // 在 communicator 结构体的头部和尾部各放一个 magic number，后续 CommCheck 会检查这两个值是否还是 NCCL_MAGIC，如果变了，说明结构体被内存越界写坏了
  comm->startMagic = comm->endMagic = NCCL_MAGIC; // Used to detect comm corruption.
  *comm->abortFlagRefCount = 1;
  // 把用户传入的 ncclConfig_t 解析到 comm 内部字段。用户通过 ncclCommInitRankConfig 传入的配置（如 NCCL_CONFIG_SPLIT_SHARED 等选项）在这里生效。
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  /*
  communicator 的初始化是异步的（后面第 2534 行会启动一个 job）。
  ncclInProgress 表示"还在初始化中"，用户调用 ncclCommInitRank 返回后，后台线程仍在工作。初始化成功后状态会变为 ncclSuccess。
  其他 API 调用时（如 ncclAllReduce）会先调用 ncclCommEnsureReady(comm)，如果发现 initState == ncclInProgress 就会等待初始化完成。
  */
  /* start with ncclInProgress and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInProgress;
  *newcomm = comm;

  // 通过 new 创建初始化 ncclCommInitRankAsyncJob 对象并赋给 job。job = new (std::nothrow) ncclCommInitRankAsyncJob{};
  NEW_NOTHROW_GOTO(job, ncclCommInitRankAsyncJob, res, fail);
  job->nId = nId;
  job->comm = comm;
  job->nranks = nranks;
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  // 保存调用方函数名，例如："ncclCommInitRank"
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", funcName);
  /*
  用户传入的 commId 是栈上的变量，并且 ncclUniqueId 内部全是 char，对齐要求只有 1 字节，所以起始地址也对齐到 1 字节。
  NCCL 后续要把这个地址强转成 ncclBootstrapHandle* 来用，内部 magic 字段是 uint64_t，起始地址对齐到 8 字节，强转会崩溃。
  ncclCalloc 中的 malloc 重新分配起始地址对齐到 max_align_t(一般是16) 的堆内存，然后 memcpy 把原始数据逐字节复制到这个对齐的地址。
  */
  // need to copy the commIds to allow async commInit and to avoid alignement issues when casting from
  // ncclUNiqueId and ncclBootstrapHandle
  // ncclUniqueIds and ncclBootstrapHandle don't have the same alignment requirements.
  // Therefore the array of Ids coming from the user might not be properly aligned to be cast into a
  // ncclBootstrapHandle
  // copying into allocated memory guarantees that the memory is properly aligned for any objects, removing that issue
  NCCLCHECKGOTO(ncclCalloc(&job->commId, nId), res, fail);
  memcpy(job->commId, commId, nId * NCCL_UNIQUE_ID_BYTES);

  commIdEnv = ncclGetEnv("NCCL_COMM_ID");
  /*
  如果用户设置了 NCCL_COMM_ID，说明走"环境变量 bootstrap"模式：
    只有 rank 0 需要创建 bootstrap root 服务
    其他 rank 通过 commId 连接
  */
  // 不一般不走这条路径，在 rank=0 获取 UniqueId 时就启动了 bootstrapRoot detached 异步线程
  if (commIdEnv && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commIdEnv);
    // 环境变量模式只支持一个 root，nId > 1 时降级为 nId = 1
    if (nId > 1) {
      INFO(NCCL_INIT | NCCL_ENV, "NCCL_COMM_ID cannot be used with more than one ncclUniqueId");
      job->nId = 1;
    }
    // start the bootstrap root before bootstrapping, use only the first handle
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&job->commId[0], true), res, fail);
  }
  // 根据 launchedJob 的值可以确定 job 的所有权在当前线程函数还是 async 系统，前者 delete job，后者由 ncclCommInitJobFree 释放。
  launchedJob = true;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res,
                fail);

exit:
  return ncclGroupErrCheck(res);
fail:
  if (job && !launchedJob) ncclCommInitJobFree(job);
  if (comm) {
    free(comm->abortFlag);
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);
    free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

// 该宏声明函数对外可见，并且有 p 开头的别名，函数原名弱引用可被插件同名符号覆盖
NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  // 加载 NCCL 配置文件到进程环境、加载环境插件
  NCCLCHECK(ncclInitEnv());
  // 启动 NVTX 范围
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)
  // 初始化 CUDA 动态库
  /*
    1.检查 CUDA 驱动版本
    2.加载 NCCL 需要的 CUDA API 为统一的 pfn_* 函数指针
    3.cuMem APIs 支持情况以及版本 bug workaround
  */
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  int cudaDev;
  // 为 NCCL 配置提供初始值，都是未定义
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  CUDACHECK(cudaGetDevice(&cudaDev));

  // 核心初始化流程
  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));

  // 添加 NVTX 范围
  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
                          NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int* gpuFlags = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  int oldDev = 0;

  NVTX3_RANGE(NcclNvtxParamsCommInitAll);

  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUDACHECKGOTO(cudaGetDeviceCount(&totalnDev), ret, fail);
  if (devlist) {
    NCCLCHECKGOTO(ncclCalloc(&gpuFlags, totalnDev), ret, fail);
    for (int i = 0; i < ndev; ++i) {
      /* invalid device check. */
      if (devlist[i] < 0 || devlist[i] >= totalnDev) {
        WARN("Invalid device %d (totalnDev=%d)", devlist[i], totalnDev);
        ret = ncclInvalidArgument;
        goto fail;
      }

      /* duplicate device check. */
      if (ncclParamMultiRankGpuEnable() == 0 && gpuFlags[devlist[i]] != 0) {
        ret = ncclInvalidUsage;
        goto fail;
      }

      gpuFlags[devlist[i]] = 1;
    }
    free(gpuFlags);
    gpuFlags = nullptr;
  }

  ncclUniqueId uniqueId;
  NCCLCHECKGOTO(ncclGetUniqueId(&uniqueId), ret, fail);
  NCCLCHECKGOTO(ncclGroupStartInternal(), ret, fail);
  for (int i = 0; i < ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    int dev = devlist ? devlist[i] : i;
    CUDACHECKGOTO(cudaSetDevice(dev), ret, fail);
    ncclCommInitRankDev(comms + i, ndev, 1, &uniqueId, i, dev, &config, __func__);
  }
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, fail);

  NVTX3_RANGE_ADD_PAYLOAD(CommInitAll, NcclNvtxParamsCommInitAllSchema, NVTX3_PAYLOAD(comms[0]->commHash, ndev));

exit:
  (void)cudaSetDevice(oldDev);
  free(gpuFlags);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  COMPILER_ATOMIC_STORE(&comm->asyncResult, nextState, std::memory_order_release);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank,
         ncclConfig_t* config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank,
                                    ncclConfig_t* config) {
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t* internalConfigPtr = NULL;

  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRankConfig);

  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL) internalConfigPtr = &internalConfig;
  else internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, internalConfigPtr, __func__), ret,
                fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void)ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankConfig, NcclNvtxParamsCommInitRankSchema,
                            NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void)ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitRankScalable, ncclComm_t* newcomm, int nranks, int myrank, int nId,
         ncclUniqueId* commId, ncclConfig_t* config);
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId,
                                      ncclConfig_t* config) {
  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRankScalable);

  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t* internalConfigPtr = NULL;
  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL) internalConfigPtr = &internalConfig;
  else internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, nId, commId, myrank, cudaDev, internalConfigPtr, __func__), ret,
                fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void)ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankScalable, NcclNvtxParamsCommInitRankSchema,
                            NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void)ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  TRACE(NCCL_DESTROY, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag,
        comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->hostStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync hostStream error %d", comm, comm->rank, ret);
    }
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync deviceStream error %d", comm, comm->rank, ret);
    }

    NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, true), ret, fail);
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);
    // And keep polling until all graphs referencing us die.
    while (comm->localPersistentRefs != 0) {
      NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
    }
    while (!ncclIntruQueueEmpty(&comm->legacyRegCleanupQueue)) {
      struct ncclCommCallback* cb = ncclIntruQueueDequeue(&comm->legacyRegCleanupQueue);
      if (cb->fn(comm, cb) != ncclSuccess) {
        WARN("Legacy IPC cleanup callback failed comm %p (rank = %d) cb %p", comm, comm->rank, cb);
      }
    }
  }

  if ((ret = ncclProxyStop(comm)) != ncclSuccess) {
    WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, ret);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->finalize(comm->tunerContext));
    NCCLCHECK(ncclTunerPluginUnload(comm));
  }
  NCCLCHECK(commFree(comm));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommFinalize);

  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob* job = NULL;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  comm->finalizeCalled = true;
  /* launch async thread to finalize comm. */
  NEW_NOTHROW_GOTO(job, ncclCommFinalizeAsyncJob, ret, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commDestroySync, nullptr, ncclCommFinalizeAsyncJobFree,
                                comm),
                ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommFinalize, NcclNvtxParamsCommFinalizeSchema, NVTX3_PAYLOAD(comm->commHash));
  }
  return ret;
fail:
  if (comm && !comm->config.blocking) (void)ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (comm->intraComm0 != NULL) {
    int curRankCnt;
    int curRank; /* Debug info */
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int* finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = COMPILER_ATOMIC_ADD_FETCH(finalizeRankCnt, 1, std::memory_order_acq_rel);
    if (curRankCnt == intraRanks) {
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      /* this is  the last call to ncclCommDestroy/Abort, we need to make sure all comms
       * in the process have been finalized before we free local resources. */
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (curIntraComm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob job;
          job.comm = curIntraComm;
          /* every comm aborts, commDestroySync should not be blocked. */
          if ((ret = commDestroySync((struct ncclAsyncJob*)&job)) != ncclSuccess) {
            WARN("commReclaim: comm %p (rank = %d) in commDestroySync, error %d", curIntraComm, curRank, ret);
          }
        }
      }

      /* free local resources. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          // We pass a freed pointer, but we don't dereference; we merely print its value, so it's OK.
          // coverity[pass_freed_arg]
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL) {
    NCCL_NVTX3_FUNC_RANGE;
    return ncclSuccess;
  }

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob* job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_FUNC_WITH_PARAMS(CommDestroy, NcclNvtxParamsCommInitRank, NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_DESTROY, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);
  NCCLCHECK(ncclGroupStartInternal());
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));
  NEW_NOTHROW_GOTO(job, ncclCommFinalizeAsyncJob, res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, nullptr, ncclCommFinalizeAsyncJobFree, comm),
                res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

static ncclResult_t setCommAbortFlags(ncclComm_t comm, int value) {
  // Set abort flags
  uint32_t uval = static_cast<uint32_t>(value);
  if (comm->childAbortFlag != nullptr) {
    COMPILER_ATOMIC_STORE(comm->childAbortFlag, uval, std::memory_order_release);
    COMPILER_ATOMIC_STORE(comm->childAbortFlagDev, uval, std::memory_order_release);
  }
  COMPILER_ATOMIC_STORE(comm->abortFlag, uval, std::memory_order_release);
  COMPILER_ATOMIC_STORE(comm->abortFlagDev, uval, std::memory_order_release);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRevoke, ncclComm_t comm, int revokeFlags);
struct ncclCommRevokeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static ncclResult_t commRevokeAsync(struct ncclAsyncJob* job_) {
  struct ncclCommRevokeAsyncJob* job = (struct ncclCommRevokeAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  NCCLCHECKGOTO(PtrCheck(comm, "CommRevokeAsync", "comm"), res, exit);
  INFO(NCCL_DESTROY, "CommRevokeAsync START comm %p rank %d nRanks %d nNodes %d localRank %d cudaDev %d", comm,
       comm->rank, comm->nRanks, comm->nNodes, comm->localRank, comm->cudaDev);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->hostStream), res, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
  NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/true), res, exit);
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), res, exit);
  {
    ncclResult_t _tmpret = ncclSuccess;
    if ((_tmpret = ncclProxyStop(comm)) != ncclSuccess) {
      WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, _tmpret);
    }
    if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread.joinable()) {
      comm->proxyState->thread.join();
      if (comm->proxyState->threadUDS.joinable()) {
        // UDS support
        comm->proxyState->threadUDS.join();
      }
    }
  }
  NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
exit:
  (void)ncclCommSetAsyncError(comm, res);
  INFO(NCCL_DESTROY, "CommRevokeAsync END comm %p result %d", comm, res);
  return res;
}

ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommRevoke);

  if (comm == NULL) {
    return ncclSuccess;
  }
  // For now only NCCL_REVOKE_DEFAULT (0) is supported
  if (revokeFlags != NCCL_REVOKE_DEFAULT) {
    return ncclInvalidArgument;
  }
  // Disallow revoke if destroy/finalize in progress
  if (comm->destroyFlag || comm->finalizeCalled) {
    return ncclInvalidArgument;
  }
  // Disallow revoke if revoke in progress
  if (comm->revokedFlag) {
    return ncclInvalidArgument;
  }
  INFO(NCCL_DESTROY, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Revoke START", comm, comm->rank, comm->nRanks,
       comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());
  (void)setCommAbortFlags(comm, 1);
  comm->revokedFlag = 1;
  (void)ncclCommEnsureReady(comm);
  comm->finalizeCalled = true;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommRevokeAsyncJob* job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommRevoke, NcclNvtxParamsCommInitRankSchema,
                          NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NEW_NOTHROW_GOTO(job, ncclCommRevokeAsyncJob, res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commRevokeAsync, nullptr, ncclCommFinalizeAsyncJobFree,
                                comm),
                res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &res));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommRevoke, NcclNvtxParamsCommInitRankSchema,
                            NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));
  }
  INFO(NCCL_DESTROY, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Revoke COMPLETE, result %d", comm, rank, nranks,
       cudaDev, comm->busId, res);
  return res;
fail:
  if (comm && !comm->config.blocking) (void)ncclCommSetAsyncError(comm, res);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommAbort);

  if (comm == NULL) {
    return ncclSuccess;
  }

  INFO(NCCL_DESTROY, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Abort START", comm, comm->rank, comm->nRanks,
       comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());
  // Ask anything that might still be running on the device to quit
  NCCLCHECK(setCommAbortFlags(comm, 1));
  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  (void)ncclCommEnsureReady(comm);

  // once the comm is ready, we can access ranks etc
  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob* job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommAbort, NcclNvtxParamsCommInitRankSchema,
                          NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NEW_NOTHROW_GOTO(job, ncclCommFinalizeAsyncJob, res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, nullptr, ncclCommFinalizeAsyncJobFree, comm),
                res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

static void childCommCleanupJob(void* job) {
  struct ncclCommInitRankAsyncJob* initJob = (struct ncclCommInitRankAsyncJob*)job;
  if (initJob->excludeRanksList) free(initJob->excludeRanksList);
  delete initJob;
}

// initializing a child communicator (for both split and shrink)
static ncclResult_t ncclCommInitChildComm(ncclComm_t comm, ncclComm_t* newcomm, bool isShrink, int flags, int color,
                                          int key, int* excludeRanksList, int excludeRanksCount, ncclConfig_t* config,
                                          const char* caller) {
  struct ncclCommInitRankAsyncJob* job = NULL;
  struct ncclComm* childComm = NCCL_COMM_NULL;
  ncclResult_t res = ncclSuccess;

  int oldDev;
  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(CommCheck(comm, caller, "comm"), res, exit);
  NCCLCHECKGOTO(PtrCheck(newcomm, caller, "newcomm"), res, exit);
  if (isShrink) {
    NCCLCHECKGOTO(PtrCheck(excludeRanksList, caller, "excludeRanksList"), res, exit);
    NCCLCHECKGOTO(excludeRanksCount > 0 ? ncclSuccess : ncclInvalidArgument, res, exit);
    // excludeRanksList may not be sorted, need to sort it
    qsort(excludeRanksList, excludeRanksCount, sizeof(int), compareInts);
    // ranks in excludeRanksList should not call into this function
    NCCLCHECKGOTO(bsearch(&comm->rank, excludeRanksList, excludeRanksCount, sizeof(int), compareInts) ?
                    ncclInvalidArgument :
                    ncclSuccess,
                  res, exit);
  }
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);

  /* *newcomm should be NCCL_COMM_NULL until comm split fully complete. */
  *newcomm = NCCL_COMM_NULL;
  if (!isShrink && color == NCCL_SPLIT_NOCOLOR) {
    INFO(NCCL_INIT, "Rank %d has color with NCCL_SPLIT_NOCOLOR, not creating a new communicator", comm->rank);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&childComm, 1), res, fail);
    childComm->startMagic = childComm->endMagic = NCCL_MAGIC;

    // Set the shareResource field, this is used throughout the init and must be reset every time.
    // Never share resources if the parent communicator has been revoked.
    // If we shrink, we only reuse resources in default mode.
    comm->shareResources =
      !comm->revokedFlag &&
      (isShrink ? (!(flags & NCCL_SHRINK_ABORT) && comm->config.shrinkShare) : comm->config.splitShare);
    if (comm->shareResources) {
      childComm->abortFlag = comm->abortFlag;
      childComm->abortFlagDev = comm->abortFlagDev;
      childComm->abortFlagRefCount = comm->abortFlagRefCount;
      comm->childAbortFlag = NULL;
      ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
    } else {
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlag, 1), res, fail);
      NCCLCHECKGOTO(ncclCudaHostCalloc(&childComm->abortFlagDev, 1), res, fail);
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlagRefCount, 1), res, fail);
      /* temporarily used to abort everything during child comm init. */
      comm->childAbortFlag = childComm->abortFlag;
      comm->childAbortFlagDev = childComm->abortFlagDev;
      *childComm->abortFlagRefCount = 1;
    }
    if (config == NULL) {
      NCCLCHECKGOTO(copyCommConfig(childComm, comm), res, fail);
    } else {
      NCCLCHECKGOTO(parseCommConfig(childComm, config), res, fail);
    }

    /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
    childComm->initState = ncclInternalError;
  }

  NEW_NOTHROW_GOTO(job, ncclCommInitRankAsyncJob, res, fail);
  job->comm = childComm;
  job->newcomm = newcomm;
  job->parent = comm;
  job->color = color;
  job->key = key;
  if (excludeRanksList) {
    // need to copy the list of ranks to exclude because the job is async
    job->excludeRanksCount = excludeRanksCount;
    NCCLCHECKGOTO(ncclCalloc(&job->excludeRanksList, excludeRanksCount), res, fail);
    memcpy(job->excludeRanksList, excludeRanksList, excludeRanksCount * sizeof(int));
  } else {
    // each split has to lead to a unique comm, so increment the childCount
    job->childCount = ++comm->childCount;
    job->excludeRanksList = NULL;
  }
  job->cudaDev = comm->cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", caller);
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, /*undo=*/nullptr,
                                /*destructor=*/childCommCleanupJob, comm),
                res, fail);

exit:
  (void)cudaSetDevice(oldDev);
  return res;
fail:
  if (childComm) {
    if (!comm->shareResources) {
      if (childComm->abortFlag) free(childComm->abortFlag);
      if (childComm->abortFlagDev) ncclCudaHostFree(childComm->abortFlagDev);
      if (childComm->abortFlagRefCount) free(childComm->abortFlagRefCount);
    }
    free(childComm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommShrink, ncclComm_t comm, int* excludeRanksList, int excludeRanksCount,
         ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm,
                            ncclConfig_t* config, int shrinkFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommShrink)
  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  // Handle error mode by setting abort flags and waiting for kernels to complete and unset the flags
  // to avoid bootstrap issues
  if (shrinkFlags & NCCL_SHRINK_ABORT) {
    NCCLCHECKGOTO(setCommAbortFlags(comm, 1), res, exit);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
    NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
  }
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/true, shrinkFlags, /*color=*/0, /*key=*/comm->rank,
                                      excludeRanksList, excludeRanksCount, config, __func__),
                res, exit);

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());

  if (newcomm && *newcomm) {
    NVTX3_RANGE_ADD_PAYLOAD(CommShrink, NcclNvtxParamsCommShrinkSchema,
                            NVTX3_PAYLOAD(comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, excludeRanksCount));
  }

  return res;
}

NCCL_API(ncclResult_t, ncclCommGetUniqueId, ncclComm_t comm, ncclUniqueId* uniqueId);
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  NCCLCHECK(PtrCheck(uniqueId, "CommGetUniqueId", "uniqueId"));

  struct ncclBootstrapHandle growHandle;
  NCCLCHECK(bootstrapGetUniqueId(&growHandle, comm));

  // Broadcast the grow handle to boundary ranks (rank 0 and N-1)
  NCCLCHECK(bcastGrowHandle(&growHandle, comm, /*isRoot=*/true));

  memset(uniqueId, 0, sizeof(ncclUniqueId));
  memcpy(uniqueId, &growHandle, sizeof(growHandle));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommGrow, ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank,
         ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm,
                          ncclConfig_t* config) {
  NVTX3_RANGE(NcclNvtxParamsCommGrow)

  if (newcomm == NULL) return ncclInvalidArgument;
  if (nRanks <= 0) {
    WARN("ncclCommGrow: total ranks must be positive, got %d", nRanks);
    return ncclInvalidArgument;
  }
  if (comm && nRanks <= comm->nRanks) {
    WARN("ncclCommGrow: total ranks %d is less than current ranks %d", nRanks, comm->nRanks);
    return ncclInvalidArgument;
  }

  ncclResult_t res = ncclSuccess;
  bool isExistingRank = (comm != NULL);
  struct ncclCommInitRankAsyncJob* job = NULL;
  ncclComm_t newComm = NULL;
  struct ncclBootstrapHandle recvHandle;

  *newcomm = NULL; // Initialize output parameter early in case of early errors

  NCCLCHECK(ncclGroupStartInternal());

  if (isExistingRank) {
    NCCLCHECKGOTO(CommCheck(comm, __func__, "comm"), res, exit);
    NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
    if (rank != -1) {
      WARN("ncclCommGrow: existing ranks must pass rank=-1, got %d", rank);
      res = ncclInvalidArgument;
      goto exit;
    }

    // each grow/shrink/split has to lead to a unique comm->magic value, increment before using to be
    // consistent with ncclCommInitChildComm
    ++comm->childCount;

    // Only boundary ranks (0 and N-1) receive grow handle from coordinator
    if ((comm->nRanks > 1) && (comm->rank == 0 || comm->rank == comm->nRanks - 1)) {
      NCCLCHECKGOTO(bcastGrowHandle(&recvHandle, comm, /*isRoot=*/false), res, exit);
      // verify the magic is the same as the one computed by the root
      if (recvHandle.magic != hashCombine(comm->magic, comm->childCount)) {
        WARN("ncclCommGrow: magic mismatch computed by the root, got %lx expected %lx", recvHandle.magic,
             hashCombine(comm->magic, comm->childCount));
        res = ncclInvalidArgument;
        goto exit;
      }
      uniqueId = (const ncclUniqueId*)&recvHandle;
    }
  } else {
    // New ranks: validate parameters
    if (rank < 0) {
      WARN("ncclCommGrow: new ranks must pass valid rank >= 0, got %d", rank);
      res = ncclInvalidArgument;
      goto exit;
    }
    if (uniqueId == NULL) {
      WARN("ncclCommGrow: new ranks must pass non-NULL uniqueId");
      res = ncclInvalidArgument;
      goto exit;
    }
    if (rank >= nRanks) {
      WARN("ncclCommGrow: new rank %d exceeds total ranks %d", rank, nRanks);
      res = ncclInvalidArgument;
      goto exit;
    }

    // Initialize NCCL and CUDA for new ranks (same sequence as ncclCommInitRank)
    NCCLCHECKGOTO(ncclInitEnv(), res, exit); // Environment plugins
    (void)ncclCudaLibraryInit(); // CUDA driver and dlsym hooks
    NCCLCHECKGOTO(ncclInit(), res, exit); // Bootstrap network, CPU stack, GDR
    if (ncclDebugLevel > NCCL_LOG_WARN || (ncclDebugLevel != NCCL_LOG_NONE && rank == 0)) {
      static std::once_flag once;
      std::call_once(once, showVersion); // Version display
    }
    CUDACHECKGOTO(cudaFree(NULL), res, exit); // CUDA runtime initialization
  }

  INFO(NCCL_INIT, "ncclCommGrow: %s rank creating new communicator with %d total ranks",
       isExistingRank ? "existing" : "new", nRanks);

  // All ranks allocate a NEW comm structure for the grown communicator
  NCCLCHECKGOTO(ncclCalloc(&newComm, 1), res, fail);
  newComm->startMagic = newComm->endMagic = NCCL_MAGIC;

  // All ranks allocate fresh resources for grown communicator
  NCCLCHECKGOTO(ncclCalloc(&newComm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&newComm->abortFlagDev, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&newComm->abortFlagRefCount, 1), res, fail);
  *newComm->abortFlagRefCount = 1;

  // Configure the new communicator
  if (isExistingRank && config == NULL) {
    NCCLCHECKGOTO(copyCommConfig(newComm, comm), res, fail);
  } else {
    NCCLCHECKGOTO(parseCommConfig(newComm, config), res, fail);
  }

  newComm->initState = ncclInProgress;
  *newcomm = newComm;

  NEW_NOTHROW_GOTO(job, ncclCommInitRankAsyncJob, res, fail);
  job->nId = (uniqueId != NULL) ? 1 : 0;
  if (job->nId == 1) {
    NCCLCHECKGOTO(ncclCalloc(&job->commId, 1), res, fail);
  }
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "ncclCommGrow");

  if (isExistingRank) {
    // Existing rank: new comm, parent is old comm
    job->parent = comm;
    job->cudaDev = comm->cudaDev;
    job->myrank = comm->rank; // Keep same rank in expanded comm
  } else {
    // New rank: new comm, no parent
    int device;
    CUDACHECK(cudaGetDevice(&device));
    job->parent = NULL;
    job->cudaDev = device;
    job->myrank = rank;
  }

  // All ranks use the newly allocated comm
  job->comm = newComm;

  if (job->nId == 1) memcpy(job->commId, uniqueId, sizeof(ncclUniqueId));
  job->nranks = nRanks;
  job->newcomm = newcomm;
  job->isGrow = 1;
  job->color = 0;
  job->key = job->myrank;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, childCommCleanupJob, newComm),
                res, fail);

exit:
  if (*newcomm) {
    uint64_t parentHash = isExistingRank ? comm->commHash : 0;
    NVTX3_RANGE_ADD_PAYLOAD(CommGrow, NcclNvtxParamsCommGrowSchema,
                            NVTX3_PAYLOAD((*newcomm)->commHash, parentHash, nRanks, (*newcomm)->rank,
                                          (*newcomm)->cudaDev));
  }
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;

fail:
  if (job) {
    if (job->commId) {
      free(job->commId);
      job->commId = NULL;
    }
    delete job;
  }
  // Clean up newly allocated comm on failure
  if (newComm) {
    // Only free abort resources if we allocated them (not shared)
    if (!isExistingRank || !comm->shareResources) {
      free(newComm->abortFlag);
      if (newComm->abortFlagDev) (void)ncclCudaHostFree((void*)newComm->abortFlagDev);
      free(newComm->abortFlagRefCount);
    }
    free(newComm);
    if (newcomm) *newcomm = NULL;
  }
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t* newcomm, ncclConfig_t* config) {
  NVTX3_RANGE(NcclNvtxParamsCommSplit)

  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/false, /*shrink mode=*/NCCL_SHRINK_DEFAULT, color,
                                      key, NULL, 0, config, __func__),
                res, exit);

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());

  if (newcomm && *newcomm) {
    NVTX3_RANGE_ADD_PAYLOAD(CommSplit, NcclNvtxParamsCommSplitSchema,
                            NVTX3_PAYLOAD((*newcomm)->commHash, comm->commHash, comm->nRanks, comm->rank, comm->cudaDev,
                                          color, key));
  }

  return res;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
  case ncclSuccess:
    return "no error";
  case ncclUnhandledCudaError:
    return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
  case ncclSystemError:
    return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
  case ncclInternalError:
    return "internal error - please report this issue to the NCCL developers";
  case ncclInvalidArgument:
    return "invalid argument (run with NCCL_DEBUG=WARN for details)";
  case ncclInvalidUsage:
    return "invalid usage (run with NCCL_DEBUG=WARN for details)";
  case ncclRemoteError:
    return "remote process exited or there was a network error";
  case ncclInProgress:
    return "NCCL operation in progress";
  case ncclTimeout:
    return "timeout";
  default:
    return "unknown result code";
  }
}

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);
const char* ncclGetLastError(ncclComm_t comm) {
  return ncclLastError;
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError) {
  NCCLCHECK(CommCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = COMPILER_ATOMIC_LOAD(&comm->asyncResult, std::memory_order_acquire);
  if (*asyncError == ncclSuccess && comm->proxyState) {
    *asyncError = COMPILER_ATOMIC_LOAD(&comm->proxyState->asyncResult, std::memory_order_acquire);
  }

  /* Check gin status */
  if (*asyncError == ncclSuccess && comm->sharedRes && comm->sharedRes->ginState.connected) {
    struct ncclGinState* ginState = &comm->sharedRes->ginState;
    // Gin progress thread status
    if (ginState->needsProxyProgress) {
      *asyncError = COMPILER_ATOMIC_LOAD(&comm->sharedRes->ginState.asyncResult, std::memory_order_acquire);
    }
    // Gin side errors, also works when we have no GIN progress thread.
    if (*asyncError == ncclSuccess) {
      bool ginError;
      NCCLCHECK(ncclGinQueryLastError(&comm->sharedRes->ginState, &ginError));
      if (ginError) {
        WARN("GIN Error detected");
        *asyncError = ncclRemoteError;
      }
    }
  }

  /* if there is linked group job, we should complete it. */
  if (*asyncError == ncclSuccess && comm->groupJob) {
    NCCLCHECK(ncclGroupJobComplete(comm->groupJob));
    comm->groupJob = NULL;
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}
