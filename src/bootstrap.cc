/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include "proxy.h"
#include "param.h"
#include "ras.h"
#include <mutex>
#include "os.h"
#include <thread>
#include <chrono>

#define BOOTSTRAP_N_CHECK_ABORT 10000
#define BOOTSTRAP_TAG_CONNECT (0x1 << 31)
#define BOOTSTRAP_TAG_ALLGATHER (0x1 << 30)
#define BOOTSTRAP_TAG_COMMSPLIT (0x1 << 29)
#define BOOTSTRAP_TAG_INTRANODE_ALLGATHER (0x1 << 28)
#define BOOTSTRAP_TAG_GROW_BOUNDARY (0x1 << 27)

#define BOOTSTRAP_INIT_TIME_CREATE 0
#define BOOTSTRAP_INIT_TIME_SEND 1
#define BOOTSTRAP_INIT_TIME_RECV 2
#define BOOTSTRAP_INIT_TIME_RING 3
#define BOOTSTRAP_INIT_TIME_TOTAL 4
#define BOOTSTRAP_INIT_TIME_DELAY 5
#define BOOTSTRAP_INIT_TIME_N 6
#define BOOTSTRAP_INIT_ROOT_WAIT 0
#define BOOTSTRAP_INIT_ROOT_SEND 1
#define BOOTSTRAP_INIT_ROOT_RECV 2
#define BOOTSTRAP_INIT_ROOT_N 3
#define BOOTSTRAP_PROF_OPEN(time) \
  do { \
    time = clockNano(); \
  } while (0)
#define BOOTSTRAP_PROF_CLOSE(time) \
  do { \
    time = clockNano() - time; \
  } while (0)

#define BOOTSTRAP_PID(i, n) (((i) + (n)) % (n))

// 计算当前 root 的起始 rank 索引
// returns the first rank associated to the root. must have root >=0
// if root >= n_roots, it does NOT assume periodicity
static int firstRankFromRoot(int root, int n_ranks, int nRoots, int offset) {
  if (root == -1) return 0;
  // only distribute the n_ranks - offset on the roots
  n_ranks -= offset;
  return offset + root * (n_ranks / nRoots) + std::min(root, n_ranks % nRoots);
}

/*
计算当前 rank 映射到的负责撮合它的 root 编号
offset 表示存量 rank 数偏移
nRoots 表示新增 rank 的 root 数量
nRanks 表示所有 rank 数量
*/
// returns the root of a rank, must have rank >=0
// if rank >= n_ranks, it does NOT assume periodicity
static int rootIdFromRank(int rank, int nRanks, int nRoots, int offset) {
  // 排除存量 rank
  // ranks < offset have no root (id = -1), ranks above the offset will get assigned to their respective root
  if (nRoots == 0 || rank < offset) return -1;
  nRanks -= offset;
  rank -= offset;
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  int D = rmr * (rpr + 1);
  if (rank < D) return rank / (rpr + 1);
  else return (rank - D) / rpr + rmr;
}

// 计算当前 root 包含几个 ranks
// return the number of child for a root, root will be periodized
static int nRankFromRoot(int root, int nRanks, int nRoots, int offset) {
  if (root == -1) return 0;
  // grow 场景去掉旧 ranks，但要包含旧通信域最后一个 rank
  nRanks -= offset;
  int ir = BOOTSTRAP_PID(root, nRoots);  // root 索引
  // 多个 root 情况下且 ranks 不能被 roots 均分，前多少个 root 分到多一个 rank
  int rmr = nRanks % nRoots; // rank mod root
  // 每个 root 分到的基础 ranks 数
  int rpr = nRanks / nRoots; // rank per root
  // 如果当前 root 是多分一个 rank 的前几个 roots 范围内，则 + 1
  return rpr + ((ir < rmr) ? 1 : 0);
}

// 计算 rank 的 localId
// return the local id of a given rank for a given root
// root will be periodize, rank will not
static int localIdFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  // any rank for root -1 has a local id that is the rank id
  if (root == -1) return rank;
  int ir = BOOTSTRAP_PID(root, nRoots);
  // 当前 rank - 当前 root 的起始 rank 索引 = 当前人 root 下 rank 的 localId
  return rank - firstRankFromRoot(ir, nRanks, nRoots, offset);
}

// Check if the given rank is the first rank from the root
static int isFirstFromRoot(int rank, int root, int nRanks, int nRoots, int offset) {
  return (rank == firstRankFromRoot(root, nRanks, nRoots, offset));
}

struct bootstrapRootArgs {
  struct ncclSocket* listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
static union ncclSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
static std::mutex bootstrapNetMutex;

// 控制 bootstrap（引导/带外控制通道）走哪条传输后端
NCCL_PARAM(BootstrapNetEnable, "OOB_NET_ENABLE", 0);

// 负责选出一个本进程后续 bootstrap 通信用的本地网卡地址，并缓存到全局变量
ncclResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (bootstrapNetInitDone == 0) {
      // 获取指定 root/bootstrap 服务地址，类似：192.168.1.10:12345、hostname:12345、[fe80::1%eth0]:12345
      const char* env = ncclGetEnv("NCCL_COMM_ID");
      int nIfs = 0;
      if (env) {
        union ncclSocketAddress remoteAddr;
        // 将 <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port> 字符串解析为 ncclSocketAddress 结构体数据类型
        if (ncclSocketGetAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          return ncclInvalidArgument;
        }
        // 遍历本机所有网络接口，找到与目标远程地址处于同一子网的本地接口
        NCCLCHECK(ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE,
                                               &nIfs));
        if (nIfs <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          return ncclSystemError;
        }
      } else {
        // 用户没有显式指定 NCCL_COMM_ID，NCCL 自动选择本地 bootstrap 网卡
        NCCLCHECK(ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1, &nIfs));
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          return ncclInvalidUsage;
        }
      }
      char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
      snprintf(line, sizeof(line), " %s:", bootstrapNetIfName);
      // 将找到的网口格式化输出形如：
      ncclSocketToString(&bootstrapNetIfAddr, line + strlen(line));
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using%s", line);
      bootstrapNetInitDone = 1;
    }
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t {
  findSubnetIf = -1,
  dontCareIf = -2
};

// check abort function
static ncclResult_t checkAbort(volatile uint32_t* flag, int* cntr) {
  if ((*cntr % BOOTSTRAP_N_CHECK_ABORT) == 0) {
    if (flag && COMPILER_ATOMIC_LOAD(flag, std::memory_order_acquire)) {
      TRACE(NCCL_BOOTSTRAP, "bootstrap: abort called");
      return ncclInternalError;
    }
  }
  *cntr = (*cntr + 1) % BOOTSTRAP_N_CHECK_ABORT;
  return ncclSuccess;
}
// send/recv functions
static ncclResult_t netReg(ncclNet_t* net, void* comm, void* data, int size, void** handle) {
  NCCLCHECK(net->regMr(comm, data, size, NCCL_PTR_HOST, handle));
  return ncclSuccess;
}
static ncclResult_t netDereg(ncclNet_t* net, void* comm, void** handle) {
  NCCLCHECK(net->deregMr(comm, *handle));
  *handle = NULL;
  return ncclSuccess;
}
static ncclResult_t netIsend(ncclNet_t* net, void* sendComm, void* data, int size, void* dataHandle, int tag,
                             void** sendReq, int* done) {
  if (*done) return ncclSuccess;
  if (!*sendReq) {
    NCCLCHECK(net->isend(sendComm, data, (size_t)size, tag, dataHandle, NULL, sendReq));
  }
  if (*sendReq) {
    NCCLCHECK(net->test(*sendReq, done, NULL));
    if (*done) {
      *sendReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netIrecv(ncclNet_t* net, void* recvComm, void* data, int size, void* dataHandle, int tag,
                             void** recvReq, int* done) {
  if (*done) return ncclSuccess;
  if (!*recvReq) {
    size_t size64 = size;
    NCCLCHECK(net->irecv(recvComm, 1, &data, &size64, &tag, &dataHandle, NULL, recvReq));
  }
  if (*recvReq) {
    NCCLCHECK(net->test(*recvReq, done, NULL));
    if (*done) {
      *recvReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netSendRecv(ncclNet_t* net, void* sendComm, void* sendData, int sendSize, void* sendDataHandle,
                                void* recvComm, void* recvData, int recvSize, void* recvDataHandle, int tag,
                                volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  int doneSend = 0, doneRecv = 0;
  void *sendReq = NULL, *recvReq = NULL;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!doneRecv) {
      NCCLCHECK(netIrecv(net, recvComm, recvData, recvSize, recvDataHandle, tag, &recvReq, &doneRecv));
    }
    if (!doneSend) {
      NCCLCHECK(netIsend(net, sendComm, sendData, sendSize, sendDataHandle, tag, &sendReq, &doneSend));
    }
  } while (!doneSend || !doneRecv);
  return ncclSuccess;
}

// Additional socket based functions, first send the size, then send the message
static ncclResult_t socketSend(struct ncclSocket* sock, void* data, int size) {
  // 先发送数据大小
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  // 再发送实际 payload
  if (size > 0) NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}

static ncclResult_t socketRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  // NCCL 的长度前缀协议：先收 4 字节 size，再收 payload；若 recvSize > sizeof(info) 直接判为消息截断报错。这提供了轻量的版本/结构体不匹配检测。
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  // 实际要接收的数据大小可能小于等于 buffer 大小
  int actualSize = std::min(recvSize, size);
  // 接收 payload
  if (actualSize > 0) NCCLCHECK(ncclSocketRecv(sock, data, actualSize));
  return ncclSuccess;
}

static ncclResult_t socketSendRecv(struct ncclSocket* sendSock, void* sendData, int sendSize,
                                   struct ncclSocket* recvSock, void* recvData, int recvSize) {
  int senderRecvSize;
  // 先发送之后要发送的数据大小值，并接收之后要接收的数据大小值
  NCCLCHECK(ncclSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int)));
  // 判断要接收的数据大小是否大于预期要接收的数据大小
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return ncclInternalError;
  }
  // 正式交换数据
  NCCLCHECK(ncclSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, std::min(recvSize, senderRecvSize)));
  return ncclSuccess;
}

static ncclResult_t socketDoubleSendRecv(struct ncclSocketOp ops[4]) {
  // ops synchronously exchange size then asynchronously exchange data in send->recv->send->recv order
  // 先完成一次之后 Ring0 和 Ring1 要通信的数据大小的同步，判断要接收的数据大小是否大于预期要接收的数据大小
  int senderRecvSize1, senderRecvSize2;
  NCCLCHECK(ncclSocketSendRecv(ops[0].sock, &ops[0].size, sizeof(int), ops[1].sock, &senderRecvSize1, sizeof(int)));
  NCCLCHECK(ncclSocketSendRecv(ops[2].sock, &ops[2].size, sizeof(int), ops[3].sock, &senderRecvSize2, sizeof(int)));
  if (senderRecvSize1 > ops[1].size || senderRecvSize2 > ops[3].size) {
    WARN("Message truncated : received %d,%d bytes instead of %d,%d", senderRecvSize1, senderRecvSize2, ops[1].size,
         ops[3].size);
    return ncclInternalError;
  }
  // 截断接收的数据大小
  ops[1].size = std::min(ops[1].size, senderRecvSize1);
  ops[3].size = std::min(ops[3].size, senderRecvSize2);
  // 并发交换 4 路数据
  NCCLCHECK(ncclSocketMultiOp(ops, 4));
  return ncclSuccess;
}

union ringConnectInfo {
  union ncclSocketAddress addr;
  char handle[NCCL_NET_HANDLE_MAXSIZE];
};

struct extInfo {
  int rank;                                  // rank of the process reaching out
  int nranks;                                // total number of ranks
  int iroot;                                 // current root index， 在向第几个 root 报道
  int nroots;                                // total number of roots， 全局 root 总数
  int offset;                                // offset for rank distribution， grow 场景的偏移
  union ncclSocketAddress listenRootAddress; // address of my listenSocket for the root，专供 root 回连、投递撮合结果，一次性，收完即 close
  /*
  环上的长期监听端点，整个通信域生命周期，环上的前驱 rank 来连它。
  注意 connectInfo 是个 union，socket 模式下是 ncclSocketAddress，走网络插件（NCCL_OOB_NET_ENABLE）时是不透明的 handle[NCCL_NET_HANDLE_MAXSIZE]。
  root 从不解析它的内容，只当作黑盒字节搬运——这是很干净的解耦，让 root 逻辑对传输层无感知。
  */
  union ringConnectInfo connectInfo;
};
#define NET_HANDLE(h, rank) ((h) + (rank * NCCL_NET_HANDLE_MAXSIZE))
#define BOOTSTRAP_HANDLE(h, i) ((struct ncclBootstrapHandle*)((char*)h + i * NCCL_UNIQUE_ID_BYTES))

static ncclResult_t rootSend(union ncclSocketAddress* addr, uint64_t magic, union ringConnectInfo* info) {
  ncclResult_t res = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECKGOTO(ncclSocketInit(&sock, addr, magic, ncclSocketTypeBootstrap), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(union ringConnectInfo)), res, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return res;
fail:
  (void)ncclSocketClose(&sock);
  return res;
}

/*
函数签名遵循 POSIX pthread 线程入口函数的标准形式：
  线程入口函数只能接收一个统一类型的参数，所以用 void* 来传递
  POSIX pthread 的线程函数约定要求返回 void*，用来表示线程退出时的返回值
但 std::thread 的构造函数支持任意可调用对象，实际上，如果完全按现代 C++ 风格写，可以写成：
  static void bootstrapRoot(struct bootstrapRootArgs* args);
*/
static void* bootstrapRoot(void* rargs) {
  uint64_t timers[BOOTSTRAP_INIT_ROOT_N] = {0};
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct ncclSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  int iroot = 0, nroots = 0, localId = 0;
  int nrecv = 0, n2send = 0, offset = 0;
  struct extInfo info;
  // 缓存所有 ranks 的 AddressesRoot、connectInfo 数组，后续用 ncclCalloc 分配，天然全零，因此存在隐式契约：槽位全零 ⟺ 该 localId 尚未报到
  union ringConnectInfo* rankInfo = NULL;
  union ncclSocketAddress* rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  // get zeros for comparison
  char zeroHandle[NCCL_NET_HANDLE_MAXSIZE];
  union ncclSocketAddress zeroAddress;
  union ringConnectInfo zeroInfo;
  memset(&zeroAddress, 0, sizeof(union ncclSocketAddress));
  memset(&zeroHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memset(&zeroInfo, 0, sizeof(union ringConnectInfo));
  // ncclOsSetFilesLimit() 把进程 fd 软上限抬到硬上限
  ncclOsSetFilesLimit();

  TRACE(NCCL_BOOTSTRAP, "BEGIN");
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
  /* Receive addresses from all ranks */
  do {
    struct ncclSocket sock;
    // 没传 addr，则清空 sock->addr
    NCCLCHECKGOTO(ncclSocketInit(&sock), res, out);
    // 与 accepted 的某个对端 socket 进行握手（接收 magic 和 type，并进行对比），当然异步情况下不是一次做完
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);
    // 接收 extInfo 数据
    NCCLCHECKGOTO(socketRecv(&sock, &info, sizeof(info)), res, out);
    // 关闭 accepted socket。
    // root 不能通过这条连接回复结果（连接已断），必须反过来主动 connect 到 listenRootAddress
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);

    // 让 root 从第一个报到者那里“学到”整个作业的拓扑参数，并据此规划自己的工作容量
    if (c == 0) {
      /*
      OPEN 记起始时刻，CLOSE 算耗时。root 的生命周期被切成三段：
        WAIT：线程启动 → 第一个 rank 报到，root 空转等待(通常很短)
        RECV：第一个报到 → 收齐 nrecv 个，收集阶段，大头是 rank 侧 stagger 延迟，rank 数超过 UID_STAGGER_THRESHOLD(默认 256)时按 UID_STAGGER_RATE(默认 7000/s)错峰报到，防止 root 被连接风暴打死)
        SEND：收齐后的 drain 循环，补发剩余撮合结果
      */
      BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
      BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_RECV]);
      /*
      两个值得注意的点：
        root 自己不知道自己的编号。info.iroot 是报到 rank 算好的 curr_root,由它告诉 root “你是几号 root”。
        offset 只在 grow(扩容)场景非零:offset = parent->nRanks - 1。语义是：ranks [0, offset) 是老 rank，它们的环上连接已存在或走父通信域的 bootstrap 通道重建，
      不参与本轮 root 撮合;从 rank offset(父通信域最后一个 rank)开始才是本轮参与者。
      */
      nranks = info.nranks;  // 通信域总 rank 数
      iroot = info.iroot;    // 我是第几个 root(所有 root 中的下标)
      nroots = info.nroots;  // root 总数(= ncclCommInitRankConfig 传入的 uniqueId 句柄数 nHandles)
      offset = info.offset;  // 排除在撮合之外的头部 rank 数(grow 场景)
      // n2send 表示当前 root 负责的 ranks 数
      // if the number of root > 1, we will receive one extra info from the first local_id of the next root
      n2send = nRankFromRoot(iroot, nranks, nroots, offset);
      /*
      如果 root 数大于 1 或者 grow 场景，每个 root 会收到下一个 root 中第一个 local_id 的这个额外 info
      */
      // offset>0 automatically means that we need to switch to the multiroot logic
      nrecv = n2send + ((offset > 0 || nroots > 1) ? 1 : 0);
      /*
      rankInfo[localId]：该 rank 的环监听端点(ringConnectInfo,socket 地址或网络插件不透明 handle)，被撮合时作为“next peer 信息”发给前驱
      rankAddressesRoot[localId]，该 rank 的 listenRootAddress(专为 root 回连准备的一次性回复通道)，root rootSend 的投递目标
      */
      NCCLCHECKGOTO(ncclCalloc(&rankInfo, nrecv), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nrecv), res, out);
    }

    // 检验后续 ranks 的元信息
    if (nranks != info.nranks || nroots != info.nroots || iroot != info.iroot || offset != info.offset) {
      WARN("Bootstrap Root : mismatch in info from procs, nranks %d vs %d, nroots %d vs %d, iroot %d vs %d, offset %d "
           "vs %d",
           nranks, info.nranks, nroots, info.nroots, iroot, info.iroot, offset, info.offset);
      goto out;
    }

    // 获取接收到的 rank 在当前 root 下的 localId
    localId = localIdFromRoot(info.rank, iroot, nranks, nroots, offset);
    if (localId < 0 || localId >= nrecv) {
      WARN("Bootstrap Root : localId %d is out of range", localId);
      goto out;
    }
    // 目前 rank 的缓存不为空属于异常情况
    if (memcmp(&zeroAddress, &rankAddressesRoot[localId], sizeof(union ncclSocketAddress)) != 0 ||
        memcmp(&zeroInfo, &rankInfo[localId], sizeof(union ringConnectInfo)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }

    /*
    计算当前 localId 的前继索引，要包含在 [0, n2send) 范围内
      1. 先确定前继索引已经收到并缓存过，则直接将当前 rank 的 connectInfo 发送给 前继节点，地址为 rankAddressesRoot[prev]
      2. 前继节点还没到，则先缓存当前 rank 的信息
    */
    // if the previous has already checked in, send the newly received handle, if not save the handle for later
    // if we have more than 1 root, I do not own the previous of local_id = 0
    // if we have prev > n2send, we do not send anything
    // 如果多个 root，则 localId=0 不用管发给前继节点，如果只有一个 root，localId=0 要发给 localId=N-1
    int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
    if (prev >= 0 && prev < n2send &&
        memcmp(&zeroAddress, &rankAddressesRoot[prev], sizeof(union ncclSocketAddress)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &info.connectInfo), res, out);
    } else {
      memcpy(&rankInfo[localId], &info.connectInfo, sizeof(union ringConnectInfo));
    }
    
    // 有可能后继先到，当前 rank 作为前继需要触发接收缓存的后继节点 connectInfo
    // if the next rank has checked in, send the newly received info, if not save the addr for later
    // for nroots >=1, I will always own the information of the next connection
    // if the local_id id must be [0 ; n2send[ otherwise we do not answer
    int next = BOOTSTRAP_PID(localId + 1, nrecv);
    if (localId >= 0 && localId < n2send && memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&info.listenRootAddress, magic, &rankInfo[next]), res, out);
    } else {
      memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
    }
    ++c;
    TRACE(NCCL_BOOTSTRAP, "Received connect from rank %d total %d/%d", info.rank, c, nrecv);
  } while (c < nrecv);
  TRACE(NCCL_BOOTSTRAP, "COLLECTED ALL %d HANDLES", nrecv);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_RECV]);

  // send the remaining info to the ranks who haven't received anything
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  // here we need to send info only to my own local process
  for (int r = 0; r < n2send; ++r) {
    // use nrecv to periodize: if 1 root, we will send the first one to the last one,
    // if >1 roots we will send the additional one we have received
    int next = BOOTSTRAP_PID(r + 1, nrecv);
    if (memcmp(&zeroAddress, &rankAddressesRoot[r], sizeof(union ncclSocketAddress)) != 0 &&
        memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[r], magic, &rankInfo[next]), res, out);
    }
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "Root timings (wait %f, recv %f, send %f)",
        timers[BOOTSTRAP_INIT_ROOT_WAIT] / 1e9, timers[BOOTSTRAP_INIT_ROOT_RECV] / 1e9,
        timers[BOOTSTRAP_INIT_ROOT_SEND] / 1e9);
out:
  if (listenSock != NULL) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankInfo) free(rankInfo);
  if (rankAddressesRoot) free(rankAddressesRoot);
  // 负责释放外部申请的内存
  free(rargs);

  TRACE(NCCL_BOOTSTRAP, "DONE");
  return NULL;
}

// bootstrapCreateRoot 的作用是：在 rank 0 上启动 bootstrap root 服务，作为所有 rank 的"中介"，帮助它们互相发现并建立连接
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;  // 监听 socket
  struct bootstrapRootArgs* args = NULL;  // 传给后台线程的参数
  std::thread thread;

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  // 初始化 socket，用 handle->addr 中的地址和 handle->magic 作为认证，bootstrap 用的 socket 是同步且不用 abort
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  // 开始监听，绑定地址 + 端口，调用 listen()
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  // 获取实际地址，回写 handle->addr（如果用户没指定端口，OS 会分配一个随机端口）
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  // bootstrapRootArgs 一起保存 sock 和 magic
  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;
  // 启动一个线程执行 bootstrapRoot(args)
  // std::thread 不要求特定签名（不像 pthread_create），它接受任何可调用对象。返回值会被 std::thread 忽略
  thread = std::thread(bootstrapRoot, args);
  // 设置线程名
  ncclSetThreadName(thread, "NCCL BootstrapR");
  /*
  分离线程：
    detach 前：
      thread 对象 ───管理───► 线程
      （thread 析构时如果线程还在运行 → std::terminate 崩溃）
    detach 后：
      thread 对象 ──✗───► 线程（断开关联）
      线程在后台独立运行，自己结束后 OS 自动回收资源
      thread 对象析构 → 安全，不影响线程
  */
  /*
  因为本函数创建的 thread 是局部变量，返回后会被析构，线程会被 std::terminate()。
  一般来说要么 return 前执行 join 阻塞等待线程结束；要么 detach 让线程后台执行，线程与 thread 对象断开关联。
  detach 后线程独立运行，bootstrapRoot 函数自己负责清理，
  std::thread 一旦启动成功，args 的所有权就转移给新线程，由 bootstrapRoot 结尾的 free(rargs) 释放。主线程不再碰它。
  */
  thread.detach();
exit:
  return ret;
fail:
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}

ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) {
  // 清空 40 字节 handle
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  // NCCL_COMM_ID 如果设置，表示用户希望跳过 NCCL 自己创建 root 服务的过程，
  // 由外部指定一个固定的 root 地址（例如 <ip>:<port>），rank 0 在该地址监听，其他 rank 直接连过来。
  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    // NCCL_COMM_ID 不支持 grow 多 root 场景
    // If comm is provided (grow operation), NCCL_COMM_ID should not be set
    if (comm) {
      WARN("ncclCommGetUniqueId should not be called when NCCL_COMM_ID is set");
      return ncclInvalidUsage;
    }
    // Normal init: use NCCL_COMM_ID from environment
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    /*
    环境变量字符串解析成 socket 地址，存到 handle->addr。格式支持：
      IPv4：192.168.1.1:12345
      IPv6：[2001:db8::1]:12345
      主机名：hostname:12345
    */
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    // handle->magic 设置为固定的 NCCL_MAGIC，用于后续校验连接双方是否匹配。
    handle->magic = NCCL_MAGIC;
  } else {
    if (comm) {  // grow 场景 (comm != NULL)，用现有 comm 的 magic 和 childCount + 1 做 hash 组合，保证新 handle 和原 comm 有关联但又唯一
      // comm->childCount will be increment in ncclCommGrow for all existing ranks, use +1 here
      handle->magic = hashCombine(comm->magic, comm->childCount + 1);
    } else {
      // 随机生成 magic
      NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    }
    // handle->nRanks：grow 场景下记录当前 communicator 的 rank 数；普通初始化时设为 0。
    handle->nRanks = comm ? comm->nRanks : 0;
    // handle->addr：复制 bootstrapNetIfAddr，也就是 NCCL 选定的本机 bootstrap 网络接口地址。
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    // bootstrapCreateRoot(handle, false)：在本机启动一个 bootstrap root 监听服务，监听地址就是 handle->addr。参数 false 表示不是由环境变量触发的 root，而是 NCCL 自己创建的。
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

ncclResult_t bcastGrowHandle(struct ncclBootstrapHandle* handle, struct ncclComm* parent, bool isRoot) {
  if (!parent || !handle) {
    WARN("bcastGrowHandle: parent comm and handle must be provided");
    return ncclInvalidArgument;
  }

  // Single rank parent already has the handle, no need to broadcast
  if (parent->nRanks == 1) return ncclSuccess;
  if (isRoot) {
    NCCLCHECK(bootstrapSend(parent->bootstrap, 0, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
    NCCLCHECK(bootstrapSend(parent->bootstrap, parent->nRanks - 1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
  } else {
    NCCLCHECK(bootstrapRecv(parent->bootstrap, -1, BOOTSTRAP_TAG_GROW_BOUNDARY, handle,
                            sizeof(struct ncclBootstrapHandle)));
  }

  return ncclSuccess;
}

// 意外连接链表节点，单链表节点，缓存“意外到达”的连接
struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

// 环形连接端点。NCCL 在 bootstrap 阶段把所有 rank 组织成一个逻辑环，用于环形的 allgather 等信息交换。这是个 union，两种后端二选一
struct bootstrapRing_t {
  union {
    //  ncclParamBootstrapNetEnable() 打开时，bootstrap 直接走网络插件，sendComm/recvComm 是插件的连接句柄，外加设备句柄。
    struct {
      void *sendComm, *recvComm;
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;
    } net;
    // 普通 TCP 后端，send/recv 是连向环上下一跳/上一跳的 socket
    struct {
      struct ncclSocket recv;
      struct ncclSocket send;
    } socket;
  };
};

// 本 rank 的被动监听端点。
struct bootstrapListen_t {
  // 用于 mesh 任意 P2P 通信的监听 socket，它的地址就是后面 peerP2pAddresses 里的内容
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P
  // 环的监听端，TCP 后端用 socket，网络后端用 net(comm + handle)。
  union {
    struct {
      int dev;
      void* comm;
      char handle[NCCL_NET_HANDLE_MAXSIZE];
    } net;
    // 用于 ring 的监听 socket
    struct ncclSocket socket; // socket to be used for the ring
  };
};

// bootstrapState 是 bootstrap 层的"会话对象"，挂到 comm->bootstrap，贯穿整个通信域生命期。
struct bootstrapState {
  // 环形拓扑的收发端点(TCP socket 或 net 插件，union 二选一)
  struct bootstrapRing_t ring;
  // 本 rank 的被动监听端点：peerSocket(供 P2P)+ 环监听端
  struct bootstrapListen_t listen;
  /*
  bootstrap 用的网络插件接口，来自 comm->ncclNet，仅当 bootstrap 走网络时用到。
  state->net = comm->ncclNet。它让 bootstrap 自身也能跑在 IB 等 fabric 上(而非只能 TCP)，
  netRingConnect(comm->netContext, state->net, &state->listen, ...) 就用它建环。
  */
  ncclNet_t* net;
  /*
  全 rank 的 proxy UDS 标识(hash) 数组，用于 proxy 间 SCM_RIGHTS fd 传递。
  虽然保存了全部 ranks 的 UDS hash，但 UDS 本质是 AF_UNIX(同内核 IPC)，只有同节点 rank 之间才能真正用它。
  */
  uint64_t* peerProxyAddressesUDS;
  // 全 rank 的 proxy 服务 TCP 地址 数组
  union ncclSocketAddress* peerProxyAddresses;
  // 全 rank 的 bootstrap Mesh P2P TCP 地址 数组
  union ncclSocketAddress* peerP2pAddresses;
  /*
  由于 bootstrap 是全 mesh 异步连接，某 rank 可能先收到一个当前并不在等的 (peer, tag) 连接。
  这类连接被 enqueueUnexpected 挂进链表，之后当真正 bootstrapRecv(peer, tag) 时，
  先 searchUnexpected 在链表里找已接受的连接，命中就摘链复用。销毁时 unexpectedFree 清空，
  若中止时链表非空还会 WARN("Unexpected connections are not empty")。
  */
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};


#define STATE_RING(s, f) (s->ring.f)
#define STATE_LISTEN(s, f) (s->listen.f)

// helper functions
static ncclResult_t createListenSocket(struct ncclComm* comm, uint64_t magic, struct ncclSocket* socket,
                                       union ncclSocketAddress* addr, ncclSocketType type) {
  // 用 bootstrapNetIfAddr（bootstrap 网络接口地址）初始化 socket，端口通常是 0，意为"让内核随便分配一个空闲端口"
  NCCLCHECK(ncclSocketInit(socket, &bootstrapNetIfAddr, magic, type, comm->abortFlag));
  // bind、listen
  NCCLCHECK(ncclSocketListen(socket));
  // 将 socket 中的 addr 写到 addr 中
  NCCLCHECK(ncclSocketGetAddr(socket, addr));
  return ncclSuccess;
}

static ncclResult_t getUDS(uint64_t* peerUDS) {
  uint64_t randId;
  // 生成 uint64_t 随机数
  NCCLCHECK(getRandomData(&randId, sizeof(randId)));
  // 进程 PID + PID namespace 的 hash 与随机数相加
  *peerUDS = getPidHash() + randId;
  return ncclSuccess;
}

#define MAX_OOB_DEVS 16
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev) {
  static int devOOB = -1;
  if (devOOB < 0) {
    std::lock_guard<std::mutex> lock(bootstrapNetMutex);
    if (devOOB < 0) {
      const char* userIfEnv = ncclGetEnv("NCCL_OOB_NET_IFNAME");
      if (userIfEnv && strlen(userIfEnv) > 0) {
        INFO(NCCL_BOOTSTRAP | NCCL_ENV, "NCCL_OOB_NET_IFNAME set to %s", userIfEnv);
        bool searchNot = userIfEnv && userIfEnv[0] == '^';
        if (searchNot) userIfEnv++;
        bool searchExact = userIfEnv && userIfEnv[0] == '=';
        if (searchExact) userIfEnv++;
        struct netIf userIfs[MAX_OOB_DEVS];
        int nUserIfs = parseStringList(userIfEnv, userIfs, MAX_OOB_DEVS);
        // loop over the device and return the first one matching
        int nDev = 0;
        NCCLCHECK(comm->ncclNet->devices(&nDev));
        int devId = 0;
        while (devId < nDev) {
          ncclNetProperties_t props;
          comm->ncclNet->getProperties(devId, &props);
          // check against user specified HCAs/ports
          if (matchIfList(props.name, props.port, userIfs, nUserIfs, searchExact) ^ searchNot) {
            // All plain physical devices have been initialized at this point
            devOOB = devId;
            break;
          }
          devId++;
        }
        if (devOOB == -1) {
          if (!searchNot) {
            WARN("no device found matching %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          } else {
            WARN("no device found after excluding %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "",
                 userIfEnv);
          }
          return ncclInvalidArgument;
        }
      } else {
        // default choice is device 0
        devOOB = 0;
      }
      // display info on the chosen device
      ncclNetProperties_t props;
      ncclResult_t res = comm->ncclNet->getProperties(devOOB, &props);
      bool hasProp = res == ncclSuccess;
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using %s:%d", (hasProp) ? props.name : "N/A", (hasProp) ? props.port : -1);
    }
  }
  *dev = devOOB;
  return ncclSuccess;
}

static ncclResult_t netRingConnect(void* ctx, ncclNet_t* net, struct bootstrapListen_t* listen,
                                   char peerHandle[NCCL_NET_HANDLE_MAXSIZE], void** sendComm,
                                   ncclNetDeviceHandle_t** sendDevHandle, void** recvComm,
                                   ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!*sendComm) NCCLCHECK(net->connect(ctx, listen->net.dev, peerHandle, sendComm, sendDevHandle));
    if (!*recvComm) NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  } while (!*sendComm || !*recvComm);
  return ncclSuccess;
}

static ncclResult_t socketRingConnect(ncclSocketAddress* addr, struct ncclSocket* sendSocket,
                                      struct ncclSocket* listenSock, struct ncclSocket* recvSocket, uint64_t magic,
                                      volatile uint32_t* abortFlag) {
  // 根据 addr 地址创建一个 socket 到 sendSocket 中
  NCCLCHECK(ncclSocketInit(sendSocket, addr, magic, ncclSocketTypeBootstrap, abortFlag));
  // 完成对后继 rank 的 connect 和握手
  NCCLCHECK(ncclSocketConnect(sendSocket));
  // 清空 recvSocket->addr，用于接收 accept 获取的 socket 和 addr
  NCCLCHECK(ncclSocketInit(recvSocket));
  // 从 listenSock 中 accept 一个 socket 并放到 recvSocket 中
  NCCLCHECK(ncclSocketAccept(recvSocket, listenSock));
  return ncclSuccess;
}

static ncclResult_t ringAllInfo(struct ncclComm* comm, struct bootstrapState* state,
                                union ncclSocketAddress* peerAddresss, union ncclSocketAddress* peerProxy,
                                uint64_t* peerUDS, struct rasRankInit* rasRanks) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  // 以下数据结构聚合一个 rank 的 4 种全局信息
  struct bootstrapRingData {
    union ncclSocketAddress peerAddress;
    union ncclSocketAddress peerProxy;
    uint64_t peerUDS;
    struct rasRankInit rasRank;
  }* ringData = NULL;

  NCCLCHECK(ncclCalloc(&ringData, nRanks));
  // 打包当前 rank 的 4 种信息到 ringData
  // pack
  if (peerAddresss) memcpy(&(ringData[rank].peerAddress), peerAddresss + rank, sizeof(union ncclSocketAddress));
  if (peerProxy) memcpy(&(ringData[rank].peerProxy), peerProxy + rank, sizeof(union ncclSocketAddress));
  if (peerUDS) memcpy(&(ringData[rank].peerUDS), peerUDS + rank, sizeof(uint64_t));
  if (rasRanks) memcpy(&(ringData[rank].rasRank), rasRanks + rank, sizeof(*rasRanks));

  // allgather
  NCCLCHECKGOTO(bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingData)), res, exit);

  // 从 ringData 中依次解包所有 ranks 的 4 种信息
  // unpack
  for (int irank = 0; irank < nRanks; ++irank) {
    if (peerAddresss) memcpy(peerAddresss + irank, &(ringData[irank].peerAddress), sizeof(union ncclSocketAddress));
    if (peerProxy) memcpy(peerProxy + irank, &(ringData[irank].peerProxy), sizeof(union ncclSocketAddress));
    if (peerUDS) memcpy(peerUDS + irank, &(ringData[irank].peerUDS), sizeof(uint64_t));
    if (rasRanks) memcpy(rasRanks + irank, &(ringData[irank].rasRank), sizeof(*rasRanks));
  }

exit:
  free(ringData);
  return ncclSuccess;
}

static ncclResult_t sendToRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm, struct extInfo* info) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  // 创建一个 socket
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  // 完成 connect，并且发送 magic 和 type 与 root 握手建联
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  // 向 root 发送元信息、回连地址和 ring 连接地址
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct extInfo)), ret, fail);
  // 关闭当前 connected socket
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

NCCL_PARAM(StaggerRate, "UID_STAGGER_RATE", 7000);
NCCL_PARAM(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);

NCCL_PARAM(RasEnable, "RAS_ENABLE", 1);

/*
建立 NCCL 的 bootstrap 通信层，一个在真正的高速传输（NVLink/IB/…）建立之前就存在的、基于 socket（或 OOB 网络）的带外（out-of-band）控制通道。
它的产物是两样东西：
  1. 一个连接所有 rank 的环（ring）：每个 rank 知道自己的"下一个"邻居并建好收发连接。
  2. 一张全局地址表：通过这个环做一次 AllGather，让每个 rank 拿到所有其他 rank 的 proxy 地址、P2P 地址、UDS 等联系方式。
之后 NCCL 所有初始化阶段的"控制面消息"（拓扑交换、连接协商等）都跑在这个 bootstrap 通道上。
核心机制是"root 撮合 + 环形 AllGather"。root 是一个协调者线程（bootstrapRoot），负责把每个 rank 的联系方式"撮合"成环。
每个 rank 把自己的门牌号（监听的 server）报给 root，root 再回复每个 rank "你的下家是谁"。环一旦建成，后续的全局信息交换就不再需要 root，直接沿环传。
*/
ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t result = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  // char nextPeerHandle[NCCL_NET_HANDLE_MAXSIZE];
  // 保存所有 bootstrap 建链信息
  struct bootstrapState* state;
  // proxy 服务的长期监听端
  struct ncclSocket* proxySocket;
  // listenSockRoot 供 root 回连投递撮合结果，sock 从 root 接收一次的临时连接
  struct ncclSocket sock, listenSockRoot;
  // 上行报给 root 的建链信息
  struct extInfo info = {0};
  // 下行从 root 收到的"下家"地址
  union ringConnectInfo nextPeer;
  bool performRasAddRanks = true;
  struct rasRankInit* rasRanks = nullptr;

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};

  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;

  // Set magic: for grow existing ranks, receive from coordinator; otherwise use handle magic.
  // This is consistent with the magic created in ncclCommGetUniqueId.
  if (handles != NULL) {
    // 获取索引为 0 的 ncclBootstrapHandle 中的 magic 字段
    // state and comm magic set to the first magic ID
    comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic;
  } else if (parent != NULL) {
    comm->magic = state->magic = hashCombine(parent->magic, parent->childCount);
  } else {
    WARN("bootstrapInit: handles and parent are NULL");
    return ncclSystemError;
  }

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  // 填充要上报给 root 的 info 基本信息
  // fill up the info
  info.nranks = nranks;
  info.nroots = nHandles;
  // 初始化置 0 后继 rank 地址
  // get the ring connection info
  memset(&nextPeer, 0, sizeof(union ringConnectInfo));
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);
  if (ncclParamBootstrapNetEnable()) {  // 如果采用 NET bootstrap
    // Create net interface for other ranks to contact me (all gather)
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                 &STATE_LISTEN(state, net.comm)));
    memcpy(info.connectInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {  // 采用 TCP socket bootstrap
    // 创建一个 ring server socket，完成 bind、listen
    // create socket for ring neightbor to contact mee
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr,
                                 ncclSocketTypeBootstrap));
  }
  // Create socket for root to contact me using the root's magic
  // For grow operations, offset is parent->nRanks - 1 (last existing rank joins the root)
  // For normal init, offset is 0
  int offset = 0;
  if (comm->isGrow) {
    if (parent != NULL) {
      offset = parent->nRanks - 1;
    } else {
      if (handles != NULL) {
        offset = BOOTSTRAP_HANDLE(handles, 0)->nRanks - 1;
      } else {
        WARN("bootstrapInit: handles and parent are NULL");
        return ncclSystemError;
      }
    }
  }
  // 获取当前 rank 映射的 root 编号
  int curr_root = rootIdFromRank(rank, nranks, nHandles, offset);
  if (curr_root >= 0) {  // 当前是非存量 rank
    // 再创建一个监听 root 回传的 server socket
    NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot,
                                 &info.listenRootAddress, ncclSocketTypeBootstrap));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // stagger connection times to avoid an overload of the root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);
  // 计算当前 root 包含几个 ranks
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles, offset);
  /*
  NCCL_UID_STAGGER_THRESHOLD 表示错峰连接的"规模闸门"，当单个 root 要撮合的 rank 数 nRankRoot 超过这个阈值时，启用"按 localId 分批延迟连 root"的错峰，
  否则所有 rank 立即连接、不加任何人为延迟，默认值为 256。
  */
  if (nRankRoot > ncclParamStaggerThreshold()) {
    // NCCL_UID_STAGGER_RATE 表示错峰连接 root 的速率，默认值 7000 rank/s
    // for socket the message rate in microsec
    double msg_rate = ncclParamStaggerRate() / 1.0e6;
    // localId=k 的 rank 睡约 k×143μs,相邻 rank 间隔约 143μs，整体以 ~7000 rank/s 的速率涓流连上 root
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles, offset) / msg_rate;
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    std::this_thread::sleep_for(std::chrono::microseconds(musec));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // send info on my listening socket to root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);
  // 与 root 握手建联（connect、发送 magic 和 type），并且发送 info
  // send contact info to my own root
  info.rank = rank;
  info.iroot = curr_root;
  info.offset = offset;
  if (curr_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));
  if (parent && comm->isGrow && rank != 0) {
    // Grow: Ranks 1 to N-1 use the parent bootstrap to send connection information to the previous rank
    NCCLCHECK(bootstrapSend(parent->bootstrap, rank - 1, 0, &info.connectInfo, sizeof(info.connectInfo)));
  }
  // if needed, send the connection info to the previous root
  // commGrow with more than = 1 rank in the parent comm is a special case of multiroot
  if (((comm->isGrow && parent && (parent->nRanks > 1)) || nHandles > 1) &&
      isFirstFromRoot(rank, curr_root, nranks, nHandles, offset)) {
    /*
    多 root 情况下并且当前是某个 root 的第一个 rank 时，
    每个 root 组的第一个 rank 要额外把自己的连接信息发给前一个 root，
    好让那个 root 告诉它管的最后一个 rank 你的下家是当前这个边界 rank。
    */
    // 当前 rank 的前驱 rank
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
    // 前驱 rank 映射到的 root 编号
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles, offset);
    info.rank = prev_rank + 1; // my rank as seen by the previous root
    info.iroot = prev_root;
    // only send if the root is valid, existing rank N-1 will use the bootstrapSend just above
    if (prev_root >= 0) NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // get info on my "next" rank in the bootstrap ring from root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);
  if (curr_root >= 0) {
    // 清空 sock->addr
    NCCLCHECK(ncclSocketInit(&sock));
    // 从 listenSockRoot 监听 socket 中 accept 一个 socket 放到 sock 中，并完成 magic+type 握手
    NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
    // 从 sock 中接收获取后继地址
    NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
    // 关闭 accepted socket
    NCCLCHECK(ncclSocketClose(&sock));
    // 关闭监听 root 回传的 socket，就用这一次
    NCCLCHECK(ncclSocketClose(&listenSockRoot));
  }
  if (parent && comm->isGrow && rank != parent->nRanks - 1) {
    // Grow: Ranks 0 to N-2 use the parent bootstrap to recv connection information to the next rank.
    // This is consistent with the bootstrapSend above.
    NCCLCHECK(bootstrapRecv(parent->bootstrap, rank + 1, 0, &nextPeer, sizeof(nextPeer)));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // accept and connect the ring network
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                             &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                             &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle),
                             state->abortFlag));
  } else {
    /*
    1. 创建一个 socket 到 state->ring.socket.send，并与 nextPeer.addr 后继 rank connect 和握手
    2. 从 state->listen.socket accept 一个 socket 存放到 state->ring.socket.recv 中
    */
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket),
                                &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // AllGather all listen handlers
  // in case of failure, those resources will be free'd when calling bootstrapDestroy, so we can return immediatly
  // 初始化保存全局 proxy address 的数组，以及当前 proxy sock
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  // 为 proxy 初始化一个监听的 socket，并完成 listen、bind，然后将地址保存到当前 rank 的 state->peerProxyAddresses[rank]
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                   ncclSocketTypeProxy),
                result, fail);

  // 初始化保存全局 proxy address UDS 的数组
  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  // 获取当前 rank 的 UDS hash 并保存到 state->peerProxyAddressesUDS[rank]
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  /*
  建 mesh 监听端 state->listen.peerSocket，地址先写栈临时 peerSocketAddress，再 memcpy 进 peerP2pAddresses[rank]
  */
  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress,
                                   ncclSocketTypeBootstrap),
                result, fail);
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // NCCL_RAS_ENABLE，默认为 1 启用 RAS
  // Initialize RAS
  if (ncclParamRasEnable() == 1) {
    /*
    填写当前 rank 的 RAS 信息，包括：地址、pid、cudaDev、nvmlDev、hostHash、pidHash。
    关键在 ncclRasCommInit 的地址输入/输出语义，myRank->addr 初始化为 bootstrap 网卡 IP，
    成功返回后，addr 被更新为 RAS 网络监听 socket 的端口。
    */
    // The RAS thread will take ownership after ncclRasAddRanks succeeds.
    NCCLCHECKGOTO(ncclCalloc(&rasRanks, nranks), result, fail);
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    rasRanks[rank].pid = ncclOsGetPid();
    rasRanks[rank].cudaDev = comm->cudaDev;
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    rasRanks[rank].hostHash = getHostHash();
    rasRanks[rank].pidHash = getPidHash();
    if (ncclRasCommInit(comm, rasRanks + rank) != ncclSuccess) {
      // RAS 失败也不能破坏 bootstrap 集合操作
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // We should still participate in the ringAllInfo below as the peers will be waiting for us.
      // Just make sure that the address is clearly invalid...
      memset(rasRanks + rank, '\0', sizeof(*rasRanks));
      // 不将完整全员数组喂给本地 RAS 子系统，构建 RAS overlay 网络
      performRasAddRanks = false;
    }
  }

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);
  // AllGather 交换全局信息
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses,
                            state->peerProxyAddressesUDS, rasRanks),
                result, fail);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  /*
  1. 初始化 comm->sharedRes->proxyState，也是 comm->proxyState
  2. proxySocket 保存到 omm->proxyState->listenSock
  3. 当前 rank 创建一个 UDS 保存到 comm->proxyState->ipcSock
  */
  // Create the service proxy and get the UDS
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result,
                fail);

  /*
  把 ringAllInfo allgather 好的完整全员 rasRanks 数组经 ncclRasAddRanks 喂给本地 RAS 线程去构建 overlay 网络，
  成功后 rasRanks = nullptr 完成所有权移交(数组归 RAS 线程、由它 free)，使末尾 exit: free(rasRanks) 变成 no-op，
  避免 use-after-free / double-free。
  */
  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    if (ncclRasAddRanks(rasRanks, nranks) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
    } else {
      // 移交所有权后，rasRanks 分配的内存由 RAS 线程负责管理，置 nullptr，使末尾 exit: free(rasRanks) 变成 no-op
      rasRanks = nullptr;
    }
  }

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)",
       timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9, timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9, timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9, timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);
exit:
  free(rasRanks);
  return result;
fail:
  free(proxySocket);
  goto exit;
}

ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key,
                            int* parentRanks) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int prev, next;
  union ringConnectInfo info;
  union ringConnectInfo nextPeer;
  struct ncclSocket* proxySocket = NULL;
  struct bootstrapState* state;

  NCCLCHECKGOTO(ncclCalloc(&state, 1), ret, fail);
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;
  comm->magic = state->magic = magic;

  prev = parentRanks[(rank - 1 + nranks) % nranks];
  next = parentRanks[(rank + 1) % nranks];

  // create a handle for the others to reach out to me
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), ret, fail);
    NCCLCHECKGOTO(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle),
                                     &STATE_LISTEN(state, net.comm)),
                  ret, fail);
    memcpy(info.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neightbor to contact mee
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.addr, ncclSocketTypeBootstrap));
  }
  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress,
                               ncclSocketTypeBootstrap));

  if (ncclParamRasEnable() == 1) {
    if (ncclRasCommInit(comm, nullptr) != ncclSuccess) {
      INFO(NCCL_INIT | NCCL_RAS, "Continuing in spite of a RAS initialization error");
    }
  }

  // Get addr from next rank using the parent's connections
  NCCLCHECKGOTO(bootstrapSend(parent->bootstrap, prev, BOOTSTRAP_TAG_COMMSPLIT, &info, sizeof(union ringConnectInfo)),
                ret, fail);
  NCCLCHECKGOTO(bootstrapRecv(parent->bootstrap, next, BOOTSTRAP_TAG_COMMSPLIT, &nextPeer,
                              sizeof(union ringConnectInfo)),
                ret, fail);
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                                 &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                 &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle),
                                 state->abortFlag),
                  ret, fail);
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket),
                                &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), ret, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));
  if (parent->shareResources) {
    /* map local rank to top parent local rank. */
    for (int i = 0; i < nranks; ++i) {
      comm->topParentRanks[i] = parent->topParentRanks[parentRanks[i]];
    }
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, NULL, NULL, NULL), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), ret, fail);
    // Create the service proxy and get the UDS
    NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), ret, fail);
    NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), ret, fail);
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank,
                                     ncclSocketTypeProxy),
                  ret, fail);
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses,
                              state->peerProxyAddressesUDS, NULL),
                  ret, fail);
    NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), ret, fail);
  }

  TRACE(NCCL_BOOTSTRAP, "bootstrapSplit: comm %p parent %p rank %d nranks %d color %d key %d prev %d next %d - DONE",
        comm, parent, rank, nranks, color, key, prev, next);

exit:
  return ret;
fail:
  free(proxySocket);
  goto exit;
}

struct socketAckInfo {
  int rank;  // src rank
  int tag;  // 用途的 tag 标识
};

/*
bootstrap 层主动连接原语，对准 peer 的 mesh 监听地址建一条 TCP 连接，连上后第一件事就是发一个 ack = {rank, tag} 自报家门（我是谁、为什么事连你）。
它是被动端 socketAccept 的镜像，因为一个 rank 只有一个 mesh 监听点、所有 peer 所有用途的连接都往这里连，光看 TCP 层无法分辨来者身份，
只能靠这个 ack 做"连接与用途的解耦匹配"。
*/
static ncclResult_t socketConnect(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  struct socketAckInfo ack = (struct socketAckInfo){state->rank, tag};  // 初始化 ACK
  // 用之前 bootstrap 建立的 P2P 网络的 peer 的监听地址初始化建联 socket
  NCCLCHECKGOTO(ncclSocketInit(sock, state->peerP2pAddresses + peer, state->magic, ncclSocketTypeBootstrap,
                               state->abortFlag),
                ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), ret, fail);  // 发起 connect 并发送 magic+type 进行握手
  NCCLCHECKGOTO(socketSend(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);  // 发送 ACK 数据
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}

ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  TRACE(NCCL_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(socketConnect(commState, peer, tag, &sock));
  NCCLCHECKGOTO(socketSend(&sock, data, size), ret, fail);
  TRACE(NCCL_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

// Bootstrap send/receive functions
static ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));  // 初始化一个空链表成员
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));  // 将非预期接收的 sock 拷贝到空链表成员

  // 入链表队列
  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}

/*
  遍历 state->unexpectedConnections 链表，找目标节点，命中就摘链、memcpy 把存的 sock 取出来、free 节点、*found=1 直接返回。
  这保证了早到但当时没在等的连接不会被漏掉，上一次别的 socketAccept 调用把它 enqueue 了，这次正好要用就先 dequeue。
*/
static ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock,
                                      int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    // peer < 0 表示通配符（接受来自任何对等方的连接）
    // peer < 0 means wildcard (accept from any peer)
    if ((peer < 0 || elem->peer == peer) && elem->tag == tag) {  // 匹配 rank 和 tag
      if (prev == NULL) {  // 说明链表第一个旧匹配成功了
        state->unexpectedConnections = elem->next;
      } else {  // 需要将前一个成员链接到当前成员的下一个成员
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));  // 拷贝 sock
      free(elem);  // 释放当前链表成员
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;  // 没找到需要更新非匹配成员
    elem = elem->next;
  }
  return ncclSuccess;
}

static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
static ncclResult_t socketAccept(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // 先去之前到达的 sock 队列中搜索
  // Search unexpected connections first
  int found;
  NCCLCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) return ncclSuccess;

  // Then look for new connections
  while (1) {
    struct socketAckInfo ack = {0};
    NCCLCHECKGOTO(ncclSocketInit(sock), ret, fail);  // 清空 sock addr
    // peerSocket 就是当前 rank 的 P2P 监听 sock，accept 一个 socket 
    NCCLCHECKGOTO(ncclSocketAccept(sock, &STATE_LISTEN(state, peerSocket)), ret, fail);
    NCCLCHECKGOTO(socketRecv(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);  // 接收一个 ack 数据
    // Match: tag must match, and peer must match (peer < 0 means wildcard)
    if (ack.tag == tag && (peer < 0 || ack.rank == peer)) return ncclSuccess;  // 如果 ack 匹配直接返回
    // No match: queue for later and try next connection
    NCCLCHECKGOTO(unexpectedEnqueue(state, ack.rank, ack.tag, sock), ret, fail);  // 否则将接收到的 sock 和 ack 保存到链表
  }
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}

// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret;
  struct ncclSocket sock;
  NCCLCHECK(socketAccept(commState, peer, tag, &sock));
  TRACE(NCCL_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  NCCLCHECKGOTO(socketRecv(&sock, ((char*)data), size), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock, /*wait*/ true), ret, fail);
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

static ncclResult_t netRingAllGather(ncclNet_t* net, void* sendComm, void* recvComm, int rank, int nranks, char* data,
                                     int size, volatile uint32_t* abortFlag) {
  ncclResult_t res;
  uint64_t tFirst = 0, tRest = 0;
  void* sendDataHandle = NULL;
  void* recvDataHandle = NULL;
  NCCLCHECKGOTO(netReg(net, sendComm, data, nranks * size, &sendDataHandle), res, exit);
  NCCLCHECKGOTO(netReg(net, recvComm, data, nranks * size, &recvDataHandle), res, exit);
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "NetRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int i = 0; i < nranks - 1; i++) {
    int tag = i;
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(netSendRecv(net, sendComm, send_data, size, sendDataHandle, recvComm, recv_data, size, recvDataHandle,
                              tag, abortFlag),
                  res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "netRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  // do not fail in case of error, try to deregister as much as possible
  if (sendDataHandle) netDereg(net, sendComm, &sendDataHandle);
  if (recvDataHandle) netDereg(net, recvComm, &recvDataHandle);
  return res;
}

static ncclResult_t socketRingAllGather(struct ncclSocket* nextSock, struct ncclSocket* prevSock, int rank, int nranks,
                                        char* data, int size) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGather started: rank=%d nranks=%d", rank, nranks);
  /*
  计算 Ring 要迭代多少步 step
  两个反向 Ring：
    Ring0（prev→next，正向），往 nextSock 发、从 prevSock 收。k 步后，当前 rank 从 Ring0 收到 rank-1, rank-2, …, rank-k。
    Ring1（next→prev，反向），往 prevSock 发、从 nextSock 收。k 步后，当前 rank 从 Ring1 收到 rank+1, rank+2, …, rank+k。
  k 步后,rank r 拥有自己 r + Ring0 的 r-1..r-k + Ring1 的 r+1..r+k = 1 + 2k 片。
  要集齐 nranks 片，1 + 2k ≥ nranks → k ≥ (nranks-1)/2，取 totalSteps = nranks / 2(整除)即可。
  */
  int totalSteps = nranks / 2;
  TRACE(NCCL_BOOTSTRAP, "bidirectional bootstrap: totalSteps=%d", totalSteps);
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int step = 0; step < totalSteps; step++) {
    /*
    由于 nranks 为偶数的情况下，Ring0 + Ring1 总共需要迭代的 step 数为 nranks-1，是奇数，不能被 Ring0 和 Ring1 均分。
    所以需要判断当前是不是最后一次 step，如果是最后一次 step，只在 Ring0 上执行。
    */
    // N ranks requires (N-1)/2 steps for the double ring  algorithm.
    // If N is even, the last step is requires a single send/recv
    bool isFinalUnidirectional = (step == totalSteps - 1) && (nranks % 2 == 0);
    // Ring0: ring from previous to next
    // 计算当前 step Ring0 要发送的分片，其实就是在上一步接收保存的分片
    int sendSliceRing0 = (rank - step + nranks) % nranks;      // Send this slice to next neighbor
    // 计算当前 step Ring0 要接收的分片
    int recvSliceRing0 = (rank - step - 1 + nranks) % nranks;  // Receive this slice from prev neighbor
    // Ring1: ring from next to previous
    // 计算当前 step Ring1 要发送的分片，其实就是在上一步接收保存的分片
    int sendSliceRing1 = (rank + step) % nranks;               // Send this slice to prev neighbor
    // 计算当前 step Ring1 要接收的分片
    int recvSliceRing1 = (rank + step + 1) % nranks;           // Receive this slice from next neighbor
    if (isFinalUnidirectional) {
      // 最后一次 step 只由 Ring0 负责
      // Final unidirectional step, only Ring0 is used
      NCCLCHECKGOTO(socketSendRecv(nextSock, data + sendSliceRing0 * size, size, prevSock, data + recvSliceRing0 * size,
                                   size),
                    res, exit);
    } else {
      // Bidirectional step: Ring0 and Ring1 are used simultaneously
      // clang-format off
      /*
      先构建 Ring0 和 Ring1 4 次收发操作的参数，轮转时先把手头的 send 推出去(对端对称地也在 send)，
      数据先上线，再去 recv 就不会干等，避免"互等对方先收"的死锁。全双工是关键，
      nextSock 上同时承载 Ring0 的 send 和 Ring1 的 recv，prevSock 上同时承载 Ring0 的 recv 和 Ring1 的 send，四路互不干扰。
      */
      struct ncclSocketOp ops[4] = {
        {NCCL_SOCKET_SEND, nextSock, data + sendSliceRing0 * size, size, 0},  // Ring0: send to next
        {NCCL_SOCKET_RECV, prevSock, data + recvSliceRing0 * size, size, 0},  // Ring0: recv from prev
        {NCCL_SOCKET_SEND, prevSock, data + sendSliceRing1 * size, size, 0},  // Ring1: send to prev
        {NCCL_SOCKET_RECV, nextSock, data + recvSliceRing1 * size, size, 0}   // Ring1: recv from next
      };
      // clang-format on
      // 并发执行 4 路收发操作
      NCCLCHECKGOTO(socketDoubleSendRecv(ops), res, exit);
    }
    // 记录第一个 step 的执行时间，并记录剩下其他 step 执行总时间
    if (step == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)",
        tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  return res;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather", rank, nranks, size);

  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  /*
  NCCL_OOB_NET_ENABLE 控制 bootstrap（引导/带外控制通道）的环形 AllGather 走哪条传输后端：
    默认 0 → 走普通 TCP socket（socketRingAllGather）；
    非 0 → 走网络插件 fabric（netRingAllGather），即让 bootstrap 也能跑在 IB/RoCE 这类高速网络上，而不只是 TCP。
  */
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingAllGather(state->net, STATE_RING(state, net.sendComm), STATE_RING(state, net.recvComm), rank,
                                   nranks, (char*)allData, size, state->abortFlag),
                  res, exit);
  } else {
    NCCLCHECKGOTO(socketRingAllGather(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), rank, nranks,
                                      (char*)allData, size),
                  res, exit);
  }
exit:
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapAllGather for %d B done in %f sec: %f MB/sec", size, time / 1e9,
        (nranks * size / 1e6) / (time / 1e9));
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather DONE", rank, nranks, size);
  return res;
}

static ncclResult_t bootstrapP2PBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  if (nranks == 1) return ncclSuccess;
  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1] = {0};
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data, sizeof(data)));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, ranks, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, NULL, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

// 实现节点内或者 MNNVL 域内的 AllGather
ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size) {
  if (nranks == 1) return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  int prevRank = ranks[(rank - 1 + nranks) % nranks];  // 组内环的前驱 rank
  int nextRank = ranks[(rank + 1) % nranks];  // 组内环的后继 rank
  // intraNode bootstrap is done defacto using the socket-based implementation
  struct ncclSocket recvSocket, sendSocket;
  // 根据后继 rank 的 P2P 监听地址创建 socket，发送 connect 到 P2P，发送 magic+type 进行握手，发送 ack 包含当前 rank 和用途 tag
  NCCLCHECK(socketConnect(commState, nextRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &sendSocket));
  // 先搜索非预期链表中是否有匹配 ack 的 sock，其次从 P2P 监听 sock 中 accept 匹配 ack 的 sock，前期所有不匹配的 sock 都保存到非预期链表队列
  NCCLCHECK(socketAccept(commState, prevRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &recvSocket));

  // 组内双向 AllGather
  NCCLCHECK(socketRingAllGather(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size));

  NCCLCHECK(ncclSocketClose(&sendSocket));  // 用完关闭 socket
  NCCLCHECK(ncclSocketClose(&recvSocket));

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

// [IntraNode] in-place Broadcast
static ncclResult_t bootstrapP2PBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                          int size) {
  if (nranks == 1) return ncclSuccess;
  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      if (i != root) {
        NCCLCHECK(bootstrapSend(commState, ranks ? ranks[i] : i, /*tag=*/ranks ? ranks[i] : i, bcastData, size));
      }
    }
  } else {
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[root] : root, /*tag=*/ranks ? ranks[rank] : rank, bcastData,
                            size));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData,
                                         int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, ranks, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBroadcast for %d B done in %f sec: %f MB/sec", size,
        time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  return ncclSuccess;
}
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, NULL, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBroadcast done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  if (commState == NULL) return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // close unexpected and return an error if we are not aborting and still operations in the pipe
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (COMPILER_ATOMIC_LOAD(state->abortFlag, std::memory_order_acquire) == 0) {
      WARN("Unexpected connections are not empty");
      return ncclInternalError;
    }
  }
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(state->net->closeSend(STATE_RING(state, net.sendComm)));
    NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.recvComm)));
    NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
  } else {
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.send)));
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.recv)));
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket)));
  }
  // close the p2p socket
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  // proxy things are free'd elsewhere
  free(state->peerP2pAddresses);
  free(state);
  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  if (commState == NULL) return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // when aborting we need to close the proxy here (maybe?)
  free(state->peerProxyAddresses);
  free(state->peerProxyAddressesUDS);
  NCCLCHECK(bootstrapClose(commState));
  return ncclSuccess;
}
