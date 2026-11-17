/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <cstddef>
#include <mutex>
#include <poll.h>
#include <unistd.h>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "diagnostics_log.h"
#include "diagnostics.h"
#include "diagnostics_checks.h"
#include "nccl.h"
#include "param/param.h"
#include "profiler.h"
#include "utils.h"
#include "ras_internal.h"
#include "os.h"
#include "os_socket_pair.h"

// RAS 的通知类型
// Type of a notification from a local NCCL thread.
typedef enum {
  RAS_ADD_RANKS = 0,
  RAS_TERMINATE = 1,
  RAS_RUN_DIAG = 2
} rasNotificationType;

// Used for communication from local NCCL threads to the RAS thread.
struct rasNotification {
  rasNotificationType type;
  union {
    struct {
      struct rasRankInit* ranks;
      int nranks;
    } addRanks;
    struct {
      struct rasDiagnosticsContext ctx;
    } runDiag;
  };
};
static_assert(sizeof(struct rasNotification) <= PIPE_BUF, "The rasNotification structure is too large");

// These ensure that we get only one RAS port/thread per process.
static std::mutex rasInitMutex;
static bool rasInitialized = false;
static int rasInitRefCount = 0;

// The RAS network listening socket of this RAS thread (random port).
struct ncclSocket rasNetListeningSocket;

static std::thread rasThread;

// Used for communication from regular NCCL threads to the RAS thread.
static std::mutex rasNotificationMutex;
static ncclSocketPairDescriptor rasNotificationPipe[2] = {NCCL_SOCKET_PAIR_INVALID, NCCL_SOCKET_PAIR_INVALID};

// Data for the main poll() in the RAS thread.
struct pollfd* rasPfds;
static int nRasPfds;

// We use it all over the place; no point in wasting the stack...
char rasLine[SOCKET_NAME_MAXLEN + 1];

// An array holding the addresses of all NCCL communicators.  Modified by the NCCL threads (hence the mutex), read by
// the RAS thread.
std::mutex ncclCommsMutex;
struct ncclComm** ncclComms = nullptr;
int nNcclComms = 0;
bool ncclCommsSorted = false; // Whether the array is currently sorted. We sort by the comms' commHash and rank.

static ncclResult_t rasLocalNotify(const struct rasNotification* msg);
static ncclResult_t rasLocalHandle(bool* terminate);
static void rasThreadCleanup();

static ncclResult_t rasMsgHandleConnInit(const struct rasMsg* msg, struct rasSocket* sock);
static ncclResult_t rasMsgHandleConnInitAck(const struct rasMsg* msg, struct rasSocket* sock);
static ncclResult_t rasNetSendNack(struct rasSocket* sock);

static void* rasThreadMain(void*);

static void rasTerminate();

// enable to run passive RAS diagnostics
NCCL_PARAM(RasDiagnostics, "RUN_RAS_DIAGNOSTICS", 0);
NCCL_PARAM(RasEnable, "RAS_ENABLE", 1);

/*
RAS 把函数按"哪个线程调用它"分成两大类：
  1. regular NCCL threads，用户调 ncclCommInitRank 等的普通线程调用的 RAS 入口，如 ncclRasCommInit、ncclRasCommFini、ncclRasAddRanks。
它们是普通 NCCL 流程到 RAS的入口，必须线程安全(多 comm/多线程可能并发调)，且不亲自干 RAS 的重活，而是通过 self-pipe(rasNotificationPipe)把请求投递给 RAS 线程。
  2. the RAS thread，rasThreadMain 及其调用的 rasLocalHandle、rasLocalHandleAddRanks、overlay 网络维护/健康检查等，是真正干 RAS 活的地方。
*/

//////////////////////////////////////////////////
// Functions invoked from regular NCCL threads. //
//////////////////////////////////////////////////

/*
每次 comm 初始化都会调它，不是只调一次。但进程级 RAS 子系统只建一次，函数体用 DCLP(双重检查锁定)。
在所有 RAS 入口里它必须最先调，因为只有它把 rasInitialized 置 true，其它入口依赖这个前提。
myRank 是 struct rasRankInit*，调用方必须预先把 myRank->addr 填成 bootstrap 网卡的 IP。
成功返回后，函数把 RAS 网络监听 socket 的端口回写进 myRank->addr。
*/
// Invoked by regular NCCL threads on every comm initialization.  This is the first function to call.
// The myRank structure should be passed with the addr element initialized to the IP address of the bootstrap
// network interface to use.  On a successful return, the address will be updated with the port number of the
// RAS network listening socket.
ncclResult_t ncclRasCommInit(struct ncclComm* comm, struct rasRankInit* myRank) {
  ncclResult_t ret = ncclSuccess;
  // 双重检查锁(DCLP)，一个进程中只有第一个 comm 做重活
  if (!rasInitialized) {  // 外层，快路径，不加锁
    std::lock_guard<std::mutex> lock(rasInitMutex);
    if (!rasInitialized) {  // // 内层，加锁后防止两个线程同时通过外层判定后重复初始化
      union ncclSocketAddress addr;

      // 将 bootstrapNetIfAddr 拷来的 bootstrap 网卡 IP 拷贝到 addr
      memcpy(&addr, &myRank->addr, sizeof(addr));
      // 端口 htons(0)，bind 到 0 端口，OS 分配临时端口(RAS overlay 用随机端口，区别于客户端口的固定 28028)。
      (addr.sa.sa_family == AF_INET ? addr.sin.sin_port : addr.sin6.sin6_port) = htons(0);
      /*
      初始化一个 socket
        ncclSocketDefaultMagic() 用进程级默认 magic(NCCL_SOCKET_MAGIC，可 env 覆盖)，不是 per-comm magic，
      因为 RAS 跨所有 comm，同进程/节点的 peer 都是同一套 RAS 协议。
        abortFlag=nullptr(关键设计)。RAS 不绑定任何 comm 的 abortFlag，因为 RAS 必须在 comm abort/hang 时照常工作去诊断上报，
      若绑了某 comm 的 abortFlag，abort 那个 comm 就会连带杀掉 RAS，恰好在最需要它的时候失效。RAS 有自己的生命周期
      (rasInitialized + rasInitRefCount + atexit + 管道的 RAS_TERMINATE 通知)。对比:bootstrap socket 带 abortFlag 求可中止、
      proxy ipcSock 带 comm->abortFlag，唯独 RAS 故意不带。
        asyncFlag=1，RAS 网络 socket 非阻塞，由 rasThreadMain 的 reactor poll 循环驱动。
      RAS 单线程要复用大量 peer 连接 + 集合操作，必须异步。
      */
      NCCLCHECKGOTO(ncclSocketInit(&rasNetListeningSocket, &addr, ncclSocketDefaultMagic(), ncclSocketTypeRasNetwork,
                                   /*abortFlag*/ nullptr, /*asyncFlag*/ 1),
                    ret, fail);
      // RAS socket 完成 bind、listen
      NCCLCHECKGOTO(ncclSocketListen(&rasNetListeningSocket), ret, fail);
      INFO(NCCL_RAS, "RAS network listening socket at %s", ncclSocketToString(&rasNetListeningSocket.addr, rasLine));

      /*
      创建一个本地 28028 监听 socket 给外部 RAS 客户端工具(nccl-ras CLI / nvidia 调试工具)连进来查询 RAS 状态。
      所以 RAS 有两个监听端，rasNetListeningSocket(随机端口，peer RAS 线程互联的 overlay) 和 rasClientListeningSocket(固定端口，外部工具查询)。
      返回值被 (void) 丢弃，客户端口建失败(如端口被占)不致命，RAS 核心(overlay 监控)照常。
      */
      (void)rasClientInitSocket();

      /*
      self-pipe 唤醒管道(reactor 的跨线程通知)
      通过 pipe 创建一个管道的读、写端口 fds，保存到 rasNotificationPipe.
        写端 [1]：普通 NCCL 线程通过 rasLocalNotify(msg) 写入(持 rasNotificationMutex 防多线程交错)。
      触发者 ncclRasAddRanks(发 RAS_ADD_RANKS)、rasTerminate(发 RAS_TERMINATE)。
        读端 [0]：RAS 线程在 rasThreadMain 的 poll 里监听，可读时调 rasLocalHandle 读出 msg 并分派
      (RAS_ADD_RANKS → rasLocalHandleAddRanks;RAS_TERMINATE → *terminate=true)。
        作用：让单线程 reactor 接收异步跨线程通知而无需轮询/条件变量，通知被统一成"一个可读 fd"融进同一个 poll 循环。
        顺序关键：管道必须在 rasThread 启动之前建好(线程一上来就 poll [0])。
      */
      NCCLCHECKGOTO(ncclOsSocketPairCreate(rasNotificationPipe), ret, fail);

      rasDiagnosticsInit();
      /*
      启动 RAS 主线程
      rasThreadMain 是 RAS 的单线程 reactor(poll 一批 rasPfds：网络监听端、客户端监听端、通知管道读端、各 peer 连接)。
      */
      rasThread = std::thread(rasThreadMain, nullptr);
      // 线程名 "NCCL RAS" 便于 ps -T/gdb/htop 识别。
      ncclSetThreadName(rasThread, "NCCL RAS");
      // 状态置位
      rasInitialized = true;
      // 进程退出时发 RAS_TERMINATE 并 join 线程，干净收尾
      atexit(rasTerminate);
    }
  }

  // 增加引用计数，每个 comm +1
  ncclAtomicRefCountIncrement(&rasInitRefCount);

  /*
  在 DCLP 块之外，每个 comm 都执行。ncclComms 是进程内所有 comm 的注册表，带空洞(ncclRasCommFini 把终止的 comm 置 nullptr)。
  RAS 线程据此监控本进程全部 comm。扩容粒度 RAS_INCREMENT*8 = 32。
  */
  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);

    int i;
    // 找空槽(允许空洞)
    for (i = 0; i < nNcclComms; i++) {
      if (ncclComms[i] == nullptr) break;
    }
    // 没空槽则扩容 +32
    if (i == nNcclComms) {
      NCCLCHECK(ncclRealloc(&ncclComms, nNcclComms, nNcclComms + RAS_INCREMENT * 8));
      nNcclComms += RAS_INCREMENT * 8;
    }
    // comm 登记
    ncclComms[i] = comm;
    // 标记需重排(RAS 监控时按序查找)
    ncclCommsSorted = false;
  }

  // 回写实际 RAS 地址(供 allgather)，主要是随机分配的端口号
  if (myRank != nullptr) memcpy(&myRank->addr, &rasNetListeningSocket.addr, sizeof(myRank->addr));

exit:
  return ret;
fail:
  (void)ncclOsSocketPairClose(rasNotificationPipe);
  (void)close(rasClientListeningSocket);
  (void)ncclSocketClose(&rasNetListeningSocket);
  goto exit;
}

// Invoked by regular NCCL threads on every comm termination.
ncclResult_t ncclRasCommFini(const struct ncclComm* comm) {
  if (!rasInitialized) return ncclSuccess;
  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);
    for (int i = 0; i < nNcclComms; i++) {
      if (ncclComms[i] == comm) {
        ncclComms[i] = nullptr;
        ncclCommsSorted = false;
        break;
      }
    }
  }
  ncclAtomicRefCountDecrement(&rasInitRefCount);
  return ncclSuccess;
}

// Global destructor.  Notifies the RAS thread to release all the resources
// and terminate.  Waits for the thread to terminate.
static void rasTerminate() {
  struct rasNotification msg;
  if (!rasInitialized) return;
  memset(&msg, '\0', sizeof(msg));
  msg.type = RAS_TERMINATE;
  if (rasLocalNotify(&msg) == ncclSuccess) rasThread.join();
}

/*
ncclRasAddRanks 写完管道就返回，RAS 线程稍后(rasThreadMain 的 poll 轮到管道读端时)才真正读指针、调 rasLocalHandleAddRanks
*/
// Invoked by regular NCCL threads on every (non-split) comm initialization.  Provides info on all the ranks within
// the communicator.
ncclResult_t ncclRasAddRanks(struct rasRankInit* ranks, int nranks) {
  // 创建一个 RAS 消息
  struct rasNotification msg;
  memset(&msg, '\0', sizeof(msg));
  msg.type = RAS_ADD_RANKS;
  /*
  这里传入的是 RAS 所有 ranks 的初始化数据指针。所以数组本身必须在移交后继续存活，直到 RAS 线程异步读取处理。
  为什么"传指针"在这里成立，有三个前提：
    1. 同进程，过管道的指针仍然有效；
    2. 为什么不拷贝数组?大小 + 管道原子性
      ranks 是 nranks × sizeof(rasRankInit),可能上千 rank、几百 KB，塞不进定长小 msg。
      那条 static_assert(sizeof(struct rasNotification) <= PIPE_BUF) 是设计护栏，保证整个 msg(含指针)≤ PIPE_BUF(4096),
    于是 write 到管道是原子的，多线程通知不交错、单条通知不被截断。传 8 字节指针正好让 msg 极小、满足原子写，真拷贝大数组就破坏了这个前提。
    3. 管道提供 happens-before，数组移交前已填好。写管道(rasLocalNotify)→ 读管道(rasLocalHandle)构成同步点。
  数组在 ringAllInfo 里全部填好之后才由 ncclRasAddRanks 把指针写进管道，RAS 线程读到指针时，数组已是完整数据。
  */
  msg.addRanks.ranks = ranks;
  msg.addRanks.nranks = nranks;
  // 把 msg(含指针)写进 rasNotificationPipe[1]
  NCCLCHECK(rasLocalNotify(&msg));
  return ncclSuccess;
}

// Requests the RAS thread to run RAS diagnostics for this communicator.
ncclResult_t ncclRunRasDiagnostics(struct ncclComm* comm) {
  struct rasNotification msg;
  ncclResult_t ret = ncclSuccess;

  if (ncclParamRasEnable() != 1) {
    DIAG_PRINT("NCCL DIAG INFO RAS diagnostics skipped: NCCL_RAS_ENABLE=0");
    return ncclSuccess;
  }

  memset(&msg, '\0', sizeof(msg));
  msg.type = RAS_RUN_DIAG;
  ret = rasDiagnosticsContextInit(&msg.runDiag.ctx, comm);
  if (ret == ncclSuccess) {
    if (comm != nullptr && comm->rank == 0) DIAG_PRINT("NCCL DIAG === RAS Diagnostics ===");
    ret = rasLocalNotify(&msg);
  }
  if (ret != ncclSuccess) {
    if (comm != nullptr) {
      INFO(NCCL_RAS, "RAS diagnostics trigger returned %d for comm 0x%lx", ret, comm->commHash);
    } else {
      INFO(NCCL_RAS, "RAS diagnostics trigger returned %d", ret);
    }
  }
  return ret;
}

// Internal function running on regular NCCL threads -- asynchronously notifies the RAS thread.
static ncclResult_t rasLocalNotify(const struct rasNotification* msg) {
  if (!rasInitialized) return ncclSuccess;

  // 在此获取独占锁，以避免多个用户线程之间的多路复用（不确定这是否绝对必要，但这样做也没坏处）
  // Take an exclusive lock here to avoid multiplexing between multiple user threads (not sure if it's
  // strictly required, but it won't hurt)...
  std::lock_guard<std::mutex> lock(rasNotificationMutex);
  size_t done = 0;
  // 循环写入通知
  while (done < sizeof(*msg)) {
    size_t written;
    // 异步向管道写端写入一次数据
    NCCLCHECK(ncclOsSocketPairWrite(rasNotificationPipe[1], (char*)msg + done, sizeof(*msg) - done, &written));
    done += written;
  }
  return ncclSuccess;
}

/////////////////////////////////////////////////////////////////////////////////
// Functions related to the handling of local notifications from NCCL threads. //
/////////////////////////////////////////////////////////////////////////////////

// Handles asynchronous local notifications arriving from regular NCCL threads.
static ncclResult_t rasLocalHandle(bool* terminate) {
  struct rasNotification msg;

  size_t done = 0;
  while (done < sizeof(msg)) {
    size_t nread;
    NCCLCHECK(ncclOsSocketPairRead(rasNotificationPipe[0], (char*)&msg + done, sizeof(msg) - done, &nread));
    if (nread == 0) {
      // EOF
      return ncclSystemError;
    }
    done += nread;
  }

  if (msg.type == RAS_ADD_RANKS) {
    (void)rasLocalHandleAddRanks(msg.addRanks.ranks, msg.addRanks.nranks);
    // Not great if the above fails, but it shouldn't be critical; better to keep going.
  } else if (msg.type == RAS_RUN_DIAG) {
    ncclResult_t ret = rasLocalHandleRunDiag(&msg.runDiag.ctx);
    if (ret != ncclSuccess) INFO(NCCL_RAS, "RAS diagnostics returned %d", ret);
  } else if (msg.type == RAS_TERMINATE) {
    INFO(NCCL_RAS, "RAS handling local termination request");
    *terminate = true;
  } else {
    WARN("RAS received unknown notification type %d", msg.type);
    return ncclInternalError;
  }

  return ncclSuccess;
}

// Cleans up local RAS state, normally in response to a RAS_TERMINATE notification.
static void rasThreadCleanup() {
  rasClientSupportTerminate();
  rasNetTerminate();
  rasCollectivesTerminate();
  rasPeersTerminate();

  {
    std::lock_guard<std::mutex> lock(rasInitMutex);
    (void)ncclOsSocketPairClose(rasNotificationPipe);
    // rasClientListeningSocket is taken care of by rasClientSupportTerminate().
    (void)ncclSocketClose(&rasNetListeningSocket);
    rasInitRefCount = 0;
    rasInitialized = false;
  }

  {
    std::lock_guard<std::mutex> lock(ncclCommsMutex);
    free(ncclComms);
    ncclComms = nullptr;
    nNcclComms = 0;
    ncclCommsSorted = false;
  }

  free(rasPfds);
  rasPfds = nullptr;
  nRasPfds = 0;
}

////////////////////////////////////////////////
// Generic functions related to RAS messages. //
////////////////////////////////////////////////

// Allocates a RAS message of the desired length for sending.
// Behind the scenes allocates encapsulating rasMsgMeta structure, which includes local metadata stored in front
// of the message.
// Must use rasMsgFree to free.
// ncclCallocQuiet is limited to rasMsgAlloc/rasMsgRecv: routine and keep-alive traffic otherwise floods ALLOC_HOST.
ncclResult_t rasMsgAlloc(struct rasMsg** msg, size_t msgLen) {
  struct rasMsgMeta* meta = nullptr;
  NCCLCHECK(ncclCallocQuiet((char**)&meta, offsetof(struct rasMsgMeta, msg) + msgLen));
  *msg = &meta->msg;
  // coverity[leaked_storage:FALSE] => rasMsgFree is used to free it
  return ncclSuccess;
}

// To be used only with messages allocated with rasMsgAlloc.  I.e., it should be used for sent messages, not
// for received ones.
void rasMsgFree(struct rasMsg* msg) {
  if (msg) {
    struct rasMsgMeta* meta = (struct rasMsgMeta*)((char*)msg - offsetof(struct rasMsgMeta, msg));
    free(meta);
  }
}

// Enqueues a message for sending down a RAS connection.
void rasConnEnqueueMsg(struct rasConnection* conn, struct rasMsg* msg, size_t msgLen, bool front) {
  // Get to the metadata of this message.
  struct rasMsgMeta* meta = (struct rasMsgMeta*)((char*)msg - offsetof(struct rasMsgMeta, msg));
  bool ready = false;

  meta->enqueueTime = clockNano();
  meta->offset = 0;
  meta->length = msgLen;

  if (front) ncclIntruQueueEnqueueFront(&conn->sendQ, meta);
  else ncclIntruQueueEnqueue(&conn->sendQ, meta);

  if (conn->sock) {
    if (conn->sock->status == RAS_SOCK_READY ||
        (conn->sock->status == RAS_SOCK_HANDSHAKE && msg->type == RAS_MSG_CONNINIT)) {
      rasPfds[conn->sock->pfd].events |= POLLOUT;
      ready = true;
    }
  }
  if (!ready) {
    // It's not a bug, unless it's for things like keep-alive messages...
    INFO(NCCL_RAS,
         "RAS enqueued message type %d on a non-ready connection with %s "
         "(experiencingDelays %d, startRetryTime %.2fs, socket status %d)",
         msg->type, ncclSocketToString(&conn->addr, rasLine), conn->experiencingDelays,
         (conn->startRetryTime ? (double)(clockNano() - conn->startRetryTime) / CLOCK_UNITS_PER_SEC : 0.0),
         (conn->sock ? conn->sock->status : -1));
  }
}

// Attempts to send the queued RAS messages to another RAS thread.
ncclResult_t rasConnSendMsg(struct rasConnection* conn, bool* closed, bool* allSent) {
  struct rasMsgMeta* meta;
  *closed = false;
  while ((meta = ncclIntruQueueHead(&conn->sendQ)) != nullptr) {
    if (conn->sock->status == RAS_SOCK_HANDSHAKE && meta->msg.type != RAS_MSG_CONNINIT) {
      // We don't send anything beyond the handshake at this point.
      meta = nullptr;
      break;
    }
    if (meta->offset < sizeof(meta->length)) {
      // Send the length of the message.
      NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &conn->sock->sock, &meta->length, sizeof(meta->length),
                                   &meta->offset, closed));
      if (*closed) return ncclSuccess;
      if (meta->offset < sizeof(meta->length)) break;
    }
    // Send the body of the message.
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &conn->sock->sock, ((char*)&meta->msg) - sizeof(meta->length),
                                 meta->length + sizeof(meta->length), &meta->offset, closed));
    if (*closed) return ncclSuccess;
    if (meta->offset < meta->length + sizeof(meta->length)) break;
    ncclIntruQueueDequeue(&conn->sendQ);
    free(meta);
  }

  *allSent = !meta;

  return ncclSuccess;
}

// Attempts to receive a message through a RAS socket.
ncclResult_t rasMsgRecv(struct rasSocket* sock, struct rasMsg** msg, bool* closed) {
  *closed = false;
  if (sock->recvOffset < sizeof(sock->recvLength)) {
    // Receive the length of the message.
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &sock->sock, &sock->recvLength, sizeof(sock->recvLength),
                                 &sock->recvOffset, closed));
    if (*closed || sock->recvOffset < sizeof(sock->recvLength)) return ncclSuccess;
    NCCLCHECK(ncclCallocQuiet((char**)&sock->recvMsg, sock->recvLength));
  }
  // Receive the body of the message.
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &sock->sock, ((char*)sock->recvMsg) - sizeof(sock->recvLength),
                               sock->recvLength + sizeof(sock->recvLength), &sock->recvOffset, closed));
  if (*closed || sock->recvOffset < sock->recvLength + sizeof(sock->recvLength)) return ncclSuccess;

  *msg = sock->recvMsg;
  sock->recvMsg = nullptr;
  sock->recvOffset = sock->recvLength = 0;

  return ncclSuccess;
}

//////////////////////////////////////////////////////////////////
// Functions related to the handling of specific message types. //
//////////////////////////////////////////////////////////////////

// Invoked from the main RAS thread to dispatch incoming messages to the appropriate handler.
ncclResult_t rasMsgHandle(struct rasMsg* msg, struct rasSocket* sock) {
  if (msg->type == RAS_MSG_CONNINIT) {
    NCCLCHECK(rasMsgHandleConnInit(msg, sock));
  } else if (msg->type == RAS_MSG_CONNINITACK) {
    NCCLCHECK(rasMsgHandleConnInitAck(msg, sock));
  } else if (msg->type == RAS_MSG_KEEPALIVE) {
    NCCLCHECK(rasMsgHandleKeepAlive(msg, sock));
  } else if (msg->type == RAS_MSG_PEERSUPDATE) {
    NCCLCHECK(rasMsgHandlePeersUpdate(msg, sock));
  } else if (msg->type == RAS_MSG_COLLREQ) {
    NCCLCHECK(rasMsgHandleCollReq(msg, sock));
  } else if (msg->type == RAS_MSG_COLLRESP) {
    NCCLCHECK(rasMsgHandleCollResp(msg, sock));
  } else {
    WARN("RAS received unknown message type (%d) from %s", msg->type, ncclSocketToString(&sock->sock.addr, rasLine));
    return ncclInternalError;
  }

  return ncclSuccess;
}

// Handles the first message sent over a RAS socket as part of the handshake.
static ncclResult_t rasMsgHandleConnInit(const struct rasMsg* msg, struct rasSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct rasConnection* conn = nullptr;
  int peerIdx;
  struct rasMsg* newMsg = nullptr;
  int newMsgLen;
  char line[SOCKET_NAME_MAXLEN + 1];
  char versionLocal[16], versionRemote[16];

  INFO(NCCL_RAS, "RAS handling connInit from %s (version %s, listeningAddr %s, peersHash 0x%lx, deadPeersHash 0x%lx)",
       ncclSocketToString(&sock->sock.addr, rasLine),
       ncclVersionToString(msg->connInit.ncclVersion, versionRemote, sizeof(versionRemote)),
       ncclSocketToString(&msg->connInit.listeningAddr, line), msg->connInit.peersHash, msg->connInit.deadPeersHash);

  if (msg->connInit.ncclVersion != NCCL_VERSION_CODE) {
    // Close any such sockets immediately!  This is basically unrecoverable...
    WARN("NCCL version mismatch with remote peer %s (local: %s, remote %s)",
         ncclSocketToString(&sock->sock.addr, rasLine),
         ncclVersionToString(NCCL_VERSION_CODE, versionLocal, sizeof(versionLocal)),
         ncclVersionToString(msg->connInit.ncclVersion, versionRemote, sizeof(versionRemote)));
    rasNetSendNack(sock);
    rasSocketTerminate(sock, /*finalize*/ true);
    ret = ncclInvalidUsage;
    goto exit;
  }

  if (rasPeerIsDead(&msg->connInit.listeningAddr)) {
    // A peer long declared dead is suddenly alive again?!
    INFO(NCCL_RAS, "RAS connection from peer %s that is considered dead!",
         ncclSocketToString(&msg->connInit.listeningAddr, rasLine));
    rasNetSendNack(sock);
    rasSocketTerminate(sock, /*finalize*/ true);
    goto exit;
  }

  // Check for any existing connection with that RAS thread (could happen due to a network issue, or possibly a race).
  conn = rasConnFind(&msg->connInit.listeningAddr);
  if (conn) {
    INFO(NCCL_RAS,
         "RAS found a matching existing connection (sendQ %sempty, experiencingDelays %d, startRetryTime %.2fs)",
         (ncclIntruQueueEmpty(&conn->sendQ) ? "" : "not "), conn->experiencingDelays,
         (conn->startRetryTime ? (double)(clockNano() - conn->startRetryTime) / CLOCK_UNITS_PER_SEC : 0.0));

    if (conn->sock) {
      INFO(NCCL_RAS, "RAS found an alternative existing socket (status %d, createTime %.2fs)", conn->sock->status,
           (double)(clockNano() - conn->sock->createTime) / CLOCK_UNITS_PER_SEC);
      // In general we prefer to keep the newer connection, but "newer" can be a relative term: we may have
      // a race where both sides attempt to establish a connection at roughly the same time, so the other side's
      // incoming connection ends up looking newer than the locally-initiated one -- for *both* of them.
      // If each side closed the "old" one, both would end up being closed.
      // As we normally try to initiate connections from the side with a lower address (precisely to avoid such
      // situations), we'll follow the same logic here: the "lower" side will reject the new connection (as it
      // came from the "wrong" side), whereas the "higher" side will keep the new one (as it came from the correct
      // side) and terminate the old one (that it presumably just opened).
      if (ncclSocketsCompare(&rasNetListeningSocket.addr, &conn->addr) < 0) {
        INFO(NCCL_RAS, "RAS terminating the new socket");
        rasSocketTerminate(sock, /*finalize*/ true);
        goto exit;
      } else {
        INFO(NCCL_RAS, "RAS keeping the new socket and terminating the existing one");
        rasSocketTerminate(conn->sock);
      }
    }
  } else {
    // conn == nullptr
    NCCLCHECK(getNewConnEntry(&conn));
    memcpy(&conn->addr, &msg->connInit.listeningAddr, sizeof(conn->addr));
  }

  sock->status = RAS_SOCK_READY;
  // rasConnResume will reset any experiencingDelays, startRetryTime, etc.

  {
    struct rasEventNotification event = {
      .eventType = "PEER_CONNECTING", .details = "", .peerInfo = nullptr, .peerAddr = &msg->connInit.listeningAddr
    };
    rasClientsNotifyEvent(RAS_EVENT_TRACE, &event);
  }

  conn->sock = sock;
  sock->conn = conn;
  memcpy(&sock->sock.addr, &msg->connInit.listeningAddr, sizeof(sock->sock.addr));

  // Make sure that the connection is part of the right links forming the RAS network.  At this point we only
  // update the expected (non-external) connections; external ones will be added during keep-alive handling.
  peerIdx = rasPeerFind(&conn->addr);
  // Note: it's possible for peerIdx to be -1 at this point if, due to races, the connInit arrives before
  // the peers update.
  if (peerIdx != -1) {
    (void)rasLinkConnUpdate(&rasNextLink, conn, peerIdx);
    (void)rasLinkConnUpdate(&rasPrevLink, conn, peerIdx);
  }

  // Send a confirmation to the server that requested the connection (so that the resilience code can mark
  // the connection as live).
  newMsgLen = rasMsgLength(RAS_MSG_CONNINITACK);
  NCCLCHECK(rasMsgAlloc(&newMsg, newMsgLen));
  newMsg->type = RAS_MSG_CONNINITACK;
  newMsg->connInitAck.nack = 0;
  rasConnEnqueueMsg(conn, newMsg, newMsgLen, /*front*/ true);

  conn->lastRecvPeersHash = msg->connInit.peersHash;
  conn->lastRecvDeadPeersHash = msg->connInit.deadPeersHash;

  if (msg->connInit.peersHash != rasPeersHash || msg->connInit.deadPeersHash != rasDeadPeersHash) {
    // Send my rasPeers and request the same in return.
    INFO(NCCL_RAS, "RAS connInit hash mismatch (my peersHash 0x%lx, deadPeersHash 0x%lx); sending my (dead) peers",
         rasPeersHash, rasDeadPeersHash);
    NCCLCHECK(rasConnSendPeersUpdate(conn, rasPeers, nRasPeers));
  }
exit:
  return ret;
}

// Handles the second message sent over a RAS socket as part of the handshake.
static ncclResult_t rasMsgHandleConnInitAck(const struct rasMsg* msg, struct rasSocket* sock) {
  INFO(NCCL_RAS, "RAS handling connInitAck from %s (nack %d)", ncclSocketToString(&sock->sock.addr, rasLine),
       msg->connInitAck.nack);

  if (msg->connInitAck.nack) {
    // The remote peer doesn't want to talk to us.  The easiest way to prevent it is by declaring it dead.
    // We make a copy of the address because rasConnDisconnect will terminate the rasSocket.
    union ncclSocketAddress addr;
    memcpy(&addr, &sock->sock.addr, sizeof(addr));
    rasConnDisconnect(&addr);
    (void)rasPeerDeclareDead(&addr);

    return ncclSuccess;
  }

  sock->status = RAS_SOCK_READY;
  // rasConnResume will reset any experiencingDelays, startRetryTime, etc.

  return ncclSuccess;
}

// Handles the deadPeer broadcast.
void rasMsgHandleBCDeadPeer(struct rasCollRequest** pReq, size_t* pReqLen, bool* pDone) {
  INFO(NCCL_RAS, "RAS handling deadPeer (addr %s)", ncclSocketToString(&(*pReq)->deadPeer.addr, rasLine));

  *pReqLen = rasCollDataLength(RAS_BC_DEADPEER);
  if (!rasPeerIsDead(&(*pReq)->deadPeer.addr)) {
    rasConnDisconnect(&(*pReq)->deadPeer.addr);
    (void)rasPeerDeclareDead(&(*pReq)->deadPeer.addr);
    *pDone = false;
  } else {
    INFO(NCCL_RAS, "RAS already knew it was dead");
    // No point in re-broadcasting what's already known.
    *pDone = true;
  }
}

// Handles the profilerMask broadcast.
void rasMsgHandleBCProfilerMask(struct rasCollRequest** pReq, size_t* pReqLen, bool* pDone) {
  INFO(NCCL_RAS, "RAS handling profilerMask (mask 0x%x)", (*pReq)->profilerMask.eventMask);
  *pReqLen = rasCollDataLength(RAS_BC_PROFILER_MASK);
  ncclProfilerSetRasOverride((*pReq)->profilerMask.eventMask);
  // A mask value has no "already known" terminal state, so keep propagating to every peer; the
  // rootAddr/rootId history still bounds the broadcast.
  *pDone = false;
}

// Attempts to immediately send a fatal NACK connInitAck response to a socket.  A bit of a hack (as it doesn't
// follow our usual message queuing and polling convention) but, since this can be invoked only for newly opened
// connections, and the message is tiny, it should be OK.  We can't use the regular path because the socket is
// about to be terminated.
static ncclResult_t rasNetSendNack(struct rasSocket* sock) {
  struct rasMsg msg;
  int length = rasMsgLength(RAS_MSG_CONNINITACK);
  bool closed = false;
  int offset;

  INFO(NCCL_RAS, "RAS sending NACK to %s", ncclSocketToString(&sock->sock.addr, rasLine));

  memset(&msg, '\0', sizeof(msg));
  msg.type = RAS_MSG_CONNINITACK;
  msg.connInitAck.nack = 1;
  offset = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &sock->sock, &length, sizeof(length), &offset, &closed));
  if (closed || offset < sizeof(length)) return ncclSuccess;
  offset = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &sock->sock, &msg, length, &offset, &closed));
  // We are closing this socket anyway -- it doesn't matter to us if we succeeded or not.

  return ncclSuccess;
}

/////////////////////////////////////////////////////////////////
// Functions related to the main event loop of the RAS thread. //
/////////////////////////////////////////////////////////////////

// Main function of the RAS thread.
static void* rasThreadMain(void*) {
  ncclResult_t ret = ncclSuccess; // Unused.
  int pfd;
  int rasNetListeningSocketFd;

  INFO(NCCL_RAS, "RAS thread started");

  // Initialize the global pollfd with the file descriptors we already have (the pipe and the listening socket).
  NCCLCHECKGOTO(rasGetNewPollEntry(&pfd), ret, exit);
  rasPfds[pfd].fd = rasNotificationPipe[0];
  rasPfds[pfd].events = POLLIN;

  NCCLCHECKGOTO(rasGetNewPollEntry(&pfd), ret, exit);
  NCCLCHECKGOTO(ncclSocketGetFd(&rasNetListeningSocket, &rasNetListeningSocketFd), ret, exit);
  rasPfds[pfd].fd = rasNetListeningSocketFd;
  rasPfds[pfd].events = POLLIN;

  NCCLCHECKGOTO(rasGetNewPollEntry(&pfd), ret, exit);
  rasPfds[pfd].fd = rasClientListeningSocket;
  rasPfds[pfd].events = POLLIN;

  // Main event loop of the RAS thread.
  for (int64_t nextWakeup = 0;;) {
    int timeoutMs, nEvents;
    int64_t now = clockNano();
    if (nextWakeup > 0) {
      // The "1" below helps avoid round-downs and especially zeroes.
      timeoutMs = std::max(nextWakeup - now, (int64_t)0) / (CLOCK_UNITS_PER_SEC / 1000) + 1;
    } else {
      timeoutMs = rasTimeoutFactorSec(1) * 1000;
    }
    timeoutMs = std::min(timeoutMs, 1000); // At most 1 actual second.

    nEvents = poll(rasPfds, nRasPfds, timeoutMs);

    nextWakeup = clockNano() + rasTimeoutFactorNs(1); // 1 second (possibly stretched).
    if (nEvents == -1 && errno != EINTR) {
      INFO(NCCL_RAS, "RAS continuing in spite of an unexpected error from poll: %s", strerror(errno));
    }

    // Handle any poll-related events.
    for (int pollIdx = 0; pollIdx < nRasPfds && nEvents > 0; pollIdx++) {
      if (rasPfds[pollIdx].revents) {
        nEvents--;
        if (rasPfds[pollIdx].revents & POLLNVAL) {
          // We passed an invalid file descriptor.  For now mark it to be ignored at the next iteration.  Hopefully
          // in the meantime one of the event handlers will terminate it.
          INFO(NCCL_RAS, "RAS poll returned POLLNVAL on pollIdx %d (fd %d) -- internal error?", pollIdx,
               rasPfds[pollIdx].fd);
          rasPfds[pollIdx].fd = POLL_FD_IGNORE;
        }
        if (rasPfds[pollIdx].fd == rasNotificationPipe[0]) {
          bool terminate = false;
          NCCLCHECKGOTO(rasLocalHandle(&terminate), ret, exit);
          if (terminate) goto exit;
        } else if (rasPfds[pollIdx].fd == rasNetListeningSocketFd) {
          (void)rasNetAcceptNewSocket();
        } else if (rasPfds[pollIdx].fd == rasClientListeningSocket) {
          (void)rasClientAcceptNewSocket();
        } else {
          // Check if it's one of the RAS sockets.
          struct rasSocket* sock;
          for (sock = rasSocketsHead; sock;) {
            struct rasSocket* sockNext = sock->next;
            if (rasPfds[pollIdx].fd == sock->sock.socketDescriptor) {
              rasSockEventLoop(sock, pollIdx);
              break;
            }
            sock = sockNext;
          } // for (sock)

          if (sock == nullptr) {
            // Try a client socket instead.
            for (struct rasClient* client = rasClientsHead; client;) {
              struct rasClient* clientNext = client->next;
              if (rasPfds[pollIdx].fd == client->sock) {
                rasClientEventLoop(client, pollIdx);
                break;
              }
              client = clientNext;
            } // for (client)
          } // if (sock == nullptr)
        } // dynamic fds
      } // if (revents)
    } // for (pollIdx)

    now = clockNano();

    rasSocksHandleTimeouts(now, &nextWakeup);

    rasConnsHandleTimeouts(now, &nextWakeup);

    rasNetHandleTimeouts(now, &nextWakeup);

    rasCollsHandleTimeouts(now, &nextWakeup);
  } // for (;;)

exit:
  rasThreadCleanup();
  INFO(NCCL_RAS, "RAS thread terminating");
  return nullptr;
}

// Returns the index of the first available entry in the rasPfds array, enlarging the array if necessary.
ncclResult_t rasGetNewPollEntry(int* index) {
  int i;
  for (i = 0; i < nRasPfds; i++)
    if (rasPfds[i].fd == NCCL_INVALID_SOCKET) break;
  if (i == nRasPfds) {
    NCCLCHECK(ncclRealloc(&rasPfds, nRasPfds, nRasPfds + RAS_INCREMENT));
    nRasPfds += RAS_INCREMENT;
    for (int j = i; j < nRasPfds; j++) rasPfds[j].fd = NCCL_INVALID_SOCKET;
  }

  memset(rasPfds + i, '\0', sizeof(*rasPfds));
  rasPfds[i].fd = NCCL_INVALID_SOCKET;

  *index = i;
  return ncclSuccess;
}
