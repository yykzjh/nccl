/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "os.h"

#include "checks.h"
#include "crypt.h"
#include "utils.h"

#include <cstdint>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cstring>
#include "socket.h"
#include "utils.h"
#include "checks.h"
#include "param.h"
#include <pthread.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <atomic>
#include <fcntl.h>
#include <limits.h>
#include <cstdio>
#include <cctype>

static thread_local char ncclDlErrorBuf[256] = {0};

static void saveDlError() {
  // 获取动态链接器的错误信息
  const char* err = dlerror();
  if (err) {
    snprintf(ncclDlErrorBuf, sizeof(ncclDlErrorBuf), "%s", err);
  } else {
    ncclDlErrorBuf[0] = '\0';
  }
}

ncclOsLibraryHandle ncclOsDlopen(const char* filename) {
  // 如果只给库名，动态链接器会按照系统配置的动态库搜索路径查找
  // RTLD_NOW：要求加载动态库时立即解析其未定义符号。（RTLD_LAZY：使用某个函数时再延迟解析。）
  // RTLD_LOCAL：表示这个动态库导出的符号默认只用于它自身及其依赖解析，不放入全局符号作用域。（RTLD_GLOBAL：符号可用于解析后续动态库的依赖。）
  ncclOsLibraryHandle handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    saveDlError();
  }
  return handle;
}

void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) {
  void* ptr = dlsym(handle, symbol);
  if (ptr == NULL) {
    saveDlError();
  }
  return ptr;
}

const char* ncclOsDlerror() {
  return ncclDlErrorBuf;
}

ncclOsLibraryHandle ncclOsDlopen(const char* path, int mode) {
  ncclOsLibraryHandle handle = dlopen(path, (mode == NCCL_OS_DL_NOW) ? RTLD_NOW : RTLD_LAZY);
  if (handle == NULL) saveDlError();
  return handle;
}

void ncclOsDlclose(ncclOsLibraryHandle handle) {
  if (handle) dlclose(handle);
}

// Process Management
uint64_t ncclOsGetPid() {
  return (uint64_t)getpid();
}

std::tm* ncclOsLocaltime(const time_t* timer, std::tm* buf) {
  return localtime_r(timer, buf);
}

uint64_t ncclOsGetTid() {
  return (uint64_t)syscall(SYS_gettid);
}

size_t ncclOsGetPageSize() {
  return (size_t)sysconf(_SC_PAGESIZE);
}

size_t ncclOsGetCommMempoolMaxSize() {
  return 0;
}

void* ncclOsAlignedAlloc(size_t alignment, size_t size) {
  return aligned_alloc(alignment, size);
}

void ncclOsAlignedFree(void* ptr) {
  free(ptr);
}

void ncclOsSetEnv(const char* name, const char* value) {
  setenv(name, value, 0);
}

char* ncclOsStrSep(char** stringp, const char* delim) {
  return strsep(stringp, delim);
}

char* ncclOsRealpath(const char* path, char* resolved_path) {
  return realpath(path, resolved_path);
}

// The default Linux stack size (8MB) is safe.
#define SAFE_STACK_SIZE (8192 * 1024)

static ncclResult_t setCpuStackSize() {
  // 获取当前线程的栈大小
  // Query the stack size used for newly launched threads.
  pthread_attr_t attr;
  size_t stackSize;
  PTHREADCHECK(pthread_attr_init(&attr), "pthread_attr_init");
  PTHREADCHECK(pthread_attr_getstacksize(&attr, &stackSize), "pthread_attr_getstacksize");

  // 如果当前栈大小小于安全栈大小 8 MiB
  if (stackSize < SAFE_STACK_SIZE) {
    // Linux/glibc 下有一个特殊情况：如果进程的 RLIMIT_STACK 是 unlimited，glibc 可能给新 pthread 使用较小的 fallback 默认栈，例如 2 MiB。
    // GNU libc normally uses RLIMIT_STACK as the default pthread stack size, unless it's set to "unlimited" --
    // in that case a fallback value of 2MB (!) is used.

    // 查询当前 stack resource limit
    // 当前默认栈配置不安全，NCCL 将为后续新建线程设置 8 MiB 默认栈
    // Query the actual resource limit so that we can distinguish between the settings of 2MB and unlimited.
    struct rlimit stackLimit;
    char buf[30];
    SYSCHECK(getrlimit(RLIMIT_STACK, &stackLimit), "getrlimit");
    if (stackLimit.rlim_cur == RLIM_INFINITY) strcpy(buf, "unlimited");
    else snprintf(buf, sizeof(buf), "%ldKB", stackLimit.rlim_cur / 1024);
    INFO(NCCL_INIT | NCCL_ENV, "Stack size limit (%s) is unsafe; will use %dKB for newly launched threads", buf,
         SAFE_STACK_SIZE / 1024);

    // 修改默认 pthread 栈大小
    // Change the default pthread stack size (via a nonportable API as the feature is not available in std::thread)
    PTHREADCHECK(pthread_attr_setstacksize(&attr, SAFE_STACK_SIZE), "pthread_attr_setstacksize");
    // pthread_setattr_default_np 是 glibc 的非标准扩展 API，_np 表示 non-portable
    PTHREADCHECK(pthread_setattr_default_np(&attr), "pthread_setattr_default_np");
  }

  PTHREADCHECK(pthread_attr_destroy(&attr), "pthread_attr_destroy");
  return ncclSuccess;
}

extern int ncclParamSetCpuStackSize();

ncclResult_t ncclOsInitialize() {
  // 根据参数 NCCL_SET_CPU_STACK_SIZE 决定是否调整后续新建 pthread 的默认栈大小，默认为 1 启用
  if (ncclParamSetCpuStackSize() != 0) {
    NCCLCHECK(setCpuStackSize());
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSetFilesLimit() {
  struct rlimit filesLimit;
  /*
  调用 getrlimit 获取当前进程的文件描述符限制：
    RLIMIT_NOFILE 表示"一个进程最多能同时打开多少个文件/socket"
  */
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  // 把软限制直接拉到硬限制。也就是说，能开多少 fd，就开到系统允许的最高值。
  filesLimit.rlim_cur = filesLimit.rlim_max;
  // 调用 setrlimit 把新的限制写回内核。
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

bool ncclOsSocketDescriptorIsValid(ncclSocketDescriptor sockDescriptor) {
  return sockDescriptor >= 0;
}

bool ncclOsSocketIsValid(struct ncclSocket* sock) {
  return ncclOsSocketDescriptorIsValid(sock->socketDescriptor);
}

extern int ncclParamPollTimeOut();
extern int ncclParamRetryTimeOut();

void ncclOsPollSocket(int socketDescriptor, int op) {
  struct pollfd pfd;
  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = socketDescriptor;
  pfd.events = (op == NCCL_SOCKET_RECV) ? POLLIN : POLLOUT;
  // 让出 CPU 一段时间，避免自旋空转
  (void)poll(&pfd, 1, ncclParamPollTimeOut());
}

extern long int ncclParamRetryCnt();

ncclResult_t ncclOsSocketTryAccept(struct ncclSocket* sock) {
  socklen_t socklen = sizeof(union ncclSocketAddress);
  /*
  调用标准的 Linux accept() 系统调用：
    sock->acceptSocketDescriptor	监听 socket 的文件描述符
    (struct sockaddr*)&sock->addr	输出缓冲区，accept 成功时填入对端地址
    &socklen	输入/输出参数，传入缓冲区大小，返回实际地址长度
  返回新的连接 socket fd，存入 sock->socketDescriptor
  */
  sock->socketDescriptor = accept(sock->acceptSocketDescriptor, (struct sockaddr*)&sock->addr, &socklen);
  if (ncclOsSocketIsValid(sock)) {
    // accept 成功，新连接的 socket fd 有效，把状态从 Accepting 推进到 Accepted
    sock->state = ncclSocketStateAccepted;
  } else if (errno == ENETDOWN || errno == EPROTO || errno == ENOPROTOOPT || errno == EHOSTDOWN || errno == ENONET ||
             errno == EHOSTUNREACH || errno == EOPNOTSUPP || errno == ENETUNREACH || errno == EINTR) {
    /*
    这一组错误（ENETDOWN、EPROTO、ENOPROTOOPT、EHOSTDOWN、ENONET、EHOSTUNREACH、EOPNOTSUPP、ENETUNREACH、EINTR）
    在 Linux accept(2) man page 中被描述为可能是已经 pending 的协议层错误，可以像 EAGAIN 一样重试。
    但 NCCL 不想无限重试，所以用 errorRetries 计数：
    */
    /* per accept's man page, for linux sockets, the following errors might be already pending errors
     * and should be considered as EAGAIN. To avoid infinite loop in case of errors, we use the retry count*/
    if (++sock->errorRetries == ncclParamRetryCnt()) {  // 34 次
      WARN("ncclOsSocketTryAccept: exceeded error retry count after %d attempts, %s", sock->errorRetries,
           strerror(errno));
      return ncclSystemError;
    }
    INFO(NCCL_NET | NCCL_INIT, "Call to accept returned %s, retrying", strerror(errno));
  } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
    // 如果错误不是 EINTR、EAGAIN、EWOULDBLOCK，说明是真正的 accept 失败（比如 fd 无效、协议错误等），直接打印 WARN 并返回 ncclSystemError。
    WARN("ncclOsSocketTryAccept: Accept failed: %s", strerror(errno));
    return ncclSystemError;
  }
  /*
  隐式分支 4：EAGAIN / EWOULDBLOCK / EINTR：
    EAGAIN / EWOULDBLOCK	非阻塞 socket 上没有待接受的连接，这是正常情况
    EINTR	accept 被信号中断，可以重试
  */
  return ncclSuccess;
}

/*
定义在 src/misc/socket.cc
*/
extern int ncclParamSocketMaxRecvBuff();
extern int ncclParamSocketMaxSendBuff();

// 给 socket 设置各种选项（非阻塞模式、TCP_NODELAY、收发缓冲区大小），优化网络通信性能
ncclResult_t ncclOsSocketSetFlags(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  const int one = 1;
  int flags;
  int rcvBuf, sndBuf;
  if (!ncclOsSocketIsValid(sock)) {
    WARN("ncclOsSocketSetFlags: invalid socket");
    ret = ncclInvalidArgument;
    goto fail;
  }
  /*
  sock->asyncFlag	异步模式：I/O 操作不阻塞，立即返回
  sock->abortFlag	需要 abort 能力：随时能中断阻塞的操作
  */
  /* Set socket as non-blocking if async or if we need to be able to abort */
  if (sock->asyncFlag || sock->abortFlag) {
    // 读取当前 flags，例如：flags = 0x2 (O_RDWR)
    SYSCHECKGOTO(flags = fcntl(sock->socketDescriptor, F_GETFL), "fcntl", ret, fail);
    // 添加 O_NONBLOCK 后写回，flags = O_RDWR | O_NONBLOCK，F_SETFL 会整体覆盖
    SYSCHECKGOTO(fcntl(sock->socketDescriptor, F_SETFL, flags | O_NONBLOCK), "fcntl", ret, fail);
  }
  /*
  Nagle 算法（默认开启）：
    没有 Nagle 算法时的问题：
      send(1 字节)  → 立即发出一个小包
      send(1 字节)  → 又一个小包
      → 网络上充满小包，浪费带宽（每个包头部 40+ 字节），低延迟

    Nagle 算法：
      send(1 字节) → 先不发，缓存起来
      等待 ACK 或攒够 MSS（最大段大小）才发
      → 减少小包数量，增加延迟
  */
 // NCCL bootstrap 传的是小控制消息（地址交换、magic 认证），对延迟极其敏感，TCP_NODELAY=1 意思是“TCP 不要延迟”，即关闭 Nagle。
  SYSCHECKGOTO(setsockopt(sock->socketDescriptor, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)),
               "setsockopt TCP NODELAY", ret, fail);
  /*
  获取 NCCL_SOCKET_RCVBUF 和 NCCL_SOCKET_SNDBUF 环境变量的值
  SO_SNDBUF	send() 把数据拷进来就返回；内核在这个队列里保存数据直到对端 ACK（因为可能要重传）。队列满了，阻塞 socket 的 send() 会阻塞，非阻塞 socket 返回 EAGAIN。
  SO_RCVBUF	网卡收到的数据先进这里等应用来取。它还直接决定 TCP 通告给对端的接收窗口（receive window）——这是它比 SO_SNDBUF 影响更大的原因。
  Linux 内核的 SO_SNDBUF/SO_RCVBUF 处理是 clamp（钳制）而非 reject（拒绝），根据如下上下限静默截断：
    sysctl net.core.wmem_max net.core.rmem_max
    net.core.wmem_max = 212992
    net.core.rmem_max = 212992
  也可以设置内核参数喜修改上下限：
    sysctl -w net.core.rmem_max=134217728
    sysctl -w net.core.wmem_max=134217728
  如果用 SO_SNDBUFFORCE/SO_RCVBUFFORCE（需要 CAP_NET_ADMIN）可以突破 *_max 限制。NCCL 没用这个，因为它不能假设自己有特权。
  内核会把值乘 2：sk->sk_sndbuf = val * 2;
  */
  // setsockopt should not fail even if the sizes are too large, do not change the default if unset by the user (=-1)
  rcvBuf = ncclParamSocketMaxRecvBuff();
  sndBuf = ncclParamSocketMaxSendBuff();
  if (sndBuf > 0) {
    SYSCHECKGOTO(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_SNDBUF, (char*)&sndBuf, sizeof(int)),
                 "setsockopt SO_SNDBUF", ret, fail);
  }
  if (rcvBuf > 0) {
    SYSCHECKGOTO(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_RCVBUF, (char*)&rcvBuf, sizeof(int)),
                 "setsockopt SO_RCVBUF", ret, fail);
  }
exit:
  return ret;
fail:
  goto exit;
}

void ncclOsSocketResetAccept(struct ncclSocket* sock) {
  // 关闭 accepted 的对等连接，并恢复监听其他连接。
  // Close the accepted peer and return to listening for another connection (see socketFinalizeAccept logging).
  (void)close(sock->socketDescriptor);
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->state = ncclSocketStateBadHandshake;
  sock->finalizeCounter = 0;
}

ncclResult_t ncclOsSocketResetFd(struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  // socket 状态初始化为 -1
  int socketDescriptor = NCCL_INVALID_SOCKET;
  /*
  创建新的 TCP socket：
    sock->addr.sa.sa_family：地址族，通常是 AF_INET 或 AF_INET6；
    SOCK_STREAM：面向连接的字节流，即 TCP；
    0：由地址族和 socket 类型自动选择默认协议。
  */
  SYSCHECKGOTO(socketDescriptor = socket(sock->addr.sa.sa_family, SOCK_STREAM, 0), "socket", ret, cleanup);
  // sock->socketDescriptor >= 0，复用原 fd 编号
  // if sock->socketDescriptor is valid, reuse its file descriptor number
  if (ncclOsSocketIsValid(sock)) {
    /*
    dup2 会关闭 sock->socketDescriptor fd 对应的旧 socket 内核对象，
    让 sock->socketDescriptor fd 指向 socketDescriptor fd 对应的新 socket 内核对象。
    好处是：能保留 fd 数值，上层结构不需要更新 descriptor，但它现在代表一个全新的 socket。
    */
    SYSCHECKGOTO(dup2(socketDescriptor, sock->socketDescriptor), "dup2", ret, cleanup);
    // 关闭临时 socketDescriptor fd，新 socket 内核对象不会被销毁，因为旧 sock->socketDescriptor fd 仍引用它。
    SYSCHECKGOTO(close(socketDescriptor), "close", ret, cleanup);
  } else {
    // 旧 fd 无效，直接采用新的 fd
    sock->socketDescriptor = socketDescriptor;
  }
  // 设置非阻塞、TCP_NODELAY、收发缓冲区大小
  NCCLCHECKGOTO(ncclOsSocketSetFlags(sock), ret, exit);
exit:
  return ret;
cleanup:
  // cleanup socketDescriptor, leave sock->socketDescriptor untouched
  if (socketDescriptor != NCCL_INVALID_SOCKET) {
    (void)close(socketDescriptor);
  }
  goto exit;
}

static ncclResult_t socketConnectCheck(struct ncclSocket* sock, int errCode, const char funcName[]) {
  char line[SOCKET_NAME_MAXLEN + 1];
  if (errCode == 0) {
    // 连接成功，接下来可以发送 magic+type 与 root 握手把状态从 Connected 推进到 Ready
    sock->state = ncclSocketStateConnected;
  } else if (errCode == EINPROGRESS) {
    // 非阻塞 connect 已发起、尚未完成。置 ConnectPolling,上层随后用 ncclOsSocketPollConnect 轮询 POLLOUT 再回本函数裁决。
    sock->state = ncclSocketStateConnectPolling;
  } else if (errCode == EINTR || errCode == EWOULDBLOCK || errCode == EAGAIN || errCode == ETIMEDOUT ||
             errCode == EHOSTUNREACH || errCode == ECONNREFUSED) {
    /*
    瞬时、可重试错误。"对端/网络只是暂时没准备好"，可恢复：
      ECONNREFUSED：bootstrap 期最常见，目标 rank 还没执行到 listen()，竞态下先连过去的就会被拒；
      EHOSTUNREACH/ETIMEDOUT：路由/网络瞬时不可达；
      EINTR：被信号打断；
      EAGAIN/EWOULDBLOCK：非阻塞资源暂不可用。
    */
   /*
   customRetry == 0 默认策略走内置重试，默认策略带"封顶 + 线性退避"。若 customRetry != 0(如 RAS 传 1)，则跳过封顶和 sleep，
   但仍执行 resetFd + 置 Connecting，把重试节奏交给调用方自管(异步友好)
   */
    if (sock->customRetry == 0) {
      /*
      封顶判定，errorRetries 从 0 起，每遇瞬时错误 +1，累计到 34 次仍失败就 return ncclRemoteError。
      退避 sleepTime = errorRetries * 100ms 是线性增长(100、200、300…ms)，累计约 sum(1..34)×100ms ≈ 59.5s，即约 1 分钟内持续重试才判死。
      NCCL_SOCKET_RETRY_CNT 默认值为 34；
      NCCL_SOCKET_RETRY_SLEEP_MSEC 默认值为 100
      */
      if (sock->errorRetries++ == ncclParamRetryCnt()) {
        sock->state = ncclSocketStateError;
        WARN("%s: connect to %s returned %s, exceeded error retry count after %d attempts", funcName,
             ncclSocketToString(&sock->addr, line), strerror(errCode), sock->errorRetries);
        return ncclRemoteError;
      }
      unsigned int sleepTime = sock->errorRetries * ncclParamRetryTimeOut();
      INFO(NCCL_NET | NCCL_INIT, "%s: connect to %s returned %s, retrying (%d/%ld) after sleep for %u msec", funcName,
           ncclSocketToString(&sock->addr, line), strerror(errCode), sock->errorRetries, ncclParamRetryCnt(),
           sleepTime);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
    }
    /*
    为什么必须 ncclOsSocketResetFd？POSIX 规定 connect 失败后该 socket 处于"未指定状态",不能直接在其上重连。
    ncclOsSocketResetFd 新建一个 TCP socket，再用 dup2(新fd, sock->socketDescriptor) 把旧 fd 号指向新 socket 内核对象，
    保留 fd 数值不变(上层结构无需更新 descriptor)，然后重设非阻塞/TCP_NODELAY/缓冲区标志。
    */
    NCCLCHECK(ncclOsSocketResetFd(sock)); /* in case of failure in connect, socket state is unspecified */
    sock->state = ncclSocketStateConnecting;
  } else {  // 致命错误
    sock->state = ncclSocketStateError;
    WARN("%s: connect to %s failed : %s", funcName, ncclSocketToString(&sock->addr, line), strerror(errCode));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSocketStartConnect(struct ncclSocket* sock) {
  // 连接远端 root 地址
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = connect(sock->socketDescriptor, &sock->addr.sa, sock->salen);
  // 检查 connect 状态
  return socketConnectCheck(sock, (ret == -1) ? errno : 0, __func__);
}

/*
异步(非阻塞)connect 的"完成探测器"。当 ncclOsSocketStartConnect 里的 connect() 返回 EINPROGRESS(状态被置为 ConnectPolling)后，
上层循环反复调用本函数，用 poll(POLLOUT) + getsockopt(SO_ERROR) 判断后台 connect 到底成功了还是失败了，再交给 socketConnectCheck 裁决状态。
*/
ncclResult_t ncclOsSocketPollConnect(struct ncclSocket* sock) {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);
  char line[SOCKET_NAME_MAXLEN + 1];

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = sock->socketDescriptor;
  /*
  关注可写，一个正在 connect 的非阻塞 socket，当连接尝试结束(无论成功或失败)时会变为"可写(POLLOUT)"。
  所以监听 POLLOUT 是异步 connect 完成检测的标准手法，可写 = "connect 有结果了"。
  */
  pfd.events = POLLOUT;
  /*
  timeout = 1(1ms)，这个超时极短，是刻意的：
    它让本函数快速返回，于是上层 do-while 每转一圈都能重新检查 abortFlag，保证 ncclCommAbort 能及时把卡在连接上的 rank 主线程拉出来；
    又不至于纯自旋空转：进内核 poll 让出 CPU 约 1ms，兼顾效率与响应性。
  */
  ret = poll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && errno == EINTR)) {
    // 超时(连接仍在进行) / poll 被信号中断
    return ncclSuccess;
  } else if (ret < 0) {
    // poll 系统调用失败
    WARN("ncclOsSocketPollConnect to %s failed with error %s", ncclSocketToString(&sock->addr, line), strerror(errno));
    return ncclSystemError;
  }
  // ret>0，socket 可写 = connect 结束

  /*
  poll 只告诉你"connect 结束了"。但绝不告诉你是成功还是失败，因为失败的 connect(如 ECONNREFUSED)同样会让 socket 变为可写(POLLOUT)
  真正的结果被内核存放在该 socket 的 SO_ERROR 选项里：
    getsockopt(SO_ERROR) 读出并清除这个待处理错误，存入 ret；
    ret == 0 → connect 成功；ret == ECONNREFUSED/ETIMEDOUT/… → connect 失败。
  */
  /* check socket status */
  SYSCHECK(getsockopt(sock->socketDescriptor, SOL_SOCKET, SO_ERROR, (void*)&ret, &rlen), "getsockopt");
  return socketConnectCheck(sock, ret, __func__);
}

/*
NCCL Linux 侧 socket I/O 的最底层原语——整个代码库里 recv()/send() 只在这一处出现，
上层所有收发 API（ncclSocketSend/Recv、握手、ncclSocketMultiOp）最终都收敛到它。
函数名中的 Progress 是核心哲学：它不承诺“完成收发”，只承诺“尽力推进一次”。是否重试、是否等待，全部交给上层。
参数	                        含义
op	                         NCCL_SOCKET_RECV 或 NCCL_SOCKET_SEND
ptr/size/offset	             缓冲区、总字节数、已完成的偏移（由调用方持有，函数只通过它回传进度）
block	                       形参：1 = 阻塞式 syscall（recv flags=0）
closed	                     出参：1 = 对端断连（有序 FIN 或 RST）
*/
ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block,
                                     int* closed) {
  int bytes = 0;
  *closed = 0;
  char* data = (char*)ptr;
  char line[SOCKET_NAME_MAXLEN + 1];
  // 异步 socket 或需要可中止的 socket，强制降级为非阻塞。但实际没有任何调用方传 block=1
  if (sock->asyncFlag || sock->abortFlag) block = 0;
  do {
    if (op == NCCL_SOCKET_RECV) {
      /*
      第 1 个参数：socket 文件描述符，指定从哪个已连接的 fd 上收数据
      第 2 个参数：接收缓冲区指针，data 是上层传入的缓冲区首地址，+ (*offset) 表示从已收到数据的末尾继续写，支持断点续传
      第 3 个参数：本次最多收多少字节，总目标字节数 size 减去已经收到的 *offset，即还剩多少字节要收
      第 4 个参数：flags，控制本次 recv 是否阻塞：block 为 1 时传 0（阻塞等待）；否则传 MSG_DONTWAIT（非阻塞，没数据立即返回 EAGAIN）
      */
      bytes = recv(sock->socketDescriptor, data + (*offset), size - (*offset), block ? 0 : MSG_DONTWAIT);
    }
    if (op == NCCL_SOCKET_SEND) {
      /*
      第 1 个参数：socket 文件描述符，指定往哪个已连接的 fd 上发数据
      第 2 个参数：发送缓冲区指针，从 data 首地址偏移 *offset 处继续发，支持断点续传
      第 3 个参数：本次最多发多少字节，剩余未发送的字节数
      第 4 个参数：flags，控制发送行为，MSG_NOSIGNAL：禁止触发 SIGPIPE 信号。MSG_DONTWAIT：本次 send 以非阻塞方式执行。
      */
      bytes = send(sock->socketDescriptor, data + (*offset), size - (*offset),
                   block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);
    }
    if (op == NCCL_SOCKET_RECV && bytes == 0) {  // recv 返回 0 的唯一语义是对端有序关闭（发出 FIN）
      *closed = 1;
      return ncclSuccess;
    }
    if (bytes == -1) {
      // 短连：send→EPIPE / recv→ECONNRESET，对端已关（写入端撞上 RST）或对端异常终止（RST）
      if ((op == NCCL_SOCKET_SEND && errno == EPIPE) || (op == NCCL_SOCKET_RECV && errno == ECONNRESET)) {
        *closed = 1;
        return ncclSuccess;
      }
      // 致命：其余（如 EBADF、ENOTCONN、ETIMEDOUT…），socket 本身坏了
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("ncclOsSocketProgressOpt: Call to %s %s failed : %s", (op == NCCL_SOCKET_RECV ? "recv from" : "send to"),
             ncclSocketToString(&sock->addr, line), strerror(errno));
        return ncclRemoteError;
      } else {
        // 良性：EINTR / EAGAIN / EWOULDBLOCK，信号中断 / 非阻塞下暂无数据可收可发
        // EAGAIN 与 EWOULDBLOCK 在 Linux 上是同一个值（11），两个都写是为了跨平台可移植；
        bytes = 0;
      }
    }
    (*offset) += bytes;
    /*
    位置在循环体内：即使同步 socket 正在连续排空一个大缓冲区（多次迭代），每次 syscall 之间都会检查一次 abort，中止响应延迟上界 = 一次 syscall；
    即使本次 EAGAIN 零推进（bytes=0），检查依然执行——空转的调用也能感知中止；
    acquire 加载与 ncclCommAbort 侧的 release 存储配对：一旦观察到 abort=1，保证也能看到中止方此前写入的所有状态（如 comm 的析构标记），不会读到半新半旧的撕裂视图；
    volatile uint32_t* 强转 std::atomic<uint32_t>*：写入方本就用原子 store，这里只是以原子语义读回，防止编译器把它缓存到寄存器里。
    */
    if (sock->abortFlag &&
        std::atomic_load_explicit((std::atomic<uint32_t>*)sock->abortFlag, std::memory_order_acquire)) {
      INFO(NCCL_NET, "ncclOsSocketProgressOpt: abort called");
      return ncclInternalError;
    }
    /*
    注意这里的循环是决定是否一次性处理指定大小的数据，退出条件：异步 || 发送队列满（没必要空转）|| 机收队列空 (没必要空转等数据) || 处理完指定数据大小了
    */
  } while (sock->asyncFlag == 0 && bytes > 0 && (*offset) < size);
  return ncclSuccess;
}

ncclResult_t ncclOsFindInterfaces(const char* prefixList, char* names, union ncclSocketAddress* addrs, int sock_family,
                                  int maxIfNameSize, int maxIfs, int* found) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';  // ^ 表示排除
  if (searchNot) prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';  // = 表示精确匹配
  if (searchExact) prefixList++;
  // 解析多个接口字符串为结构体
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  *found = 0;
  struct ifaddrs *interfaces, *interface;
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");
  // 遍历所有接口
  /*
  过滤条件：
    跳过空地址
    只支持 IPv4/IPv6
    必须 UP 且 RUNNING（物理连接的网卡）
    如果指定了地址族，必须匹配
    跳过 IPv6 回环
  */
  for (interface = interfaces; interface && *found < maxIfs; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    /* Only consider running interfaces, i.e. UP and physically attached. */
    if (!(interface->ifa_flags & IFF_RUNNING)) continue;

    TRACE(NCCL_INIT | NCCL_NET, "Found interface %s:%s", interface->ifa_name,
          ncclSocketToString((union ncclSocketAddress*)interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family) continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      // 判断是否是回环地址 ::1
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
    }

    // 匹配成功（ture）且不是排除 (false)，匹配不成功（false）且是排除（true），不会被过滤
    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^ searchNot)) {
      continue;
    }

    // 检查是否重复接口名，同一网卡可能有 IPv4 和 IPv6 两个地址，只保留第一个
    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < *found; i++) {
      // 可能多个接口名拼在一个字符串数组
      if (strcmp(interface->ifa_name, names + i * maxIfNameSize) == 0) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      // 保存接口名，按最大长度拼接到 names 字符数组后面
      // Store the interface name
      strncpy(names + (*found) * maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // 保存 IP 地址
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
      memset(addrs + *found, '\0', sizeof(*addrs));  // 初始化填充数据部分内存为 '\0'
      memcpy(addrs + *found, interface->ifa_addr, salen);  // 复制找到的网口 IP 地址
      (*found)++;  // 找到的网口数+1
    }
  }
  freeifaddrs(interfaces);
  return ncclSuccess;
}

static bool matchSubnet(struct ifaddrs local_if, union ncclSocketAddress* remote) {
  // family 匹配
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    /*
    本地子网 = 本地IP & 子网掩码
    远程子网 = 远程IP & 子网掩码
    如果 本地子网 == 远程子网 → 同子网 ✓
    */
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr == remote_subnet.s_addr);
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote->sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;  // IPv6 address is 16 unsigned char
    // 遍历字节对比，IPv6地址是16个字节构成
    for (int c = 0; c < len; c++) {
      // Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // link-local 场景可以通过 scope_id 判断是同一条链路
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they are not in the same scope
    // For Global type, this field is 0, so a comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    INFO(NCCL_NET, "Net : Unsupported address family type");
    return false;
  }
}

// 遍历本机所有网络接口，找到与目标远程地址处于同一子网的本地接口
ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
  char line_a[SOCKET_NAME_MAXLEN + 1];
#endif
  *found = 0;
  /*
  struct ifaddrs {
    struct ifaddrs  *ifa_next;          // 下一个接口
    char            *ifa_name;          // 接口名，如 "eth0"
    unsigned int     ifa_flags;         // 接口标志
    struct sockaddr *ifa_addr;          // 接口地址
    struct sockaddr *ifa_netmask;       // 子网掩码
    struct sockaddr *ifa_broadaddr;     // 广播地址
    struct sockaddr *ifa_dstaddr;       // P2P 目的地址
  };
  */
  struct ifaddrs *interfaces, *interface;
  // getifaddrs 是 Linux 系统调用，返回一个链表，每个节点描述一个网络接口的地址信息
  SYSCHECK(getifaddrs(&interfaces), "getifaddrs");
  for (interface = interfaces; interface && !*found; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue; // 结束

    // 只支持 IPv4/IPv6
    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6) continue;

    // 必须与 remoteAddr 同子网
    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // 找到相同子网的接口，保存地址
    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    memcpy(localAddr, interface->ifa_addr, salen);

    // 保存相同子网的接口名
    // Store the interface name
    strncpy(ifName, interface->ifa_name, ifNameMaxSize);

    TRACE(NCCL_INIT | NCCL_NET, "NET : Found interface %s:%s in the same subnet as remote address %s",
          interface->ifa_name, ncclSocketToString(localAddr, line), ncclSocketToString(remoteAddr, line_a));
    *found = 1;
  }

  freeifaddrs(interfaces);
  return ncclSuccess;
}

/*
OS 抽象层的 socket 关闭函数，职责是安全、优雅地关闭一个 ncclSocket 并清理其状态。
  wait: 是否等待对端关闭
*/
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait) {
  if (sock != NULL) {
    if (sock->state > ncclSocketStateNone && sock->state < ncclSocketStateNum && ncclOsSocketIsValid(sock)) {
      if (wait) {  // wait 分支：等待对端先关闭
        char data;
        bool closed = false;
        /*
        这段是优雅关闭的关键:
          以 RECV 模式反复推进接收 1 字节，直到 closed == 1——即底层 recv() 返回 0(读到 EOF),表示对端已经完成了它那一侧的关闭并发来了 FIN;
          offset 每轮重置为 0,data 只是个“垃圾桶”缓冲，目的不是读数据，而是排空对端残留数据并确认对端关闭事件；
          ncclSocketProgress 出错则 break,不无限重试——出错意味着连接已坏，无需再等；
          典型场景是双边对称关闭协议：双方都 wait=true 各自等对方 FIN，保证关闭前双方数据都已收完，避免 RST 截断数据。
        */
        do {
          int offset = 0;
          if (ncclSocketProgress(NCCL_SOCKET_RECV, sock, &data, sizeof(char), &offset, &closed) != ncclSuccess) break;
        } while (!closed);
      }
      /*
      close() 受 fd 引用计数影响：如果 fd 被 dup()/fork() 复制过(内核 file 结构引用计数 > 1),close() 只是递减计数，不会发送 FIN,对端会一直傻等；
      shutdown() 不受引用计数影响：直接作用于 socket 本身，无条件切断双向读写并触发 FIN 发送；
      SHUT_RDWR 表示彻底关闭收发两个方向，保证对端的 recv 能拿到 EOF;
      (void) 强转忽略返回值：清理路径上这两个调用几乎不可能有效失败，且失败也无从恢复，尽力而为即可。
      */
      /* shutdown() is needed to send FIN packet to proxy thread; shutdown() is not affected
       * by refcount of fd, but close() is. close() won't close a fd and send FIN packet if
       * the fd is duplicated (e.g. fork()). So shutdown() guarantees the correct and graceful
       * connection close here. */
      (void)shutdown(sock->socketDescriptor, SHUT_RDWR);
      (void)close(sock->socketDescriptor);
    }
    if (sock->crypto) {
      ncclCryptFree(sock->crypto);
      sock->crypto = nullptr;
    }
    sock->state = ncclSocketStateClosed;
    sock->socketDescriptor = NCCL_INVALID_SOCKET;
  }
  return ncclSuccess;
}

void ncclOsSetMutexCondShared(std::mutex& mutex, std::condition_variable& cond, int* initialized) {
  if (initialized != NULL && *initialized) return;
  pthread_mutexattr_t mutexAttr;
  pthread_mutexattr_init(&mutexAttr);
  pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
  pthread_mutex_t* mutexHandle = mutex.native_handle();
  pthread_mutex_init(mutexHandle, &mutexAttr);
  pthread_mutexattr_destroy(&mutexAttr);

  pthread_condattr_t condAttr;
  pthread_condattr_init(&condAttr);
  pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
  pthread_cond_t* condHandle = cond.native_handle();
  pthread_cond_init(condHandle, &condAttr);
  pthread_condattr_destroy(&condAttr);
  if (initialized != NULL) *initialized = 1;
}

void ncclOsUnsetMutexCondShared(std::mutex& mutex, std::condition_variable& cond, int* initialized) {
  if (initialized != NULL && *initialized == 0) return;
  pthread_cond_destroy(cond.native_handle());
  pthread_mutex_destroy(mutex.native_handle());
  if (initialized != NULL) *initialized = 0;
}

// 清空一个 CPU 亲和性集合
void ncclOsCpuZero(ncclAffinity& affinity) {
  // glibc 宏 CPU_ZERO(&affinity)，CPU_ZERO 要的是指针，所以取 &affinity
  CPU_ZERO(&affinity);
}

int ncclOsCpuCount(const ncclAffinity& affinity) {
  return CPU_COUNT(&affinity);
}

void ncclOsCpuSet(ncclAffinity& affinity, int cpu) {
  CPU_SET(cpu, &affinity);
}

bool ncclOsCpuIsSet(const ncclAffinity& affinity, int cpu) {
  return CPU_ISSET(cpu, &affinity);
}

ncclAffinity ncclOsCpuAnd(const ncclAffinity& a, const ncclAffinity& b) {
  ncclAffinity result;
  CPU_AND(&result, &a, &b);
  return result;
}

ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity) {
  /*
  pid = 0 在 Linux 上特指「调用线程自身」，不是「当前进程」。这是整个绑核机制作用范围的关键：
    接口	                          作用对象
    sched_getaffinity(0, ...)	     仅当前线程的亲和性
    sched_setaffinity(0, ...)	     仅当前线程的亲和性
    pthread_setaffinity_np(...)	   指定线程
    cgroup cpuset.cpus	           整个 cgroup 的所有任务
  所以只影响执行 initTransportsRank 的那个主线程，进程里其他已存在的线程完全不受影响。
  */
  int result = sched_getaffinity(0, sizeof(ncclAffinity), affinity);
  if (result == -1) {
    WARN("sched_getaffinity failed with error: %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity) {
  int result = sched_setaffinity(0, sizeof(ncclAffinity), &affinity);
  if (result == -1) {
    WARN("sched_setaffinity failed with error: %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

int ncclOsGetCpu() {
  return sched_getcpu();
}

ncclResult_t ncclOsNvmlOpen(ncclOsLibraryHandle* handle) {
  *handle = nullptr;

  // 打开 libnvidia-ml.so.1 动态库 handle
  *handle = ncclOsDlopen("libnvidia-ml.so.1");
  if (*handle == nullptr) {
    WARN("Failed to open libnvidia-ml.so.1: %s", ncclOsDlerror());
    return ncclSystemError;
  }

  INFO(NCCL_INIT, "Loaded NVML from libnvidia-ml.so.1");
  return ncclSuccess;
}

/* Topology/PCI detection functions */
#define BUSID_SIZE (sizeof("0000:00:00.0"))  // = 13（12 字符 + '\0'），对应完整 busId "domain:bus:device.function"（0000:00:00.0，12 字符）
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))  // = 8 （7  字符 + '\0'），对应前缀 "domain:bus"（0000:00，7 字符）

/*
拷贝并逐字符转小写，因为 sysfs 路径里的十六进制一律是小写，而调用方传入的 busId 可能是大写（如 NVIDIA 工具常给出 0000:3B:00.0），
必须归一化，否则 realpath 会解析失败。
*/
static void memcpylower(char* dst, const char* src, const size_t size) {
  for (size_t i = 0; i < size; i++) dst[i] = tolower(src[i]);
}

/*
给定一个 PCI 设备的 busId（如 "0000:3b:00.0"），构造出它在 sysfs 中对应的真实文件系统路径，
供后续拓扑探测（读取该设备的 PCI 层级、link 速度、NUMA 归属等）使用。

sysfs 的物理目录如下所示：
/sys/devices/pci0000:00/                      ← 根复合体（host bridge，域 0 总线 0）
    └── 0000:00:1c.0/                         ← PCIe 桥/根端口，它的次级总线就是 3b
        ├── 0000:3b:00.0/                     ← ★ 我们真正想找的设备目录
        └── pci_bus/
            └── 0000:3b/                      ← 总线 3b 自身的"总线设备"目录
root port 桥目录 0000:00:1c.0/ 下既有所有次级总线的目录（pci_bus/ 下），又有总线上的设备 0000:3b:00.0/。同时，/sys/class/pci_bus/ 下有一个符号链接指向总线目录：
/sys/class/pci_bus/0000:3b  →  ../../devices/pci0000:00/0000:00:1c.0/pci_bus/0000:3b，所以
*/
ncclResult_t ncclOsGetPciPath(const char* busId, char** path) {
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  // 根据 busId 小写填入路径模板
  memcpylower(busPath + sizeof("/sys/class/pci_bus/") - 1, busId, BUSID_REDUCED_SIZE - 1);
  memcpylower(busPath + sizeof("/sys/class/pci_bus/0000:00/../../") - 1, busId, BUSID_SIZE - 1);
  *path = realpath(busPath, NULL);  // realpath 会先解析符号链接、再按物理路径处理 ..
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

/*
这是 NCCL 拓扑探测的最底层 sysfs 读取原语：给定一个设备目录路径 path 和一个属性文件名 fileName，
把该文件的内容读进调用方缓冲区 strValue，并做去尾换行处理。上层大量用它读取 PCI 设备的 
class、max_link_speed、max_link_width、numa_node、vendor 等属性来还原拓扑。
*/
ncclResult_t ncclOsTopoGetStrFromSys(const char* path, const char* fileName, char* strValue, int maxLen) {
  char filePath[PATH_MAX];
  // 拼接完整文件路径，filePath 为：/sys/devices/pci0000:00/0000:00:1c.0/0000:3b:00.0/{fileName}
  snprintf(filePath, sizeof(filePath), "%s/%s", path, fileName);
  int offset = 0;
  FILE* file;
  // 打开并循环读取
  if ((file = fopen(filePath, "r")) != NULL) {
    // 未到 EOF（feof == 0）、无读错误（ferror == 0）、缓冲区未满（offset < maxLen），任一不满足即停
    while (feof(file) == 0 && ferror(file) == 0 && offset < maxLen) {
      // 每次最多读 maxLen - offset 字节，offset 累加已读量，实现"分块填满、绝不越界"。sysfs 属性文件通常极短（几十字节），一般一轮就读完。
      int len = fread(strValue + offset, 1, maxLen - offset, file);
      offset += len;
    }
    fclose(file);
  }
  // 收尾，区分"读到了"与"没读到"
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(NCCL_GRAPH, "Topology detection : could not read %s, ignoring", filePath);
  } else {
    // 因为 sysfs 文件内容几乎都以 \n 结尾（如 "0x060000\n"），这一步正好剥掉尾部换行，得到干净的值。
    strValue[offset - 1] = '\0';
  }
  return ncclSuccess;
}

ncclResult_t ncclOsGetPciDeviceClassByBusId(const char* busId, char* deviceClass, size_t maxLen) {
  char* path = NULL;
  // 获取指定 busId（如 "0000:3b:00.0"）在 sysfs 中对应的真实文件系统路径
  NOWARN(ncclOsGetPciPath(busId, &path), NCCL_GRAPH);
  if (path == NULL) {
    deviceClass[0] = '\0';
    return ncclSystemError;
  }
  // 读取 sysfs 中对应的真实 class 文件中的数据
  ncclResult_t ret = ncclOsTopoGetStrFromSys(path, "class", deviceClass, maxLen);
  free(path);
  return ret;
}

// 扫描 Broadcom PCIe Switch 驱动在 sysfs 导出的"虚拟交换链接"目录，找出该 Switch 下所有可 P2P 直通的 peer 设备 busId，返回一个扁平数组
ncclResult_t ncclOsGetBcmLinks(const char* busId, int* nlinks, char** peers) {
  ncclResult_t ret = ncclSuccess;
  *nlinks = 0;
  *peers = NULL;

  /*
  把 busId 的前 12 字符（完整 BDF 地址）小写化后填入。结果如：
  /sys/kernel/pci_switch_link/virtual_switch_links/0000:00:1c.0
  */
  // Path to Broadcom switch virtual links in sysfs
  char dirPath[] = "/sys/kernel/pci_switch_link/virtual_switch_links/0000:00:00.0";
  memcpylower(dirPath + sizeof("/sys/kernel/pci_switch_link/virtual_switch_links/") - 1, busId, BUSID_SIZE - 1);

  DIR* dir = opendir(dirPath);  // 打开目录并遍历
  if (dir) {
    struct dirent* file;
    while ((file = readdir(dir)) != NULL) {
      // 长度必须是 BUSID_SIZE - 1 = 12（即 "0000:00:00.0" 格式）
      // Check if the entry is a valid PCI bus ID format
      if (strlen(file->d_name) != BUSID_SIZE - 1) continue;

      // 该 busId 在 sysfs 里确实对应一个真实 PCI 设备
      // Validate that this is a real PCI device
      char* path;
      if (ncclOsGetPciPath(file->d_name, &path) != ncclSuccess) continue;
      free(path);

      /*
        NCCL 的 realloc 封装，把缓冲区从 nlinks * 13 字节扩展到 (nlinks+1) * 13 字节。每次发现一个 peer 就扩一格，
      虽然是 N 平方复杂度，但 peer 数量通常极少（一颗 Switch 下一般 2–8 个端口），性能不是问题。
      */
      // Add to peers list
      NCCLCHECKGOTO(ncclRealloc(peers, (*nlinks) * BUSID_SIZE, ((*nlinks) + 1) * BUSID_SIZE), ret, exit);
      memcpy((*peers) + BUSID_SIZE * (*nlinks)++, file->d_name, BUSID_SIZE);
    }
  }

exit:
  if (dir) closedir(dir);  // also on the realloc failure path

  return ret;
}

// 取指定 NUMA 节点的 CPU 亲和性位图（cpumap），返回十六进制掩码字符串
ncclResult_t ncclOsGetNumaNodeAffinity(unsigned int numaId, char* affinityStr, size_t maxLen, int* cpuOffset) {
  *cpuOffset = 0;
  char filePath[PATH_MAX];
  // 构造 sysfs 路径
  snprintf(filePath, sizeof(filePath), "/sys/devices/system/node/node%u/cpumap", numaId);
  int offset = 0;
  /*
  Linux 为每个 NUMA 节点在 /sys/devices/system/node/node<N>/ 下导出一系列属性文件，其中 cpumap 的内容是该节点所拥有的 CPU 位图，
  格式为逗号分隔的十六进制 32 位字（从高位到低位），例如：00000000,0000ffff,ffff0000,00000000
  每个 bit 对应一个逻辑 CPU 编号，置 1 表示该 CPU 属于此 NUMA 节点。
  */
  FILE* file = fopen(filePath, "r");
  if (file != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < (int)maxLen - 1) {
      int len = fread(affinityStr + offset, 1, maxLen - 1 - offset, file);
      offset += len;
    }
    fclose(file);
  }
  // 剥离尾部换行
  while (offset > 0 && (affinityStr[offset - 1] == '\n' || affinityStr[offset - 1] == '\r')) offset--;
  affinityStr[offset] = '\0';
  if (offset == 0) {
    INFO(NCCL_GRAPH, "Topology detection: could not read %s, using empty affinity", filePath);
  }
  return ncclSuccess;
}

// Shared memory implementation for Linux
#include "comm.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <stdlib.h>

void ncclOsShmHandleInit(ncclShmDescriptor shmDesc, char* shmPath, size_t shmSize, size_t realShmSize, char* hptr,
                         void* dptr, bool create, struct ncclShmHandleInternal* handle) {
  handle->shmDesc = shmDesc;
  handle->shmPtr = hptr;
  handle->devShmPtr = dptr;
  handle->shmSize = shmSize;
  handle->realShmSize = realShmSize;
  handle->refcount = (hptr != NULL) ? (int*)(hptr + shmSize) : NULL;
  if (create) {
    int slen = strlen(shmPath);
    handle->shmPath = (char*)malloc(slen + 1);
    memcpy(handle->shmPath, shmPath, slen + 1);
    INFO_LOC(NCCL_ALLOC_HOST, "SHM path string Size %d pointer %p", slen + 1, handle->shmPath);
    if (hptr) memset(hptr, 0, shmSize);
  } else {
    handle->shmPath = NULL;
  }
}

ncclResult_t ncclOsShmOpen(char* shmPath, size_t shmPathSize, size_t shmSize, void** shmPtr, void** devShmPtr,
                           int refcount, struct ncclShmHandleInternal** handle) {
  int fd = -1;
  char* hptr = NULL;
  void* dptr = NULL;
  ncclResult_t ret = ncclSuccess;
  struct ncclShmHandleInternal* tmphandle;
  bool create = refcount > 0 ? true : false;
  const size_t refSize = sizeof(int);
  const size_t realShmSize = shmSize + refSize;

  *handle = NULL;
  *shmPtr = NULL;
  EQCHECKGOTO(tmphandle = (struct ncclShmHandleInternal*)calloc(1, sizeof(struct ncclShmHandleInternal)), NULL, ret,
              fail);

  if (create) {
    if (shmPath[0] == '\0') {
      snprintf(shmPath, shmPathSize, "/dev/shm/nccl-XXXXXX");
    retry_mkstemp:
      fd = mkstemp(shmPath);
      if (fd < 0) {
        if (errno == EINTR) {
          INFO(NCCL_ALL, "mkstemp: Failed to create %s, error: %s (%d) - retrying", shmPath, strerror(errno), errno);
          goto retry_mkstemp;
        }
        WARN("Error: failed to create shared memory file %s, error %s (%d)", shmPath, strerror(errno), errno);
        ret = ncclSystemError;
        goto fail;
      }
    } else {
      SYSCHECKGOTO(fd = open(shmPath, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
    }

  retry_fallocate:
    if (fallocate(fd, 0, 0, realShmSize) != 0) {
      if (errno == EINTR) {
        INFO(NCCL_ALL, "fallocate: Failed to extend %s to %ld bytes, error: %s (%d) - retrying", shmPath, realShmSize,
             strerror(errno), errno);
        goto retry_fallocate;
      }
      WARN("Error: failed to extend %s to %ld bytes, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
      ret = ncclSystemError;
      goto fail;
    }
    INFO_LOC(NCCL_ALLOC_HOST, "Allocated %ld bytes of shared memory in %s", realShmSize, shmPath);
  } else {
    SYSCHECKGOTO(fd = open(shmPath, O_RDWR, S_IRUSR | S_IWUSR), "open", ret, fail);
  }

  hptr = (char*)mmap(NULL, realShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (hptr == MAP_FAILED) {
    WARN("Error: Could not map %s size %zu, error: %s (%d)", shmPath, realShmSize, strerror(errno), errno);
    ret = ncclSystemError;
    hptr = NULL;
    goto fail;
  }

  if (create) {
    *(int*)(hptr + shmSize) = refcount;
  } else {
    int remref = ncclAtomicRefCountDecrement((int*)(hptr + shmSize));
    if (remref == 0) {
      if (unlink(shmPath) != 0) {
        INFO(NCCL_ALLOC, "unlink shared memory %s failed, error: %s (%d)", shmPath, strerror(errno), errno);
      }
    }
  }

  if (devShmPtr) {
    INFO(NCCL_ALLOC,
         "SHM legacy: sharing buffer with GPU via cudaHostRegister + cudaHostGetDevicePointer (host %p size %ld)",
         (void*)hptr, (long)realShmSize);
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
    CUDACHECKGOTO(cudaHostRegister((void*)hptr, realShmSize, cudaHostRegisterPortable | cudaHostRegisterMapped), ret,
                  fail);
    CUDACHECKGOTO(cudaHostGetDevicePointer(&dptr, (void*)hptr, 0), ret, fail);
    CUDACHECKGOTO(cudaThreadExchangeStreamCaptureMode(&mode), ret, fail);
  }

  ncclOsShmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
exit:
  *shmPtr = hptr;
  if (devShmPtr) *devShmPtr = dptr;
  *handle = tmphandle;
  return ret;
fail:
  WARN("Error while %s shared memory segment %s (size %ld), error: %s (%d)", create ? "creating" : "attaching to",
       shmPath, shmSize, strerror(errno), errno);
  if (tmphandle) {
    ncclOsShmHandleInit(fd, shmPath, shmSize, realShmSize, hptr, dptr, create, tmphandle);
    (void)ncclOsShmClose(tmphandle);
    tmphandle = NULL;
  }
  hptr = NULL;
  dptr = NULL;
  goto exit;
}

ncclResult_t ncclOsShmClose(struct ncclShmHandleInternal* handle) {
  ncclResult_t ret = ncclSuccess;
  if (handle) {
    if (handle->shmDesc >= 0) {
      close(handle->shmDesc);
      if (handle->shmPath != NULL && handle->refcount != NULL && *handle->refcount > 0) {
        if (unlink(handle->shmPath) != 0) {
          WARN("unlink shared memory %s failed, error: %s (%d)", handle->shmPath, strerror(errno), errno);
          ret = ncclSystemError;
        }
      }
      free(handle->shmPath);
    }

    if (handle->shmPtr) {
      if (handle->devShmPtr) CUDACHECK(cudaHostUnregister(handle->shmPtr));
      if (munmap(handle->shmPtr, handle->realShmSize) != 0) {
        WARN("munmap of shared memory %p size %ld failed, error: %s (%d)", handle->shmPtr, handle->realShmSize,
             strerror(errno), errno);
        ret = ncclSystemError;
      }
    }
    free(handle);
  }
  return ret;
}

ncclResult_t ncclOsShmUnlink(struct ncclShmHandleInternal* handle) {
  ncclResult_t ret = ncclSuccess;
  if (handle) {
    if (handle->shmPath != NULL && handle->refcount != NULL && *handle->refcount > 0) {
      if (unlink(handle->shmPath) != 0) {
        WARN("unlink shared memory %s failed, error: %s (%d)", handle->shmPath, strerror(errno), errno);
        ret = ncclSystemError;
      }
      free(handle->shmPath);
      handle->shmPath = NULL;
    }
  }
  return ret;
}
