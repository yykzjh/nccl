/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "socket.h"
#include "utils.h"
#include "os.h"
#include "crypt.h"
#include <stdlib.h>
#include <cstdlib>

#include "param.h"
#include <time.h>
#include <mutex>

NCCL_PARAM(RetryCnt, "SOCKET_RETRY_CNT", 34);
NCCL_PARAM(RetryTimeOut, "SOCKET_RETRY_SLEEP_MSEC", 100);
NCCL_PARAM(PollTimeOut, "SOCKET_POLL_TIMEOUT_MSEC", 0);
NCCL_PARAM(SocketMaxRecvBuff, "SOCKET_RCVBUF", -1);
NCCL_PARAM(SocketMaxSendBuff, "SOCKET_SNDBUF", -1);

uint64_t ncclSocketDefaultMagic(void) {
  /* Default is the historical constant; env may override on first init. */
  static uint64_t cached = NCCL_SOCKET_MAGIC;
  static std::once_flag once;
  std::call_once(once, []() {
    const char* env = ncclGetEnv("NCCL_SOCKET_MAGIC");
    bool fromEnv = false;
    if (env != NULL && env[0] != '\0') {
      char* endptr = NULL;
      unsigned long long v = std::strtoull(env, &endptr, 0);
      if (endptr != env && endptr != NULL && *endptr == '\0') {
        cached = (uint64_t)v;
        fromEnv = true;
      } else {
        INFO(NCCL_ENV, "NCCL_SOCKET_MAGIC invalid value \"%s\", using built-in default", env);
      }
    }
    INFO(NCCL_ENV, "Socket handshake magic 0x%016llx (%s)", (unsigned long long)cached,
         fromEnv ? "NCCL_SOCKET_MAGIC" : "built-in default");
  });
  return cached;
}

void ncclSocketMove(struct ncclSocket* dst, struct ncclSocket* src) {
  memcpy(dst, src, sizeof(struct ncclSocket));
  ncclCryptRebindSocket(dst);
  src->socketDescriptor = NCCL_INVALID_SOCKET;
  src->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  src->state = ncclSocketStateNone;
  src->crypto = nullptr;
}

static ncclResult_t socketProgressRaw(int op, struct ncclSocket* sock, void* ptr, int size, int* offset,
                                      bool* pclosed = nullptr) {
  int closed;
  NCCLCHECK(ncclOsSocketProgressOpt(op, sock, ptr, size, offset, 0, &closed));
  // 输出远端关闭 socket 日志
  if (closed) {
    // 有 pclosed 参数时把关闭状态传给调用者
    if (pclosed) {
      *pclosed = true;
      return ncclSuccess;
    } else {
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("socketProgress: Connection closed by remote peer %s",
           ncclSocketToString(&sock->addr, line, /*numericHostForm*/ 0));
      return ncclRemoteError;
    }
  }
  return ncclSuccess;
}

static ncclResult_t socketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset,
                                   bool* pclosed = nullptr) {
  if (sock->crypto) {
    if (size <= *offset) return ncclSuccess;
    if (op == NCCL_SOCKET_SEND) NCCLCHECK(ncclCryptSocketSend(sock, ptr, size, offset, pclosed));
    else NCCLCHECK(ncclCryptSocketRecv(sock, ptr, size, offset, pclosed));
  } else {
    NCCLCHECK(socketProgressRaw(op, sock, ptr, size, offset, pclosed));
  }
  return ncclSuccess;
}

static ncclResult_t socketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  while (*offset < size) {  // 已经处理的字节数小于预期处理的字节数
    NCCLCHECK(socketProgress(op, sock, ptr, size, offset));
    /*
    采用 poll 系统调用等待 socket 可读或可写
    SOCKET_POLL_TIMEOUT_MSEC 可以设置超时时间，默认为0，比如设成 100，则每轮 socketProgress 没做完，就调 poll(..., 100ms)：
      事件就绪或超时，二者先到先返回
      超时后回到循环头再 socketProgress 一次
      这样释放了 CPU，代价是每次事件到达最多多一个 timeout 的等待粒度（事件就绪时 poll 会立即返回，所以正常情况下延迟影响很小）
      这个开关主要用于：非阻塞 socket + 希望降低自旋 CPU 占用的场景（比如配合 abortFlag 非阻塞 socket 时避免空转烧 CPU）。
    */
    // If we have more data to read or write, use the poll system call to wait
    // until the socket becomes readable or writable again.
    if ((*offset < size) && ncclParamPollTimeOut()) {
      // 将 fd 交给 poll 管理
      ncclOsPollSocket(sock->socketDescriptor, op);
    }
    // 否则自选忙等
  }
  return ncclSuccess;
}

uint16_t ncclSocketToPort(union ncclSocketAddress* addr) {
  return ntohs(addr->sa.sa_family == AF_INET ? addr->sin.sin_port : addr->sin6.sin6_port);
}

ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how) {
  if (sock != NULL) {
    if (ncclOsSocketIsValid(sock)) {
      SYSCHECK(shutdown(sock->socketDescriptor, how), "shutdown");
    }
    sock->state = ncclSocketStateTerminating;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* socketDescriptor) {
  if (sock == NULL) {
    WARN("ncclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (socketDescriptor) *socketDescriptor = sock->socketDescriptor;
  return ncclSuccess;
}

ncclResult_t ncclSocketSetFd(ncclSocketDescriptor socketDescriptor, struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketSetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  sock->socketDescriptor = socketDescriptor;
  return ncclSuccess;
}

ncclResult_t ncclSocketListen(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketListen: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (!ncclOsSocketIsValid(sock)) {
    WARN("ncclSocketListen: socket is invalid");
    return ncclInvalidArgument;
  }

  // 返回端口号（ntohs 把网络字节序转主机字节序）。返回 0 表示未指定端口
  if (ncclSocketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(NCCL_OS_LINUX)
    // SO_REUSEADDR 旧连接还处于 TIME_WAIT 状态的端口复用
    // SO_REUSEPORT（Linux 3.9+）更激进：允许多个 socket 同时 bind 同一端口，内核做负载均衡。主要用于高并发服务器，这里一并设置增加容错
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#endif
#elif defined(NCCL_OS_WINDOWS)
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)), "setsockopt");
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->socketDescriptor, SOL_SOCKET, SO_REUSEPORT, (char*)&opt, sizeof(opt)), "setsockopt");
#endif
#endif
  }

  // 端口值为0，不设置选项，bind 时 OS 分配随机端口
  // addr port should be 0 (Any port)
  SYSCHECK(bind(sock->socketDescriptor, &sock->addr.sa, sock->salen), "bind");

  // 获取实际端口，getsockname 查询 fd 实际绑定的地址，写回 sock->addr
  /* Get the assigned Port */
  socklen_t size = sock->salen;
  SYSCHECK(getsockname(sock->socketDescriptor, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
  TRACE(NCCL_INIT | NCCL_NET, "Listening on socket %s", ncclSocketToString(&sock->addr, line));
#endif

  /*
  listen 把 socket 从主动转为被动模式，开始监听。
    listen 前：
      socket 可以 connect()（主动外连）
      不能 accept()
    listen 后：
      socket 开始等待连接
      来的连接进入 backlog 队列
      可以 accept() 从队列取连接
  半连接+全连接队列的最大长度 backlog = 16384：
    内核为监听 socket 维护两个队列：SYN 队列（半连接）和 accept 队列（全连接）
    backlog 限制 accept 队列长度
    NCCL 设 16384 是一个非常大的值，应对大量 rank 同时连接的场景（比如几千卡训练，几千个 rank 同时 connect root）
  */
  SYSCHECK(listen(sock->socketDescriptor, 16384), "listen");

  /*
  acceptSocketDescriptor：NCCL 的 socket 结构区分两种 fd：
    socketDescriptor：数据收发用的 fd
    acceptSocketDescriptor：监听/接受连接用的 fd
  对监听 socket 来说两者相同（就是同一个 fd）
  */
  // Set acceptSocketDescriptor to the same value as socketDescriptor for listening sockets
  sock->acceptSocketDescriptor = sock->socketDescriptor;
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

// 把 ncclSocketAddress 转换成人类可读的 "IP<port>" 字符串
/* Format a string representation of a (union ncclSocketAddress *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
// numericHostForm 是否以数字形式显示 IP（1=显示 IP，0=反查主机名）
const char* ncclSocketToString(const union ncclSocketAddress* addr, char* buf, const int numericHostForm /*= 1*/) {
  // 取通用地址，getnameinfo 接受的是通用 sockaddr*，它内部会根据 sa_family 判断是 IPv4 还是 IPv6
  const struct sockaddr* saddr = &addr->sa;
  // NI_MAXHOST	1025	主机名/IP 的最大长度
  // NI_MAXSERV	32	服务名/端口的最大长度
  char host[NI_MAXHOST], service[NI_MAXSERV];
  /*
  NI_NUMERICSERV	端口以数字形式返回	返回 "12345" 而不是 "http" 之类的服务名
  NI_NUMERICHOST	IP 以数字形式返回	返回 "192.168.1.1" 而不是 "myserver.local"
  */
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  if (buf == NULL || addr == NULL) goto fail;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) goto fail;
  /*
  getnameinfo 是 getaddrinfo 的逆操作：
    getaddrinfo : 字符串 → sockaddr
    getnameinfo : sockaddr → 字符串
  */
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be determined.)
   */
  if (getnameinfo(saddr, sizeof(union ncclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag)) goto fail;
  // 格式化输出，形如：192.168.1.1<12345>
  sprintf(buf, "%s<%s>", host, service);
  return buf;
fail:
  if (buf) buf[0] = '\0';
  return buf;
}

// 允许用户指定使用 IPv4/IPv6
/* Allow the user to force the IPv4/IPv6 interface selection */
int ncclEnvSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  const char* env = ncclGetEnv("NCCL_SOCKET_FAMILY");
  if (env == NULL) return family;

  INFO(NCCL_ENV, "NCCL_SOCKET_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0) family = AF_INET6; // IPv6
  return family;
}

// 实现了一套优先级降级（fallback）策略，从最优选择到最差选择依次尝试，确保最终总能找到一块可用网卡
ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress* ifAddrs, int ifNameMaxSize, int maxIfs,
                                int* nIfs) {
  static int shownIfName = 0;
  // 通过 NCCL_SOCKET_FAMILY 环境变量可以指定 AF_INET / AF_INET6
  // Allow user to force the INET socket family selection
  int sock_family = ncclEnvSocketFamily();
  // 用户显示指定接口名，例如：eth0
  // User specified interface
  const char* env = ncclGetEnv("NCCL_SOCKET_IFNAME");
  *nIfs = 0;
  if (env && strlen(env) > 1) {
    INFO(NCCL_ENV, "NCCL_SOCKET_IFNAME set by environment to %s", env);
    // 静态变量，只在第一次调用输出日志
    // Specified by user : find or fail
    if (shownIfName++ == 0) INFO(NCCL_NET, "NCCL_SOCKET_IFNAME set to %s", env);
    // 找到所有符合筛选条件的网口以及对应IP地址
    NCCLCHECK(ncclOsFindInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
  } else {
    // 完全自动检测网口设备
    // Try to automatically pick the right one

    // 首先找 (InfiniBand）网口，接口名通常以 ib 开头，如 ib0、ib1
    // Start with IB
    NCCLCHECK(ncclOsFindInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    // else see if we can get some hint from COMM ID
    if (*nIfs == 0) {
      const char* commId = ncclGetEnv("NCCL_COMM_ID");
      if (commId && strlen(commId) > 1) {
        INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commId);
        // Try to find interface that is in the same subnet as the IP in comm id
        union ncclSocketAddress idAddr;
        // 找到与 NCCL_COMM_ID 中指定网口子网匹配的本地网口
        NCCLCHECK(ncclSocketGetAddrFromString(&idAddr, commId));
        NCCLCHECK(ncclFindInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr, ifNameMaxSize, nIfs));
      }
    }
    // 排除虚拟接口：docker 虚拟网卡、本地回环、虚拟网桥
    // Then look for anything else (but not docker,lo, or virtual)
    if (*nIfs == 0) {
      NCCLCHECK(ncclOsFindInterfaces("^docker,lo,virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    }
    // 如果前面都找不到，逐步放宽要求，甚至接受 docker 虚拟网卡或 lo 回环。这是为了在受限环境（如容器内）也能工作
    // Finally look for docker, then lo.
    if (*nIfs == 0) {
      NCCLCHECK(ncclOsFindInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    }
    if (*nIfs == 0) NCCLCHECK(ncclOsFindInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    if (*nIfs == 0) {
      NCCLCHECK(ncclOsFindInterfaces("virbr", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs, nIfs));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    WARN("Net : string is null");
    return ncclInvalidArgument;
  }

  // 判断是否为IPv6地址
  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {  // IPv4 或者主机名
    struct netIf ni;
    // 将 <ip_or_hostname>:<port> 按照 : 分割到 netIf 中，返回值为解析多少个 netIf，当前场景必须就一个
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return ncclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;  // 支持 IPv4 和 IPv6
    hints.ai_socktype = SOCK_STREAM;  // TCP

    /*
    解析主机名或者 IP，出错原因可能包括：
      EAI_NONAME：主机名不存在或无法解析
      EAI_AGAIN：DNS 临时失败
      EAI_FAIL：DNS 服务器返回失败
      EAI_FAMILY：不支持的地址族
      EAI_MEMORY：内存不足
      EAI_SYSTEM：系统错误
    */
    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      // 声明 ua->sin 字段的引用
      struct sockaddr_in& sin = ua->sin;
      /*
      将 getaddrinfo 已经填充好的 sockaddr_in 复制到 ua->sin 中，包含：
        sin_family
        sin_addr（IP 地址）
        sin_port（可能为 0，因为传入 service 是 NULL）
      */
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4 显式再设置一次，确保正确
      // inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      /*
      htons 是 Host TO Network Short：
        主机字节序（x86 是小端）→ 网络字节序（大端）
        端口号是 16 位整数，网络传输要求大端
      */
      sin.sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;                     // IPv6
      sin6.sin6_port = htons(ni.port);                 // port
      // IPv6 的流标签字段，用于标识一组需要相同处理的包。现代网络很少用它，置 0 即可
      sin6.sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      // Scope ID 用于链路本地地址（link-local address），这里 IPv4 风格的主机名解析路径
      sin6.sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      freeaddrinfo(p);
      return ncclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    // 例如 "[fe80::1%eth0]:12345"
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;  // 记录 % 位置
      if (ip_port_pair[i] == ']') break;  // 找到结束括号
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return ncclInvalidArgument;
    }
    bool global_scope = (j == -1 ? true : false);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[16];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair + 1, global_scope ? i - 1 : j - 1);  // IP 部分
    strncpy(port_str, ip_port_pair + i + 2, len - i - 1);  // 端口部分
    int port = atoi(port_str);
    if (!global_scope) {
      // If not global scope, we need the intf name
      strncpy(if_name, ip_port_pair + j + 1, i - j - 1);  // 接口名，如 "eth0"
    }

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    // if_nametoindex("eth0") 把接口名转成数字索引，链路本地地址（如 fe80::1）必须指定这个，否则内核不知道走哪块网卡。
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr) {
  if (sock == NULL) {
    WARN("ncclSocketGetAddr: pass NULL socket");
    return ncclInvalidArgument;
  }
  // 需要确保 socket 初始化已经完成
  if (sock->state != ncclSocketStateReady) return ncclInternalError;
  // 把实际地址写回 addr
  memcpy(addr, &sock->addr, sizeof(union ncclSocketAddress));
  return ncclSuccess;
}

static void socketResetAccept(struct ncclSocket* sock) {
  ncclOsSocketResetAccept(sock);
  ncclCryptFree(sock->crypto);
  sock->crypto = nullptr;
}

static ncclResult_t socketFinalizeAccept(struct ncclSocket* sock) {
  NCCLCHECK(ncclOsSocketSetFlags(sock));

  ncclResult_t ret;
  do {
    // Runs the TLS handshake and reads the hello in encrypted mode, or reads the
    // stock plaintext hello otherwise; all of the socket I/O lives in crypt.cc.
    enum ncclCryptHelloVerdict verdict;
    NCCLCHECK(ret = ncclCryptAcceptHello(sock, &verdict));
    if (verdict == NCCL_CRYPT_HELLO_VERDICT_RESET) {
      socketResetAccept(sock);
      return ncclSuccess;
    }
    if (verdict == NCCL_CRYPT_HELLO_VERDICT_READY) {
      sock->state = ncclSocketStateReady;
      return ncclSuccess;
    }
  } while (sock->asyncFlag == 0 && ret == ncclInProgress);
  return ncclSuccess;
}

ncclResult_t ncclSocketPollConnect(struct ncclSocket* sock) {
  if (sock == NULL) {
    WARN("ncclSocketPollConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(ncclOsSocketPollConnect(sock));
  return ncclSuccess;
}

static ncclResult_t socketFinalizeConnect(struct ncclSocket* sock) {
  ncclResult_t ret;
  do {
    NCCLCHECK(ret = ncclCryptConnectHello(sock));
  } while (sock->asyncFlag == 0 && ret == ncclInProgress);
  if (ret == ncclInProgress) return ncclSuccess;
  sock->state = ncclSocketStateReady;
  return ncclSuccess;
}

// 根据当前 sock->state，调用对应的处理函数把连接推进到下一个状态。
// 关键设计：一次调用可能推进多个状态
static ncclResult_t socketProgressState(struct ncclSocket* sock) {
  /*
  这是服务端的第一步。底层调用 accept()：
    如果有客户端连上来了：创建新连接 fd，状态推进到 Accepted
    如果暂时没有连接：在非阻塞 socket 上返回 EAGAIN/EWOULDBLOCK，状态保持 Accepting
    函数本身返回 ncclSuccess（即使还没连上），由外层循环决定是否继续等
  */
  if (sock->state == ncclSocketStateAccepting) {
    NCCLCHECK(ncclOsSocketTryAccept(sock));
  }
  /*
  accept 成功后，需要完成 NCCL 的握手认证。这个函数会：
    接收对方发送的 magic
    接收对方发送的 socket type
    和本端预期的 magic / type 比对
    如果匹配：状态推进到 Ready
    如果不匹配：状态变成 BadHandshake，随后丢弃这个连接
    在同步模式下，socketFinalizeAccept 内部会阻塞等待接收完整的 magic；在异步模式下，它用 finalizeCounter 记录已接收字节数，分多次推进。
  */
  if (sock->state == ncclSocketStateAccepted) {
    NCCLCHECK(socketFinalizeAccept(sock));
  }
  /*
  这是客户端的第一步。底层调用 connect()：
    如果 connect 立即成功：状态推进到 Connected
    如果 connect 正在建立中（非阻塞 socket 返回 EINPROGRESS）：状态推进到 ConnectPolling
    如果失败：进入 Error
  */
  if (sock->state == ncclSocketStateConnecting) {
    NCCLCHECK(ncclOsSocketStartConnect(sock));
  }
  /*
  connect 发起后进入阻塞/轮询阶段。这个函数会：
    用 select/poll 检查 socket 是否可写
    通过 getsockopt(SO_ERROR) 检查 connect 是否真正成功
    成功则推进到 Connected
    失败则进入 Error
  */
  if (sock->state == ncclSocketStateConnectPolling) {
    NCCLCHECK(ncclOsSocketPollConnect(sock));
  }
  /*
  客户端 connect 成功后，需要主动发送自己的 magic 和 type 给服务端，完成握手：
    发送 magic
    发送 type
    发送完成后状态推进到 Ready
    和 socketFinalizeAccept 对称，一个负责收，一个负责发。
  */
  if (sock->state == ncclSocketStateConnected) {
    NCCLCHECK(socketFinalizeConnect(sock));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* running) {
  if (sock == NULL) {
    *running = 0;
    return ncclSuccess;
  }
  if (sock->state == ncclSocketStateError || sock->state == ncclSocketStateClosed) {
    WARN("ncclSocketReady: unexpected socket state %d", sock->state);
    return ncclRemoteError;
  }
  *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  if (*running == 0) {
    NCCLCHECK(socketProgressState(sock));
    if (sock->state == ncclSocketStateBadHandshake) sock->state = ncclSocketStateAccepting;
    *running = (sock->state == ncclSocketStateReady) ? 1 : 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(struct ncclSocket* sock) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif

  if (sock == NULL) {
    WARN("ncclSocketConnect: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (!ncclOsSocketIsValid(sock)) {
    WARN("ncclSocketConnect: socket is invalid");
    return ncclInvalidArgument;
  }

  if (sock->state != ncclSocketStateInitialized) {
    WARN("ncclSocketConnect: wrong socket state %d", sock->state);
    if (sock->state == ncclSocketStateError) return ncclRemoteError;
    return ncclInternalError;
  }
  TRACE(NCCL_INIT | NCCL_NET, "Connecting to socket %s", ncclSocketToString(&sock->addr, line));

  NCCLCHECK(ncclCryptStartConnect(sock));

  // 初始化 sock 状态和异步缓冲区计数
  sock->state = ncclSocketStateConnecting;
  sock->finalizeCounter = 0;
  do {
    // 推进 sock 状态
    NCCLCHECK(socketProgressState(sock));
    /*
    同步模式下一般内部还有循环，到这里收发数据完成。异步模式下内部收发数据不一定完成，到达这里的粒度会更细。
    但在这里无论同步还是异步都会检测是否 abort 以及状态是否达到最终状态
    */
  } while (sock->asyncFlag == 0 &&
           (sock->abortFlag == NULL || COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire) == 0) &&
           (sock->state == ncclSocketStateConnecting || sock->state == ncclSocketStateConnectPolling ||
            sock->state == ncclSocketStateConnected));

  // abort
  if (sock->abortFlag && COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire)) return ncclInternalError;

  switch (sock->state) {
  case ncclSocketStateConnecting:
  case ncclSocketStateConnectPolling:
  case ncclSocketStateConnected:
  case ncclSocketStateReady:
    // 中间状态和最终状态都是成功返回
    return ncclSuccess;
  case ncclSocketStateError:
    return ncclSystemError;
  default:
    WARN("ncclSocketConnect: wrong socket state %d", sock->state);
    return ncclInternalError;
  }
}

// 在一个监听 socket 上接受一个客户端连接并完成握手
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* listenSock, bool retry) {
  ncclResult_t ret = ncclSuccess;

  if (listenSock == NULL || sock == NULL) {
    WARN("ncclSocketAccept: pass NULL socket");
    ret = ncclInvalidArgument;
    goto exit;
  }
  // 监听 socket 必须已经初始化并处于 Ready 状态
  if (listenSock->state != ncclSocketStateReady) {
    WARN("ncclSocketAccept: wrong socket state %d", listenSock->state);
    if (listenSock->state == ncclSocketStateError) ret = ncclSystemError;
    else ret = ncclInternalError;
    goto exit;
  }

  // 如果 sock 还没有被初始化为 accept socket（通过 acceptSocketDescriptor 判断），
  // 就把整个 listenSock 复制过来，然后把自己的状态改为 ncclSocketStateAccepting。
  if (!ncclOsSocketDescriptorIsValid(sock->acceptSocketDescriptor)) {
    // Clone the listener configuration; the accepted socket owns its own TLS state.
    memcpy(sock, listenSock, sizeof(struct ncclSocket));
    sock->acceptSocketDescriptor = listenSock->acceptSocketDescriptor;
    sock->state = ncclSocketStateAccepting;
    sock->finalizeCounter = 0;
    sock->crypto = nullptr;
  }

  do {
    /*
    NCCL socket 抽象层的核心状态机推进函数。它的作用非常单纯：根据当前 sock->state，调用对应的处理函数把连接推进到下一个状态。
    */
    NCCLCHECKGOTO(socketProgressState(sock), ret, exit);
    /*
    如果握手失败进入 BadHandshake 状态：
      把状态重置回 Accepting
      如果 retry == false，跳出循环，不再重试
      如果 retry == true，继续循环，重新尝试接受和握手
    */
    if (sock->state == ncclSocketStateBadHandshake) {
      // Most likely some issue with magic.  We will retry from the beginning, unless the caller requested not to.
      sock->state = ncclSocketStateAccepting;
      if (!retry) break;
    }
  /*
  循环继续的条件是：
    sock->asyncFlag == 0：当前是同步模式，异步模式下不阻塞循环
    未被中止：abortFlag 为空或未被设置
    状态还没稳定：仍处于 Accepting 或 Accepted，一旦变成 Ready/Error 就退出
  */
  } while (sock->asyncFlag == 0 &&
           (sock->abortFlag == NULL || COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire) == 0) &&
           (sock->state == ncclSocketStateAccepting || sock->state == ncclSocketStateAccepted));

  // 检查是否被中止
  if (sock->abortFlag && COMPILER_ATOMIC_LOAD(sock->abortFlag, std::memory_order_acquire)) return ncclInternalError;

  /*
  根据状态机最终状态返回：
    Accepting / Accepted / Ready → 成功
    Error → 系统错误
    其他异常状态 → 内部错误
  */
  switch (sock->state) {
  case ncclSocketStateAccepting:
  case ncclSocketStateAccepted:
  case ncclSocketStateReady:
    ret = ncclSuccess;
    break;
  case ncclSocketStateError:
    ret = ncclSystemError;
    break;
  default:
    WARN("ncclSocketAccept: wrong socket state %d", sock->state);
    ret = ncclInternalError;
    break;
  }

exit:
  return ret;
}

ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr, uint64_t magic,
                            enum ncclSocketType type, volatile uint32_t* abortFlag, int asyncFlag, int customRetry) {
  ncclResult_t ret = ncclSuccess;

  if (sock == NULL) goto exit;
  sock->errorRetries = 0;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->state = ncclSocketStateInitialized;
  sock->magic = magic;
  sock->type = type;
  sock->socketDescriptor = NCCL_INVALID_SOCKET;
  sock->acceptSocketDescriptor = NCCL_INVALID_SOCKET;
  sock->customRetry = customRetry;
  sock->finalizeCounter = 0;
  sock->crypto = nullptr;
#ifdef NCCL_OS_WINDOWS
  sock->socketBlockingMode = 1;
#endif

  // 传入了 addr 需要创建一个 socket
  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    // 拷贝 addr
    memcpy(&sock->addr, addr, sizeof(union ncclSocketAddress));
    family = sock->addr.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("ncclSocketInit: connecting to address %s with family %d is neither AF_INET(%d) nor AF_INET6(%d)",
           ncclSocketToString(&sock->addr, line), family, AF_INET, AF_INET6);
      ret = ncclInternalError;
      goto exit;
    }
    sock->salen = (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    /*
    内部调用 socket(family, SOCK_STREAM, 0)，向 OS 申请一个 TCP socket，返回一个文件描述符 fd（比如 fd=5）
    */
    // in case of error, we close the descriptor before returning as it's unclear if the caller has to
    // use ncclSocketClose for cleanup
    NCCLCHECKGOTO(ncclOsSocketResetFd(sock), ret, fail);
  } else {
    // 如果没传入 addr，则只做初始化，socket 从 accept 获取
    memset(&sock->addr, 0, sizeof(union ncclSocketAddress));
  }
exit:
  return ret;
fail:
  (void)ncclSocketClose(sock);
  goto exit;
}

ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, bool* closed) {
  if (sock == NULL) {
    WARN("ncclSocketProgress: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketProgress(op, sock, ptr, size, offset, closed));
  return ncclSuccess;
}

ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset) {
  if (sock == NULL) {
    WARN("ncclSocketWait: pass NULL socket");
    return ncclInvalidArgument;
  }
  NCCLCHECK(socketWait(op, sock, ptr, size, offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketSend: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (sock->state != ncclSocketStateReady) {
    WARN("ncclSocketSend: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  NCCLCHECK(socketWait(NCCL_SOCKET_SEND, sock, ptr, size, &offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size) {
  int offset = 0;
  // 检查 ncclSocket
  if (sock == NULL) {
    WARN("ncclSocketRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  // 确保 sock 状态合法
  if (sock->state != ncclSocketStateReady && sock->state != ncclSocketStateTerminating) {
    WARN("ncclSocketRecv: socket state (%d) is not ready", sock->state);
    return ncclInternalError;
  }
  // 同步自旋或 poll 接收数据
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, sock, ptr, size, &offset));
  return ncclSuccess;
}

ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock,
                                void* recvPtr, int recvSize) {
  int sendOffset = 0, recvOffset = 0;
  if (sendSock == NULL || recvSock == NULL) {
    WARN("ncclSocketSendRecv: invalid socket %p/%p", sendSock, recvSock);
    return ncclInternalError;
  }
  if (sendSock->state != ncclSocketStateReady ||
      (recvSock->state != ncclSocketStateReady && recvSock->state != ncclSocketStateTerminating)) {
    WARN("ncclSocketSendRecv: socket state (%d/%d) is not ready", sendSock->state, recvSock->state);
    return ncclInternalError;
  }
  // 循环发送和接收数据，内部采用非阻塞调用
  while (sendOffset < sendSize || recvOffset < recvSize) {
    if (sendOffset < sendSize) NCCLCHECK(socketProgress(NCCL_SOCKET_SEND, sendSock, sendPtr, sendSize, &sendOffset));
    if (recvOffset < recvSize) NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, recvSock, recvPtr, recvSize, &recvOffset));
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps) {
  if (ops == NULL || numOps <= 0) {
    WARN("ncclSocketMultiOp: invalid arguments ops=%p numOps=%d", ops, numOps);
    return ncclInvalidArgument;
  }

  // 先初始化异步收发的偏移记录为 0
  for (int i = 0; i < numOps; i++) {
    if (ops[i].sock == NULL) {
      WARN("ncclSocketMultiOp: invalid socket at index %d", i);
      return ncclInvalidArgument;
    }
    ops[i].offset = 0;
  }
  int completedOps = 0, i = 0;
  /*
  轮转 i=(i+1)%numOps 让 4 个 op 交替推进、谁就绪推谁，没有任何一个 op 能独占阻塞别人，
  这就是在 2 条全双工连接上、单线程里同时收发 4 路的"迷你 reactor"。
  */
  while (completedOps < numOps) {
    // 如果收发数据偏移小于数据大小说明操作没有完成
    if (ops[i].offset < ops[i].size) {
      NCCLCHECK(socketProgress(ops[i].op, ops[i].sock, ops[i].ptr, ops[i].size, &ops[i].offset));
      if (ops[i].offset >= ops[i].size) completedOps++;
    }
    i = (i + 1) % numOps;
  }
  return ncclSuccess;
}

// Receive or detect connection closed
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, bool* closed, bool blocking) {
  int offset = 0;
  if (sock == NULL) {
    WARN("ncclSocketTryRecv: pass NULL socket");
    return ncclInvalidArgument;
  }
  *closed = false;
  NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, ptr, size, &offset, closed));
  if (*closed) return ncclSuccess;
  if (offset == 0 && !blocking) return ncclInProgress;
  while (offset < size) {
    NCCLCHECK(socketProgress(NCCL_SOCKET_RECV, sock, ptr, size, &offset, closed));
    if (*closed) return ncclSuccess;
  }
  return ncclSuccess;
}
