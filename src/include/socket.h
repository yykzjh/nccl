/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include "nccl.h"
#include "os.h"

#if !defined(__CUDA_ARCH__)
/* Ensure full socket types are visible when this header is included from .cu or other TU without os.h socket context. */
#if defined(NCCL_OS_WINDOWS)
#include <winsock2.h>
#include <ws2tcpip.h>
#elif defined(NCCL_OS_LINUX)
#include <sys/socket.h>
#include <netinet/in.h>
#endif
#endif

#define MAX_IFS 16
#if NCCL_OS_LINUX
#define MAX_IF_NAME_SIZE 16
#elif NCCL_OS_WINDOWS
#define MAX_IF_NAME_SIZE 64
#endif
#if defined(__CUDA_ARCH__) || (!defined(NCCL_OS_WINDOWS) && !defined(NCCL_OS_LINUX))
/* Device compilation or stub build (no OS): no system socket headers; use placeholder for union size. */
#ifndef NI_MAXHOST
#define NI_MAXHOST 1024
#endif
#ifndef NI_MAXSERV
#define NI_MAXSERV 32
#endif
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define NCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

// 实际使用时被强制类型转换为系统的 sockaddr 类型 (struct sockaddr_in*)
/* Placeholder; real definition needs sockaddr from system headers */
union ncclSocketAddress {
  char _storage[28];
};
#else
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define NCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union ncclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};
#endif

struct ncclSocketCrypto;

enum ncclSocketState {
  ncclSocketStateNone = 0,
  ncclSocketStateInitialized = 1,
  ncclSocketStateAccepting = 2,  // 服务端：等待 accept
  ncclSocketStateAccepted = 3,  // 服务端：已 accept，待握手
  ncclSocketStateConnecting = 4,  // 客户端：开始 connect
  ncclSocketStateConnectPolling = 5,  // 客户端：connect 阻塞中，轮询
  ncclSocketStateConnected = 6,  // 客户端：connect 完成，待握手
  ncclSocketStateReady = 7,  // 双方：连接建立且握手完成
  ncclSocketStateTerminating = 8,
  ncclSocketStateClosed = 9,
  ncclSocketStateError = 10,
  ncclSocketStateBadHandshake = 11,  // 握手失败（通常是 magic 不匹配）
  ncclSocketStateNum = 12
};

enum ncclSocketType {
  ncclSocketTypeUnknown = 0,
  ncclSocketTypeBootstrap = 1,
  ncclSocketTypeProxy = 2,
  ncclSocketTypeNetSocket = 3,
  ncclSocketTypeNetIb = 4,
  ncclSocketTypeRasNetwork = 5,
  ncclSocketTypeNetNd = 6
};

// The stock hello: socket magic + type, sent by the connector as the first
// application data. In encrypted mode it travels inside the TLS channel.
#define NCCL_SOCKET_PLAIN_HELLO_BYTES (sizeof(uint64_t) + sizeof(enum ncclSocketType))

// Contains owned OS and TLS state. Do not copy with raw struct assignment or
// memcpy; use ncclSocketMove when transferring socket ownership.
/*
NCCL 统一的 TCP socket 抽象——把 IPv4/IPv6、同步/异步、连接握手、可中止、可重试全部封装进一个结构
*/
struct ncclSocket {
  // 数据收发 fd(accept 后 = 新连接 fd)
  ncclSocketDescriptor socketDescriptor;
  // 监听 fd(accept 的来源)
  ncclSocketDescriptor acceptSocketDescriptor;
  // 瞬时错误重试计数
  int errorRetries;
  // 	IPv4/IPv6 统一地址。connect 时是对端地址,listen 时是本地绑定地址,accept 时被内核填入对端地址
  union ncclSocketAddress addr;
  volatile uint32_t* abortFlag;
  // 同步阻塞 or 异步增量
  int asyncFlag;
  enum ncclSocketState state;
  /*
  sockaddr 长度(随地址族变)。IPv4 是 sizeof(sockaddr_in)、IPv6 是 sizeof(sockaddr_in6)，
  传给 bind/connect/accept/getsockname 这些需要地址长度的系统调用。
  */
  int salen;
  // 连接状态机
  uint64_t magic;
  // socket 用途类别(bootstrap/proxy/…)
  enum ncclSocketType type;
  // 是否关闭内置重试、交调用方自管
  int customRetry;
  // 异步握手已传输字节数
  int finalizeCounter; // Used to keep track of initial handshake for async sockets.
  // 异步收 magic 的中转缓冲
  char finalizeBuffer[NCCL_SOCKET_PLAIN_HELLO_BYTES]; // Used to keep track of initial handshake for async sockets.
  struct ncclSocketCrypto* crypto;
#ifdef NCCL_OS_WINDOWS
  int socketBlockingMode; // 0 - blocking mode; 1 - non-blocking mode
#endif
};
struct ncclSocketOp {
  int op; // NCCL_SOCKET_SEND or NCCL_SOCKET_RECV
  struct ncclSocket* sock; // Socket to operate on
  void* ptr; // Data pointer
  int size; // Size of data
  int offset; // Current progress offset
};
const char* ncclSocketToString(const union ncclSocketAddress* addr, char* buf, const int numericHostForm = 1);
ncclResult_t ncclSocketGetAddrFromString(union ncclSocketAddress* ua, const char* ip_port_pair);
ncclResult_t ncclFindInterfaceMatchSubnet(char* ifName, union ncclSocketAddress* localAddr,
                                          union ncclSocketAddress* remoteAddr, int ifNameMaxSize, int* found);
ncclResult_t ncclFindInterfaces(char* ifNames, union ncclSocketAddress* ifAddrs, int ifNameMaxSize, int maxIfs,
                                int* nIfs);

// Magic used for NCCL-internal TCP handshakes (bootstrap uses comm magic separately). Honors NCCL_SOCKET_MAGIC env.
uint64_t ncclSocketDefaultMagic(void);

// Initialize a socket
ncclResult_t ncclSocketInit(struct ncclSocket* sock, const union ncclSocketAddress* addr = NULL,
                            uint64_t magic = ncclSocketDefaultMagic(), enum ncclSocketType type = ncclSocketTypeUnknown,
                            volatile uint32_t* abortFlag = NULL, int asyncFlag = 0, int customRetry = 0);
// Move an initialized socket into an uninitialized destination.
void ncclSocketMove(struct ncclSocket* dst, struct ncclSocket* src);
// Create a listening socket. sock->addr can be pre-filled with IP & port info. sock->fd is set after a successful call
ncclResult_t ncclSocketListen(struct ncclSocket* sock);
ncclResult_t ncclSocketGetAddr(struct ncclSocket* sock, union ncclSocketAddress* addr);
// Connect to sock->addr. sock->socketDescriptor is set after a successful call.
ncclResult_t ncclSocketConnect(struct ncclSocket* sock);
// Return socket connection state.
ncclResult_t ncclSocketReady(struct ncclSocket* sock, int* running);
// Accept an incoming connection from listenSock->socketDescriptor and keep the file descriptor in
// sock->socketDescriptor, with the remote side IP/port in sock->addr.
ncclResult_t ncclSocketAccept(struct ncclSocket* sock, struct ncclSocket* ulistenSock, bool retry = true);
ncclResult_t ncclSocketGetFd(struct ncclSocket* sock, ncclSocketDescriptor* socketDescriptor);
ncclResult_t ncclSocketSetFd(ncclSocketDescriptor socketDescriptor, struct ncclSocket* sock);

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

int ncclEnvSocketFamily(void);
ncclResult_t ncclSocketProgress(int op, struct ncclSocket* sock, void* ptr, int size, int* offset,
                                bool* closed = nullptr);
ncclResult_t ncclSocketWait(int op, struct ncclSocket* sock, void* ptr, int size, int* offset);
ncclResult_t ncclSocketSend(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketRecv(struct ncclSocket* sock, void* ptr, int size);
ncclResult_t ncclSocketSendRecv(struct ncclSocket* sendSock, void* sendPtr, int sendSize, struct ncclSocket* recvSock,
                                void* recvPtr, int recvSize);
ncclResult_t ncclSocketMultiOp(struct ncclSocketOp* ops, int numOps);
ncclResult_t ncclSocketTryRecv(struct ncclSocket* sock, void* ptr, int size, bool* closed, bool blocking);
ncclResult_t ncclSocketShutdown(struct ncclSocket* sock, int how);
ncclResult_t ncclSocketClose(struct ncclSocket* sock, bool wait = false);
uint16_t ncclSocketToPort(union ncclSocketAddress* addr);
#endif
