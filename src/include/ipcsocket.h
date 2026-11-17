/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_IPCSOCKET_H
#define NCCL_IPCSOCKET_H

#include "nccl.h"
#include <stdio.h>
#include "os.h"
#include <errno.h>
#include <memory.h>
#include <inttypes.h>

#define NCCL_IPC_SOCKNAME_LEN 64

#if defined(NCCL_OS_WINDOWS)
typedef intptr_t ncclIpcFd;
#else
typedef int ncclIpcFd;
#endif

#define NCCL_INVALID_IPC_FD ((ncclIpcFd) - 1)

/*
AF_UNIX 数据报 socket 的文件描述符。这是结构里唯一的 OS 资源，另两个字段都是围绕它的元数据。
写入：ncclIpcSocketInit 里 socket(AF_UNIX, SOCK_DGRAM, 0) + bind() 成功后 handle->fd = fd
消费：
  recvmsg(handle->fd, &msg, 0) / sendmsg(handle->fd, &msg, 0)，收发数据报，辅助数据里携带 SCM_RIGHTS 的 fd。
  ncclIpcSocketGetFd(handle, &fd) 取出它，UDS 服务线程放进 poll()。
  ncclIpcSocketClose 里 close(handle->fd)。
*/
struct ncclIpcSocket {
  ncclIpcFd fd;
  /*
  自己的绑定名，长度 64 字节，格式为 "/tmp/nccl-socket-%d-%" PRIx64 (rank + 64 位 hash 的十六进制)。
  消费：
    ncclIpcSocketClose：非 abstract 模式下 unlink(handle->socketName) 删除 /tmp 下的 socket 文件。
    ncclIpcSocketRecvMsg 的 WARN 和 TRACE 日志里打印，便于定位端点。
  它不是发送目的地址，ncclIpcSocketSendMsg 发送时，目的地址是用入参(rank, 对端 hash)现场 snprintf 进一个局部 cliaddr 的，
  与 handle->socketName 无关。所以 socketName 存的是"我自己绑到哪个名"，仅供清理 + 日志，对端名每次发送时按 (rank, hash) 重新拼。

  Unix 域套接字靠什么定位？
  进程间通信时，一个 socket 要能被别人找到，就得有个"地址"。对 Unix 域套接字(AF_UNIX)，地址结构是 struct sockaddr_un，核心字段是 sun_path。
  sun_path 怎么填，就分出了两种命名空间。
    1. pathname(路径名)命名空间，传统方式。sun_path 填一个真实的文件系统路径，比如 /tmp/nccl-socket-0-abc123。
  当 bind() 到它时，内核会在文件系统里真的创建一个文件(一个特殊的"socket 文件"，ls -l 会看到类型是 s)。别的进程通过这个路径找到它。
    缺点：这个文件在 socket 关闭后不会自动消失，留在 /tmp 里，必须自己 unlink() 删掉它，否则:
      残留文件越积越多；
      下次用同名 bind 会失败(Address already in use)；
      还受文件系统权限、是否有可写目录等约束。
    2. abstract(抽象)命名空间，Linux 特有的扩展。如果把 sun_path[0] 设成 '\0'(空字节)，
  那么后面的字节就被当作一个只存在于内核内存里的"抽象名字"，不在文件系统里创建任何文件。
  例如名字字符串本来是 /tmp/nccl-socket-...，把第 0 个字节 / 改成 \0 后，
  它就从"文件路径"变成了"抽象名字" tmp/nccl-socket-...(内核只认 \0 后面那串字节当名字，不再当文件路径)。
    优点:
      1. 不建文件，没有 /tmp 残留，socket 一关(或进程一退出)名字自动释放，无需 unlink；
      2. 不受文件系统权限/可写目录限制；
      3. 对容器更友好(不依赖挂载的 /tmp)。
    缺点：
      1. Linux 专有，abstract namespace 不是 POSIX 标准，BSD/macOS 没有，
    所以用 NCCL_IPC_USE_ABSTRACT_SOCKET 开关以便在不支持的环境回退到 pathname 模式；
      2. 作用域是 network namespace(不是文件系统/挂载命名空间)，同一个 network namespace 里的进程能看到彼此的抽象 socket，
    不同 network namespace 的容器互相看不到。这对容器化部署有影响。
      3. 没有文件可见性，无法用 ls 看到它(调试时不如 pathname 直观)，但换来了干净。
      4. 访问控制不由文件权限管，任何在同 network namespace、知道名字的进程都能连。
    NCCL 靠在名字里放一个随机 hash(getUDS 生成的)来降低被猜中/串扰的风险。
  */
  char socketName[NCCL_IPC_SOCKNAME_LEN];
  /*
  作用一：置非阻塞，即决定 fd 是否非阻塞。
  作用二：recv/send 的 while 循环里，遇 EAGAIN/EWOULDBLOCK/EINTR 不报错而重试，于是卡住的 UDS 收发能被拉出。
  */
  volatile uint32_t* abortFlag;
};

ncclResult_t ncclIpcSocketInit(struct ncclIpcSocket* handle, int rank, uint64_t hash, volatile uint32_t* abortFlag);
ncclResult_t ncclIpcSocketClose(struct ncclIpcSocket* handle);
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, ncclIpcFd* fd);

ncclResult_t ncclIpcSocketRecvFd(struct ncclIpcSocket* handle, ncclIpcFd* fd);
ncclResult_t ncclIpcSocketSendFd(struct ncclIpcSocket* handle, ncclIpcFd fd, int rank, uint64_t hash);

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket* handle, void* hdr, int hdrLen, ncclIpcFd sendFd, int rank,
                                  uint64_t hash);
ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket* handle, void* hdr, int hdrLen, ncclIpcFd* recvFd);
int ncclIpcFdClose(ncclIpcFd fd);

#endif /* NCCL_IPCSOCKET_H */
