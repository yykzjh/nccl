/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os_socket_pair.h"
#include "checks.h"

#include <unistd.h>

/*
在 Linux 上用 pipe() 建一条匿名、单向管道，pair[0]=读端、pair[1]=写端。
它是 RAS self-pipe trick 的底层原语，把"唤醒事件"从任意 NCCL 线程安全投递给阻塞在 poll() 里的 RAS 线程。
*/
// TODO: switch from pipe() to socketpair(AF_UNIX, SOCK_STREAM, 0) to align with the Windows implementation
ncclResult_t ncclOsSocketPairCreate(ncclSocketPairDescriptor pair[2]) {
  int fds[2];
  /*
  建管道，EINTR/EAGAIN 自动重试，硬失败 WARN+return ncclSystemError
  pipe() 接口详解：
    int pipe(int fds[2])，(POSIX,<unistd.h>)，创建一条匿名、单向(半双工)的字节流管道。
  出参约定 fds[0]=读端，fds[1]=写端。返回成功 0，失败 -1 并置 errno。
    单向：数据只能 写端→读端。在 Linux 上对 fds[1] 读、或对 fds[0] 写，会得到 EBADF(两端分别只开写/只读)，所以 header 注释强调"典型用法是单向"。
  ⚠️ 一个细节:header 注释说"Either endpoint can technically perform both operations"，那是按 socketpair(两端都全双工)的语义写的，
  但当前 Linux 用 pipe()，两端严格单向，必须 pair[0] 读、pair[1] 写。RAS 正是这么用的(rasLocalNotify 写 [1]、rasLocalHandle 读 [0])。
    字节流、无消息边界：像文件一样按字节流读，write 两次的数据会被 read 连成一片，没有"一条消息"的边界，要自己分帧(RAS 的通知是定长 struct，靠 read 循环读满一个结构来分帧)。
    内核缓冲区：管道有个内核 buffer(Linux 默认 64KB / 16 页，可用 fcntl(fd, F_SETPIPE_SZ, ...) 调)。buffer 满时 write 阻塞(非阻塞则 EAGAIN)，
  buffer 空时 read 阻塞(非阻塞则 EAGAIN)。
    匿名：没有文件系统名字，只能靠这两个 fd 共享，通常 fork 后父子进程共享，或像这里单进程内自用(self-pipe)。
    EOF 语义：当所有写端 fd 都被关闭后，读端 read 返回 0(EOF)。这对 self-pipe 的"对端关闭检测"很关键。
    原子写：不超过 PIPE_BUF(Linux 4096 字节)的 write 是原子的，不会和其它写者的数据交错，多线程/多信号处理函数并发写通知时能保证不撕裂。
  
  为什么用 pipe:self-pipe trick
    经典问题：一个线程阻塞在 poll()/select() 等一批 fd，如何被"另一个线程的事件"或"信号"安全唤醒?poll 只认 fd，不认你内存里的标志位。
    解法：建一条管道，把读端加进 poll 集合，别的线程(或信号处理函数)想唤醒 poller，就往写端写一个字节→读端变可读→poll 返回→poller 把管道读干、处理事件。
    NCCL RAS 里的落地：rasThreadMain 的 poll 集合含 rasNotificationPipe[0](读端)，普通 NCCL 线程调 rasLocalNotify(msg) 往 
  rasNotificationPipe[1](写端)写→唤醒 RAS 线程→rasLocalHandle 读出→分派(RAS_ADD_RANKS→rasLocalHandleAddRanks，RAS_TERMINATE→置 terminate)。
    价值：把"谁触发"(任意 NCCL 线程，同步写)与"谁执行"(RAS 线程，异步处理)解耦，让 RAS 线程的等待是事件驱动而非忙轮询，
  这正是前面聊 ncclRasCommInit 时说的 self-pipe。
  */
  SYSCHECK(pipe(fds), "pipe");
  pair[0] = fds[0];  // 读端
  pair[1] = fds[1];  // 写端
  return ncclSuccess;
}

ncclResult_t ncclOsSocketPairClose(ncclSocketPairDescriptor pair[2]) {
  ncclResult_t firstError = ncclSuccess;
  for (int i = 0; i < 2; i++) {
    if (pair[i] != NCCL_SOCKET_PAIR_INVALID) {
      if (close(pair[i]) == -1 && firstError == ncclSuccess) {
        WARN("close failed: %s", strerror(errno));
        firstError = ncclSystemError;
      }
      pair[i] = NCCL_SOCKET_PAIR_INVALID;
    }
  }
  return firstError;
}

ncclResult_t ncclOsSocketPairWrite(ncclSocketPairDescriptor descriptor, const void* buf, size_t len, size_t* written) {
  ssize_t n;
  // 可重入异步写入，buf 为本次写入的偏移地址，len 为剩余可写入数据长度，n 为本次实际写入字节数
  SYSCHECK(n = write(descriptor, buf, len), "write");
  *written = (size_t)n;
  return ncclSuccess;
}

ncclResult_t ncclOsSocketPairRead(ncclSocketPairDescriptor descriptor, void* buf, size_t len, size_t* nread) {
  ssize_t n;
  SYSCHECK(n = read(descriptor, buf, len), "read");
  *nread = (size_t)n;
  return ncclSuccess;
}
