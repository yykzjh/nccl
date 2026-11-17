#ifndef NCCL_OS_LINUX_H_
#define NCCL_OS_LINUX_H_

#include <sys/syscall.h>
#include <sys/types.h>
#include <strings.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include <getopt.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <linux/types.h>
#include <endian.h>
#include <sys/resource.h>
#include <link.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <dirent.h>
#include <sched.h>

#define NCCL_INVALID_SOCKET -1
typedef int ncclSocketDescriptor;

/*
不管哪个平台，ncclAffinity 都是一个位图，第 i 位 = 1 表示 CPU i 属于这个集合。
cpu_set_t来自 <sched.h>，glibc 里它是个定长位集结构，本质等价于：
    #define __CPU_SETSIZE 1024                       // 最多表示 CPU 0..1023
    typedef unsigned long int __cpu_mask;            // 64 位平台 = 8 字节
    typedef struct {
        __cpu_mask __bits[__CPU_SETSIZE / (8 * sizeof(__cpu_mask))];  // = __bits[16]
    } cpu_set_t;

大小：16 × 8 = 128 字节，共 1024 位；
CPU → 位的映射：CPU n 落在 __bits[n / 64] 的第 n % 64 位。即 __bits[0] 管 CPU 0–63、__bits[1] 管 CPU 64–127，以此类推；
操作全靠 glibc 宏：CPU_ZERO（清空）、CPU_SET(cpu,s)（加入）、CPU_CLR(cpu,s)（移除）、CPU_ISSET(cpu,s)（测试）、CPU_COUNT(s)（数有几个）、CPU_AND/OR/XOR（集合运算）。
*/
typedef cpu_set_t ncclAffinity;

typedef pid_t ncclPid_t;

#define NCCL_POLLIN POLLIN
#define NCCL_POLLERR POLLHUP

#endif
