# NCCL 测试程序编译、调试、执行指南

## 一、前置条件：编译 Debug 版 NCCL 库

编译带完整调试符号的 NCCL 库（仅编译当前 GPU 架构 sm_90，适用于 H20）：

```bash
cd /root/code/nccl
make clean
make -j$(nproc) DEBUG=1 NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

编译产物：
- 动态库：`/root/code/nccl/build/lib/libnccl.so`
- 静态库：`/root/code/nccl/build/lib/libnccl_static.a`
- 头文件：`/root/code/nccl/build/include/nccl.h`

验证调试符号是否存在：

```bash
file /root/code/nccl/build/lib/libnccl.so.2.*
# 应输出: with debug_info, not stripped

readelf -S /root/code/nccl/build/lib/libnccl.so.2.* | grep debug
# 应看到 .debug_info, .debug_line, .debug_str 等 section
```

---

## 二、测试程序编译

### 2.1 通信域初始化测试（test_init）

源码路径：`/root/code/nccl/docs/examples/01_communicators/01_multiple_devices_single_process/main.cc`

功能：检测所有 GPU，创建 NCCL 通信域，验证通信域属性，清理资源。

```bash
nvcc -g -O0 \
  -I/root/code/nccl/build/include \
  -L/root/code/nccl/build/lib -lnccl \
  -lcudart \
  -gencode=arch=compute_90,code=sm_90 \
  -o /root/code/nccl/build/bin/test_init \
  /root/code/nccl/docs/examples/01_communicators/01_multiple_devices_single_process/main.cc
```

### 2.2 AllReduce 集合通信测试（test_allreduce）

源码路径：`/root/code/nccl/docs/examples/03_collectives/01_allreduce/main.cc`

功能：在多 GPU 间执行 AllReduce Sum 操作，验证归约结果的正确性。

```bash
nvcc -g -O0 \
  -I/root/code/nccl/build/include \
  -L/root/code/nccl/build/lib -lnccl \
  -lcudart \
  -gencode=arch=compute_90,code=sm_90 \
  -o /root/code/nccl/build/bin/test_allreduce \
  /root/code/nccl/docs/examples/03_collectives/01_allreduce/main.cc
```

### 2.3 编译选项说明

| 选项 | 含义 |
|------|------|
| `-g` | 生成调试符号（host 端） |
| `-O0` | 不做优化，确保调试时变量不被优化掉 |
| `-I/root/code/nccl/build/include` | NCCL 头文件路径 |
| `-L/root/code/nccl/build/lib -lnccl` | 链接 debug 版 libnccl.so |
| `-gencode=arch=compute_90,code=sm_90` | 仅编译 H20 GPU 架构 |

---

## 三、测试程序执行

运行前需设置动态库搜索路径，确保加载的是 debug 版 NCCL：

```bash
export LD_LIBRARY_PATH=/root/code/nccl/build/lib:$LD_LIBRARY_PATH
```

### 3.1 直接运行

```bash
/root/code/nccl/build/bin/test_init
/root/code/nccl/build/bin/test_allreduce
```

### 3.2 带 NCCL 调试日志运行

```bash
# INFO 级别日志：显示初始化、拓扑发现等关键信息
NCCL_DEBUG=INFO /root/code/nccl/build/bin/test_init

# TRACE 级别日志：显示所有子系统的详细跟踪信息
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL /root/code/nccl/build/bin/test_init

# 仅跟踪集合通信子系统
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL /root/code/nccl/build/bin/test_allreduce
```

---

## 四、GDB 调试

### 4.1 启动 GDB

```bash
export LD_LIBRARY_PATH=/root/code/nccl/build/lib:$LD_LIBRARY_PATH
gdb /root/code/nccl/build/bin/test_init
```

### 4.2 GDB 常用命令速查

| 命令 | 缩写 | 作用 |
|------|------|------|
| `run` | `r` | 从头启动程序 |
| `start` | | 启动程序，自动停在 main() 第一行 |
| `break ncclCommInitAll` | `b ncclCommInitAll` | 在函数入口设断点 |
| `break init.cc:100` | `b init.cc:100` | 在文件某行设断点 |
| `next` | `n` | 执行下一行（不进入函数） |
| `step` | `s` | 执行下一行（进入函数） |
| `finish` | `fin` | 跑完当前函数，返回调用者 |
| `continue` | `c` | 继续运行到下一个断点 |
| `print var` | `p var` | 打印变量值 |
| `print *ptr` | `p *ptr` | 打印指针指向的内容 |
| `backtrace` | `bt` | 查看调用栈 |
| `list` | `l` | 显示当前位置附近的源码 |
| `info locals` | | 查看所有局部变量 |
| `info threads` | | 查看所有线程 |
| `thread apply all bt` | | 打印所有线程的调用栈 |
| `quit` | `q` | 退出 GDB |

### 4.3 调试 test_init 示例流程

```bash
(gdb) break ncclCommInitAll
(gdb) run
# 命中断点后
(gdb) step                     # 进入 ncclCommInitAll 内部
(gdb) list                     # 查看当前源码
(gdb) bt                       # 查看调用栈
(gdb) next                     # 逐行执行
(gdb) print num_gpus           # 查看变量
(gdb) finish                   # 跳出当前函数
(gdb) continue                 # 继续运行
```

### 4.4 调试 test_allreduce 示例流程

```bash
gdb /root/code/nccl/build/bin/test_allreduce

(gdb) break ncclAllReduce
(gdb) run
# 命中断点后
(gdb) bt                       # 查看调用栈
(gdb) step                     # 进入 ncclAllReduce 内部
(gdb) info locals              # 查看局部变量
(gdb) break enqueue.cc:1 if sendbuff != 0   # 条件断点
(gdb) continue
```

### 4.5 在 GDB 中启用 NCCL 日志

```bash
(gdb) set environment NCCL_DEBUG INFO
(gdb) set environment LD_LIBRARY_PATH /root/code/nccl/build/lib
(gdb) run
```

---

## 五、CUDA Kernel 调试（cuda-gdb）

如需调试 GPU device 端 kernel 代码，使用 cuda-gdb 替代 gdb：

```bash
export LD_LIBRARY_PATH=/root/code/nccl/build/lib:$LD_LIBRARY_PATH
cuda-gdb /root/code/nccl/build/bin/test_allreduce
```

cuda-gdb 额外命令：

| 命令 | 作用 |
|------|------|
| `info cuda kernels` | 查看所有活跃的 kernel |
| `info cuda threads` | 查看当前 kernel 的线程网格 |
| `cuda thread (1,0,0)` | 切换到指定 GPU 线程 |
| `cuda block (2,0,0)` | 切换到指定 block |
| `print threadIdx.x` | 打印 GPU 线程索引 |

---

## 六、推荐的 NCCL 代码调试学习路径

1. **初始化流程** — 用 `test_init` 在 `ncclCommInitAll` 设断点，跟进 `src/init.cc`
2. **拓扑发现** — 跟进 `src/graph/` 下的代码，理解 GPU 拓扑检测
3. **集合通信** — 用 `test_allreduce` 在 `ncclAllReduce` 设断点，跟进 `src/enqueue.cc` 和 `src/collectives.cc`
4. **传输层** — 跟进 `src/transport/` 下的代码，理解数据在 GPU 间的传输机制
