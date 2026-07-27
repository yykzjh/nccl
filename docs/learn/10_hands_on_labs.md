# 10｜动手实验与源码阅读计划

源码学习最有效的循环不是“把文件从头看到尾”，而是：

```text
提出一个可验证的问题
  → 跑最小程序并保留日志
  → 从公共 API 向下追一条调用链
  → 改一个变量做对照
  → 用结果修正自己的理解
```

本章把前九章变成一组递进实验。基础实验只需单机多 GPU；多节点、NVLS、GIN 和 Device API 实验都标为选做，不具备硬件时照样可以完成主线。

## 1. 准备一份学习记录

每个实验保留五项信息：

| 项目 | 示例 |
| --- | --- |
| 问题 | “4 GPU AllReduce 为什么选 Ring + Simple？” |
| 环境 | NCCL commit、CUDA/driver、GPU/NIC、rank 数 |
| 命令 | 完整环境变量和启动命令 |
| 证据 | 关键日志、时间线、源码函数 |
| 结论 | 哪个假设成立，哪些仍未知 |

不要只记“成功/失败”。半年后真正有用的是当时的 commit、配置和证据链。

## 2. 构建当前源码

先确认版本和工作区：

```bash
git describe --tags --always
sed -n '1,40p' makefiles/version.mk
nvidia-smi
nvcc --version
```

建议使用独立 build 目录，并只编译本机架构：

```bash
make -j src.build \
  BUILDDIR="$PWD/build-learn" \
  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

将 `sm_90` 换成目标 GPU 对应架构。不确定时先使用项目默认值，正确构建比节省编译时间重要。

仓库构建说明在 [`README.md`](../../README.md)，调试构建见 [`docs/examples/DEBUG_GUIDE.md`](../examples/DEBUG_GUIDE.md)。

## 3. 实验 0：没有 GPU 也能做的“地图练习”

目标：建立源码导航习惯。

```bash
rg -n "ncclAllReduce" src/nccl.h.in src
rg -n "ncclCommInitRank" src/nccl.h.in src/init.cc
rg -n "ncclKernelMain|RunWorkBatch" src/device
rg -n "ncclProxyProgress|proxyProgress" src
```

产出一张四列小表：公共 API、host enqueue、device runner、transport/proxy。然后与 [`01_codebase_map.md`](01_codebase_map.md) 的模块图对照。

完成标准：不依赖 IDE 全局索引，也能在 2 分钟内定位一个公共 API 的实现入口。

## 4. 实验 1：一个进程管理多块 GPU

先运行最小 communicator 示例：

- 上游示例：[`docs/examples/01_communicators/01_multiple_devices_single_process/`](../examples/01_communicators/01_multiple_devices_single_process/)
- 本目录保留的 CMake 版本：[`docs/learn/01_communicators/01_multiple_devices_single_process/`](01_communicators/01_multiple_devices_single_process/)

观察三个问题：

1. `ncclCommInitAll` 为什么适合一个进程管理多 GPU？
2. 每个 communicator 的 rank、CUDA device 和 stream 如何对应？
3. 为什么多 device 操作要用 `ncclGroupStart/End`？

源码追踪：

```text
ncclCommInitAll
  → ncclCommInitRankDev
  → ncclCommInitRankFunc
  → initTransportsRank
```

对照 [`02_communicator_bootstrap.md`](02_communicator_bootstrap.md) 标出 `InitAll` 为每个 device 生成 UniqueId/comm 的方式，以及它与多进程 `InitRank` 的边界。

## 5. 实验 2：线程模型与进程模型

依次阅读或运行：

- [`02_one_device_per_pthread`](../examples/01_communicators/02_one_device_per_pthread/)
- [`03_one_device_per_process_mpi`](../examples/01_communicators/03_one_device_per_process_mpi/)

比较表：

| 问题 | 单进程多线程 | MPI 多进程 |
| --- | --- | --- |
| UniqueId 如何分发 | 进程内共享 | MPI broadcast / 外部 rendezvous |
| 每 rank 的地址空间 | 共享 | 独立 |
| CUDA context | 每线程切 device | 每进程通常固定 local GPU |
| bootstrap 是否仍需要 | 需要 | 需要 |
| 崩溃影响范围 | 整个进程 | 单个进程，可传播到 communicator |

完成标准：能解释 rank 不是进程号，local rank 也不是全局 rank。

## 6. 实验 3：用 Send/Recv 手写一圈 Ring

运行 [`docs/examples/02_point_to_point/01_ring_pattern/`](../examples/02_point_to_point/01_ring_pattern/)。给 rank `r` 填入值 `r`，让每个 rank 向 `(r+1)%N` 发送、从 `(r-1+N)%N` 接收。

重点验证：

- send 与 recv 为什么要放在一个 group 中，避免所有 rank 都先等待 send/recv 的环形依赖；
- P2P task 在 [`src/enqueue.cc`](../../src/enqueue.cc) 如何匹配 peer、生成 channel work；
- P2P Ring 与 AllReduce Ring 的相同点只是拓扑，后者还包含 chunk ownership 与 reduction。

扩展：把一个 rank 的 peer 写错，观察日志和挂住位置。只在可控测试环境中做，并给程序加超时。

## 7. 实验 4：先证明 AllReduce 语义

运行 [`docs/examples/03_collectives/01_allreduce/`](../examples/03_collectives/01_allreduce/)，先不要看性能。

令 rank `r` 的每个元素为 `r+1`，`N=4` 时 Sum AllReduce 的期望值是：

```text
1 + 2 + 3 + 4 = 10
```

依次改变：

1. out-of-place → in-place；
2. `float` → `int`；
3. `ncclSum` → `ncclMax`；
4. count 从 1、1 Ki、1 Mi 元素逐步增大。

每次先写出期望结果，再运行。这样能把“collective 语义错误”和“性能路径差异”分离。

## 8. 实验 5：手工模拟 4-rank Ring AllReduce

拿一张纸，把 16 个元素分成 4 个 chunk：`C0 C1 C2 C3`。按 [`05_collective_algorithms.md`](05_collective_algorithms.md) 的 Ring 图完成：

1. `N-1=3` 轮 ReduceScatter；
2. `N-1=3` 轮 AllGather；
3. 每轮写下每个 rank 的 send chunk、recv chunk 和 owner；
4. 验证每个 rank 最终都有全部 4 个 reduced chunks。

再打开 [`src/device/all_reduce.h`](../../src/device/all_reduce.h) 的 `runRing`，把纸上的六轮映射到：

```text
send
recvReduceSend × (N-2)
recvReduceCopySend
recvCopySend × (N-2)
recv
```

完成标准：不看文档也能解释为什么总共 `2(N-1)` 步，以及每 rank 算法通信量为何是 `2(N-1)M/N`。

## 9. 实验 6：观察算法、协议和 channel 的选择

对同一个 AllReduce 扫描消息大小：

```bash
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=TUNING,COLL,GRAPH,NET \
./allreduce
```

记录：message bytes、algorithm、protocol、channel/CTA、耗时。随后各做一次受控强制：

```bash
NCCL_ALGO=Ring NCCL_PROTO=Simple ./allreduce
NCCL_ALGO=Tree NCCL_PROTO=LL ./allreduce
```

不要把强制变量写入长期 shell profile。它们用于构造对照，不是普适调优结论。

源码追踪：

```text
ncclTaskAppend
  → ncclGetAlgoInfo / cost table
  → scheduleCollTasksToPlan
  → computeCollChunking
```

对照 [`04_host_execution_pipeline.md`](04_host_execution_pipeline.md) 和 [`05_collective_algorithms.md`](05_collective_algorithms.md)。

## 10. 实验 7：从硬件拓扑追到 transport

先记录系统视图：

```bash
nvidia-smi topo -m
```

再让 NCCL 导出 XML：

```bash
NCCL_TOPO_DUMP_FILE=/tmp/nccl-topo.xml \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=GRAPH,NET,INIT \
./allreduce
```

选择一对 GPU，回答：

1. 它们是 NVLink、同 PCIe switch、跨 CPU 还是跨主机？
2. `ncclTopoCheckP2p` 会允许 P2P 吗？
3. 最终 connector 是 P2P、SHM 还是 NET？
4. 若为 NET，GPU↔NIC 是否走 GDR？

对照 [`03_topology_transport_proxy.md`](03_topology_transport_proxy.md) 逐层画出 `hardware path → algorithm graph → transport connector`，不要把 XML 中的链路直接当成 Ring 顺序。

## 11. 实验 8：追一个 chunk 到 GPU 与 Proxy

选择 Ring + Simple 的一个 chunk，按下面顺序设断点或静态追踪：

1. [`src/enqueue.cc`](../../src/enqueue.cc)：plan/work upload。
2. [`src/device/common.cu`](../../src/device/common.cu)：`ncclKernelMain`。
3. [`src/device/all_reduce.h`](../../src/device/all_reduce.h)：Ring runner。
4. [`src/device/prims_simple.h`](../../src/device/prims_simple.h)：`genericOp`。
5. [`src/proxy.cc`](../../src/proxy.cc)：proxy progress。
6. [`src/transport/net.cc`](../../src/transport/net.cc)：network progress。

若只有单机 P2P，proxy 可能不是主要数据路径；可以先把静态调用链补全，再到多节点环境验证 NET。

建议输出一张时序表：

| 时刻 | GPU kernel | conn FIFO | CPU proxy | NIC |
| --- | --- | --- | --- | --- |
| t0 | 等待空 step | head/tail | idle | idle |
| t1 | 写 chunk | tail++ | 发现新 step | post send |
| t2 | 处理下一 slice | in flight | poll | transmit |
| t3 | 等待/继续 | head++ | publish completion | done |

完成标准：能指出 head/tail/step 中哪个变量表达 producer 进度，哪个表达 consumer 进度。

## 12. 实验 9：测量算法带宽与总线带宽

使用 NVIDIA [`nccl-tests`](https://github.com/NVIDIA/nccl-tests) 跑固定拓扑下的 AllReduce，记录 `algbw` 和 `busbw`。

对于 `N` 个 rank 的 Ring AllReduce：

```text
busbw = algbw × 2(N-1)/N
```

对于 AllGather、ReduceScatter、AllToAll：

```text
busbw = algbw × (N-1)/N
```

公式用于把 collective 的应用视角带宽换算为近似链路负载，详细定义见 [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)。不要拿不同 collective 的 raw algbw 直接比较链路效率。

## 13. 实验 10：Buffer Registration 与 CUDA Graph

先运行普通 registration 示例：

- [`docs/examples/04_user_buffer_registration/`](../examples/04_user_buffer_registration/)

做四组 A/B：未注册/已注册 × 普通 launch/Graph replay。每组至少区分：

- 首次 setup/capture 时间；
- warm-up 后单次 iteration；
- CPU 提交时间；
- GPU/NCCL 执行时间。

源码检查：

- [`src/register/register.cc`](../../src/register/register.cc)
- [`src/register/coll_reg.cc`](../../src/register/coll_reg.cc) 与 [`src/register/sendrecv_reg.cc`](../../src/register/sendrecv_reg.cc)
- [`src/enqueue.cc`](../../src/enqueue.cc) 中的 persistent plan。

扩展错误实验：在 graph executable 仍可能重放时提前释放 buffer，先在纸上说明为什么这是生命周期错误；不要在重要进程中实际制造 use-after-free。

## 14. 实验 11：对称内存与 Zero-CTA（选做）

硬件和 CUDA 版本满足条件时，运行：

- [`05_symmetric_memory/01_allreduce`](../examples/05_symmetric_memory/01_allreduce/)
- [`05_symmetric_memory/02_allgather`](../examples/05_symmetric_memory/02_allgather/)

比较普通 policy 与 `CTAPolicy=ZERO`：

1. 日志是否显示 symmetric/CE path 可用？
2. profiler 中通信是否占用 NCCL CTA？
3. SM 上同时运行的 compute kernel 是否更容易 overlap？
4. 端到端延迟是否也改善，还是只减少了 SM 占用？

对照 [`src/ce_coll.cc`](../../src/ce_coll.cc) 的 availability 条件。无法满足条件时，记录具体缺失项本身就是有效实验结果。

## 15. 实验 12：Device API 与融合（进阶选做）

按难度递增：

1. [`06_device_api/01_allreduce_lsa`](../examples/06_device_api/01_allreduce_lsa/)：只用 LSA 域内访问。
2. [`06_device_api/02_alltoall_gin`](../examples/06_device_api/02_alltoall_gin/)：纯 GIN AllToAll。
3. [`06_device_api/03_alltoall_hybrid`](../examples/06_device_api/03_alltoall_hybrid/)：LSA + GIN 分层。
4. [`07_kernel_fusion/01_rmsnorm_lsa`](../examples/07_kernel_fusion/01_rmsnorm_lsa/)：把计算与域内通信放进一个 kernel。
5. `multimem`、GIN、hybrid RMSNorm 示例：观察硬件能力如何改变同一算法的后端。

每个例子都只回答三个问题：host setup 了什么资源、device kernel 如何寻址和同步、硬件不支持时如何回退/报错。不要一开始钻进所有模板实现。

## 16. 四周学习安排

### 第 1 周：建立正确心智模型

- 第 1～2 天：读 `README`、00、01，完成实验 0。
- 第 3～4 天：读 02，完成实验 1、2。
- 第 5～7 天：读 collective 语义，完成实验 3、4。

阶段产出：能独立写一个 communicator 初始化、AllReduce、结果校验和清理程序。

### 第 2 周：算法、拓扑与调度

- 第 1～2 天：读 05，完成 4-rank Ring 手算。
- 第 3～4 天：读 03，完成拓扑/transport 实验。
- 第 5～7 天：读 04，完成算法与协议 A/B。

阶段产出：能从日志解释一次 collective 选择了什么以及为什么可能降级。

### 第 3 周：GPU 数据面与性能

- 第 1～3 天：读 06，追踪 Ring + Simple chunk。
- 第 4 天：对比 LL/LL128 数据布局。
- 第 5～7 天：跑 nccl-tests，理解 algbw/busbw，并用 profiler 找 GPU 空洞。

阶段产出：能描述 kernel、primitive、FIFO、proxy 和 NIC 的协作关系。

### 第 4 周：生命周期、工程与高级能力

- 第 1～2 天：读 07，完成 registration/Graph A/B。
- 第 3～4 天：读 09，整理自己的排障 SOP。
- 第 5～7 天：按硬件条件选读 08，完成一个 symmetric/Device API 示例。

阶段产出：能从一个性能或挂死现象提出分层假设，并用日志、源码和实验逐个排除。

## 17. 推荐的源码入口

如果今天只开始看一条链，选择 AllReduce：

```text
src/collectives.cc                 ncclAllReduce 公共实现
  → src/enqueue.cc                 task、cost、plan、launch
  → src/init.cc                    comm 中的 graph/channel/connector 来源
  → src/graph/search.cc            Ring/Tree 等图如何搜索
  → src/transport.cc               transport 如何选和连接
  → src/device/common.cu           kernel 主循环
  → src/device/all_reduce.h        算法 runner
  → src/device/prims_simple.h      一次真实 send/recv/reduce
  → src/proxy.cc + transport/net.cc 跨节点 progress
```

这条链横跨 NCCL 最核心的控制面和数据面。读通以后，再按兴趣横向展开：

- 初始化/容错：`src/init.cc`、`src/bootstrap.cc`、`src/ras/`；
- 网络：`src/transport/net.cc`、`src/transport/net_ib/`、`src/plugin/net.cc`；
- 算法：`src/device/{all_reduce,all_gather,reduce_scatter}.h`；
- 高级 device 通信：`src/dev_runtime.cc`、`src/include/nccl_device/`、`src/gin/`；
- 工程构建：`src/device/generate.py`、`src/device/Makefile`、`makefiles/`。

## 18. 最终自测题

1. `ncclCommInitAll` 与 `ncclCommInitRank` 的进程模型和 UniqueId 分发有何不同？
2. hardware topology、algorithm graph、transport connection 分别描述什么？
3. AllReduce 的 Ring 为什么是 ReduceScatter + AllGather？
4. Simple、LL、LL128 改变的是拓扑、算法还是连接协议？
5. channel、CTA、chunk、slice、step 分别处于哪个尺度？
6. 什么时候 GPU kernel 仍需要 CPU proxy？
7. buffer registration 与 window registration 的语义差异是什么？
8. CUDA Graph 复用了什么，为什么不能减少理论网络通信量？
9. LSA、multimem、GIN 各自解决“可达性”的哪一层？
10. 一次挂死应该如何判断是 collective 不匹配、proxy 不推进还是远端 rank 已退出？

如果能结合源码函数和一个实际日志回答这些问题，已经不再只是会调用 NCCL，而是具备继续研究新算法和真实性能问题的基础。
