# 总进度

- [ ] 第 1 周：先把"怎么用 NCCL"学扎实
- [ ] 第 2 周：建立核心数据结构和初始化主线
- [ ] 第 3 周：攻克 graph、transport、proxy 和执行主线
- [ ] 第 4 周：高级特性、Device API、观测与调优

---

# 第 1 周：先把"怎么用 NCCL"学扎实

## Day 1 仓库总览与构建入口

- [ ] 阅读 `README.md`
- [ ] 阅读 `docs/examples/README.md`
- [ ] 阅读 `Makefile`
- [ ] 执行 `make -j src.build`
- [ ] `[条件允许]` 执行 `make -j examples`
- [ ] 回答：NCCL 解决的核心问题是什么
- [ ] 回答：仓库内 `docs/examples` 和仓库外 `nccl-tests` 的分工是什么
- [ ] 回答：构建产物大概会落到哪里
- [ ] 画出一张仓库地图

## Day 2 communicator 最小闭环：单进程多卡

- [ ] 阅读 `docs/examples/01_communicators/README.md`
- [ ] 阅读 `docs/examples/01_communicators/01_multiple_devices_single_process/README.md`
- [ ] 阅读 `docs/examples/01_communicators/01_multiple_devices_single_process/main.cc`
- [ ] 执行 `cd docs/examples/01_communicators/01_multiple_devices_single_process && make && ./multiple_devices_single_process`
- [ ] 回答：`ncclCommInitAll` 解决了什么问题
- [ ] 回答：为什么这个模式不需要外部协调
- [ ] 回答：单进程多卡模式的边界是什么
- [ ] 写出 communicator 生命周期笔记

## Day 3 communicator 进阶：一线程一卡

- [ ] 阅读 `docs/examples/01_communicators/02_one_device_per_pthread/README.md`
- [ ] 阅读 `docs/examples/01_communicators/02_one_device_per_pthread/main.cc`
- [ ] 阅读 `docs/examples/common/README.md`
- [ ] 执行 `cd docs/examples/01_communicators/02_one_device_per_pthread && make && NTHREADS=2 ./one_device_per_pthread`
- [ ] 回答：线程如何共享 unique ID
- [ ] 回答：线程和 rank 的关系是什么
- [ ] 回答：为什么 pthread 方案比 MPI 轻，但不能天然跨节点
- [ ] 做一张 `ncclCommInitAll` vs `ncclCommInitRank + pthreads` 对比表

## Day 4 communicator 进阶：MPI 多进程

- [ ] 阅读 `docs/examples/01_communicators/03_one_device_per_process_mpi/README.md`
- [ ] 阅读 `docs/examples/01_communicators/03_one_device_per_process_mpi/main.cc`
- [ ] 再次回看 `docs/examples/common/README.md` 的 MPI 部分
- [ ] `[条件允许]` 执行 `cd docs/examples/01_communicators/03_one_device_per_process_mpi && make MPI=1 && mpirun -np 2 ./one_device_per_process_mpi`
- [ ] 回答：MPI 在这个例子里到底帮了什么
- [ ] 回答：为什么 `ncclCommInitRank` 能跨进程
- [ ] 回答：local rank 和 global rank 分别用于什么
- [ ] 整理 3 种 communicator 初始化方式总结

## Day 5 collective 最小闭环：AllReduce

- [ ] 阅读 `docs/examples/03_collectives/README.md`
- [ ] 阅读 `docs/examples/03_collectives/01_allreduce/README.md`
- [ ] 阅读 `docs/examples/03_collectives/01_allreduce/main.cc`
- [ ] 执行 `cd docs/examples/03_collectives/01_allreduce && make && ./allreduce`
- [ ] 回答：为什么 expected sum 是 `0+1+...+(n-1)`
- [ ] 回答：`ncclAllReduce` 的输入和输出缓冲区分别扮演什么角色
- [ ] 回答：为什么示例里要配 `cudaStream_t`
- [ ] 画一张最小 AllReduce 时序图

## Day 6 P2P 最小闭环：Ring Pattern

- [ ] 阅读 `docs/examples/02_point_to_point/README.md`
- [ ] 阅读 `docs/examples/02_point_to_point/01_ring_pattern/README.md`
- [ ] 阅读 `docs/examples/02_point_to_point/01_ring_pattern/main.cc`
- [ ] 执行 `cd docs/examples/02_point_to_point/01_ring_pattern && make && ./ring_pattern`
- [ ] 回答：为什么这个例子特别强调 `ncclGroupStart()` / `ncclGroupEnd()`
- [ ] 回答：如果不 group，可能会发生什么
- [ ] 回答：P2P 和 collective 的关系是什么
- [ ] 画一张 ring 数据流草图

## Day 7 第一周整合与日志观察

- [ ] 阅读 `src/CMakeLists.txt`
- [ ] 回看本周所有 example 的 README 标题和运行命令
- [ ] 重跑 Day 2 的例子并加 `NCCL_DEBUG=INFO`
- [ ] 重跑 Day 5 的例子并加 `NCCL_DEBUG=INFO`
- [ ] 回答：日志里能看到哪些初始化 / 执行阶段
- [ ] 回答：example 和 `src/` 的哪些模块开始建立映射
- [ ] 回答：我现在能否脱离代码解释 communicator 与 allreduce 的最小路径
- [ ] 整理第 1 周 cheat sheet

---

# 第 2 周：建立核心数据结构和初始化主线

## Day 8 认识核心结构体

- [ ] 阅读 `src/CMakeLists.txt`
- [ ] 阅读 `src/include/comm.h`
- [ ] 重点追 `ncclComm`
- [ ] 重点追 `ncclChannel`
- [ ] 重点追 `ncclTaskColl`
- [ ] 重点追 `ncclKernelPlan`
- [ ] 回看 `docs/examples/03_collectives/01_allreduce/main.cc`，尝试映射这些结构
- [ ] 回答：`comm`、`channel`、`task`、`plan` 分别站在什么抽象层
- [ ] 回答：为什么内部会有 task 和 plan 两层
- [ ] 回答：哪些结构是 host 侧的中心
- [ ] 做 4 张结构体卡片

## Day 9 从 API 调用到内部描述对象

- [ ] 阅读 `src/include/info.h`
- [ ] 阅读 `src/include/collectives.h`
- [ ] 阅读 `src/include/channel.h`
- [ ] 做静态追踪，不要求跑程序
- [ ] 回答：一次 collective 调用在内部最小会被描述成哪些字段
- [ ] 回答：`info` 和 `task` 的关系是什么
- [ ] 回答：channel 为什么会成为 graph、transport、kernel 的交汇点
- [ ] 画一张 API 调用对象模型图

## Day 10 初始化主线前半段

- [ ] 阅读 `src/init.cc`
- [ ] 只追 `commAlloc`
- [ ] 只追 `ncclCommInitRankFunc`
- [ ] 配套回看 `src/include/comm.h`
- [ ] 重跑 Day 3 或 Day 4 的 communicator 例子，并加 `NCCL_DEBUG=INFO`
- [ ] 回答：初始化的第一批资源是什么
- [ ] 回答：为什么 communicator 初始化不是一次简单分配
- [ ] 回答：在 topo / transport 之前已经准备了什么
- [ ] 画一张 `ncclCommInitRank` 前半段顺序图

## Day 11 bootstrap：先认识彼此

- [ ] 阅读 `src/bootstrap.cc`
- [ ] 阅读 `src/include/bootstrap.h`
- [ ] 重点追 `bootstrapInit`
- [ ] 重点追 `bootstrapSplit`
- [ ] 重点追 `bootstrapAllGather`
- [ ] 重跑前一天同一个 communicator 例子
- [ ] 回答：bootstrap 交换了哪些控制面数据
- [ ] 回答：为什么需要 OOB ring
- [ ] 回答：bootstrap 和真正的数据 transport 为什么要分开
- [ ] 写一页"先认识彼此，再正式通信"的解释

## Day 12 channel：把抽象落到通道

- [ ] 阅读 `src/channel.cc`
- [ ] 阅读 `src/init.cc` 中的 `setupChannel`
- [ ] 阅读 `src/init.cc` 中的 `devCommSetup`
- [ ] 重跑 `docs/examples/03_collectives/01_allreduce/allreduce`
- [ ] 回答：channel 到底更像"逻辑通道"还是"物理链路"
- [ ] 回答：channel peer 在什么时候被配齐
- [ ] 回答：哪些 host 侧状态会复制到 device 侧
- [ ] 画一张 channel 结构示意图

## Day 13 graph / transport / proxy 预热

- [ ] 阅读 `src/include/graph.h`
- [ ] 阅读 `src/include/transport.h`
- [ ] 阅读 `src/include/proxy.h`
- [ ] 只做静态梳理，不要求跑程序
- [ ] 回答：graph、transport、proxy 三个子系统分别回答什么问题
- [ ] 回答：它们之间的边界是什么
- [ ] 回答：为什么这 3 个模块是理解 NCCL 的关键转折点
- [ ] 画一张模块职责边界图

## Day 14 第二周整合：初始化主线

- [ ] 回看本周全部笔记
- [ ] 把 `src/init.cc`、`src/bootstrap.cc`、`src/channel.cc` 的关键函数名整理到同一页
- [ ] 跑任一 communicator 例子
- [ ] 口头复述一次 communicator 从创建到 ready 的路径
- [ ] 回答：我能不能完整讲完 communicator 的初始化主线
- [ ] 回答：我能不能说出 5 个关键结构体和 5 个关键文件
- [ ] 回答：哪些地方我还只是"知道名字"，没有真正理解
- [ ] 写第 2 周总结稿

---

# 第 3 周：攻克 graph、transport、proxy 和执行主线

## Day 15 topology 与 path

- [ ] 阅读 `src/graph/topo.cc`
- [ ] 阅读 `src/graph/paths.cc`
- [ ] `[条件允许]` 查看 `nvidia-smi topo -m`
- [ ] 回答：NCCL 会采集哪些拓扑信息
- [ ] 回答：path 在这里代表什么
- [ ] 回答：为什么 trim 之后还要重新计算 path
- [ ] 写一页 topology 词汇表

## Day 16 graph 搜索与 channel 落地

- [ ] 阅读 `src/graph/search.cc`
- [ ] 阅读 `src/graph/connect.cc`
- [ ] 快速浏览 `src/graph/rings.cc`
- [ ] 快速浏览 `src/graph/trees.cc`
- [ ] 回答：ring / tree / NVLS / CollNet 图是怎么被搜出来的
- [ ] 回答：graph 搜索的输出是什么
- [ ] 回答：它是如何落到具体 channel 上的
- [ ] 画一张 `graph -> channel` 映射图

## Day 17 tuning：算法与协议选择

- [ ] 阅读 `src/graph/tuning.cc`
- [ ] 在 `src/enqueue.cc` 中定位 `ncclGetAlgoInfo`
- [ ] 在 `src/enqueue.cc` 中定位 `topoGetAlgoInfo`
- [ ] 重跑 `docs/examples/03_collectives/01_allreduce/allreduce`
- [ ] 尝试加 `NCCL_DEBUG=INFO`
- [ ] 回答：algorithm 和 protocol 的区别是什么
- [ ] 回答：谁负责做选择
- [ ] 回答：选择时会考虑哪些输入
- [ ] 整理一张算法 / 协议选择表

## Day 18 transport：p2p 与 shm

- [ ] 阅读 `src/transport.cc`
- [ ] 阅读 `src/transport/generic.cc`
- [ ] 阅读 `src/transport/p2p.cc`
- [ ] 阅读 `src/transport/shm.cc`
- [ ] 可选重跑 `docs/examples/02_point_to_point/01_ring_pattern/ring_pattern`
- [ ] 回答：transport 抽象层想屏蔽什么差异
- [ ] 回答：`p2p` 和 `shm` 各适合什么场景
- [ ] 回答：transport 在什么时候从 graph 手里接过接力棒
- [ ] 画一张 transport 对照表

## Day 19 transport：网络路径

- [ ] 阅读 `src/transport/net.cc`
- [ ] 阅读 `src/transport/net_socket.cc`
- [ ] 快速浏览 `src/transport/net_ib/init.cc`
- [ ] 快速浏览 `src/transport/net_ib/connect.cc`
- [ ] `[条件允许]` 跑 MPI 例子
- [ ] 回答：哪些代码是通用网络 transport 壳层
- [ ] 回答：哪些是 socket / IB 专属实现
- [ ] 回答：为什么多机路径更依赖 host 参与
- [ ] 写一页 `intra-node` vs `inter-node` 比较

## Day 20 proxy：host 侧推进器

- [ ] 阅读 `src/proxy.cc`
- [ ] 阅读 `src/include/proxy.h`
- [ ] 重点追 `ncclProxyCreate`
- [ ] 重跑 `docs/examples/03_collectives/01_allreduce/allreduce`
- [ ] 回答：proxy 为什么存在
- [ ] 回答：schedule 和 progress 分别做什么
- [ ] 回答：哪些 transport 更依赖 proxy
- [ ] 画一张 proxy 生命周期图

## Day 21 从 API 到 launch 的完整主线

- [ ] 阅读 `src/collectives.cc`
- [ ] 阅读 `src/enqueue.cc`
- [ ] 阅读 `src/group.cc`
- [ ] 重点追 `ncclEnqueueCheck`
- [ ] 重点追 `taskAppend`
- [ ] 重点追 `ncclTasksRegAndEnqueue`
- [ ] 重点追 `finishPlan`
- [ ] 重点追 `ncclGroupEndInternal`
- [ ] 重点追 `groupLaunch`
- [ ] 运行 `docs/examples/03_collectives/01_allreduce/allreduce`
- [ ] 运行 `docs/examples/02_point_to_point/01_ring_pattern/ring_pattern`
- [ ] 回答：API 调用是怎么变成 task 的
- [ ] 回答：group 边界到底影响了什么
- [ ] 回答：plan 什么时候形成，kernel launch 什么时候发生
- [ ] 写一页从 `ncclAllReduce` 到 launch 的完整主线

---

# 第 4 周：高级特性、Device API、观测与调优

## Day 22 user buffer registration

- [ ] 阅读 `docs/examples/04_user_buffer_registration/README.md`
- [ ] 阅读 `docs/examples/04_user_buffer_registration/01_allreduce/README.md`
- [ ] 阅读 `docs/examples/04_user_buffer_registration/01_allreduce/main.cc`
- [ ] 阅读 `src/register/register.cc`
- [ ] 阅读 `src/register/coll_reg.cc`
- [ ] 执行 `cd docs/examples/04_user_buffer_registration/01_allreduce && make && ./allreduce_ub`
- [ ] 回答：为什么这里强调 `ncclMemAlloc`，而不是 `cudaMalloc`
- [ ] 回答：为什么 registered buffer 不能混用
- [ ] 回答：为什么注册 / 反注册顺序重要
- [ ] 写一张 buffer registration 检查清单

## Day 23 symmetric memory

- [ ] 阅读 `docs/examples/05_symmetric_memory/README.md`
- [ ] 阅读 `docs/examples/05_symmetric_memory/01_allreduce/README.md`
- [ ] 阅读 `docs/examples/05_symmetric_memory/01_allreduce/main.cc`
- [ ] 阅读 `src/scheduler/symmetric_sched.cc`
- [ ] 阅读 `src/sym_kernels.cc`
- [ ] 执行 `cd docs/examples/05_symmetric_memory/01_allreduce && make && ./allreduce_sm`
- [ ] 回答：symmetric window 比普通注册多了什么约束
- [ ] 回答：这些约束换来了什么优化空间
- [ ] 回答：为什么它常被看成 Device API 的前置知识
- [ ] 做一张"普通注册 vs 对称窗口"比较表

## Day 24 Device API 总览

- [ ] 阅读 `docs/examples/06_device_api/README.md`
- [ ] 阅读 `src/include/nccl_device/README.md`
- [ ] 阅读 `src/include/nccl_device.h`
- [ ] 阅读 `src/include/nccl_device/comm.h`
- [ ] 阅读 `src/include/nccl_device/barrier.h`
- [ ] 阅读 `src/include/nccl_device/gin.h`
- [ ] 阅读 `src/include/nccl_device/core.h`
- [ ] 回答：Device API 把哪些能力下放到了 kernel 里
- [ ] 回答：公有头文件和 `impl/` 分层的目的是什么
- [ ] 回答：host API 和 device API 的边界在哪里
- [ ] 写一页 device API 术语表

## Day 25 Device API：LSA AllReduce

- [ ] 阅读 `docs/examples/06_device_api/01_allreduce_lsa/README.md`
- [ ] 阅读 `docs/examples/06_device_api/01_allreduce_lsa/main.cu`
- [ ] 阅读 `src/device/common.cu`
- [ ] 阅读 `src/device/common.h`
- [ ] 阅读 `src/device/primitives.h`
- [ ] 阅读 `src/device/all_reduce.h`
- [ ] 可选阅读 `src/device/prims_simple.h`
- [ ] 可选阅读 `src/device/prims_ll.h`
- [ ] 可选阅读 `src/device/prims_ll128.h`
- [ ] `[条件允许]` 执行 `cd docs/examples/06_device_api/01_allreduce_lsa && make && ./allreduce_lsa`
- [ ] 回答：哪些原本由 host 组织的工作被移到了 kernel
- [ ] 回答：LSA barrier 保证了什么
- [ ] 回答：device primitives 在隐藏什么复杂度
- [ ] 画一张 kernel 内 AllReduce 路径图

## Day 26 Device API：GIN 与 Hybrid

- [ ] 阅读 `docs/examples/06_device_api/02_alltoall_gin/README.md`
- [ ] 阅读 `docs/examples/06_device_api/02_alltoall_gin/main.cu`
- [ ] 阅读 `docs/examples/06_device_api/03_alltoall_hybrid/README.md`
- [ ] 阅读 `docs/examples/06_device_api/03_alltoall_hybrid/main.cu`
- [ ] 阅读 `src/gin/gin_host.cc`
- [ ] 阅读 `src/transport/net_ib/gin.cc`
- [ ] 阅读 `src/include/plugin/nccl_gin.h`
- [ ] `[条件允许]` 运行 `./alltoall_gin`
- [ ] `[条件允许]` 运行 `./alltoall_hybrid`
- [ ] 回答：pure GIN 和 hybrid 的本质差异是什么
- [ ] 回答：local peer 和 remote peer 是怎么区分的
- [ ] 回答：为什么这些例子要先查 communicator properties
- [ ] 画一张 LSA 本地路径 / GIN 远程路径决策树

## Day 27 profiler 与 inspector

- [ ] 阅读 `plugins/profiler/README.md`
- [ ] 阅读 `plugins/profiler/example/README.md`
- [ ] 阅读 `plugins/profiler/inspector/README.md`
- [ ] 阅读 `plugins/profiler/inspector/exporter/example/README.md`
- [ ] 阅读 `src/include/plugin/nccl_profiler.h`
- [ ] `[条件允许]` 在 `plugins/profiler/example` 执行 `make`
- [ ] `[条件允许]` 在 `plugins/profiler/inspector` 执行 `make`
- [ ] `[条件允许]` 选择简单 workload（如 `allreduce`），按 README 配置环境变量并运行
- [ ] `[条件允许]` 运行 `python perf_summary_exporter.py --input_dir <日志目录>`
- [ ] 回答：debug log、profiler trace、inspector JSON 分别解决什么问题
- [ ] 回答：trace 里的 group / coll / kernel / proxy 分别对应主线的哪一段
- [ ] 回答：哪些指标最适合判断算法 / 协议选择是不是合理
- [ ] 画一张观测栈地图

## Day 28 tuner 与最终整合

- [ ] 阅读 `plugins/tuner/README.md`
- [ ] 阅读 `plugins/tuner/example/README.md`
- [ ] 阅读 `plugins/tuner/example/test/README.md`
- [ ] 阅读 `plugins/tuner/example/scripts/README.md`
- [ ] 阅读 `src/include/plugin/nccl_tuner.h`
- [ ] 可选预读 `contrib/nccl_ep/README.md`
- [ ] 执行 `cd plugins/tuner/example && make`
- [ ] 执行 `make test`
- [ ] 执行 `python scripts/optimize_config.py --dry-run scripts/sample_performance_data.csv`
- [ ] `[条件允许]` 跑外部 `nccl-tests` 的 `all_reduce_perf`
- [ ] 回答：tuner 在哪个层级影响 NCCL 决策
- [ ] 回答：cost table 能改什么，不能改什么
- [ ] 回答：如果性能异常，我会先查哪一层
- [ ] 写最终 2 页总结："我现在如何解释一次 NCCL AllReduce"

---

# 最终验收清单

- [ ] 我能不用看代码，讲清 `ncclAllReduce` 从 API 到 kernel 的主线
- [ ] 我能解释 `comm`、`channel`、`graph`、`transport`、`proxy`、`task`、`plan`、`device kernel` 的职责
- [ ] 我能说清 `docs/examples` 中 communicator、collective、P2P、registration、symmetric memory、Device API 各自教什么
- [ ] 我能用 `NCCL_DEBUG`、profiler、inspector、tuner 这几条线观察和解释行为
- [ ] 我知道 `contrib/nccl_ep`、GIN、hybrid device communication 为什么属于高级主题

# 学完后的加练题

- [ ] 如果只给你一个 `allreduce` 例子，你能顺着源码追到哪些关键函数
- [ ] 如果日志显示选了某个 algorithm / protocol，你会去哪个文件验证
- [ ] 如果多机性能差，你会优先怀疑 topology、transport、proxy 还是 tuner
- [ ] 如果你要给别人讲 NCCL，你会用哪 5 个文件做最小教学路径
