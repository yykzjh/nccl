# 06. 插件、参数、调试、构建与工程支撑

这一篇收拢不属于单次 collective 主链路、但阅读源码和排障时一定会遇到的支撑模块：插件 ABI、环境参数、日志/RAS、构建生成、OS wrapper、bindings、examples、tests 和 benchmark。

## 插件系统

插件加载的公共入口在 [src/plugin/plugin_open.cc](../../src/plugin/plugin_open.cc#L17)。这里维护 `NET`、`GIN`、`RMA`、`TUNER`、`PROFILER`、`ENV` 六类插件，按 `libnccl-net.so`、`libnccl-net-xxx.so` 这类命名规则 `dlopen`，并记录实际加载路径。

| 插件 | 入口 | 作用 |
| --- | --- | --- |
| NET / CollNet | [src/plugin/net.cc](../../src/plugin/net.cc#L97)、[plugins/net/README.md](../../plugins/net/README.md) | 提供跨节点网络、GDR、fabricId、CollNet 等能力。 |
| Tuner | [src/plugin/tuner.cc](../../src/plugin/tuner.cc#L39)、[plugins/tuner/README.md](../../plugins/tuner/README.md) | 允许外部覆盖或修正算法/协议选择。 |
| Profiler | [src/plugin/profiler.cc](../../src/plugin/profiler.cc#L46)、[plugins/profiler/README.md](../../plugins/profiler/README.md) | profiler 订阅 NCCL 事件，适合把 kernel、proxy、network 时间线接到外部工具。 |
| GIN / RMA | [src/plugin/gin.cc](../../src/plugin/gin.cc#L70)、[src/plugin/rma.cc](../../src/plugin/rma.cc#L70)、[plugins/gin/example/plugin.c](../../plugins/gin/example/plugin.c#L332) | 面向 GPU initiated networking 和 one-sided RMA 的扩展 ABI。 |
| ENV | [src/plugin/env.cc](../../src/plugin/env.cc) | 外部环境变量或配置来源适配。 |

`src/plugin/` 是 NCCL core 侧的 ABI 适配层，`plugins/` 是示例或说明。真正运行时，NCCL 会在初始化 communicator 或第一次需要该能力时把外部函数表接进来；后续 transport、scheduler、profiler 才通过这些函数表工作。

## 参数系统

NCCL 参数不是散落的 `getenv`。核心宏在 [src/include/param.h#L21](../../src/include/param.h#L21)，很多 `NCCL_PARAM` 会生成懒加载的查询函数。参数注册、枚举和文档化相关代码在 [src/param](../../src/param/)。

| 想看什么 | 入口 |
| --- | --- |
| 算法/协议覆盖 | [src/graph/tuning.cc#L453](../../src/graph/tuning.cc#L453)、[src/graph/tuning.cc#L458](../../src/graph/tuning.cc#L458) |
| 拓扑/连接相关参数 | [src/graph](../../src/graph/)、[src/transport.cc](../../src/transport.cc#L75)、[src/transport/p2p.cc](../../src/transport/p2p.cc#L104) |
| device runtime / RMA / GIN 参数 | [src/dev_runtime.cc](../../src/dev_runtime.cc#L28)、[src/rma/rma_proxy.cc](../../src/rma/rma_proxy.cc#L26)、[src/gin/gin_host.cc](../../src/gin/gin_host.cc#L19) |
| debug 参数 | [src/debug.cc](../../src/debug.cc#L45)、[src/include/debug.h](../../src/include/debug.h#L50) |

## 日志、调试与 RAS

日志入口集中在 [src/debug.cc](../../src/debug.cc#L45) 和 [src/include/debug.h](../../src/include/debug.h#L50)。`WARN`、`INFO`、`TRACE` 宏最终会走 `ncclDebugLogInternal`。常用调试组合：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET ./your_program
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL ./your_program
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=PROXY,TUNING ./your_program
```

| 场景 | 建议入口 |
| --- | --- |
| 初始化卡住 | `INIT,BOOTSTRAP,GRAPH,NET`，对照 [bootstrap](../../src/bootstrap.cc#L672)、[topology](../../src/graph/topo.cc#L1765)、[transport connect](../../src/transport.cc#L21)。 |
| 性能不稳定 | `TUNING,COLL,PROXY,NET`，对照 [tuning model](../../src/graph/tuning.cc#L351)、[enqueue](../../src/enqueue.cc#L3124)、[proxy progress](../../src/proxy.cc#L1164)。 |
| device kernel 选择不符合预期 | `COLL,TUNING`，再查 [03_collective_algorithms_tuning.md](03_collective_algorithms_tuning.md) 和 [04_device_data_plane.md](04_device_data_plane.md)。 |
| 拓扑路径不符合预期 | `GRAPH,INIT,NET`，再查 [02_topology_transport_proxy.md](02_topology_transport_proxy.md)。 |

RAS 是 reliability、availability、serviceability 的缩写，核心线程在 [src/ras/ras.cc](../../src/ras/ras.cc#L90)。初始化时 [bootstrap.cc](../../src/bootstrap.cc#L672) 通过 `NCCL_RAS_ENABLE` 控制是否启用；启用后 communicator 初始化会调用 [ncclRasCommInit](../../src/ras/ras.cc#L90) 和 [ncclRasAddRanks](../../src/ras/ras.cc#L174)。

## 构建与 device 代码生成

构建入口包括根目录 [Makefile](../../Makefile)、[CMakeLists.txt](../../CMakeLists.txt)、[makefiles/common.mk](../../makefiles/common.mk) 和 [pkg/Makefile](../../pkg/Makefile)。Device 端代码比较特殊：[src/device/Makefile](../../src/device/Makefile#L80) 会调用 [src/device/generate.py](../../src/device/generate.py) 生成不同 collective、datatype、reduction op、algorithm、protocol 组合需要的源文件；symmetric kernel 还有 [src/device/symmetric/generate.py](../../src/device/symmetric/generate.py)。

这对应 device kernel 的“模板 + 代码生成 + 函数表”模式。阅读 [04_device_data_plane.md](04_device_data_plane.md) 时，如果找不到某个 concrete kernel 名称，通常要回到生成脚本和 Makefile。

## OS wrapper 与外部库封装

`src/os/` 和 `src/include/os/` 封装 pthread、socket、shared memory、dlopen、CUDA driver 加载、NUMA、文件系统等平台能力。调试动态库加载、共享内存创建失败、socket 连接失败时，这层经常是入口之一。

| 模块 | 作用 |
| --- | --- |
| [src/misc](../../src/misc/) | 通用工具、alloc、timer、checks、profiler helper。 |
| [src/os](../../src/os/) | OS 能力封装。 |
| [src/transport/net_ib](../../src/transport/net_ib/) | IB/RDMA 相关 transport 支撑。 |
| [src/include/mlx5](../../src/include/mlx5/) | mlx5 相关头文件和能力适配。 |

## Examples、tests、benchmarks、bindings

| 目录 | 用法 |
| --- | --- |
| [docs/examples](../examples/README.md) | 从 communicator、P2P、collectives 到 registration、symmetric memory、Device API、kernel fusion 的示例。 |
| [contrib/nccl_ep/ep_test.cu](../../contrib/nccl_ep/ep_test.cu) | Expert Parallel 示例测试入口。 |
| [contrib/nccl_ep/ep_bench.cu](../../contrib/nccl_ep/ep_bench.cu) | Expert Parallel benchmark 入口。 |
| [bindings/nccl4py](../../bindings/nccl4py/README.md) | Python binding。 |
| [bindings/ir](../../bindings/ir/) | device wrapper / IR 相关接口。 |
| [contrib](../../contrib/README.md) | 实验扩展和场景化封装，细节见 [05_memory_device_api_extensions.md](05_memory_device_api_extensions.md)。 |

## 推荐排障顺序

1. 初始化失败：先看 [01_initialization_control_plane.md](01_initialization_control_plane.md)，同时开 `INIT,BOOTSTRAP,GRAPH,NET`。
2. 没走预期链路：看 [02_topology_transport_proxy.md](02_topology_transport_proxy.md)，确认 topology、graph search、transport `canConnect`。
3. 算法或协议不符合预期：看 [03_collective_algorithms_tuning.md](03_collective_algorithms_tuning.md)，再开 `TUNING,COLL`。
4. kernel 行为不符合预期：看 [04_device_data_plane.md](04_device_data_plane.md)，对照生成脚本和 primitives。
5. 特殊路径失败：看 [05_memory_device_api_extensions.md](05_memory_device_api_extensions.md)，重点查 registration、symmetric window、GIN/RMA plugin、proxy。
