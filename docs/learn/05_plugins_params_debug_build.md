# 05. Plugins、参数、调试、构建与扩展模块

插件、参数、日志、RAS、构建、bindings 和 contrib 位于主数据通路之外，但常用于排障、调优和扩展。它们提供运行时扩展点、观测点和构建发布入口。

## 插件系统：把网络、调优和观测做成 ABI

插件加载的公共入口在 [src/plugin/plugin_open.cc](../../src/plugin/plugin_open.cc#L17)。这里维护了 `NET`、`GIN`、`RMA`、`TUNER`、`PROFILER`、`ENV` 六类插件，按 `libnccl-net.so`、`libnccl-net-xxx.so` 这类命名规则 `dlopen`，并记录实际加载路径。

| 插件 | 核心文件 | 阅读入口 |
| --- | --- | --- |
| Net / CollNet | [src/plugin/net.cc](../../src/plugin/net.cc#L97)、[plugins/net/README.md](../../plugins/net/README.md) | `ncclNetPluginLoad` 如何按 v12 到 v6 找函数表，`plugins/net/example/plugin.c` 需要实现哪些回调。 |
| Tuner | [src/plugin/tuner.cc](../../src/plugin/tuner.cc#L39)、[plugins/tuner/README.md](../../plugins/tuner/README.md) | tuner 不搬数据，它影响 algorithm/protocol/channel 等选择。 |
| Profiler | [src/plugin/profiler.cc](../../src/plugin/profiler.cc#L46)、[plugins/profiler/README.md](../../plugins/profiler/README.md) | profiler 订阅 NCCL 事件，适合把 kernel、proxy、network 的时间线接到外部工具。 |
| Env | [src/plugin/env.cc](../../src/plugin/env.cc#L93)、[plugins/env/README.md](../../plugins/env/README.md) | env plugin 可以替换/扩展环境变量读取来源，参数系统会通过它取值。 |
| GIN / RMA | [src/plugin/gin.cc](../../src/plugin/gin.cc#L70)、[src/plugin/rma.cc](../../src/plugin/rma.cc#L70)、[plugins/gin/example/plugin.c](../../plugins/gin/example/plugin.c#L332) | 面向 GPU initiated networking 和 one-sided RMA 的扩展 ABI。 |

理解插件时要分清两层：`src/plugin/` 是 NCCL core 侧的 ABI 适配层，`plugins/` 是示例或说明。真正运行时，NCCL 会在初始化 communicator 或第一次需要该能力时把外部函数表接进来；后续 transport、scheduler、profiler 才通过这些函数表工作。

## 参数系统：从环境变量到可查询 registry

老式参数宏在 [src/include/param.h](../../src/include/param.h#L21)，例如 [src/enqueue.cc](../../src/enqueue.cc#L29) 里的 `NCCL_PARAM`。它的特点是简单：第一次读取 `NCCL_XXX` 时解析并缓存。

较新的类型化参数系统在 [src/include/param/param.h](../../src/include/param/param.h#L28)，通过 `DEFINE_NCCL_PARAM` 注册类型、默认值、合法取值和描述。所有参数会进入 [ncclParamRegistry](../../src/include/param/param_registry.h#L25)，外部可以通过 [ncclParamBind](../../src/param/c_api.cc#L55)、[ncclParamGetAllParameterKeys](../../src/param/c_api.cc#L124)、[ncclParamGetParameter](../../src/param/c_api.cc#L156) 查询。命令行工具 [ncclparam](../../src/param/ncclparam.cc#L14) 也是围绕这些 API 做的。

| 目标 | 入口 |
| --- | --- |
| 找某个环境变量对应的源码 | `rg "NCCL_PARAM\\(|DEFINE_NCCL_PARAM" src`。 |
| 看日志相关参数 | [src/debug.cc](../../src/debug.cc#L45)。 |
| 看拓扑/连接相关参数 | [src/graph/](../../src/graph)、[src/transport.cc](../../src/transport.cc#L75)、[src/transport/p2p.cc](../../src/transport/p2p.cc#L104)。 |
| 看 device runtime / RMA / GIN 参数 | [src/dev_runtime.cc](../../src/dev_runtime.cc#L28)、[src/rma/rma_proxy.cc](../../src/rma/rma_proxy.cc#L26)、[src/gin/gin_host.cc](../../src/gin/gin_host.cc#L19)。 |

常见注意点：参数名在源码里常写成 `"FOO"`，实际环境变量通常是 `NCCL_FOO`。例如 `NCCL_PARAM(GraphRegister, "GRAPH_REGISTER", 1)` 对应 `NCCL_GRAPH_REGISTER`。

## 日志与调试：从 `NCCL_DEBUG` 追到 subsystem

日志入口集中在 [src/debug.cc](../../src/debug.cc#L45) 和 [src/include/debug.h](../../src/include/debug.h#L50)。`WARN`、`INFO`、`TRACE` 宏最终会走 `ncclDebugLogInternal`。调试时最常用的是：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,NET,PROXY ./your_program
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL ./your_program
NCCL_DEBUG=INFO NCCL_TOPO_DUMP_FILE=/tmp/nccl-topo.xml ./your_program
```

[docs/examples/DEBUG_GUIDE.md](../examples/DEBUG_GUIDE.md) 已经给了本仓库示例程序的编译、运行、gdb/lldb 调试方式。配合源码阅读时，可按以下顺序缩小问题：

1. 初始化卡住：开 `INIT,BOOTSTRAP,GRAPH,NET`，看 [bootstrap](../../src/bootstrap.cc#L672)、[topology](../../src/graph/topo.cc#L1765)、[transport connect](../../src/transport.cc#L21)。
2. 性能不稳定：开 `TUNING,COLL,PROXY,NET`，对照 [tuning model](../../src/graph/tuning.cc#L351)、[enqueue](../../src/enqueue.cc#L3124)、[proxy progress](../../src/proxy.cc#L1164)。
3. kernel 行为不清楚：开 `COLL`，再去看 [src/device/](../../src/device/) 生成的 kernel 组合。
4. 插件没生效：开 `INIT,NET,ENV,PROFILE`，看 `plugin_open.cc` 输出的库名和实际路径。

官方资料中，[Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 适合查参数，[Logging and troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/logging.html) 适合查日志级别和 subsystem。

## RAS：运行时健康与故障观测

RAS 是 reliability、availability、serviceability 的缩写，核心线程在 [src/ras/ras.cc](../../src/ras/ras.cc#L90)。初始化时 [bootstrap.cc](../../src/bootstrap.cc#L672) 通过 `NCCL_RAS_ENABLE` 控制是否启用；启用后，communicator 初始化会调用 [ncclRasCommInit](../../src/ras/ras.cc#L90) 和 [ncclRasAddRanks](../../src/ras/ras.cc#L174)，把 rank 信息交给 RAS 线程。

| 文件 | 负责内容 |
| --- | --- |
| [src/ras/ras.cc](../../src/ras/ras.cc#L90) | RAS 线程、消息循环、本地通知。 |
| [src/ras/peers.cc](../../src/ras/peers.cc#L69) | peer 列表维护、dead peer 传播、RAS 网络重配置。 |
| [src/ras/rasnet.cc](../../src/ras/rasnet.cc#L196) | RAS socket 连接、keepalive、超时处理。 |
| [src/ras/collectives.cc](../../src/ras/collectives.cc#L200) | RAS 内部 collective request/response。 |
| [src/ras/client_support.cc](../../src/ras/client_support.cc#L165) | `ncclras` 客户端连接和 JSON dump 支持。 |
| [src/ras/client.cc](../../src/ras/client.cc#L458) | `ncclras` 命令行工具入口。 |

通信挂住或 rank 异常时，RAS 是运行时健康信息的主要入口；数据归约路径仍以 01 到 03 篇中的 host 调度、transport/proxy 和 device kernel 为主线。

## 构建系统：Makefile、CMake 和 device 代码生成

这个仓库同时保留 Makefile 和 CMake。顶层 [Makefile](../../Makefile#L33) 会把 `src`、`docs/examples`、`bindings`、`pkg` 等目标分发出去；[src/Makefile](../../src/Makefile#L11) 定义 public headers、库文件、工具二进制和安装目标；公共 CUDA 编译参数在 [makefiles/common.mk](../../makefiles/common.mk#L8)。

Device 端代码比较特殊。[src/device/Makefile](../../src/device/Makefile#L80) 会调用 [src/device/generate.py](../../src/device/generate.py) 生成不同 collective、datatype、reduction op、algorithm、protocol 组合需要的源文件；symmetric kernel 还有 [src/device/symmetric/generate.py](../../src/device/symmetric/generate.py)。这对应 device kernel 的“模板 + 代码生成 + 函数表”模式。

CMake 入口在 [CMakeLists.txt](../../CMakeLists.txt#L25)，其中 [CMakeLists.txt#L262](../../CMakeLists.txt#L262) 会按选项加入 plugin examples，[CMakeLists.txt#L269](../../CMakeLists.txt#L269) 加入核心 `src`，后面再接 bindings、contrib 和包格式。发行包逻辑在 [pkg/](../../pkg/)。

## OS wrapper 与外部库封装

NCCL 需要同时碰 CUDA、NVML、IB verbs、GDRCopy、socket、IPC、动态库等系统接口。为了降低平台差异，源码里有一层 wrapper：

| 模块 | 入口 |
| --- | --- |
| OS/socket/thread/dlopen | [src/os/linux.cc](../../src/os/linux.cc)、[src/os/linux_socket_pair.cc](../../src/os/linux_socket_pair.cc)、[src/include/os.h](../../src/include/os.h) |
| CUDA Driver/Runtime wrapper | [src/include/cudawrap.h](../../src/include/cudawrap.h)、[src/misc/cudawrap.cc](../../src/misc/cudawrap.cc) |
| NVML / IB / GDR wrapper | [src/include/nvmlwrap.h](../../src/include/nvmlwrap.h)、[src/include/ibvwrap.h](../../src/include/ibvwrap.h)、[src/include/gdrwrap.h](../../src/include/gdrwrap.h) |
| Windows stubs | [src/os/windows_stubs.cc](../../src/os/windows_stubs.cc) |

调试“为什么这台机器没有走 NVLink/IB/GDR”时，这层经常是入口之一：它能显示某个动态库是否存在、某个系统能力是否初始化成功。

## Bindings 与 contrib：公共 API 上的扩展示例

[bindings/nccl4py](../../bindings/nccl4py/README.md) 展示了 Python 绑定，适合从应用侧理解 `nccl.h` 和 `nccl_device.h` 怎么暴露出去。[bindings/ir](../../bindings/ir/nccl_device_wrapper.h) 更偏 device API/IR 包装。

[contrib/README.md](../../contrib/README.md) 明确说明 `contrib/` 里的项目应基于 NCCL public API，而不是依赖内部头文件。当前比较值得扫一眼的有：

| contrib | 用途 |
| --- | --- |
| [contrib/nccl_ep](../../contrib/nccl_ep/README.md) | 面向专家并行/分布式训练场景的扩展。 |
| [contrib/nccl_checkpoint](../../contrib/nccl_checkpoint/README.md) | checkpoint 相关辅助能力。 |
| [contrib/nccl_ubx](../../contrib/nccl_ubx/README.md) | 基于 symmetric memory / multimem 思路的实验扩展。 |
| [contrib/nccl_xfer](../../contrib/nccl_xfer/README.md) | 数据传输相关扩展。 |
| [contrib/custom_algos](../../contrib/custom_algos/README.md) | 自定义算法示例。 |

## 推荐的源码阅读路线

1. Net plugin：读 [plugins/net/README.md](../../plugins/net/README.md)，再读 [plugins/net/example/plugin.c](../../plugins/net/example/plugin.c#L15)，最后回到 [src/plugin/net.cc](../../src/plugin/net.cc#L97) 看 NCCL core 如何适配版本。
2. 算法选择调优：先看 [02_topology_transport_proxy.md](02_topology_transport_proxy.md)，再读 [src/plugin/tuner.cc](../../src/plugin/tuner.cc#L39) 和 [plugins/tuner/README.md](../../plugins/tuner/README.md)。
3. 运行问题定位：先跑 [docs/examples/DEBUG_GUIDE.md](../examples/DEBUG_GUIDE.md)，再按日志 subsystem 回到对应源码目录。
4. 构建产物：从 [Makefile](../../Makefile#L33) 进 [src/Makefile](../../src/Makefile#L11)，再看 [src/device/Makefile](../../src/device/Makefile#L80) 为什么要生成 device 源码。

## 参考资料

- 官方 [Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。
- 官方 [Logging and troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/logging.html)。
- 官方 [NCCL GitHub repository](https://github.com/NVIDIA/nccl)，适合检索 issue/discussion 中的插件、RAS、debug 经验。
