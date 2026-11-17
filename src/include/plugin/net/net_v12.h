/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_V12_H_
#define NET_V12_H_

#define NCCL_NET_MAX_DEVS_PER_NIC_V12 8

typedef struct {
  int ndevs;
  int devs[NCCL_NET_MAX_DEVS_PER_NIC_V12];
} ncclNetVDeviceProps_v12_t;

#define NCCL_NET_TRAFFIC_CLASS_UNDEF -1

typedef struct {
  // Plugin-specific TC value
  int trafficClass;
} ncclNetCommConfig_v12_t;

typedef struct {
  // 网卡名，主要用于日志和填写拓扑
  char* name;
  // /sys 里的 PCI 设备路径，定位 NIC 在 PCI 拓扑中的位置，决定它与哪个 GPU/CPU 亲和，ncclTopoFillNet 用它建 net 节点
  char* pciPath;
  // NIC 芯片唯一标识，一块卡有多个 PCI function(物理/虚拟)时，靠它区分"是不是同一颗芯片"，写入 XML guid 属性
  uint64_t guid;
  /*
  位标志：NCCL_PTR_HOST=0x1 ｜ NCCL_PTR_CUDA=0x2 ｜ NCCL_PTR_DMABUF=0x4
  决定能注册哪类内存：
    含 NCCL_PTR_CUDA 即表示支持 GPUDirect RDMA，
  IB 插件只有在 ncclIbGdrSupport() 成功时才 ptrSupport |= NCCL_PTR_CUDA(GDR support via nv_peermem)，
  用 (props.ptrSupport & NCCL_PTR_CUDA)==0 → continue 过滤掉不支持 GDR 的设备，
  进而决定 useGdr / NCCL_DIRECT_NIC 标志。
    DMABUF 位表示可走 regMrDmaBuf
  */
  int ptrSupport;
  /*
  声明插件的内存注册(regMr)是不是"全局"的，即：
    =1：注册有快速注册缓存(fast registration cache)，产出的 mhandle 不绑定某个具体 comm(sendComm/recvComm)，可跨连接/跨 comm 复用;
    =0：注册是按 comm 的 / 没有这种缓存，每次注册都可能是"真注册"(开销大)。
  NCCL 有个零拷贝优化叫 user buffer registration：把用户的 send/recv buffer 向网卡注册一次，
  之后多轮集合操作复用同一个 mhandle，省掉每次把数据拷进 staging buffer 再发的开销。
  但 regMr(向网卡注册内存：pin 页、在 NIC 建 MR)本身可能很贵。所以:
    若插件有快速注册缓存(regIsGlobal=1)，注册/复用便宜，user buffer registration 划算，可以开;
    若没有(regIsGlobal=0)，注册昂贵且不能跨 comm 复用，"user buffer registration 可能被禁用"，因为注册成本会盖过零拷贝收益。
  这个字段存在的意图是让插件告诉 NCCL "我的注册便不便宜、能不能全局复用"，NCCL 据此判断要不要走 user buffer 注册这条路。
  */
  int regIsGlobal;
  // 接收后是否强制 flush，某些 NIC/GDR 场景需显式 iflush 保证写入 CUDA 内存后对 GPU 可见(对应 transport 里的 needFlush/ncclTopoFlushType)
  int forceFlush;
  // 端口速率(Mbps)，写入 XML speed，建图时换算成网卡带宽,影响 ring/tree 选路与算法调优
  int speed;
  // 端口号，多端口卡区分端口号，写入 XML port
  int port;
  // 网络延迟，写入 XML latency，供延迟相关的调优判断
  float latency;
  // 最多能创建多少 comm(IB 用 ibDev->maxQp，socket 用 65536)，写入 XML maxconn
  int maxComms;
  // 单次 grouped receive 最多合并多少个 recv，对应 irecv(n, data[], sizes[], tags[], ...) 里 n 的上限(IB=NCCL_NET_IB_MAX_RECVS，socket=1 即不能合并)
  int maxRecvs;
  /*
  网络卸载类型，卸载(offload)= 把原本由 CPU(host)承担的活儿,转移给专用硬件去做。
  传统路径 NCCL_NET_DEVICE_HOST(不卸载)
    CPU 上的 proxy 线程负责调插件的 isend/irecv/test,在网络和 GPU 显存之间搬数据、推进请求;
    GPU kernel 只管在 buffer 里产出/消费数据,网络 I/O 全由 CPU proxy 代劳;
    代价:CPU proxy 处在关键路径上,带来延迟、CPU 占用,以及 GPU 与 proxy 之间的同步开销。
  卸载到 GPU device：NCCL_NET_DEVICE_UNPACK
    把接收侧的"拆包/落位"交给 GPU kernel 直接做；
    网卡把数据 DMA 过来后，由 GPU device 代码用设备句柄直接把数据 unpack 到最终位置，省掉 CPU 中转和一次拷贝。
  卸载到"GPU 发起"：三个 GIN 后端
    值	              后端	                                  谁敲网卡	                      是否还需 CPU proxy
    GIN_PROXY	        proxy 型	                             GPU 提请求，CPU proxy 协助推进	    需要
    GIN_GDAKI	        GPUDirect Async Kernel-Initiated	     GPU 直接敲网卡 doorbell	        不需要
    GIN_GPI	          另一种 GPU 发起后端	                     GPU 发起	                        视实现
  */
  ncclNetDeviceType netDeviceType;
  /*
  该卸载类型的版本号，与 NCCL 支持的版本(如 NCCL_NET_DEVICE_UNPACK_VERSION)比对，
  不匹配会导致该插件不被 assign (isAssigned=false 的来源之一)
  */
  int netDeviceVersion;
  /*
  ncclNetVDeviceProps_v12_t { int ndevs; int devs[8]; }，
  描述由最多 8 个物理设备聚合成的虚拟网卡，配合 makeVDevice 做多轨合并 (multi-rail bonding)
  */
  ncclNetVDeviceProps_v12_t vProps;
  // 单次点对点传输最大字节数，约束 ring/tree 路径的 isend/irecv 点对点操作
  size_t maxP2pBytes;
  /*
  单次集合操作传输最大字节数，约束 collnet 路径的 iallreduce 网内规约操作
  collnet = collective network，指在网络硬件(交换机/网卡)里直接完成规约、聚合的能力，典型实现是 NVIDIA SHARP。
  比如 AllReduce 不再靠 GPU 之间来回搬数据做 ring/tree，而是把数据发进网络，由交换机里的规约引擎算完再广播回来。
  */
  size_t maxCollBytes;
  // 单个 multi-request 里最多支持多少请求(批量 isend/irecv 分组上限)
  int maxMultiRequestSize;
  // 该 netdev 关联的 rail ID，rail-optimized(轨道优化)网络拓扑，NCCL_NET_ID_UNDEF(-1)表示未定义
  int16_t railId;
  // 该 netdev 关联的 plane ID，多平面网络，写入 XML plane 属性
  int16_t planeId;
} ncclNetProperties_v12_t;

#define NCCL_NET_ATTR_UNDEF -1

#define NCCL_NET_ID_UNDEF -1

#define NCCL_NET_ATTR_INIT \
  { \
    {NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF}, /* sendCommAttr */ \
    {NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF}, /* recvCommAttr */ \
    (uint32_t)NCCL_NET_ATTR_UNDEF, /* op */ \
    (uint32_t)NCCL_NET_ATTR_UNDEF, /* algo */ \
    (uint32_t)NCCL_NET_ATTR_UNDEF, /* proto */ \
  }

typedef struct {
  int32_t maxConcurrentPeers;
  int32_t minConcurrentPeers;
  int32_t maxFlowsPerPeer;
  int32_t minFlowsPerPeer;
} ncclNetCommAttr_v12_t;

typedef struct {
  ncclNetCommAttr_v12_t sendCommAttr;
  ncclNetCommAttr_v12_t recvCommAttr;
  uint32_t op;
  uint32_t algo;
  uint32_t proto;
} ncclNetAttr_v12_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclNetCommConfig_v12_t* config, ncclDebugLogger_t logFunction,
                       ncclProfilerCallback_t profFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  // If *sendDevComm points to a valid object, then NCCL is requesting device offload for this connection
  ncclResult_t (*connect)(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v12_t** sendDevComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  // If *recvDevComm points to a valid object, then NCCL is requesting device offload for this connection
  ncclResult_t (*accept)(void* listenComm, void** recvComm, ncclNetDeviceHandle_v12_t** recvDevComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles,
                        void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Copy the given mhandle to a dptr in a format usable by this plugin's device code
  ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);

  // Notify the plugin that a recv has completed by the device
  ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);

  // Virtual NIC APIs. makeVDevice will create a virtual NIC given the specified properties, and tell the caller
  // what index this new vNIC exists at
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v12_t* props);
  // Finalize the network.
  ncclResult_t (*finalize)(void* ctx);

  ncclResult_t (*setNetAttr)(void* ctx, ncclNetAttr_v12_t* netAttr);
} ncclNet_v12_t;

typedef struct {
  void* mhandle;
  void* address;
  size_t size;
} ncclNetSGE_v12_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* collComm, void* data, size_t size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                              void** mhandle);
  ncclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, size_t count, ncclDataType_t dataType,
                             ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  ncclResult_t (*iallgather)(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v12_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes, void* sendMhandle,
                             void** request);
  ncclResult_t (*ireducescatter)(void* collComm, int nSendParts, ncclNetSGE_v12_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes, ncclDataType_t dataType,
                                 ncclRedOp_t redOp, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Create a virtual NIC given the specified properties, which can be accessed at device index d
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v12_t* props);
  // Finalize the collective network.
  ncclResult_t (*finalize)(void* ctx);
} ncclCollNet_v12_t;
#endif // end include guard
