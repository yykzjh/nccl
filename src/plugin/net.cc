/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net.h"
#include "bootstrap.h"
#include "checks.h"
#include "plugin.h"
#include "nccl_net.h"

#include <string.h>
#include <errno.h>
#include <mutex>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <unistd.h>

#if defined(NCCL_OS_LINUX)
typedef ncclNet_t* getNcclNet_t(void* netPluginLib);
typedef ncclCollNet_t* getNcclCollNet_t(void* netPluginLib);

extern getNcclNet_t getNcclNet_v6;
extern getNcclNet_t getNcclNet_v7;
extern getNcclNet_t getNcclNet_v8;
extern getNcclNet_t getNcclNet_v9;
extern getNcclNet_t getNcclNet_v10;
extern getNcclNet_t getNcclNet_v11;
extern getNcclNet_t getNcclNet_v12;
extern getNcclCollNet_t getNcclCollNet_v6;
extern getNcclCollNet_t getNcclCollNet_v7;
extern getNcclCollNet_t getNcclCollNet_v8;
extern getNcclCollNet_t getNcclCollNet_v9;
extern getNcclCollNet_t getNcclCollNet_v10;
extern getNcclCollNet_t getNcclCollNet_v11;
extern getNcclCollNet_t getNcclCollNet_v12;

// 表示 NET 插件被多少 comm 引用的基础计数，设为非 0 决定 NET 插件 .so 不会被释放
NCCL_PARAM(NetPluginRefCount, "NET_PLUGIN_REF_COUNT", 0);
#define NCCL_NET_VERSION_COUNT 7
int ncclNetVersion[NCCL_NET_VERSION_COUNT] = {12, 11, 10, 9, 8, 7, 6};
getNcclNet_t* getNcclNet[NCCL_NET_VERSION_COUNT] = {getNcclNet_v12, getNcclNet_v11, getNcclNet_v10, getNcclNet_v9,
                                                    getNcclNet_v8,  getNcclNet_v7,  getNcclNet_v6};
getNcclCollNet_t* getNcclCollNet[NCCL_NET_VERSION_COUNT] = {getNcclCollNet_v12, getNcclCollNet_v11, getNcclCollNet_v10,
                                                            getNcclCollNet_v9,  getNcclCollNet_v8,  getNcclCollNet_v7,
                                                            getNcclCollNet_v6};
#endif

// Count only built-in transports registered below. Windows does not build the
// real IB transport (only a failure stub is linked), so it registers
// NetworkDirect and Socket instead.
#define NCCL_NET_NUM_INTERNAL_PLUGINS 2

typedef enum ncclNetPluginState {
  ncclNetPluginStateDisabled = -2,       // Plugin library failed to initialize
  ncclNetPluginStateLoadFailed = -1,       // Plugin library failed to load
  ncclNetPluginStateLoadReady = 0,        // Plugin library is ready to be loaded
  ncclNetPluginStateInitReady = 1,        // Plugin library is loaded and ready to be initialized
  ncclNetPluginStateEnabled = 2,        // Plugin library is loaded and initialized
} ncclNetPluginState_t;

#define MAX_STR_LEN 255
typedef struct netPluginLib {
  char name[MAX_STR_LEN];                       // Name of the plugin library
  void* dlHandle;                               // Handle to the plugin library
  ncclNet_t* ncclNet;                           // Pointer to the ncclNet_t structure
  int ncclNetVer;                               // Version of the nccl net plugin
  ncclCollNet_t* ncclCollNet;                   // Pointer to the ncclCollNet_t structure
  ncclNetPluginState_t ncclNetPluginState;      // State of the nccl net plugin
  ncclNetPluginState_t ncclCollNetPluginState;  // State of the nccl coll net plugin
  int ncclNetPluginRefCount;                    // Reference count for the nccl net plugin
  int netPhysDevs;                              // ncclNet - number of physical devices
  int netVirtDevs;                              // ncclNet - number of virtual devices
  int collNetPhysDevs;                          // ncclCollNet -  number of physical devices
  int collNetVirtDevs;                          // ncclCollNet -  number of virtual devices
} netPluginLib_t;

static int pluginCount = 0;
static netPluginLib_t netPluginLibs[NCCL_NET_MAX_PLUGINS] = {0};  // 最多 16 个插件
static std::mutex netPluginMutex;
static std::once_flag initPluginLibsOnceFlag;

#if defined(NCCL_OS_LINUX)
static ncclResult_t ncclNetPluginUnload(netPluginLib_t* pluginLib) {
  // 如果有 .so 动态库 handle 并且 NET 插件的引用计数为 0，则释放 handle
  if ((pluginLib->dlHandle) && ((pluginLib->ncclNetPluginRefCount) == 0)) {
    INFO(NCCL_DESTROY | NCCL_NET, "Unloading plugin %s", pluginLib->name);
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));

    // 重置字段但是保留名称以重用
    // Reset fields but preserve name, to be reused when reloading
    pluginLib->dlHandle = NULL;
    pluginLib->ncclNet = NULL;
    pluginLib->ncclNetVer = 0;
    pluginLib->ncclCollNet = NULL;
    pluginLib->ncclNetPluginState = ncclNetPluginStateLoadReady;
    pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadReady;
    pluginLib->ncclNetPluginRefCount = 0;
    pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
    pluginLib->collNetPhysDevs = pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
  }
  return ncclSuccess;
}

static ncclResult_t ncclNetPluginLoad(netPluginLib_t* pluginLib) {
  // 打开 .so 句柄
  pluginLib->dlHandle = ncclOpenNetPluginLib(pluginLib->name);

  if (pluginLib->dlHandle == nullptr) goto fail;

  // 从新到旧遍历所有 NET 插件 ABI 版本
  // load ncclNet
  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
    pluginLib->ncclNetVer = ncclNetVersion[i];
    // 通过最新版本的 getNcclNet 函数从 .so handle 中解析出 ncclNet_t 插件对象
    pluginLib->ncclNet = getNcclNet[i](pluginLib->dlHandle);
    if (pluginLib->ncclNet) break;
  }

  // if we fail to find a net, exit
  if (pluginLib->ncclNet == nullptr) {
    const char* netPlugin = ncclGetEnv("NCCL_NET_PLUGIN");
    if (netPlugin) {
      ATTN("External network plugin %s is unsupported",
           (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));
    } else {
      INFO(NCCL_INIT | NCCL_NET, "External network plugin %s is unsupported",
           (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));
    }
    goto fail;
  }

  pluginLib->ncclNetPluginState = ncclNetPluginStateInitReady;

  // load ncclCollNet
  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
    pluginLib->ncclCollNet = getNcclCollNet[i](pluginLib->dlHandle);
    if (pluginLib->ncclCollNet) break;
  }

  if (pluginLib->ncclCollNet == nullptr) pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;
  else pluginLib->ncclCollNetPluginState = ncclNetPluginStateInitReady;

  INFO(NCCL_INIT | NCCL_NET, "Successfully loaded external network plugin %s",
       (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));
exit:
  return ncclSuccess;
fail:
  if (pluginLib->dlHandle) {
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));
  }
  pluginLib->dlHandle = nullptr;
  pluginLib->ncclNetPluginState = ncclNetPluginStateLoadFailed;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;
  goto exit;
}
#endif

// 获取指定索引的 NET 设备属性
ncclResult_t ncclNetCheckDeviceVersion(struct ncclComm* comm, ncclNet_t* net, int dev) {
  ncclNetProperties_t props;

  // 获取指定 NET 设备属性
  NCCLCHECK(net->getProperties(dev, &props));
  // 网络卸载类型
  ncclNetDeviceType type = props.netDeviceType;
  switch (type) {
  case NCCL_NET_DEVICE_HOST:
    break;
  case NCCL_NET_DEVICE_UNPACK:
    if (props.netDeviceVersion == NCCL_NET_DEVICE_UNPACK_VERSION) {
      INFO(NCCL_INIT, "Using NCCL_NET_DEVICE_UNPACK net plugin version %d", props.netDeviceVersion);
      return ncclSuccess;
    } else {
      WARN("NCCL_DEVICE_UNPACK plugin has incompatible version %d, this NCCL build is compatible with %d, not using it",
           props.netDeviceVersion, NCCL_NET_DEVICE_UNPACK_VERSION);
      return ncclInternalError;
    }
  default:  // NET 插件不支持 GIN 相关卸载类型，GIN 是另一个子系统，契约违规报错
    WARN("Unknown device code index %d", type);
    return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclProfilerCallback_t ncclNetGetProfilerCallback() {
#if defined(NCCL_OS_LINUX)
  return ncclProfilerCallback;
#elif defined(NCCL_OS_WINDOWS)
  return nullptr;
#endif
}

static ncclResult_t ncclNetPluginInit(struct ncclComm* comm, netPluginLib_t* pluginLib) {
  int ndev;
  bool initCompleted = false;
  // netContext 是每个 comm 独立的，所以哪怕插件已被别的 comm 启用，新 comm 仍要再 init() 一次拿到自己的 context
  // Init must be called for each new comm to set the right context
  if (pluginLib->ncclNetPluginState >= ncclNetPluginStateInitReady && pluginLib->ncclNet) {
    ncclNetCommConfig_t commConfig = {};
    // 如果未定义就是 -1，否则采用 comm->config.trafficClass 指定值
    commConfig.trafficClass =
      comm->config.trafficClass == NCCL_CONFIG_UNDEF_INT ? NCCL_NET_TRAFFIC_CLASS_UNDEF : comm->config.trafficClass;
    /*
    调用 NET 插件的初始化接口：
      &comm->netContext:输出参数,插件把它的每-comm 上下文句柄写这里,后续 listen/connect/isend/irecv 都靠它;
      comm->commHash:64 位 communicator 标识;
      &commConfig:上面的 trafficClass 配置;
      ncclDebugLog:让插件复用 NCCL 的日志系统;
      ncclProfilerCallback:让插件能向 NCCL profiler 发事件。
    */
    if (pluginLib->ncclNet->init(&comm->netContext, comm->commHash, &commConfig, ncclDebugLog,
                                 ncclNetGetProfilerCallback()) != ncclSuccess) {
      goto fail;
    }
    initCompleted = true;
  }
  // 设备数是硬件/插件属性,与具体 comm 无关,第一个 comm 探测一次缓存下来即可,避免每个 comm 重复探测
  // Detection of the devices is only done when the plugin is being initialized the first time
  if (pluginLib->ncclNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclNet) {
    // 获取物理网卡数
    if (pluginLib->ncclNet->devices(&ndev) != ncclSuccess || ndev <= 0) goto fail;
    // 这两个值后面被 ncclNetGetDevCount 读取,用于拓扑探测(ncclTopoGetSystem)
    pluginLib->netPhysDevs = ndev;
    pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;  // -1，表示虚拟设备数还没定(后续 makeVDevice / ncclNetSetVirtDevCount 再填)
  }
  pluginLib->ncclNetPluginState = ncclNetPluginStateEnabled;  // NET 插件 Enabled
  INFO(NCCL_INIT | NCCL_NET, "Initialized NET plugin %s", pluginLib->ncclNet->name);

  // COLL 插件进行与 NET 插件相同初始化流程

  // Init must be called for each new comm to set the right context
  if (pluginLib->ncclCollNetPluginState >= ncclNetPluginStateInitReady && pluginLib->ncclCollNet) {
    if (pluginLib->ncclCollNet->init(&comm->collNetContext, comm->commHash, ncclDebugLog) != ncclSuccess) {
      pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
    }
  }
  // Detection of the devices is only done when the plugin is being initialized the first time
  if (pluginLib->ncclCollNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclCollNet) {
    if (pluginLib->ncclCollNet->devices(&ndev) != ncclSuccess || ndev <= 0) {
      pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
    } else {
      pluginLib->collNetPhysDevs = ndev;
      pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
      pluginLib->ncclCollNetPluginState = ncclNetPluginStateEnabled;
    }
  }

exit:
  return ncclSuccess;
fail:
  ATTN("Failed to initialize NET plugin %s", pluginLib->ncclNet->name);
  if (initCompleted) pluginLib->ncclNet->finalize(comm->netContext);
  pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->collNetPhysDevs = pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->ncclNetPluginState = ncclNetPluginStateDisabled;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
  goto exit;
}

static ncclResult_t ncclNetPluginAssignToComm(struct ncclComm* comm, int pluginIndex, bool* isAssigned) {
  *isAssigned = false;

  // 选第一个 NET 设备检查网络卸载类型是否正常
  if (ncclSuccess != ncclNetCheckDeviceVersion(comm, netPluginLibs[pluginIndex].ncclNet, 0)) return ncclSuccess;

  // 将 NET 相关内容保存到 comm 中，引用计数 +1
  if (netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateEnabled) {
    comm->ncclNet = netPluginLibs[pluginIndex].ncclNet;
    comm->ncclNetVer = netPluginLibs[pluginIndex].ncclNetVer;
    comm->netPluginIndex = pluginIndex;
    netPluginLibs[pluginIndex].ncclNetPluginRefCount++;
    *isAssigned = true;
    INFO(NCCL_INIT | NCCL_NET, "Assigned NET plugin %s to comm", netPluginLibs[pluginIndex].ncclNet->name);
    if (netPluginLibs[pluginIndex].ncclCollNetPluginState >= ncclNetPluginStateEnabled) {
      comm->ncclCollNet = netPluginLibs[pluginIndex].ncclCollNet;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclNetPluginDisableOtherExternal(int pluginIndex) {
  // 只有启用外部插件才禁用其他插件
  // Only if an external plugin is enabled, disable other external plugins
  if (pluginIndex >= (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) return ncclSuccess;
  char names[MAX_STR_LEN * (NCCL_NET_MAX_PLUGINS - NCCL_NET_NUM_INTERNAL_PLUGINS)] = {0};
  for (int i = 0; i < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS); i++) {
    if (i != pluginIndex) {
      // Append all disabled plugin names to a string
      snprintf(names + strlen(names), sizeof(names) - strlen(names), (strlen(names) == 0) ? "%s" : ", %s",
               netPluginLibs[i].name);
      netPluginLibs[i].ncclNetPluginState = ncclNetPluginStateDisabled;
    }
  }
  if (strlen(names) > 0) {
    INFO(NCCL_INIT | NCCL_NET, "Disabling external plugins: %s", names);
  }
  return ncclSuccess;
}

static void ncclNetPluginRegisterInternal(int* pluginCounter, ncclNet_t* net, int version) {
  netPluginLib_t* pluginLib = &netPluginLibs[*pluginCounter];
  pluginLib->ncclNet = net;
  pluginLib->ncclNetVer = version;
  pluginLib->ncclNetPluginState = ncclNetPluginStateInitReady;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
  pluginLib->netPhysDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->collNetPhysDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
  (*pluginCounter)++;
}

static void initPluginLibsOnceFunc() {
  int pluginCounter = 0;
  memset(netPluginLibs, 0, NCCL_NET_MAX_PLUGINS * sizeof(netPluginLib_t));
#if defined(NCCL_OS_LINUX)
  char* netPluginName = nullptr;
  const char* defaultNetPlugin = "libnccl-net.so";
  const char* envNetPlugin = nullptr;
  char* envNetPluginList = nullptr;
  char* savePtr = nullptr;
  envNetPlugin = ncclGetEnv("NCCL_NET_PLUGIN");  // 获取 NET 插件名称或路径
  if (envNetPlugin) {
    INFO(NCCL_ENV | NCCL_NET, "NCCL_NET_PLUGIN set by environment to %s", envNetPlugin);
    if (strcasecmp(envNetPlugin, "none") == 0) envNetPlugin = "";  // 如果指定 None 就是不用插件
    envNetPluginList = strdup(envNetPlugin);
    // 分割多个插件列表：gcp,efa,accl
    // Iterate over list until the list is empty
    netPluginName = strtok_r(envNetPluginList, ",", &savePtr);
    while (netPluginName) {
      // 有 ib 和 socket 两个内部插件，所以最多能指定 16 - 2 = 14 个外部插件列表
      // We have 2 internal plugins (ib and socket)
      // So, we can have at most( NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS)) in the NCCL_NET_PLUGIN list
      if (pluginCounter >= (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS))) {
        INFO(NCCL_NET | NCCL_ENV, "NCCL_NET_PLUGIN list contains more than %d plugins, ignoring the rest",
             (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS + 1)));
        break;
      }
      // need to leave space for the name + "\n"
      if ((strlen(netPluginName) + 1) <= MAX_STR_LEN) {  // 长度限制 255
        // 初始化当前插件 slot 的状态
        netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;
        netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();
        strcpy(netPluginLibs[pluginCounter].name, netPluginName);
        pluginCounter++;
      } else {
        INFO(NCCL_NET | NCCL_ENV,
             "NCCL_NET_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", netPluginName,
             MAX_STR_LEN);
      }
      netPluginName = strtok_r(nullptr, ",", &savePtr);
    }
    free(envNetPluginList);
  } else {
    // 没有参数指定外部插件，使用默认外部插件名：libnccl-net.so
    // Add default net plugin
    netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;
    netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();
    strcpy(netPluginLibs[pluginCounter++].name, defaultNetPlugin);
  // 添加两个内部插件 slots
  }

  ncclNetPluginRegisterInternal(&pluginCounter, &ncclNetIb, 0);
  ncclNetPluginRegisterInternal(&pluginCounter, &ncclNetSocket, 0);
#elif defined(NCCL_OS_WINDOWS)
  ncclNetPluginRegisterInternal(&pluginCounter, &ncclNetNd, 12);
  ncclNetPluginRegisterInternal(&pluginCounter, &ncclNetSocket, 12);
#endif
  pluginCount = pluginCounter;
}

static ncclResult_t ncclNetPluginFinalize(struct ncclComm* comm, int pluginIndex) {
  // NET 插件内部销毁
  NCCLCHECK(netPluginLibs[pluginIndex].ncclNet->finalize(comm->netContext));
  // CollNet 插件内部销毁
  if (netPluginLibs[pluginIndex].ncclCollNet &&
      netPluginLibs[pluginIndex].ncclCollNetPluginState == ncclNetPluginStateEnabled) {
    NCCLCHECK(netPluginLibs[pluginIndex].ncclCollNet->finalize(comm->collNetContext));
  }
  netPluginLibs[pluginIndex].ncclNetPluginRefCount--;
#if defined(NCCL_OS_LINUX)
  // 外部 NET 插件尝试释放 handle 并重置字段
  if (pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) {
    NCCLCHECK(ncclNetPluginUnload(&netPluginLibs[pluginIndex]));
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclNetInit(struct ncclComm* comm) {
  bool ncclNetPluginInitialized = false;
  comm->ncclCollNet = nullptr;
  // 初始化 NET 插件状态，此时还没有打开 .so 和插件对象，准备好 Load
  std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);
  // 加锁
  std::lock_guard<std::mutex> lock(netPluginMutex);
  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
#if defined(NCCL_OS_LINUX)
    // 外部插件并且状态正常，加载插件
    if ((pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) &&
        (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateLoadReady)) {
      // 加载 .so handle 和 ncclNet_t 插件对象，准备好 Init
      NCCLCHECK(ncclNetPluginLoad(&netPluginLibs[pluginIndex]));
    }
#endif
    // comm->config.netName 可以限定 NET 插件名
    if ((netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateInitReady) &&
        (!comm->config.netName || (strcasecmp(comm->config.netName, netPluginLibs[pluginIndex].ncclNet->name) == 0))) {
      // 初始化 NET 插件，状态为 Enabled
      // plugin init must be done by all comms to setup the context, therefore we use ">="
      NCCLCHECK(ncclNetPluginInit(comm, &netPluginLibs[pluginIndex]));
      if (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateEnabled) {
        bool isAssigned = false;
        NCCLCHECK(ncclNetPluginAssignToComm(comm, pluginIndex, &isAssigned));
        if (isAssigned) {
          // 一个外部插件成功赋予给 comm。disable 其他所有插件。
          // If one external plugin is assigned to a comm, then disable all other external plugins
          ncclNetPluginDisableOtherExternal(pluginIndex);
          ncclNetPluginInitialized = true;
          break;
        } else {
          ncclNetPluginFinalize(comm, pluginIndex);
        }
      }
    }
  }
  if (ncclNetPluginInitialized) return ncclSuccess;
  WARN("Failed to initialize any NET plugin");
  return ncclInvalidUsage;
}

ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t ret = ncclSuccess;
  comm->netContext = parent->netContext;
  comm->collNetContext = parent->collNetContext;
  comm->ncclNet = parent->ncclNet;
  comm->ncclCollNet = parent->ncclCollNet;
  comm->netPluginIndex = parent->netPluginIndex;
  comm->ncclNetVer = parent->ncclNetVer;
  if (comm->config.netName != NCCL_CONFIG_UNDEF_PTR && strcasecmp(comm->config.netName, parent->config.netName)) {
    WARN("Comm config netName (%s) does not match the parent (%s)", comm->config.netName, parent->config.netName);
    ret = ncclInvalidUsage;
  }
  if (comm->config.trafficClass != NCCL_CONFIG_UNDEF_INT && comm->config.trafficClass != parent->config.trafficClass) {
    INFO(NCCL_INIT, "Comm config trafficClass (%d) does not match the parent (%d)", comm->config.trafficClass,
         parent->config.trafficClass);
  }
  return ret;
}

ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->netPluginIndex;
  std::lock_guard<std::mutex> lock(netPluginMutex);
  NCCLCHECK(ncclNetPluginFinalize(comm, pluginIndex));
  return ncclSuccess;
}

ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclNetPluginState != ncclNetPluginStateEnabled ||
      netPluginLibs[netPluginIndex].netPhysDevs == NCCL_UNDEF_DEV_COUNT)
    goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = netPluginLibs[netPluginIndex].netPhysDevs;
  *nVirtDevs = netPluginLibs[netPluginIndex].netVirtDevs;
  return ncclSuccess;
fail:
  WARN("trying to access the number of devices of an uninitialized netPlugin[%d]", netPluginIndex);
  return ncclInternalError;
}

ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclCollNetPluginState != ncclNetPluginStateEnabled ||
      netPluginLibs[netPluginIndex].collNetPhysDevs == NCCL_UNDEF_DEV_COUNT)
    goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = netPluginLibs[netPluginIndex].collNetPhysDevs;
  *nVirtDevs = netPluginLibs[netPluginIndex].collNetVirtDevs;
  return ncclSuccess;
fail:
  WARN("trying to access the number of devices of an uninitialized netPlugin[%d]", netPluginIndex);
  return ncclInternalError;
}

ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclNetPluginState != ncclNetPluginStateEnabled || nVirtDevs < 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  netPluginLibs[netPluginIndex].netVirtDevs = nVirtDevs;
  return ncclSuccess;
fail:
  WARN("failed to set the number of devices for netPlugin[%d] to %d", netPluginIndex, nVirtDevs);
  return ncclInternalError;
}

ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclCollNetPluginState != ncclNetPluginStateEnabled || nVirtDevs < 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  netPluginLibs[netPluginIndex].collNetVirtDevs = nVirtDevs;
  return ncclSuccess;
fail:
  WARN("failed to set the number of devices for netPlugin[%d] to %d", netPluginIndex, nVirtDevs);
  return ncclInternalError;
}

ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport) {
  constexpr int GPU_BUF_SIZE = 2 * 1024 * 1024;
  // 新驱动直接查属性(CUDA ≥ 11.3)
#if CUDART_VERSION >= 11030  // 编译期 NCCL 构建用的 CUDA 版本
  // In CUDA 11.3 and later we can now query the cudaDevAttrGPUDirectRDMASupported attribute
  int driverVersion;
  CUDACHECK(cudaDriverGetVersion(&driverVersion));
  if (driverVersion >= 11030) {  // 运行期实际装的驱动版本
    int cudaDev, attr = 0;
    CUDACHECK(cudaGetDevice(&cudaDev));
    CUDACHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrGPUDirectRDMASupported, cudaDev));
    *gdrSupport = attr;
    return ncclSuccess;
  }
#endif
  // 老驱动"经验实测"(真的去注册一次 GPU 显存)
  // static 数组跨调用持久、进程内共享,最多 32 块 GPU
  static int gdrSupportMatrix[32] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  // 只有第一次(-1)才跑昂贵的实测，之后直接返回缓存的 0/1
  if (gdrSupportMatrix[comm->cudaDev] == -1) {
    int netDevs;
    NCCLCHECK(comm->ncclNet->devices(&netDevs));
    gdrSupportMatrix[comm->cudaDev] = 0;
    // 遍历网卡，过滤出"能注册 CUDA 指针"的
    for (int dev = 0; dev < netDevs; dev++) {
      // Find a net device which is GDR-capable
      ncclNetProperties_t props;
      NCCLCHECK(comm->ncclNet->getProperties(dev, &props));
      // 判断是否支持 GDR
      if ((props.ptrSupport & NCCL_PTR_CUDA) == 0) continue;

      // Allocate memory on the GPU and try to register it on the NIC.
      void *lComm = NULL, *sComm = NULL, *rComm = NULL;
      ncclNetHandle_t handle;
      char* gpuPtr = NULL;
      void* mHandle = NULL;
      ncclResult_t ret;
      // 在同一块网卡上 listen 出一个监听端 lComm,
      NCCLCHECKGOTONOWARN(comm->ncclNet->listen(comm->netContext, dev, &handle, &lComm), ret, cleanup1, NCCL_NET);

      bool connected;
      connected = false;
      // while 循环应对 connect/accept 的异步性(未就绪时 comm 为 NULL 就重试),并带 abortFlag 检查以便中止时跳出
      while (!connected) {
        // If we're aborting now, skip to cleanup
        if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
          goto cleanup2;
        }

        if (sComm == NULL) {
          // connect 到自己的 handle 得到发送端 sComm
          NCCLCHECKGOTONOWARN(comm->ncclNet->connect(comm->netContext, dev, &handle, &sComm, NULL), ret, cleanup2,
                              NCCL_NET);
        }
        // accept 得到接收端 rComm
        if (rComm == NULL) NCCLCHECKGOTONOWARN(comm->ncclNet->accept(lComm, &rComm, NULL), ret, cleanup2, NCCL_NET);

        connected = (rComm != NULL) && (sComm != NULL);
      }

      // 分配 2MB GPU buffer
      NCCLCHECKGOTONOWARN(ncclCudaMalloc(&gpuPtr, GPU_BUF_SIZE, comm->memManager), ret, cleanup2, NCCL_NET);
      /*
      发送端注册 GPU 显存。regMr 必须挂在一个 comm(send 或 recv)上，要验证收、发两个方向都能注册 GPU 显存，
      就需要一对连通的 comm。让网卡 listen + connect 到自己 + accept，就地造出 sComm/rComm 一对，是不依赖任何远端 rank 的最省事自测法。
      */
      NOWARN(ret = comm->ncclNet->regMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle), NCCL_NET);
      if (ret == ncclSuccess) {
        NCCLCHECKNOWARN(comm->ncclNet->deregMr(sComm, mHandle), NCCL_NET);  // 注销
        // 接受端也试试
        NCCLCHECKNOWARN(comm->ncclNet->regMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle), NCCL_NET);
        NCCLCHECKNOWARN(comm->ncclNet->deregMr(rComm, mHandle), NCCL_NET);
        gdrSupportMatrix[comm->cudaDev] = 1;  // 收发都支持注册 GPU 显存，GDR 可用
      }
      NCCLCHECK(ncclCudaFree(gpuPtr, comm->memManager));
    cleanup2:
      if (rComm != NULL) NCCLCHECK(comm->ncclNet->closeRecv(rComm));
      if (sComm != NULL) NCCLCHECK(comm->ncclNet->closeSend(sComm));
      NCCLCHECK(comm->ncclNet->closeListen(lComm));
    cleanup1:
      break;
    }
  }
  *gdrSupport = gdrSupportMatrix[comm->cudaDev];
  return ncclSuccess;
}
