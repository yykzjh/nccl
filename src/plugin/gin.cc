/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gin.h"
#include "bootstrap.h"
#include "checks.h"
#include "plugin.h"
#include "nccl_gin.h"
#include "gin/gin_host_proxy.h"

#include <string.h>
#include <errno.h>
#include <mutex>

typedef ncclGin_t* getNcclGin_t(void* ginPluginLib);

extern getNcclGin_t getNcclGin_v13;
extern getNcclGin_t getNcclGin_v14;
NCCL_PARAM(GinPluginRefCount, "GIN_PLUGIN_REF_COUNT", 0);
#define NCCL_GIN_VERSION_COUNT 2
int ncclGinVersion[NCCL_GIN_VERSION_COUNT] = {14, 13};
getNcclGin_t* getNcclGin[NCCL_GIN_VERSION_COUNT] = {getNcclGin_v14, getNcclGin_v13};

#define NCCL_GIN_NUM_RESERVED_PLUGINS 3
#define NCCL_GIN_NUM_INTERNAL_PLUGINS 2

typedef enum ncclGinPluginState {
  ncclGinPluginStateDisabled = -2,       // Plugin library failed to initialize
  ncclGinPluginStateLoadFailed = -1,       // Plugin library failed to load
  ncclGinPluginStateLoadReady = 0,        // Plugin library is ready to be loaded
  ncclGinPluginStateInitReady = 1,        // Plugin library is loaded and ready to be initialized
  ncclGinPluginStateEnabled = 2,        // Plugin library is loaded and initialized
} ncclGinPluginState_t;

#define MAX_STR_LEN 255
typedef struct ginPluginLib {
  char name[MAX_STR_LEN];                       // Name of the plugin library
  void* dlHandle;                               // Handle to the plugin library
  ncclGin_t* ncclGin;                           // Pointer to the plugin structure
  int version;                                  // Version of the plugin
  ncclGinPluginState_t state;                   // State of the plugin
  int refCount;                                 // Reference count
  int physDevs;                                 // Number of physical devices
} ginPluginLib_t;

static int pluginCount = 0;
static ginPluginLib_t pluginLibs[NCCL_GIN_MAX_PLUGINS] = {0};
static std::mutex pluginMutex;
static std::once_flag initPluginLibsOnceFlag;

static ncclResult_t ncclGinPluginUnload(ginPluginLib_t* pluginLib) {
  if (pluginLib->dlHandle && pluginLib->refCount == 0) {
    INFO(NCCL_DESTROY | NCCL_NET, "GIN/Plugin: Unloading plugin %s", pluginLib->name);
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeGin));

    // Reset fields but preserve name, to be reused when reloading
    pluginLib->dlHandle = NULL;
    pluginLib->ncclGin = NULL;
    pluginLib->state = ncclGinPluginStateLoadReady;
    pluginLib->refCount = 0;
    pluginLib->physDevs = 0;
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinPluginLoad(ginPluginLib_t* pluginLib) {
  // 打开 GIN 插件 .so 动态库句柄
  // Entries borrowed from RMA/NET may already have a handle;
  // after unload, reopen the same library by its saved name/path.
  if (pluginLib->dlHandle == NULL) {
    pluginLib->dlHandle = ncclOpenGinPluginLib(pluginLib->name);
    if (pluginLib->dlHandle == nullptr) goto fail;
  }

  // 按照 ABI 版本从新到旧调用 getNcclGin_v** 获取 getNcclGin_t 符号对象
  // load gin
  for (int i = 0; i < NCCL_GIN_VERSION_COUNT; i++) {
    pluginLib->version = ncclGinVersion[i];
    pluginLib->ncclGin = getNcclGin[i](pluginLib->dlHandle);
    if (pluginLib->ncclGin) break;
  }

  if (pluginLib->ncclGin == nullptr) goto fail;

  pluginLib->state = ncclGinPluginStateInitReady;
  INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Successfully loaded external plugin %s",
       (ncclPluginLibPaths[ncclPluginTypeGin] ? ncclPluginLibPaths[ncclPluginTypeGin] : pluginLib->name));
exit:
  return ncclSuccess;
fail:
  INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Failed to load external plugin %s, dlHandle: %p, ncclGin: %p",
       (ncclPluginLibPaths[ncclPluginTypeGin] ? ncclPluginLibPaths[ncclPluginTypeGin] : pluginLib->name),
       pluginLib->dlHandle, pluginLib->ncclGin);
  if (pluginLib->dlHandle) {
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeGin));
  }
  pluginLib->dlHandle = nullptr;
  pluginLib->ncclGin = nullptr;
  pluginLib->version = 0;
  pluginLib->state = ncclGinPluginStateLoadFailed;
  goto exit;
}

static ncclResult_t ncclGinPluginInit(struct ncclComm* comm, ginPluginLib_t* pluginLib) {
  int ndev;
  bool ginInitCompleted = false;
  // 调用 GIN 插件内部的初始化获取 comm->ginContext 上下文
  // Init must be called for each new comm to set the right context
  if (pluginLib->state >= ncclGinPluginStateInitReady && pluginLib->ncclGin) {
    if (!pluginLib->ncclGin->init ||
        pluginLib->ncclGin->init(&comm->ginContext, comm->commHash, ncclDebugLog) != ncclSuccess) {
      pluginLib->state = ncclGinPluginStateDisabled;
    } else {
      ginInitCompleted = true;
    }
  }
  if (pluginLib->state == ncclGinPluginStateInitReady && pluginLib->ncclGin) {
    if (pluginLib->ncclGin->devices(&ndev) != ncclSuccess || ndev <= 0) {
      if (ginInitCompleted) {
        pluginLib->ncclGin->finalize(comm->ginContext);
        comm->ginContext = nullptr;
      }
      pluginLib->state = ncclGinPluginStateDisabled;
    } else {
      pluginLib->physDevs = ndev;
      pluginLib->state = ncclGinPluginStateEnabled;
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinPluginAssignToComm(struct ncclComm* comm, int pluginIndex, bool* isAssigned) {
  *isAssigned = false;

  if (pluginLibs[pluginIndex].state >= ncclGinPluginStateEnabled) {
    ncclGin_t* gin = pluginLibs[pluginIndex].ncclGin;
    ncclNetProperties_t props;
    // 获取第一个设备属性
    NCCLCHECK(gin->getProperties(0, &props));

    // 获取指定的 GIN 卸载类型
    int64_t ginType = ncclParamGinType();
    bool isExternal = pluginIndex < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS);

    // 设备支持的卸载类型与指定的 GIN 卸载类型不同
    if (ginType != -1 && props.netDeviceType != ginType) {
      INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Skipping plugin %s index %d type %d: NCCL_GIN_TYPE=%ld requested",
           gin->name, pluginIndex, props.netDeviceType, ginType);
      return ncclSuccess;
    }
    
    /*
    外部插件里，真硬件卸载的欢迎(不满足条件二，正常往下走绑定)；只是 proxy 的谢绝(命中本分支)，这个活儿交给我自带的 proxy-over-RMA。
    GIN 插件表尾部固定有两个内置插件(NCCL_GIN_NUM_INTERNAL_PLUGINS == 2)，由 initPluginLibsOnceFunc 注册：
      1. ncclGinIbGdaki —— 真正的 IB 硬件 GPU-initiated 卸载(GDAKI)；
      2. ncclGinProxy —— 内置的 GIN proxy 兜底，跑在 RMA 后端之上。
    */
    if (isExternal && props.netDeviceType == NCCL_NET_DEVICE_GIN_PROXY) {
      INFO(NCCL_INIT | NCCL_NET,
           "GIN/Plugin: Skipping external proxy plugin %s index %d; using NCCL GIN proxy over RMA backend %s",
           gin->name, pluginIndex,
           (comm->rmaState.rmaProxyState.ncclRma ? comm->rmaState.rmaProxyState.ncclRma->name : "none"));
      return ncclSuccess;
    }

    // 将 GIN 插件和相关信息保存到 sharedRes
    INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Assigned plugin %s type %d to comm", gin->name, props.netDeviceType);
    comm->sharedRes->ginState.ncclGin = gin;
    comm->sharedRes->ginState.ginVersion = pluginLibs[pluginIndex].version;
    /*
    把一个枚举裸转成另一个枚举，之所以合法，是因为两张枚举表被刻意设计成同值对齐：
    值  ncclNetDeviceType(net 侧)	    ncclGinType_t(GIN 侧)
    0   NCCL_NET_DEVICE_HOST(不卸载)	 NCCL_GIN_TYPE_NONE
    1	  NCCL_NET_DEVICE_UNPACK	      (故意空缺，注释 "intentionally not 1")
    2	  NCCL_NET_DEVICE_GIN_PROXY	    NCCL_GIN_TYPE_PROXY
    3	  NCCL_NET_DEVICE_GIN_GDAKI	    NCCL_GIN_TYPE_GDAKI
    4	  NCCL_NET_DEVICE_GIN_GPI	      NCCL_GIN_TYPE_GPI
    */
    // NOTE: The following cast is valid because ncclGinType_t variant values
    // should match NCCL_NET_DEVICE_GIN_* values from `enum ncclNetDeviceType`.
    comm->sharedRes->ginState.ginType = static_cast<ncclGinType_t>(props.netDeviceType);
    // GIN 插件索引单独保存到 comm
    comm->ginPluginIndex = pluginIndex;

    // 查询并缓存 GIN 能力标志
    ncclGinProperties_t ginProperties;
    NCCLCHECK(gin->getGinProperties(&ginProperties));
    // 是否支持“强信号”语义
    comm->sharedRes->ginState.supportsStrongSignals = ginProperties.supportsStrongSignals;
    // 是否支持基于虚拟地址(VA)的信号
    comm->sharedRes->ginState.supportsVASignals = ginProperties.supportsVASignals;
  }
  pluginLibs[pluginIndex].refCount++;
  *isAssigned = true;
  return ncclSuccess;
}

static ncclResult_t ncclGinPluginDisableOtherExternal(int pluginIndex) {
  // Only if an external plugin is enabled, disable other external plugins
  if (pluginIndex >= (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS)) return ncclSuccess;
  char names[MAX_STR_LEN * (NCCL_GIN_MAX_PLUGINS - NCCL_GIN_NUM_INTERNAL_PLUGINS)] = {0};
  for (int i = 0; i < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS); i++) {
    if (i != pluginIndex && pluginLibs[i].state >= ncclGinPluginStateEnabled) {
      // Append all disabled plugin names to a string
      snprintf(names + strlen(names), sizeof(names) - strlen(names), (strlen(names) == 0) ? "%s" : ", %s",
               pluginLibs[i].name);
      pluginLibs[i].state = ncclGinPluginStateDisabled;
    }
  }
  if (strlen(names) > 0) {
    INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Disabling external plugins: %s", names);
  }
  return ncclSuccess;
}

static void initPluginLibsOnceFunc() {
  char* ginPluginName = nullptr;
  const char* defaultGinPlugin = "libnccl-gin.so";
  const char* envGinPlugin = nullptr;
  char* envGinPluginList = nullptr;
  char* savePtr = nullptr;
  int pluginCounter = 0;

  memset(pluginLibs, 0, NCCL_GIN_MAX_PLUGINS * sizeof(ginPluginLib_t));
  envGinPlugin = ncclGetEnv("NCCL_GIN_PLUGIN");
  if (envGinPlugin) {
    INFO(NCCL_ENV | NCCL_NET, "NCCL_GIN_PLUGIN set by environment to %s", envGinPlugin);
    if (strcasecmp(envGinPlugin, "none") == 0) envGinPlugin = "";
    envGinPluginList = strdup(envGinPlugin);
    // Iterate over list until the list is empty
    ginPluginName = strtok_r(envGinPluginList, ",", &savePtr);
    while (ginPluginName) {
      // So, we can have at most( NCCL_GIN_MAX_PLUGINS - (NCCL_GIN_NUM_RESERVED_PLUGINS)) in the NCCL_GIN_PLUGIN list

      if (pluginCounter >= (NCCL_GIN_MAX_PLUGINS - NCCL_GIN_NUM_RESERVED_PLUGINS)) {
        INFO(NCCL_NET | NCCL_ENV, "NCCL_GIN_PLUGIN list contains more than %d plugins, ignoring the rest",
             (NCCL_GIN_MAX_PLUGINS - NCCL_GIN_NUM_RESERVED_PLUGINS));
        break;
      }
      // need to leave space for the name + "\n"
      if ((strlen(ginPluginName) + 1) <= MAX_STR_LEN) {
        pluginLibs[pluginCounter].state = ncclGinPluginStateLoadReady;
        pluginLibs[pluginCounter].refCount = ncclParamGinPluginRefCount();
        strcpy(pluginLibs[pluginCounter].name, ginPluginName);
        pluginCounter++;
      } else {
        INFO(NCCL_NET | NCCL_ENV,
             "NCCL_GIN_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", ginPluginName,
             MAX_STR_LEN);
      }
      ginPluginName = strtok_r(nullptr, ",", &savePtr);
    }
    if (envGinPluginList) free(envGinPluginList);
  } else {
    // Add default gin plugin
    pluginLibs[pluginCounter].state = ncclGinPluginStateLoadReady;
    pluginLibs[pluginCounter].refCount = ncclParamGinPluginRefCount();
    strcpy(pluginLibs[pluginCounter++].name, defaultGinPlugin);
  }

  // NET 插件加载了拉进 GIN 插件候选池，NET 的库被 GIN 借用当后端候选
  // Also check if the NET plugin has GIN support
  if ((pluginLibs[pluginCounter].dlHandle = ncclGetNetPluginLib(ncclPluginTypeGin)) != NULL) {
    pluginLibs[pluginCounter].state = ncclGinPluginStateLoadReady;
    strcpy(pluginLibs[pluginCounter++].name, ncclGetPluginLibName(ncclPluginTypeGin));
  }

  // 添加内部 IB GDAKI 插件
  // Add internal ib plugin
  pluginLibs[pluginCounter].ncclGin = &ncclGinIbGdaki;
  pluginLibs[pluginCounter].state = ncclGinPluginStateInitReady;
  pluginLibs[pluginCounter].version = ncclGinVersion[0];
  pluginCounter++;
  // 添加 ncclGinProxy 插件
  // Add gin proxy as fallback
  pluginLibs[pluginCounter].ncclGin = &ncclGinProxy;
  pluginLibs[pluginCounter].state = ncclGinPluginStateInitReady;
  pluginLibs[pluginCounter].version = ncclGinProxyVersion;
  pluginCounter++;
  pluginCount = pluginCounter;
}

static ncclResult_t ncclGinPluginFinalize(struct ncclComm* comm, int pluginIndex) {
  if (pluginLibs[pluginIndex].ncclGin && pluginLibs[pluginIndex].state == ncclGinPluginStateEnabled) {
    NCCLCHECK(pluginLibs[pluginIndex].ncclGin->finalize(comm->ginContext));
  }
  pluginLibs[pluginIndex].refCount--;
  if (pluginIndex < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS)) {
    NCCLCHECK(ncclGinPluginUnload(&pluginLibs[pluginIndex]));
  }
  return ncclSuccess;
}

ncclResult_t ncclGinInit(struct ncclComm* comm) {
  // GIN 仅支持 CUDA CC 版本大于等于 7.0
  if (comm->compCap < 70) {
    /* GIN only supported for Volta and later */
    INFO(NCCL_INIT, "Compute Capability (%d) is not sufficient to enable GIN.  Require Volta (70) or newer.",
         comm->compCap);
    return ncclSuccess;
  }

  bool initialized = false;
  // 解析 GIN 插件名并初始化状态
  std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);
  // 加锁遍历所有 GIN 插件
  std::lock_guard<std::mutex> lock(pluginMutex);
  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
    if (pluginIndex < (pluginCount - NCCL_GIN_NUM_INTERNAL_PLUGINS) &&
        pluginLibs[pluginIndex].state == ncclGinPluginStateLoadReady) {
      // 打开 GIN 插件的 .so 动态库 handle，并获取 getNcclGin_t 符号对象
      NCCLCHECK(ncclGinPluginLoad(&pluginLibs[pluginIndex]));
    }
    if (pluginLibs[pluginIndex].state >= ncclGinPluginStateInitReady) {
      // 初始化获取 GIN 上下文，保存到 comm
      // plugin init must be done by all comms to setup the context, therefore we use ">="
      NCCLCHECK(ncclGinPluginInit(comm, &pluginLibs[pluginIndex]));
      if (pluginLibs[pluginIndex].state == ncclGinPluginStateEnabled) {
        bool isAssigned = false;
        // 校验卸载类型，将 GIN 相关属性信息保存到 comm->sharedRes->ginState
        NCCLCHECK(ncclGinPluginAssignToComm(comm, pluginIndex, &isAssigned));
        if (isAssigned) {
          // If one external plugin is assigned to a comm, then disable all other external plugins
          ncclGinPluginDisableOtherExternal(pluginIndex);
          initialized = true;
          break;
        } else {
          ncclGinPluginFinalize(comm, pluginIndex);
        }
      }
    }
  }
  if (!initialized) INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Failed to initialize any GIN plugin");
  return ncclSuccess;
}

ncclResult_t ncclGinInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  comm->ginContext = parent->ginContext;
  comm->ginPluginIndex = parent->ginPluginIndex;
  return ncclSuccess;
}

ncclResult_t ncclGinFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->ginPluginIndex;
  std::lock_guard<std::mutex> lock(pluginMutex);
  NCCLCHECK(ncclGinPluginFinalize(comm, pluginIndex));
  return ncclSuccess;
}

ncclResult_t ncclGinGetDevCount(int pluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (pluginLibs[pluginIndex].state != ncclGinPluginStateEnabled || pluginLibs[pluginIndex].physDevs == 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = pluginLibs[pluginIndex].physDevs;
  return ncclSuccess;
fail:
  WARN("trying to access the number of devices of an uninitialized ginPlugin[%d]", pluginIndex);
  return ncclInternalError;
}
