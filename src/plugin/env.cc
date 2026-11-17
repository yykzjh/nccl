/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <errno.h>
#include <stdlib.h>
#include <mutex>
#include <atomic>

#include "checks.h"
#include "debug.h"
#include "env.h"
#include "param.h"
#include "plugin.h"

extern ncclEnv_t* getNcclEnv_v2(void* lib);
extern ncclEnv_t* getNcclEnv_v1(void* lib);

static void* envPluginLib = nullptr;
static ncclEnv_t* ncclEnvPlugin = nullptr;
extern ncclEnv_v2_t ncclIntEnv_v2;

#define EXT_ENV_PLUGIN 0
#define INT_ENV_PLUGIN 1
#define NUM_ENV_PLUGIN 2
static ncclEnv_t* ncclEnvPlugins[NUM_ENV_PLUGIN] = {nullptr, &ncclIntEnv_v2};

enum {
  envPluginLoadFailed = -1,
  envPluginLoadReady = 0,
  envPluginLoadSuccess = 1,
};
static int envPluginStatus = envPluginLoadReady;

static ncclResult_t ncclEnvPluginLoad(void) {
  const char* envName = nullptr;
  bool envPluginRequested = false;
  if (envPluginStatus != envPluginLoadReady) goto exit;

  // 从环境变量 NCCL_ENV_PLUGIN 获取插件名称，例如 NCCL_ENV_PLUGIN=gcp，可能是从之前的 NCCL 配置文件导出的该环境变量
  if ((envName = std::getenv("NCCL_ENV_PLUGIN")) != nullptr) {
    envPluginRequested = true;
    INFO(NCCL_ENV, "NCCL_ENV_PLUGIN set by environment to %s", envName);
    // 如果显式设置为 none，则不加载插件
    if (strcasecmp(envName, "none") == 0) { // 不区分大小写比较字符串是否相等
      goto fail;
    }
  }
  // 打开环境插件动态库句柄
  envPluginLib = ncclOpenEnvPluginLib(envName);
  if (nullptr == envPluginLib) {
    goto fail;
  } else if (ncclPluginLibPaths[ncclPluginTypeEnv]) {
    envName = ncclPluginLibPaths[ncclPluginTypeEnv];
  }

  // 查找 v2 版本的插件符号 ncclEnvPlugin_v2，类型为 ncclEnv_t*
  ncclEnvPlugins[EXT_ENV_PLUGIN] = getNcclEnv_v2(envPluginLib);
  if (ncclEnvPlugins[EXT_ENV_PLUGIN] == nullptr) {
    // fallback to v1
    ncclEnvPlugins[EXT_ENV_PLUGIN] = getNcclEnv_v1(envPluginLib);
    if (ncclEnvPlugins[EXT_ENV_PLUGIN] == nullptr) {
      if (envPluginRequested) ATTN("External env plugin %s is unsupported", envName);
      else INFO(NCCL_INIT, "External env plugin %s is unsupported", envName);
      goto fail;
    }
  }
  INFO(NCCL_INIT, "Successfully loaded external env plugin %s", envName);

  envPluginStatus = envPluginLoadSuccess;

exit:
  return ncclSuccess;
fail:
  // Fallback to internal/default plugin
  if (envPluginLib) NCCLCHECK(ncclClosePluginLib(envPluginLib, ncclPluginTypeEnv));
  envPluginLib = nullptr;
  envPluginStatus = envPluginLoadFailed;
  goto exit;
}

static ncclResult_t ncclEnvPluginUnload(void) {
  if (ncclEnvPlugin) {
    INFO(NCCL_DESTROY, "ENV/Plugin: Closing env plugin %s", ncclEnvPlugin->name);
  }
  if (ncclEnvPlugins[EXT_ENV_PLUGIN]) {
    ncclEnvPlugin = ncclEnvPlugins[INT_ENV_PLUGIN];
    ncclEnvPlugins[EXT_ENV_PLUGIN] = nullptr;
  }
  // 关闭环境插件动态库句柄
  NCCLCHECK(ncclClosePluginLib(envPluginLib, ncclPluginTypeEnv));
  return ncclSuccess;
}

void ncclEnvPluginFinalize(void);

static bool initialized;

ncclResult_t ncclEnvPluginInit(void) {
  // 先尝试从 NCCL 配置文件解析环境变量到当前进程环境
  initEnv();
  // 加载环境插件，即 ncclEnv_t* 类型结构体对象
  NCCLCHECK(ncclEnvPluginLoad());
  ncclEnvPlugin =
    (envPluginLoadSuccess == envPluginStatus) ? ncclEnvPlugins[EXT_ENV_PLUGIN] : ncclEnvPlugins[INT_ENV_PLUGIN];
  // 环境插件初始化
  NCCLCHECK(ncclEnvPlugin->init(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_SUFFIX, ncclDebugLog));
  // 注册进程退出时的清理函数
  atexit(ncclEnvPluginFinalize);
  // 以 release 语义把初始化完成标志发布出去
  COMPILER_ATOMIC_STORE(&initialized, true, std::memory_order_release);
  return ncclSuccess;
}

void ncclEnvPluginFinalize(void) {
  if (ncclEnvPlugin->finalize) {
    // 调用插件自己的 finalize()
    ncclEnvPlugin->finalize();
    // 卸载插件动态库
    ncclEnvPluginUnload();
  }
}

const char* ncclEnvPluginGetEnv(const char* name) {
  return ncclEnvPlugin->getEnv(name);
}

bool ncclEnvPluginInitialized(void) {
  return COMPILER_ATOMIC_LOAD(&initialized, std::memory_order_acquire);
}
