/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "os.h"

#include "debug.h"
#include "plugin.h"

#define MAX_STR_LEN 255

#define NUM_LIBS 6
static char* libNames[NUM_LIBS];
char* ncclPluginLibPaths[NUM_LIBS];
static void* libHandles[NUM_LIBS];
static const char* pluginNames[NUM_LIBS] = {"NET", "GIN", "RMA", "TUNER", "PROFILER", "ENV"};
static const char* pluginPrefix[NUM_LIBS] = {"libnccl-net",   "libnccl-gin",      "libnccl-rma",
                                             "libnccl-tuner", "libnccl-profiler", "libnccl-env"};
static const char* pluginFallback[NUM_LIBS] = {"", "", "", "", "", ""};
static unsigned long subsys[NUM_LIBS] = {
  NCCL_INIT | NCCL_NET, NCCL_INIT | NCCL_NET, NCCL_INIT | NCCL_NET, NCCL_INIT | NCCL_TUNING, NCCL_INIT,
  NCCL_INIT | NCCL_ENV
};

static void* tryOpenLib(char* name, int* err, char* errStr) {
  *err = 0;
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;
  }

  // NCCL 打开动态库句柄
  void* handle = ncclOsDlopen(name);
  if (nullptr == handle) {
    const char* dlErr = ncclOsDlerror();
    if (dlErr) {
      strncpy(errStr, dlErr, MAX_STR_LEN);
      errStr[MAX_STR_LEN] = '\0';
    } else {
      errStr[0] = '\0';
    }
    // 如果 name 不为空，并且 errStr 中包含 name 以及 "No such file or directory"，则设置 err 为 ENOENT
    // "handle" and "name" won't be NULL at the same time.
    // coverity[var_deref_model]
    if (name && strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }
  return handle;
}

static void appendNameToList(char* nameList, int* leftChars, char* name) {
  snprintf(nameList + PATH_MAX - *leftChars, *leftChars, " %s", name);
  *leftChars -= strlen(name) + 1;
}

#if defined(NCCL_OS_LINUX)
static char* getLibPath(void* handle) {
  // link_map 是 Linux 动态链接器用于描述已加载共享对象的数据结构，其中 lm->l_name 表示共享对象的文件名或加载路径。
  struct link_map* lm;
  // 使用 GNU/Linux 动态链接器接口查询该句柄对应的 link_map
  if (dlinfo(handle, RTLD_DI_LINKMAP, &lm) != 0) return nullptr;
  else return strdup(lm->l_name);
}
#elif defined(NCCL_OS_WINDOWS)
static char* getLibPath(void* handle) {
  char path[PATH_MAX];
  DWORD len = GetModuleFileNameA((HMODULE)handle, path, PATH_MAX);
  if (len == 0 || len >= PATH_MAX) return nullptr;
  return strdup(path);
}
#endif

static void* openPluginLib(enum ncclPluginType type, const char* libName) {
  int openErr, len = PATH_MAX;
  char libName_[MAX_STR_LEN] = {0};
  char openErrStr[MAX_STR_LEN + 1] = {0};
  char eNoEntNameList[PATH_MAX] = {0};

  // 首先按照 libName 作为动态库名称进行尝试，例如 libName = "/opt/plugins/libcustom.so"
  if (libName && strlen(libName)) {
    snprintf(libName_, MAX_STR_LEN, "%s", libName);
  } else {
    snprintf(libName_, MAX_STR_LEN, "%s.so", pluginPrefix[type]);
  }

  // 尝试打开动态库
  // 如果报错，libName_ 为 nullptr，openErr 为错误码，openErrStr 为错误信息
  // 保存该类型插件，只有一个同类型插件动态库句柄
  libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
  // 加载成功保存动态库 handle、名称和路径
  if (libHandles[type]) {
    libNames[type] = strdup(libName_);
    // 通过 dlinfo 从句柄获取动态库的路径 link_map->l_name
    ncclPluginLibPaths[type] = getLibPath(libHandles[type]);
    return libHandles[type];
  }
  // 找不到动态库
  if (openErr == ENOENT) {
    // 记录所有找不到的动态库名称
    appendNameToList(eNoEntNameList, &len, libName_);
  } else {  // 其他错误
    INFO(subsys[type], "%s/Plugin: %s: %s", pluginNames[type], libName_, openErrStr);
  }

  // 检测第二种指定动态库名称的方式，不能为空、不能包含 /、不能以 lib 开头、不能以 .so 结尾
  // libName can't be a relative or absolute path (start with '.' or contain any '/'). It can't be a
  // library name either (start with 'lib' or end with '.so')
  if (libName && strlen(libName) && strchr(libName, '/') == nullptr &&
      (strncmp(libName, "lib", strlen("lib")) || strlen(libName) < strlen(".so") ||
       strncmp(libName + strlen(libName) - strlen(".so"), ".so", strlen(".so")))) {
    // 构造动态库名称，例如 libName = gcp、type = ncclPluginTypeNet，则 libName_ = libnccl-net-gcp.so
    snprintf(libName_, MAX_STR_LEN, "%s-%s.so", pluginPrefix[type], libName);

    libHandles[type] = tryOpenLib(libName_, &openErr, openErrStr);
    if (libHandles[type]) {
      libNames[type] = strdup(libName_);
      ncclPluginLibPaths[type] = getLibPath(libHandles[type]);
      return libHandles[type];
    }
    if (openErr == ENOENT) {
      appendNameToList(eNoEntNameList, &len, libName_);
    } else {
      INFO(subsys[type], "%s/Plugin: %s: %s", pluginNames[type], libName_, openErrStr);
    }
  }

  // 统一输出所有未找到的名称
  if (strlen(eNoEntNameList)) {
    INFO(subsys[type], "%s/Plugin: Could not find:%s%s%s", pluginNames[type], eNoEntNameList,
         (strlen(pluginFallback[type]) > 0 ? ". " : ""), pluginFallback[type]);
  } else if (strlen(pluginFallback[type])) {
    INFO(subsys[type], "%s/Plugin: %s", pluginNames[type], pluginFallback[type]);
  }
  return nullptr;
}

void* ncclOpenNetPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeNet, name);
}

void* ncclOpenGinPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeGin, name);
}

void* ncclOpenRmaPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeRma, name);
}

void* ncclOpenTunerPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeTuner, name);
}

void* ncclOpenProfilerPluginLib(const char* name) {
  return openPluginLib(ncclPluginTypeProfiler, name);
}

// 打开环境插件库
void* ncclOpenEnvPluginLib(const char* name) {
  // 通用的打开插件库
  return openPluginLib(ncclPluginTypeEnv, name);
}

void* ncclGetGinPluginLib(enum ncclPluginType type) {
  // 探测：如果之前 NET/GIN 子系统已经加载过一个 GIN 插件(libNames[Gin] 非空)，说明磁盘上有个能用的多面手 .so
  if (libNames[ncclPluginTypeGin]) {
    /*
    1. 把 GIN 的 .so 名字/路径 strdup 一份填进 Rma 槽。
    2. 再 dlopen 一次，对同一个 .so 文件再次 dlopen，操作系统加载器不会把它在内存里加载第二份，只是把该库的引用计数 +1，
    返回同一个底层映像的句柄，所以 RMA 和 GIN 共享同一份代码/数据段，代价极低。
    */
    // increment the reference counter of the gin library
    libNames[type] = strdup(libNames[ncclPluginTypeGin]);
    ncclPluginLibPaths[type] = strdup(ncclPluginLibPaths[ncclPluginTypeGin]);
    libHandles[type] = ncclOsDlopen(libNames[ncclPluginTypeGin]);
  }
  return libHandles[type];
}

void* ncclGetNetPluginLib(enum ncclPluginType type) {
  if (libNames[ncclPluginTypeNet]) {
    // increment the reference counter of the net library
    libNames[type] = strdup(libNames[ncclPluginTypeNet]);
    ncclPluginLibPaths[type] = strdup(ncclPluginLibPaths[ncclPluginTypeNet]);
    libHandles[type] = ncclOsDlopen(libNames[ncclPluginTypeNet]);
  }
  return libHandles[type];
}

ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type) {
  if (handle && libHandles[type] == handle) {
    ncclOsDlclose(handle);  // 关闭动态库句柄
    libHandles[type] = nullptr;
    free(ncclPluginLibPaths[type]);  // 释放动态库路径字符串
    ncclPluginLibPaths[type] = nullptr;
    free(libNames[type]);  // 释放动态库名称字符串
    libNames[type] = nullptr;
  }
  return ncclSuccess;
}

const char* ncclGetPluginLibName(enum ncclPluginType type) {
  return libNames[type];
}