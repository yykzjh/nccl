/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nvmlwrap.h"
#include "checks.h"
#include "debug.h"
#include "os.h"
#include "param.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <mutex>

int ncclNvmlDeviceCount = 0;
ncclNvmlDeviceInfo ncclNvmlDevices[ncclNvmlMaxDevices];
ncclNvmlDevicePairInfo ncclNvmlDevicePairs[ncclNvmlMaxDevices][ncclNvmlMaxDevices];

#if NCCL_NVML_DIRECT
// 如果直接链接 nvml.h 头文件，则创建新的 pfn_<symbol_name> 符号直接静态指向 symbol 函数 
#define NCCL_NVML_FN(name, rettype, arglist) constexpr rettype(*pfn_##name) arglist = name;
#else
// 如果动态加载 nvml 动态库中的符号，则创建新的 pfn_<symbol_name> 符号置为 nullptr
#define NCCL_NVML_FN(name, rettype, arglist) rettype(*pfn_##name) arglist = nullptr;
#endif

namespace {
NCCL_NVML_FN(nvmlInit, nvmlReturn_t, ())
NCCL_NVML_FN(nvmlInit_v2, nvmlReturn_t, ())
NCCL_NVML_FN(nvmlShutdown, nvmlReturn_t, ())
NCCL_NVML_FN(nvmlDeviceGetCount, nvmlReturn_t, (unsigned int*))
NCCL_NVML_FN(nvmlDeviceGetCount_v2, nvmlReturn_t, (unsigned int*))
NCCL_NVML_FN(nvmlDeviceGetHandleByPciBusId, nvmlReturn_t, (const char* pciBusId, nvmlDevice_t* device))
NCCL_NVML_FN(nvmlDeviceGetHandleByIndex, nvmlReturn_t, (unsigned int index, nvmlDevice_t* device))
NCCL_NVML_FN(nvmlDeviceGetIndex, nvmlReturn_t, (nvmlDevice_t device, unsigned* index))
NCCL_NVML_FN(nvmlDeviceGetName, nvmlReturn_t, (nvmlDevice_t device, char* name, unsigned int length))
NCCL_NVML_FN(nvmlSystemGetDriverVersion, nvmlReturn_t, (char* version, unsigned int length))
NCCL_NVML_FN(nvmlDeviceGetMemoryErrorCounter, nvmlReturn_t,
             (nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType,
              nvmlMemoryLocation_t locationType, unsigned long long* count))
NCCL_NVML_FN(nvmlErrorString, char const*, (nvmlReturn_t r))
NCCL_NVML_FN(nvmlDeviceGetNvLinkState, nvmlReturn_t,
             (nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive))
NCCL_NVML_FN(nvmlDeviceGetNvLinkRemotePciInfo, nvmlReturn_t,
             (nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci))
NCCL_NVML_FN(nvmlDeviceGetNvLinkRemoteDeviceType, nvmlReturn_t,
             (nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* NvLinkDeviceType))
NCCL_NVML_FN(nvmlDeviceGetNvLinkCapability, nvmlReturn_t,
             (nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult))
NCCL_NVML_FN(nvmlDeviceGetCudaComputeCapability, nvmlReturn_t, (nvmlDevice_t device, int* major, int* minor))
NCCL_NVML_FN(nvmlDeviceGetP2PStatus, nvmlReturn_t,
             (nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex,
              nvmlGpuP2PStatus_t* p2pStatus))
NCCL_NVML_FN(nvmlDeviceGetFieldValues, nvmlReturn_t, (nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values))
NCCL_NVML_FN(nvmlDeviceGetPciInfo_v3, nvmlReturn_t, (nvmlDevice_t device, nvmlPciInfo_t* pciInfo))
NCCL_NVML_FN(nvmlDeviceGetPciInfoExt, nvmlReturn_t, (nvmlDevice_t device, nvmlPciInfoExt_t* pciInfoExt))
NCCL_NVML_FN(nvmlDeviceGetPcieLinkMaxSpeed, nvmlReturn_t, (nvmlDevice_t device, int* maxSpeed))
NCCL_NVML_FN(nvmlDeviceGetCurrPcieLinkGeneration, nvmlReturn_t, (nvmlDevice_t device, unsigned int* currLinkGen))
NCCL_NVML_FN(nvmlDeviceGetCurrPcieLinkWidth, nvmlReturn_t, (nvmlDevice_t device, unsigned int* currLinkWidth))
// MNNVL support
NCCL_NVML_FN(nvmlDeviceGetGpuFabricInfoV, nvmlReturn_t, (nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo))
NCCL_NVML_FN(nvmlDeviceGetPlatformInfo, nvmlReturn_t, (nvmlDevice_t device, nvmlPlatformInfo_t* platfromInfo))
// CC support
NCCL_NVML_FN(nvmlSystemGetConfComputeState, nvmlReturn_t, (nvmlConfComputeSystemState_t* state));
NCCL_NVML_FN(nvmlSystemGetConfComputeSettings, nvmlReturn_t, (nvmlSystemConfComputeSettings_t* setting));

std::mutex lock; // NVML has had some thread safety bugs
bool initialized = false;
thread_local bool threadInitialized = false;
ncclResult_t initResult;

union nvmlCCInfoInternal {
  nvmlConfComputeSystemState_t settingV12020;
  nvmlSystemConfComputeSettings_t settingV12040;
};

void ncclNvmlParseCudaVisibleDevices(const char* env, int deviceCount, bool* visible) {
  for (int a = 0; a < deviceCount; a++) visible[a] = (env == nullptr);
  if (env == nullptr || env[0] == '\0') return;

  const char* p = env;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t') p++;

    char* end = nullptr;
    errno = 0;
    long value = strtol(p, &end, 10);
    if (p == end) {
      // CUDA accepts UUID and MIG identifiers, which cannot be mapped to NVML
      // ordinals here. Avoid treating a potentially visible device as invisible.
      for (int a = 0; a < deviceCount; a++) visible[a] = true;
      return;
    }
    if (errno != 0 || value > INT_MAX) return;
    if (value < 0) return;

    while (*end == ' ' || *end == '\t') end++;
    if (*end != '\0' && *end != ',') return;

    // CUDA stops processing the sequence at the first invalid index.
    if (value >= deviceCount) return;
    visible[value] = true;
    if (*end == '\0') break;
    p = end + 1;
  }
}

void ncclNvmlCacheCudaVisibleDevices() {
  bool visible[ncclNvmlMaxDevices];
  ncclNvmlParseCudaVisibleDevices(ncclGetEnv("CUDA_VISIBLE_DEVICES"), ncclNvmlDeviceCount, visible);
  for (int a = 0; a < ncclNvmlDeviceCount; a++) {
    ncclNvmlDevices[a].cudaVisible = visible[a];
  }
}
} // namespace

/*
ncclNvmlEnsureInitialized 承担三件事，且保证全进程只做一次：
  动态加载 NVML 库，把二十多个符号解析成函数指针（弱依赖，缺失不崩）。
  初始化 NVML 并枚举整机所有 GPU，缓存句柄和 compute capability 到 ncclNvmlDevices[]。
  预查所有 GPU 两两之间的 P2P 状态，缓存成 N×N 矩阵 ncclNvmlDevicePairs[][]。
*/
ncclResult_t ncclNvmlEnsureInitialized() {
  // 由于 ncclNvmlEnsureInitialized 被每个 NVML 包装函数在每次调用时都执行一遍，线程级无锁快速路径
  // Optimization to avoid repeatedly grabbing the lock when we only want to
  // read from the global tables.
  if (threadInitialized) return initResult;
  threadInitialized = true;

  std::lock_guard<std::mutex> locked(lock);  // 加锁

  // 进程级一次性判断
  if (initialized) return initResult;
  initialized = true;

/*
header 里默认 NCCL_NVML_DIRECT 0，而这个宏决定了整套机制的形态：

默认走动态加载，好处是 libnccl.so 不依赖 NVML 库存在——在没装完整驱动工具链的容器里也能加载，
只是 NVML 相关功能降级。这就是"弱依赖"。注意 header 里那大段 End of nvml.h 之前的内容，
就是为了在不 include nvml.h 的情况下自备类型定义。
*/
#if !NCCL_NVML_DIRECT
  /*
  NCCL_NVML_DIRECT=1 时，pfn_nvmlInit 是 constexpr 常量指针，编译期直接指向真函数，硬链接 libnvidia-ml，
  header 里的 nvml 声明	#include "nvml.h"；
  NCCL_NVML_DIRECT=0（默认）时，pfn_nvmlInit 运行时 nullptr，靠 dlsym 填，无链接依赖，
  header 自带一份 nvml 类型/常量的复刻声明。
  */
  if (pfn_nvmlInit == nullptr) {
    ncclOsLibraryHandle libhandle;
    // 打开 NVML 动态库
    ncclResult_t openRes = ncclOsNvmlOpen(&libhandle);
    if (openRes != ncclSuccess) {
      initResult = openRes;
      return initResult;
    }

    struct Symbol {
      void** ppfn;
      char const* name;
    };
    std::initializer_list<Symbol> symbols = {
      {(void**)&pfn_nvmlInit, "nvmlInit"},
      {(void**)&pfn_nvmlInit_v2, "nvmlInit_v2"},
      {(void**)&pfn_nvmlShutdown, "nvmlShutdown"},
      {(void**)&pfn_nvmlDeviceGetCount, "nvmlDeviceGetCount"},
      {(void**)&pfn_nvmlDeviceGetCount_v2, "nvmlDeviceGetCount_v2"},
      {(void**)&pfn_nvmlDeviceGetHandleByPciBusId, "nvmlDeviceGetHandleByPciBusId"},
      {(void**)&pfn_nvmlDeviceGetHandleByIndex, "nvmlDeviceGetHandleByIndex"},
      {(void**)&pfn_nvmlDeviceGetIndex, "nvmlDeviceGetIndex"},
      {(void**)&pfn_nvmlDeviceGetName, "nvmlDeviceGetName"},
      {(void**)&pfn_nvmlSystemGetDriverVersion, "nvmlSystemGetDriverVersion"},
      {(void**)&pfn_nvmlDeviceGetMemoryErrorCounter, "nvmlDeviceGetMemoryErrorCounter"},
      {(void**)&pfn_nvmlErrorString, "nvmlErrorString"},
      {(void**)&pfn_nvmlDeviceGetNvLinkState, "nvmlDeviceGetNvLinkState"},
      {(void**)&pfn_nvmlDeviceGetNvLinkRemotePciInfo, "nvmlDeviceGetNvLinkRemotePciInfo"},
      {(void**)&pfn_nvmlDeviceGetNvLinkRemoteDeviceType, "nvmlDeviceGetNvLinkRemoteDeviceType"},
      {(void**)&pfn_nvmlDeviceGetNvLinkCapability, "nvmlDeviceGetNvLinkCapability"},
      {(void**)&pfn_nvmlDeviceGetCudaComputeCapability, "nvmlDeviceGetCudaComputeCapability"},
      {(void**)&pfn_nvmlDeviceGetP2PStatus, "nvmlDeviceGetP2PStatus"},
      {(void**)&pfn_nvmlDeviceGetFieldValues, "nvmlDeviceGetFieldValues"},
      {(void**)&pfn_nvmlDeviceGetPciInfo_v3, "nvmlDeviceGetPciInfo_v3"},
      {(void**)&pfn_nvmlDeviceGetPciInfoExt, "nvmlDeviceGetPciInfoExt"},
      {(void**)&pfn_nvmlDeviceGetPcieLinkMaxSpeed, "nvmlDeviceGetPcieLinkMaxSpeed"},
      {(void**)&pfn_nvmlDeviceGetCurrPcieLinkGeneration, "nvmlDeviceGetCurrPcieLinkGeneration"},
      {(void**)&pfn_nvmlDeviceGetCurrPcieLinkWidth, "nvmlDeviceGetCurrPcieLinkWidth"},
      // MNNVL support
      {(void**)&pfn_nvmlDeviceGetGpuFabricInfoV, "nvmlDeviceGetGpuFabricInfoV"},
      {(void**)&pfn_nvmlDeviceGetPlatformInfo, "nvmlDeviceGetPlatformInfo"},
      // CC support
      {(void**)&pfn_nvmlSystemGetConfComputeState, "nvmlSystemGetConfComputeState"},
      {(void**)&pfn_nvmlSystemGetConfComputeSettings, "nvmlSystemGetConfComputeSettings"}
    };
    for (Symbol sym : symbols) {
      *sym.ppfn = ncclOsDlsym(libhandle, sym.name);  // 失败就留 nullptr，不报错
    }

    /*
    libhandle 是局部变量，出了作用域就丢了，永远不会 dlclose。这是刻意的——那二十多个 pfn_* 函数指针指向的代码必须一直有效，
    一旦 dlclose 全部变野指针。末尾那行 coverity[leaked_storage] 是静态分析工具的抑制注解，等于在说"我知道，别报了"。
    这和之前看过的 NCCL_NVML_FN 之外的另一处，网络插件用 NCCL_NET_PLUGIN_REF_COUNT 预置引用计数永久阻止 dlclose，
    是同一个思路的两种实现。
    */
    // Coverity complains that we never dlclose this object, but that's
    // deliberate, since we want the loaded object to remain in memory until
    // the process terminates, so that we can use its code.
    // coverity[leaked_storage]
  }
#endif

// 同一个功能加载了两个版本的符号，有 _v2 就用 _v2，没有就退回老版。
#if NCCL_NVML_DIRECT
  // 直接链接模式下 pfn 是 constexpr 非空常量，写 != nullptr 会被 GCC 警告"永远为真"，所以硬编码 true 绕开
  bool have_v2 = true;
#else
  // if this compare is done in the NCCL_NVML_DIRECT=1 case then GCC warns about it never being null
  bool have_v2 = pfn_nvmlInit_v2 != nullptr;
#endif
  // 初始化 NVML
  nvmlReturn_t res1 = (have_v2 ? pfn_nvmlInit_v2 : pfn_nvmlInit)();
  if (res1 != NVML_SUCCESS) {
    WARN("nvmlInit%s() failed: %s", have_v2 ? "_v2" : "", pfn_nvmlErrorString(res1));
    initResult = ncclSystemError;
    return initResult;
  }

  unsigned int ndev;
  // 获取整机 GPU 数
  res1 = (have_v2 ? pfn_nvmlDeviceGetCount_v2 : pfn_nvmlDeviceGetCount)(&ndev);
  if (res1 != NVML_SUCCESS) {
    WARN("nvmlDeviceGetCount%s() failed: %s", have_v2 ? "_v2" : "", pfn_nvmlErrorString(res1));
    initResult = ncclSystemError;
    return initResult;
  }

  ncclNvmlDeviceCount = int(ndev);
  // 整机 GPU 数不可能大于 32，静态分配换取零动态内存管理
  if (ncclNvmlMaxDevices < ncclNvmlDeviceCount) {
    WARN("nvmlDeviceGetCount() reported more devices (%d) than the internal maximum (ncclNvmlMaxDevices=%d)",
         ncclNvmlDeviceCount, ncclNvmlMaxDevices);
    initResult = ncclInternalError;
    return initResult;
  }

  ncclNvmlCacheCudaVisibleDevices();

  for (int a = 0; a < ncclNvmlDeviceCount; a++) {
    // 获取指定索引的 NVML 设备句柄
    res1 = pfn_nvmlDeviceGetHandleByIndex(a, &ncclNvmlDevices[a].handle);
    if (res1 != NVML_SUCCESS) {
      if (!ncclNvmlDevices[a].cudaVisible) {
        ncclNvmlDevices[a].handle = nullptr;
        continue;
      }
      WARN("nvmlDeviceGetHandleByIndex(%d) failed: %s", int(a), pfn_nvmlErrorString(res1));
      initResult = ncclSystemError;
      return initResult;
    }

    // 获取指定 NVML 设备句柄的计算能力
    res1 = pfn_nvmlDeviceGetCudaComputeCapability(ncclNvmlDevices[a].handle, &ncclNvmlDevices[a].computeCapabilityMajor,
                                                  &ncclNvmlDevices[a].computeCapabilityMinor);
    if (res1 != NVML_SUCCESS) {
      if (!ncclNvmlDevices[a].cudaVisible) continue;
      WARN("nvmlDeviceGetCudaComputeCapability(%d) failed: %s", int(a), pfn_nvmlErrorString(res1));
      initResult = ncclSystemError;
      return initResult;
    }
  }

  // 两层循环遍历 N×N P2P 状态矩阵
  for (int a = 0; a < ncclNvmlDeviceCount; a++) {
    for (int b = 0; b < ncclNvmlDeviceCount; b++) {
      if (ncclNvmlDevices[a].handle == nullptr || ncclNvmlDevices[b].handle == nullptr) {
        ncclNvmlDevicePairs[a][b].p2pStatusRead = NVML_P2P_STATUS_UNKNOWN;
        ncclNvmlDevicePairs[a][b].p2pStatusWrite = NVML_P2P_STATUS_UNKNOWN;
        continue;
      }
      nvmlDevice_t da = ncclNvmlDevices[a].handle;
      nvmlDevice_t db = ncclNvmlDevices[b].handle;

      // 查询指定两个 NVML 设备间是否支持 P2P 读
      res1 = pfn_nvmlDeviceGetP2PStatus(da, db, NVML_P2P_CAPS_INDEX_READ, &ncclNvmlDevicePairs[a][b].p2pStatusRead);
      if (res1 != NVML_SUCCESS) {
        if (!ncclNvmlDevices[a].cudaVisible || !ncclNvmlDevices[b].cudaVisible) {
          ncclNvmlDevicePairs[a][b].p2pStatusRead = NVML_P2P_STATUS_UNKNOWN;
        } else {
          WARN("nvmlDeviceGetP2PStatus(%d,%d,NVML_P2P_CAPS_INDEX_READ) failed: %s", a, b, pfn_nvmlErrorString(res1));
          initResult = ncclSystemError;
          return initResult;
        }
      }

      // 查询指定两个 NVML 设备间是否支持 P2P 写
      res1 = pfn_nvmlDeviceGetP2PStatus(da, db, NVML_P2P_CAPS_INDEX_WRITE, &ncclNvmlDevicePairs[a][b].p2pStatusWrite);
      if (res1 != NVML_SUCCESS) {
        if (!ncclNvmlDevices[a].cudaVisible || !ncclNvmlDevices[b].cudaVisible) {
          ncclNvmlDevicePairs[a][b].p2pStatusWrite = NVML_P2P_STATUS_UNKNOWN;
          continue;
        }
        WARN("nvmlDeviceGetP2PStatus(%d,%d,NVML_P2P_CAPS_INDEX_WRITE) failed: %s", a, b, pfn_nvmlErrorString(res1));
        initResult = ncclSystemError;
        return initResult;
      }
    }
  }

  initResult = ncclSuccess;
  return initResult;
}

#define NVMLCHECK(name, ...) \
  do { \
    nvmlReturn_t e44241808 = pfn_##name(__VA_ARGS__); \
    if (e44241808 != NVML_SUCCESS) { \
      WARN(#name "() failed: %s", pfn_nvmlErrorString(e44241808)); \
      return ncclSystemError; \
    } \
  } while (0)

#define NVMLTRY(name, ...) \
  do { \
    if (!NCCL_NVML_DIRECT && pfn_##name == nullptr) \
      return ncclInternalError; /* missing symbol is not a warned error */ \
    nvmlReturn_t e44241808 = pfn_##name(__VA_ARGS__); \
    if (e44241808 != NVML_SUCCESS) { \
      if (e44241808 != NVML_ERROR_NOT_SUPPORTED) \
        INFO(NCCL_INIT, #name "() failed: %s", pfn_nvmlErrorString(e44241808)); \
      return ncclSystemError; \
    } \
  } while (0)

ncclResult_t ncclNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  // 懒初始化：首次用时才 dlopen libnvml
  NCCLCHECK(ncclNvmlEnsureInitialized());
  // NVML 调用全局串行化
  std::lock_guard<std::mutex> locked(lock);
  // 调真正的 NVML 函数指针
  NVMLCHECK(nvmlDeviceGetHandleByPciBusId, pciBusId, device);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  *device = ncclNvmlDevices[index].handle;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetName, device, name, length);
  return ncclSuccess;
}

ncclResult_t ncclNvmlSystemGetDriverVersion(char* version, unsigned int length) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlSystemGetDriverVersion, version, length);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType,
                                                 nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType,
                                                 unsigned long long* count) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetMemoryErrorCounter, device, errorType, counterType, locationType, count);
  return ncclSuccess;
}

// Re-queries NVML on each call, unlike the cached ncclNvmlDeviceCount global.
ncclResult_t ncclNvmlDeviceGetCount(unsigned int* deviceCount) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
#if NCCL_NVML_DIRECT
  bool have_v2 = true;
#else
  // if this compare is done in the NCCL_NVML_DIRECT=1 case then GCC warns about it never being null
  bool have_v2 = pfn_nvmlInit_v2 != nullptr;
#endif
  if (have_v2) {
    NVMLTRY(nvmlDeviceGetCount_v2, deviceCount);
  } else {
    NVMLTRY(nvmlDeviceGetCount, deviceCount);
  }
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  // 懒初始化：首次用时才 dlopen libnvml
  NCCLCHECK(ncclNvmlEnsureInitialized());
  // NCCL 初始化时已经把整机 NVML 设备缓存进 ncclNvmlDevices[]，这里靠句柄相等反查数组下标，就是这块卡的 NVML 索引
  for (int d = 0; d < ncclNvmlDeviceCount; d++) {
    if (ncclNvmlDevices[d].handle == device) {
      *index = d;
      return ncclSuccess;
    }
  }
  return ncclInvalidArgument;
}

ncclResult_t ncclNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetNvLinkState, device, link, isActive);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetNvLinkRemotePciInfo, device, link, pci);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link,
                                                     nvmlIntNvLinkDeviceType_t* NvLinkDeviceType) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetNvLinkRemoteDeviceType, device, link, NvLinkDeviceType);
  return ncclSuccess;
}

// 询某块 GPU 的某条 NVLink 链路是否具备某种能力（如是否支持 P2P、是否支持系统内存访问等）
ncclResult_t ncclNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                               nvmlNvLinkCapability_t capability, unsigned int* capResult) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  /*
  device：NVML 设备句柄
  link：NVLink 链路编号（0、1、2…，一块 Hopper GPU 最多 18 条）
  capability：要查询的能力枚举 nvmlNvLinkCapability_t，如 NVML_NVLINK_CAP_P2P_SUPPORTED、NVML_NVLINK_CAP_SYS_MEM_ACCESS 等
  capResult：出参，返回该能力是否存在（非 0 = 支持）
  */
  NVMLTRY(nvmlDeviceGetNvLinkCapability, device, link, capability, capResult);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  NCCLCHECK(ncclNvmlEnsureInitialized());

  for (int d = 0; d < ncclNvmlDeviceCount; d++) {
    if (device == ncclNvmlDevices[d].handle) {
      *major = ncclNvmlDevices[d].computeCapabilityMajor;
      *minor = ncclNvmlDevices[d].computeCapabilityMinor;
      return ncclSuccess;
    }
  }
  return ncclInvalidArgument;
}

ncclResult_t ncclNvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex,
                                        nvmlGpuP2PStatus_t* p2pStatus) {
  NCCLCHECK(ncclNvmlEnsureInitialized());

  if (p2pIndex == NVML_P2P_CAPS_INDEX_READ || p2pIndex == NVML_P2P_CAPS_INDEX_WRITE) {
    int a = -1, b = -1;
    for (int d = 0; d < ncclNvmlDeviceCount; d++) {
      if (device1 == ncclNvmlDevices[d].handle) a = d;
      if (device2 == ncclNvmlDevices[d].handle) b = d;
    }
    if (a == -1 || b == -1) return ncclInvalidArgument;
    if (p2pIndex == NVML_P2P_CAPS_INDEX_READ) *p2pStatus = ncclNvmlDevicePairs[a][b].p2pStatusRead;
    else *p2pStatus = ncclNvmlDevicePairs[a][b].p2pStatusWrite;
  } else {
    std::lock_guard<std::mutex> locked(lock);
    NVMLCHECK(nvmlDeviceGetP2PStatus, device1, device2, p2pIndex, p2pStatus);
  }
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(nvmlDeviceGetFieldValues, device, valuesCount, values);
  return ncclSuccess;
}

// MNNVL support
ncclResult_t ncclNvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t* gpuFabricInfo) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  // NVML 的版本化结构要求调用方先填版本号，NVML 据此知道该按哪个版本的布局填充
  gpuFabricInfo->version = nvmlGpuFabricInfo_v2;
  // 查询一块 GPU 在机架级 NVLink 系统里的物理位置信息
  NVMLTRY(nvmlDeviceGetGpuFabricInfoV, device, gpuFabricInfo);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t* platformInfo) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  // 也是必须先填写版本号
  platformInfo->version = nvmlPlatformInfo_v2;
  // 
  NVMLTRY(nvmlDeviceGetPlatformInfo, device, platformInfo);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pciInfo) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLCHECK(nvmlDeviceGetPciInfo_v3, device, pciInfo);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t* pciInfoExt) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  INFO(NCCL_GRAPH, "ncclNvmlDeviceGetPciInfoExt: nvmlPciInfoExt_v1: %u", nvmlPciInfoExt_v1);
  pciInfoExt->version = nvmlPciInfoExt_v1;
  NVMLCHECK(nvmlDeviceGetPciInfoExt, device, pciInfoExt);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, int* maxSpeed) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLCHECK(nvmlDeviceGetPcieLinkMaxSpeed, device, maxSpeed);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLCHECK(nvmlDeviceGetCurrPcieLinkGeneration, device, currLinkGen);
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLCHECK(nvmlDeviceGetCurrPcieLinkWidth, device, currLinkWidth);
  return ncclSuccess;
}

ncclResult_t ncclNvmlGetCCStatus(struct ncclNvmlCCStatus* status) {
  NCCLCHECK(ncclNvmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  nvmlCCInfoInternal ccInfo;
  if (pfn_nvmlSystemGetConfComputeSettings != NULL) {
    ccInfo.settingV12040.version = nvmlSystemConfComputeSettings_v1;
    NVMLTRY(nvmlSystemGetConfComputeSettings, &ccInfo.settingV12040);
    if (ccInfo.settingV12040.ccFeature == NVML_CC_SYSTEM_FEATURE_ENABLED) status->CCEnabled = true;
    else status->CCEnabled = false;

    if (ccInfo.settingV12040.multiGpuMode == NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE) {
      status->multiGpuProtectedPCIE = true;
    } else {
      status->multiGpuProtectedPCIE = false;
    }
    if (ccInfo.settingV12040.multiGpuMode == NVML_CC_SYSTEM_MULTIGPU_NVLE) status->multiGpuNVLE = true;
    else status->multiGpuNVLE = false;
  } else if (pfn_nvmlSystemGetConfComputeState != NULL) {
    NVMLTRY(nvmlSystemGetConfComputeState, &ccInfo.settingV12020);
    if (ccInfo.settingV12020.ccFeature == NVML_CC_SYSTEM_FEATURE_ENABLED) status->CCEnabled = true;
    else status->CCEnabled = false;
    status->multiGpuProtectedPCIE = false;
    status->multiGpuNVLE = false;
  } else {
    status->CCEnabled = false;
    status->multiGpuProtectedPCIE = false;
    status->multiGpuNVLE = false;
  }
  return ncclSuccess;
}
