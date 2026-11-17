/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ALLOC_H_
#define NCCL_ALLOC_H_

#include "nccl.h"
#include "checks.h"
#include "bitops.h"
#include "utils.h"
#include "p2p.h"
#include "mem_manager.h"
struct ncclComm;
#include "os.h"
#include <memory>
#include <stdlib.h>
#include <string.h>

#if CUDART_VERSION >= 11030
#include <cuda.h>
#include "cudawrap.h"
#endif

#if defined(NCCL_OS_LINUX)
#include <unistd.h>
#endif

uint64_t clockNano(); // from utils.h with which we have a circular dependency

template <typename T>
constexpr size_t ncclSizeOfT() {
  return sizeof(T);
}
template <>
constexpr size_t ncclSizeOfT<void>() {
  return 1;
}

// 标准库的 std::unique_ptr<T> 默认删除器调用 delete，由于 NCCL 用 malloc/calloc 分配内存，不能用 delete 释放，所以 NCCL 自定义了删除器，调用 std::free
// C++14-compatible wrapper that captures function pointers through template parameters.
template <typename FunctionPtr, FunctionPtr Function>
struct ncclDeleterWrapper {
  template <typename... Args>
  constexpr auto operator()(Args&&... args) const {
    return Function(std::forward<Args>(args)...);
  }
}; // struct ncclDeleterWrapper

using ncclDeleterFree = ncclDeleterWrapper<decltype(&std::free), std::free>;
template <typename T>
using ncclUniquePtr = std::unique_ptr<T, ncclDeleterFree>;
template <typename T>
using ncclUniqueArrayPtr = std::unique_ptr<T[], ncclDeleterFree>;

#if CUDART_VERSION >= 12020

static inline ncclResult_t ncclCuMemHostAllocDebug(void** ptr, CUmemGenericAllocationHandle* handlep, size_t size,
                                                   const char* file, int line, const char* callerFunc) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int cpuNumaNodeId = -1;
  CUmemAllocationHandleType type = ncclCuMemHandleType;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  CUCHECK(cuDeviceGetAttribute(&cpuNumaNodeId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, currentDev));
  if (cpuNumaNodeId < 0) cpuNumaNodeId = 0;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = type; // So it can be exported
  prop.location.id = cpuNumaNodeId;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation */
  CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
  /* Now allow RW access to the newly mapped memory for local GPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  /* Now allow RW access to the newly mapped memory from the CPU */
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  accessDesc.location.id = cpuNumaNodeId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));

  if (handlep) *handlep = handle;
  INFO_LOC_FN(NCCL_ALLOC_HOST, file, line, callerFunc,
              "CUMEM Host Alloc Size %zi pointer %p handle %llx numa %d dev %d granularity %ld", size, *ptr, handle,
              cpuNumaNodeId, cudaDev, granularity);
  return result;
}
#define ncclCuMemHostAlloc(ptr, handlep, size) \
  ncclCuMemHostAllocDebug((ptr), (handlep), (size), __FILE__, __LINE__, __func__)

static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  CUCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  TRACE(NCCL_ALLOC, "CUMEM Host Free Size %zi pointer %p handle 0x%llx", size, ptr, handle);
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CUCHECK(cuMemRelease(handle));
  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  return result;
}

#else /* CUDART_VERSION >= 12020 */

static inline ncclResult_t ncclCuMemHostAllocDebug(void** ptr, void* handlep, size_t size, const char* file, int line,
                                                   const char* callerFunc) {
  (void)ptr;
  (void)handlep;
  (void)size;
  (void)file;
  (void)line;
  (void)callerFunc;
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}
#define ncclCuMemHostAlloc(ptr, handlep, size) \
  ncclCuMemHostAllocDebug((ptr), (handlep), (size), __FILE__, __LINE__, __func__)

static inline ncclResult_t ncclCuMemHostFree(void* ptr) {
  WARN("CUMEM Host is not supported prior to CUDA 12.2");
  return ncclInternalError;
}

#endif  /* CUDART_VERSION >= 12020 */

template <typename T>
ncclResult_t ncclCudaHostCallocDebug(T** ptr, size_t nelem, const char* file, int line, const char* callerFunc) {
  ncclResult_t result = ncclSuccess;
  // cudaStreamCaptureModeRelaxed 表示放宽限制，允许在 capture 期间执行不可录制的操作
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  // 把当前线程的 capture 模式临时换成 Relaxed
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    // 分配 pinned + GPU 可映射的主机内存
    CUDACHECKGOTO(cudaHostAlloc(ptr, nelem * ncclSizeOfT<T>(), cudaHostAllocMapped), result, finish);
    memset(*ptr, 0, nelem * ncclSizeOfT<T>());
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA host alloc %ld bytes", nelem * ncclSizeOfT<T>());
  INFO_LOC_FN(NCCL_ALLOC_HOST, file, line, callerFunc, "Cuda Host Alloc Size %ld pointer %p", nelem * ncclSizeOfT<T>(),
              *ptr);
  return result;
}

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

#define ncclCudaHostCalloc(...) ncclCudaHostCallocDebug(__VA_ARGS__, __FILE__, __LINE__, __func__)

template <typename T>
ncclResult_t ncclCallocDebug(T** ptr, size_t nelem, const char* file, int line, const char* callerFunc,
                             bool logHostAlloc) {
  if (nelem > 0) {
    T* p = (T*)malloc(nelem * ncclSizeOfT<T>());
    if (p == NULL) {
      WARN("Failed to malloc %ld bytes", nelem * ncclSizeOfT<T>());
      return ncclSystemError;
    }
    memset((void*)p, 0, nelem * ncclSizeOfT<T>());
    *ptr = p;
    if (logHostAlloc)
      INFO_LOC_FN(NCCL_ALLOC_HOST, file, line, callerFunc, "Host Calloc Size %ld pointer %p", nelem * ncclSizeOfT<T>(),
                  p);
  } else {
    *ptr = NULL;
  }
  return ncclSuccess;
}

template <typename T>
ncclResult_t ncclCallocDebug(ncclUniquePtr<T>& ptr, size_t nelem, const char* file, int line, const char* callerFunc,
                             bool logHostAlloc) {
  typename ncclUniquePtr<T>::pointer p = nullptr;
  ncclResult_t result = ncclCallocDebug(&p, nelem, file, line, callerFunc, logHostAlloc);
  ptr.reset(p);
  return result;
}

template <typename T>
ncclResult_t ncclCallocDebug(ncclUniqueArrayPtr<T>& ptr, size_t nelem, const char* file, int line,
                             const char* callerFunc, bool logHostAlloc) {
  typename ncclUniqueArrayPtr<T>::pointer p = nullptr;
  ncclResult_t result = ncclCallocDebug(&p, nelem, file, line, callerFunc, logHostAlloc);
  ptr.reset(p);
  return result;
}

#define ncclCalloc(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__, __func__, true)
/* Quiet calloc/realloc skip NCCL_ALLOC_HOST INFO on very high-churn host paths only. */
#define ncclCallocQuiet(...) ncclCallocDebug(__VA_ARGS__, __FILE__, __LINE__, __func__, false)

template <typename T>
ncclResult_t ncclReallocDebug(T** ptr, size_t oldNelem, size_t nelem, const char* file, int line,
                              const char* callerFunc, bool logHostAlloc) {
  T* oldp = *ptr;
  if (nelem < oldNelem || (oldp == NULL && oldNelem > 0)) return ncclInternalError;
  if (nelem == oldNelem) return ncclSuccess;

  T* p = (T*)malloc(nelem * ncclSizeOfT<T>());
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem * ncclSizeOfT<T>());
    return ncclSystemError;
  }
  if (oldp && oldNelem) memcpy(p, oldp, oldNelem * ncclSizeOfT<T>());
  if (oldp) free(oldp);
  memset(p + oldNelem, 0, (nelem - oldNelem) * ncclSizeOfT<T>());
  *ptr = (T*)p;
  if (logHostAlloc)
    INFO_LOC_FN(NCCL_ALLOC_HOST, file, line, callerFunc, "Mem Realloc old size %ld, new size %ld pointer %p",
                oldNelem * ncclSizeOfT<T>(), nelem * ncclSizeOfT<T>(), *ptr);
  return ncclSuccess;
}
#define ncclRealloc(ptr, oldNelem, nelem) \
  ncclReallocDebug((ptr), (oldNelem), (nelem), __FILE__, __LINE__, __func__, true)
#define ncclReallocQuiet(ptr, oldNelem, nelem) \
  ncclReallocDebug((ptr), (oldNelem), (nelem), __FILE__, __LINE__, __func__, false)

#if CUDART_VERSION >= 11030

#include <cuda.h>
#include "cudawrap.h"

// Helper function to map memory and set access permissions for a device
static inline ncclResult_t ncclCuMemMapAndSetAccess(void* ptr, size_t size, CUmemGenericAllocationHandle handle,
                                                    int cudaDev) {
  ncclResult_t result = ncclSuccess;
  // Map the virtual address range to the physical allocation
  CUCHECK(cuMemMap((CUdeviceptr)ptr, size, 0, handle, 0));
  // Set access permissions for the device
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess((CUdeviceptr)ptr, size, &accessDesc, 1));
  return result;
}

// ncclCuMemAllocAddr takes memory handle and size and returns the mapped address pointer
static inline ncclResult_t ncclCuMemAllocAddr(void** ptr, CUmemGenericAllocationHandle* handleIn, size_t size) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUmemAllocationProp prop = {};
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, *handleIn));
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation and set access permissions */
  NCCLCHECK(ncclCuMemMapAndSetAccess(*ptr, size, *handleIn, cudaDev));
  TRACE(NCCL_ALLOC, "CuMem Map Size %zu pointer %p handle %llx", size, *ptr, *handleIn);
  return result;
}

static inline ncclResult_t ncclCuMemFreeAddr(void* ptr, struct ncclMemManager* manager, int numSegments = 1) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  size_t totalSize = 0;
  for (int segment = 0; segment < numSegments; segment++) {
    size_t segmentSize = 0;
    CUCHECK(cuMemGetAddressRange(NULL, &segmentSize, (CUdeviceptr)ptr + totalSize));
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr + totalSize, segmentSize));
    totalSize += segmentSize;
  }

  // Untrack from memory manager
  if (manager != nullptr) {
    NCCLCHECK(ncclMemUntrack(manager, ptr, totalSize));
  }

  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, totalSize));
  return result;
}

static inline ncclResult_t ncclCuMemAlloc(void** ptr, CUmemGenericAllocationHandle* handlep,
                                          CUmemAllocationHandleType type, size_t size, struct ncclMemManager* manager,
                                          ncclMemType_t memType = ncclMemPersist) {
  ncclResult_t result = ncclSuccess;
  size_t granularity = 0;
  CUdevice currentDev;
  CUmemAllocationProp prop = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int flag = 0;
  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = type;
  prop.location.id = currentDev;
  // Query device to see if RDMA support is available
  CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, currentDev));
  if (flag) prop.allocFlags.gpuDirectRDMACapable = 1;
  CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ALIGN_SIZE(size, granularity);
  /* Allocate the physical memory on the device */
  CUCHECK(cuMemCreate(&handle, size, &prop, 0));
  /* Reserve a virtual address range */
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
  /* Map the virtual address range to the physical allocation and set access permissions */
  NCCLCHECK(ncclCuMemMapAndSetAccess(*ptr, size, handle, currentDev));
  if (handlep) *handlep = handle;
  TRACE(NCCL_ALLOC, "CuMem Alloc Size %zu pointer %p handle %llx", size, *ptr, handle);
  /* Track allocation in memory manager */
  if (manager != nullptr) {
    ncclMemTrack(manager, *ptr, size, handle, type, memType);
  }
  return result;
}

static inline ncclResult_t ncclCuMemFree(void* ptr, struct ncclMemManager* manager, int numSegments = 1) {
  if (ptr == NULL) return ncclSuccess;
  ncclResult_t result = ncclSuccess;
  size_t totalSize = 0;
  for (int segment = 0; segment < numSegments; segment++) {
    CUmemGenericAllocationHandle handle;
    size_t segmentSize = 0;
    CUCHECK(cuMemRetainAllocationHandle(&handle, (void*)((char*)ptr + totalSize)));
    CUCHECK(cuMemRelease(handle));
    CUCHECK(cuMemGetAddressRange(NULL, &segmentSize, (CUdeviceptr)ptr + totalSize));
    TRACE(NCCL_ALLOC, "CuMem Free Size %zu pointer %p handle 0x%llx segment %d numSegments %d", segmentSize, ptr,
          handle, segment, numSegments);
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr + totalSize, segmentSize));
    CUCHECK(cuMemRelease(handle));
    totalSize += segmentSize;
  }

  // Update tracking with total size after processing all segments
  if (manager != nullptr) {
    NCCLCHECK(ncclMemUntrack(manager, ptr, totalSize));
  }

  CUCHECK(cuMemAddressFree((CUdeviceptr)ptr, totalSize));
  return result;
}

// Get the base and size of all segments that span a given user buffer
static inline ncclResult_t ncclCuMemGetAddressRange(CUdeviceptr userBuff, size_t userBuffSize,
                                                    CUdeviceptr* mappedPtrBase, size_t* totalMappedBufferSize,
                                                    int* numSegments, bool* hasSysmemSegment = nullptr) {
  *totalMappedBufferSize = 0;
  *mappedPtrBase = 0;
  if (numSegments) *numSegments = 0;
  CUdeviceptr userBuffStart = userBuff;
  CUdeviceptr userBuffEnd = userBuffStart + userBuffSize;
  CUdeviceptr mappedPtrEnd = userBuffStart;
  CUdeviceptr baseSend;
  size_t baseSendSize;

  if (hasSysmemSegment != nullptr) {
    *hasSysmemSegment = false;
  }

  while (mappedPtrEnd < userBuffEnd) {
    CUCHECK(cuMemGetAddressRange(&baseSend, &baseSendSize, mappedPtrEnd));

    if (hasSysmemSegment != nullptr && *hasSysmemSegment == false) {
      CUmemGenericAllocationHandle handle;
      CUmemAllocationProp prop;
      CUCHECK(cuMemRetainAllocationHandle(&handle, (void*)mappedPtrEnd));
      CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, handle));
      if (prop.location.type == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
        *hasSysmemSegment = true;
      }
      CUCHECK(cuMemRelease(handle));
    }

    if (*totalMappedBufferSize == 0) {
      // Workaround for CPU backed buffers since baseSend can be 0 in some CUDA driver versions
      if (baseSend == 0) {
        *mappedPtrBase = userBuffStart;
      } else {
        *mappedPtrBase = baseSend;
      }
    }
    *totalMappedBufferSize += baseSendSize;
    // Workaround for CPU backed buffers since baseSend can be 0 in some CUDA driver versions
    if (baseSend == 0) {
      mappedPtrEnd = mappedPtrEnd + baseSendSize;
    } else {
      mappedPtrEnd = baseSend + baseSendSize;
    }

    if (numSegments) *numSegments = *numSegments + 1;
  }
  return ncclSuccess;
}

#else

extern int ncclCuMemEnable();

static inline ncclResult_t ncclCuMemAlloc(void** ptr, void* handlep, int type, size_t size,
                                          struct ncclMemManager* manager, ncclMemType_t memType = ncclMemScratch) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}
static inline ncclResult_t ncclCuMemFree(void* ptr, struct ncclMemManager* manager, int numSegments = 1) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemAllocAddr(void** ptr, CUmemGenericAllocationHandle* handleIn, size_t size) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemFreeAddr(void* ptr, struct ncclMemManager* manager, int numSegments = 1) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

static inline ncclResult_t ncclCuMemGetAddressRange(CUdeviceptr userBuff, size_t userBuffSize,
                                                    CUdeviceptr* mappedPtrBase, size_t* totalMappedBufferSize,
                                                    int* numSegments) {
  WARN("CUMEM not supported prior to CUDA 11.3");
  return ncclInternalError;
}

#endif

/*
  T** ptr + size_t nelem：模板化的"分配 nelem 个 T"，实际字节数 = nelem * ncclSizeOfT<T>()。
ncclSizeOfT<T>() 就是 sizeof(T)，但对 void 特化为 1（让"分配 N 字节裸内存"也能用同一套接口）。
  file/line/callerFunc：调试元数据，仅用于日志定位。调用方不手写它们，而是通过宏自动注入。
  struct ncclMemManager* manager：内存管理器，负责这块显存的记账与 suspend/resume（只在 VMM 路径真正用到）。
  ncclMemType_t memType = ncclMemPersist：内存"驻留策略"，默认 Persist。
*/
template <typename T>
ncclResult_t ncclCudaMallocDebug(T** ptr, size_t nelem, const char* file, int line, const char* callerFunc,
                                 struct ncclMemManager* manager, ncclMemType_t memType = ncclMemPersist) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  /*
  交换语义，把传入的 mode（此刻是 Relaxed）设为本线程新的捕获模式，同时把线程原来的模式写回 mode。
  调用后：线程处于 Relaxed，变量 mode 保存着"原模式"，供 finish 处还原。这是"两阶段交换"的第一步。

  为什么要 Relaxed：cudaMalloc / cuMem 这类分配是不可被捕获进 CUDA graph 的"不安全 API"。
  如果调用线程恰好处在 cudaStreamBeginCapture 的捕获状态中，
  在默认的 Global/ThreadLocal 模式下调用它会直接报 cudaErrorStreamCaptureUnsupported。
  而 NCCL 的内部分配（abort flag、通信结构、缓冲区）常常发生在用户正把 NCCL 集合通信录进 graph 的时候，
  这些分配是一次性 setup，本就不该被录进可重放的图里，也绝不能失败。切到 Relaxed 关闭本线程的捕获安全检查，让分配照常执行但不被捕获。
  */
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    if (ncclCuMemEnable()) {
      /*
      走 VMM 路径 ncclCuMemAlloc：用 cuMem 驱动 API（cuMemCreate/cuMemAddressReserve/cuMemMap）分配，
      ncclCuMemHandleType（默认 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR）指定可跨进程共享的句柄类型，并把 manager+memType 传进去做记账
      */
      NCCLCHECKGOTO(ncclCuMemAlloc((void**)ptr, NULL, ncclCuMemHandleType, nelem * ncclSizeOfT<T>(), manager, memType),
                    result, finish);
    } else {
      // 走传统 cudaMalloc：一把梭的普通设备分配，没有可共享句柄、不进 memManager 记账
      CUDACHECKGOTO(cudaMalloc(ptr, nelem * ncclSizeOfT<T>()), result, finish);
    }
  }
finish:
  // 线程恢复到进入前的捕获模式
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA malloc %ld bytes", nelem * ncclSizeOfT<T>());
  INFO_LOC_FN(NCCL_ALLOC, file, line, callerFunc, "Cuda Alloc Size %ld pointer %p memType %d", nelem * ncclSizeOfT<T>(),
              *ptr, memType);
  return result;
}

#define ncclCudaMalloc(ptr, nelem, manager, ...) \
  ncclCudaMallocDebug(ptr, nelem, __FILE__, __LINE__, __func__, manager, ##__VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaCallocDebug(T** ptr, size_t nelem, const char* file, int line, const char* callerFunc,
                                 struct ncclMemManager* manager, ncclMemType_t memType = ncclMemPersist) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    // Need a side stream so as not to interfere with graph capture.
    cudaStream_t stream;
    CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void**)ptr, NULL, ncclCuMemHandleType, nelem * ncclSizeOfT<T>(), manager, memType),
                    result, finish);
    } else {
      CUDACHECKGOTO(cudaMalloc(ptr, nelem * ncclSizeOfT<T>()), result, finish);
    }
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem * ncclSizeOfT<T>(), stream), result, finish);
    CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
    CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc %ld bytes", nelem * ncclSizeOfT<T>());
  INFO_LOC_FN(NCCL_ALLOC, file, line, callerFunc, "Cuda Calloc Size %ld pointer %p memType %d",
              nelem * ncclSizeOfT<T>(), *ptr, memType);
  return result;
}
#define ncclCudaCalloc(ptr, nelem, manager, ...) \
  ncclCudaCallocDebug(ptr, nelem, __FILE__, __LINE__, __func__, manager, ##__VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaCallocAsyncDebug(T** ptr, size_t nelem, cudaStream_t stream, const char* file, int line,
                                      const char* callerFunc, struct ncclMemManager* manager,
                                      ncclMemType_t memType = ncclMemPersist) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  *ptr = nullptr;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (nelem > 0) {
    if (ncclCuMemEnable()) {
      NCCLCHECKGOTO(ncclCuMemAlloc((void**)ptr, NULL, ncclCuMemHandleType, nelem * ncclSizeOfT<T>(), manager, memType),
                    result, finish);
    } else {
      CUDACHECKGOTO(cudaMalloc(ptr, nelem * ncclSizeOfT<T>()), result, finish);
    }
    CUDACHECKGOTO(cudaMemsetAsync(*ptr, 0, nelem * ncclSizeOfT<T>(), stream), result, finish);
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (*ptr == nullptr && nelem > 0) WARN("Failed to CUDA calloc async %ld bytes", nelem * ncclSizeOfT<T>());
  INFO_LOC_FN(NCCL_ALLOC, file, line, callerFunc, "Cuda CallocAsync Size %ld pointer %p memType %d",
              nelem * ncclSizeOfT<T>(), *ptr, memType);
  return result;
}
#define ncclCudaCallocAsync(ptr, nelem, stream, manager, ...) \
  ncclCudaCallocAsyncDebug(ptr, nelem, stream, __FILE__, __LINE__, __func__, manager, ##__VA_ARGS__)

template <typename T>
ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  // Need a side stream so as not to interfere with graph capture.
  cudaStream_t stream;
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), result, finish);
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(dst, src, nelem, stream), result, finish);
  CUDACHECKGOTO(cudaStreamSynchronize(stream), result, finish);
  CUDACHECKGOTO(cudaStreamDestroy(stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCudaMemcpyAsync(T* dst, T* src, size_t nelem, cudaStream_t stream) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  CUDACHECKGOTO(cudaMemcpyAsync(dst, src, nelem * ncclSizeOfT<T>(), cudaMemcpyDefault, stream), result, finish);
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

template <typename T>
ncclResult_t ncclCudaFree(T* ptr, struct ncclMemManager* manager, int numSegments = 1) {
  ncclResult_t result = ncclSuccess;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  TRACE(NCCL_ALLOC, "Cuda Free pointer %p", ptr);
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  if (ncclCuMemEnable()) {
    NCCLCHECKGOTO(ncclCuMemFree((void*)ptr, manager, numSegments), result, finish);
  } else {
    if (numSegments > 1) {
      result = ncclUnhandledCudaError;
      goto finish;
    } else {
      CUDACHECKGOTO(cudaFree(ptr), result, finish);
    }
  }
finish:
  CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
  return result;
}

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// allocated on separate pages as those pages will be marked DONTFORK
// and if they are shared, that could cause a crash in a child process
inline ncclResult_t ncclIbMallocDebug(void** ptr, size_t size, const char* file, int line, const char* callerFunc) {
  if (size > 0) {
    void* p = NULL;
    size_t page_size = ncclOsGetPageSize();
#if defined(NCCL_OS_LINUX)
    size_t size_aligned = ROUNDUP(size, page_size);
    int ret = posix_memalign(&p, page_size, size_aligned);
    if (ret != 0) return ncclSystemError;
#elif defined(NCCL_OS_WINDOWS)
    size_t size_aligned = ROUNDUP(size, page_size);
    p = _aligned_malloc(size_aligned, page_size);
    if (p == NULL) return ncclSystemError;
#endif
    if (p != NULL) {
      memset(p, 0, size);
    }
    *ptr = p;
  } else {
    *ptr = NULL;
  }
  INFO_LOC_FN(NCCL_ALLOC, file, line, callerFunc, "Ib Alloc Size %ld pointer %p", size, *ptr);
  return ncclSuccess;
}
#define ncclIbMalloc(...) ncclIbMallocDebug(__VA_ARGS__, __FILE__, __LINE__, __func__)

#endif
