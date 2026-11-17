/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_DEVICE_H_
#define NCCL_NET_DEVICE_H_

#define NCCL_NET_DEVICE_INVALID_VERSION 0x0
#define NCCL_NET_MTU_SIZE 4096

// Arbitrary version number - A given NCCL build will only be compatible with a single device networking plugin
// version. NCCL will check the supplied version number from net->getProperties() and compare to its internal version.
#define NCCL_NET_DEVICE_UNPACK_VERSION 0x7

typedef enum {
  NCCL_NET_DEVICE_HOST = 0,  // 不卸载：CPU proxy 驱动网络
  NCCL_NET_DEVICE_UNPACK = 1,  // 卸载到 GPU：recv 侧数据由 device 直接 unpack 落位
  NCCL_NET_DEVICE_GIN_PROXY = 2,  // GIN：GPU 发起，CPU proxy 协助推进
  NCCL_NET_DEVICE_GIN_GDAKI = 3,  // GIN：GPUDirect Async Kernel-Initiated，GPU 直接敲网卡
  NCCL_NET_DEVICE_GIN_GPI = 4,  // GIN：另一种 GPU 发起后端
} ncclNetDeviceType;

typedef struct {
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
  int needsProxyProgress;
} ncclNetDeviceHandle_v7_t;

typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_v8_t;
typedef ncclNetDeviceHandle_v8_t ncclNetDeviceHandle_v9_t;
typedef ncclNetDeviceHandle_v9_t ncclNetDeviceHandle_v10_t;
typedef ncclNetDeviceHandle_v10_t ncclNetDeviceHandle_v11_t;
typedef ncclNetDeviceHandle_v11_t ncclNetDeviceHandle_v12_t;
typedef ncclNetDeviceHandle_v12_t ncclNetDeviceHandle_t;

#endif
