/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION &
 *AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cuda_runtime.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// 用于 NCCL 操作的增强型错误检查宏
// 提供详细的错误信息，包括失败的操作
#define NCCLCHECK(cmd)                                                         \
    do {                                                                       \
        ncclResult_t res = cmd;                                                \
        if (res != ncclSuccess) {                                              \
            fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__,       \
                    __LINE__, ncclGetErrorString(res));                        \
            fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t err = cmd;                                                 \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "Failed CUDA error %s:%d '%s'\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err));                        \
            fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(int argc, char *argv[]) {
    // 管理多个 GPU communicators 的变量
    int num_gpus;             // 可用的 CUDA 设备数量
    ncclComm_t *comms = NULL; // NCCL communicators 的数组 (每个 GPU 一个)
    cudaStream_t *streams = NULL; // CUDA streams 的数组 (每个 GPU 一个)
    int *devices = NULL;          // 设备 ID 的数组

    // 确定有多少可用的 CUDA 设备，这决定了需要创建多少 communicators
    CUDACHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        fprintf(stderr, "ERROR: No CUDA devices found on this system\n");
        fprintf(stderr, "Please ensure CUDA is properly installed and GPUs are "
                        "available\n");
        return 1;
    }
    printf("Found %d CUDA device(s) available\n\n", num_gpus);

    // ========================================================================
    // STEP 1: 准备设备信息并分配内存
    // ========================================================================

    // 分配数组来保存我们的 per-device 资源
    // 每个 GPU 需要一个 communicator, stream, 和 device ID
    devices = (int *)malloc(num_gpus * sizeof(int));
    comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
    streams = (cudaStream_t *)malloc(num_gpus * sizeof(cudaStream_t));
    if (!devices || !comms || !streams) {
        fprintf(stderr, "ERROR: Failed to allocate resources for device arrays\n");
        return 1;
    }

    // 创建设备列表并显示设备信息
    // 默认情况下，使用所有可用的设备 (0, 1, 2, ...)
    printf("Available GPU devices:\n");
    for (int i = 0;i<num_gpus;i++) {
        devices[i] = i; // communicator_i 使用 device_i
        // 查询设备属性以显示信息
        cudaDeviceProp prop;
        CUDACHECK(cudaGetDeviceProperties(&prop, devices[i]));
        printf("  GPU %d: %s (CUDA Device %d)\n", i, prop.name, devices[i]);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    // 创建每个 GPU 的 CUDA stream
    for (int i = 0;i<num_gpus;i++) {
        // 在创建设备资源之前先设置激活的 CUDA 设备，确保 stream 在正确的 GPU 上创建
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }

    // ========================================================================
    // STEP 2: 初始化 NCCL Communicators
    // ========================================================================
    printf("Using ncclCommInitAll() to create all communicators simultaneously\n");
    // ncclCommInitAll() 创建所有 communicators 并处理内部协调
    // 参数:
    // - comms: 存储创建的 communicators 的数组
    // - num_gpus: 要创建的 communicators 数量
    // - devices: 要使用的 CUDA device IDs 数组
    // 调用后:
    // - comms[0] 将是 devices[0] 的 communicator, 且 rank 为 0
    // - comms[1] 将是 devices[1] 的 communicator, 且 rank 为 1
    // - ... 以此类推
    // - 所有 communicators 将有相同的 'size' (总参与者数)
    NCCLCHECK(ncclCommInitAll(comms, num_gpus, devices));
    printf("All %d NCCL communicators initialized successfully\n\n", num_gpus);

    // ========================================================================
    // STEP 3: 验证 Communicator 属性
    // ========================================================================
    printf("Communicator Details:\n");
    bool sizes_match = true;
    for (int i = 0;i<num_gpus;i++) {
        // 查询 communicator 以验证它是否正确设置
        int rank, size, device;
        // 获取这个 communicator 的 rank
        NCCLCHECK(ncclCommUserRank(comms[i], &rank));
        // 获取总参与者数
        NCCLCHECK(ncclCommCount(comms[i], &size));
        // 获取分配的 CUDA device
        NCCLCHECK(ncclCommCuDevice(comms[i], &device));
        printf("  Communicator %d: Rank %d/%d on CUDA device %d\n", i, rank, size, device);
        // 验证属性是否正确
        if (rank != i) {
            printf(" [WARNING: Communicator %d expected rank %d], actual rank %d\n", i, i, rank);
        }
        if (device != devices[i]) {
            printf(" [WARNING: Communicator %d expected device %d], actual device %d\n", i, devices[i], device);
        }
        if (size != num_gpus) {
            printf("WARNING: Communicator %d expected size %d, actual size %d\n", i, num_gpus, size);
            sizes_match = false;
        }
    }
    if (sizes_match) {
        printf("All communicators have the expected size of %d\n", num_gpus);
    }
    printf("\n");

    // ========================================================================
    // STEP 4: 清理资源
    // ========================================================================
    // 重要提示: 清理资源是 NCCL 应用的关键,资源必须以正确的顺序清理，以避免问题
    // 首先，同步所有流，确保没有操作在执行中，这可以防止在销毁资源时它们仍在使用
    printf("Synchronizing all CUDA streams...\n");
    for (int i = 0;i<num_gpus;i++) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    printf("All streams synchronized\n");

    // 接下来，先销毁 NCCL communicators，这必须在销毁 CUDA 资源之前完成，因为它们依赖于这些资源
    printf("Destroying NCCL communicators...\n");
    for (int i = 0;i<num_gpus;i++) {
        NCCLCHECK(ncclCommFinalize(comms[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }
    printf("All NCCL communicators destroyed\n");

    // 最后，销毁 CUDA streams，现在 communicators 已经销毁，这是安全的
    printf("Destroying CUDA streams...\n");
    for (int i = 0;i<num_gpus;i++) {
        CUDACHECK(cudaSetDevice(devices[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
    }
    printf("All CUDA streams destroyed\n");

    // 最后，释放主机内存分配
    free(devices);
    free(comms);
    free(streams);

    printf("\n=============================================================\n");
    printf("SUCCESS: Multiple devices single process example completed!\n");
    printf("=============================================================\n\n");

    return 0;
}
