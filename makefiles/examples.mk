#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Make sure NCCL headers are found and libraries are linked
ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

# Build configuration
INCLUDES = -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include
LIBRARIES = -L$(NCCL_HOME)/lib -L$(CUDA_HOME)/lib64
LDFLAGS = -lcudart -lnccl -Wl,-rpath,$(NCCL_HOME)/lib


# MPI configuration
ifeq ($(MPI), 1)

ifdef MPI_HOME
MPICXX ?= $(MPI_HOME)/bin/mpicxx
MPIRUN ?= $(MPI_HOME)/bin/mpirun
else
MPICXX ?= mpicxx
MPIRUN ?= mpirun
endif

CXXFLAGS += -DMPI_SUPPORT
endif
