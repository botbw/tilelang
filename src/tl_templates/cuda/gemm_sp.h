// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once
#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
#include "gemm_sp_sm90.h"
#endif
