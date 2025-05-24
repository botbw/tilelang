#include <torch/extension.h>

#include <iostream>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

using namespace cute;

#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

// A matrix configuration
using ElementA = cutlass::half_t;  // Element type for A matrix operand
using LayoutTagA =
    cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<
              ElementA>::value;  // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB = cutlass::half_t;  // Element type for B matrix operand
using LayoutTagB =
    cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<
              ElementB>::value;  // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementC = float;  // Element type for C and D matrix operands
using LayoutTagC =
    cutlass::layout::ColumnMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<
              ElementC>::value;  // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator = float;  // Element type for internal accumulation
using TileShape =
    Shape<_128, _128, _128>;  // Threadblock-level tile size for sparse kernel
using TileShapeRef = Shape<_128, _128, _64>;  // Threadblock-level tile size for
                                              // reference (dense) kernel
using ClusterShape =
    Shape<_1, _2, _1>;  // Shape of the threadblocks in a cluster
using KernelSchedule =
    cutlass::gemm::collective::KernelScheduleAuto;  // Kernel schedule policy
using EpilogueSchedule =
    cutlass::epilogue::collective::EpilogueScheduleAuto;  // Epilogue schedule
                                                          // policy

using ProblemShape = Shape<int, int, int, int>;

// Sparse kernel setup

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassSparseTensorOp, TileShape,
        ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, ElementC, LayoutTagC,
        AlignmentC, ElementC, LayoutTagC, AlignmentC,
        EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassSparseTensorOp, ElementA,
        LayoutTagA, AlignmentA, ElementB, LayoutTagB, AlignmentB,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernel =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                         CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference (dense) kernel setup

using CollectiveEpilogueRef =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShapeRef,
        ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, ElementC, LayoutTagC,
        AlignmentC, ElementC, LayoutTagC, AlignmentC,
        EpilogueSchedule>::CollectiveOp;

using CollectiveMainloopRef =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA,
        LayoutTagA, AlignmentA, ElementB, LayoutTagB, AlignmentB,
        ElementAccumulator, TileShapeRef, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernelRef =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopRef,
                                         CollectiveEpilogue>;

using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

// Layouts
using LayoutA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutE = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// Layouts for reference (non-sparse) tensors
using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
using StrideE = StrideA;

using ElementE = typename Gemm::GemmKernel::CollectiveMainloop::ElementE;
using SparseConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;

// Offline compressor kernel
using CompressorUtility =
    cutlass::transform::kernel::StructuredSparseCompressorUtility<
        ProblemShape, ElementA, LayoutTagA, SparseConfig>;

using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
    ProblemShape, ElementA, LayoutTagA, SparseConfig, cutlass::arch::Sm90>;

using Compressor =
    cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;



template <typename T>
void print_class_name() {
#if defined(__clang__) || defined(__GNUC__)
    std::string func = __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
    std::string func = __FUNCSIG__;
#endif
    std::cout << "Function signature: " << func << std::endl;
}
std::tuple<torch::Tensor, torch::Tensor> compress_sm90(torch::Tensor A) {
  int M = A.size(0);
  int N = M;  // assume square matrix
  int K = A.size(1);
  int L = 1;
  ProblemShape problem_shape = make_tuple(M, N, K, L);
  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));

  CompressorUtility compressor_utility(problem_shape, stride_A);
  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int KC = compressor_utility.get_tensorA_k_physical();

  std::cout << "M: " << M << ", N: " << N << ", K: " << K << ", L: " << L
            << std::endl;
  std::cout << "ME: " << ME << ", KE: " << KE << ", KC: " << KC << std::endl;
  print_class_name<ElementE>();
  StrideE stride_E =
      cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(ME, KE, L));

  torch::Tensor A_compressed = torch::zeros(
      {M, KC}, torch::TensorOptions().dtype(torch::kHalf).device(A.device()));

  torch::Tensor E = torch::zeros(
      {ME, KE}, torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
  assert(A_compressed.size(0) == M);
  assert(A_compressed.size(1) == KC);
  assert(E.size(0) == ME);
  assert(E.size(1) == KE);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  typename Compressor::Arguments arguments{problem_shape,
                                           {
                                               A.data_ptr(),
                                               stride_A,
                                               A_compressed.data_ptr(),
                                               E.data_ptr(),
                                           },
                                           {hw_info}};

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(compressor_op.can_implement(arguments));
  CUTLASS_CHECK(compressor_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(compressor_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  return std::make_tuple(A_compressed, E);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compress_sm90", torch::wrap_pybind_function(compress_sm90),
        "compress_sm90");
}
