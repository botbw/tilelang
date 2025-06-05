#include <iostream>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

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

using ElementA = cutlass::half_t;
using ElementE = unsigned char;
using LayoutTagA = cutlass::layout::RowMajor;

using ProblemShape = Shape<int, int, int, int>;

using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
using StrideE = StrideA;

using SparseConfig = cutlass::Sm90GemmSparseConfig<
    cute::sparse_elem<2, ElementA>, cute::SM90::GMMA::Major::K,
    cute::sparse_elem<8, ElementE>, cute::C<32> >;

using CompressorUtility =
    cutlass::transform::kernel::StructuredSparseCompressorUtility<
        ProblemShape, ElementA, LayoutTagA, SparseConfig>;

using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
    ProblemShape, ElementA, LayoutTagA, SparseConfig, cutlass::arch::Sm90>;

using Compressor =
    cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

template <typename T>
void print_type_name() {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
}

int main() {
//   SparseConfig config;
//   print_type_name<SparseConfig>();
  print_latex(SparseConfig::TensorEAtom_16bit{});
  return 0;
}