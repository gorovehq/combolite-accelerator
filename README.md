# combolite-accelerator

`combolite-accelerator` is the optional hardware acceleration library (`libcombolite_accel.so`) for the `combolite` database engine (spec section 18). It accelerates vector similarity search, Top-K shortlisting, and neighbour-graph refresh operations across GPU hardware using NVIDIA CUDA and Vulkan compute backends.

The accelerator is completely optional: when omitted or disabled at runtime, `combolite-core` falls back seamlessly to its internal, bit-exact C99 CPU execution paths.

---

## Architectural Principles

1. **Certified Bounds & Exact Verification**:
   - Floating-point calculations on GPU hardware differ slightly across vendor architectures due to FMA contraction and floating-point non-associativity.
   - The accelerator computes mathematically proven error bounds (`cax_bound`). When a shortlist's margin between the $K$-th and $(K+1)$-th result is wider than the bound, the GPU result is certified as exact. If a bound check fails, the query gracefully widens or falls back to the CPU without sacrificing correctness.
2. **Zero-Copy Host Integration**:
   - Interfaces through the `cdb_env_ops.accel` dispatch hook in `combolite-core`.
   - Engine buffers and vector mirrors are managed through caller-allocated memory limits (`accel_mem`).
3. **Dual Backend Support**:
   - **CUDA Backend**: High-performance kernels tuned for NVIDIA Turing, Ampere, Ada Lovelace, and Blackwell microarchitectures (sm_75, 86, 89, 120).
   - **Vulkan Backend**: Cross-platform compute shaders compiled to standard SPIR-V for Intel, AMD, and mobile GPUs supporting Vulkan 1.2+ with buffer device address extensions.

---

## Prerequisites

- **CUDA Backend**: NVIDIA CUDA Toolkit >= 11.8 (with `nvcc` and `cublas`).
- **Vulkan Backend**: Vulkan SDK 1.3+ (headers, Vulkan loader, and `glslc` shader compiler).
- **Core Engine**: `combolite-core` headers in sibling directory.

---

## Building

### Umbrella Presets (from Repository Root)

```sh
# Build core with both CUDA and Vulkan accelerators in Debug mode:
cmake --preset core-both-debug
cmake --build --preset core-both-debug -j

# Build standalone accelerator tests:
ctest --preset core-both-debug -L combolite-accelerator
```

### Standalone Build

```sh
cd combolite-accelerator
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCDB_ACCEL_CUDA=ON -DCDB_ACCEL_VULKAN=ON
cmake --build build -j
```

---

## Shader Generation (Vulkan)

Vulkan compute shaders in `vulkan/shaders/` are compiled into checked-in SPIR-V bytecode arrays in `vulkan/cav_spirv.c`. To regenerate after modifying GLSL shaders:

```sh
python3 tools/gen_spirv.py
```

To verify that the checked-in SPIR-V arrays match current shader sources:

```sh
python3 tools/gen_spirv.py --check
```

---

## Features as Built

- **Exact Vector Distance Kernels**: Dense float32 (cosine, L2 Euclidean, inner product), int8 quantized cosine, and 1-bit binary Hamming distance.
- **Top-K Selection & Reduction**: Parallel radix / bitonic reduction kernels isolating top-$K$ candidates on GPU.
- **On-Device Neighbour Graph Refresh**: Bulk distance matrix evaluation and candidate selection for graph maintenance on device (M25 Layer 4).
- **Runtime Backend Auto-Selection**: Automatically selects CUDA when an NVIDIA GPU is detected, falling back to Vulkan, and ultimately falling back to CPU.

---

## Known Issues

1. **Vulkan Driver Clock Drops During Sparse Bursts**: Desktop NVIDIA drivers drop to P8 memory states between isolated queries, causing brief latency increases on initial Vulkan invocations compared to persistent CUDA contexts.
2. **Euclidean Error Bounds**: Euclidean $L2$ distance bounds are more conservative than cosine bounds, occasionally triggering CPU fallback for tight distance clusters.

---

## ToDos

- Add ARM Mali and Apple Silicon (MoltenVK) validation suites.
- Implement tightened Euclidean error bounds based on vector difference norms.
- Add support for FP16 half-precision similarity scoring.

---

## Provisioned but Not Yet Implemented

1. **Approximate Search Pending-Pass GPU Acceleration (Spec 18.4)**:
   - The initial pass of approximate nearest neighbour search evaluates candidate vectors on GPU, but the final pending-list sweep is currently processed on the host CPU.
2. **Direct Studio GPU Acceleration**:
   - Combolite Studio currently interfaces only with `libcombolite.so`; direct integration with `libcombolite_accel.so` for UI layout physics is provisioned but not yet connected.

---

## License

MIT: See [`LICENSE`](../LICENSE) or root license.
