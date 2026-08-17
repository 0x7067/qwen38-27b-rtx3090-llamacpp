FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

ARG LLAMA_CPP_REF=4df29be4f4c3673f428170fda944a5b19f743bb8
# llama-swap: OpenAI-compatible proxy that hot-swaps llama-server backends so a
# single GPU can serve multiple models (one resident at a time).
ARG LLAMA_SWAP_VERSION=v230

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libcurl4-openssl-dev \
      libopenblas-dev \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
    && git checkout "${LLAMA_CPP_REF}"

# Vendored performance patches over the pinned ref (see patches/*.patch):
# 0001: DFlash draft-side greedy fast path + LLAMA_SPEC_PROF instrumentation
# 0002: Ampere MMQ small-batch (J=16) tile config — 128x64 tiles for Q4_K/Q5_K
COPY patches/ /src/llama.cpp/patches/
RUN git apply --stat --apply patches/*.patch

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=ON \
      -DGGML_OPENMP=ON \
      -DGGML_BLAS=ON \
      -DGGML_BLAS_VENDOR=OpenBLAS \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DLLAMA_BUILD_TESTS=ON \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64/stubs" \
    && cmake --build build --target llama-server llama-bench llama-perplexity test-backend-ops -j"$(nproc)"

# Fetch the llama-swap release binary (static Go binary, linux/amd64).
RUN curl -fL "https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION#v}_linux_amd64.tar.gz" \
      -o /tmp/llama-swap.tar.gz \
    && tar -xzf /tmp/llama-swap.tar.gz -C /usr/local/bin llama-swap \
    && chmod +x /usr/local/bin/llama-swap

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      libcurl4 \
      libgomp1 \
      libopenblas0-pthread \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/llama.cpp/build/bin/llama-bench /usr/local/bin/llama-bench
# Kernel-development harnesses: per-op CUDA-vs-CPU correctness/perf, and the
# KL-divergence quality gate used to validate hand-written kernels.
COPY --from=build /src/llama.cpp/build/bin/llama-perplexity /usr/local/bin/llama-perplexity
COPY --from=build /src/llama.cpp/build/bin/test-backend-ops /usr/local/bin/test-backend-ops
COPY --from=build /src/llama.cpp/build/bin/*.so /usr/local/lib/
COPY --from=build /usr/local/bin/llama-swap /usr/local/bin/llama-swap

RUN ldconfig

WORKDIR /app
EXPOSE 8080
