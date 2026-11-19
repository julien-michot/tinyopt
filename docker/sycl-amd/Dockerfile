FROM ghcr.io/prefix-dev/pixi:0.81.0 AS pixi

FROM rocm/dev-ubuntu-22.04:6.3.2-complete
ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/adaptivecpp/bin:${PATH}

COPY --from=pixi /usr/local/bin/pixi /usr/local/bin/pixi

RUN apt-get update && apt-get install -y --no-install-recommends \
      libboost-all-dev \
      ninja-build \
      python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN pixi install --frozen -e bench-sycl-amd
ADD https://github.com/AdaptiveCpp/AdaptiveCpp/archive/refs/tags/v25.02.0.tar.gz /tmp/AdaptiveCpp.tar.gz
RUN mkdir -p /tmp/AdaptiveCpp \
    && tar -xzf /tmp/AdaptiveCpp.tar.gz --strip-components=1 -C /tmp/AdaptiveCpp \
    && pixi run -e bench-sycl-amd cmake -S /tmp/AdaptiveCpp -B /tmp/AdaptiveCpp-build -G Ninja \
      -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_INSTALL_PREFIX=/opt/adaptivecpp \
      -DLLVM_DIR=/opt/rocm/llvm/lib/cmake/llvm \
      -DClang_DIR=/opt/rocm/llvm/lib/cmake/clang \
      -DROCM_PATH="${ROCM_PATH}" \
      -DWITH_ROCM_BACKEND=ON \
    && pixi run -e bench-sycl-amd cmake --build /tmp/AdaptiveCpp-build --parallel 2 \
    && pixi run -e bench-sycl-amd cmake --install /tmp/AdaptiveCpp-build \
    && rm -rf /tmp/AdaptiveCpp /tmp/AdaptiveCpp-build

CMD ["bash"]
