FROM ghcr.io/prefix-dev/pixi:0.81.0 AS pixi

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

COPY --from=pixi /usr/local/bin/pixi /usr/local/bin/pixi

WORKDIR /workspace
COPY . .

RUN pixi install --frozen -e bench-sycl-cuda \
    && ACPP_CUDA_CONFIG=/workspace/.pixi/envs/bench-sycl-cuda/etc/AdaptiveCpp/acpp-cuda.json \
    && ACPP_CUDA_ROOT="$(sed -n 's/.*"default-cuda-path" *: *"\([^"]*\)".*/\1/p' \
      /workspace/.pixi/envs/bench-sycl-cuda/etc/AdaptiveCpp/acpp-cuda.json)" \
    && test -n "${ACPP_CUDA_ROOT}" \
    && test -d /workspace/.pixi/envs/bench-sycl-cuda/nvvm/libdevice \
    && sed -i "s|${ACPP_CUDA_ROOT}|/workspace/.pixi/envs/bench-sycl-cuda|g" "${ACPP_CUDA_CONFIG}" \
    && mkdir -p "${ACPP_CUDA_ROOT}/nvvm" \
    && ln -s /workspace/.pixi/envs/bench-sycl-cuda/nvvm/libdevice \
      "${ACPP_CUDA_ROOT}/nvvm/libdevice" \
    && mkdir -p /workspace/.pixi/envs/bench-sycl-cuda/lib/lib \
    && ln -s /workspace/.pixi/envs/bench-sycl-cuda/lib/hipSYCL \
      /workspace/.pixi/envs/bench-sycl-cuda/lib/lib/hipSYCL

CMD ["bash"]
