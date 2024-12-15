# syntax=docker/dockerfile:1.2

# -----------------------------------------------------
# DOCKER BUILD SHOULD BE RUN FROM THE PARENT DIRECTORY!
# docker build -f simulator-rust/Dockerfile -t simulator .
# -----------------------------------------------------
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
USER root
WORKDIR /root

RUN \
  apt-get update && \
  apt-get install -y wget && \
  apt-get install -y curl && \
  curl --proto '=https' --t\lsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y && \
  . "$HOME/.cargo/env" && \
  cargo install bindgen-cli && \
  apt-get install -y python3 && \
  apt-get install -y python3-pip  && \
  pip install meson && \
  apt-get install -y lsb-release && \
  apt-get install -y software-properties-common && \
  apt-get install -y gnupg && \
  wget -qO- https://apt.llvm.org/llvm.sh | bash -s -- 19 && \
  apt-get install -y ninja-build && \
  apt-get install -y libssl-dev && \
  apt-get install -y protobuf-compiler && \
  apt-get install -y pkg-config

# This is where the libsimulator-cuda.so ends up, but is not picked up by
# by the linker without adding to the path.
ENV LD_LIBRARY_PATH="/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Need to copy rust as well for cuda compilation because bindgen will write a 
# file into simulator-rust.
COPY simulator-rust /root/simulator-rust
COPY simulator-cuda /root/simulator-cuda

WORKDIR /root/simulator-cuda
RUN meson setup buildrelease --buildtype=release -Dtest_mode=false && \
  cd buildrelease && \
  ninja install

WORKDIR /root/simulator-rust
RUN \
  . "$HOME/.cargo/env" && \ 
  cargo build --release

ENTRYPOINT /root/simulator-rust/target/release/simulator serve

