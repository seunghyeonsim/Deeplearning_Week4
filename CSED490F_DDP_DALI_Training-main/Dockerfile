# File: Dockerfile
# FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

# Robust 설치: HTTPS + signed-by + 키링 + dirmngr
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg dirmngr lsb-release; \
    arch="$(dpkg --print-architecture)"; \
    ver="$(. /etc/os-release; echo ${VERSION_ID} | tr -d .)"; \
    # NVIDIA devtools repo GPG 키 (우선 nvidia.pub, 실패 시 7fa2af80.pub로 폴백)
    curl -fsSL "https://developer.download.nvidia.com/devtools/repos/ubuntu${ver}/${arch}/nvidia.pub" \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg \
      || curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub" \
      | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg; \
    # devtools APT repo 등록(HTTPS + signed-by)
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu${ver}/${arch}/ /" \
      > /etc/apt/sources.list.d/nvidia-devtools.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends nsight-systems-cli; \
    rm -rf /var/lib/apt/lists/*

# (권장) CUPTI 경로 보장
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# 설치 확인(권한 경고는 무시 가능)
RUN nsys --version && nsys status -e || true

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt