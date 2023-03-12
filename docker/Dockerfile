FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

# docker build -f ./docker/Dockerfile  -t onnx_wasm_example .

# for linux
# docker run --rm -it -p 8080:8080 -v $(pwd):/develop onnx_wasm_example bash

# for windows
# docker run --rm -it -p 8080:8080 -v %cd%:/develop onnx_wasm_example bash

# make


# base things
RUN apt-get update --fix-missing
RUN apt-get install -y git make cmake gcc python3 build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget


# emsdk for opencv and whole project
WORKDIR /
RUN git clone https://github.com/emscripten-core/emsdk.git
WORKDIR /emsdk
RUN ./emsdk install 3.1.3
RUN ./emsdk activate 3.1.3
RUN bash ./emsdk_env.sh
# source /emsdk/emsdk_env.sh

# opencv
WORKDIR /
RUN git clone --branch 4.7.0 --single-branch https://github.com/opencv/opencv.git

WORKDIR /
# uses 3.1.3 emsdk
RUN sed -i 's/DBUILD_opencv_imgcodecs=OFF/DBUILD_opencv_imgcodecs=ON/g' /opencv/platforms/js/build_js.py
RUN sed -i 's#DCMAKE_INSTALL_PREFIX=/usr/local#DCMAKE_INSTALL_PREFIX=/opencv_wasm_simd_install#g' /opencv/platforms/js/build_js.py
RUN sed -i 's/DWITH_PNG =OFF/DWITH_PNG =ON/g' /opencv/platforms/js/build_js.py 
RUN python3 /opencv/platforms/js/build_js.py /opencv_wasm --build_wasm --threads --simd --build_flags="-Oz -flto" --emscripten_dir /emsdk/upstream/emscripten
WORKDIR /opencv_wasm/modules/imgcodecs
RUN make -j16
WORKDIR /opencv_wasm
RUN make install


# onnxruntime
WORKDIR /
RUN git clone --recursive --branch v1.12.1 --single-branch https://github.com/Microsoft/onnxruntime
WORKDIR /onnxruntime
# uses 3.1.3 emsdk

# fix cmake version - push it upper
# https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
# RUN apt purge --auto-remove -y cmake
# RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
# RUN apt-get install -y software-properties-common && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-get update && apt-get -y install cmake
RUN apt-get install -y nodejs

RUN ./build.sh --config Release --build_wasm_static_lib --enable_wasm_simd --enable_wasm_threads --skip_tests
# --parallel
# RUN ./build.sh --config Release --build_wasm_static_lib

RUN apt-get install -y python3 python3-pip
# RUN pip3 install onnx==1.12.1 onnxruntime==1.12.1

# make cute packs of libs
WORKDIR  /develop
RUN apt-get install -y npm && npm install -g http-server

# fix nodejs 10
RUN apt-get install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_current.x | bash
# RUN apt-get install -y nodejs


# build the project

