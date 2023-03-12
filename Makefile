# docker build -f ./docker/Dockerfile  -t onnx_wasm_example .

# for linux
# docker run --rm -it -p 8080:8080 -v $(pwd):/develop onnx_wasm_example bash

# for windows
# docker run --rm -it -p 8080:8080 -v %cd%:/develop onnx_wasm_example bash

# make

# open 


SHELL := /bin/bash


run: build
	cd build && http-server

build: clean
	mkdir build
	cp -r ./staff/* ./build
	source /emsdk/emsdk_env.sh && cd build && emcmake cmake ../src && emmake make

clean:
	rm -rf build
