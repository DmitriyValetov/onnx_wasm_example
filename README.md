An example of onnxruntime & opencv use in browser via WEBASSEMBLY (WASM).

To run this:

    1) build a docker image: 

        docker build -f ./docker/Dockerfile  -t onnx_wasm_example .

    2) run docker:

        docker run --rm -it -p 8080:8080 -v $(pwd):/develop onnx_wasm_example bash

        or 

        docker run --rm -it -p 8080:8080 -v %cd%:/develop onnx_wasm_example bash

    3) run make:

        make

    4) open browser at http://127.0.0.1:8080 and choose a demo or a speed test