cmake_minimum_required(VERSION 3.0.0)
project(main)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



if(EMSCRIPTEN)
    add_executable(main main_batch_wasm_port.cpp)

    # opencv
    set(OPENCV_PATH /opencv_wasm_simd_install)
    include_directories(${OPENCV_PATH}/include/opencv4)
    set(OPENCV_PATH_LIB ${OPENCV_PATH}/lib)
    target_link_libraries(main ${OPENCV_PATH_LIB}/libopencv_core.a)    
    target_link_libraries(main ${OPENCV_PATH_LIB}/libopencv_imgproc.a)    


    # onnxruntime
    target_include_directories(main PRIVATE "/onnxruntime/include/onnxruntime/core/session/")
    target_link_libraries(main "/onnxruntime/build/Linux/Release/libonnxruntime_webassembly.a")

    # 9 mb
    # -fexceptions 
    set(flags "-Oz -flto -msimd128")
    set_target_properties(${prj_name} PROPERTIES COMPILE_FLAGS ${flags})
    set_target_properties(main PROPERTIES LINK_FLAGS "${flags} -s ERROR_ON_UNDEFINED_SYMBOLS=0 -s ALLOW_MEMORY_GROWTH=1 --bind --export-all --preload-file model.onnx@model.onnx")

else()
    find_package(OpenCV REQUIRED)
    set(ONNXRUNTIME_DIR /onnxruntime_pack)

    add_executable(main_single main_single.cpp)
    target_include_directories(main_single PRIVATE "/onnxruntime/include/onnxruntime/core/session/")
    target_link_libraries(main_single ${OpenCV_LIBS})
    target_link_libraries(main_single "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.12.0")


    add_executable(main_batch main_batch.cpp)
    target_include_directories(main_batch PRIVATE "${ONNXRUNTIME_DIR}/include")
    target_link_libraries(main_batch ${OpenCV_LIBS})
    target_link_libraries(main_batch "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.12.0")

endif()