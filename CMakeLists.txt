cmake_minimum_required(VERSION 3.15)
project(my_inference_onnx)

set(CMAKE_CXX_STANDARD 11)


SET(INC_DIR  ./include)
SET(LIB_DIR  ./lib/cudnn ./lib/onnxruntime  ./lib/opencv  ./lib/trt /usr/local/cuda/lib64)

set(TRT_LIBS nvinfer nvinfer_plugin )
set(CUDNN_LIB cudnn )
set(CUDA_LIBS cublas cudart curand cufft)
set(OPENCV_LIBS opencv_world)

set(LINK_LIBS ${TRT_LIBS} ${CUDNN_LIB} ${CUDA_LIBS} ${OPENCV_LIBS})

include_directories(${INC_DIR})
link_directories(${LIB_DIR})
link_libraries(${LINK_LIBS})

add_library(my_inference_onnx
        SHARED
        common.h
        my_interface.h
        my_onnx_inference.cpp
        my_onnx_inference.h
        my_interface.cpp
        aes.h
        aes.cpp my_memory.h my_memory.cpp my_utils.h my_utils.cpp)

target_link_libraries(my_inference_onnx ${LINK_LIBS} )