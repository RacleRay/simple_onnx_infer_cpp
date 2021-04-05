

#ifndef MY_INFERENCE_ONNX_MY_MEMORY_H
#define MY_INFERENCE_ONNX_MY_MEMORY_H
#include "common.h"

result_t alloc_tensor_arry(tensor_params_array_t *tensor_params_array, tensor_array_t **tensor_array);

result_t release_tensor_arry(tensor_array_t *tensor_array);

#endif //MY_INFERENCE_ONNX_MY_MEMORY_H
