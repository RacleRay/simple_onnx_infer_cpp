
#include "common.h"
#include "my_memory.h"
#include "my_onnx_inference.h"

/**
 * @brief  init process
 * 
 * @param input_tensors_params  输入tensor信息对象，类型见common.h
 * @param output_tensors_params  输出tensor信息对象
 * @param input_tensors  输入tensor data对象
 * @param output_tensors  输出tensor data对象
 * @return result_t 
 */
result_t my_init_tensors(tensor_params_array_t *input_tensors_params,
                         tensor_params_array_t *output_tensors_params,
                         tensor_array_t **input_tensors,
                         tensor_array_t **output_tensors)
{
    MY_CHECK_NULL(input_tensors_params, MY_PARAM_NULL);
    MY_CHECK_NULL(output_tensors_params, MY_PARAM_NULL);
    MY_CHECK_NULL(input_tensors, MY_PARAM_NULL);
    MY_CHECK_NULL(output_tensors, MY_PARAM_NULL);

    result_t res = MY_SUCCESS;

    //分配输入tensors 内存
    res = alloc_tensor_arry(input_tensors_params, input_tensors);
    if (MY_SUCCESS != res)
    {
        MY_ERROR("alloc input tensors  error!\n");
    }

    //分配输出tensors 内存
    res = alloc_tensor_arry(output_tensors_params, output_tensors);
    if (MY_SUCCESS != res)
    {
        MY_ERROR("alloc output tensors  error!\n");
    }

    return res;
}

/**
 * @brief  release resources
 * 
 * @param input_tensors  输入tensor data对象
 * @param output_tensors  输出tensor data对象
 * @return result_t 
 */
result_t my_deinit_tensors(tensor_array_t *input_tensors, tensor_array_t *output_tensors)
{
    MY_CHECK_NULL(input_tensors, MY_PARAM_NULL);
    MY_CHECK_NULL(output_tensors, MY_PARAM_NULL);

    result_t res = MY_SUCCESS;

    res = release_tensor_arry(input_tensors);
    if (MY_SUCCESS != res)
    {
        MY_ERROR("release input tensor array error!\n");
    }

    res = release_tensor_arry(output_tensors);
    if (MY_SUCCESS != res)
    {
        MY_ERROR("release output tensor array error!\n");
    }

    return res;
}

/**
 * @brief  load_model_handle model
 * 
 * @param load_model_param  GPU、推理引擎设置等
 * @param input_tensors  输入tensor data对象
 * @param output_tensors   输出tensor data对象
 * @param load_model_handle  模型句柄，只有一个指针成员
 * @return result_t 
 */
result_t my_load_model(model_params_t *load_model_param,
                       tensor_array_t *input_tensors,
                       tensor_array_t *output_tensors,
                       model_handle_t *load_model_handle)
{

    OnnxRuntimeModelHandle *pOnnxHdl = new OnnxRuntimeModelHandle(load_model_param);
    pOnnxHdl->set_input_tensor_array(input_tensors);
    pOnnxHdl->set_output_tensor_array(output_tensors);
    load_model_handle->model_handle = pOnnxHdl;
    pOnnxHdl->my_onnxruntime_open_model();
    return MY_SUCCESS;
}

/**
 * @brief  release resources
 * 
 * @param load_model_handle  模型句柄
 * @return result_t 
 */
result_t my_release_model(model_handle_t *load_model_handle)
{
    OnnxRuntimeModelHandle *pOnnxHdl = (OnnxRuntimeModelHandle *)load_model_handle->model_handle;
    pOnnxHdl->my_onnxruntime_release_model();
    return MY_SUCCESS;
}

/**
 * @brief  run inference. Results are stored in output_tensors object.
 * 
 * @param load_model_handle  模型句柄
 * @return result_t 
 */
result_t my_inference_tensors(model_handle_t *load_model_handle)
{
    OnnxRuntimeModelHandle *pOnnxHdl = (OnnxRuntimeModelHandle *)load_model_handle->model_handle;
    pOnnxHdl->my_onnxruntime_inference_tensors();
    return MY_SUCCESS;
}
