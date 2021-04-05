#include <cstring>
#include "my_utils.h"
#include "my_memory.h"

/**
 * @brief 分配tensor所需的内存
 *
 * @param tensor_params_array 名称、类型、shape等
 * @param tensor_array tensor info和tensor数据
 * @return result_t 返回执行结果状态码
 */
result_t alloc_tensor_arry(tensor_params_array_t *tensor_params_array, tensor_array_t **tensor_array) {
    MY_CHECK_NULL(tensor_params_array, MY_PARAM_NULL);
    MY_CHECK_NULL(tensor_array, MY_PARAM_NULL);

    tensor_array_t *ptTensorArray = new tensor_array_t();
    MY_CHECK_NULL(ptTensorArray, MY_TENSOR_ALLOC_FAILED);

    ptTensorArray->nArraySize = tensor_params_array->nArraySize;
    ptTensorArray->pTensorArray = new tensor_t[ptTensorArray->nArraySize];
    MY_CHECK_NULL(ptTensorArray->pTensorArray, MY_TENSOR_ALLOC_FAILED);

    for (int(i) = 0; (i) < ptTensorArray->nArraySize; ++(i)) {
        tensor_params_t *cur_tensor_param = &(tensor_params_array->pTensorParamArray[i]);
        tensor_t *cur_tensor = &(ptTensorArray->pTensorArray[i]);

        cur_tensor->pTensorInfo = new tensor_params_t();
        MY_CHECK_NULL(cur_tensor->pTensorInfo, MY_TENSOR_ALLOC_FAILED);
        memcpy(cur_tensor->pTensorInfo, cur_tensor_param, sizeof(tensor_params_t));

        GetTensorSize(cur_tensor);

       // MY_DEBUG("alloced tensor %s memory length: %d\n", cur_tensor_param->aTensorName, cur_tensor->pTensorInfo->nLength);

        cur_tensor->pValue = new my_u8[cur_tensor->pTensorInfo->nLength];
    }

    // strcpy(ptTensorArray->pcSignatureDef, tensor_params_array->pcSignatureDef);  // tensorflow specific
    *tensor_array = ptTensorArray;

    return MY_SUCCESS;
}

/**
 * @brief 释放tensor占用空间
 * 
 * @param tensor_array array of tensors
 * @return result_t 返回执行结果状态码
 */
result_t release_tensor_arry(tensor_array_t *tensor_array) {
    MY_CHECK_NULL(tensor_array, MY_PARAM_NULL);

    for (int i = 0; i < tensor_array->nArraySize; ++i) {
        tensor_t *cur_tensor = &(tensor_array->pTensorArray[i]);
        tensor_params_t *cur_tensor_params = cur_tensor->pTensorInfo;

        if (cur_tensor->pValue != nullptr) {
            delete[]((my_u8 *) (cur_tensor->pValue));
            cur_tensor->pValue = nullptr;
        }

        if (cur_tensor->pTensorInfo) {
            delete cur_tensor->pTensorInfo;
            cur_tensor->pTensorInfo = nullptr;
        }
    }

    if (tensor_array->pTensorArray) {
        delete[] tensor_array->pTensorArray;
        tensor_array->pTensorArray = nullptr;
    }
    if (tensor_array) {
        delete tensor_array;
        tensor_array = nullptr;
    }

    return MY_SUCCESS;
}


