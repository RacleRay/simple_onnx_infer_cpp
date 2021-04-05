#include <iostream>
#include "my_utils.h"

unsigned int ElementSize(tensor_types_t t)
{
    size_t nDataSize = 0;
    switch (t)
    {
    case DT_FLOAT:
        nDataSize = sizeof(float);
        break;
    case DT_DOUBLE:
        nDataSize = sizeof(double);
        break;
    case DT_INT32:
        nDataSize = sizeof(int);
        break;
    case DT_UINT8:
        nDataSize = sizeof(my_u8);
        break;
    case DT_STRING:
        nDataSize = sizeof(char);
        break;
    case DT_INT64:
        nDataSize = sizeof(long long);
        break;
    case DT_BOOL:
        nDataSize = sizeof(bool);
        break;
    default:
        nDataSize = 1;
        break;
    }

    return nDataSize;
}

/**
 * @brief Get the Tensor Size and Shape of input object
 *
 * @param cur_tensor tensor object
 */
void GetTensorSize(tensor_t *cur_tensor)
{
    tensor_params_t *cur_tensor_param = cur_tensor->pTensorInfo;

    unsigned int nDataSize = ElementSize(cur_tensor_param->type);

    cur_tensor_param->nElementSize = 1;

    MY_DEBUG("cur_tensor[%s]->nDims:  %d\n", cur_tensor_param->aTensorName, cur_tensor_param->nDims);

    for (int j = 0; j < cur_tensor_param->nDims; ++j)
    {
        if (cur_tensor_param->pShape[j] < 0)
        {
            std::cout << "tensor :" << cur_tensor_param->aTensorName << "shape [" << j << "]"
                      << "should be  > 0 !!!!" << std::endl;
            exit(-1);
        }

        MY_DEBUG("cur_tensor[%s]->shape[%d]:  %d\n", cur_tensor_param->aTensorName, j, cur_tensor_param->pShape[j]);

        cur_tensor_param->nElementSize *= cur_tensor_param->pShape[j];
    }

    cur_tensor_param->nLength = cur_tensor_param->nElementSize * nDataSize;

    MY_DEBUG("cur_tensor->nValueLen:  %d\n", cur_tensor_param->nLength);
}
