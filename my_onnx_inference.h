#ifndef MY_INFERENCE_ONNX_MY_ONNX_INFERENCE_H
#define MY_INFERENCE_ONNX_MY_ONNX_INFERENCE_H
#include <vector>
#include <memory>
#include <mutex>
#include "common.h"
#include "onnxruntime/onnxruntime_c_api.h"
#include "onnxruntime/cuda_provider_factory.h"

#define USE_TRT

#ifdef USE_TRT
#include "onnxruntime/tensorrt_provider_factory.h"
#endif

class OnnxRuntimeModelHandle
{
public:
    OnnxRuntimeModelHandle(model_params_t *tModelParam);
    ~OnnxRuntimeModelHandle();
    result_t my_onnxruntime_open_model();
    result_t my_onnxruntime_inference_tensors();
    result_t my_onnxruntime_release_model();
    void set_input_tensor_array(tensor_array_t *input_tensor_array);
    void set_output_tensor_array(tensor_array_t *ouput_tensor_array);

private:
    void GetModelInfo();
    void CheckStatus(OrtStatus *status);
    inline bool FindNameInTensorNames(const char *cur_name, std::vector<const char *> &node_names);

private:
    model_params_t *m_tModelParam;
    tensor_array_t *m_input_tensor_array;
    tensor_array_t *m_ouput_tensor_array;
    OrtSessionOptions *m_pSessionOptions;
    OrtSession *m_pSession;

    std::vector<const char *> m_vecInputNodesName;
    std::vector<ONNXTensorElementDataType> m_vecInputNodesType;
    std::vector<std::vector<int64_t>> m_vecInputNodesDims;

    std::vector<const char *> m_vecOutputNodesName;
    std::vector<ONNXTensorElementDataType> m_vecOutputNodesType;
    std::vector<std::vector<int64_t>> m_vecOutputNodesDims;
    std::mutex m_onnx_mutex;
};

#endif //MY_INFERENCE_ONNX_MY_ONNX_INFERENCE_H
