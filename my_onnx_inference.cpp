#include "my_onnx_inference.h"

#include <iostream>
#include <chrono>
#include <string>
#include <unistd.h>
#include <assert.h>
#include "aes.h"
#include "my_utils.h"

static const OrtApi *g_pOrt = OrtGetApiBase()->GetApi(ORT_API_VERSION); // global api manager
static OrtEnv *g_pEnv = nullptr;
static int g_pEnv_ref_count = 0;

/**
 * @brief get saved model dir
 *
 * @return path string
 */
static std::string getCurrentModelDir()
{
    char buffer[256];
    getcwd(buffer, 256);
    std::string pDir = buffer;
    std::cout << "Original dir: " << pDir << std::endl;
    std::string myPath = pDir;
    return myPath;
}

/**
 * @brief get runtime env and load encrypted model
 * 
 * @return result_t 
 */
result_t OnnxRuntimeModelHandle::my_onnxruntime_open_model()
{
    // 线程安全
    m_onnx_mutex.lock();

    if (g_pEnv == nullptr)  // 主线程中初始化OnnxRuntime Env
    {
        CheckStatus(g_pOrt->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntime", &g_pEnv));
        std::cout << "Create  env =========" << std::endl;
    }
    g_pEnv_ref_count++;

    // session option
    CheckStatus(g_pOrt->CreateSessionOptions(&m_pSessionOptions));
    g_pOrt->SetIntraOpNumThreads(m_pSessionOptions, 1);

    // Sets graph optimization level.  For TensorRT
    GraphOptimizationLevel optmizeLevel = (GraphOptimizationLevel)m_tModelParam->model_optimize_level;
    g_pOrt->SetSessionGraphOptimizationLevel(m_pSessionOptions, optmizeLevel);

    if (m_tModelParam->cpu_or_gpu == 1)  // GPU or CPU
    {
#ifdef USE_TRT
        if (optmizeLevel > ORT_DISABLE_ALL)
        {
            OrtSessionOptionsAppendExecutionProvider_Tensorrt(m_pSessionOptions, m_tModelParam->gpu_id);
        }
        else
        {
#endif
            OrtSessionOptionsAppendExecutionProvider_CUDA(m_pSessionOptions, m_tModelParam->gpu_id);

#ifdef USE_TRT
        }
#endif
    }

    std::string strModelAbsolutePath = getCurrentModelDir() + "/" + m_tModelParam->model_path;
    if (m_tModelParam->model_path[0] == '/')  // absolute path
    {
        strModelAbsolutePath = std::string(m_tModelParam->model_path);
    }

    // 解密加载模型
    if (m_tModelParam->bIsCipher)
    {
        int encStartPoint = m_tModelParam->encStartPoint / 16;
        int encLength = m_tModelParam->encLength / 16;

        std::string strOutFileContent = my_onnx::DecryptionModelPartial(strModelAbsolutePath, encStartPoint, encLength);

        CheckStatus(g_pOrt->CreateSessionFromArray(g_pEnv, strOutFileContent.c_str(), strOutFileContent.size(),
                                                   m_pSessionOptions, &m_pSession));
    }
    else
    {
        std::cout << "Begin to load onnx model  " << strModelAbsolutePath << std::endl;
        CheckStatus(g_pOrt->CreateSession(g_pEnv, strModelAbsolutePath.c_str(), m_pSessionOptions, &m_pSession));
    }

    m_onnx_mutex.unlock();

    GetModelInfo();

    std::cout << "Load onnx model  " << strModelAbsolutePath << "  succeed!!" << std::endl;

    return MY_SUCCESS;
}

/**
 * @brief release onnxruntime resources with mutex lock
 * 
 * @return result_t 
 */
result_t OnnxRuntimeModelHandle::my_onnxruntime_release_model()
{
    m_onnx_mutex.lock();

    if (m_pSession)
    {
        g_pOrt->ReleaseSession(m_pSession);
        m_pSession = nullptr;
    }

    if (m_pSessionOptions)
    {
        g_pOrt->ReleaseSessionOptions(m_pSessionOptions);
        m_pSessionOptions = nullptr;
    }

    g_pEnv_ref_count--;
    if (g_pEnv_ref_count <= 0)
    {
        g_pOrt->ReleaseEnv(g_pEnv);
        g_pEnv = nullptr;
    }

    m_onnx_mutex.unlock();

    return MY_SUCCESS;
}

/**
 * @brief check onnxruntime error and show in stderr
 * 
 * @param status 
 */
void OnnxRuntimeModelHandle::CheckStatus(OrtStatus *status)
{
    if (status != NULL)
    {
        const char *msg = g_pOrt->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_pOrt->ReleaseStatus(status);
        exit(1);
    }
}

/**
 * @brief 加载模型后，保存和打印模型信息
 * 
 */
void OnnxRuntimeModelHandle::GetModelInfo()
{
    size_t num_input_nodes, num_output_nodes;
    OrtStatus *status;
    OrtAllocator *allocator;
    CheckStatus(g_pOrt->GetAllocatorWithDefaultOptions(&allocator));

    /*===================== get input  nodes information =====================*/
    // print number of model input nodes
    status = g_pOrt->SessionGetInputCount(m_pSession, &num_input_nodes);
    printf("Number of inputs = %zu\n", num_input_nodes);

    m_vecInputNodesName.resize(num_input_nodes);

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++)
    {
        // print input node names
        char *input_name;
        status = g_pOrt->SessionGetInputName(m_pSession, i, allocator, &input_name);
        m_vecInputNodesName[i] = input_name;
        printf("Input %zu : name=%s\n", i, input_name);

        // print input node types
        OrtTypeInfo *typeinfo;
        status = g_pOrt->SessionGetInputTypeInfo(m_pSession, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo *tensor_info;
        CheckStatus(g_pOrt->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_pOrt->GetTensorElementType(tensor_info, &type));
        m_vecInputNodesType.push_back(type);
        printf("Input %zu : type=%d\n", i, type);

        // print input shapes/dims
        size_t num_dims;
        std::vector<int64_t> cur_node_dims;
        CheckStatus(g_pOrt->GetDimensionsCount(tensor_info, &num_dims));
        printf("Input %zu : num_dims=%zu\n", i, num_dims);

        cur_node_dims.resize(num_dims);
        g_pOrt->GetDimensions(tensor_info, (int64_t *)cur_node_dims.data(), num_dims);
        for (size_t j = 0; j < num_dims; j++)
            printf("Input %zu : dim %zu=%jd\n", i, j, cur_node_dims[j]);

        m_vecInputNodesDims.push_back(cur_node_dims);
        g_pOrt->ReleaseTypeInfo(typeinfo);
    }

    /*===================== get output  nodes information =====================*/
    status = g_pOrt->SessionGetOutputCount(m_pSession, &num_output_nodes);
    printf("\nNumber of outputs = %zu\n", num_output_nodes);

    m_vecOutputNodesName.resize(num_output_nodes);

    // iterate over all output nodes
    for (size_t i = 0; i < num_output_nodes; i++)
    {
        // print output node names
        char *output_name;
        status = g_pOrt->SessionGetOutputName(m_pSession, i, allocator, &output_name);
        printf("Output %zu : name=%s\n", i, output_name);
        m_vecOutputNodesName[i] = output_name;

        // print output node types
        OrtTypeInfo *typeinfo;
        status = g_pOrt->SessionGetOutputTypeInfo(m_pSession, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo *tensor_info;
        CheckStatus(g_pOrt->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_pOrt->GetTensorElementType(tensor_info, &type));
        m_vecOutputNodesType.push_back(type);
        printf("Output %zu : type=%d\n", i, type);

        // print input shapes/dims
        size_t num_dims;
        std::vector<int64_t> cur_node_dims;
        CheckStatus(g_pOrt->GetDimensionsCount(tensor_info, &num_dims));
        printf("Input %zu : num_dims=%zu\n", i, num_dims);

        cur_node_dims.resize(num_dims);
        g_pOrt->GetDimensions(tensor_info, (int64_t *)cur_node_dims.data(), num_dims);
        for (size_t j = 0; j < num_dims; j++)
            printf("Output %zu : dim %zu=%jd\n", i, j, cur_node_dims[j]);

        m_vecOutputNodesDims.push_back(cur_node_dims);
        g_pOrt->ReleaseTypeInfo(typeinfo);
    }
}

/**
 * @brief find specific tensor by name in a node
 * 
 * @param cur_name  name to find
 * @param node_names  node with some tensors
 * @return bool
 */
bool OnnxRuntimeModelHandle::FindNameInTensorNames(const char *cur_name, std::vector<const char *> &node_names)
{
    for (size_t i = 0; i < node_names.size(); i++)
    {
        if (strcmp(cur_name, node_names[i]) == 0)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief main inference process
 * 
 * @return result_t 
 */
result_t OnnxRuntimeModelHandle::my_onnxruntime_inference_tensors()
{
    m_onnx_mutex.lock();

    /*===================== process input tensor =====================*/
    MY_DEBUG("Begin onnx inference tensors!\n");
    assert(m_input_tensor_array->nArraySize == m_vecInputNodesType.size());

    // OrtValue *input_tensors[1];
    std::vector<OrtValue *> input_tensors(m_input_tensor_array->nArraySize);
    OrtValue *input_tensor = NULL;

    for (int i = 0; i < m_input_tensor_array->nArraySize; i++)
    {
        tensor_t *cur_tensor = &(m_input_tensor_array->pTensorArray[i]);
        tensor_params_t *cur_tensor_param = cur_tensor->pTensorInfo;

        // Check shape of inputs
        for (int j = 0; j < cur_tensor_param->nDims; j++)
        {
            if (cur_tensor_param->pShape[j] <= 0)
            {
                std::cout << "Tensor " << cur_tensor_param->aTensorName << " shape[ " << j << " ] should be  > 0" << std::endl;
                return MY_FAILED;
            }
        }

        // input size
        GetTensorSize(cur_tensor);

        // input dims
        m_vecInputNodesDims[i].clear();
        for (int j = 0; j < cur_tensor_param->nDims; j++)
        {
            m_vecInputNodesDims[i].push_back(cur_tensor_param->pShape[j]);
        }

        // create inputs memory
        OrtMemoryInfo *memory_info;
        CheckStatus(g_pOrt->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

        if (cur_tensor_param->type == DT_FLOAT)
        {
            CheckStatus(g_pOrt->CreateTensorWithDataAsOrtValue(
                memory_info, cur_tensor->pValue, cur_tensor_param->nLength, m_vecInputNodesDims[i].data(),
                m_vecInputNodesDims[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
        }
        else if (cur_tensor_param->type == DT_UINT8)
        {
            CheckStatus(g_pOrt->CreateTensorWithDataAsOrtValue(
                memory_info, cur_tensor->pValue, cur_tensor_param->nLength, m_vecInputNodesDims[i].data(),
                m_vecInputNodesDims[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &input_tensor));
        }
        else if (cur_tensor_param->type == DT_INT32)
        {
            CheckStatus(g_pOrt->CreateTensorWithDataAsOrtValue(
                memory_info, cur_tensor->pValue, cur_tensor_param->nLength, m_vecInputNodesDims[i].data(),
                m_vecInputNodesDims[i].size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &input_tensor));
        }
        else
        {
            std::cout << "Now the tensor data type not supported!!!" << std::endl;
            return MY_FAILED;
        }

        // store input data and release resources of info
        int is_tensor;
        CheckStatus(g_pOrt->IsTensor(input_tensor, &is_tensor));
        assert(is_tensor);
        g_pOrt->ReleaseMemoryInfo(memory_info);

        input_tensors[i] = input_tensor;
    }

    /*===================== process output tensor =====================*/
    std::vector<const char *> output_node_names;
    for (int i = 0; i < m_ouput_tensor_array->nArraySize; i++)
    {
        tensor_t *cur_tensor = &(m_ouput_tensor_array->pTensorArray[i]);
        tensor_params_t *cur_tensor_param = cur_tensor->pTensorInfo;
        if (!FindNameInTensorNames(cur_tensor_param->aTensorName, m_vecOutputNodesName))
        {
            std::cout << "Can't find output tensor names " << cur_tensor_param->aTensorName << " in model " << std::endl;
            return MY_FAILED;
        }

        output_node_names.push_back(cur_tensor_param->aTensorName);
    }

    std::vector<OrtValue *> output_tensors(output_node_names.size());

    CheckStatus(g_pOrt->Run(m_pSession,                                    // session
                                   NULL,                                          // run_options
                                   m_vecInputNodesName.data(),                    // input_names
                                   (const OrtValue *const *)input_tensors.data(), // input   values
                                   input_tensors.size(),                          // input_len
                                   m_vecOutputNodesName.data(),                   // output_names
                                   m_vecOutputNodesName.size(),                   // output_names_len
                                   output_tensors.data()));                       // OrtValue** output

    for (size_t i = 0; i < output_tensors.size(); i++)
    {
        int is_tensor;
        CheckStatus(g_pOrt->IsTensor(output_tensors[i], &is_tensor));
        assert(is_tensor);

        float *floatarr;
        CheckStatus(g_pOrt->GetTensorMutableData(output_tensors[i], (void **)&floatarr));

        tensor_t *cur_tensor = &(m_ouput_tensor_array->pTensorArray[i]);
        tensor_params_t *cur_tensor_param = cur_tensor->pTensorInfo;

        GetTensorSize(cur_tensor);

        memcpy(cur_tensor->pValue, floatarr, cur_tensor_param->nLength);
    }

    // Release output tensor if set
    for (auto &tensor : output_tensors)
    {
        if (tensor != nullptr)
        {
            g_pOrt->ReleaseValue(tensor);
        }
    }
    input_tensors.clear();
    output_tensors.clear();

    m_onnx_mutex.unlock();

    MY_DEBUG("End onnx  inference tensors succeed!!!\n");
    return MY_SUCCESS;
}

/**
 * @brief Construct a new Onnx Runtime Model Handle:: Onnx Runtime Model Handle object
 * 
 * @param tModelParam 
 */
OnnxRuntimeModelHandle::OnnxRuntimeModelHandle(model_params_t *tModelParam)
{
    m_tModelParam = new model_params_t();
    memcpy(m_tModelParam, tModelParam, sizeof(model_params_t));
}

/**
 * @brief Destroy the Onnx Runtime Model Handle:: Onnx Runtime Model Handle object
 * 
 */
OnnxRuntimeModelHandle::~OnnxRuntimeModelHandle()
{
    if (m_tModelParam)
    {
        delete m_tModelParam;
    }
}

/**
 * @brief member set
 * 
 * @param input_tensor_array 
 */
void OnnxRuntimeModelHandle::set_input_tensor_array(tensor_array_t *input_tensor_array)
{
    m_input_tensor_array = input_tensor_array;
}

/**
 * @brief member set
 * 
 * @param ouput_tensor_array 
 */
void OnnxRuntimeModelHandle::set_output_tensor_array(tensor_array_t *ouput_tensor_array)
{
    m_ouput_tensor_array = ouput_tensor_array;
}
