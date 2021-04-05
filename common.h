#ifndef _MY_COMMON_H_
#define _MY_COMMON_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdio.h>

    typedef signed char my_s8;
    typedef unsigned char my_u8;
    typedef signed short my_s16;
    typedef unsigned short my_u16;
    typedef signed int my_s32;
    typedef unsigned int my_u32;
    typedef signed char MY_BOOL;

#define TRUE 1
#define FALSE 0

    //#define DEBUG_ON

#ifdef DEBUG_ON
#define MY_DEBUG(...)                                                                          \
    do                                                                                         \
    {                                                                                          \
        fprintf(stdout, "[DEBUG]  %s    %s  (Line  %d) : ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stdout, __VA_ARGS__);                                                          \
    } while (0)
#else
#define MY_DEBUG(...)
#endif

#define MY_ERROR(...)                                                                          \
    do                                                                                         \
    {                                                                                          \
        fprintf(stderr, "[ERROR]  %s    %s  (Line  %d) : ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                                          \
    } while (0)

#define MY_CHECK_NULL(a, errcode)     \
    do                                \
    {                                 \
        if (NULL == (a))              \
        {                             \
            MY_ERROR("NULL DATA \n"); \
            return errcode;           \
        }                             \
    } while (0)

    // 返回值的数据结构
    typedef enum
    {
        MY_SUCCESS = 0, //成功
        MY_FAILED,      //失败
        MY_PARAM_NULL,  //参数为空
        MY_PARAM_SET_ERROR,
        MY_FILE_NOT_EXIST,       //文件不存在
        MY_MEMORY_MALLOC_FAILED, //内存分配失败
        MY_MODEL_LOAD_FAILED,    //模型加载失败
        MY_TENSOR_ALLOC_FAILED,  //tensor内存分配失败
    } result_t;

    typedef enum
    {
        TF_SAVED_MODEL = 0,
        TF_FROZEN_PB,
    } tf_model_type_t;

    typedef enum
    {
        TF_GPU_MEM_FRACTION = 0,
        TF_GPU_MEM_VIRTUAL_GPU,
    } gpu_part_type_t;

    typedef enum
    {
        TRT_DISABLE_ALL = 0,
        TRT_ENABLE_BASIC,
        TRT_ENABLE_CUSTOM,
    } tf_trt_optimize_level_t;

    typedef enum
    {
        TRT_MODE_FP32 = 0,
        TRT_MODE_FP16,
        TRT_MODE_INT8
    } tf_trt_precision_t;

    typedef struct
    {
        bool is_dynamic_op;                        //是否动态转换
        int max_batch_size;                        //最大的batch size
        signed long long max_workspace_size_bytes; //转换时需要的最大空间
        tf_trt_precision_t precision_mode;         //转换的精度
        signed long long min_segment_size;         //控制最少几个TF算子融合成TRT算子
        signed long long max_cached_engines;       //控制可以缓存的engine数量
    } tf_trt_custom_config_t;

    typedef struct
    {
        int cpu_or_gpu;           //模型加载再cpu：０；　　gpu: 1
        char visibleCard[32];     //设置哪些ＧＰＵ卡是可见的
        int gpu_id;               //虚拟的gpu id
        float gpu_memory_faction; //设置ＧＰＵ显存的比例： tensorflow参数
        char model_path[256];     //模型的路径名
        char paModelTagSet[256];  //模型的 tagset
        MY_BOOL bIsCipher;        //模型文件是否加密
        int encStartPoint;
        int encLength;
        tf_model_type_t model_type;

        //虚拟gpu参数
        gpu_part_type_t part_type;
        int visibleCardNum;
        int vGPUNumber[16]; //表示每张卡虚拟几个GPU
        float vGPU[16][16]; //表示每个虚拟GPU的显存大小

        //TF-TRT参数
        tf_trt_optimize_level_t model_optimize_level;
        tf_trt_custom_config_t tf_trt_config_st;
    } model_params_t;

    typedef struct
    {
        void *model_handle; //模型句柄
    } model_handle_t;

    typedef enum
    {
        DT_INVALID = 0,
        DT_FLOAT = 1,
        DT_DOUBLE = 2,
        DT_INT32 = 3,
        DT_UINT8 = 4,
        DT_INT16 = 5,
        DT_INT8 = 6,
        DT_STRING = 7,
        DT_INT64 = 9,
        DT_BOOL = 10,
    } tensor_types_t;

    //Tensor参数的数据结构
    typedef struct
    {
        tensor_types_t type;   //Tensor的类型
        char aTensorName[256]; //Tensor的名字
        int nDims;             //Tensor的rank
        int pShape[8];         //shape
        int nElementSize;      //多少个元素
        int nLength;           //多少个字节长度
    } tensor_params_t;

    //定义Tensor的数据结构
    typedef struct
    {
        tensor_params_t *pTensorInfo;
        void *pValue;
    } tensor_t;

    typedef struct
    {
        int nArraySize;        // 多少个参数
        tensor_params_t *pTensorParamArray;
        char pcSignatureDef[256]; //函数签名
    } tensor_params_array_t;

    typedef struct
    {
        int nArraySize;
        tensor_t *pTensorArray;
        char pcSignatureDef[256]; //函数签名
    } tensor_array_t;

#ifdef __cplusplus
}
#endif

#endif