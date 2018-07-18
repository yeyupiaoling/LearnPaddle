#include "image_recognition.h"
#include <android/asset_manager_jni.h>
#include <time.h>

extern "C" {

paddle_gradient_machine gradient_machine_;

JNIEXPORT void
Java_com_yeyupiaoling_testpaddle_ImageRecognition_initPaddle(JNIEnv *env, jobject thiz) {
    static bool called = false;
    if (!called) {
        // Initalize Paddle
        char *argv[] = {const_cast<char *>("--use_gpu=False"),
                        const_cast<char *>("--pool_limit_size=0")};
        CHECK(paddle_init(2, (char **) argv));
        LOGI("初始化PaddlePaddle成功");
        called = true;
    }
}

JNIEXPORT void
Java_com_yeyupiaoling_testpaddle_ImageRecognition_loadModel(JNIEnv *env,
                                                            jobject thiz,
                                                            jobject jasset_manager,
                                                            jstring modelPath) {
    //加载上下文
    AAssetManager *aasset_manager = AAssetManager_fromJava(env, jasset_manager);
    BinaryReader::set_aasset_manager(aasset_manager);

    const char *merged_model_path = env->GetStringUTFChars(modelPath, 0);
    // Step 1: Reading merged model.
    LOGI("merged_model_path = %s", merged_model_path);
    long size;
    void *buf = BinaryReader()(merged_model_path, &size);
    // Create a gradient machine for inference.
    CHECK(paddle_gradient_machine_create_for_inference_with_parameters(
            &gradient_machine_, buf, size));
    // 释放空间
    env->ReleaseStringUTFChars(modelPath, merged_model_path);
    LOGI("加载模型成功");
    free(buf);
    buf = nullptr;
}

JNIEXPORT jfloatArray
Java_com_yeyupiaoling_testpaddle_ImageRecognition_infer(JNIEnv *env,
                                                        jobject thiz,
                                                        jbyteArray jpixels,
                                                        size_t height_,
                                                        size_t width_,
                                                        size_t channel_) {

    //网络的输入和输出被组织为paddle_arguments对象
    //在C-API中。在下面的评论中，“argument”具体指的是一个输入
    //PaddlePaddle C-API中的神经网络。
    paddle_arguments in_args = paddle_arguments_create_none();

    //调用函数来创建一个参数。
    CHECK(paddle_arguments_resize(in_args, 1));

    //每个参数需要一个矩阵或一个ivector（整数向量，稀疏
    //索引输入，通常用于NLP任务）来保存真实的输入数据。
    //在下面的评论中，“matrix”具体指的是需要的对象
    //参数来保存数据。这里我们为上面创建的矩阵创建
    //储存测试样品的存量。
    paddle_matrix mat = paddle_matrix_create(1, 3072, false);

    paddle_real *array;
    //获取指向第一行开始地址的指针
    //创建矩阵。
    CHECK(paddle_matrix_get_row(mat, 0, &array));

    //获取字节数组转换成浮点数组
    unsigned char *pixels =
            (unsigned char *) env->GetByteArrayElements(jpixels, 0);

    // 加载数据
    size_t index = 0;
    std::vector<float> means;
    means.clear();
    for (size_t i = 0; i < channel_; ++i) {
        means.push_back(0.0f);
    }
    for (size_t c = 0; c < channel_; ++c) {
        for (size_t h = 0; h < height_; ++h) {
            for (size_t w = 0; w < width_; ++w) {
                array[index] =
                        (static_cast<float>(
                                 pixels[(h * 32 + w) * 3 + c]) - means[c]) / 255;
                index++;
            }
        }
    }

    env->ReleaseByteArrayElements(jpixels, (jbyte *) pixels, 0);

    //将矩阵分配给输入参数。
    CHECK(paddle_arguments_set_value(in_args, 0, mat));

    //创建输出参数。
    paddle_arguments out_args = paddle_arguments_create_none();

    //调用向前计算。
    CHECK(paddle_gradient_machine_forward(gradient_machine_, in_args, out_args, false));

    //创建矩阵来保存神经网络的向前结果。
    paddle_matrix prob = paddle_matrix_create_none();
    //访问输出参数的矩阵，预测结果存储在哪个。
    CHECK(paddle_arguments_get_value(out_args, 0, prob));

    uint64_t height;
    uint64_t width;
    //获取矩阵的大小
    CHECK(paddle_matrix_get_shape(prob, &height, &width));
    //获取预测结果矩阵
    CHECK(paddle_matrix_get_row(prob, 0, &array));
    for (int i = 0; i < sizeof(array); ++i) {
        LOGI("array:%f", array[i]);
    }

    jfloatArray result = env->NewFloatArray(height * width);
    env->SetFloatArrayRegion(result, 0, height * width, array);

    // 清空内存
    CHECK(paddle_matrix_destroy(prob));
    CHECK(paddle_arguments_destroy(out_args));
    CHECK(paddle_matrix_destroy(mat));
    CHECK(paddle_arguments_destroy(in_args));

    return result;
}
}