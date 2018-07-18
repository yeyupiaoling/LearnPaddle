package com.yeyupiaoling.testpaddle;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * Created by 15696 on 2018/3/1.
 */

public class ImageRecognition {
    // 加载C++编写的库
    static {
        System.loadLibrary("image_recognition");
    }
    // 类别名称
    private String[] clasName = {"airplane", "automobile", "bird", "cat",
            "deer", "deer", "frog", "horse", "ship", "truck"};

    public byte[] getPixelsBGR(Bitmap bitmap) {
        // 计算我们的图像包含多少字节
        int bytes = bitmap.getByteCount();

        ByteBuffer buffer = ByteBuffer.allocate(bytes);
        // 将字节数据移动到缓冲区
        bitmap.copyPixelsToBuffer(buffer);

        // 获取包含数据的基础数组
        byte[] temp = buffer.array();

        byte[] pixels = new byte[(temp.length/4) * 3];
        // 进行像素复制
        for (int i = 0; i < temp.length/4; i++) {

            pixels[i * 3] = temp[i * 4 + 2];        //B
            pixels[i * 3 + 1] = temp[i * 4 + 1];    //G
            pixels[i * 3 + 2] = temp[i * 4 ];       //R
        }
        return pixels;
    }

    public String infer(String img_path) {
        //把图像读取成一个Bitmap对象
        Bitmap bitmap = BitmapFactory.decodeFile(img_path);
        Bitmap mBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        mBitmap.setWidth(32);
        mBitmap.setHeight(32);
        int width = mBitmap.getWidth();
        int height = mBitmap.getHeight();
        int channel = 3;
        //把图像生成一个数组
        byte[] pixels = getPixelsBGR(mBitmap);
        // 获取预测结果
        float[] result = infer(pixels, width, height, channel);
        // 把概率最大的结果提取出来
        float max = 0;
        int number = 0;
        for (int i = 0; i < result.length; i++) {
            if (result[i] > max) {
                max = result[i];
                number = i;
            }
        }
        String msg = "类别为：" + clasName[number] + "，可信度为：" + max;
        Log.i("ImageRecognition", msg);

        return msg;
    }

    // CPP中初始化PaddlePaddle
    public native void initPaddle();

    // CPP中加载预测合并模型
    public native void loadModel(AssetManager assetManager, String modelPath);

    // CPP中获取预测结果
    private native float[] infer(byte[] pixels, int width, int height, int channel);
}
