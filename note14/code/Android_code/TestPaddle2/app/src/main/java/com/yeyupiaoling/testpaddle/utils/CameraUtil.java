package com.yeyupiaoling.testpaddle.utils;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.util.Log;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;

/**
 * Created by 夜雨飘零 on 2017/9/20.
 */

public class CameraUtil {

    public static Uri startCamera(Activity activity, int requestCode) {
        Uri imageUri;
        File outputImage = new File(activity.getExternalCacheDir(), "out_image.jpg");
        try {
            if (outputImage.exists()) {
                outputImage.delete();
            }
            outputImage.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (Build.VERSION.SDK_INT >= 24) {
            //兼容Android 7.0以上
            imageUri = FileProvider.getUriForFile(activity,
                    "com.wang.testface.fileprovider", outputImage);
        } else {
            imageUri = Uri.fromFile(outputImage);
        }
        // 指定开启系统相机的Action
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // 设置系统相机拍摄照片完成后图片文件的存放地址
        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        // 此值在最低质量最小文件尺寸时是0，在最高质量最大文件尺寸时是１
        intent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 0);
        activity.startActivityForResult(intent, requestCode);
        return imageUri;
    }

    /**
     * 启动相机
     */
    public static Uri startCameraSave(Activity activity, int requestCode) {

        // 指定相机拍摄照片保存地址
        String state = Environment.getExternalStorageState();
        if (state.equals(Environment.MEDIA_MOUNTED)) {
            Intent intent = new Intent();
            // 指定开启系统相机的Action
            intent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);
            File outDir = Environment
                    .getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
            if (!outDir.exists()) {
                outDir.mkdirs();
            }
            File outFile = new File(outDir, System.currentTimeMillis() + ".jpg");
            // 把文件地址转换成Uri格式
            Uri uri;
            if (Build.VERSION.SDK_INT >= 24) {
                //兼容Android 7.0以上
                uri = FileProvider.getUriForFile(activity,
                        "com.wang.testface.fileprovider", outFile);
            } else {
                uri = Uri.fromFile(outFile);
            }
            Log.e("getAbsolutePath=", outFile.getAbsolutePath());
            // 设置系统相机拍摄照片完成后图片文件的存放地址
            intent.putExtra(MediaStore.EXTRA_OUTPUT, uri);
            // 此值在最低质量最小文件尺寸时是0，在最高质量最大文件尺寸时是１
            intent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 0);
            activity.startActivityForResult(intent, requestCode);
            return uri;
        } else {
            Toast.makeText(activity, "请确认已经插入SD卡",
                    Toast.LENGTH_LONG).show();
            return null;
        }
    }

    //获取图片的路径
    public static String getRealPathFromURI(Context context, Uri uri) {
        String result;
        Cursor cursor = context.getContentResolver().query(uri, null, null, null, null);
        if (cursor == null) {
            result = uri.getPath();
        } else {
            cursor.moveToFirst();
            int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
            result = cursor.getString(idx);
            cursor.close();
        }
        return result;
    }
}
