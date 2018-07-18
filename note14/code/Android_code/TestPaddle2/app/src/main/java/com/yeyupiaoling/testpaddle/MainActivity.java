package com.yeyupiaoling.testpaddle;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.bumptech.glide.Glide;
import com.yeyupiaoling.testpaddle.utils.CameraUtil;
import com.yeyupiaoling.testpaddle.utils.ToastUtil;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;

public class MainActivity extends AppCompatActivity {
    private ProgressDialog pd;
    private Uri imageUri;
    private ToastUtil toastUtil;
    private ImageView imageView;
    private TextView inferResult;
    private TextView inferTime;
    private ImageRecognition imageRecognition;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        toastUtil = new ToastUtil(this);

        // 初始化PaddlePaddle
        imageRecognition = new ImageRecognition();
        imageRecognition.initPaddle();
        imageRecognition.loadModel(this.getAssets(), "model/include/mobile_net.paddle");

        // 获取控件
        imageView = findViewById(R.id.infer_image);
        inferResult = findViewById(R.id.infer_result);
        inferTime = findViewById(R.id.use_time);
        Button getPhotoBtn = findViewById(R.id.use_photo);

        //从相册获取照片
        getPhotoBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (ContextCompat.checkSelfPermission(MainActivity.this,
                        Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(MainActivity.this,
                            new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
                } else {
                    getPhoto();
                }
            }
        });
    }


    // 动态申请权限回调
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        switch (requestCode) {
            case 1:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    getPhoto();
                } else {
                    toastUtil.showToast("你拒绝了授权");
                }
                break;
        }
    }

    // 拍照或从相册获取照片回调
    @SuppressLint("SetTextI18n")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == Activity.RESULT_OK) {
            switch (requestCode) {
                case 1:
                    // 显示对话框
                    showPD();
                    //获取相册的URI
                    Uri uri = data.getData();
                    //在界面显示图像
                    Glide.with(MainActivity.this).load(uri).into(imageView);
                    // 获取图像的路径
                    String imagePath = CameraUtil.getRealPathFromURI(MainActivity.this, uri);
                    // 获取开始预测时间
                    long startTime = System.currentTimeMillis();
                    // 获取预测结果
                    String resutl = imageRecognition.infer(imagePath);
                    // 获取结束预测时间
                    long endTime = System.currentTimeMillis();
                    // 隐藏对话框
                    dismissPD();
                    // 输出预测结果
                    inferResult.setText("预测结果：" + resutl);
                    // 输出预测时间
                    inferTime.setText("预测时间：" + String.valueOf(endTime - startTime) + "ms");
                    break;
            }
        }
    }

    //打开相册
    private void getPhoto() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, 1);
    }

    //显示对话框
    private void showPD() {
        pd = new ProgressDialog(this);
        pd.setTitle("预测图像");
        pd.setMessage("正在识别...");
        pd.setProgressStyle(ProgressDialog.STYLE_SPINNER);
        pd.show();
    }

    // 隐藏对话框
    private void dismissPD() {
        if (pd != null) {
            pd.dismiss();
        }
    }

}
