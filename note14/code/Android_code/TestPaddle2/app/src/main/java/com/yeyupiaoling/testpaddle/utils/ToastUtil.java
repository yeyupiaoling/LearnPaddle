package com.yeyupiaoling.testpaddle.utils;

import android.content.Context;
import android.widget.Toast;

/**
 * 弹出Toast
 * Created by 夜雨飘零 on 2017/11/3.
 */

public class ToastUtil {
    private Context context;
    private Toast toast;

    public ToastUtil(Context context){
        this.context = context;
    }

    public void showToast(String text){
        if (toast == null) {
            toast = Toast.makeText(context,text,Toast.LENGTH_SHORT);
        }else {
            toast.setText(text);
        }
        toast.show();
    }
}
