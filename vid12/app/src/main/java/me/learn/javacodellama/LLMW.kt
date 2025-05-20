package me.learn.javacodellama

import android.llama.cpp.LLamaAndroid
import android.llama.cpp.LLamaAndroid.Companion.instance
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch

class LLMW(private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()) {
    interface MessageHandler {
        fun h(msg: String)
    }

    fun load(path: String) {
        GlobalScope.launch {
            llamaAndroid.load(path)
        }
    }


    fun unload() {
        GlobalScope.launch {
            llamaAndroid.unload()
        }
    }

    fun send(msg: String, mh: MessageHandler){
        GlobalScope.launch {
            llamaAndroid.send(msg).collect{mh.h(it)}
        }
    }

}