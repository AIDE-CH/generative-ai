package me.learn.javacodellama;

import android.os.Bundle;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.lifecycle.DefaultLifecycleObserverAdapter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import me.learn.javacodellama.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {
    private LLMW llmw = new LLMW();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ActivityMainBinding binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        try{
            llmw.load(preparePath());
            binding.editTextText.setOnEditorActionListener(new TextView.OnEditorActionListener() {
                @Override
                public boolean onEditorAction(TextView textView, int actionID, KeyEvent keyEvent) {
                    if(actionID == 0 || actionID == EditorInfo.IME_ACTION_DONE
                            || actionID == EditorInfo.IME_ACTION_NEXT
                            || actionID == EditorInfo.IME_ACTION_GO
                            || actionID == EditorInfo.IME_ACTION_SEND){
                        String question = binding.editTextText.getText().toString();
                        textView.setText("");
                        llmw.send(question, new LLMW.MessageHandler() {
                            @Override
                            public void h(@NonNull String msg) {
                                MainActivity.this.runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        String all = binding.textView.getText().toString() +
                                                msg;
                                        binding.textView.setText(all);
                                    }
                                });
                            }
                        });
                    }

                    return false;
                }
            });

        }catch (Exception ex){
            Toast.makeText(this, "Load error: " + ex.getMessage(), Toast.LENGTH_LONG);
        }

    }


    private String preparePath() throws IOException {
        InputStream inputStream = getResources().openRawResource(R.raw.doa);

        File tempFile = File.createTempFile("myfile", ".gguf", getCacheDir());
        OutputStream out = new FileOutputStream(tempFile);

        byte[] buffer = new byte[1024];
        int read;
        while ((read = inputStream.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
        inputStream.close();
        out.close();
        return tempFile.getAbsolutePath().toString();
    }
}