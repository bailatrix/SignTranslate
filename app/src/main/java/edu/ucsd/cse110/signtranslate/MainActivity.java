package edu.ucsd.cse110.signtranslate;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.VideoView;

public class MainActivity extends AppCompatActivity {

    VideoView video;
    ImageView photo;
    Button videoCameraBtn;
    Button photoCameraBtn;
    Button playBtn;
    Boolean tag;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        videoCameraBtn = (Button) findViewById(R.id.cameraBtn);
        photoCameraBtn = (Button) findViewById(R.id.photoCameraBtn);
        playBtn = (Button) findViewById(R.id.playBtn);
        video = (VideoView) findViewById(R.id.videoView);
        photo = (ImageView) findViewById(R.id.photoView);


        videoCameraBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent callVideoIntent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
                //callVideoIntent.setAction(MediaStore.ACTION_VIDEO_CAPTURE);
                //callVideoIntent.putExtra("intentIs", "videoIntent");
                tag = true;
                startActivityForResult(callVideoIntent, 0);
            }
        });

        photoCameraBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent imageIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                //imageIntent.putExtra("intentIs", "imageIntent");
                tag = false;
                startActivityForResult(imageIntent, 0);
            }
        });

        playBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                video.start();

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 0 && resultCode == RESULT_OK){
           /* if(data.getExtras().get("intentIs") == "videoIntent") {
                Uri videoUri = data.getData();
                video.setVideoURI(videoUri);
            }
            else if(data.getExtras().get("intentIs") == "imageIntent"){
                Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                photo.setImageBitmap(bitmap);
            }*/
            if(tag == true) {
                Uri videoUri = data.getData();
                video.setVideoURI(videoUri);
            }
            else if(tag == false){
                Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                photo.setImageBitmap(bitmap);
            }

        }

    }
}
