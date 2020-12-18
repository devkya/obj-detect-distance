/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.media.MediaPlayer;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/*private static HashMap<String, Double> objWdith = new HashMap<>();
objWdith.put("car", 180.0);
objWdith.put("bicycle", 70.0);
objWdith.put("motorcycle", 80.0);
objWdith.put("person", 40.0);
objWdith.put("scooter", 30.0);
objWdith.put("truck", 218.0);
objWdith.put("wheelchair", 59.5);
objWdith.put("chair", 90.0);*/


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */

public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.65f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Detector detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  // Measure distance variable
  private static double known_distance = 250.0;
  public static HashMap<String, Double> distances = new HashMap<String, Double>();
  public boolean isplay = false;
  MediaPlayer mp;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              this,
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing Detector!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Detector.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Detector.Recognition> mappedRecognitions =
                new ArrayList<Detector.Recognition>();

            for (final Detector.Recognition result : results) {
              final RectF location = result.getLocation();

              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);
                cropToFrameTransform.mapRect(location);
                final double distance = getDistance(result);
                Log.i("distance :", ""+distance);
                distances.put(result.getId(), distance);
                result.setLocation(location);
                mappedRecognitions.add(result);

                if (result.getTitle().equals(("person")) && (isplay == false) && (distance >= 280 && distance <=320)){
                  mp = MediaPlayer.create(getApplicationContext() ,R.raw.people_3);
                  mp.setLooping(false);
                  mp.start();
                  Log.i("isplaying : ", "" + mp.isPlaying());
                  isplay = true;
                }
                else if (result.getTitle().equals(("person")) && (isplay == false) && (distance >= 480 && distance <= 520)){
                  mp = MediaPlayer.create(getApplicationContext() ,R.raw.people_3);
                  mp.setLooping(false);
                  mp.start();
                  Log.i("isplaying : ", "" + mp.isPlaying());
                  isplay = true;
                }


                if (result.getTitle().equals(("car")) && (isplay == false) && (distance >= 480 && distance <= 520)) {
                  mp = MediaPlayer.create(getApplicationContext() ,R.raw.car_5);
                  mp.setLooping(false);
                  mp.start();
                  Log.i("isplaying : ", "" + mp.isPlaying());
                  isplay = true;
                }
                else if (result.getTitle().equals(("car")) && (isplay == false) && (distance >= 780 && distance <= 820)) {
                  mp = MediaPlayer.create(getApplicationContext() ,R.raw.car_5);
                  mp.setLooping(false);
                  mp.start();
                  Log.i("isplaying : ", "" + mp.isPlaying());
                  isplay = true;
                }
                else if (result.getTitle().equals(("car")) && (isplay == false) && (distance >= 980 && distance <= 1020)) {
                  mp = MediaPlayer.create(getApplicationContext() ,R.raw.car_5);
                  mp.setLooping(false);
                  mp.start();
                  Log.i("isplaying : ", "" + mp.isPlaying());
                  isplay = true;
                }

                try {
                  if (mp.isPlaying() == false) isplay = false;
                }
                catch (Exception e){
                  Log.i("catch :", ""+e);
                  continue;
                }

              }
            }
            tracker.setDistance(distances);
            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            // tracker.resetDistance();
            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  // Measure actual distance
  protected double getDistance(Detector.Recognition result) {
    // Hash Map : focus Length
    HashMap<String, Double> focustLength = new HashMap<String, Double>();
    focustLength.put("chair", (8892 * known_distance) / 46.0);
    focustLength.put("mouse", (4052 * known_distance) / 15.0);
    focustLength.put("person", (8000 * known_distance) / 50.0);
    focustLength.put("car", (23000 * known_distance) / 190.0);

    // Hash Map : Known Width
    HashMap<String, Double> knownWidth = new HashMap<String, Double>();
    knownWidth.put("chair", 46.0);
    knownWidth.put("mouse", 15.0);
    knownWidth.put("person", 50.0);
    knownWidth.put("car", 190.0);

     if(focustLength.containsKey(result.getTitle())){
      final RectF location = result.getLocation();
      final double obj_psize = location.width() * location.height();
      final double known_width =  knownWidth.get(result.getTitle());

      Log.i("objectSize : ", "" + obj_psize);

      final double focus_length = focustLength.get(result.getTitle());
      final double distance = (known_width * focus_length) / obj_psize;
      return distance;
    }else{
       return 0;
    }
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(
        () -> {
          try {
            detector.setUseNNAPI(isChecked);
          } catch (UnsupportedOperationException e) {
            LOGGER.e(e, "Failed to set \"Use NNAPI\".");
            runOnUiThread(
                () -> {
                  Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                });
          }
        });
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(
        () -> {
          try {
            detector.setNumThreads(numThreads);
          } catch (IllegalArgumentException e) {
            LOGGER.e(e, "Failed to set multithreads.");
            runOnUiThread(
                () -> {
                  Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                });
          }
        });
  }
}
