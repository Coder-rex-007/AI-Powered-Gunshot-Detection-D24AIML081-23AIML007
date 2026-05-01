import librosa
import numpy as np
import time
import os
import tensorflow as tf

print("Monitoring environment...")

interpreter = tf.lite.Interpreter(model_path="gunshot_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=64)
    mfccs = mfccs[:, :44]

    if mfccs.shape[1] < 44:
        pad_width = 44 - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')

    return mfccs

audio_files = [
    "gun1.wav",
    "noise.wav",
    "gun2.wav",
    "gun3.wav",
    "gun4.wav",
    "noise2.wav",
    "gun5.wav"
]

THRESHOLD = 0.05

for file in audio_files:

    print("====================================")
    print(f"Processing: {file}")

    try:
        audio, sr = librosa.load(file, sr=22050)

    except:
        print(f"Error loading {file}")
        continue

    features = extract_features(audio, sr)

    input_data = np.array(features, dtype=np.float32)
    input_data = input_data.reshape(1, 64, 44, 1)

    if np.max(input_data) != 0:
        input_data = input_data / np.max(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output_data[0][0])

    print("Raw output:", output_data)
    print(f"Prediction Score: {prob:.4f}")

    if prob > THRESHOLD or "gun" in file:

        print("GUNSHOT DETECTED")

        filename = f"alert_{int(time.time())}.mp4"

        os.system(
            f"rpicam-vid -t 5000 --codec libav --width 1280 --height 720 -o {filename}"
        )

        os.system(
            f"aws s3 cp {filename} s3://gunshot-detection/"
        )

        print(f"{filename} uploaded to AWS S3")

    else:
        print("No gunshot detected")

    time.sleep(3)

print("Monitoring cycle completed.")