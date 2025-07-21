import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import os
def train_dummy_lstm():
    x_train = np.random.randint(5, 30, (100, 10, 1)) 
    y_train = np.mean(x_train, axis=1) / 30 
    model = Sequential([
        tf.keras.layers.Input(shape=(10, 1)),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    model.fit(x_train, y_train, epochs=10, verbose=0)
    model.save("lstm_model.h5")

def predict_congestion(vehicle_counts, model):
    """Uses LSTM model to predict congestion level."""
    x_input = np.array(vehicle_counts[-10:]).reshape(1, 10, 1)
    prediction = model.predict(x_input, verbose=0)[0][0]
    if prediction > 0.7:
        return "High Congestion"
    elif prediction > 0.4:
        return "Moderate Congestion"
    else:
        return "Low Congestion"
st.title("🚦 Real-Time Traffic Density Detection and Forecasting")
video_file = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

if video_file:
    stframe = st.empty()
    with open("input_video.mp4", "wb") as f:
        f.write(video_file.read())
    model = YOLO("yolov8n.pt")  
    tracker = DeepSort(max_age=30)
    if not os.path.exists("lstm_model.h5"):
        train_dummy_lstm()
    lstm_model = load_model("lstm_model.h5", compile=False)

    cap = cv2.VideoCapture("input_video.mp4")
    vehicle_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) in [2, 3, 5, 7]:  
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'vehicle'))

        tracks = tracker.update_tracks(detections, frame=frame)
        vehicle_count = len([t for t in tracks if t.is_confirmed()])
        vehicle_counts.append(vehicle_count)

        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            track_id = track.track_id
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if len(vehicle_counts) >= 10:
            congestion = predict_congestion(vehicle_counts, lstm_model)
        else:
            congestion = "Analyzing..."

        cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Congestion: {congestion}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR", use_container_width=True)


    cap.release()
    st.success("🎉 Video Processing Completed!")
