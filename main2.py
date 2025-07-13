import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from speed1 import SpeedEstimator

st.set_page_config(page_title="YOLOv8 Speed & Counting", layout="wide")
st.title("🚗 Vehicle Speed Estimation & Counting with YOLOv8")

# Inisialisasi state
if "paused" not in st.session_state:
    st.session_state.paused = False
if "frame_pos" not in st.session_state:
    st.session_state.frame_pos = 0
if "step" not in st.session_state:
    st.session_state.step = False
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

video_file = st.file_uploader("📁 Upload Video", type=["mp4", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    model = YOLO("yolov8n.pt")  # Ganti ke "best.pt" kalau pakai model custom
    line_pts = [(0, 288), (1019, 288)]
    names = model.names
    speed_obj = SpeedEstimator(reg_pts=line_pts, names=names)

    crossed_up, crossed_down = set(), set()
    count_up, count_down = 0, 0
    track_history = {}
    stframe = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sidebar kontrol
    with st.sidebar:
        st.header("📊 Statistik Kendaraan")
        up_count_display = st.empty()
        down_count_display = st.empty()

        st.markdown("## 🎮 Kontrol Playback")
        if st.button("⏸ Pause" if not st.session_state.paused else "▶️ Lanjutkan"):
            st.session_state.paused = not st.session_state.paused

        if st.session_state.paused:
            if st.button("➡️ Step Satu Frame"):
                st.session_state.step = True

        st.session_state.frame_pos = st.slider("📍 Posisi Frame", 0, total_frames - 1, st.session_state.frame_pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)

    def get_centroid(box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    while cap.isOpened():
        if st.session_state.paused and not st.session_state.step:
            stframe.image(frame, channels="BGR", use_container_width=True)
            continue

        st.session_state.step = False  # Reset step flag
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))

        results = model.track(frame, persist=True, classes=[2, 5, 7])
        if not results or results[0].boxes.id is None:
            stframe.image(frame, channels="BGR", use_container_width=True)
            st.session_state.frame_pos += 1
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()

        frame = speed_obj.estimate_speed(frame, results)

        for box, obj_id in zip(boxes, ids):
            cx, cy = get_centroid(box)

            if obj_id in track_history:
                prev_cx, prev_cy = track_history[obj_id]
                direction = cy - prev_cy

                if direction > 0 and prev_cy < line_pts[0][1] <= cy:
                    if obj_id not in crossed_down:
                        crossed_down.add(obj_id)
                        count_down += 1

                elif direction < 0 and prev_cy > line_pts[0][1] >= cy:
                    if obj_id not in crossed_up:
                        crossed_up.add(obj_id)
                        count_up += 1

            track_history[obj_id] = (cx, cy)

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f'ID:{obj_id}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        up_count_display.metric("⬆️ Kendaraan Naik", count_up)
        down_count_display.metric("⬇️ Kendaraan Turun", count_down)

        cv2.putText(frame, f'⬆️ Up: {count_up}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'⬇️ Down: {count_down}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.line(frame, line_pts[0], line_pts[1], (255, 0, 0), 2)

        stframe.image(frame, channels="BGR", use_container_width=True)
        st.session_state.frame_pos += 1

    cap.release()
    st.success("✅ Video processing complete.")
    st.markdown("### 🧾 Hasil Akhir")
    st.info(f"⬆️ **Total Kendaraan Naik:** {count_up}  \n⬇️ **Total Kendaraan Turun:** {count_down}")
