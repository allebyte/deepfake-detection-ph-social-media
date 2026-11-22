import sys
import subprocess
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import imageio_ffmpeg

# ==========================================
# 1. FOLDER PATHS
# ==========================================
RAW_DIR = Path(r"D:\Deepfake_Thesis\01_Raw_Datasets\PH_Dataset")
OUT_DIR = Path(r"D:\Deepfake_Thesis\PH_Dataset_Preprocessed")
FRAMES_OUT = OUT_DIR / "ph_in_the_wild"

RAW_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_OUT.mkdir(parents=True, exist_ok=True)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
NUM_FRAMES = 16
IMG_SIZE = 299       # Crisp face crop; evaluation script resizes to 224 or 299 per model
BBOX_MARGIN = 0.25   # 25% margin around detected face (standard FaceForensics++ / Celeb-DF crop)

# ==========================================
# 2. ALL 10 PAIRS (20 VIDEOS)
# Format: (video_id, pair_id, label [1=fake, 0=real], url, start_ts, end_ts, crop_half)
# ==========================================
CLIPS = [
    # --- Pairs 1-4 ---
    ("PH_FAKE_001", "pair_01", 1, None, None, None, None),
    ("PH_REAL_001", "pair_01", 0, None, None, None, None),

    ("PH_FAKE_002", "pair_02", 1, None, None, None, None),
    ("PH_REAL_002", "pair_02", 0, None, None, None, None),

    ("PH_FAKE_003", "pair_03", 1, None, None, None, None),
    ("PH_REAL_003", "pair_03", 0, None, None, None, None),

    ("PH_FAKE_004", "pair_04", 1, None, None, None, None),
    ("PH_REAL_004", "pair_04", 0, None, None, None, None),

    # --- Pairs 5-10 (Downloaded & trimmed automatically into PH_Dataset) ---
    # Pair 5 (Amy Perez; 01:51-01:54 avoids yellow chin captions, left half isolates Amy Perez)
    ("PH_FAKE_005", "pair_05", 1, "https://www.youtube.com/watch?v=FbQaCq068Tw", "00:01:51", "00:01:54", "left"),
    ("PH_REAL_005", "pair_05", 0, "https://www.youtube.com/watch?v=TrtUdhSMtqU", "00:04:06", "00:04:15", None),

    # Pair 6 (Fake timestamp 00:36-00:41; same @oneph_cignal clip)
    ("PH_FAKE_006", "pair_06", 1, "https://www.tiktok.com/@oneph_cignal/video/7353983171781741829", "00:00:36", "00:00:41", None),
    ("PH_REAL_006", "pair_06", 0, "https://www.tiktok.com/@oneph_cignal/video/7353983171781741829", "00:00:51", "00:00:56", None),

    # Pair 7 (Marian Rivera)
    ("PH_FAKE_007", "pair_07", 1, "https://www.tiktok.com/@behborro/video/7054244648679673114", "00:00:00", "00:00:12", None),
    ("PH_REAL_007", "pair_07", 0, "https://www.tiktok.com/@marianrivera/video/7669789540181085460", "00:00:00", "00:00:15", None),

    # Pair 8
    ("PH_FAKE_008", "pair_08", 1, "https://www.tiktok.com/@allyvidmor/video/7654978224044150037", "00:00:03", "00:00:07", None),
    ("PH_REAL_008", "pair_08", 0, "https://www.tiktok.com/@kolokoystv13/video/7630411239902186772", "00:00:00", "00:00:07", None),

    # Pair 9
    ("PH_FAKE_009", "pair_09", 1, "https://www.tiktok.com/@bitoystory/video/7572136550054546708", "00:00:00", "00:00:15", None),
    ("PH_REAL_009", "pair_09", 0, "https://www.tiktok.com/@allanyebes/video/7690973159016664321", "00:00:00", "00:00:10", None),

    # Pair 10
    ("PH_FAKE_010", "pair_10", 1, "https://www.tiktok.com/@lavieenrozs/video/7625625722014403861", "00:00:23", "00:00:33", None),
    ("PH_REAL_010", "pair_10", 0, "https://www.tiktok.com/@psngayon/video/7456609600398314759", "00:00:00", "00:00:10", None),
]

# ==========================================
# 3. INITIALIZE OPENCV 5 YuNet FACE DETECTOR
# ==========================================
YUNET_PATH = RAW_DIR / "face_detection_yunet_2023mar.onnx"
if not YUNET_PATH.exists():
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        str(YUNET_PATH),
    )

face_detector = cv2.FaceDetectorYN.create(
    str(YUNET_PATH), "", (320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=5000
)


def get_or_download_mp4(vid_id: str, url, start_ts, end_ts) -> Path:
    out_mp4 = RAW_DIR / f"{vid_id}.mp4"
    if out_mp4.exists() and out_mp4.stat().st_size > 10_000:
        print(f"[FOUND LOCAL] {out_mp4.name}")
        return out_mp4

    if url is None:
        raise FileNotFoundError(f"Expected local file {out_mp4} was not found!")

    section = f"*{start_ts}-{end_ts}"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", FFMPEG_EXE,
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "--merge-output-format", "mp4",
        "-o", str(out_mp4),
    ]
    if "tiktok.com" in url:
        cmd.extend(["--impersonate", "chrome"])
    cmd.append(url)

    print(f"[DOWNLOADING] {vid_id} ({section}) ...")
    subprocess.run(cmd, check=True)
    return out_mp4


def detect_and_crop_face(frame_bgr: np.ndarray, last_box=None):
    h, w, _ = frame_bgr.shape
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame_bgr)

    if faces is not None and len(faces) > 0:
        best = max(faces, key=lambda f: f[2] * f[3])
        x, y, fw, fh = best[0], best[1], best[2], best[3]
        last_box = (x, y, fw, fh)
    elif last_box is not None:
        x, y, fw, fh = last_box
    else:
        side = int(min(h, w) * 0.6)
        x = (w - side) // 2
        y = max(0, int(h * 0.15))
        fw = fh = side

    cx, cy = x + fw / 2.0, y + fh / 2.0
    side = int(max(fw, fh) * (1.0 + BBOX_MARGIN))
    x1 = max(0, int(cx - side / 2.0))
    y1 = max(0, int(cy - side / 2.0))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    crop = frame_bgr[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return crop_resized, last_box


def process_all():
    meta_rows = []
    preview_tiles = []

    for vid_id, pair_id, label, url, start_ts, end_ts, crop_half in CLIPS:
        try:
            mp4_path = get_or_download_mp4(vid_id, url, start_ts, end_ts)
        except Exception as e:
            print(f"[ERROR] {vid_id}: {e}")
            continue

        cap = cv2.VideoCapture(str(mp4_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"[ERROR] Could not read frames from {mp4_path}")
            cap.release()
            continue

        indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        vid_out_dir = FRAMES_OUT / vid_id
        vid_out_dir.mkdir(parents=True, exist_ok=True)

        last_box = None
        mid_crop = None

        for out_idx, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            if crop_half == "left":
                frame = frame[:, : frame.shape[1] // 2]
            elif crop_half == "right":
                frame = frame[:, frame.shape[1] // 2 :]

            face_img, last_box = detect_and_crop_face(frame, last_box)
            cv2.imwrite(str(vid_out_dir / f"frame_{out_idx:02d}.jpg"), face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if out_idx == NUM_FRAMES // 2:
                mid_crop = face_img.copy()

        cap.release()

        if mid_crop is not None:
            cv2.putText(mid_crop, vid_id, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            preview_tiles.append(cv2.resize(mid_crop, (224, 224)))

        platform = "local_pair1_4" if url is None else ("youtube" if "youtu" in url else "tiktok")
        meta_rows.append({
            "video_id": vid_id,
            "pair_id": pair_id,
            "label": label,
            "platform": platform,
            "segment": "full_clip" if start_ts is None else f"{start_ts}-{end_ts}",
        })
        print(f"[OK] Saved {NUM_FRAMES} face crops -> {vid_out_dir}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(OUT_DIR / "ph_test.csv", index=False)

    if preview_tiles:
        cols = 4
        rows_n = int(np.ceil(len(preview_tiles) / cols))
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        while len(preview_tiles) < rows_n * cols:
            preview_tiles.append(blank)
        grid = np.vstack([np.hstack(preview_tiles[r * cols : (r + 1) * cols]) for r in range(rows_n)])
        cv2.imwrite(str(OUT_DIR / "ph_pairs_preview.jpg"), grid)
        print(f"\nSaved visual crop verification grid: {OUT_DIR / 'ph_pairs_preview.jpg'}")

    print(f"Saved anonymized metadata CSV: {OUT_DIR / 'ph_test.csv'}")


if __name__ == "__main__":
    process_all()
