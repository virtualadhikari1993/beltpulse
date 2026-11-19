"""Step 3d - SEE the labels: draw them on the photos.

1) The 67 hand/LLM-labeled keyframes with their boxes
     outputs/03_labeling/labels_keyframes/<video>__<frame>.jpg   (one per keyframe, full size)
     outputs/03_labeling/labels_keyframes/_overview_<video>.jpg  (all keyframes of one video side by side)
2) The propagated labels as a video clip per recording (every labeled frame, boxes drawn)
     outputs/03_labeling/labels_videos/<video>.mp4

Run:  .venv\\Scripts\\python src\\03d_view_labels.py [keyframe_box_file.json]
"""
import os
import sys
import json
import cv2
import numpy as np

from common import ROOT, frame_path, put_label

BOX_FILE = sys.argv[1] if len(sys.argv) > 1 else "keyframe_boxes_llm.json"
LAB = os.path.join(ROOT, "data", "labels")
OUT_K = os.path.join(ROOT, "outputs", "03_labeling", "labels_keyframes")
OUT_V = os.path.join(ROOT, "outputs", "03_labeling", "labels_videos")
os.makedirs(OUT_K, exist_ok=True)
os.makedirs(OUT_V, exist_ok=True)
COLORS = {"wood": (0, 200, 255), "plastic": (255, 255, 255), "canga": (0, 120, 255),
          "jaspilito": (200, 120, 255), "tool": (0, 255, 0)}


def draw(img, boxes, thick=4):
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[c], thick)
        put_label(img, f"{c}  [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]", (int(x1) + 6, int(y1) + 30), 0.8)
    return img


# 1) keyframes (regular ones and the "extra" ones added where objects roll, see 03f)
lab_json = json.load(open(os.path.join(LAB, BOX_FILE)))
kf = lab_json["keyframes"]
extra = set(lab_json.get("extra_keyframes", {}).get("keys", []))
EXTRA_COLOR = (255, 0, 255)  # purple border = extra keyframe (rolling region)


def grid_page(ims, cols=3):
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    return np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])


by_video, extra_ims = {}, []
for key, boxes in sorted(kf.items(), key=lambda kv: (kv[0].split("/")[0], int(kv[0].split("/")[1]))):
    video, fid = key.split("/")
    img = draw(cv2.imread(frame_path(video, int(fid))), boxes)
    tag_txt = "EXTRA keyframe (rolling)" if key in extra else "keyframe"
    put_label(img, f"{video.replace('boardcamera_', '')}  frame {fid}  {tag_txt}", (10, 700), 1.0)
    if key in extra:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), EXTRA_COLOR, 14)
    cv2.imwrite(os.path.join(OUT_K, f"{video}__{fid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    small = cv2.resize(img, (640, 360))
    by_video.setdefault(video, []).append(small)
    if key in extra:
        extra_ims.append(small)
for video, ims in by_video.items():
    cv2.imwrite(os.path.join(OUT_K, f"_overview_{video.replace('boardcamera_', '')}.jpg"), grid_page(ims))
for p in range(0, len(extra_ims), 12):
    cv2.imwrite(os.path.join(OUT_K, f"_review_EXTRA_rolling_p{p // 12 + 1}.jpg"), grid_page(extra_ims[p:p + 12]))
print(f"{len(kf)} keyframes drawn ({len(extra)} extra) -> {OUT_K}")

# 2) propagated labels as clips
labels = json.load(open(os.path.join(LAB, "boxes.json")))
keyset = set(kf)
per_video = {}
for key, boxes in labels.items():
    v, f = key.split("/")
    per_video.setdefault(v, []).append((int(f), boxes))
for video, items in sorted(per_video.items()):
    vw = cv2.VideoWriter(os.path.join(OUT_V, f"{video.replace('boardcamera_', '')}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 360))
    for fid, boxes in sorted(items):
        img = draw(cv2.imread(frame_path(video, fid)), boxes, 3)
        k = f"{video}/{fid}"
        txt = "   EXTRA KEYFRAME (LLM, rolling)" if k in extra else ("   KEYFRAME (LLM)" if k in keyset else "   (physics)")
        put_label(img, f"frame {fid}" + txt, (10, 40), 1.0)
        vw.write(cv2.resize(img, (640, 360)))
    vw.release()
print(f"{len(per_video)} clips -> {OUT_V}")

# 3) propagated labels as contact sheets (pictures), every EVERY-th frame, 4 x 5 thumbnails per page
EVERY, COLS, ROWS, TW, TH = 4, 4, 5, 480, 270
OUT_P = os.path.join(ROOT, "outputs", "03_labeling", "labels_pages")
os.makedirs(OUT_P, exist_ok=True)
n_pages = 0
for video, items in sorted(per_video.items()):
    items = sorted(items)
    show = [it for i, it in enumerate(items) if i % EVERY == 0 or f"{video}/{it[0]}" in keyset]
    for p in range(0, len(show), COLS * ROWS):
        thumbs = []
        for fid, bxs in show[p:p + COLS * ROWS]:
            img = cv2.resize(draw(cv2.imread(frame_path(video, fid)), bxs, 5), (TW, TH))
            k = f"{video}/{fid}"
            if k in extra:
                put_label(img, f"{fid}  LLM EXTRA (rolling)", (6, 26), 0.7)
                cv2.rectangle(img, (0, 0), (TW - 1, TH - 1), EXTRA_COLOR, 8)
            elif k in keyset:
                put_label(img, f"{fid}  LLM keyframe", (6, 26), 0.7)
                cv2.rectangle(img, (0, 0), (TW - 1, TH - 1), (0, 0, 255), 8)
            else:
                put_label(img, f"{fid}  physics", (6, 26), 0.7)
            thumbs.append(img)
        while len(thumbs) < COLS * ROWS:
            thumbs.append(np.zeros((TH, TW, 3), np.uint8))
        page = np.vstack([np.hstack(thumbs[r * COLS:(r + 1) * COLS]) for r in range(ROWS)])
        name = f"{video.replace('boardcamera_', '')}_p{p // (COLS * ROWS) + 1:02d}.jpg"
        cv2.imwrite(os.path.join(OUT_P, name), page, [cv2.IMWRITE_JPEG_QUALITY, 80])
        n_pages += 1
print(f"{n_pages} contact-sheet pages -> {OUT_P}")
