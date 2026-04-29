"""Batch re-annotate all reference and recorded videos.

Runs HandAnnotation.createAnnotatedVideo on every video, regenerating the
_handLandmarks.json files with wrist position data included.

Usage:
    ./venv/bin/python reannotate.py [--dry-run]
"""

import argparse
import os
import sys
import cv2

from HandAnnotation import HandAnnotation

REFERENCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "videos", "referenceVideos"))
RECORDED_DIR = os.path.join(REFERENCE_DIR, ".recorded")
ANNOTATED_DIR = os.path.join(REFERENCE_DIR, ".annotated")

os.makedirs(ANNOTATED_DIR, exist_ok=True)


def get_all_videos():
    videos = []
    for f in sorted(os.listdir(REFERENCE_DIR)):
        if f.endswith(".mp4"):
            videos.append(os.path.join(REFERENCE_DIR, f))
    for f in sorted(os.listdir(RECORDED_DIR)):
        if f.endswith(".mp4"):
            videos.append(os.path.join(RECORDED_DIR, f))
    return videos


def reannotate(video_path, dry_run=False):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(ANNOTATED_DIR, f"{stem}_annotated.mp4")
    json_path = os.path.join(ANNOTATED_DIR, f"{stem}_annotated_handLandmarks.json")

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Processing: {os.path.basename(video_path)}")
    print(f"  -> {out_path}")

    if dry_run:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return
    cap.release()

    ann = HandAnnotation(cv2.VideoCapture(video_path))
    ann.cam.release()
    ann.out.release()
    ann.createAnnotatedVideo(video_path, out_path)
    print(f"  JSON saved: {os.path.exists(json_path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    videos = get_all_videos()
    print(f"Found {len(videos)} videos to process.")
    for v in videos:
        reannotate(v, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
