import argparse
import os
import cv2

from HandAnnotation import HandAnnotation
from Scoring import Scoring


DEFAULT_WEBCAM_VIDEO = "/home/emmika/Videos/Screencasts/Screencast From 2026-04-13 15-29-13.mp4"
DEFAULT_REFERENCE_VIDEO = "slovo"
DEFAULT_REFERENCE_FOLDER = "../videa"
DEFAULT_ANNOTATED_FOLDER = "../videa/.annotated"


def resolve_reference_video(reference_arg, reference_folder):
    """Resolve reference input as either direct file path or gesture name."""
    if os.path.isfile(reference_arg):
        return os.path.abspath(reference_arg)

    candidate_with_ext = os.path.join(reference_folder, f"{reference_arg}.mp4")
    if os.path.isfile(candidate_with_ext):
        return os.path.abspath(candidate_with_ext)

    candidate_raw = os.path.join(reference_folder, reference_arg)
    if os.path.isfile(candidate_raw):
        return os.path.abspath(candidate_raw)

    raise FileNotFoundError(
        f"Could not resolve reference video '{reference_arg}'. Tried: "
        f"'{reference_arg}', '{candidate_with_ext}', '{candidate_raw}'"
    )


def ensure_landmarks(video_path, annotated_folder):
    """Load cached landmarks if available; otherwise create annotated video + landmarks."""
    os.makedirs(annotated_folder, exist_ok=True)

    filename = os.path.basename(video_path)
    stem, ext = os.path.splitext(filename)
    annotated_path = os.path.join(annotated_folder, f"{stem}_annotated{ext or '.mp4'}")
    landmarks_path = os.path.splitext(annotated_path)[0] + "_handLandmarks.json"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    annotation = HandAnnotation(cap)

    cached_ok = False
    if os.path.isfile(landmarks_path):
        loaded = annotation.loadLandmarksFromFile(landmarks_path)
        cached_ok = loaded is not None and len(loaded) > 0

    if not cached_ok:
        annotation.createAnnotatedVideo(video_path, annotated_path)

    # Cleanup resources not needed after landmark extraction.
    annotation.cam.release()
    annotation.out.release()

    return annotation


def summarize_similarity(scorer, user_sequence, reference_sequence):
    """Return detailed metric values and frame counters."""
    trimmed_reference = scorer._trimReferenceSequenceByMarkers(reference_sequence)
    user_frames = scorer._extractPerHandArrays(user_sequence)
    reference_frames = scorer._extractPerHandArrays(trimmed_reference)

    if len(user_frames) == 0 or len(reference_frames) == 0:
        return {
            "user_frames": len(user_frames),
            "reference_frames_total": len(reference_sequence),
            "reference_frames_trimmed": len(reference_frames),
            "hand_weights": [0.5, 0.5],
            "dtw_distance": float("inf"),
            "euclidean_distance": float("inf"),
            "cosine_similarity": 0.0,
            "final_score": 0.0,
        }

    hand_weights = scorer._calculateHandMotionWeights(reference_frames)

    dtw_distance = scorer._dtwDistance(user_frames, reference_frames, hand_weights)
    euclidean_distance = scorer._averageEuclideanDistance(user_frames, reference_frames, hand_weights)
    cosine_similarity = scorer._averageCosineSimilarity(user_frames, reference_frames, hand_weights)
    final_score = scorer.calculateScore(user_sequence)

    return {
        "user_frames": len(user_frames),
        "reference_frames_total": len(reference_sequence),
        "reference_frames_trimmed": len(reference_frames),
        "hand_weights": hand_weights,
        "dtw_distance": dtw_distance,
        "euclidean_distance": euclidean_distance,
        "cosine_similarity": cosine_similarity,
        "final_score": final_score,
    }


def marker_indices(reference_annotation, total_frames):
    """Convert 0-1000 marker positions to frame indices for display."""
    marker_start = int(max(0, min(1000, getattr(reference_annotation, "markerStart", 0))))
    marker_end = int(max(0, min(1000, getattr(reference_annotation, "markerEnd", 1000))))
    if marker_end < marker_start:
        marker_start, marker_end = marker_end, marker_start

    if total_frames <= 1:
        return marker_start, marker_end, 0, max(total_frames - 1, 0)

    start_idx = int((marker_start / 1000.0) * (total_frames - 1))
    end_idx = int((marker_end / 1000.0) * (total_frames - 1))
    return marker_start, marker_end, start_idx, end_idx


def main():
    parser = argparse.ArgumentParser(
        description="Compare a webcam/user video against a reference gesture video and print similarity metrics."
    )
    parser.add_argument("--webcam", default=DEFAULT_WEBCAM_VIDEO, help="Path to user/webcam video")
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE_VIDEO,
        help="Reference video path or gesture name (example: slovo)",
    )
    parser.add_argument(
        "--reference-folder",
        default=DEFAULT_REFERENCE_FOLDER,
        help="Folder used when --reference is a gesture name",
    )
    parser.add_argument(
        "--annotated-folder",
        default=DEFAULT_ANNOTATED_FOLDER,
        help="Folder for cached annotated videos and landmarks",
    )

    args = parser.parse_args()

    webcam_path = os.path.abspath(args.webcam)
    reference_path = resolve_reference_video(args.reference, args.reference_folder)

    if not os.path.isfile(webcam_path):
        raise FileNotFoundError(f"Webcam input video not found: {webcam_path}")

    print("=== Video Similarity Comparison ===")
    print(f"Webcam video   : {webcam_path}")
    print(f"Reference video: {reference_path}")

    webcam_annotation = ensure_landmarks(webcam_path, args.annotated_folder)
    reference_annotation = ensure_landmarks(reference_path, args.annotated_folder)

    scorer = Scoring(webcam_annotation, reference_annotation)

    user_sequence = webcam_annotation.handLandmarksTimestamped
    reference_sequence = reference_annotation.handLandmarksTimestamped

    metrics = summarize_similarity(scorer, user_sequence, reference_sequence)

    marker_start, marker_end, start_idx, end_idx = marker_indices(reference_annotation, len(reference_sequence))

    print("\n--- Reference Marker Window (used for scoring) ---")
    print(f"Marker start/end (0-1000): {marker_start}/{marker_end}")
    print(f"Reference frame range     : {start_idx}-{end_idx} of {max(len(reference_sequence) - 1, 0)}")

    print("\n--- Sequence Stats ---")
    print(f"User frames (used)        : {metrics['user_frames']}")
    print(f"Reference frames (total)  : {metrics['reference_frames_total']}")
    print(f"Reference frames (trimmed): {metrics['reference_frames_trimmed']}")

    print("\n--- Similarity Metrics ---")
    print(f"Hand weights (h0/h1)      : {metrics['hand_weights'][0]:.3f}/{metrics['hand_weights'][1]:.3f}")
    print(f"DTW distance              : {metrics['dtw_distance']:.6f}")
    print(f"Euclidean distance        : {metrics['euclidean_distance']:.6f}")
    print(f"Cosine similarity         : {metrics['cosine_similarity']:.6f}")
    print(f"Final score (0-1)         : {metrics['final_score']:.6f}")
    print(f"Final score (%)           : {metrics['final_score'] * 100:.2f}%")


if __name__ == "__main__":
    main()
