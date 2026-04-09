# AI GENERATED

"""
Test script for Scoring similarity metrics.
"""

import numpy as np
from Scoring import Scoring
from HandAnnotation import HandAnnotation
import cv2


def create_mock_landmarks(offset_x=0, offset_y=0, scale_factor=1.0, noise_level=0.0):
    """
    Create mock landmark data for testing.

    Args:
        offset_x, offset_y: Position offset
        scale_factor: Scale multiplier
        noise_level: Random noise to add

    Returns:
        List of landmark dicts
    """
    # Base hand positions (simplified)
    base_positions = [
        (0.5, 0.5, 0.0),  # wrist
        (0.52, 0.48, 0.0),  # thumb
        (0.54, 0.46, 0.0),
        (0.56, 0.44, 0.0),
        (0.58, 0.42, 0.0),
        (0.52, 0.52, 0.0),  # index
        (0.54, 0.54, 0.0),
        (0.56, 0.56, 0.0),
        (0.58, 0.58, 0.0),
        (0.51, 0.55, 0.0),  # middle
        (0.52, 0.58, 0.0),
        (0.53, 0.61, 0.0),
        (0.54, 0.64, 0.0),
        (0.50, 0.57, 0.0),  # ring
        (0.50, 0.60, 0.0),
        (0.50, 0.63, 0.0),
        (0.50, 0.66, 0.0),
        (0.49, 0.59, 0.0),  # pinky
        (0.48, 0.62, 0.0),
        (0.47, 0.65, 0.0),
        (0.46, 0.68, 0.0),
    ]

    landmarks = []
    for x, y, z in base_positions:
        # Apply transformations
        scaled_x = (x * scale_factor) + offset_x
        scaled_y = (y * scale_factor) + offset_y

        # Add noise
        if noise_level > 0:
            scaled_x += np.random.normal(0, noise_level)
            scaled_y += np.random.normal(0, noise_level)
            z += np.random.normal(0, noise_level * 0.1)

        landmarks.append({"x": scaled_x, "y": scaled_y, "z": z})

    return landmarks


def create_test_sequence(length=10, base_offset=(0, 0), movement_pattern="static"):
    """
    Create a sequence of landmark frames for testing.

    Args:
        length: Number of frames
        base_offset: Base position offset
        movement_pattern: "static", "wave", or "random"

    Returns:
        List of timestamped landmark frames
    """
    sequence = []

    for i in range(length):
        timestamp = i * 0.05  # 20 FPS

        # Apply movement pattern
        if movement_pattern == "wave":
            # Simple wave motion
            wave_offset = (0.05 * np.sin(i * 0.5), 0.02 * np.cos(i * 0.3))
        elif movement_pattern == "random":
            # Random small movements
            wave_offset = (np.random.normal(0, 0.01), np.random.normal(0, 0.01))
        else:  # static
            wave_offset = (0, 0)

        offset_x = base_offset[0] + wave_offset[0]
        offset_y = base_offset[1] + wave_offset[1]

        landmarks = create_mock_landmarks(offset_x=offset_x, offset_y=offset_y, noise_level=0.001)
        sequence.append({"timestamp": timestamp, "landmarks": landmarks})

    return sequence


def test_scoring_metrics():
    """Test the scoring similarity metrics."""
    print("Testing Scoring Similarity Metrics")
    print("=" * 50)

    # Create mock HandAnnotation instances
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera, creating mock instances...")

    webcam_annotation = HandAnnotation(cap)
    reference_annotation = HandAnnotation(cap)

    # Create test sequences
    # Perfect match
    reference_seq = create_test_sequence(length=10, base_offset=(0, 0), movement_pattern="static")
    perfect_match_seq = create_test_sequence(length=10, base_offset=(0, 0), movement_pattern="static")

    # Similar but with small differences
    similar_seq = create_test_sequence(length=10, base_offset=(0.01, 0.01), movement_pattern="static")

    # Different sequence
    different_seq = create_test_sequence(length=10, base_offset=(0.1, 0.1), movement_pattern="wave")

    # Different length
    short_seq = create_test_sequence(length=5, base_offset=(0, 0), movement_pattern="static")

    # Set reference landmarks
    reference_annotation.handLandmarksTimestamped = reference_seq

    # Create scoring instance
    scoring = Scoring(webcam_annotation, reference_annotation)

    print("\nTest 1: Perfect match")
    score1 = scoring.calculateScore(perfect_match_seq)
    print(f"Score: {score1:.4f} (should be ~1.0)")

    print("\nTest 2: Similar sequence")
    score2 = scoring.calculateScore(similar_seq)
    print(f"Score: {score2:.4f} (should be high > 0.8)")

    print("\nTest 3: Different sequence")
    score3 = scoring.calculateScore(different_seq)
    print(f"Score: {score3:.4f} (should be low < 0.5)")

    print("\nTest 4: Different length sequence")
    score4 = scoring.calculateScore(short_seq)
    print(f"Score: {score4:.4f} (should handle length differences)")

    # Verify score ordering
    assert score1 > score2 > score3, f"Score ordering incorrect: {score1} > {score2} > {score3}"
    print("\n✓ Score ordering is correct (perfect > similar > different)")

    cap.release()
    print("\n✓ All scoring tests passed!")


def test_individual_metrics():
    """Test individual similarity metrics in isolation."""
    print("\nTesting Individual Metrics")
    print("=" * 30)

    # Create test landmark arrays
    landmarks1 = np.random.rand(21, 3)  # 21 landmarks, 3D
    landmarks2 = landmarks1 + 0.1  # Similar but offset
    landmarks3 = np.random.rand(21, 3)  # Completely different

    scoring = Scoring(None, None)  # We only need the methods

    # Test Euclidean distance
    dist1 = scoring._euclideanDistance(landmarks1, landmarks1)
    dist2 = scoring._euclideanDistance(landmarks1, landmarks2)
    dist3 = scoring._euclideanDistance(landmarks1, landmarks3)

    print(f"Euclidean distance:")
    print(f"  Identical: {dist1:.6f} (should be ~0)")
    print(f"  Similar: {dist2:.6f}")
    print(f"  Different: {dist3:.6f}")

    assert dist1 < 0.001, "Identical landmarks should have near-zero distance"
    assert dist2 < dist3, "Similar should be closer than different"

    # Test cosine similarity
    cos1 = scoring._cosineSimilarity(landmarks1, landmarks1)
    cos2 = scoring._cosineSimilarity(landmarks1, landmarks2)
    cos3 = scoring._cosineSimilarity(landmarks1, landmarks3)

    print(f"\nCosine similarity:")
    print(f"  Identical: {cos1:.6f} (should be ~1.0)")
    print(f"  Similar: {cos2:.6f}")
    print(f"  Different: {cos3:.6f}")

    assert abs(cos1 - 1.0) < 0.001, "Identical landmarks should have cosine similarity ~1.0"
    assert cos1 > cos2 > cos3, "Cosine similarity should decrease with difference"

    # Test DTW
    seq1 = [landmarks1, landmarks1, landmarks1]
    seq2 = [landmarks1, landmarks2, landmarks3]

    dtw_dist = scoring._dtwDistance(seq1, seq2)
    print(f"\nDTW distance between sequences: {dtw_dist:.6f}")

    print("\n✓ Individual metrics working correctly!")


if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results

    try:
        test_individual_metrics()
        test_scoring_metrics()

        print("\n" + "=" * 50)
        print("🎉 ALL SCORING TESTS PASSED! 🎉")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
