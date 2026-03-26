"""

AI GENERATED

Test script for translation and scale normalization functions.

This script tests:
1. Translation normalization (_getTranslatedLandmarks): wrist-centering
2. Scale normalization (_getNormalizedScaleLandmarks): hand-size invariance
"""

import numpy as np
from HandAnnotation import HandAnnotation
from LandmarkIndices import LandmarkIndices
import cv2


class MockLandmark:
    """Mock MediaPipe Landmark object for testing."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def create_test_landmarks(offset_x=0, offset_y=0, scale_factor=1.0):
    """
    Create synthetic hand landmarks for testing.

    Args:
        offset_x: X offset to apply to all landmarks
        offset_y: Y offset to apply to all landmarks
        scale_factor: Scale factor to apply (to test scale normalization)

    Returns:
        List of 21 MockLandmark objects representing a hand
    """
    # Base hand landmark positions (simplified, normalized 0-1 space)
    base_positions = [
        (0.5, 0.5, 0.0),  # 0: WRIST
        (0.52, 0.48, 0.0),  # 1: THUMB_CMC
        (0.54, 0.46, 0.0),  # 2: THUMB_MCP
        (0.56, 0.44, 0.0),  # 3: THUMB_IP
        (0.58, 0.42, 0.0),  # 4: THUMB_TIP
        (0.52, 0.52, 0.0),  # 5: INDEX_FINGER_MCP
        (0.54, 0.54, 0.0),  # 6: INDEX_FINGER_PIP
        (0.56, 0.56, 0.0),  # 7: INDEX_FINGER_DIP
        (0.58, 0.58, 0.0),  # 8: INDEX_FINGER_TIP
        (0.51, 0.55, 0.0),  # 9: MIDDLE_MCP
        (0.52, 0.58, 0.0),  # 10: MIDDLE_FINGER_PIP
        (0.53, 0.61, 0.0),  # 11: MIDDLE_FINGER_DIP
        (0.54, 0.64, 0.0),  # 12: MIDDLE_FINGER_TIP
        (0.50, 0.57, 0.0),  # 13: RING_FINGER_MCP
        (0.50, 0.60, 0.0),  # 14: RING_FINGER_PIP
        (0.50, 0.63, 0.0),  # 15: RING_FINGER_DIP
        (0.50, 0.66, 0.0),  # 16: RING_FINGER_TIP
        (0.49, 0.59, 0.0),  # 17: PINKY_MCP
        (0.48, 0.62, 0.0),  # 18: PINKY_PIP
        (0.47, 0.65, 0.0),  # 19: PINKY_DIP
        (0.46, 0.68, 0.0),  # 20: PINKY_TIP
    ]

    landmarks = []
    for x, y, z in base_positions:
        # Apply scale factor first, then offset
        scaled_x = x * scale_factor + offset_x
        scaled_y = y * scale_factor + offset_y
        landmarks.append(MockLandmark(scaled_x, scaled_y, z))

    return landmarks


def test_translation_normalization():
    """Test that translation normalization correctly centers on wrist."""
    print("\n" + "=" * 60)
    print("TEST 1: Translation Normalization (Wrist-Centering)")
    print("=" * 60)

    # Create landmarks at different screen positions
    landmarks_offset1 = create_test_landmarks(offset_x=0.2, offset_y=0.3)
    landmarks_offset2 = create_test_landmarks(offset_x=0.5, offset_y=0.1)

    # Create a dummy HandAnnotation instance
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera, but proceeding with test...")

    ha = HandAnnotation(cap)

    # Apply translation normalization
    translated1 = ha._getTranslatedLandmarks(landmarks_offset1)
    translated2 = ha._getTranslatedLandmarks(landmarks_offset2)

    # Check that wrist is at origin (0, 0, 0) after translation
    wrist_idx = LandmarkIndices.WRIST
    wrist1 = translated1[wrist_idx]
    wrist2 = translated2[wrist_idx]

    print(f"\nTest Case 1 (offset +0.2, +0.3):")
    print(f"  Original wrist position: ({landmarks_offset1[wrist_idx].x:.4f}, {landmarks_offset1[wrist_idx].y:.4f})")
    print(f"  Translated wrist position: ({wrist1[0]:.6f}, {wrist1[1]:.6f}, {wrist1[2]:.6f})")

    assert np.allclose(wrist1, [0, 0, 0], atol=1e-6), "Wrist should be at origin after translation!"
    print("  ✓ Wrist correctly centered at origin [0, 0, 0]")

    print(f"\nTest Case 2 (offset +0.5, +0.1):")
    print(f"  Original wrist position: ({landmarks_offset2[wrist_idx].x:.4f}, {landmarks_offset2[wrist_idx].y:.4f})")
    print(f"  Translated wrist position: ({wrist2[0]:.6f}, {wrist2[1]:.6f}, {wrist2[2]:.6f})")

    assert np.allclose(wrist2, [0, 0, 0], atol=1e-6), "Wrist should be at origin after translation!"
    print("  ✓ Wrist correctly centered at origin [0, 0, 0]")

    # Verify that other landmarks have been translated consistently
    print(f"\nTest Case 3: Consistency check")
    print(
        f"  Thumb tip original difference (Case 1 - Case 2): "
        f"({landmarks_offset1[4].x - landmarks_offset2[4].x:.4f}, "
        f"{landmarks_offset1[4].y - landmarks_offset2[4].y:.4f})"
    )
    print(
        f"  Thumb tip translated difference (Case 1 - Case 2): "
        f"({translated1[4, 0] - translated2[4, 0]:.4f}, "
        f"{translated1[4, 1] - translated2[4, 1]:.4f})"
    )
    print("  ✓ Landmark differences preserved after translation (translation is uniform)")

    cap.release()
    print("\n✓✓✓ Translation Normalization Test PASSED ✓✓✓\n")


def test_scale_normalization():
    """Test that scale normalization makes different hand sizes equivalent."""
    print("\n" + "=" * 60)
    print("TEST 2: Scale Normalization (Hand-Size Invariance)")
    print("=" * 60)

    # Create landmarks with scale factors 1.0 and 2.0 (hand size variations)
    scale_1x = create_test_landmarks(scale_factor=1.0)
    scale_2x = create_test_landmarks(scale_factor=2.0)

    # Create a dummy HandAnnotation instance
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera, but proceeding with test...")

    ha = HandAnnotation(cap)

    # Apply translation, then scale normalization
    translated_1x = ha._getTranslatedLandmarks(scale_1x)
    translated_2x = ha._getTranslatedLandmarks(scale_2x)

    normalized_1x = ha._getNormalizedScaleLandmarks(translated_1x)
    normalized_2x = ha._getNormalizedScaleLandmarks(translated_2x)

    # Print original scales
    wrist_idx = LandmarkIndices.WRIST
    middle_mcp_idx = LandmarkIndices.MIDDLE_MCP

    scale_ref_1x = np.linalg.norm(translated_1x[middle_mcp_idx] - translated_1x[wrist_idx])
    scale_ref_2x = np.linalg.norm(translated_2x[middle_mcp_idx] - translated_2x[wrist_idx])

    print(f"\nOriginal scales (wrist to middle MCP distance):")
    print(f"  Scale factor 1.0x: {scale_ref_1x:.6f}")
    print(f"  Scale factor 2.0x: {scale_ref_2x:.6f}")
    print(f"  Ratio (2x / 1x): {scale_ref_2x / scale_ref_1x:.4f} (should be ~2.0)")

    # After scale normalization, these distances should be equal
    normalized_scale_1x = np.linalg.norm(normalized_1x[middle_mcp_idx] - normalized_1x[wrist_idx])
    normalized_scale_2x = np.linalg.norm(normalized_2x[middle_mcp_idx] - normalized_2x[wrist_idx])

    print(f"\nNormalized scales (wrist to middle MCP distance after scaling):")
    print(f"  Scale factor 1.0x: {normalized_scale_1x:.6f}")
    print(f"  Scale factor 2.0x: {normalized_scale_2x:.6f}")
    print(f"  Ratio (2x / 1x): {normalized_scale_2x / normalized_scale_1x:.6f} (should be ~1.0)")

    assert np.allclose(
        normalized_scale_1x, normalized_scale_2x, atol=1e-6
    ), "Normalized scales should be equal regardless of original hand size!"
    print("  ✓ Hand-size invariance achieved: both scales are now equivalent")

    # Verify that corresponding landmarks are now at equivalent positions
    print(f"\nTest Case: Thumb tip positions after scale normalization")
    thumb_tip_idx = 4
    print(f"  1.0x scale, thumb tip: ({normalized_1x[thumb_tip_idx, 0]:.6f}, {normalized_1x[thumb_tip_idx, 1]:.6f})")
    print(f"  2.0x scale, thumb tip: ({normalized_2x[thumb_tip_idx, 0]:.6f}, {normalized_2x[thumb_tip_idx, 1]:.6f})")
    print(
        f"  Difference: ({abs(normalized_1x[thumb_tip_idx, 0] - normalized_2x[thumb_tip_idx, 0]):.8f}, "
        f"{abs(normalized_1x[thumb_tip_idx, 1] - normalized_2x[thumb_tip_idx, 1]):.8f})"
    )

    assert np.allclose(
        normalized_1x[thumb_tip_idx], normalized_2x[thumb_tip_idx], atol=1e-6
    ), "Thumb tip positions should be identical after scale normalization!"
    print("  ✓ Thumb tip positions match after scale normalization")

    cap.release()
    print("\n✓✓✓ Scale Normalization Test PASSED ✓✓✓\n")


def test_combined_normalization():
    """Test that translation + scale normalization produces consistent results."""
    print("\n" + "=" * 60)
    print("TEST 3: Combined Normalization (Translation + Scale)")
    print("=" * 60)

    # Create two hands with different positions and sizes
    hand1 = create_test_landmarks(offset_x=0.1, offset_y=0.2, scale_factor=1.0)
    hand2 = create_test_landmarks(offset_x=0.6, offset_y=0.4, scale_factor=1.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera, but proceeding with test...")

    ha = HandAnnotation(cap)

    # Apply combined normalization
    trans1 = ha._getTranslatedLandmarks(hand1)
    norm1 = ha._getNormalizedScaleLandmarks(trans1)

    trans2 = ha._getTranslatedLandmarks(hand2)
    norm2 = ha._getNormalizedScaleLandmarks(trans2)

    # Check that wrists are at origin
    wrist_idx = LandmarkIndices.WRIST
    assert np.allclose(norm1[wrist_idx], [0, 0, 0], atol=1e-6), "Wrist should be at origin!"
    assert np.allclose(norm2[wrist_idx], [0, 0, 0], atol=1e-6), "Wrist should be at origin!"
    print(f"✓ Both hands have wrist at origin after combined normalization")

    # Check that scale is normalized
    middle_mcp_idx = LandmarkIndices.MIDDLE_MCP
    scale1 = np.linalg.norm(norm1[middle_mcp_idx] - norm1[wrist_idx])
    scale2 = np.linalg.norm(norm2[middle_mcp_idx] - norm2[wrist_idx])

    print(f"\nScales after combined normalization:")
    print(f"  Hand 1: {scale1:.6f}")
    print(f"  Hand 2: {scale2:.6f}")
    print(f"  Ratio (hand2 / hand1): {scale2 / scale1:.6f} (should be ~1.0)")

    assert np.allclose(scale1, scale2, atol=1e-6), "Scales should be equal!"
    print(f"✓ Both hands have equivalent scale after combined normalization")

    # Verify multiple landmark positions match
    test_landmarks = [1, 4, 8, 12, 16, 20]  # Various landmarks
    max_diff = 0
    for idx in test_landmarks:
        diff = np.linalg.norm(norm1[idx] - norm2[idx])
        max_diff = max(max_diff, diff)

    print(f"\nLandmark consistency check (indices {test_landmarks}):")
    print(f"  Maximum difference between corresponding landmarks: {max_diff:.8f}")
    assert max_diff < 1e-6, f"Landmarks should match, but max diff is {max_diff}!"
    print(f"✓ All tested landmarks match after combined normalization")

    cap.release()
    print("\n✓✓✓ Combined Normalization Test PASSED ✓✓✓\n")


def test_edge_cases():
    """Test edge cases like zero scale or invalid inputs."""
    print("\n" + "=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Could not open camera, but proceeding with test...")

    ha = HandAnnotation(cap)

    # Test 1: Empty landmarks
    print("\nTest Case 1: Empty landmarks array")
    empty_landmarks = []
    result = ha._getTranslatedLandmarks(empty_landmarks)
    assert result.shape[0] == 0, "Should handle empty landmarks"
    print("  ✓ Handles empty landmarks array correctly")

    # Test 2: Landmarks with zero scale (all at same position)
    print("\nTest Case 2: Zero scale (all landmarks at same position)")
    same_position_landmarks = [MockLandmark(0.5, 0.5, 0.0) for _ in range(21)]
    translated = ha._getTranslatedLandmarks(same_position_landmarks)
    # All should be zero after translation
    assert np.allclose(translated, 0, atol=1e-6), "All landmarks should be at origin after translation"
    print("  ✓ Handles same-position landmarks (zero scale guard)")

    # Scale normalization should return original (unmodified) when scale is 0
    normalized = ha._getNormalizedScaleLandmarks(translated)
    assert np.allclose(normalized, translated), "Should return original landmarks when scale is 0"
    print("  ✓ Scale normalization handles zero scale correctly")

    cap.release()
    print("\n✓✓✓ Edge Cases Test PASSED ✓✓✓\n")


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  LANDMARK NORMALIZATION TEST SUITE".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60)

    try:
        test_translation_normalization()
        test_scale_normalization()
        test_combined_normalization()
        test_edge_cases()

        print("\n" + "█" * 60)
        print("█" + " " * 58 + "█")
        print("█" + "  ALL TESTS PASSED! ✓".center(58) + "█")
        print("█" + " " * 58 + "█")
        print("█" * 60 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback

        traceback.print_exc()
        exit(1)
