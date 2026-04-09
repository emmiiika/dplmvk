# AI GENERATED

from HandAnnotation import HandAnnotation
import cv2
import numpy as np
import math


class Scoring:
    def __init__(self, webcamAnnotation, referenceAnnotation):
        self.webcamAnnotation = webcamAnnotation
        self.referenceAnnotation = referenceAnnotation

    def calculateScore(self, user_landmarks=None):
        """
        Calculate similarity score between user landmarks and reference landmarks.

        Args:
            user_landmarks: List of timestamped landmark sequences from user webcam.
                          Format: [{"timestamp": float, "landmarks": [{"x": float, "y": float, "z": float}, ...]}, ...]

        Returns:
            float: Similarity score between 0.0 (no similarity) and 1.0 (perfect match)
        """
        print(
            f"Calculating score, user_landmarks: {len(user_landmarks) if user_landmarks else 0}, reference_landmarks: {len(self.referenceAnnotation.handLandmarksTimestamped)}"
        )

        if user_landmarks is None or len(user_landmarks) == 0:
            # Fallback to current landmarks if no sequence provided
            webcamHandLandmarks = self.webcamAnnotation.getHandLandmarks()
            referenceHandLandmarks = self.referenceAnnotation.getHandLandmarks()
            # Placeholder for single-frame scoring
            score = 0.0
            return score

        # Compare timestamped sequences
        reference_landmarks = self.referenceAnnotation.handLandmarksTimestamped

        if len(reference_landmarks) == 0:
            print("Warning: No reference landmarks available for comparison")
            return 0.0

        # Calculate comprehensive similarity score
        score = self._calculateSequenceSimilarity(user_landmarks, reference_landmarks)

        print(f"Calculated similarity score: {score:.4f}")
        return score

    def _calculateSequenceSimilarity(self, user_sequence, reference_sequence):
        """
        Calculate similarity between two landmark sequences using multiple metrics.

        Args:
            user_sequence: List of timestamped user landmarks
            reference_sequence: List of timestamped reference landmarks

        Returns:
            float: Combined similarity score (0.0 to 1.0)
        """
        # Extract per-frame hand arrays: each frame is (hand0_array_or_None, hand1_array_or_None)
        user_frames = self._extractPerHandArrays(user_sequence)
        ref_frames = self._extractPerHandArrays(reference_sequence)

        if len(user_frames) == 0 or len(ref_frames) == 0:
            return 0.0

        # Compute per-hand motion from reference gesture and use it as comparison weight.
        # If one hand moves more, it gets higher weight.
        hand_weights = self._calculateHandMotionWeights(ref_frames)

        # Estimate motion energy before trimming. This is used to penalize sequences
        # where the user mostly holds the neutral/rest position.
        reference_motion_energy = self._averageMotionEnergy(ref_frames, hand_weights)
        user_motion_energy = self._averageMotionEnergy(user_frames, hand_weights)

        # Remove low-motion start/end segments (relaxed neutral pose) from both sequences
        # so scoring focuses primarily on the actual gesture movement.
        user_frames = self._trimLowMotionEdges(user_frames, hand_weights)
        ref_frames = self._trimLowMotionEdges(ref_frames, hand_weights)

        if len(user_frames) == 0 or len(ref_frames) == 0:
            return 0.0

        # Method 1: DTW for temporal alignment (handles different speeds)
        dtw_distance = self._dtwDistance(user_frames, ref_frames, hand_weights)
        # Tight calibration: DTW distances around ~1 should already be considered fairly dissimilar.
        dtw_similarity = self._distanceToSimilarity(dtw_distance, max_possible_distance=2.0)

        # Method 2: Average Euclidean distance between corresponding frames
        euclidean_distance = self._averageEuclideanDistance(user_frames, ref_frames, hand_weights)
        # Looser calibration to reduce over-penalization from viewpoint changes.
        euclidean_similarity = self._distanceToSimilarity(euclidean_distance, max_possible_distance=1.5)

        # Method 3: Cosine similarity for pose comparison
        cosine_sim = self._averageCosineSimilarity(user_frames, ref_frames, hand_weights)

        # Combine metrics with weights.
        # Favor temporal + structural similarity so the same gesture from different
        # camera viewpoints remains comparably scored.
        combined_score = 0.55 * dtw_similarity + 0.10 * euclidean_similarity + 0.35 * cosine_sim

        # Motion-activity penalty:
        # If the reference contains meaningful motion, but the user sequence stays
        # mostly static (e.g., hands at belly position), reduce final score strongly.
        activity_factor = 1.0
        min_reference_motion = 0.003
        if reference_motion_energy > min_reference_motion:
            expected_user_motion = reference_motion_energy * 0.6
            if expected_user_motion > 0:
                activity_factor = max(0.0, min(1.0, user_motion_energy / expected_user_motion))

        combined_score *= activity_factor

        print(f"  Hand weights (h0, h1): ({hand_weights[0]:.3f}, {hand_weights[1]:.3f})")
        print(f"  Frames used (user/reference): {len(user_frames)}/{len(ref_frames)}")
        print(f"  Motion energy (user/reference): {user_motion_energy:.5f}/{reference_motion_energy:.5f}")
        print(f"  Activity factor: {activity_factor:.3f}")
        print(f"  DTW similarity: {dtw_similarity:.4f}")
        print(f"  Euclidean similarity: {euclidean_similarity:.4f}")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        print(f"  Combined score: {combined_score:.4f}")

        return combined_score

    def _extractPerHandArrays(self, sequence):
        """
        Extract per-frame hand landmark arrays.

        Args:
            sequence: List of frames in one of these formats:
                - {"timestamp": float, "landmarks": list_of_dicts}
                - (timestamp, landmarks) or [timestamp, landmarks]
              where landmarks can be either:
                - list_of_dicts (flattened landmarks)
                - list_of_hands, each hand is list_of_dicts

        Returns:
            List of tuples (hand0, hand1), each hand is np.ndarray shape (21, 3) or None.
        """
        frames = []
        for frame in sequence:
            landmarks = []

            # Frame can be dict: {"timestamp": ..., "landmarks": ...}
            if isinstance(frame, dict):
                landmarks = frame.get("landmarks", [])
            # Frame can be tuple/list: (timestamp, landmarks)
            elif isinstance(frame, (tuple, list)) and len(frame) >= 2:
                landmarks = frame[1]

            if len(landmarks) == 0:
                continue

            hands = self._extractHandsFromFrameLandmarks(landmarks)
            if len(hands) == 0:
                continue

            hand0 = hands[0] if len(hands) > 0 else None
            hand1 = hands[1] if len(hands) > 1 else None
            frames.append((hand0, hand1))

        return frames

    def _extractHandsFromFrameLandmarks(self, landmarks):
        """
        Convert frame landmark payload into up to two hand arrays.

        Args:
            landmarks: Either list_of_dicts or list_of_hands(list_of_dicts)

        Returns:
            List of np.ndarray hands, each shape (21, 3)
        """
        hands = []

        # Flat format: list of dict landmarks (possibly flattened two hands: 42 points)
        if isinstance(landmarks[0], dict):
            hand_count = len(landmarks) // 21
            for i in range(min(hand_count, 2)):
                chunk = landmarks[i * 21 : (i + 1) * 21]
                if len(chunk) == 21:
                    hands.append(np.array([[lm["x"], lm["y"], lm["z"]] for lm in chunk]))
            return hands

        # Nested format: list of hands
        for hand in landmarks:
            if isinstance(hand, list) and len(hand) >= 21 and isinstance(hand[0], dict):
                chunk = hand[:21]
                hands.append(np.array([[lm["x"], lm["y"], lm["z"]] for lm in chunk]))
                if len(hands) == 2:
                    break

        return hands

    def _calculateHandMotionWeights(self, reference_frames):
        """
        Compute motion-based weights for hand 0 and hand 1 from reference sequence.

        Higher motion means higher comparison weight.
        """
        motion = [0.0, 0.0]
        presence = [0, 0]

        for frame in reference_frames:
            for idx in (0, 1):
                if frame[idx] is not None:
                    presence[idx] += 1

        for idx in (0, 1):
            prev = None
            total = 0.0
            count = 0
            for frame in reference_frames:
                curr = frame[idx]
                if curr is None:
                    continue
                if prev is not None and prev.shape == curr.shape:
                    total += float(np.mean(np.linalg.norm(curr - prev, axis=1)))
                    count += 1
                prev = curr

            motion[idx] = total / count if count > 0 else 0.0

        motion_sum = motion[0] + motion[1]
        if motion_sum > 0:
            return [motion[0] / motion_sum, motion[1] / motion_sum]

        presence_sum = presence[0] + presence[1]
        if presence_sum > 0:
            return [presence[0] / presence_sum, presence[1] / presence_sum]

        return [0.5, 0.5]

    def _trimLowMotionEdges(self, frames, hand_weights, motion_floor_ratio=0.2, min_active_frames=5):
        """
        Trim low-motion frames at the beginning and end of a sequence.

        This reduces the influence of neutral/rest hand pose before and after gesture execution.
        """
        if len(frames) <= min_active_frames:
            return frames

        motion = [0.0]
        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            energy = 0.0

            for idx in (0, 1):
                w = hand_weights[idx]
                if w <= 0:
                    continue

                prev_hand = prev_frame[idx]
                curr_hand = curr_frame[idx]
                if prev_hand is None or curr_hand is None:
                    continue

                energy += w * float(np.mean(np.linalg.norm(curr_hand - prev_hand, axis=1)))

            motion.append(energy)

        max_motion = max(motion)
        if max_motion <= 1e-8:
            return frames

        threshold = max(max_motion * motion_floor_ratio, 1e-4)
        active_indices = [i for i, e in enumerate(motion) if e >= threshold]

        if not active_indices:
            return frames

        start = max(0, active_indices[0] - 1)
        end = min(len(frames), active_indices[-1] + 2)

        trimmed = frames[start:end]
        if len(trimmed) < min_active_frames:
            return frames

        return trimmed

    def _averageMotionEnergy(self, frames, hand_weights):
        """
        Compute average frame-to-frame motion energy for a sequence.

        Uses weighted per-hand mean landmark displacement.
        """
        if len(frames) < 2:
            return 0.0

        total = 0.0
        count = 0

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]
            energy = 0.0

            for idx in (0, 1):
                w = hand_weights[idx]
                if w <= 0:
                    continue

                prev_hand = prev_frame[idx]
                curr_hand = curr_frame[idx]
                if prev_hand is None or curr_hand is None:
                    continue

                energy += w * float(np.mean(np.linalg.norm(curr_hand - prev_hand, axis=1)))

            total += energy
            count += 1

        if count == 0:
            return 0.0

        return total / count

    def _dtwDistance(self, seq1, seq2, hand_weights=None):
        """
        Calculate Dynamic Time Warping distance between two sequences.

        DTW finds optimal alignment between sequences of different lengths,
        allowing for different gesture speeds.

        Args:
            seq1, seq2: Either:
                - lists of per-frame tuples (hand0, hand1), or
                - lists of ndarray landmarks (backward-compatible mode)
            hand_weights: [weight_hand0, weight_hand1] for per-hand mode

        Returns:
            float: Length-normalized DTW distance (lower = more similar)
        """
        n = len(seq1)
        m = len(seq2)

        if n == 0 or m == 0:
            return float("inf")

        if hand_weights is None:
            hand_weights = [0.5, 0.5]

        # Initialize DTW matrix
        dtw_matrix = np.full((n + 1, m + 1), float("inf"))
        dtw_matrix[0, 0] = 0

        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = seq1[i - 1]
                right = seq2[j - 1]

                # Backward-compatible mode: plain ndarray sequence.
                if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                    cost = self._euclideanDistance(left, right)
                else:
                    # Per-hand mode.
                    cost = self._weightedFrameDistance(left, right, hand_weights)

                # DTW recurrence relation
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],  # insertion
                    dtw_matrix[i, j - 1],  # deletion
                    dtw_matrix[i - 1, j - 1],  # match
                )

        # Normalize cumulative DTW cost by sequence length so longer gestures
        # are not unfairly penalized compared to shorter ones.
        return dtw_matrix[n, m] / (n + m)

    def _euclideanDistance(self, landmarks1, landmarks2):
        """
        Calculate Euclidean distance between two sets of landmarks.

        Args:
            landmarks1, landmarks2: Numpy arrays of shape (21, 3)

        Returns:
            float: Average Euclidean distance across all landmarks
        """
        if landmarks1.shape != landmarks2.shape:
            return float("inf")

        # Calculate Euclidean distance for each landmark pair
        distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)

        # Return average distance across all landmarks
        return np.mean(distances)

    def _weightedFrameDistance(self, frame1, frame2, hand_weights):
        """Compute weighted per-frame distance with mild penalty for missing hand."""
        total = 0.0
        missing_hand_penalty = 1.0

        for idx in (0, 1):
            w = hand_weights[idx]
            if w <= 0:
                continue

            h1 = frame1[idx]
            h2 = frame2[idx]

            if h1 is None and h2 is None:
                dist = 0.0
            elif h1 is None or h2 is None:
                dist = missing_hand_penalty
            else:
                dist = self._euclideanDistance(h1, h2)

            total += w * dist

        return total

    def _averageEuclideanDistance(self, seq1, seq2, hand_weights):
        """
        Calculate average Euclidean distance between corresponding frames.

        This assumes sequences are roughly the same length and timing.

        Args:
            seq1, seq2: Lists of per-frame tuples (hand0, hand1)
            hand_weights: [weight_hand0, weight_hand1]

        Returns:
            float: Average Euclidean distance
        """
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return float("inf")

        total_distance = 0
        for i in range(min_len):
            total_distance += self._weightedFrameDistance(seq1[i], seq2[i], hand_weights)

        return total_distance / min_len

    def _cosineSimilarity(self, landmarks1, landmarks2):
        """
        Calculate cosine similarity between two landmark configurations.

        Cosine similarity measures the angle between landmark vectors,
        being more robust to scale differences than Euclidean distance.

        Args:
            landmarks1, landmarks2: Numpy arrays of shape (21, 3)

        Returns:
            float: Cosine similarity (-1 to 1, higher = more similar)
        """
        if landmarks1.shape != landmarks2.shape:
            return -1.0

        # Flatten landmarks into vectors
        vec1 = landmarks1.flatten()
        vec2 = landmarks2.flatten()

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _weightedFrameCosineSimilarity(self, frame1, frame2, hand_weights):
        """Compute weighted per-frame cosine similarity in [0, 1]."""
        total = 0.0

        for idx in (0, 1):
            w = hand_weights[idx]
            if w <= 0:
                continue

            h1 = frame1[idx]
            h2 = frame2[idx]

            if h1 is None and h2 is None:
                sim01 = 1.0
            elif h1 is None or h2 is None:
                sim01 = 0.0
            else:
                sim = self._cosineSimilarity(h1, h2)
                sim01 = (sim + 1) / 2

            total += w * sim01

        return total

    def _averageCosineSimilarity(self, seq1, seq2, hand_weights):
        """
        Calculate average cosine similarity across sequences.

        Args:
            seq1, seq2: Lists of per-frame tuples (hand0, hand1)
            hand_weights: [weight_hand0, weight_hand1]

        Returns:
            float: Average cosine similarity (converted to 0-1 range)
        """
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0

        total_similarity = 0
        for i in range(min_len):
            total_similarity += self._weightedFrameCosineSimilarity(seq1[i], seq2[i], hand_weights)

        return total_similarity / min_len

    def _distanceToSimilarity(self, distance, max_possible_distance=1.0):
        """
        Convert a distance metric to a similarity score (0.0 to 1.0).

        Args:
            distance: Raw distance value (higher = less similar)
            max_possible_distance: Expected maximum reasonable distance

        Returns:
            float: Similarity score (0.0 = no similarity, 1.0 = perfect match)
        """
        if distance == float("inf") or distance < 0:
            return 0.0

        # Exponential decay: similarity = e^(-distance/scale)
        # This gives high similarity for small distances, low for large ones
        scale = max_possible_distance / 3  # 3-sigma rule for reasonable similarity
        similarity = math.exp(-distance / scale)

        return min(similarity, 1.0)
