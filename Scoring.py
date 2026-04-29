import numpy as np
import math


# Scoring strategy presets — change the strategy name to switch behaviour.
# Each strategy defines the tunable calibration parameters.
# This is not a part of the aplication itself, it exists just to experiment with
#   different scoring parameters without changing the code.
SCORING_STRATEGIES = {
    "original": {
        "euclideanMaxDistance": 1.5,
        "euclideanWeight": 0.10,
        "cosineWeight": 0.35,
        "activityThreshold": 0.6,
    },
    "relaxed_distances": {
        "euclideanMaxDistance": 3.0,
        "euclideanWeight": 0.10,
        "cosineWeight": 0.35,
        "activityThreshold": 0.6,
    },
    "lower_activity": {
        "euclideanMaxDistance": 1.5,
        "euclideanWeight": 0.10,
        "cosineWeight": 0.35,
        "activityThreshold": 0.35,
    },
    "cosine_heavy": {
        "euclideanMaxDistance": 1.5,
        "euclideanWeight": 0.10,
        "cosineWeight": 0.45,
        "activityThreshold": 0.6,
    },
    "all_combined": {
        "euclideanMaxDistance": 3.0,
        "euclideanWeight": 0.10,
        "cosineWeight": 0.45,
        "activityThreshold": 0.35,
    },
}


class Scoring:
    TRIM_MOTION_FLOOR_RATIO = 0.4  # fraction of peak motion below which a frame is considered idle
    TRIM_PADDING_FRAMES = 2  # extra frames kept before/after the detected active region
    TRIM_MIN_ACTIVE_FRAMES = 5  # minimum frames to retain; skip trimming if result would be shorter
    TRIM_ABSOLUTE_MOTION_FLOOR = 0.02  # minimum motion energy to count as genuine movement (not jitter)

    def __init__(self, webcamAnnotation, referenceAnnotation, strategy="original"):
        """Initialize scoring with user and reference annotation sources.

        Args:
            webcamAnnotation: Annotation provider for user/webcam landmarks.
            referenceAnnotation: Annotation provider for reference landmarks and markers.
            strategy: Name of a scoring strategy from SCORING_STRATEGIES.
        """
        self.webcamAnnotation = webcamAnnotation
        self.referenceAnnotation = referenceAnnotation
        if strategy not in SCORING_STRATEGIES:
            raise ValueError(f"Unknown scoring strategy '{strategy}'. " f"Available: {list(SCORING_STRATEGIES.keys())}")
        self.strategy = SCORING_STRATEGIES[strategy]
        print(f"Scoring strategy: {strategy}")

    # ------------------------------------------------------------------
    # Reference sequence preparation
    # ------------------------------------------------------------------

    def _trimReferenceSequenceByMarkers(self, referenceSequence):
        """Keep only the marker-selected interval from the reference sequence.

        Marker positions are expected in 0-1000 scale (same as progress bar).

        Args:
            referenceSequence: Timestamped reference landmark sequence.

        Returns:
            Trimmed reference sequence restricted to marker boundaries.
        """
        if not referenceSequence:
            return referenceSequence

        # Marker values come from the UI progress-bar scale (0..1000).
        markerStart = getattr(self.referenceAnnotation, "markerStart", 0)
        markerEnd = getattr(self.referenceAnnotation, "markerEnd", 1000)

        # Clamp to valid range and normalize order.
        markerStart = int(max(0, min(1000, markerStart)))
        markerEnd = int(max(0, min(1000, markerEnd)))
        if markerEnd < markerStart:
            markerStart, markerEnd = markerEnd, markerStart

        total = len(referenceSequence)
        if total <= 1:
            return referenceSequence

        # Convert marker positions to inclusive frame indices.
        startIdx = int((markerStart / 1000.0) * (total - 1))
        endIdx = int((markerEnd / 1000.0) * (total - 1))

        # Keep indices safe and ensure end is not before start.
        startIdx = max(0, min(total - 1, startIdx))
        endIdx = max(startIdx, min(total - 1, endIdx))

        trimmed = referenceSequence[startIdx : endIdx + 1]
        print(f"Reference trim by markers: {startIdx}-{endIdx} ({len(trimmed)}/{total} frames retained)")
        return trimmed

    # ------------------------------------------------------------------
    # Landmark extraction
    # ------------------------------------------------------------------

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
            handCount = len(landmarks) // 21
            for i in range(min(handCount, 2)):
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

            # Skip frame if no hand detected
            if not landmarks or (isinstance(landmarks, list) and len(landmarks) == 0):
                continue

            hands = self._extractHandsFromFrameLandmarks(landmarks)
            # Skip frame if no valid hand arrays extracted
            if not hands or all(h is None or (isinstance(h, np.ndarray) and h.size == 0) for h in hands):
                continue

            hand0 = hands[0] if len(hands) > 0 else None
            hand1 = hands[1] if len(hands) > 1 else None
            # Only include frame if at least one hand is present and non-empty
            if (hand0 is not None and (not isinstance(hand0, np.ndarray) or hand0.size > 0)) or (
                hand1 is not None and (not isinstance(hand1, np.ndarray) or hand1.size > 0)
            ):
                frames.append((hand0, hand1))

        return frames

    def _extractRawWristSequences(self, sequence):
        """Extract per-frame raw wrist positions from a timestamped landmark sequence.

        Returns a list of (wrist0, wrist1) tuples where each wrist is [x,y,z] or None.
        """
        result = []
        for frame in sequence:
            wrists = frame[2] if isinstance(frame, (tuple, list)) and len(frame) > 2 else None
            if wrists is None:
                result.append((None, None))
                continue
            w0 = wrists[0] if len(wrists) > 0 else None
            w1 = wrists[1] if len(wrists) > 1 else None
            result.append((w0, w1))
        return result

    def _extractWristTrajectory(self, annotation):
        """Extract wrist trajectories for both hands from an annotation.

        Reads raw (pre-normalization) wrist positions stored as the 3rd element of
        each entry in ``handLandmarksTimestamped`` (populated by HandAnnotation when
        the JSON file was generated with wrist data).

        Args:
            annotation: A HandAnnotation instance (webcam or reference).

        Returns:
            Tuple (traj0, traj1) where each is a list of [x, y, z] per frame.
            Frames where a hand was not detected contain None.
        """
        traj0 = []  # type: ignore
        traj1 = []  # type: ignore
        for frame in getattr(annotation, "handLandmarksTimestamped", []):
            # frame[2] is the list of raw wrist positions per hand (or absent in old files)
            wrists = frame[2] if isinstance(frame, (tuple, list)) and len(frame) > 2 else None
            for idx, traj in enumerate([traj0, traj1]):
                if wrists is not None and len(wrists) > idx and wrists[idx] is not None:
                    traj.append(wrists[idx])
                else:
                    traj.append(None)
        return traj0, traj1

    # ------------------------------------------------------------------
    # Motion and activity analysis
    # ------------------------------------------------------------------

    def _calculateHandMotionWeights(self, referenceFrames):
        """
        Compute motion-based weights for hand 0 and hand 1 from reference sequence.

        Higher motion means higher comparison weight.

        Args:
            referenceFrames: Sequence of per-frame tuples (hand0, hand1), each hand as ndarray or None.

        Returns:
            Two-element list of normalized hand weights [w0, w1].
        """
        motion = [0.0, 0.0]
        presence = [0, 0]

        # Count how often each hand is visible as a fallback signal.
        for frame in referenceFrames:
            for idx in (0, 1):
                if frame[idx] is not None:
                    presence[idx] += 1

        # Estimate per-hand motion from frame-to-frame landmark displacement.
        for idx in (0, 1):
            prev = None
            total = 0.0
            count = 0
            for frame in referenceFrames:
                curr = frame[idx]
                if curr is None:
                    continue
                if prev is not None and prev.shape == curr.shape:
                    total += float(np.mean(np.linalg.norm(curr - prev, axis=1)))
                    count += 1
                prev = curr

            motion[idx] = total / count if count > 0 else 0.0

        # Prefer motion-based weights, then fall back to visibility-based weights.
        motionSum = motion[0] + motion[1]
        if motionSum > 0:
            return [motion[0] / motionSum, motion[1] / motionSum]

        presenceSum = presence[0] + presence[1]
        if presenceSum > 0:
            return [presence[0] / presenceSum, presence[1] / presenceSum]

        return [0.5, 0.5]

    def _averageMotionEnergy(self, frames, handWeights):
        """
        Compute median frame-to-frame motion energy for a sequence.

        Uses the median (not mean) of weighted per-hand landmark displacements
        to be robust against outlier spikes from tracker glitches (hand loss,
        handedness flips, re-detection jumps).

        Args:
            frames: Sequence of per-frame tuples (hand0, hand1).
            handWeights: Two-element list of per-hand weights.

        Returns:
            float: Median weighted motion energy across consecutive frames.
        """
        # Need at least two frames to compute frame-to-frame motion.
        if len(frames) < 2:
            return 0.0

        energies = []

        # Collect weighted displacement between consecutive frames.
        for i in range(1, len(frames)):
            prevFrame = frames[i - 1]
            currFrame = frames[i]
            energy = 0.0

            for idx in (0, 1):
                w = handWeights[idx]
                if w <= 0:
                    continue

                prevHand = prevFrame[idx]
                currHand = currFrame[idx]
                if prevHand is None or currHand is None:
                    continue

                energy += w * float(np.mean(np.linalg.norm(currHand - prevHand, axis=1)))

            energies.append(energy)

        # Guard against empty effective comparisons (e.g., missing hands everywhere).
        if len(energies) == 0:
            return 0.0

        # Median is robust to outlier spikes from tracker glitches.
        return float(np.median(energies))

    def _trimLowMotionEdges(self, frames, handWeights):
        """
        Trim low-motion frames at the beginning and end of a sequence.

        This reduces the influence of neutral/rest hand pose before and after gesture execution.
        Uses class constants TRIM_MOTION_FLOOR_RATIO, TRIM_PADDING_FRAMES, and
        TRIM_MIN_ACTIVE_FRAMES (analogous to HandAnnotation.MOVEMENT_THRESHOLD /
        TRIM_PADDING_SECONDS).

        Args:
            frames: Sequence of per-frame tuples (hand0, hand1), each hand as ndarray or None.
            handWeights: Two-element list of per-hand weights used to compute motion energy.

        Returns:
            Trimmed frame sequence, or the original sequence when trimming is not reliable.
        """
        if len(frames) <= self.TRIM_MIN_ACTIVE_FRAMES:
            return frames

        # Per-frame weighted motion energy (frame 0 has no previous frame).
        motion = [0.0]
        for i in range(1, len(frames)):
            prevFrame = frames[i - 1]
            currFrame = frames[i]
            energy = 0.0

            for idx in (0, 1):
                w = handWeights[idx]
                if w <= 0:
                    continue

                prevHand = prevFrame[idx]
                currHand = currFrame[idx]
                if prevHand is None or currHand is None:
                    continue

                energy += w * float(np.mean(np.linalg.norm(currHand - prevHand, axis=1)))

            motion.append(energy)

        # If sequence is essentially static, keep it unchanged.
        maxMotion = max(motion)
        if maxMotion <= 1e-8:
            return frames

        # Use the higher of the relative floor (fraction of peak) and the absolute floor.
        # The absolute floor ensures that pure jitter (static user) isn't mistaken for motion.
        threshold = max(maxMotion * self.TRIM_MOTION_FLOOR_RATIO, self.TRIM_ABSOLUTE_MOTION_FLOOR)
        activeIndices = [i for i, e in enumerate(motion) if e >= threshold]

        if not activeIndices:
            return frames

        # Expand by padding so transitions around active motion are preserved.
        pad = self.TRIM_PADDING_FRAMES
        start = max(0, activeIndices[0] - pad)
        end = min(len(frames), activeIndices[-1] + pad + 1)

        trimmed = frames[start:end]
        # Avoid returning overly short clips that could destabilize scoring.
        if len(trimmed) < self.TRIM_MIN_ACTIVE_FRAMES:
            return frames

        return trimmed

    # ------------------------------------------------------------------
    # Wrist displacement
    # ------------------------------------------------------------------

    def _wristMaxDisplacement(self, traj):
        """Compute the maximum wrist displacement from the starting position.

        Returns the largest distance from the first valid point to any later
        point in the trajectory.  This captures gestures that move away from
        the start and return (net displacement = 0) while still being large.

        Args:
            traj: List of [x, y, z] array-likes. None entries are skipped.

        Returns:
            float >= 0, or None if fewer than 2 valid points.
        """
        pts = [p for p in traj if p is not None]
        if len(pts) < 2:
            return None
        origin = np.array(pts[0], dtype=float)
        dists = [np.linalg.norm(np.array(p, dtype=float) - origin).item() for p in pts[1:]]
        return float(max(dists))

    def _wristMaxDispFromRawSeq(self, rawSeq, handWeights):
        """Compute weighted maximum wrist displacement from raw wrist sequences.

        Args:
            rawSeq: List of (wrist0, wrist1) tuples, each wrist is [x,y,z] or None.
            handWeights: [weight_hand0, weight_hand1].

        Returns:
            float >= 0, or None if no wrist data is available.
        """
        total = 0.0
        weightSum = 0.0
        for idx in (0, 1):
            w = handWeights[idx]
            if w <= 0:
                continue
            traj = [pair[idx] for pair in rawSeq]
            d = self._wristMaxDisplacement(traj)
            if d is not None:
                total += w * d
                weightSum += w
        return total / weightSum if weightSum > 0 else None

    def _weightedWristMaxDisplacement(self, frames, handWeights):
        """Compute hand-weighted average maximum wrist displacement.

        Extracts wrist positions (landmark 0) from per-frame hand arrays and
        returns the weighted average of per-hand max displacement.

        Args:
            frames: Sequence of per-frame tuples (hand0, hand1), each an ndarray
                    of shape (21, 3) or None.
            handWeights: [weight_hand0, weight_hand1].

        Returns:
            float >= 0, or None if no wrist data is available.
        """
        total = 0.0
        weightSum = 0.0
        for idx in (0, 1):
            w = handWeights[idx]
            if w <= 0:
                continue
            traj = [frame[idx][0] if frame[idx] is not None else None for frame in frames]
            d = self._wristMaxDisplacement(traj)
            if d is not None:
                total += w * d
                weightSum += w
        return total / weightSum if weightSum > 0 else None

    # ------------------------------------------------------------------
    # Distance and similarity helpers
    # ------------------------------------------------------------------

    def _distanceToSimilarity(self, distance, maxPossibleDistance=1.0):
        """
        Convert a distance metric to a similarity score (0.0 to 1.0).

        Args:
            distance: Raw distance value (higher = less similar)
            maxPossibleDistance: Expected maximum reasonable distance

        Returns:
            float: Similarity score (0.0 = no similarity, 1.0 = perfect match)
        """
        # Invalid or degenerate distances map to zero similarity.
        if distance == float("inf") or distance < 0:
            return 0.0

        # Exponential decay: similarity = e^(-distance/scale).
        # Unlike linear mapping, this is forgiving for small errors (stays near 1)
        # but drops off sharply once the distance grows past 'scale'.
        scale = maxPossibleDistance / 3  # At distance == maxPossibleDistance, similarity ≈ e^-3 ≈ 0.05
        similarity = math.exp(-distance / scale)

        # Clamp to [0, 1] (exp never exceeds 1 for non-negative input, but guard anyway).
        return min(similarity, 1.0)

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

    def _weightedFrameDistance(self, frame1, frame2, handWeights):
        """Compute weighted per-frame distance with mild penalty for missing hand.

        Args:
            frame1: First frame tuple (hand0, hand1).
            frame2: Second frame tuple (hand0, hand1).
            handWeights: Two-element list of per-hand weights.

        Returns:
            float: Weighted frame distance.
        """
        total = 0.0
        missingHandPenalty = 1.0

        for idx in (0, 1):
            w = handWeights[idx]
            if w <= 0:
                continue

            h1 = frame1[idx]
            h2 = frame2[idx]

            # Both hands missing: treat as perfect match for this hand
            if h1 is None and h2 is None:
                dist = 0.0
            # Only one hand missing: apply penalty
            elif h1 is None or h2 is None:
                dist = missingHandPenalty
            else:
                # Both hands present: compute Euclidean distance
                dist = self._euclideanDistance(h1, h2)

            # Accumulate weighted distance for this hand
            total += w * dist

        return total

    def _averageEuclideanDistance(self, seq1, seq2, handWeights):
        """
        Calculate average Euclidean distance between corresponding frames.

        This assumes sequences are roughly the same length and timing.

        Args:
            seq1, seq2: Lists of per-frame tuples (hand0, hand1)
            handWeights: [weight_hand0, weight_hand1]

        Returns:
            float: Average Euclidean distance
        """
        # Only compare up to the shorter sequence; extra frames are ignored.
        minLen = min(len(seq1), len(seq2))
        if minLen == 0:
            return float("inf")

        totalDistance = 0
        for i in range(minLen):
            # Sum weighted per-frame distances across aligned frame pairs.
            totalDistance += self._weightedFrameDistance(seq1[i], seq2[i], handWeights)

        # Return mean distance per frame.
        return totalDistance / minLen

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
            return -1.0  # Incompatible shapes cannot be compared.

        # Flatten (21, 3) arrays into 1-D vectors of length 63.
        vec1 = landmarks1.flatten()
        vec2 = landmarks2.flatten()

        # cos(θ) = (vec1 · vec2) / (‖vec1‖ * ‖vec2‖)
        dotProduct = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Zero-length vector has no direction; treat as no similarity.
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dotProduct / (norm1 * norm2)

    def _weightedFrameCosineSimilarity(self, frame1, frame2, handWeights):
        """Compute weighted per-frame cosine similarity in [0, 1].

        Args:
            frame1: First frame tuple (hand0, hand1).
            frame2: Second frame tuple (hand0, hand1).
            handWeights: Two-element list of per-hand weights.

        Returns:
            float: Weighted cosine similarity in the range [0, 1].
        """
        total = 0.0

        for idx in (0, 1):
            w = handWeights[idx]
            if w <= 0:
                continue

            h1 = frame1[idx]
            h2 = frame2[idx]

            # Both hands missing: treat as identical for this hand
            if h1 is None and h2 is None:
                sim01 = 1.0
            # Only one hand missing: no meaningful similarity
            elif h1 is None or h2 is None:
                sim01 = 0.0
            else:
                sim = self._cosineSimilarity(h1, h2)
                # Remap from [-1, 1] to [0, 1] so that all values are non-negative.
                sim01 = (sim + 1) / 2

            # Accumulate weighted similarity for hands, giving more influence to the hand with higher motion.
            total += w * sim01

        return total

    def _averageCosineSimilarity(self, seq1, seq2, handWeights):
        """
        Calculate average cosine similarity across sequences.

        Args:
            seq1, seq2: Lists of per-frame tuples (hand0, hand1)
            handWeights: [weight_hand0, weight_hand1]

        Returns:
            float: Average cosine similarity (converted to 0-1 range)
        """
        # Only compare up to the shorter sequence; extra frames are ignored.
        minLen = min(len(seq1), len(seq2))
        if minLen == 0:
            return 0.0

        totalSimilarity = 0
        for i in range(minLen):
            # Sum weighted cosine similarities across aligned frame pairs.
            totalSimilarity += self._weightedFrameCosineSimilarity(seq1[i], seq2[i], handWeights)

        # Return mean similarity per frame.
        return totalSimilarity / minLen

    # ------------------------------------------------------------------
    # DTW
    # ------------------------------------------------------------------

    def _buildDtwMatrix(self, seq1, seq2, handWeights):
        """Fill and return the DTW cost matrix for two sequences.

        Args:
            seq1, seq2: Lists of per-frame tuples (hand0, hand1).
            handWeights: [weight_hand0, weight_hand1] for per-hand mode

        Returns:
            np.ndarray of shape (n+1, m+1) with accumulated DTW costs,
            or None if either sequence is empty.
        """
        n = len(seq1)
        m = len(seq2)

        if n == 0 or m == 0:
            return None

        # Initialize DP table with infinity; (0,0) is the alignment origin.
        dtwMatrix = np.full((n + 1, m + 1), float("inf"))
        dtwMatrix[0, 0] = 0

        # Fill DP table using standard DTW recurrence.
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = seq1[i - 1]
                right = seq2[j - 1]

                cost = self._weightedFrameDistance(left, right, handWeights)

                # Best predecessor: insertion, deletion, or diagonal match.
                dtwMatrix[i, j] = cost + min(
                    dtwMatrix[i - 1, j],  # insertion
                    dtwMatrix[i, j - 1],  # deletion
                    dtwMatrix[i - 1, j - 1],  # match
                )

        return dtwMatrix

    def _dtwWithPath(self, seq1, seq2, handWeights=None):
        """Compute DTW distance and the optimal warping path between two sequences.

        Args:
            seq1, seq2: Lists of per-frame tuples (hand0, hand1).
            handWeights: [weight_hand0, weight_hand1] for per-hand mode

        Returns:
            Tuple (distance, path) where distance is the length-normalized DTW
            distance and path is a list of (i, j) index pairs (0-indexed into
            seq1 and seq2) tracing the optimal alignment from start to end.
        """
        if handWeights is None:
            handWeights = [0.5, 0.5]

        dtwMatrix = self._buildDtwMatrix(seq1, seq2, handWeights)
        if dtwMatrix is None:
            return float("inf"), []

        n, m = len(seq1), len(seq2)
        distance = dtwMatrix[n, m] / (n + m)

        # Backtrack from (n, m) to (1, 1) to recover the optimal warping path.
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))  # store as 0-indexed
            diag = dtwMatrix[i - 1, j - 1]
            ins = dtwMatrix[i - 1, j]
            de = dtwMatrix[i, j - 1]
            best = min(diag, ins, de)
            if best == diag:
                i -= 1
                j -= 1
            elif best == ins:
                i -= 1
            else:
                j -= 1
        path.reverse()

        return distance, path

    # ------------------------------------------------------------------
    # Sequence similarity and top-level scoring
    # ------------------------------------------------------------------

    def _calculateSequenceSimilarity(self, userSequence, referenceSequence):
        """
        Calculate similarity between two landmark sequences using multiple metrics.

        Args:
            userSequence: List of timestamped user landmarks
            referenceSequence: List of timestamped reference landmarks

        Returns:
            float: Combined similarity score (0.0 to 1.0)
        """
        # Score only within the active reference interval marked on the progress bar.
        referenceSequence = self._trimReferenceSequenceByMarkers(referenceSequence)

        # Extract raw (pre-normalization) wrist positions for displacement penalty.
        rawUserWrists = self._extractRawWristSequences(userSequence)
        rawRefWrists = self._extractRawWristSequences(referenceSequence)

        # Extract per-frame hand arrays: each frame is (hand0_array_or_None, hand1_array_or_None)
        userFrames = self._extractPerHandArrays(userSequence)
        refFrames = self._extractPerHandArrays(referenceSequence)

        if len(userFrames) == 0 or len(refFrames) == 0:
            return 0.0, [0.5, 0.5]

        # Compute per-hand motion from reference gesture and use it as comparison weight.
        # If one hand moves more, it gets higher weight.
        handWeights = self._calculateHandMotionWeights(refFrames)

        # Estimate motion energy before trimming. This is used to penalize sequences
        # where the user mostly holds the neutral/rest position.
        referenceMotionEnergy = self._averageMotionEnergy(refFrames, handWeights)
        userMotionEnergy = self._averageMotionEnergy(userFrames, handWeights)

        # Trim idle/rest frames from the edges of the user sequence.
        userFrames = self._trimLowMotionEdges(userFrames, handWeights)
        print(f"  User frames after trim: {len(userFrames)}")

        s = self.strategy

        # Method 1: DTW for temporal alignment — produces the warping path
        # used to align frames for Euclidean and Cosine metrics.
        _, warpingPath = self._dtwWithPath(userFrames, refFrames, handWeights)

        # Reindex both sequences according to the DTW warping path so that
        # Euclidean and Cosine metrics compare temporally matched frames.
        alignedUserFrames = [userFrames[i] for (i, j) in warpingPath]
        alignedRefFrames = [refFrames[j] for (i, j) in warpingPath]

        # Method 2: Average Euclidean distance on DTW-aligned frames
        euclideanDistance = self._averageEuclideanDistance(alignedUserFrames, alignedRefFrames, handWeights)
        euclideanSimilarity = self._distanceToSimilarity(
            euclideanDistance, maxPossibleDistance=s["euclideanMaxDistance"]
        )

        # Method 3: Cosine similarity on DTW-aligned frames
        cosineSim = self._averageCosineSimilarity(alignedUserFrames, alignedRefFrames, handWeights)

        # Combine metrics with weights (DTW similarity excluded; weights normalized).
        euclideanCosineWeight = s["euclideanWeight"] + s["cosineWeight"]
        combinedScore = (
            (s["euclideanWeight"] * euclideanSimilarity + s["cosineWeight"] * cosineSim) / euclideanCosineWeight
            if euclideanCosineWeight > 0
            else 0.0
        )

        # Motion-activity penalty:
        # If the reference contains meaningful motion, but the user sequence stays
        # mostly static (e.g., hands at belly position), reduce final score strongly.
        activityFactor = 1.0
        minReferenceMotion = 0.003
        if referenceMotionEnergy > minReferenceMotion:
            expectedUserMotion = referenceMotionEnergy * s["activityThreshold"]
            if expectedUserMotion > 0:
                activityFactor = max(0.0, min(1.0, userMotionEnergy / expectedUserMotion))

        # Wrist max-displacement penalty:
        # If the reference wrist travels far from its starting position (e.g. hand
        # moves to the eye for 'oko'), but the user wrist barely moves (static user
        # or jitter), penalise.  This works even for gestures that return to the
        # start (net displacement = 0) because we use the *maximum* displacement
        # at any point during the sequence, not start-to-end.
        # Only applied when refMaxDisp is large enough that the reference is a
        # moving gesture — static pose gestures naturally have small refMaxDisp.
        refMaxDisp = self._wristMaxDispFromRawSeq(rawRefWrists, handWeights)
        if refMaxDisp is not None:
            userMaxDisp = self._wristMaxDispFromRawSeq(rawUserWrists, handWeights)
            if userMaxDisp is not None:
                displacementFactor = min(1.0, userMaxDisp / refMaxDisp)
            else:
                displacementFactor = 0.0
            activityFactor *= displacementFactor
            userDispStr = f"{userMaxDisp:.3f}" if userMaxDisp is not None else "None"
            print(f"  Wrist max-disp (user/ref): {userDispStr}/{refMaxDisp:.3f}, factor: {displacementFactor:.3f}")

        combinedScore *= activityFactor

        print(f"  Hand weights (h0, h1): ({handWeights[0]:.3f}, {handWeights[1]:.3f})")
        print(f"  Frames used (user/reference): {len(userFrames)}/{len(refFrames)}")
        print(f"  Motion energy (user/reference): {userMotionEnergy:.5f}/{referenceMotionEnergy:.5f}")
        print(f"  Activity factor: {activityFactor:.3f}")
        print(f"  Euclidean similarity: {euclideanSimilarity:.4f}")
        print(f"  Cosine similarity: {cosineSim:.4f}")

        return combinedScore, handWeights

    def calculateScore(self, userLandmarks=None, includeWristTrajectory=True, wristWeight=0.1):
        """Calculate similarity score between user landmarks and reference landmarks.

        Args:
            userLandmarks: List of timestamped landmark sequences captured from the webcam.
                           Each entry is a tuple (timestamp, landmarks) where landmarks is
                           a list of per-hand landmark dicts with keys 'x', 'y', 'z'.
            includeWristTrajectory: If True, blend in a wrist-trajectory similarity term.
            wristWeight: Weight of the wrist trajectory term in the final blended score
                         (0.0 = ignore wrist, 1.0 = wrist only).

        Returns:
            float: Similarity score between 0.0 (no similarity) and 1.0 (perfect match).
        """
        print(
            f"Calculating score, userLandmarks: {len(userLandmarks) if userLandmarks else 0}, referenceLandmarks: {len(self.referenceAnnotation.handLandmarksTimestamped)}"
        )

        if userLandmarks is None or len(userLandmarks) == 0:
            # Fallback to current landmarks if no sequence provided
            score = 0.0
            return score

        # Compare timestamped sequences
        referenceLandmarks = self.referenceAnnotation.handLandmarksTimestamped

        if len(referenceLandmarks) == 0:
            print("Warning: No reference landmarks available for comparison")
            return 0.0

        # Calculate hand shape similarity score
        score, handWeights = self._calculateSequenceSimilarity(userLandmarks, referenceLandmarks)

        # Optionally add wrist trajectory similarity
        if includeWristTrajectory:
            userWrist0, userWrist1 = self._extractWristTrajectory(self.webcamAnnotation)
            refWrist0, refWrist1 = self._extractWristTrajectory(self.referenceAnnotation)
            minLen = min(len(userWrist0), len(refWrist0))
            if minLen > 2:
                wristHandWeights = handWeights

                # Convert absolute positions to per-frame displacement vectors so the
                # comparison measures movement pattern, not screen location.
                def toDeltas(traj):
                    deltas = []
                    for i in range(1, len(traj)):
                        prev, curr = traj[i - 1], traj[i]
                        if prev is not None and curr is not None:
                            deltas.append(np.array(curr, dtype=float) - np.array(prev, dtype=float))
                        else:
                            deltas.append(None)
                    return deltas

                uDelta0 = toDeltas(userWrist0[:minLen])
                uDelta1 = toDeltas(userWrist1[:minLen])
                rDelta0 = toDeltas(refWrist0[:minLen])
                rDelta1 = toDeltas(refWrist1[:minLen])
                deltaLen = len(uDelta0)  # minLen - 1

                def toArr(pt):
                    return pt.reshape(1, 3) if pt is not None else None

                seq1 = [(toArr(uDelta0[i]), toArr(uDelta1[i])) for i in range(deltaLen)]
                seq2 = [(toArr(rDelta0[i]), toArr(rDelta1[i])) for i in range(deltaLen)]

                dtwDist, _ = self._dtwWithPath(seq1, seq2, handWeights=wristHandWeights)
                dtwSim = self._distanceToSimilarity(dtwDist, maxPossibleDistance=0.05)

                # Per-hand mean delta distance, weighted by presence.
                euclSims = []
                for wristHandIdx, (uDelta, rDelta) in enumerate([(uDelta0, rDelta0), (uDelta1, rDelta1)]):
                    w = wristHandWeights[wristHandIdx]
                    if w <= 0:
                        continue
                    pairs = [(u, r) for u, r in zip(uDelta, rDelta) if u is not None and r is not None]
                    if pairs:
                        dists = [np.linalg.norm(u - r) for u, r in pairs]
                        euclSims.append(w * self._distanceToSimilarity(float(np.mean(dists)), maxPossibleDistance=0.05))
                euclSim = (
                    sum(euclSims) / sum(wristHandWeights[i] for i in range(2) if wristHandWeights[i] > 0 and euclSims)
                    if euclSims
                    else 0.0
                )

                wristScore = 0.6 * dtwSim + 0.4 * euclSim
                print(
                    f"Wrist DTW similarity: {dtwSim:.4f}, Euclidean similarity: {euclSim:.4f}, Combined: {wristScore:.4f}"
                )
                score = (1 - wristWeight) * score + wristWeight * wristScore
        print(f"Calculated similarity score: {score:.4f}")
        return score
