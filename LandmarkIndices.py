class LandmarkIndices:
    """Named integer constants for MediaPipe hand landmark indices (0-based)."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 11
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


"""
handsLandmarksTimestamped structure:

[
  (0.00, [  # timestamp
    [  # hand 0
      {"x": 0.50, "y": 0.40, "z": -0.05},
      {"x": 0.51, "y": 0.42, "z": -0.04},
      ... 21 landmarks ...
    ],
    [  # hand 1 (if detected)
      {"x": 0.20, "y": 0.45, "z": -0.02},
      {"x": 0.21, "y": 0.47, "z": -0.03},
      ... 21 landmarks ...
    ]
  ]),
  (0.05, [ ... next frame sample ... ]),
  ...
]

"""
