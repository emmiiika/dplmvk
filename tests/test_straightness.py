import os, sys, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from HandAnnotation import HandAnnotation
from Scoring import Scoring

ANNOTATED = os.path.abspath('../videa/.annotated')
RECORDED  = os.path.abspath('../videa/.recorded')
REFERENCE = os.path.abspath('../videa')

def get_ann(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    lm_path = os.path.join(ANNOTATED, f'{stem}_annotated_handLandmarks.json')
    cap = cv2.VideoCapture(video_path)
    ann = HandAnnotation(cap)
    if os.path.isfile(lm_path):
        ann.loadLandmarksFromFile(lm_path)
    ann.cam.release()
    ann.out.release()
    return ann

pairs = [
    ('d_nic',   'oko'),
    ('d_oko',   'oko'),
    ('d_slovo', 'slovo'),
    ('e_oko',   'oko'),
    ('e_slovo', 'slovo'),
]
print(f"{'User':12} {'Ref':10} {'Score':>7}  {'refMaxDisp':>10}  {'userMaxDisp':>11}")
print('-' * 60)
for user_name, ref_name in pairs:
    user_ann = get_ann(os.path.join(RECORDED,  f'{user_name}.mp4'))
    ref_ann  = get_ann(os.path.join(REFERENCE, f'{ref_name}.mp4'))
    scorer = Scoring(user_ann, ref_ann)
    ref_frames  = scorer._extractPerHandArrays(ref_ann.handLandmarksTimestamped)
    hw = scorer._calculateHandMotionWeights(ref_frames)
    raw_ref = scorer._extractRawWristSequences(ref_ann.handLandmarksTimestamped)
    raw_user = scorer._extractRawWristSequences(user_ann.handLandmarksTimestamped)
    ref_disp  = scorer._wristMaxDispFromRawSeq(raw_ref, hw)
    user_disp = scorer._wristMaxDispFromRawSeq(raw_user, hw)
    score = scorer._calculateSequenceSimilarity(
        user_ann.handLandmarksTimestamped,
        ref_ann.handLandmarksTimestamped
    )
    print(f"{user_name:12} {ref_name:10} {score*100:6.1f}%  {ref_disp or 0:7.3f}  {user_disp or 0:8.3f}")
