import os
import json
import numpy as np

RAW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/raw')
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed_npy/train')

os.makedirs(OUT_ROOT, exist_ok=True)

# 프레임별 keypoint 추출 함수
def extract_keypoints_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    people = data['people']
    def reshape_kps(kps):
        arr = np.array(kps).reshape(-1, 3)
        return arr[:, :2]  # (N, 2)
    pose = reshape_kps(people.get('pose_keypoints_2d', []))
    hand_l = reshape_kps(people.get('hand_left_keypoints_2d', []))
    hand_r = reshape_kps(people.get('hand_right_keypoints_2d', []))
    keypoints = np.concatenate([pose, hand_l, hand_r], axis=0)  # (keypoint수, 2)
    return keypoints

def process_word_folder(word_folder):
    word_name = os.path.basename(word_folder)
    out_word_dir = OUT_ROOT
    json_files = sorted([f for f in os.listdir(word_folder) if f.endswith('_keypoints.json')])
    keypoint_seq = []
    for jf in json_files:
        kp = extract_keypoints_from_json(os.path.join(word_folder, jf))
        keypoint_seq.append(kp)
    keypoint_seq = np.stack(keypoint_seq, axis=0)  # (프레임수, keypoint수, 2)
    # feature만 저장
    np.save(os.path.join(out_word_dir, f'{word_name}_feature.npy'), keypoint_seq)
    print(f'{word_name}_feature.npy 저장 완료')

def main():
    for word_name in os.listdir(RAW_ROOT):
        word_folder = os.path.join(RAW_ROOT, word_name)
        if not os.path.isdir(word_folder):
            continue
        process_word_folder(word_folder)

if __name__ == '__main__':
    main() 