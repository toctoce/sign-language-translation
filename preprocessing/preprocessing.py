import os
import json
import numpy as np

# 경로 설정
RAW_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/raw')
LABEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/label_json/morpheme_sen/02')
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed_npy/train')
DICT_PATH = os.path.join(OUT_ROOT, 'gloss_dict.json')

os.makedirs(OUT_ROOT, exist_ok=True)

FPS = 30

def find_all_json_files(root):
    json_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.json'):
                json_files.append(os.path.join(dirpath, fname))
    return json_files

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


# 1. gloss 사전 생성
print('글로스 사전 생성 중...')
gloss_set = set()
label_json_files = find_all_json_files(LABEL_ROOT)
for path in label_json_files:
    with open(path, 'r', encoding='utf-8') as f:
        label_json = json.load(f)
    for seg in label_json['data']:
        for attr in seg['attributes']:
            gloss_set.add(attr['name'])

gloss2idx = {gloss: idx for idx, gloss in enumerate(sorted(gloss_set))}
idx2gloss = {idx: gloss for gloss, idx in gloss2idx.items()}

with open(DICT_PATH, 'w', encoding='utf-8') as f:
    json.dump({'gloss2idx': gloss2idx, 'idx2gloss': idx2gloss}, f, ensure_ascii=False, indent=2)
print(f'단어-인덱스 사전 {DICT_PATH} 저장 완료')

# 2. 샘플별 feature/label 생성
print('샘플별 feature/label 생성 중...')
for word_name in os.listdir(RAW_ROOT):
    word_folder = os.path.join(RAW_ROOT, word_name)
    if not os.path.isdir(word_folder):
        continue
    # keypoint feature 생성
    json_files = sorted([f for f in os.listdir(word_folder) if f.endswith('_keypoints.json')])
    keypoint_seq = []
    for jf in json_files:
        kp = extract_keypoints_from_json(os.path.join(word_folder, jf))
        keypoint_seq.append(kp)
    keypoint_seq = np.stack(keypoint_seq, axis=0)  # (프레임수, keypoint수, 2)
    np.save(os.path.join(OUT_ROOT, f'{word_name}_feature.npy'), keypoint_seq)
    print(f'{word_name}_feature.npy 저장 완료')

    # label json 찾기 (파일명 규칙에 맞게 수정 필요할 수 있음)
    label_json_name = word_name + '_morpheme.json'
    label_json_path = os.path.join(LABEL_ROOT, label_json_name)
    if not os.path.exists(label_json_path):
        print(f'label json 없음: {label_json_path}')
        continue
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_json = json.load(f)
    # label 시퀀스
    label_seq = []
    for seg in label_json['data']:
        for attr in seg['attributes']:
            label_seq.append(gloss2idx[attr['name']])
    label_seq = np.array(label_seq, dtype=np.int64)
    np.save(os.path.join(OUT_ROOT, f'{word_name}_label.npy'), label_seq)
    print(f'{word_name}_label.npy 저장 완료')
    


print('전처리 완료!')
