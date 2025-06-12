import os
import json
import numpy as np

# 데이터 디렉토리 경로 설정
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
RAW_ROOT = os.path.join(DATA_ROOT, 'raw')
LABEL_JSON_ROOT = os.path.join(DATA_ROOT, 'label_json')
PROCESSED_ROOT = os.path.join(DATA_ROOT, 'processed_npy')

FEATURE_ROOT = os.path.join(PROCESSED_ROOT, 'train')
LABEL_ROOT = os.path.join(LABEL_JSON_ROOT, 'morpheme_sen/02')
OUT_ROOT = os.path.join(PROCESSED_ROOT, 'train')
DICT_PATH = os.path.join(OUT_ROOT, 'gloss_dict.json')

os.makedirs(OUT_ROOT, exist_ok=True)

def get_label_path(feature_path):
    """
    feature 파일 경로로부터 label json 파일 경로를 찾는 함수
    Args:
        feature_path: feature npy 파일 경로 (예: .../NIA_SL_SEN0001_REAL02_F_feature.npy)
    Returns:
        label json 파일 경로
    """
    fname = os.path.basename(feature_path)
    # SEN/WRD 구분
    if 'NIA_SL_SEN' in fname:
        type_dir = 'morpheme_sen'
    else:
        type_dir = 'morpheme_word'
    # REALXX에서 XX 추출
    import re
    real_num = re.search(r'REAL(\d+)', fname).group(1)
    # feature.npy -> morpheme.json
    label_fname = fname.replace('_feature.npy', '_morpheme.json')
    return os.path.join(LABEL_JSON_ROOT, type_dir, real_num, label_fname)

def find_all_json_files(train_root):
    json_files = []
    for fname in os.listdir(train_root):
        if fname.endswith('_feature.npy'):
            feature_path = os.path.join(train_root, fname)
            label_path = get_label_path(feature_path)
            json_files.append(label_path)
    return json_files

# 1. 모든 글로스(단어) 수집
gloss_set = set()
json_files = find_all_json_files(FEATURE_ROOT)
for path in json_files:
    with open(path, 'r', encoding='utf-8') as f:
        print(path)
        label_json = json.load(f)
    for seg in label_json['data']:
        for attr in seg['attributes']:
            gloss_set.add(attr['name'])

# 2. 단어-인덱스 사전 생성
gloss2idx = {gloss: idx for idx, gloss in enumerate(sorted(gloss_set))}
idx2gloss = {idx: gloss for gloss, idx in gloss2idx.items()}

# 3. 각 샘플별 label 시퀀스 생성 및 저장
for fname in os.listdir(FEATURE_ROOT):
    if fname.endswith('_feature.npy'):
        feature_path = os.path.join(FEATURE_ROOT, fname)
        feature = np.load(feature_path)
        feature_len = len(feature)
        label_path = get_label_path(feature_path)
        with open(label_path, 'r', encoding='utf-8') as f:
            label_json = json.load(f)
        num_frames = feature_len
        label_seq = np.zeros(num_frames, dtype=np.int64)  # 프레임 수만큼 0으로 초기화
        for seg in label_json['data']:
            start_frame = int(seg['start'] * 30)
            end_frame = int(seg['end'] * 30)
            # attributes가 여러개 있지만, AIHub 데이터셋에서는 하나만 존재
            for attr in seg['attributes']:
                gloss_idx = gloss2idx[attr['name']]
                label_seq[start_frame:end_frame+1] = gloss_idx
        label_seq = np.array(label_seq, dtype=np.int64)
        base = fname.replace('_feature.npy', '')
        np.save(os.path.join(OUT_ROOT, f'{base}_label.npy'), label_seq)
        print(f'{base}_label.npy 저장 완료')

# 4. 사전 저장
with open(DICT_PATH, 'w', encoding='utf-8') as f:
    json.dump({'gloss2idx': gloss2idx, 'idx2gloss': idx2gloss}, f, ensure_ascii=False, indent=2)
print(f'단어-인덱스 사전 {DICT_PATH} 저장 완료')