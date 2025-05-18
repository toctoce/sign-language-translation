import os
import json
import numpy as np

LABEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/label_json/morpheme_sen/02')
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed_npy/train')
DICT_PATH = os.path.join(OUT_ROOT, 'gloss_dict.json')

os.makedirs(OUT_ROOT, exist_ok=True)

def find_all_json_files(root):
    json_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.json'):
                json_files.append(os.path.join(dirpath, fname))
    return json_files

# 1. 모든 글로스(단어) 수집
gloss_set = set()
json_files = find_all_json_files(LABEL_ROOT)
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
for path in json_files:
    with open(path, 'r', encoding='utf-8') as f:
        label_json = json.load(f)
    label_seq = []
    for seg in label_json['data']:
        for attr in seg['attributes']:
            label_seq.append(gloss2idx[attr['name']])
    label_seq = np.array(label_seq, dtype=np.int64)
    # 파일명에서 _morpheme.json 제거, _label.npy로 저장
    base = os.path.basename(path).replace('_morpheme.json', '')
    np.save(os.path.join(OUT_ROOT, f'{base}_label.npy'), label_seq)
    print(f'{base}_label.npy 저장 완료')

# 4. 사전 저장
with open(DICT_PATH, 'w', encoding='utf-8') as f:
    json.dump({'gloss2idx': gloss2idx, 'idx2gloss': idx2gloss}, f, ensure_ascii=False, indent=2)
print(f'단어-인덱스 사전 {DICT_PATH} 저장 완료')