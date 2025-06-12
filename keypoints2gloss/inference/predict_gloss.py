import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from models.cslr_model import CSLRModel
import json

# 경로 설정
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else '../results/cslr_lstm_ctc.pth'
FEATURE_PATH = sys.argv[2] if len(sys.argv) > 2 else None
GLOSS_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed_npy/train/gloss_dict.json')

# 하이퍼파라미터 (학습 때와 동일하게 맞춰야 함)
HIDDEN_SIZE = 128
NUM_LAYERS = 2

NUM_CLASSES = 319

# feature 불러오기
def load_feature(path):
    feature = np.load(path)
    if feature.ndim == 3:
        feature = feature.reshape(feature.shape[0], -1)
    return torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # [1, T, F]

# gloss 사전 불러오기
def load_gloss_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    idx2gloss = {int(k): v for k, v in d['idx2gloss'].items()}
    return idx2gloss, d['gloss2idx']

# CTC 디코딩 (greedy)
def ctc_greedy_decode(log_probs, blank_idx=0):
    pred = log_probs.argmax(dim=-1).squeeze(0).cpu().numpy()
    prev = -1
    output = []
    for p in pred:
        if p != prev and p != blank_idx:
            output.append(p)
        prev = p
    return output

def main():
    if FEATURE_PATH is None:
        print('사용법: python predict_gloss.py [모델경로] [feature_npy경로]')
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # gloss 사전
    idx2gloss, gloss2idx = load_gloss_dict(GLOSS_DICT_PATH)
    NUM_CLASSES = len(gloss2idx)
    # feature 불러오기
    feature = load_feature(FEATURE_PATH).to(device)
    feature_len = [feature.shape[1]]
    # 모델 불러오기
    model = CSLRModel(feature.shape[2], HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(feature, feature_len)  # [1, T, C]
        print(logits)
        print(logits.shape)
        log_probs = logits.log_softmax(2)
        print(log_probs)
        pred_all = log_probs.argmax(dim=-1).squeeze(0).cpu().numpy()
        print('모든 프레임 예측 인덱스:', pred_all)
        pred_indices = ctc_greedy_decode(log_probs)
        pred_gloss = [idx2gloss[str(idx)] for idx in pred_indices if str(idx) in idx2gloss]
    print('예측 인덱스 시퀀스:', pred_indices)
    print('예측 gloss 시퀀스:', pred_gloss)

if __name__ == '__main__':
    main()