import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.cslr_model import CSLRModel

# 하이퍼파라미터
BATCH_SIZE = 1
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 319  # 실제 gloss 개수에 맞게 수정

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/processed_npy/train')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/cslr_lstm_ctc.pth')

def load_npy(path):
    return np.load(path, allow_pickle=True)

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_feature.npy')])
        self.label_files = [f.replace('_feature.npy', '_label.npy') for f in self.feature_files]

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature = load_npy(os.path.join(self.data_dir, self.feature_files[idx]))
        # (프레임수, keypoint수, 2) → (프레임수, keypoint수*2)
        if feature.ndim == 3:
            feature = feature.reshape(feature.shape[0], -1)
        feature = torch.tensor(feature, dtype=torch.float32)
        label_path = os.path.join(self.data_dir, self.label_files[idx])
        if os.path.exists(label_path):
            label = torch.tensor(load_npy(label_path), dtype=torch.long)
        else:
            label = torch.tensor([], dtype=torch.long)
        return feature, label

def collate_fn(batch):
    features, labels = zip(*batch)
    feature_lens = [f.shape[0] for f in features]
    label_lens = [l.shape[0] for l in labels]
    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels_concat = torch.cat(labels) if len(labels) > 0 and labels[0].numel() > 0 else torch.tensor([], dtype=torch.long)
    return features_padded, labels_concat, feature_lens, label_lens

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for features, labels, feature_lens, label_lens in loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(features, feature_lens)
        log_probs = logits.log_softmax(2)
        log_probs = log_probs.transpose(0, 1)  # [T, B, C]
        loss = criterion(log_probs, labels, feature_lens, label_lens)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    if len(loader) == 0:
        print('Validation 데이터가 없습니다.')
        return 0.0
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels, feature_lens, label_lens in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features, feature_lens)
            log_probs = logits.log_softmax(2)
            log_probs = log_probs.transpose(0, 1)
            loss = criterion(log_probs, labels, feature_lens, label_lens)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SignLanguageDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    # FEATURE_DIM 자동 결정
    sample_feature, _ = dataset[0]
    FEATURE_DIM = sample_feature.shape[1]
    model = CSLRModel(FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, loader, criterion, optimizer, device)
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}')

    # 날짜/시간 문자열 생성
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'../results/cslr_lstm_ctc_{now}.pth'
    )
    torch.save(model.state_dict(), save_path)
    print('모델 저장 완료!')

if __name__ == '__main__':
    main()