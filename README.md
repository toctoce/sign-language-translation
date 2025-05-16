# 🤟 한국어 수어 영상 기반 문장 번역기
본 프로젝트는 수어 영상을 입력 받아 손동작을 인식하고, 단어(gloss) 시퀀스를 예측한 뒤, 이를 자연어 문장으로 변환하는 수어-텍스트 번역 시스템입니다.

---

## 📌 프로젝트 개요

- 입력: 수어 영상 or keypoint 시퀀스
- 처리: LSTM + CTC 기반 gloss 예측
- 출력: GPT 기반 자연어 문장 생성
- 사용 데이터: AIHub 수어 영상 데이터셋

---

## ⚙️ 실행 방법

```bash
# 1. 가상환경 생성 및 패키지 설치
pip install -r requirements.txt

# 2. JSON → npy 전처리
python preprocessing/extract_keypoints.py

# 3. 모델 학습
python training/train_ctc.py --config config.yaml

# 4. 문장 생성
python inference/generate_sentence.py
```

---

## 🗂️ 디렉토리 구조
```
sign-language-translation/
├── data/                  # 원본 JSON / 전처리된 npy / 정답 gloss
│   ├── raw/               # AIHub 원본 keypoint JSON
│   └── processed/         # 모델 입력용 npy 시퀀스
│
├── models/                # CSLR 모델, 문장 생성기
│   ├── cslr_model.py
│   └── llm_wrapper.py
│
├── training/              # 학습 루프 및 설정
│   ├── train_ctc.py
│   └── config.yaml
│
├── inference/             # 추론 및 실험용 스크립트
│   ├── extract_keypoints.py     # JSON → npy 변환 (inference용)
│   ├── predict_gloss.py         # gloss 예측
│   └── generate_sentence.py     # gloss → 문장 변환
│
├── results/               # 예측 결과, 시각화, 비교표
│   ├── gloss_output.txt
│   └── final_sentence.txt
│
├── assets/                # 프로젝트용 그림, 도식, PPT용 자료
│   └── architecture_diagram.png
│
├── docs/                  # 프로젝트 설명 문서
│   ├── project_overview.md
│   ├── model_architecture.md
│   └── training_log.md
│
├── notebooks/             # 개발, 실험용 Jupyter 노트북
│   └── visualize_sequence.ipynb
│
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt

```
### 📁 data/
프로젝트의 원본 및 전처리 데이터를 저장하는 폴더

raw/ : AIHub JSON, 영상(mp4) 등 원본 데이터

processed/ : 모델 입력용 .npy 시퀀스 (T, 42, 2 형태)

### 📁 models/
모델 정의 및 관련 함수

cslr_model.py : LSTM/TCN 기반 수어 인식 모델

loss.py : CTC Loss 및 평가 함수

llm_wrapper.py : GPT-3.5, KoGPT 등 문장 생성기 연결 모듈

### 📁 training/
모델 학습 관련 코드 및 설정

train_ctc.py : gloss 예측 모델 학습 스크립트

config.yaml : 하이퍼파라미터, 경로 등 설정 파일

### 📁 inference/
학습된 모델을 사용해 예측을 수행하는 코드

extract_keypoints.py : 추론용 JSON → .npy 변환 스크립트

predict_gloss.py : .npy 시퀀스 → gloss 시퀀스 예측

generate_sentence.py : gloss → 자연어 문장 생성

### 📁 notebooks/
실험, 시각화, 디버깅을 위한 Jupyter 노트북

01_visualize_sequence.ipynb : keypoint 시각화

02_gloss_prediction_test.ipynb : 예측 테스트

### 📁 docs/
문서화 및 설명 정리

project_overview.md : 전체 개요, 목적

data_preprocessing.md : 전처리 흐름 설명

model_architecture.md : 모델 구조 설명

training_log.md : 실험 기록, 성능 로그

### 📁 results/
실험 결과 및 예측 결과 저장

gloss_output.txt : 예측된 gloss 시퀀스

final_sentence.txt : 생성된 자연어 문장

training_curve.png : 학습 곡선 시각화

### 📁 assets/
시각 자료 및 도식 저장용

architecture_diagram.png : 전체 시스템 구조

model_flow.png, dataflow.png : 파이프라인 흐름도

### 📄 루트 파일들
README.md : 프로젝트 설명 및 실행법 안내

LICENSE : 오픈소스 라이선스 (MIT 권장)

requirements.txt : 필요한 패키지 목록

.gitignore : 불필요한 파일 무시 설정 (*.npy, __pycache__/ 등)

---

## 🔧 주요 기술 스택
Python, NumPy, PyTorch, MediaPipe

LSTM / TCN / CTC Loss

OpenAI GPT-3.5 (or KoGPT)

Google Cloud Storage, RunPod

---

## 실험 결과

---

## 라이선스

---

## TODO
