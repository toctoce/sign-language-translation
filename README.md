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

sign-language-translation/
├── data/ # 원본 및 전처리 데이터
├── preprocessing/ # keypoint 전처리 코드
├── models/ # CSLR, LLM 연동 모델
├── training/ # 학습 루프, 설정
├── inference/ # 추론 스크립트
├── docs/ # 프로젝트 문서화
├── assets/ # 도식, 그림 등 시각화 자료
└── results/ # 예측 결과

### 📁 `data/`

데이터를 저장하는 최상위 폴더입니다.

* `raw/`
  → 원본 수어 영상(mp4), AIHub 전체 JSON 데이터 등
  → *아직 전처리를 하지 않은 상태의 데이터*

* `keypoints_json/`
  → 프레임 단위로 추출된 keypoint JSON 파일들
  → *수어 영상의 좌표 정보가 들어 있음*

* `processed_npy/`
  → 전처리를 마친 `.npy` 시퀀스 파일들
  → (T, 42, 2) 형태로 저장되어 모델 학습에 바로 사용 가능


### 📁 `preprocessing/`

데이터 전처리 및 변환 작업을 담당하는 코드들

* `extract_keypoints.py`
  → JSON → numpy 시퀀스(.npy)로 변환하는 함수

* `generate_dataset.py`
  → 여러 폴더에 있는 JSON을 일괄 변환하는 배치 스크립트


### 📁 `models/`

수어 인식 모델 및 문장 생성 모델 정의

* `cslr_model.py`
  → LSTM/TCN + Linear + CTC 구조의 수어 인식 모델

* `loss.py`
  → CTC Loss, Accuracy 계산 함수

* `llm_wrapper.py`
  → GPT-3.5, KoGPT 등 문장 생성용 모델과의 연동 코드


### 📁 `training/`

학습 스크립트와 설정 파일

* `train_ctc.py`
  → CSLR 모델 학습 루프 (데이터 로딩, loss 계산, 저장 등)

* `config.yaml`
  → 배치 크기, 러닝레이트, 모델 경로 등 하이퍼파라미터 정리


### 📁 `inference/`

학습된 모델을 사용하여 추론하는 코드

* `predict_gloss.py`
  → `.npy` 파일을 입력받아 gloss 시퀀스 추론

* `generate_sentence.py`
  → gloss 시퀀스를 입력받아 LLM을 통해 문장 생성


### 📁 `notebooks/`

개발 중 실험, 시각화, 디버깅을 위한 Jupyter Notebook

* `01_visualize_sequence.ipynb`
  → keypoint 시퀀스 시각화 예시 (matplotlib 등 활용)

* `02_gloss_prediction_test.ipynb`
  → 학습된 모델로 테스트 해보는 Notebook


### 📁 `docs/`

프로젝트 문서 정리 (포트폴리오 핵심)

* `project_overview.md`
  → 전체 배경, 목표, 구성 요약

* `data_preprocessing.md`
  → 전처리 과정 설명

* `model_architecture.md`
  → CSLR + CTC 모델 구조 설명

* `training_log.md`
  → 실험 결과 기록, 성능 변화 비교 등


### 📁 `results/`

모델 추론 결과나 성능 측정 결과 저장

* `gloss_output.txt`
  → 예측된 gloss 시퀀스

* `final_sentence.txt`
  → LLM을 통해 생성된 자연어 문장

* `training_curve.png`
  → loss, accuracy 변화 시각화


### 📁 `assets/`

이미지, 도식, 시각화 자료 저장 (발표/블로그용)

* `architecture_diagram.png`
  → 전체 파이프라인 구조

* `model_flow.png`, `dataflow.png` 등


### 📄 루트 파일들

* `README.md`
  → 프로젝트 개요, 실행법, 구조 요약

* `LICENSE`
  → 오픈소스 라이선스 (MIT 추천)

* `requirements.txt`
  → 필요한 패키지 리스트 (pip install -r requirements.txt)

* `.gitignore`
  → `__pycache__/`, `.npy`, `logs/` 등 버전관리 제외

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
