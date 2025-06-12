import cv2
import mediapipe as mp
import numpy as np
import os
import json

def extract_keypoints_from_video(video_path):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")

    json_data_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Pose
        pose_results = pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            pose_kps = []
            for lm in pose_results.pose_landmarks.landmark:
                pose_kps.extend([lm.x, lm.y, 1])
            while len(pose_kps) < 25*3:
                pose_kps.extend([0,0,0])
        else:
            pose_kps = [0,0,0]*25
        # Hands
        hands_results = hands.process(frame_rgb)
        left_hand = [0,0,0]*21
        right_hand = [0,0,0]*21
        if hands_results.multi_handedness and hands_results.multi_hand_landmarks:
            for idx, handedness in enumerate(hands_results.multi_handedness):
                label = handedness.classification[0].label
                hand_landmarks = hands_results.multi_hand_landmarks[idx]
                kps = []
                for lm in hand_landmarks.landmark:
                    kps.extend([lm.x, lm.y, 1])
                while len(kps) < 21*3:
                    kps.extend([0,0,0])
                if label == 'Left':
                    left_hand = kps
                elif label == 'Right':
                    right_hand = kps
        # JSON 포맷 맞추기 (프레임별)
        json_data = {
            "version": 1.3,
            "people": {
                "person_id": -1,
                "pose_keypoints_2d": pose_kps,
                "hand_left_keypoints_2d": left_hand,
                "hand_right_keypoints_2d": right_hand
            }
        }
        json_data_list.append(json_data)

    cap.release()
    hands.close()
    pose.close()
    return json_data_list

def save_keypoints_to_json(json_data_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for frame_idx, json_data in enumerate(json_data_list):
        save_path = os.path.join(save_dir, f"{frame_idx:06d}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False)
    print(f"총 {len(json_data_list)}개 프레임의 keypoints json을 {save_dir}에 저장 완료.")
    return len(json_data_list)

if __name__ == "__main__":
    # 비디오 파일 경로와 저장 디렉토리 설정
    video_path = "./example.mp4"  
    save_dir = "./example"  # 프레임별 json 저장 디렉토리
    
    # 키포인트 추출 실행
    json_data_list = extract_keypoints_from_video(video_path)
    
    # JSON 파일로 저장
    num_frames = save_keypoints_to_json(json_data_list, save_dir)
    print(f"프레임별 keypoints json 저장 완료: {save_dir} (총 {num_frames}개)")
