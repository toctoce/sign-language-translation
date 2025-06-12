import cv2
import mediapipe as mp
import numpy as np
import os
import json

def get_video_from_path(video_path):
    """비디오 파일 경로로부터 VideoCapture 객체를 반환합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"비디오 파일을 열 수 없습니다: {video_path}")
    return cap

def extract_keypoints_from_video(video: cv2.VideoCapture):
    """비디오 데이터로부터 키포인트를 추출합니다.
    """
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
    
    json_data_list = []
    while video.isOpened():
        ret, frame = video.read()
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

    hands.close()
    pose.close()
    return json_data_list

def convert_json_keypoint_to_npy(json_data):
    """JSON 데이터로부터 키포인트를 추출합니다.
    """
    people = json_data['people']
    def reshape_kps(kps):
        arr = np.array(kps).reshape(-1, 3)
        return arr[:, :2]  # (N, 2)
    pose = reshape_kps(people.get('pose_keypoints_2d', []))      # (25, 2)
    hand_l = reshape_kps(people.get('hand_left_keypoints_2d', []))  # (21, 2)
    hand_r = reshape_kps(people.get('hand_right_keypoints_2d', [])) # (21, 2)
    keypoints = np.concatenate([pose, hand_l, hand_r], axis=0)   # (67, 2)
    return keypoints

def convert_json_list_to_npy(json_data_list):
    keypoint_seq = []
    for json_data in json_data_list:
        kp = convert_json_keypoint_to_npy(json_data)
        keypoint_seq.append(kp)
    keypoint_seq = np.stack(keypoint_seq, axis=0)  # (frame수, 67, 2)
    return keypoint_seq

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
    out_path = "./example_keypoints.npy"  # 최종 npy 파일 저장 경로
    
    # 키포인트 추출 실행
    video = get_video_from_path(video_path)
    json_data_list = extract_keypoints_from_video(video)
    video.release()

    save_keypoints_to_json(json_data_list, save_dir)
    print(f"Saved keypoints to {save_dir}")
    
    keypoint_seq = convert_json_list_to_npy(json_data_list)
    np.save(out_path, keypoint_seq)
    print(f"Saved keypoints to {out_path}, shape: {keypoint_seq.shape}")
