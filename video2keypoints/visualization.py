import cv2
import json
import os
from video2keypoints import get_video_from_path

def visualize_keypoints(video: cv2.VideoCapture, keypoint_dir):
    """비디오와 키포인트를 시각화하여 보여줍니다.
    
    Args:
        video (cv2.VideoCapture): 원본 비디오 객체
        keypoint_dir (str): 프레임별 키포인트 json이 저장된 디렉토리 경로
    """
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        json_path = os.path.join(keypoint_dir, f'{frame_idx:06d}.json')
        if not os.path.exists(json_path):
            break
        with open(json_path, 'r') as f:
            kp = json.load(f)
        # --- Pose keypoints ---
        pose = kp['people']['pose_keypoints_2d']
        for i in range(0, len(pose), 3):
            x, y, v = pose[i], pose[i+1], pose[i+2]
            if v > 0:
                px, py = int(x * frame_width), int(y * frame_height)
                cv2.circle(frame, (px, py), 3, (0,255,0), -1)
        # --- Left hand keypoints ---
        left = kp['people']['hand_left_keypoints_2d']
        for i in range(0, len(left), 3):
            x, y, v = left[i], left[i+1], left[i+2]
            if v > 0:
                px, py = int(x * frame_width), int(y * frame_height)
                cv2.circle(frame, (px, py), 3, (255,0,0), -1)
        # --- Right hand keypoints ---
        right = kp['people']['hand_right_keypoints_2d']
        for i in range(0, len(right), 3):
            x, y, v = right[i], right[i+1], right[i+2]
            if v > 0:
                px, py = int(x * frame_width), int(y * frame_height)
                cv2.circle(frame, (px, py), 3, (0,0,255), -1)
        # --- 실시간 시각화 ---
        cv2.imshow('Keypoint Visualization', frame)
        if cv2.waitKey(1) == 27:  # ESC 누르면 종료
            break
        frame_idx += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'example.mp4'
    keypoint_dir = 'example'
    video = get_video_from_path(video_path)
    visualize_keypoints(video, keypoint_dir)
    video.release()