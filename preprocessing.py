import numpy as np
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import sys
from scipy.signal import butter, filtfilt

points = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28 ,29, 30, 31, 32]

class PREPROCESS():
    def __init__(self, rootPath, marker_filename, markerless_filename):
        self.rootPath = rootPath
        self.markerless_filename = markerless_filename
        self.marker_filename = marker_filename
        self.points = points
        
    def return_marker_csv(self):
        # CSV 파일 경로 지정
        csv_file_path = f"{self.rootPath}\\marker_raw_csv\\{self.marker_filename}"

        # CSV 파일 읽기
        marker = pd.read_csv(csv_file_path, skiprows=11)

        # 열 이름 정의
        new_columns = [
            'R_CLA_FIN_x', 'R_CLA_FIN_y', 'R_CLA_FIN_z',
            'CLA_MID_x', 'CLA_MID_y', 'CLA_MID_z',
            'L_CLA_FIN_x', 'L_CLA_FIN_y', 'L_CLA_FIN_z',
            'R_ELBOW_L_x', 'R_ELBOW_L_y', 'R_ELBOW_L_z',
            'R_ELBOW_M_x', 'R_ELBOW_M_y', 'R_ELBOW_M_z',
            'R_WRIST_L_x', 'R_WRIST_L_y', 'R_WRIST_L_z',
            'R_WRIST_M_x', 'R_WRIST_M_y', 'R_WRIST_M_z',
            'L_ELBOW_L_x', 'L_ELBOW_L_y', 'L_ELBOW_L_z',
            'L_ELBOW_M_x', 'L_ELBOW_M_y', 'L_ELBOW_M_z',
            'L_WRIST_L_x', 'L_WRIST_L_y', 'L_WRIST_L_z',
            'L_WRIST_M_x', 'L_WRIST_M_y', 'L_WRIST_M_z',
            'R_ASIS_x', 'R_ASIS_y', 'R_ASIS_z',
            'L_ASIS_x', 'L_ASIS_y', 'L_ASIS_z',
            'R_PSIS_x', 'R_PSIS_y', 'R_PSIS_z',
            'L_PSIS_x', 'L_PSIS_y', 'L_PSIS_z',
            'R_KNEE_L_x', 'R_KNEE_L_y', 'R_KNEE_L_z',
            'R_KNEE_M_x', 'R_KNEE_M_y', 'R_KNEE_M_z',
            'R_ANKLE_L_x', 'R_ANKLE_L_y', 'R_ANKLE_L_z',
            'R_ANKLE_M_x', 'R_ANKLE_M_y', 'R_ANKLE_M_z',
            'R_TOE1_x', 'R_TOE1_y', 'R_TOE1_z',
            'R_TOE5_x', 'R_TOE5_y', 'R_TOE5_z',
            'R_HEEL_x', 'R_HEEL_y', 'R_HEEL_z',
            'L_KNEE_L_x', 'L_KNEE_L_y', 'L_KNEE_L_z',
            'L_KNEE_M_x', 'L_KNEE_M_y', 'L_KNEE_M_z',
            'L_ANKLE_L_x', 'L_ANKLE_L_y', 'L_ANKLE_L_z',
            'L_ANKLE_M_x', 'L_ANKLE_M_y', 'L_ANKLE_M_z',
            'L_TOE1_x', 'L_TOE1_y', 'L_TOE1_z',
            'L_TOE5_x', 'L_TOE5_y', 'L_TOE5_z',
            'L_HEEL_x', 'L_HEEL_y', 'L_HEEL_z'
        ]

        # 열 이름을 새로운 열 이름으로 변경
        marker.columns = new_columns

        # 중간값 계산 함수
        def calculate_midpoint(col1, col2):
            return (marker[col1] + marker[col2]) / 2

        # 중간값 계산 및 새로운 열 생성
        marker['R_SHOULDER_x'] = calculate_midpoint('R_CLA_FIN_x', 'CLA_MID_x')
        marker['R_SHOULDER_y'] = calculate_midpoint('R_CLA_FIN_y', 'CLA_MID_y')
        marker['R_SHOULDER_z'] = calculate_midpoint('R_CLA_FIN_z', 'CLA_MID_z')

        marker['L_SHOULDER_x'] = calculate_midpoint('CLA_MID_x', 'L_CLA_FIN_x')
        marker['L_SHOULDER_y'] = calculate_midpoint('CLA_MID_y', 'L_CLA_FIN_y')
        marker['L_SHOULDER_z'] = calculate_midpoint('CLA_MID_z', 'L_CLA_FIN_z')

        marker['R_HIP_x'] = calculate_midpoint('R_ASIS_x', 'R_PSIS_x')
        marker['R_HIP_y'] = calculate_midpoint('R_ASIS_y', 'R_PSIS_y')
        marker['R_HIP_z'] = calculate_midpoint('R_ASIS_z', 'R_PSIS_z')

        marker['L_HIP_x'] = calculate_midpoint('L_ASIS_x', 'L_PSIS_x')
        marker['L_HIP_y'] = calculate_midpoint('L_ASIS_y', 'L_PSIS_y')
        marker['L_HIP_z'] = calculate_midpoint('L_ASIS_z', 'L_PSIS_z')

        marker['R_ELBOW_x'] = calculate_midpoint('R_ELBOW_L_x', 'R_ELBOW_M_x')
        marker['R_ELBOW_y'] = calculate_midpoint('R_ELBOW_L_y', 'R_ELBOW_M_y')
        marker['R_ELBOW_z'] = calculate_midpoint('R_ELBOW_L_z', 'R_ELBOW_M_z')

        marker['R_WRIST_x'] = calculate_midpoint('R_WRIST_L_x', 'R_WRIST_M_x')
        marker['R_WRIST_y'] = calculate_midpoint('R_WRIST_L_y', 'R_WRIST_M_y')
        marker['R_WRIST_z'] = calculate_midpoint('R_WRIST_L_z', 'R_WRIST_M_z')

        marker['R_KNEE_x'] = calculate_midpoint('R_KNEE_L_x', 'R_KNEE_M_x')
        marker['R_KNEE_y'] = calculate_midpoint('R_KNEE_L_y', 'R_KNEE_M_y')
        marker['R_KNEE_z'] = calculate_midpoint('R_KNEE_L_z', 'R_KNEE_M_z')

        marker['R_ANKLE_x'] = calculate_midpoint('R_ANKLE_L_x', 'R_ANKLE_M_x')
        marker['R_ANKLE_y'] = calculate_midpoint('R_ANKLE_L_y', 'R_ANKLE_M_y')
        marker['R_ANKLE_z'] = calculate_midpoint('R_ANKLE_L_z', 'R_ANKLE_M_z')

        # R_HEEL과 R_TOE1은 그대로 사용
        marker['R_HEEL_x'] = marker['R_HEEL_x']
        marker['R_HEEL_y'] = marker['R_HEEL_y']
        marker['R_HEEL_z'] = marker['R_HEEL_z']

        marker['R_TOE_x'] = marker['R_TOE1_x']
        marker['R_TOE_y'] = marker['R_TOE1_y']
        marker['R_TOE_z'] = marker['R_TOE1_z']

        marker['L_ELBOW_x'] = calculate_midpoint('L_ELBOW_L_x', 'L_ELBOW_M_x')
        marker['L_ELBOW_y'] = calculate_midpoint('L_ELBOW_L_y', 'L_ELBOW_M_y')
        marker['L_ELBOW_z'] = calculate_midpoint('L_ELBOW_L_z', 'L_ELBOW_M_z')

        marker['L_WRIST_x'] = calculate_midpoint('L_WRIST_L_x', 'L_WRIST_M_x')
        marker['L_WRIST_y'] = calculate_midpoint('L_WRIST_L_y', 'L_WRIST_M_y')
        marker['L_WRIST_z'] = calculate_midpoint('L_WRIST_L_z', 'L_WRIST_M_z')

        marker['L_KNEE_x'] = calculate_midpoint('L_KNEE_L_x', 'L_KNEE_M_x')
        marker['L_KNEE_y'] = calculate_midpoint('L_KNEE_L_y', 'L_KNEE_M_y')
        marker['L_KNEE_z'] = calculate_midpoint('L_KNEE_L_z', 'L_KNEE_M_z')

        marker['L_ANKLE_x'] = calculate_midpoint('L_ANKLE_L_x', 'L_ANKLE_M_x')
        marker['L_ANKLE_y'] = calculate_midpoint('L_ANKLE_L_y', 'L_ANKLE_M_y')
        marker['L_ANKLE_z'] = calculate_midpoint('L_ANKLE_L_z', 'L_ANKLE_M_z')

        # L_HEEL과 L_TOE1은 그대로 사용
        marker['L_HEEL_x'] = marker['L_HEEL_x']
        marker['L_HEEL_y'] = marker['L_HEEL_y']
        marker['L_HEEL_z'] = marker['L_HEEL_z']

        marker['L_TOE_x'] = marker['L_TOE1_x']
        marker['L_TOE_y'] = marker['L_TOE1_y']
        marker['L_TOE_z'] = marker['L_TOE1_z']

        # 중간값을 구하는 데 사용된 원래 열들 제거
        columns_to_drop = [
            'R_CLA_FIN_x', 'R_CLA_FIN_y', 'R_CLA_FIN_z',
            'CLA_MID_x', 'CLA_MID_y', 'CLA_MID_z',
            'L_CLA_FIN_x', 'L_CLA_FIN_y', 'L_CLA_FIN_z',
            'R_ELBOW_L_x', 'R_ELBOW_L_y', 'R_ELBOW_L_z',
            'R_ELBOW_M_x', 'R_ELBOW_M_y', 'R_ELBOW_M_z',
            'R_WRIST_L_x', 'R_WRIST_L_y', 'R_WRIST_L_z',
            'R_WRIST_M_x', 'R_WRIST_M_y', 'R_WRIST_M_z',
            'L_ELBOW_L_x', 'L_ELBOW_L_y', 'L_ELBOW_L_z',
            'L_ELBOW_M_x', 'L_ELBOW_M_y', 'L_ELBOW_M_z',
            'L_WRIST_L_x', 'L_WRIST_L_y', 'L_WRIST_L_z',
            'L_WRIST_M_x', 'L_WRIST_M_y', 'L_WRIST_M_z',
            'R_ASIS_x', 'R_ASIS_y', 'R_ASIS_z',
            'L_ASIS_x', 'L_ASIS_y', 'L_ASIS_z',
            'R_PSIS_x', 'R_PSIS_y', 'R_PSIS_z',
            'L_PSIS_x', 'L_PSIS_y', 'L_PSIS_z',
            'R_KNEE_L_x', 'R_KNEE_L_y', 'R_KNEE_L_z',
            'R_KNEE_M_x', 'R_KNEE_M_y', 'R_KNEE_M_z',
            'R_ANKLE_L_x', 'R_ANKLE_L_y', 'R_ANKLE_L_z',
            'R_ANKLE_M_x', 'R_ANKLE_M_y', 'R_ANKLE_M_z',
            'L_KNEE_L_x', 'L_KNEE_L_y', 'L_KNEE_L_z',
            'L_KNEE_M_x', 'L_KNEE_M_y', 'L_KNEE_M_z',
            'L_ANKLE_L_x', 'L_ANKLE_L_y', 'L_ANKLE_L_z',
            'L_ANKLE_M_x', 'L_ANKLE_M_y', 'L_ANKLE_M_z',
            'R_TOE5_x', 'R_TOE5_y', 'R_TOE5_z',
            'L_TOE5_x', 'L_TOE5_y', 'L_TOE5_z',

        ]

        marker = marker.drop(columns=columns_to_drop)

        new_order = [
            'R_SHOULDER_x', 'R_SHOULDER_y', 'R_SHOULDER_z', 'L_SHOULDER_x', 'L_SHOULDER_y', 'L_SHOULDER_z',
            'R_ELBOW_x', 'R_ELBOW_y', 'R_ELBOW_z', 'L_ELBOW_x', 'L_ELBOW_y', 'L_ELBOW_z',
            'R_WRIST_x', 'R_WRIST_y', 'R_WRIST_z', 'L_WRIST_x', 'L_WRIST_y', 'L_WRIST_z',
            'R_HIP_x', 'R_HIP_y', 'R_HIP_z', 'L_HIP_x', 'L_HIP_y', 'L_HIP_z',
            'R_KNEE_x', 'R_KNEE_y', 'R_KNEE_z', 'L_KNEE_x', 'L_KNEE_y', 'L_KNEE_z',
            'R_ANKLE_x', 'R_ANKLE_y', 'R_ANKLE_z', 'L_ANKLE_x', 'L_ANKLE_y', 'L_ANKLE_z',
            'R_HEEL_x', 'R_HEEL_y', 'R_HEEL_z', 'L_HEEL_x', 'L_HEEL_y', 'L_HEEL_z',
            'R_TOE_x', 'R_TOE_y', 'R_TOE_z', 'L_TOE_x', 'L_TOE_y', 'L_TOE_z'
        ]

        # 열 순서 변경
        marker = marker[new_order]
        marker=marker.iloc[::2,:]   # 120fps에서 2frame씩 걸러 입력-> 60fps
        return marker

    def get_xyz(self, detection_result):  # 각 관절 segment를 저장하는 함수
        res = []
        for i in points:
            try:
                res.append(detection_result.pose_landmarks[0][i].x)
            except Exception as e:
                res.append(-10)

            try:
                res.append(detection_result.pose_landmarks[0][i].y)
            except Exception as e:
                res.append(-10)

            try:
                res.append(detection_result.pose_landmarks[0][i].z)
            except Exception as e:
                res.append(-10)

        return res

    def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image

    def make_markerless_csv(self):
        # 일괄처리 코드
        # Create base options for the PoseLandmarker

        base_options = python.BaseOptions(model_asset_path=f"{self.rootPath}\\pose_landmarker_heavy.task")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        # Define input and output directories
        input_dir = f"{self.rootPath}\\markerless_video"
        output_dir = f"{self.rootPath}\\markerless_raw_csv"

        # Get all video files in the input directory
        video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mov', '.mp4', '.avi'))]

        # Process each video file
        for video_file in video_files:
            video_path = os.path.join(input_dir, video_file)
            
            # Open the video file using OpenCV
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2  # Reduce frame size by half
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

            # Initialize an empty list to store DataFrame rows
            rows = []

            while cap.isOpened():
                ret, frame = cap.read()

                # End loop if the video ends or reading fails
                if not ret:
                    break

                frame = cv2.resize(frame, (frame_width, frame_height))

                # Convert OpenCV frame to Mediapipe image
                mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                # Detect pose landmarks
                detection_result = detector.detect(mp_frame)

                # Get xyz coordinates
                d = self.get_xyz(detection_result)

                # Add coordinates to rows list
                rows.append(d)

                # Optional: Visualize landmarks
                #annotated_frame = draw_landmarks_on_image(frame, detection_result)
                #cv2.imshow('Pose Detection', annotated_frame)   # landmarks 적용 video 보여주는 코드

                # Check for 'q' key press to exit loop and stop processing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting process...")
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit()  # 전체 스크립트 종료

            # Release the video and close windows
            cap.release()
            cv2.destroyAllWindows()

            # Convert rows to DataFrame
            markerless = pd.DataFrame(rows)
            
            # Assume get_xyz() returns a flat list of length 36: [x1, y1, z1, x2, y2, z2, ..., x12, y12, z12]
            # Reshape the DataFrame
            columns = []
            for point in points:
                for suffix in ['x', 'y', 'z']:
                    columns.append(f"{point}{suffix}")
            markerless.columns = ['R_SHOULDER_x', 'R_SHOULDER_y', 'R_SHOULDER_z', 'L_SHOULDER_x', 'L_SHOULDER_y', 'L_SHOULDER_z', 
                'R_ELBOW_x', 'R_ELBOW_y', 'R_ELBOW_z','L_ELBOW_x', 'L_ELBOW_y', 'L_ELBOW_z',
                'R_WRIST_x', 'R_WRIST_y', 'R_WRIST_z', 'L_WRIST_x', 'L_WRIST_y', 'L_WRIST_z',
                'R_HIP_x', 'R_HIP_y', 'R_HIP_z', 'L_HIP_x', 'L_HIP_y', 'L_HIP_z', 
                'R_KNEE_x', 'R_KNEE_y', 'R_KNEE_z', 'L_KNEE_x', 'L_KNEE_y', 'L_KNEE_z',
                'R_ANKLE_x', 'R_ANKLE_y', 'R_ANKLE_z', 'L_ANKLE_x', 'L_ANKLE_y', 'L_ANKLE_z',
            'R_HEEL_x','R_HEEL_y','R_HEEL_z','L_HEEL_x','L_HEEL_y','L_HEEL_z',
                'R_TOE_x','R_TOE_y','R_TOE_z','L_TOE_x','L_TOE_y','L_TOE_z']

            # Save DataFrame to CSV in the output directory
            output_file_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.csv")
            markerless.to_csv(output_file_path, index=False)
            
    def return_markerless_csv(self):
        markerless = pd.read_csv(f'{self.rootPath}\\markerless_raw_csv\\{self.markerless_filename}').iloc[:-19,:]
        return markerless

    def marker_markerless_cut(self, marker, markerless):
            # 길이 맞추기
        if marker.shape[0]>=markerless.shape[0]:
            marker=marker.iloc[-markerless.shape[0]:,:]
        else:
            markerless=markerless.iloc[-marker.shape[0]:,:]
        marker=marker.reset_index(drop=True)
        markerless=markerless.reset_index(drop=True)
        cut=marker.shape[1]
        # 전처리 2 (필요 행 추출)
        # 열 방향으로 데이터 결합
        conc = pd.concat([marker, markerless], axis=1,)

        # 각 열에서 모든 값이 -10인 값 제거
        conc = conc[conc.iloc[:,-1]!=-10]

        # 마커기반 및 마커리스기반 데이터 분리
        marker = conc.iloc[:, :cut]
        markerless = conc.iloc[:, cut:]
        return marker, markerless
    
    # 버터워스 저역 통과 필터 함수 정의
    def butter_lowpass_filter(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data, axis=0)
        return y
    
    def filtering(self, marker_cut, markerless_cut):
        # 필터 매개변수 정의
        cutoff = 6    # 컷오프 주파수
        fs = 60.0     # 샘플링 주파수
        order = 4      # 필터 차수
        
        marker_e={}
        markerless_e={}
        for i in marker_cut.columns:
            marker_e[i]=self.butter_lowpass_filter(marker_cut[i],cutoff,fs,order)
            markerless_e[i]=self.butter_lowpass_filter(markerless_cut[i],cutoff,fs,order)

        marker_filter=pd.DataFrame(marker_e)
        markerless_filter=pd.DataFrame(markerless_e)
        return marker_filter, markerless_filter
        
    def make_complite(self):
        marker = self.return_marker_csv()
        # self.make_markerless_csv()
        markerless = self.return_markerless_csv()
        
        marker_cut, markerless_cut = self.marker_markerless_cut(marker, markerless)
        marker_filter, markerless_filter = self.filtering(marker_cut, markerless_cut)
        
        markerless_filter = abs(1-markerless_filter)
        return marker_filter, markerless_filter
        