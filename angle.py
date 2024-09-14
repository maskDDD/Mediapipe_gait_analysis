from mediapipe.framework.formats import landmark_pb2
import math
import matplotlib.pyplot as plt
import numpy as np

class Angle():
    def __init__(self, marker, markerless):
        self.marker = marker
        self.markerless = markerless
        
    # 2D 각도 계산 함수
    def calculate_angle_2d(self, joint_coords, num=3):
        # 2D에서 AB 및 BC 벡터 생성 (xy 평면에서 작업)
        if num == 3:
            AB = (joint_coords[0][0] - joint_coords[1][0], joint_coords[0][1] - joint_coords[1][1])
            BC = (joint_coords[2][0] - joint_coords[1][0], joint_coords[2][1] - joint_coords[1][1])
        else:
            AB = (joint_coords[0][0] - joint_coords[1][0], joint_coords[0][1] - joint_coords[1][1])
            BC = (joint_coords[2][0] - joint_coords[3][0], joint_coords[2][1] - joint_coords[3][1])
            
        # AB와 BC의 내적 계산
        dot_product = AB[0] * BC[0] + AB[1] * BC[1]

        # AB와 BC 벡터의 크기 계산
        magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
        magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)

        # 벡터의 크기가 0인 경우, 계산이 불가능하므로 NaN 반환
        if magnitude_AB * magnitude_BC == 0:
            return np.nan

        # 코사인 값 계산
        cos_theta = dot_product / (magnitude_AB * magnitude_BC)

        # 각도를 라디안으로 변환 후, 다시 도(degree)로 변환
        angle_radians = math.acos(cos_theta)
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    # 3D 각도 계산 함수
    def calculate_angle_3d(self, joint_coords, num=3):
        # 3D에서 AB 및 BC 벡터 생성
        if num == 3:
            AB = (joint_coords[0][0] - joint_coords[1][0], joint_coords[0][1] - joint_coords[1][1], joint_coords[0][2] - joint_coords[1][2])
            BC = (joint_coords[2][0] - joint_coords[1][0], joint_coords[2][1] - joint_coords[1][1], joint_coords[2][2] - joint_coords[1][2])
        else:
            AB = (joint_coords[0][0] - joint_coords[1][0], joint_coords[0][1] - joint_coords[1][1], joint_coords[0][2] - joint_coords[1][2])
            BC = (joint_coords[2][0] - joint_coords[3][0], joint_coords[2][1] - joint_coords[3][1], joint_coords[2][2] - joint_coords[3][2])
            
        # AB와 BC의 내적 계산
        dot_product = AB[0] * BC[0] + AB[1] * BC[1] + AB[2] * BC[2]

        # AB와 BC 벡터의 크기 계산
        magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2 + AB[2]**2)
        magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2 + BC[2]**2)

        # 벡터의 크기가 0인 경우, 계산이 불가능하므로 NaN 반환
        if magnitude_AB * magnitude_BC == 0:
            return np.nan

        # 코사인 값 계산
        cos_theta = dot_product / (magnitude_AB * magnitude_BC)

        # 각도를 라디안으로 변환 후, 도(degree)로 변환
        angle_radians = math.acos(cos_theta)
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    # 각 관절의 2D 및 3D 각도를 계산하는 함수
    def add_joint_angles_to_data_marker(self, data):
        # # 3D 관절 각도 계산
        # data['left_shoulder_flexion_3d'] = data.apply(lambda row: calculate_angle_3d([[row['L_HIP_x'], row['L_HIP_y'], row['L_HIP_z']],
        #                                                                              [row['L_SHOULDER_x'], row['L_SHOULDER_y'], row['L_SHOULDER_z']],
        #                                                                              [row['L_ELBOW_x'], row['L_ELBOW_y'], row['L_ELBOW_z']]]), axis=1)
        
        # data['right_shoulder_flexion_3d'] = data.apply(lambda row: calculate_angle_3d([[row['R_HIP_x'], row['R_HIP_y'], row['R_HIP_z']],
        #                                                                               [row['R_SHOULDER_x'], row['R_SHOULDER_y'], row['R_SHOULDER_z']],
        #                                                                               [row['R_ELBOW_x'], row['R_ELBOW_y'], row['R_ELBOW_z']]]), axis=1)
        
        # 2D 관절 각도 계산
        data['left_shoulder_flexion_2d'] = data.apply(lambda row: self.calculate_angle_2d([[row['L_HIP_y'], row['L_HIP_z']],
                                                                                    [row['L_SHOULDER_y'], row['L_SHOULDER_z']],
                                                                                    [row['L_ELBOW_y'], row['L_ELBOW_z']]]), axis=1)
        
        data['right_shoulder_flexion_2d'] = data.apply(lambda row: self.calculate_angle_2d([[row['R_HIP_y'], row['R_HIP_z']],
                                                                                    [row['R_SHOULDER_y'], row['R_SHOULDER_z']],
                                                                                    [row['R_ELBOW_y'], row['R_ELBOW_z']]]), axis=1)
        
        # 팔꿈치, 무릎 및 엉덩이의 3D 각도 계산
        data['left_elbow_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['L_SHOULDER_x'], row['L_SHOULDER_y'], row['L_SHOULDER_z']],
                                                                            [row['L_ELBOW_x'], row['L_ELBOW_y'], row['L_ELBOW_z']],
                                                                            [row['L_WRIST_x'], row['L_WRIST_y'], row['L_WRIST_z']]]), axis=1)
        
        data['right_elbow_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['R_SHOULDER_x'], row['R_SHOULDER_y'], row['R_SHOULDER_z']],
                                                                                [row['R_ELBOW_x'], row['R_ELBOW_y'], row['R_ELBOW_z']],
                                                                                [row['R_WRIST_x'], row['R_WRIST_y'], row['R_WRIST_z']]]), axis=1)
        
        # 3D 무릎 굴곡 각도 계산
        data['left_knee_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['L_HIP_x'], row['L_HIP_y'], row['L_HIP_z']],
                                                                            [row['L_KNEE_x'], row['L_KNEE_y'], row['L_KNEE_z']],
                                                                            [row['L_ANKLE_x'], row['L_ANKLE_y'], row['L_ANKLE_z']]]), axis=1)
        
        data['right_knee_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['R_HIP_x'], row['R_HIP_y'], row['R_HIP_z']],
                                                                            [row['R_KNEE_x'], row['R_KNEE_y'], row['R_KNEE_z']],
                                                                            [row['R_ANKLE_x'], row['R_ANKLE_y'], row['R_ANKLE_z']]]), axis=1)

        # 엉덩이 굴곡 및 내전 각도 계산
        data['left_hip_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['L_SHOULDER_x'], row['L_SHOULDER_y'], row['L_SHOULDER_z']],
                                                                            [row['L_HIP_x'], row['L_HIP_y'], row['L_HIP_z']],
                                                                            [row['L_KNEE_x'], row['L_KNEE_y'], row['L_KNEE_z']]]), axis=1)
        
        data['right_hip_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['R_SHOULDER_x'], row['R_SHOULDER_y'], row['R_SHOULDER_z']],
                                                                            [row['R_HIP_x'], row['R_HIP_y'], row['R_HIP_z']],
                                                                            [row['R_KNEE_x'], row['R_KNEE_y'], row['R_KNEE_z']]]), axis=1)
        
        data['left_hip_adduction'] = data.apply(lambda row: self.calculate_angle_2d([[row['L_KNEE_z'], row['L_KNEE_y']],
                                                                            [row['L_HIP_z'], row['L_HIP_y']],
                                                                            [(row['L_HIP_z'] + row['R_HIP_z']) / 2, (row['L_HIP_y'] + row['R_HIP_y']) / 2],
                                                                            [(row['R_SHOULDER_z'] + row['L_SHOULDER_z']) / 2, (row['R_SHOULDER_y'] + row['L_SHOULDER_y']) / 2]]), axis=1)
        
        data['right_hip_adduction'] = data.apply(lambda row: self.calculate_angle_2d([[row['R_KNEE_z'], row['R_KNEE_y']],
                                                                                [row['R_HIP_z'], row['R_HIP_y']],
                                                                                [(row['L_HIP_z'] + row['R_HIP_z']) / 2, (row['L_HIP_y'] + row['R_HIP_y']) / 2],
                                                                                [(row['R_SHOULDER_z'] + row['L_SHOULDER_z']) / 2, (row['R_SHOULDER_y'] + row['L_SHOULDER_y']) / 2]]), axis=1)

        # 골반의 리스트 및 회전 측정
        data['pelvis_list'] = data['R_HIP_y'] - data['L_HIP_y']
        data['pelvis_rotation'] = data['R_HIP_x'] - data['L_HIP_x']
        
        return data

    # 각 관절의 2D 및 3D 각도를 계산하는 함수
    def add_joint_angles_to_data_markerless(self, data):
        # # 3D 관절 각도 계산
        # data['left_shoulder_flexion_3d'] = data.apply(lambda row: calculate_angle_3d([[row['L_HIP_x'], row['L_HIP_y'], row['L_HIP_z']],
        #                                                                              [row['L_SHOULDER_x'], row['L_SHOULDER_y'], row['L_SHOULDER_z']],
        #                                                                              [row['L_ELBOW_x'], row['L_ELBOW_y'], row['L_ELBOW_z']]]), axis=1)
        
        # data['right_shoulder_flexion_3d'] = data.apply(lambda row: calculate_angle_3d([[row['R_HIP_x'], row['R_HIP_y'], row['R_HIP_z']],
        #                                                                               [row['R_SHOULDER_x'], row['R_SHOULDER_y'], row['R_SHOULDER_z']],
        #                                                                               [row['R_ELBOW_x'], row['R_ELBOW_y'], row['R_ELBOW_z']]]), axis=1)
        
        # 2D 관절 각도 계산
        data['left_shoulder_flexion_2d'] = data.apply(lambda row: self.calculate_angle_2d([[row['L_HIP_x'], row['L_HIP_y']],
                                                                                    [row['L_SHOULDER_x'], row['L_SHOULDER_y']],
                                                                                    [row['L_ELBOW_x'], row['L_ELBOW_y']]]), axis=1)
        
        data['right_shoulder_flexion_2d'] = data.apply(lambda row: self.calculate_angle_2d([[row['R_HIP_x'], row['R_HIP_y']],
                                                                                    [row['R_SHOULDER_x'], row['R_SHOULDER_y']],
                                                                                    [row['R_ELBOW_x'], row['R_ELBOW_y']]]), axis=1)
        
        # 팔꿈치, 무릎 및 엉덩이의 3D 각도 계산
        data['left_elbow_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['L_SHOULDER_x'], row['L_SHOULDER_y'], row['L_SHOULDER_z']],
                                                                            [row['L_ELBOW_x'], row['L_ELBOW_y'], row['L_ELBOW_z']],
                                                                            [row['L_WRIST_x'], row['L_WRIST_y'], row['L_WRIST_z']]]), axis=1)
        
        data['right_elbow_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['R_SHOULDER_x'], row['R_SHOULDER_y'], row['R_SHOULDER_z']],
                                                                                [row['R_ELBOW_x'], row['R_ELBOW_y'], row['R_ELBOW_z']],
                                                                                [row['R_WRIST_x'], row['R_WRIST_y'], row['R_WRIST_z']]]), axis=1)
        
        # 3D 무릎 굴곡 각도 계산
        data['left_knee_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['L_HIP_x'], row['L_HIP_y'], row['L_HIP_z']],
                                                                            [row['L_KNEE_x'], row['L_KNEE_y'], row['L_KNEE_z']],
                                                                            [row['L_ANKLE_x'], row['L_ANKLE_y'], row['L_ANKLE_z']]]), axis=1)
        
        data['right_knee_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['R_HIP_x'], row['R_HIP_y'], row['R_HIP_z']],
                                                                            [row['R_KNEE_x'], row['R_KNEE_y'], row['R_KNEE_z']],
                                                                            [row['R_ANKLE_x'], row['R_ANKLE_y'], row['R_ANKLE_z']]]), axis=1)

        # 엉덩이 굴곡 및 내전 각도 계산
        data['left_hip_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['L_SHOULDER_x'], row['L_SHOULDER_y'], row['L_SHOULDER_z']],
                                                                            [row['L_HIP_x'], row['L_HIP_y'], row['L_HIP_z']],
                                                                            [row['L_KNEE_x'], row['L_KNEE_y'], row['L_KNEE_z']]]), axis=1)
        
        data['right_hip_flexion'] = data.apply(lambda row: self.calculate_angle_3d([[row['R_SHOULDER_x'], row['R_SHOULDER_y'], row['R_SHOULDER_z']],
                                                                            [row['R_HIP_x'], row['R_HIP_y'], row['R_HIP_z']],
                                                                            [row['R_KNEE_x'], row['R_KNEE_y'], row['R_KNEE_z']]]), axis=1)
        
        data['left_hip_adduction'] = data.apply(lambda row: self.calculate_angle_2d([[row['L_KNEE_x'], row['L_KNEE_y']],
                                                                            [row['L_HIP_x'], row['L_HIP_y']],
                                                                            [(row['L_HIP_x'] + row['R_HIP_x']) / 2, (row['L_HIP_y'] + row['R_HIP_y']) / 2],
                                                                            [(row['R_SHOULDER_x'] + row['L_SHOULDER_x']) / 2, (row['R_SHOULDER_y'] + row['L_SHOULDER_y']) / 2]]), axis=1)
        
        data['right_hip_adduction'] = data.apply(lambda row: self.calculate_angle_2d([[row['R_KNEE_x'], row['R_KNEE_y']],
                                                                                [row['R_HIP_x'], row['R_HIP_y']],
                                                                                [(row['L_HIP_x'] + row['R_HIP_x']) / 2, (row['L_HIP_y'] + row['R_HIP_y']) / 2],
                                                                                [(row['R_SHOULDER_x'] + row['L_SHOULDER_x']) / 2, (row['R_SHOULDER_y'] + row['L_SHOULDER_y']) / 2]]), axis=1)

        # 골반의 리스트 및 회전 측정
        data['pelvis_list'] = data['R_HIP_y'] - data['L_HIP_y']
        data['pelvis_rotation'] = data['R_HIP_x'] - data['L_HIP_x']
        
        return data

    def visualize(self):
        marker_angle=self.add_joint_angles_to_data_marker(self.marker)
        markerless_angle=self.add_joint_angles_to_data_markerless(self.markerless)
            # 결과 시각화
        plt.figure(figsize=(14, 8))

        # 마커기반 데이터 시각화
        plt.subplot(2, 1, 1)
        plt.plot(marker_angle["left_shoulder_flexion_2d"], label='left_shoulder_flexion_2d', linestyle='-', alpha=0.7)
        plt.title("Marker-based Data")
        plt.xlabel("Time")
        plt.ylabel("Flexion Angle")
        plt.legend()
        plt.grid(True)

        # 마커리스기반 데이터 시각화
        plt.subplot(2, 1, 2)
        plt.plot(markerless_angle["left_shoulder_flexion_2d"], label='left_shoulder_flexion_2d', linestyle='-', alpha=0.7)
        plt.title("Markerless-based Data")
        plt.xlabel("Time")
        plt.ylabel("Flexion Angle")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()