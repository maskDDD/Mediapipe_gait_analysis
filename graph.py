import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

class write_graph():
    def __init__(self, marker, markerless, landmark, axis):
        self.marker = marker
        self.markerless = markerless
        self.landmark = landmark
        if axis == "h": # 높이일 경우
            marker_axis = "z"
            markerless_axis = "y"
            self.markerless = abs(1-self.markerless)  # 마커리스 축 뒤집기
            
        elif axis == "w": # 너비일 경우
            marker_axis = "y"
            markerless_axis = "x"
        else:
            exit(1)
        
        self.landmark_marker_R = f"R_{landmark}_{marker_axis}"
        self.landmark_marker_L = f"L_{landmark}_{marker_axis}"
        self.landmark_markerless_R = f"R_{landmark}_{markerless_axis}"
        self.landmark_markerless_L= f"L_{landmark}_{markerless_axis}"

    def visualize(self):
        # 결과 시각화
        plt.figure(figsize=(14, 8))

        # 마커기반 데이터 시각화
        plt.subplot(2, 1, 1)
        plt.plot(self.marker[self.landmark_marker_R], label='Right', linestyle='-', alpha=0.7)
        plt.plot(self.marker[self.landmark_marker_L], label='Left', linestyle='--', alpha=0.7)
        plt.title(f"Marker-Based {self.landmark}")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        # 마커리스기반 데이터 시각화
        plt.subplot(2, 1, 2)
        plt.plot(self.markerless[self.landmark_markerless_R], label='Right', linestyle='-', alpha=0.7)
        plt.plot(self.markerless[self.landmark_markerless_L], label='Left', linestyle='--', alpha=0.7)
        plt.title(f"Markerless-Based {self.landmark}")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
    def differential(self): # 미분
        plt.figure(figsize=(14, 8))
        # 마커

        m_yR= self.marker[self.landmark_marker_R]
        m_yL= self.marker[self.landmark_marker_L]
        m_x = self.marker.index
        self.mdy_dxR = np.gradient(m_yR,m_x)
        self.mdy_dxL = np.gradient(m_yL,m_x)

        plt.subplot(2, 1, 1)
        plt.plot(self.mdy_dxR, label='Original', linestyle='-', alpha=0.7)
        plt.plot(self.mdy_dxL, label='Original', linestyle='--', alpha=0.7)
        plt.title(f"Marker-Based HEEL")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        # 마커리스
        ml_yR= self.markerless[self.landmark_markerless_R]
        ml_yL= self.markerless[self.landmark_markerless_L]
        ml_x = self.markerless.index
        self.mldy_dxR = np.gradient(ml_yR,ml_x)
        self.mldy_dxL = np.gradient(ml_yL,ml_x)

        plt.subplot(2, 1, 2)
        plt.plot(self.mldy_dxR, label='Original', linestyle='-', alpha=0.7)
        plt.plot(self.mldy_dxL, label='Original', linestyle='--', alpha=0.7)
        plt.title(f"Markerless-Based HEEL")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)
        
    def feature_extraction(self):
        mldy_dxR = list(self.mldy_dxR)
        mldy_dxL = list(self.mldy_dxL)
        
        whR =[]
        wr_flagR = 0
        for i in range(len(mldy_dxR)-1):
            if mldy_dxR[i]>0.005:
                wr_flagR=1
            c=i
            n=i+1
            if mldy_dxR[c]*mldy_dxR[n]<=0 and wr_flagR==1:
                whR.append(c)
                wr_flagR=0

        whL=[]
        wr_flagL = 0
        for i in range(len(mldy_dxL)-1):
            if mldy_dxL[i]>0.005:
                wr_flagL=1
            c=i
            n=i+1
            if mldy_dxL[c]*mldy_dxL[n]<=0 and wr_flagL==1:
                whL.append(c)
                wr_flagL=0
                
        L_HEEL=list(self.markerless[self.landmark_marker_L])
        L_HEEL_EV1=L_HEEL[whL[0]:whR[0]]
        L_HEEL_EV2=L_HEEL[whL[1]:whR[1]]
        L_HEEL_EV1_ml_min=L_HEEL_EV1.index(min(L_HEEL_EV1))
        L_HEEL_EV2_ml_min=L_HEEL_EV2.index(min(L_HEEL_EV2))
        L_HEEL_EV1_ml_min=L_HEEL_EV1_ml_min+whL[0]+1
        L_HEEL_EV2_ml_min=L_HEEL_EV2_ml_min+whL[1]+1
        
         # 마커리스 시각화
        plt.subplot(2, 1, 2)
        plt.plot(self.markerless[self.landmark_markerless_R], label='Original', linestyle='-', alpha=0.7)
        plt.plot(self.markerless[self.landmark_markerless_L], label='Original', linestyle='--', alpha=0.7)
        plt.axvline(x=L_HEEL_EV1_ml_min, color='red', linestyle='--', linewidth=2)
        plt.axvline(x=L_HEEL_EV2_ml_min, color='red', linestyle='--', linewidth=2)
        plt.title(f"Markerless-Based {self.landmark}")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
       
    def toe_off(self):
       # toe off!!!

        # 사용하려는 마커를 R_TOE로 설정
        landmark_marker = self.landmark_marker_R
        landmark_markerless =  self.landmark_markerless_R

        # 결과 시각화
        plt.figure(figsize=(14, 8))

        # 마커기반 데이터에서 봉우리 감지 (R_TOE 기준)
        marker_peaks_R, _ = find_peaks(self.marker[landmark_marker])

        # 가장 높은 봉우리(피크) 찾기 (R_TOE 기준)
        max_R_peak = marker_peaks_R[np.argmax(self.marker[landmark_marker].iloc[marker_peaks_R])]

        # 피크 앞쪽에서 최저점 찾기 (R_TOE 기준)
        min_R_before_peak = self.marker[landmark_marker][:max_R_peak].idxmin()

        # 가장 큰 봉우리 이후의 봉우리를 찾기
        peaks_after_max = marker_peaks_R[marker_peaks_R > max_R_peak]
        if len(peaks_after_max) > 0:
            second_max_R_peak = peaks_after_max[np.argmax(self.marker[landmark_marker].iloc[peaks_after_max])]
            min_R_between_peaks = self.marker[landmark_marker][max_R_peak:second_max_R_peak].idxmin()
        else:
            second_max_R_peak = None
            min_R_between_peaks = None

        # 마커리스기반 데이터에서 봉우리 감지 (R_TOE 기준)
        markerless_peaks_R, _ = find_peaks(self.markerless[landmark_markerless])

        # 가장 높은 봉우리(피크) 찾기 (R_TOE 기준)
        max_R_peak_ml = markerless_peaks_R[np.argmax(self.markerless[landmark_markerless].iloc[markerless_peaks_R])]

        # 피크 앞쪽에서 최저점 찾기 (R_TOE 기준)
        min_R_before_peak_ml = self.markerless[landmark_markerless][:max_R_peak_ml].idxmin()

        # 가장 큰 봉우리 이후의 봉우리를 찾기 (마커리스 기준)
        peaks_after_max_ml = markerless_peaks_R[markerless_peaks_R > max_R_peak_ml]
        if len(peaks_after_max_ml) > 0:
            second_max_R_peak_ml = peaks_after_max_ml[np.argmax(self.markerless[landmark_markerless].iloc[peaks_after_max_ml])]
            min_R_between_peaks_ml = self.markerless[landmark_markerless][max_R_peak_ml:second_max_R_peak_ml].idxmin()
        else:
            second_max_R_peak_ml = None
            min_R_between_peaks_ml = None

        # 마커기반 데이터 시각화 (R_TOE)
        plt.subplot(2, 1, 1)
        plt.plot(self.marker[landmark_marker], label='R_TOE', linestyle='-', alpha=0.7)
        plt.axvline(x=min_R_before_peak, color='red', linestyle='--', linewidth=2, label='TO1')  # 봉우리 앞쪽 최저점 표시
        if second_max_R_peak is not None:
            plt.axvline(x=min_R_between_peaks, color='green', linestyle='--', linewidth=2, label='TO2')  # 봉우리 사이 최저점 표시
        plt.title(f"Marker-Based {landmark_marker}")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        # 마커리스기반 데이터 시각화 (R_TOE)
        plt.subplot(2, 1, 2)
        plt.plot(self.markerless[landmark_markerless], label='R_TOE', linestyle='-', alpha=0.7)
        plt.axvline(x=min_R_before_peak_ml, color='red', linestyle='--', linewidth=2, label='TO1')  # 봉우리 앞쪽 최저점 표시
        if second_max_R_peak_ml is not None:
            plt.axvline(x=min_R_between_peaks_ml, color='green', linestyle='--', linewidth=2, label='TO2')  # 봉우리 사이 최저점 표시
        plt.title(f"Markerless-Based {landmark_markerless}")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
