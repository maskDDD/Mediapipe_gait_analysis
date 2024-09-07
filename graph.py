import matplotlib.pyplot as plt
import numpy as np

class write_graph():
    def __init__(self, marker, markerless, landmark, axis):
        self.marker = marker
        self.markerless = markerless
        self.landmark = landmark
        if axis == "h": # 높이일 경우
            marker_axis = "z"
            markerless_axis = "y"
            
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
        plt.plot(self.marker[self.landmark_marker_R], label='Original', linestyle='-', alpha=0.7)
        plt.plot(self.marker[self.landmark_marker_L], label='Original', linestyle='--', alpha=0.7)
        plt.title(f"Marker-Based {self.landmark}")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        # 마커리스기반 데이터 시각화
        plt.subplot(2, 1, 2)
        plt.plot(self.markerless[self.landmark_markerless_R], label='Original', linestyle='-', alpha=0.7)
        plt.plot(self.markerless[self.landmark_markerless_L], label='Original', linestyle='--', alpha=0.7)
        plt.title(f"Markerless-Based {self.landmark}")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
    def differential(self):
        plt.figure(figsize=(14, 8))
        # 마커

        m_yR= self.marker[self.landmark_marker_R]
        m_yL= self.marker[self.landmark_marker_L]
        m_x = self.marker.index
        mdy_dxR = np.gradient(m_yR,m_x)
        mdy_dxL = np.gradient(m_yL,m_x)

        plt.subplot(2, 1, 1)
        plt.plot(mdy_dxR, label='Original', linestyle='-', alpha=0.7)
        plt.plot(mdy_dxL, label='Original', linestyle='--', alpha=0.7)
        plt.title(f"Marker-Based HEEL")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        # 마커리스
        ml_yR= self.markerless[self.landmark_markerless_R]
        ml_yL= self.markerless[self.landmark_markerless_L]
        ml_x = self.markerless.index
        mldy_dxR = np.gradient(ml_yR,ml_x)
        mldy_dxL = np.gradient(ml_yL,ml_x)

        plt.subplot(2, 1, 2)
        plt.plot(mldy_dxR, label='Original', linestyle='-', alpha=0.7)
        plt.plot(mldy_dxL, label='Original', linestyle='--', alpha=0.7)
        plt.title(f"Markerless-Based HEEL")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

    def feature_graph(self):
        fill_this_empty_space = 0