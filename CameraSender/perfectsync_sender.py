import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow警告を完全に非表示

import cv2
import mediapipe as mp
from pythonosc import udp_client
import numpy as np
import math
from collections import deque

# OSC設定
OSC_IP = "127.0.0.1"
OSC_PORT = 9011

PARAMS = {
    "JawOpen": "/avatar/parameters/JawOpen",
    "JawRight": "/avatar/parameters/JawRight",
    "JawLeft": "/avatar/parameters/JawLeft",
    "JawForward": "/avatar/parameters/JawForward",
    "MouthCornerPullRight": "/avatar/parameters/MouthCornerPullRight",
    "MouthCornerPullLeft": "/avatar/parameters/MouthCornerPullLeft",
    "MouthPucker": "/avatar/parameters/MouthPucker",
    "CheekPuffRight": "/avatar/parameters/CheekPuffRight",
    "CheekPuffLeft": "/avatar/parameters/CheekPuffLeft",
    "TongueOut": "/avatar/parameters/TongueOut",
    "TongueUp": "/avatar/parameters/TongueUp",
    "TongueDown": "/avatar/parameters/TongueDown",
    "TongueRight": "/avatar/parameters/TongueRight",
    "TongueLeft": "/avatar/parameters/TongueLeft"
}

class FacialTracker:
    def __init__(self):
        self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        
        # MediaPipe設定
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        # スムージング用のバッファ（適応的サイズ）
        self.smoothing_buffer = {}
        self.buffer_sizes = {
            'JawOpen': 3,  # 口の開閉は反応速度重視
            'TongueOut': 3,  # 舌も反応速度重視
            'TongueUp': 3,
            'TongueDown': 3,
            'TongueRight': 3,
            'TongueLeft': 3,
            'MouthCornerPullRight': 4,  # 口角は中程度のスムージング
            'MouthCornerPullLeft': 4,
            'MouthPucker': 5,  # 口すぼめは安定性重視
            'CheekPuffRight': 6,  # 頬膨らみは最も安定性重視
            'CheekPuffLeft': 6,
            'JawRight': 4,
            'JawLeft': 4,
            'JawForward': 4
        }
        for param in PARAMS.keys():
            buffer_size = self.buffer_sizes.get(param, 5)
            self.smoothing_buffer[param] = deque(maxlen=buffer_size)
        
        # 動的キャリブレーションシステム
        self.baseline = {}  # 安静時の基準値
        self.baseline_buffer = {}  # 基準値の更新用バッファ
        self.calibration_frames = 0
        self.calibration_needed = True
        self.auto_calibration_counter = 0
        self.last_movement_time = 0
        
        # 顔の角度補正用
        self.face_orientation_buffer = deque(maxlen=10)
        self.stable_face_orientation = None
        
        # 個人差吸収用のパラメータ
        self.face_scale_factor = 1.0  # 顔のサイズ補正
        self.mouth_aspect_ratio = 1.0  # 口の縦横比補正
        
        # パラメータの基準値バッファ初期化
        for param in PARAMS.keys():
            self.baseline_buffer[param] = deque(maxlen=60)  # 2秒間のバッファ
        
        # 口のランドマークインデックス（MediaPipeの468点ランドマーク）
        self.mouth_landmarks = {
            'outer_upper': [61, 185, 40, 39, 37, 0, 267, 269, 270, 267, 271, 272, 291],
            'outer_lower': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 81, 82, 13, 312, 311, 310, 415, 308, 291],
            'inner_upper': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
            'inner_lower': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
            'corners': [61, 291],  # 口角
            'vertical': [13, 14],  # 上唇中央・下唇中央
            'tongue_tip': [17, 18],  # 舌先
            'upper_lip_top': [12, 15],  # 上唇上端
            'lower_lip_bottom': [16, 17],  # 下唇下端
        }
        
        # 顔の基準点（安定した追跡のため）
        self.face_reference_points = [1, 2, 5, 4, 6, 19, 94, 125, 141, 235, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122, 6, 202, 214, 234]

    def calibrate(self, landmarks, w, h):
        """改善されたキャリブレーションシステム"""
        if self.calibration_frames < 60:  # 2秒間で基準値を収集
            params = self.calculate_raw_params(landmarks, w, h)
            
            # 顔の基本情報を収集
            mouth_width = abs(landmarks[61].x - landmarks[291].x) * w
            mouth_height = abs(landmarks[13].y - landmarks[14].y) * h
            face_width = abs(landmarks[234].x - landmarks[454].x) * w
            
            # 個人差のパラメータを計算
            if self.calibration_frames == 0:
                self.face_scale_factor = face_width / 100.0  # 基準サイズ100px
                self.mouth_aspect_ratio = mouth_height / (mouth_width + 1e-6)
            
            # パラメータの基準値を収集
            for key, val in params.items():
                if key not in self.baseline:
                    self.baseline[key] = []
                self.baseline[key].append(val)
            
            self.calibration_frames += 1
            return False
        else:
            # 平均値を基準として設定
            for key in self.baseline:
                values = self.baseline[key]
                # 外れ値を除去して平均を計算
                sorted_values = sorted(values)
                trim_count = len(sorted_values) // 10  # 上下10%をカット
                if trim_count > 0:
                    trimmed_values = sorted_values[trim_count:-trim_count]
                else:
                    trimmed_values = sorted_values
                self.baseline[key] = np.mean(trimmed_values)
            
            self.calibration_needed = False
            print(f"Calibration completed. Face scale: {self.face_scale_factor:.2f}, Mouth ratio: {self.mouth_aspect_ratio:.2f}")
            return True

    def get_rotated_point(self, cx, cy, x, y, angle):
        """点を回転させる（強化版）"""
        s = math.sin(angle)
        c = math.cos(angle)
        x -= cx
        y -= cy
        x_new = x * c - y * s
        y_new = x * s + y * c
        return x_new + cx, y_new + cy
    
    def update_baseline_adaptively(self, params):
        """適応的な基準値更新"""
        import time
        current_time = time.time()
        
        # 動きが小さいときのみ基準値を更新（顎の左右は除外）
        movement_detected = False
        for key, val in params.items():
            if key in self.baseline and key not in ['JawRight', 'JawLeft']:
                diff = abs(val - self.baseline[key])
                if diff > 0.1:
                    movement_detected = True
                    self.last_movement_time = current_time
                    break
        
        # 2秒間動きがない場合、基準値を更新（顎以外）
        if not movement_detected and (current_time - self.last_movement_time) > 2.0:
            for key, val in params.items():
                if key in self.baseline_buffer and key not in ['JawRight', 'JawLeft']:
                    self.baseline_buffer[key].append(val)
                    if len(self.baseline_buffer[key]) >= 20:
                        new_baseline = np.mean(list(self.baseline_buffer[key]))
                        # より積極的に更新（カメラ角度変化に対応）
                        self.baseline[key] = self.baseline[key] * 0.9 + new_baseline * 0.1
                        self.baseline_buffer[key].clear()

    def calculate_face_orientation(self, landmarks, w, h):
        """強化された顔の向き計算"""
        # 顔の基準点を使って向きを計算
        # 鼻先、顎、額を使って3D座標系を推定
        nose_tip = np.array([landmarks[1].x * w, landmarks[1].y * h, landmarks[1].z * w])
        chin = np.array([landmarks[18].x * w, landmarks[18].y * h, landmarks[18].z * w])
        forehead = np.array([landmarks[9].x * w, landmarks[9].y * h, landmarks[9].z * w])
        
        # 左右の頬を使ってロール角を計算
        left_cheek = np.array([landmarks[234].x * w, landmarks[234].y * h])
        right_cheek = np.array([landmarks[454].x * w, landmarks[454].y * h])
        
        # ロール角（Z軸回転）
        dx = right_cheek[0] - left_cheek[0]
        dy = right_cheek[1] - left_cheek[1]
        roll = math.atan2(dy, dx)
        
        # ピッチ角（X軸回転） - 顔の上下の傂き
        face_vertical = chin - forehead
        pitch = math.atan2(face_vertical[2], face_vertical[1])
        
        # ヨー角（Y軸回転） - 顔の左右の向き
        face_center = (left_cheek + right_cheek) / 2
        nose_offset = nose_tip[:2] - face_center
        yaw = math.atan2(nose_offset[0], nose_offset[1])
        
        orientation = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
        
        # 安定した向きを記録
        self.face_orientation_buffer.append(orientation)
        
        # 安定した向きを計算
        if len(self.face_orientation_buffer) >= 5:
            avg_roll = np.mean([o['roll'] for o in self.face_orientation_buffer])
            avg_pitch = np.mean([o['pitch'] for o in self.face_orientation_buffer])
            avg_yaw = np.mean([o['yaw'] for o in self.face_orientation_buffer])
            self.stable_face_orientation = {'roll': avg_roll, 'pitch': avg_pitch, 'yaw': avg_yaw}
        
        return orientation

    def calculate_raw_params(self, landmarks, w, h):
        """生の顔パラメータを計算"""
        def get_xy(idx):
            return landmarks[idx].x * w, landmarks[idx].y * h
        
        def get_xyz(idx):
            return landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z
        
        # 顔の向きを取得
        orientation = self.calculate_face_orientation(landmarks, w, h)
        roll = orientation['roll']
        
        # 口の中心を計算
        mouth_center_x = (landmarks[61].x + landmarks[291].x) * w / 2
        mouth_center_y = (landmarks[13].y + landmarks[14].y) * h / 2
        
        # 回転補正関数
        def get_rotated(idx):
            x, y = get_xy(idx)
            return self.get_rotated_point(mouth_center_x, mouth_center_y, x, y, -roll)
        
        # 基本的な口の寸法
        mouth_corners = [get_rotated(61), get_rotated(291)]
        mouth_vertical = [get_rotated(13), get_rotated(14)]
        
        mouth_width = abs(mouth_corners[1][0] - mouth_corners[0][0])
        mouth_height = abs(mouth_vertical[0][1] - mouth_vertical[1][1])
        
        # より詳細な口形状解析
        # 上唇の外側ライン
        upper_lip_outer = [get_rotated(i) for i in [61, 185, 40, 39, 37, 0, 267, 269, 270, 271, 272, 291]]
        # 上唇の内側ライン
        upper_lip_inner = [get_rotated(i) for i in [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]]
        # 下唇の外側ライン
        lower_lip_outer = [get_rotated(i) for i in [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 81, 82, 13, 312, 311, 310, 415, 308, 291]]
        # 下唇の内側ライン
        lower_lip_inner = [get_rotated(i) for i in [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]]
        
        # より精密な口の開き計算
        # 上唇の内側と下唇の内側の距離を計算
        upper_lip_inner_y = np.mean([p[1] for p in upper_lip_inner[2:6]])
        lower_lip_inner_y = np.mean([p[1] for p in lower_lip_inner[2:6]])
        precise_mouth_height = abs(upper_lip_inner_y - lower_lip_inner_y)
        
        # 上唇の厚さと下唇の厚さも計算
        upper_lip_thickness = abs(np.mean([p[1] for p in upper_lip_outer[3:8]]) - upper_lip_inner_y)
        lower_lip_thickness = abs(lower_lip_inner_y - np.mean([p[1] for p in lower_lip_outer[8:13]]))
        
        # 口角の引き具合をより正確に
        left_corner_x, left_corner_y = get_rotated(61)
        right_corner_x, right_corner_y = get_rotated(291)
        mouth_center_baseline_x = (left_corner_x + right_corner_x) / 2
        
        # 口角の相対位置を計算（口の中心からの距離）
        mouth_rest_width = mouth_width
        left_corner_distance = abs(left_corner_x - mouth_center_baseline_x)
        right_corner_distance = abs(right_corner_x - mouth_center_baseline_x)
        
        # 口角の上下の動きを計算
        mouth_center_y_baseline = (landmarks[13].y + landmarks[14].y) * h / 2
        left_corner_vertical_shift = (mouth_center_y_baseline - left_corner_y) / (mouth_height + 1e-6)
        right_corner_vertical_shift = (mouth_center_y_baseline - right_corner_y) / (mouth_height + 1e-6)
        
        # 口角の引きと下がりを分離
        left_corner_pull = max(left_corner_vertical_shift, 0)
        right_corner_pull = max(right_corner_vertical_shift, 0)
        left_corner_down = max(-left_corner_vertical_shift, 0)
        right_corner_down = max(-right_corner_vertical_shift, 0)
        
        # 口すぼめ（Pucker）の計算改善
        # 内側の口の幅と外側の口の幅の比率で計算
        inner_width = abs(get_rotated(78)[0] - get_rotated(308)[0])
        outer_width = mouth_width
        pucker_ratio = inner_width / (outer_width + 1e-6)
        
        # 唇の前後の突出も考慮（Z座標使用）
        lip_protrusion = np.mean([landmarks[12].z, landmarks[15].z, landmarks[16].z, landmarks[17].z])
        pucker_intensity = (0.7 - pucker_ratio) + abs(lip_protrusion) * 0.5
        
        # 顎の動き（改善版）
        chin_point = get_rotated(18)  # 顎先
        
        # 顎の左右の動き - デッドゾーンを設けて微小な振動を防ぐ
        jaw_displacement_x_raw = (chin_point[0] - mouth_center_x) / (mouth_width + 1e-6)
        # デッドゾーンを適用
        if abs(jaw_displacement_x_raw) < 0.05:  # 閾値以下はゼロに
            jaw_displacement_x_raw = 0
        
        # 顎の前後の動き
        jaw_forward_raw = landmarks[18].z * 30  # さらに穏やかに
        
        # 顎の開き - シンプルな計算に変更
        # 口の高さを直接使用（閾値は後でキャリブレーションで処理）
        jaw_open_factor = precise_mouth_height / 10.0  # 10で割って0-1範囲に正規化
        
        # 舌の位置（口の開きを考慮）
        # 口が開いているかチェック
        mouth_is_open = jaw_open_factor > 0.1  # JawOpenが一定以上の時のみ舌を検出
        
        if mouth_is_open:
            # 舌関連のランドマーク
            tongue_tip = get_rotated(17)  # 舌先
            lower_lip_center = get_rotated(14)  # 下唇中央
            upper_lip_center = get_rotated(13)  # 上唇中央
            
            # X軸（左右）の動き - より精密な計算
            tongue_x_displacement = (tongue_tip[0] - mouth_center_x) / (mouth_width * 0.4 + 1e-6)
            # より大きなデッドゾーンで安定化
            if abs(tongue_x_displacement) < 0.1:
                tongue_x_displacement = 0
            else:
                # デッドゾーンを引いて感度調整
                if tongue_x_displacement > 0:
                    tongue_x_displacement = max(tongue_x_displacement - 0.1, 0) * 1.5
                else:
                    tongue_x_displacement = min(tongue_x_displacement + 0.1, 0) * 1.5
            
            # 舌の突出度 - より簡単な計算
            # 舌先が下唇より下にあるかどうかで判定
            tongue_below_lip = max((tongue_tip[1] - lower_lip_center[1]) / (precise_mouth_height + 1e-6), 0)
            tongue_protrusion_raw = tongue_below_lip * 3  # 突出度を簡単化
            
            # 舌の上下動作 - より簡单な計算
            # 舌が上唇に近いかどうか
            tongue_up_distance = max((upper_lip_center[1] - tongue_tip[1]) / (precise_mouth_height + 1e-6), 0)
            tongue_up_raw = max(tongue_up_distance - 0.3, 0) * 5  # 闾値を下げて感度アップ
            
            # 舌が下唇から離れているかどうか
            tongue_down_distance = max((tongue_tip[1] - lower_lip_center[1]) / (precise_mouth_height + 1e-6), 0)
            tongue_down_raw = max(tongue_down_distance - 0.1, 0) * 4  # 闾値を低くして感度保持
        else:
            # 口が閉じている時は舌のパラメータを全てゼロに
            tongue_x_displacement = 0
            tongue_protrusion_raw = 0
            tongue_up_raw = 0
            tongue_down_raw = 0
        
        # 顎の動き（改善版）
        chin_point = get_rotated(18)  # 顎先
        
        # 顎の左右の動き - 相対変化量ベースの改良版
        # 顔の中心軸からの顎の偏移を計算
        face_center_x = (landmarks[234].x + landmarks[454].x) * w / 2  # 左右頬の中点
        jaw_offset_from_center = (chin_point[0] - face_center_x) / (mouth_width + 1e-6)
        
        # 顔の傾きを考慮した補正（Roll角度）
        if self.stable_face_orientation:
            # Roll角の補正：顔が傾いているときの顎位置補正
            roll_correction = math.sin(orientation['roll']) * 0.2
            jaw_offset_from_center -= roll_correction
        
        # 適応的なデッドゾーン（顔のサイズに比例）
        adaptive_deadzone = 0.08 + (1.0 / max(self.face_scale_factor, 0.5)) * 0.02
        
        # 変化量ベースの検出（前フレームとの差分も考慮）
        if not hasattr(self, 'prev_jaw_offset'):
            self.prev_jaw_offset = 0
        
        jaw_movement = abs(jaw_offset_from_center - self.prev_jaw_offset)
        
        # 静止状態では値をゼロに、動きがあるときのみ検出
        if abs(jaw_offset_from_center) < adaptive_deadzone and jaw_movement < 0.02:
            jaw_displacement_x_raw = 0
        else:
            # デッドゾーンを超えた部分のみを使用
            if jaw_offset_from_center > adaptive_deadzone:
                jaw_displacement_x_raw = (jaw_offset_from_center - adaptive_deadzone) * 1.2
            elif jaw_offset_from_center < -adaptive_deadzone:
                jaw_displacement_x_raw = (jaw_offset_from_center + adaptive_deadzone) * 1.2
            else:
                jaw_displacement_x_raw = 0
        
        self.prev_jaw_offset = jaw_offset_from_center
        
        # 顎の前後の動き
        jaw_forward_raw = landmarks[18].z * 30  # さらに穏やかに
        
        # 顎の開き - より簡単な計算
        jaw_open_normalized = precise_mouth_height / (mouth_width * 0.25 + 1e-6)
        jaw_open_factor = max(jaw_open_normalized - 1.2, 0)  # 闾値を設定
        
        # 頬の膨らみ（改善版）- より安定した検出
        left_corner_x, left_corner_y = get_rotated(61)
        right_corner_x, right_corner_y = get_rotated(291)
        left_cheek_x, left_cheek_y = get_rotated(234)
        right_cheek_x, right_cheek_y = get_rotated(454)
        
        # 頬の膨らみを面積ベースで計算（より正確）
        # 口角と頬の点、さらに顔の中心線を使って三角形の面積を計算
        face_center_x = (left_cheek_x + right_cheek_x) / 2
        
        # 左頬の膨らみ：口角、頬、顔中心で作る三角形の面積
        left_triangle_area = abs((left_corner_x - face_center_x) * (left_cheek_y - left_corner_y) - 
                               (left_cheek_x - face_center_x) * (left_corner_y - left_corner_y)) / 2
        
        # 右頬の膨らみ：口角、頬、顔中心で作る三角形の面積  
        right_triangle_area = abs((right_corner_x - face_center_x) * (right_cheek_y - right_corner_y) - 
                                (right_cheek_x - face_center_x) * (right_corner_y - right_corner_y)) / 2
        
        # 基準面積で正規化
        base_area = mouth_width * mouth_width * 0.1 * self.face_scale_factor * self.face_scale_factor
        left_cheek_expansion = left_triangle_area / (base_area + 1e-6)
        right_cheek_expansion = right_triangle_area / (base_area + 1e-6)
        
        # より大きな中立域を設定
        cheek_neutral_zone = 0.25
        left_cheek_expansion = max(left_cheek_expansion - cheek_neutral_zone, 0)
        right_cheek_expansion = max(right_cheek_expansion - cheek_neutral_zone, 0)
        
        # パラメータの計算と正規化（最終版）
        params = {
            "JawOpen": min(max(jaw_open_factor, 0), 1),  # キャリブレーションで基準値調整
            "JawRight": min(max(jaw_displacement_x_raw * 15, 0), 1),  # 感度を適切に調整
            "JawLeft": min(max(-jaw_displacement_x_raw * 15, 0), 1),  # 感度を適切に調整
            "JawForward": min(max(jaw_forward_raw, 0), 1),
            "MouthCornerPullRight": min(max(right_corner_pull * 3, 0), 1),
            "MouthCornerPullLeft": min(max(left_corner_pull * 3, 0), 1),
            "MouthPucker": min(max(pucker_intensity, 0), 1),
            "CheekPuffRight": min(max(right_cheek_expansion * 3, 0), 1),  # 中立域考慮済み
            "CheekPuffLeft": min(max(left_cheek_expansion * 3, 0), 1),   # 中立域考慮済み
            "TongueOut": min(max(tongue_protrusion_raw, 0), 1),
            "TongueUp": min(max(tongue_up_raw, 0), 1),
            "TongueDown": min(max(tongue_down_raw, 0), 1),
            "TongueRight": min(max(tongue_x_displacement * 3, 0), 1),  # 両方向とも確実に動作
            "TongueLeft": min(max(-tongue_x_displacement * 3, 0), 1)   # 両方向とも確実に動作
        }
        
        return params

    def smooth_params(self, params):
        """パラメータのスムージング"""
        smoothed = {}
        for key, val in params.items():
            self.smoothing_buffer[key].append(val)
            # 移動平均でスムージング
            smoothed[key] = np.mean(list(self.smoothing_buffer[key]))
        return smoothed

    def process_frame(self, frame):
        """フレームを処理してOSCデータを送信"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        results = self.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # キャリブレーション中
            if self.calibration_needed:
                calibrated = self.calibrate(landmarks, w, h)
                if calibrated:
                    cv2.putText(frame, "Calibration Complete!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    progress = int((self.calibration_frames / 30) * 100)
                    cv2.putText(frame, f"Calibrating... {progress}%", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                return frame
            
            # パラメータ計算
            raw_params = self.calculate_raw_params(landmarks, w, h)
            
            # 適応的基準値更新
            self.update_baseline_adaptively(raw_params)
            
            # キャリブレーション基準値から差分を計算
            calibrated_params = {}
            for key, val in raw_params.items():
                if key in self.baseline:
                    # 基準値からの変化を計算
                    if key in ["JawOpen", "TongueOut", "TongueUp", "TongueDown", "CheekPuffRight", "CheekPuffLeft"]:
                        # 基準値からの差分で計算し、マイナス値を防ぐ
                        diff = val - self.baseline[key]
                        # 個人差を考慮したスケーリング
                        if key.startswith("Cheek"):
                            diff *= self.face_scale_factor
                        calibrated_params[key] = max(diff, 0)
                    elif key == "JawForward":
                        # JawForwardも基準値からの差分で計算
                        calibrated_params[key] = max(val - self.baseline[key], 0)
                    else:
                        calibrated_params[key] = val
                else:
                    calibrated_params[key] = val
            
            # スムージング適用
            smoothed_params = self.smooth_params(calibrated_params)
            
            # OSC送信とUI表示
            y = 60
            for key, val in smoothed_params.items():
                self.osc_client.send_message(PARAMS[key], float(val))
                color = (0, 255, 0) if val > 0.1 else (100, 100, 100)
                cv2.putText(frame, f"{key}: {val:.3f}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 20
            
            # 口のランドマークを描画（デバッグ用）
            mouth_points = []
            for idx in [61, 291, 13, 14, 17]:  # 主要な口のポイント
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                mouth_points.append((x, y))
            
            # 口の輪郭線を描画
            if len(mouth_points) >= 4:
                cv2.line(frame, mouth_points[0], mouth_points[1], (255, 0, 0), 1)  # 口角線
                cv2.line(frame, mouth_points[2], mouth_points[3], (255, 0, 0), 1)  # 縦線
        
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

def main():
    print("Initializing facial tracker...")
    tracker = FacialTracker()
    
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # カメラ設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera opened successfully!")
    print("Press 'r' to recalibrate, 'c' to calibrate, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from camera")
            break
        
        # 水平反転（鏡像）
        frame = cv2.flip(frame, 1)
        
        # トラッキング処理
        frame = tracker.process_frame(frame)
        
        # キャリブレーションボタン用のテキストを追加
        cv2.putText(frame, "Press 'c' to calibrate (face neutral position)", (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'r' to reset calibration", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('VRChat Facial OSC Tracking - Fixed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # キャリブレーションリセット
            tracker.calibration_needed = True
            tracker.calibration_frames = 0
            tracker.baseline = {}
            print("Calibration reset...")
        elif key == ord('c'):
            # 手動キャリブレーション開始
            tracker.calibration_needed = True
            tracker.calibration_frames = 0
            tracker.baseline = {}
            print("Starting calibration... Keep face in neutral position for 2 seconds")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()