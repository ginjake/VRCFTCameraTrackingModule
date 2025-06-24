import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow警告を完全に非表示

import cv2
import mediapipe as mp
from pythonosc import udp_client
import numpy as np
import math
from collections import deque
from scipy.spatial import distance

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
    "TongueLeft": "/avatar/parameters/TongueLeft",
    # アイトラッキングパラメータ（パーフェクトシンク仕様）
    "EyesX": "/avatar/parameters/EyesX",
    "EyesY": "/avatar/parameters/EyesY",
    "LeftEyeLid": "/avatar/parameters/LeftEyeLid",
    "RightEyeLid": "/avatar/parameters/RightEyeLid",
    "EyesWiden": "/avatar/parameters/EyesWiden",
    "EyeSquintRight": "/avatar/parameters/EyeSquintRight",
    "EyeSquintLeft": "/avatar/parameters/EyeSquintLeft",
    "EyeWideLeft": "/avatar/parameters/EyeWideLeft",
    "EyeWideRight": "/avatar/parameters/EyeWideRight",
    # 眉毛パラメータ（パーフェクトシンク仕様）
    "BrowInnerUpLeft": "/avatar/parameters/BrowInnerUpLeft",
    "BrowInnerUpRight": "/avatar/parameters/BrowInnerUpRight",
    "BrowLowererLeft": "/avatar/parameters/BrowLowererLeft",
    "BrowLowererRight": "/avatar/parameters/BrowLowererRight",
    "BrowOuterUpLeft": "/avatar/parameters/BrowOuterUpLeft",
    "BrowOuterUpRight": "/avatar/parameters/BrowOuterUpRight",
    # 鼻パラメータ
    "NoseSneerLeft": "/avatar/parameters/NoseSneerLeft",
    "NoseSneerRight": "/avatar/parameters/NoseSneerRight"
}

class FacialTracker:
    def __init__(self):
        self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        
        # MediaPipe設定
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.01,  # 最低限界まで下げて超至近距離検出
            min_tracking_confidence=0.01,   # 最小限で部分的顔も追跡
            static_image_mode=False
        )
        
        # スムージング用のバッファ（適応的サイズ）
        self.smoothing_buffer = {}
        self.buffer_sizes = {
            'JawOpen': 2,  # 口の開閉は少しスムーズに
            'TongueOut': 1,  # 舌は瞬時反応
            'TongueUp': 1,
            'TongueDown': 1,
            'TongueRight': 1,
            'TongueLeft': 1,
            'MouthCornerPullRight': 3,  # 口角は少しスムージング
            'MouthCornerPullLeft': 3,
            'MouthPucker': 2,  # 口すぼめは少しスムージング
            'CheekPuffRight': 4,  # 頬膨らみは安定性重視
            'CheekPuffLeft': 4,
            'JawRight': 2,
            'JawLeft': 2,
            'JawForward': 2,
            # アイトラッキング用
            'LeftEyeLid': 2,
            'RightEyeLid': 2,
            'EyesX': 3,
            'EyesY': 3
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
        
        # アイトラッキング設定
        self.eye_tracking_enabled = True  # 瞬きのため初期状態でオンに変更
        
        # パラメータの基準値バッファ初期化
        self.baseline = {}  # 基準値辞書を初期化
        for param in PARAMS.keys():
            self.baseline_buffer[param] = deque(maxlen=60)  # 2秒間のバッファ
            # キャリブレーション用の基準値は空リストで初期化（顎の左右以外）
            if param not in ['JawRight', 'JawLeft']:
                self.baseline[param] = []  # キャリブレーション時にデータを収集するためのリスト
        
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
        
        # 3D姿勢推定用
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 鼻先 (landmark 1)
            (0.0, -330.0, -65.0),        # 顎 (landmark 18)
            (-225.0, 170.0, -135.0),     # 左目外角 (landmark 33)
            (225.0, 170.0, -135.0),      # 右目外角 (landmark 263)
            (-150.0, -150.0, -125.0),    # 左口角 (landmark 61)
            (150.0, -150.0, -125.0)      # 右口角 (landmark 291)
        ], dtype=np.float64)
        
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4,1), dtype=np.float64)
        
        # 光学フロー用
        self.prev_gray = None
        self.prev_cheek_points = None
        self.cheek_movement_history = deque(maxlen=10)

    def calibrate(self, landmarks, w, h, rgb):
        """改善されたキャリブレーションシステム"""
        if self.calibration_frames < 60:  # 2秒間で基準値を収集
            params = self.calculate_raw_params(landmarks, w, h, rgb)
            
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
                if key in self.baseline:  # 初期化済みのパラメータのみ処理
                    self.baseline[key].append(val)
            
            self.calibration_frames += 1
            return False
        else:
            # 平均値を基準として設定
            for key in self.baseline:
                if isinstance(self.baseline[key], list) and len(self.baseline[key]) > 0:
                    values = self.baseline[key]
                    # 外れ値を除去して平均を計算
                    sorted_values = sorted(values)
                    trim_count = len(sorted_values) // 10  # 上下10%をカット
                    if trim_count > 0:
                        trimmed_values = sorted_values[trim_count:-trim_count]
                    else:
                        trimmed_values = sorted_values
                    self.baseline[key] = np.mean(trimmed_values)
                # リストでない場合（初期値0.0）はそのまま維持
            
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
    
    def get_3d_pose(self, landmarks, w, h):
        """OpenCV solvePnPによる正確な3D姿勢推定"""
        if self.camera_matrix is None:
            # カメラ行列の初期化
            focal_length = max(w, h)
            center = (w/2, h/2)
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
        
        # 2Dランドマークから対応する点を抽出
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),    # 鼻先
            (landmarks[18].x * w, landmarks[18].y * h),   # 顎
            (landmarks[33].x * w, landmarks[33].y * h),   # 左目外角  
            (landmarks[263].x * w, landmarks[263].y * h), # 右目外角
            (landmarks[61].x * w, landmarks[61].y * h),   # 左口角
            (landmarks[291].x * w, landmarks[291].y * h)  # 右口角
        ], dtype=np.float64)
        
        # solvePnPで3D姿勢を推定
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, 
            self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if success:
            # 回転行列に変換
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # オイラー角を計算
            sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + 
                          rotation_matrix[1,0] * rotation_matrix[1,0])
            
            singular = sy < 1e-6
            if not singular:
                x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = 0
                
            return {
                'success': True,
                'rotation_matrix': rotation_matrix,
                'translation_vector': translation_vector,
                'euler_angles': {'roll': x, 'pitch': y, 'yaw': z}
            }
        
        return {'success': False}
    
    def detect_cheek_puff_optical_flow(self, gray_frame, landmarks, w, h):
        """シンプル頬膨らみ検出"""
        # 主要な頬ランドマーク（最も確実なポイント）
        left_cheek = landmarks[234]   # 左頬の中心
        right_cheek = landmarks[454]  # 右頬の中心
        face_center_x = (left_cheek.x + right_cheek.x) * w / 2
        
        # 現在の頬の位置
        left_x = left_cheek.x * w
        right_x = right_cheek.x * w
        current_width = abs(right_x - left_x)
        
        # 基準値設定
        if not hasattr(self, 'base_cheek_positions'):
            self.base_cheek_positions = {
                'left_x': left_x,
                'right_x': right_x,
                'width': current_width,
                'center_x': face_center_x
            }
        
        # 各頬の中心からの距離変化
        base_left_distance = abs(self.base_cheek_positions['left_x'] - self.base_cheek_positions['center_x'])
        base_right_distance = abs(self.base_cheek_positions['right_x'] - self.base_cheek_positions['center_x'])
        
        current_left_distance = abs(left_x - face_center_x)
        current_right_distance = abs(right_x - face_center_x)
        
        # 膨らみ検出（外向きの移動）
        left_expansion = (current_left_distance - base_left_distance) / (base_left_distance + 1e-6)
        right_expansion = (current_right_distance - base_right_distance) / (base_right_distance + 1e-6)
        
        # 高感度変換（0-1範囲）
        left_puff = max(0, min(1.0, left_expansion * 50.0))  # 高感度
        right_puff = max(0, min(1.0, right_expansion * 50.0))
        
        # デバッグ用に値を確認
        # print(f"CheekPuff Debug - Left: {left_expansion:.4f} -> {left_puff:.2f}, Right: {right_expansion:.4f} -> {right_puff:.2f}")
        
        return left_puff, right_puff
    
    def update_baseline_adaptively(self, params):
        """適応的な基準値更新"""
        import time
        current_time = time.time()
        
        # 動きが小さいときのみ基準値を更新（顎の左右は除外）
        movement_detected = False
        for key, val in params.items():
            if key in self.baseline and key not in ['JawRight', 'JawLeft']:
                # 基準値が数値の場合のみ比較（リストの場合はスキップ）
                if isinstance(self.baseline[key], (int, float)):
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
                        if key in self.baseline and isinstance(self.baseline[key], (int, float)):
                            self.baseline[key] = self.baseline[key] * 0.9 + new_baseline * 0.1
                        else:
                            self.baseline[key] = new_baseline  # 初回は直接設定
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

    def calculate_raw_params(self, landmarks, w, h, rgb):
        """生の顔パラメータを計算（距離正規化対応）"""
        def get_xy(idx):
            return landmarks[idx].x * w, landmarks[idx].y * h
        
        def get_xyz(idx):
            return landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z
        
        # 顔のスケール計算（距離正規化の基準）
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        face_width = abs(left_cheek.x - right_cheek.x) * w
        face_height = abs(landmarks[10].y - landmarks[152].y) * h  # 額から顎
        face_scale = (face_width + face_height) / 2  # 平均スケール
        
        # 正規化係数（超至近距離10cm用に基準顔サイズを1000pxに極限拡大）
        scale_factor = 1000.0 / (face_scale + 1e-6)
        
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
        
        # 高精度口角検出システム
        # 基準状態からの口幅変化を検出
        if not hasattr(self, 'base_mouth_width'):
            self.base_mouth_width = mouth_width
        
        # 口幅の変化率を計算
        width_change_ratio = mouth_width / self.base_mouth_width
        
        # 口角の水平移動を詳細に計算
        left_corner_horizontal_shift = (left_corner_distance - mouth_rest_width/2) / (mouth_rest_width/2 + 1e-6)
        right_corner_horizontal_shift = (right_corner_distance - mouth_rest_width/2) / (mouth_rest_width/2 + 1e-6)
        
        # 口角の引きを強化（感度向上）
        left_corner_pull = max(0, min(1, left_corner_vertical_shift * 3.0 + left_corner_horizontal_shift * 2.0))
        right_corner_pull = max(0, min(1, right_corner_vertical_shift * 3.0 + right_corner_horizontal_shift * 2.0))
        
        # Eの口形状検出（横に引く動き）
        mouth_stretch = max(0, min(1, (width_change_ratio - 1.0) * 5.0))
        
        # 口角下がり検出も強化
        left_corner_down = max(0, min(1, -left_corner_vertical_shift * 2.0))
        right_corner_down = max(0, min(1, -right_corner_vertical_shift * 2.0))
        
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
        
        # 口の開閉判定を統一し、より厳密に設定
        # 口の縦横比で計算（距離に依存しない）
        mouth_aspect_ratio = precise_mouth_height / (mouth_width + 1e-6)
        
        # 口の開き判定を非常に厳密に（舌や顎の動き用）
        mouth_is_open_strict = mouth_aspect_ratio > 0.08  # 非常に厳しい閾値
        mouth_is_open_normal = mouth_aspect_ratio > 0.05  # 通常の閾値
        
        # JawOpenパラメータ用の計算（こちらは緊急でなくてもOK）
        jaw_open_factor = max(mouth_aspect_ratio - 0.03, 0) * 8.0
        
        # 舌検出：口が大きく開いている時のみ（誤検出防止）
        mouth_very_open = jaw_open_factor > 0.4  # より厳しい基準
        
        if mouth_very_open:
            # 舌関連のランドマーク
            tongue_tip = get_rotated(17)  # 舌先（改良版）
            lower_lip_center = get_rotated(14)  # 下唇中央
            upper_lip_center = get_rotated(13)  # 上唇中央
            
            # 口の中心線からの舌の左右偏移
            tongue_x_offset = (tongue_tip[0] - mouth_center_x) / (mouth_width + 1e-6)
            
            # 大きなデッドゾーンで微細な動きを無視
            tongue_deadzone = 0.15  # 15%のデッドゾーン
            if abs(tongue_x_offset) > tongue_deadzone:
                tongue_x_displacement = (tongue_x_offset - math.copysign(tongue_deadzone, tongue_x_offset)) * 3.0
            else:
                tongue_x_displacement = 0
            
            # 舌の突出度 - より簡単な計算
            # 舌先が下唇より下にあるかどうかで判定
            lip_to_tongue_distance = (tongue_tip[1] - lower_lip_center[1]) / (mouth_height + 1e-6)
            tongue_protrusion_raw = max(0, (lip_to_tongue_distance - 0.2) * 5.0)  # 20%以上の突出で検出
            
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
        
        # === 3D姿勢推定による高精度顎検出 ===
        pose_3d = self.get_3d_pose(landmarks, w, h)
        
        if pose_3d['success']:
            # 単純化：顎の中心からの横方向偏移を直接計算
            chin_point = landmarks[18]
            face_center_x = (landmarks[234].x + landmarks[454].x) * w / 2
            face_width = abs(landmarks[234].x - landmarks[454].x) * w
            
            # 顎の横方向偏移（正面時は0になるように）
            jaw_offset = (chin_point.x * w - face_center_x) / (face_width + 1e-6)
            
            # より大きなデッドゾーンで正面時の振動を防ぐ
            jaw_deadzone = 0.03  # 3%のデッドゾーン
            
            if abs(jaw_offset) > jaw_deadzone:
                jaw_displacement_x_raw = (jaw_offset - math.copysign(jaw_deadzone, jaw_offset)) * 5.0
            else:
                jaw_displacement_x_raw = 0
            
            # 顎の前後の動き
            jaw_forward_raw = max(0, landmarks[18].z * 10)
            
        else:
            # 3D姿勢推定が失敗した場合のフォールバック
            chin_point = get_rotated(18)
            face_center_x = (landmarks[234].x + landmarks[454].x) * w / 2
            jaw_offset_from_center = (chin_point[0] - face_center_x) / (mouth_width + 1e-6)
            jaw_displacement_x_raw = jaw_offset_from_center * 3.0  # 低い感度
            jaw_forward_raw = landmarks[18].z * 15
        
        # （jaw_open_factorは上で既に計算済み）
        
        # === 光学フローによる高精度頬膨らみ検出 ===
        # グレースケール変換
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # 光学フローベースの頬膨らみ検出
        left_cheek_expansion, right_cheek_expansion = self.detect_cheek_puff_optical_flow(gray, landmarks, w, h)
        
        # 従来手法との組み合わせ（フォールバック）
        if left_cheek_expansion == 0 and right_cheek_expansion == 0:
            # 光学フローで検出できない場合は従来手法
            left_corner_x, left_corner_y = get_rotated(61)
            right_corner_x, right_corner_y = get_rotated(291)
            left_cheek_x, left_cheek_y = get_rotated(234)
            right_cheek_x, right_cheek_y = get_rotated(454)
            
            face_center_x = (left_cheek_x + right_cheek_x) / 2
            face_width = abs(landmarks[234].x - landmarks[454].x) * w
            
            left_cheek_protrusion = (face_center_x - left_cheek_x) / (face_width/2 + 1e-6)
            right_cheek_protrusion = (right_cheek_x - face_center_x) / (face_width/2 + 1e-6)
            
            # 簡易的な膨らみ検出
            left_cheek_expansion = max(0, (left_cheek_protrusion - 0.9) * 8.0)
            right_cheek_expansion = max(0, (right_cheek_protrusion - 0.9) * 8.0)
        
        # パラメータの計算と正規化（最終版）
        params = {
            "JawOpen": min(max(jaw_open_factor, 0), 1),  # キャリブレーションで基準値調整
            "JawRight": min(max(jaw_displacement_x_raw, 0), 1),  # 復活
            "JawLeft": min(max(-jaw_displacement_x_raw, 0), 1),  # 復活
            "JawForward": min(max(jaw_forward_raw, 0), 1),
            "MouthCornerPullRight": min(max(right_corner_pull * 3, 0), 1),
            "MouthCornerPullLeft": min(max(left_corner_pull * 3, 0), 1),
            "MouthPucker": min(max(pucker_intensity, 0), 1),
            "CheekPuffRight": min(max(right_cheek_expansion, 0), 1),  # 復活（倍率削除）
            "CheekPuffLeft": min(max(left_cheek_expansion, 0), 1),   # 復活（倍率削除）
            "TongueOut": min(max(tongue_protrusion_raw, 0), 1) if mouth_very_open else 0,  # 改良
            "TongueUp": min(max(tongue_up_raw, 0), 1) if mouth_very_open else 0,           # 改良
            "TongueDown": min(max(tongue_down_raw, 0), 1) if mouth_very_open else 0,       # 改良
            "TongueRight": min(max(tongue_x_displacement, 0), 1) if mouth_very_open else 0,  # 改良（倍率削除）
            "TongueLeft": min(max(-tongue_x_displacement, 0), 1) if mouth_very_open else 0   # 改良（倍率削除）
        }
        
        # スケール情報を保存（デバッグ用）
        self.last_scale_info = {
            'factor': scale_factor,
            'scale': face_scale
        }
        
        # デバッグ用の口と頬のパラメータを保存
        self.last_mouth_params = {
            'jaw_open': params.get('JawOpen', 0),
            'mouth_stretch': mouth_stretch if 'mouth_stretch' in locals() else 0,
            'cheek_left': params.get('CheekPuffLeft', 0),
            'cheek_right': params.get('CheekPuffRight', 0),
            'corner_left': params.get('MouthCornerPullLeft', 0),
            'corner_right': params.get('MouthCornerPullRight', 0),
            'pucker': params.get('MouthPucker', 0),
            'mouth_ratio': mouth_aspect_ratio,
            'is_open_strict': mouth_is_open_strict,
            'is_open_normal': mouth_is_open_normal,
            'jaw_right': params.get('JawRight', 0),
            'jaw_left': params.get('JawLeft', 0)
        }
        
        return params

    def calculate_nose_params(self, landmarks, w, h):
        """鼻の動きを検出（距離に依存しない相対計算）"""
        try:
            # 鼻翼のランドマーク
            left_nostril_wing = landmarks[219]    # 左鼻翼
            right_nostril_wing = landmarks[439]   # 右鼻翼  
            nose_center = landmarks[2]            # 鼻中央
            
            # 顔の基準サイズ（左右頬の距離）を使って正規化
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            face_width = abs(left_cheek.x - right_cheek.x)
            
            # 鼻翼の中心線からの相対距離（顔幅で正規化）
            left_nostril_ratio = abs(left_nostril_wing.x - nose_center.x) / (face_width + 1e-6)
            right_nostril_ratio = abs(right_nostril_wing.x - nose_center.x) / (face_width + 1e-6)
            
            # 基準比率を設定（初回のみ）
            if not hasattr(self, 'base_nostril_ratio'):
                self.base_nostril_ratio = {
                    'left': left_nostril_ratio,
                    'right': right_nostril_ratio
                }
            
            # 比率の変化を計算（距離に依存しない）
            left_ratio_change = left_nostril_ratio - self.base_nostril_ratio['left']
            right_ratio_change = right_nostril_ratio - self.base_nostril_ratio['right']
            
            # 正規化と感度調整（相対値なので距離無関係）
            left_nose_sneer = max(0, left_ratio_change / 0.02)  # 2%の変化で1.0
            right_nose_sneer = max(0, right_ratio_change / 0.02)
            
            # 感度調整
            left_nose_sneer = min(1, left_nose_sneer * 3)
            right_nose_sneer = min(1, right_nose_sneer * 3)
            
            nose_params = {
                "NoseSneerLeft": max(0, min(1, left_nose_sneer)),
                "NoseSneerRight": max(0, min(1, right_nose_sneer))
            }
            
            # デバッグ情報を保存（描画用）
            self.nose_debug_info = {
                'left_ratio': left_nostril_ratio,
                'right_ratio': right_nostril_ratio,
                'left_base': self.base_nostril_ratio['left'],
                'right_base': self.base_nostril_ratio['right'],
                'left_change': left_ratio_change,
                'right_change': right_ratio_change,
                'face_width': face_width
            }
            
            return nose_params
            
        except (IndexError, AttributeError):
            return {"NoseSneerLeft": 0, "NoseSneerRight": 0}

    def calculate_eye_params(self, landmarks, w, h, face_scale):
        """パーフェクトシンク仕様のアイトラッキング・眉毛パラメータを計算"""
        if not self.eye_tracking_enabled:
            return {}
        
        # スケール係数を計算
        scale_factor = 200.0 / (face_scale + 1e-6)
        
        # 目のランドマーク（MediaPipe 468点）
        # 左目: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        # 右目: 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
        # 眉毛: 70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 285, 295, 296, 334, 293, 300, 276, 283, 282, 295
        
        # 左目の詳細な測定点（より正確なランドマーク使用）
        left_eye_inner = landmarks[133]  # 左目内角
        left_eye_outer = landmarks[33]   # 左目外角
        left_eye_top = landmarks[159]    # 左目上端
        left_eye_bottom = landmarks[145] # 左目下端
        # 追加：より正確な上下端点
        left_eye_top_center = landmarks[158]    # 左目上端中央
        left_eye_bottom_center = landmarks[153] # 左目下端中央
        
        left_eye_center_x = (left_eye_inner.x + left_eye_outer.x) * w / 2
        left_eye_center_y = (left_eye_top.y + left_eye_bottom.y) * h / 2
        # より正確な高さ測定（中央部分を使用）
        left_eye_height = abs(left_eye_top_center.y - left_eye_bottom_center.y) * h
        left_eye_width = abs(left_eye_outer.x - left_eye_inner.x) * w
        
        # 右目の詳細な測定点（より正確なランドマーク使用）
        right_eye_inner = landmarks[362]  # 右目内角
        right_eye_outer = landmarks[263]  # 右目外角
        right_eye_top = landmarks[386]    # 右目上端
        right_eye_bottom = landmarks[374] # 右目下端
        # 追加：より正確な上下端点
        right_eye_top_center = landmarks[385]   # 右目上端中央
        right_eye_bottom_center = landmarks[380] # 右目下端中央
        
        right_eye_center_x = (right_eye_inner.x + right_eye_outer.x) * w / 2
        right_eye_center_y = (right_eye_top.y + right_eye_bottom.y) * h / 2
        # より正確な高さ測定（中央部分を使用）
        right_eye_height = abs(right_eye_top_center.y - right_eye_bottom_center.y) * h
        right_eye_width = abs(right_eye_outer.x - right_eye_inner.x) * w
        
        # 顔の基準点
        face_center_x = landmarks[1].x * w
        face_center_y = landmarks[1].y * h
        
        # 目の視線計算（パーフェクトシンク仕様に近い）
        # 瞳孔位置の推定（アイリス追跡の代替として目の中心位置を使用）
        left_gaze_x = (left_eye_center_x - face_center_x) / (left_eye_width * 2)
        left_gaze_y = -(left_eye_center_y - face_center_y) / (left_eye_height * 2)
        right_gaze_x = (right_eye_center_x - face_center_x) / (right_eye_width * 2)
        right_gaze_y = -(right_eye_center_y - face_center_y) / (right_eye_height * 2)
        
        # 平均視線（EyesX, EyesY）
        eyes_x = (left_gaze_x + right_gaze_x) / 2
        eyes_y = (left_gaze_y + right_gaze_y) / 2
        
        # 距離に依存しない目の開閉度検出（相対比ベース）
        
        # 目の縦横比で開閉度を判定（距離に依存しない）
        left_aspect_ratio = left_eye_height / (left_eye_width + 1e-6)
        right_aspect_ratio = right_eye_height / (right_eye_width + 1e-6)
        
        # シンプルな目の開閉検出
        # 基準値設定（左右完全独立）
        if not hasattr(self, 'eye_open_baseline'):
            self.eye_open_baseline = {'left': left_aspect_ratio, 'right': right_aspect_ratio}
            self.eye_closed_baseline = {'left': 0.05, 'right': 0.05}  # 完全に閉じた時の縦横比
        
        # 正常開眼時のみ基準値を更新
        if left_aspect_ratio > 0.15:  # 左目が開いている時
            self.eye_open_baseline['left'] = self.eye_open_baseline['left'] * 0.95 + left_aspect_ratio * 0.05
        if right_aspect_ratio > 0.15:  # 右目が開いている時
            self.eye_open_baseline['right'] = self.eye_open_baseline['right'] * 0.95 + right_aspect_ratio * 0.05
        
        # 左右完全独立な開閉度計算
        def calculate_eyelid_simple(aspect_ratio, open_baseline, closed_baseline):
            """シンプルな目の開閉度計算"""
            if aspect_ratio >= open_baseline:
                return 0.0  # 完全に開いている
            elif aspect_ratio <= closed_baseline:
                return 1.0  # 完全に閉じている
            else:
                # 線形補間
                ratio = (open_baseline - aspect_ratio) / (open_baseline - closed_baseline + 1e-6)
                return max(0.0, min(1.0, ratio))
        
        left_eyelid = calculate_eyelid_simple(left_aspect_ratio, 
                                             self.eye_open_baseline['left'], 
                                             self.eye_closed_baseline['left'])
        right_eyelid = calculate_eyelid_simple(right_aspect_ratio, 
                                              self.eye_open_baseline['right'], 
                                              self.eye_closed_baseline['right'])
        
        # デバッグ出力
        # print(f"Eye Debug - Left: ratio={left_aspect_ratio:.3f}, baseline={self.eye_open_baseline['left']:.3f}, eyelid={left_eyelid:.2f}")
        # print(f"Eye Debug - Right: ratio={right_aspect_ratio:.3f}, baseline={self.eye_open_baseline['right']:.3f}, eyelid={right_eyelid:.2f}")
        
        # デバッグ用の目のパラメータを保存
        self.last_eye_params = {
            'left_eyelid': left_eyelid,
            'right_eyelid': right_eyelid,
            'left_ratio': left_aspect_ratio,
            'right_ratio': right_aspect_ratio,
            'left_width': left_eye_width,
            'right_width': right_eye_width,
            'left_height': left_eye_height,
            'right_height': right_eye_height
        }
        
        # 目の細め・見開き検出（縦横比ベース）
        left_squint = max(0, min(1, (self.eye_open_baseline['left'] * 0.7 - left_aspect_ratio) * 10.0))
        right_squint = max(0, min(1, (self.eye_open_baseline['right'] * 0.7 - right_aspect_ratio) * 10.0))
        
        # 目見開き検出
        left_wide = max(0, min(1, (left_aspect_ratio - self.eye_open_baseline['left'] * 1.2) * 8.0))
        right_wide = max(0, min(1, (right_aspect_ratio - self.eye_open_baseline['right'] * 1.2) * 8.0))
        
        # 全体の目を見開く
        eyes_widen = (left_wide + right_wide) / 2
        
        # 眉毛のランドマーク
        # 左眉毛: 70(外), 107(中), 55(内)
        # 右眉毛: 285(外), 296(中), 334(内)
        left_brow_inner = landmarks[55]   # 左眉内側
        left_brow_middle = landmarks[107] # 左眉中央
        left_brow_outer = landmarks[70]   # 左眉外側
        right_brow_inner = landmarks[285] # 右眉内側
        right_brow_middle = landmarks[296]# 右眉中央
        right_brow_outer = landmarks[334] # 右眉外側
        
        # 眉毛の基準位置（目との相対位置）
        left_brow_inner_height = (left_eye_top.y - left_brow_inner.y) * h
        left_brow_outer_height = (left_eye_top.y - left_brow_outer.y) * h
        right_brow_inner_height = (right_eye_top.y - right_brow_inner.y) * h
        right_brow_outer_height = (right_eye_top.y - right_brow_outer.y) * h
        
        # 基準眉毛高さを設定
        if not hasattr(self, 'base_brow_height'):
            self.base_brow_height = {
                'left_inner': left_brow_inner_height,
                'left_outer': left_brow_outer_height,
                'right_inner': right_brow_inner_height,
                'right_outer': right_brow_outer_height
            }
        
        # 眉毛の動きを計算（距離正規化対応）
        brow_scale = 10 * scale_factor  # スケール調整された除数
        brow_inner_up_left = max(0, (left_brow_inner_height - self.base_brow_height['left_inner']) / brow_scale)
        brow_inner_up_right = max(0, (right_brow_inner_height - self.base_brow_height['right_inner']) / brow_scale)
        brow_outer_up_left = max(0, (left_brow_outer_height - self.base_brow_height['left_outer']) / brow_scale)
        brow_outer_up_right = max(0, (right_brow_outer_height - self.base_brow_height['right_outer']) / brow_scale)
        
        # 眉をひそめる（眉毛が下がる）- 距離正規化対応
        brow_down_scale = 8 * scale_factor  # スケール調整された除数
        brow_lowerer_left = max(0, (self.base_brow_height['left_inner'] - left_brow_inner_height) / brow_down_scale)
        brow_lowerer_right = max(0, (self.base_brow_height['right_inner'] - right_brow_inner_height) / brow_down_scale)
        
        eye_params = {
            "EyesX": max(-1, min(1, eyes_x)),
            "EyesY": max(-1, min(1, eyes_y)),
            "LeftEyeLid": max(0, min(1, left_eyelid)),
            "RightEyeLid": max(0, min(1, right_eyelid)),
            "EyesWiden": max(0, min(1, eyes_widen)),
            "EyeSquintLeft": max(0, min(1, left_squint)),
            "EyeSquintRight": max(0, min(1, right_squint)),
            "EyeWideLeft": max(0, min(1, left_wide)),
            "EyeWideRight": max(0, min(1, right_wide)),
            # 眉毛パラメータ
            "BrowInnerUpLeft": max(0, min(1, brow_inner_up_left)),
            "BrowInnerUpRight": max(0, min(1, brow_inner_up_right)),
            "BrowLowererLeft": max(0, min(1, brow_lowerer_left)),
            "BrowLowererRight": max(0, min(1, brow_lowerer_right)),
            "BrowOuterUpLeft": max(0, min(1, brow_outer_up_left)),
            "BrowOuterUpRight": max(0, min(1, brow_outer_up_right))
        }
        
        return eye_params

    def smooth_params(self, params):
        """パラメータのスムージング"""
        smoothed = {}
        for key, val in params.items():
            # バッファが存在しない場合は初期化
            if key not in self.smoothing_buffer:
                buffer_size = self.buffer_sizes.get(key, 5)
                self.smoothing_buffer[key] = deque(maxlen=buffer_size)
            
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
                calibrated = self.calibrate(landmarks, w, h, rgb)
                if calibrated:
                    cv2.putText(frame, "Calibration Complete!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    progress = int((self.calibration_frames / 30) * 100)
                    cv2.putText(frame, f"Calibrating... {progress}%", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                return frame
            
            # 顔スケール計算（全体で共有）
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            face_width = abs(left_cheek.x - right_cheek.x) * w
            face_height = abs(landmarks[10].y - landmarks[152].y) * h
            face_scale = (face_width + face_height) / 2
            
            # パラメータ計算
            raw_params = self.calculate_raw_params(landmarks, w, h, rgb)
            
            # 鼻パラメータを追加
            nose_params = self.calculate_nose_params(landmarks, w, h)
            raw_params.update(nose_params)
            
            # アイトラッキングパラメータを追加（スケール情報を渡す）
            eye_params = self.calculate_eye_params(landmarks, w, h, face_scale)
            raw_params.update(eye_params)
            
            # 適応的基準値更新
            self.update_baseline_adaptively(raw_params)
            
            # キャリブレーション基準値から差分を計算
            calibrated_params = {}
            for key, val in raw_params.items():
                if key in self.baseline and isinstance(self.baseline[key], (int, float)):
                    # 基準値からの変化を計算（数値の基準値がある場合のみ）
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
                    # 基準値がない場合や顎の左右は生の値をそのまま使用
                    calibrated_params[key] = val
            
            # スムージング適用
            smoothed_params = self.smooth_params(calibrated_params)
            
            # OSC送信とUI表示 - 2列に分けて表示
            h, w, _ = frame.shape
            left_col_x = 10
            right_col_x = w // 2 + 10
            left_y = 60
            right_y = 60
            
            # パラメータを2列に分けて表示
            param_count = 0
            # 無効化されたパラメータリスト（全て有効にするため空に）
            disabled_params = set()
            
            for key, val in smoothed_params.items():
                if key in PARAMS and key not in disabled_params:  # 有効なパラメータのみ送信
                    self.osc_client.send_message(PARAMS[key], float(val))
                    color = (0, 255, 0) if abs(val) > 0.1 else (100, 100, 100)
                    
                    # 左列と右列に交互に表示（文字サイズ拡大）
                    if param_count % 2 == 0:
                        cv2.putText(frame, f"{key}: {val:.3f}", (left_col_x, left_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        left_y += 25
                    else:
                        cv2.putText(frame, f"{key}: {val:.3f}", (right_col_x, right_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        right_y += 25
                    param_count += 1
            
            # アイトラッキング状態表示（上部中央）
            eye_status = "ON" if self.eye_tracking_enabled else "OFF"
            eye_color = (0, 255, 0) if self.eye_tracking_enabled else (0, 0, 255)
            cv2.putText(frame, f"Eye Tracking: {eye_status}", (w//2 - 120, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, eye_color, 2)
            
            # 口のランドマークを描画（デバッグ用）
            mouth_points = []
            for idx in [61, 291, 13, 14, 17]:  # 主要な口のポイント
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                mouth_points.append((x, y))
            
            # 鼻のランドマークを描画（デバッグ用）
            nose_points = [219, 439, 2]  # 左鼻翼、右鼻翼、鼻中央
            for idx in nose_points:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                color = (255, 0, 255) if idx == 2 else (0, 255, 0)  # 中央は紫、翼は緑
                cv2.circle(frame, (x, y), 3, color, -1)
            
            # 鼻のデバッグ情報を表示
            if hasattr(self, 'nose_debug_info'):
                debug_info = self.nose_debug_info
                debug_y = h - 150
                cv2.putText(frame, f"Nose Debug:", (10, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"L: {debug_info['left_change']:.3f} ({debug_info['left_ratio']:.3f})", (10, debug_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"R: {debug_info['right_change']:.3f} ({debug_info['right_ratio']:.3f})", (10, debug_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"Face W: {debug_info['face_width']:.3f}", (10, debug_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # 3D姿勢情報を表示
            pose_3d = self.get_3d_pose(landmarks, w, h)
            if pose_3d['success']:
                euler = pose_3d['euler_angles']
                cv2.putText(frame, f"3D Pose: Y:{math.degrees(euler['yaw']):.1f}° P:{math.degrees(euler['pitch']):.1f}° R:{math.degrees(euler['roll']):.1f}°", 
                           (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, "3D POSE ACTIVE", (w - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "3D POSE FAILED", (w - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 光学フロー状態表示
            if hasattr(self, 'prev_gray') and self.prev_gray is not None:
                cv2.putText(frame, "OPTICAL FLOW ACTIVE", (w - 200, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "OPTICAL FLOW INIT", (w - 200, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 全体的なスケール情報を表示
            if hasattr(self, 'last_scale_info'):
                scale_info = self.last_scale_info
                cv2.putText(frame, f"Scale Factor: {scale_info['factor']:.2f}", (w - 200, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Face Scale: {scale_info['scale']:.1f}px", (w - 200, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 重要パラメータを青い文字で表示（デバッグ用）
            blue_color = (255, 0, 0)  # BGRなので青は(255,0,0)
            
            if hasattr(self, 'last_eye_params'):
                eye_params = self.last_eye_params
                cv2.putText(frame, f"EYE DEBUG:", (10, h - 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue_color, 2)
                cv2.putText(frame, f"Left Eye: Ratio={eye_params['left_ratio']:.3f} Lid={eye_params['left_eyelid']:.3f}", (10, h - 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Right Eye: Ratio={eye_params['right_ratio']:.3f} Lid={eye_params['right_eyelid']:.3f}", (10, h - 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Ratios: L={eye_params.get('left_ratio', 0):.3f} R={eye_params.get('right_ratio', 0):.3f}", (10, h - 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"H/W: L={eye_params.get('left_height', 0):.1f}/{eye_params.get('left_width', 0):.1f} R={eye_params.get('right_height', 0):.1f}/{eye_params.get('right_width', 0):.1f}", (10, h - 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, blue_color, 1)
                cv2.putText(frame, f"Eye Tracking: {'ON' if self.eye_tracking_enabled else 'OFF (Press A)'}", (10, h - 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
            
            # 口と頬のパラメータも青い文字で表示
            if hasattr(self, 'last_mouth_params'):
                mouth_params = self.last_mouth_params
                cv2.putText(frame, f"MOUTH & JAW DEBUG:", (300, h - 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue_color, 2)
                cv2.putText(frame, f"Mouth Ratio: {mouth_params.get('mouth_ratio', 0):.3f} Open: N={mouth_params.get('is_open_normal', False)} S={mouth_params.get('is_open_strict', False)}", (300, h - 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, blue_color, 1)
                cv2.putText(frame, f"Jaw Open: {mouth_params['jaw_open']:.3f} L/R: {mouth_params.get('jaw_left', 0):.3f}/{mouth_params.get('jaw_right', 0):.3f}", (300, h - 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Cheek L/R: {mouth_params['cheek_left']:.3f}/{mouth_params['cheek_right']:.3f}", (300, h - 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Corner Pull L/R: {mouth_params['corner_left']:.3f}/{mouth_params['corner_right']:.3f}", (300, h - 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Mouth Stretch: {mouth_params['mouth_stretch']:.3f}", (300, h - 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
            
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
    
    # カメラ設定 - より大きなプレビュー + VR近距離用オートフォーカス
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # VR近距離撮影用のフォーカス設定
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # オートフォーカス有効
    cap.set(cv2.CAP_PROP_FOCUS, 0)      # 最小フォーカス距離に設定
    
    print("Camera opened successfully!")
    print("Press 'r' to recalibrate, 'c' to calibrate, 'a' to toggle eye tracking, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from camera")
            break
        
        # 水平反転（鏡像）
        frame = cv2.flip(frame, 1)
        
        # トラッキング処理
        frame = tracker.process_frame(frame)
        
        # コントロール用のテキストを追加（右下に配置して重複回避）
        h, w, _ = frame.shape
        help_x = w - 400
        cv2.putText(frame, "Controls:", (help_x, h - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "'c': Calibrate | 'r': Reset | 'a': Eye tracking | 'q': Quit", (help_x, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
        elif key == ord('a'):
            # アイトラッキング切り替え
            tracker.eye_tracking_enabled = not tracker.eye_tracking_enabled
            status = "enabled" if tracker.eye_tracking_enabled else "disabled"
            print(f"Eye tracking {status}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()