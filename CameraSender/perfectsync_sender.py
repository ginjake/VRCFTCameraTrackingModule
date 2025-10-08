import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow警告を完全に非表示

import cv2
import mediapipe as mp
from pythonosc import udp_client
import numpy as np
import math
from collections import deque
from scipy.spatial import distance
import tkinter as tk
from tkinter import ttk
import threading
import logging
from PIL import Image, ImageDraw, ImageFont
import signal
import sys
import atexit

# Varjo競合対策のためのカメラ列挙（カメラを開かずにデバイスリストを取得）
try:
    from pygrabber.dshow_graph import FilterGraph
    PYGRABBER_AVAILABLE = True
except ImportError:
    PYGRABBER_AVAILABLE = False
    print("警告: pygrabberが利用できません。Varjo競合回避機能が制限されます。")

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
    def __init__(self, dual_camera=False):
        self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        self.dual_camera = dual_camera
        
        # GUI用のボタン領域
        self.buttons = {
            'calibrate': {'x': 10, 'y': 10, 'width': 120, 'height': 40, 'text': 'Calibrate'},
            'reset': {'x': 140, 'y': 10, 'width': 100, 'height': 40, 'text': 'Reset'},
            'switch_camera': {'x': 250, 'y': 10, 'width': 140, 'height': 40, 'text': 'Switch Camera'}
        }

        # 日本語フォント設定
        try:
            # Windowsの標準日本語フォント
            self.font = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 20)
            self.font_large = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 30)
            self.font_small = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 16)
        except:
            # フォールバック：デフォルトフォント
            self.font = ImageFont.load_default()
            self.font_large = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
        
        # MediaPipe設定（VRゴーグル対応・口周辺特化）
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,   # 負荷軽減のため高く設定
            min_tracking_confidence=0.5,    # 負荷軽減のため高く設定
            static_image_mode=False,
            refine_landmarks=False          # 負荷軽減のため無効化（468点→68点）
        )
        
        # スムージング用のバッファ（カクカク防止・なめらかさ重視）
        self.smoothing_buffer = {}
        self.buffer_sizes = {
            'JawOpen': 3,  # なめらかさとレスポンスのバランス
            'JawRight': 3,
            'JawLeft': 3,
            'MouthCornerPullRight': 3,
            'MouthCornerPullLeft': 3,
            'MouthPucker': 3
        }
        for param in self.buffer_sizes.keys():
            self.smoothing_buffer[param] = deque(maxlen=self.buffer_sizes[param])
        
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
        
        # 2台カメラ用の追加設定
        if self.dual_camera:
            self.prev_gray_secondary = None
            self.secondary_camera_offset = 10  # 10cm間隔
            self.stereo_depth_enabled = True
            
            # ステレオ視用の設定
            self.stereo_matcher = cv2.StereoBM_create(numDisparities=64, blockSize=15)
            
            # 頬・舌専用の高精度検出用バッファ
            self.cheek_stereo_buffer = deque(maxlen=5)
            self.tongue_stereo_buffer = deque(maxlen=3)
            
            # 3D座標計算用
            self.camera_distance = 0.1  # 10cm
            self.focal_length_mm = 2.8  # 一般的なウェブカメラ
            
        # VRゴーグル対応（目を使わない設定）
        self.vr_mode = True  # VRゴーグル着用時は目の検出を無効化
        self.focus_on_lower_face = True  # 口元・頬・舌に特化

    def put_japanese_text(self, img, text, position, font, color=(255, 255, 255)):
        """OpenCVの画像に日本語テキストを描画"""
        # OpenCV形式(BGR)からPIL形式(RGB)に変換
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color[::-1])  # BGRからRGBに変換
        # PIL形式からOpenCV形式に戻す
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def calibrate(self, landmarks, w, h):
        """軽量化されたキャリブレーションシステム"""
        if self.calibration_frames < 60:  # 2秒間で基準値を収集
            params = self.calculate_raw_params_lightweight(landmarks, w, h)
            
            # 軽量化：最小限のキャリブレーション
            # JawOpenの基準値のみ収集
            if 'JawOpen' in params and params['JawOpen'] > 0:
                if not hasattr(self, 'baseline_jaw_samples'):
                    self.baseline_jaw_samples = []
                self.baseline_jaw_samples.append(params['JawOpen'])
            
            self.calibration_frames += 1
            return False
        else:
            # JawOpenの基準値設定
            if hasattr(self, 'baseline_jaw_samples') and len(self.baseline_jaw_samples) > 10:
                # 最小値を基準として使用（口を閉じた状態）
                jaw_baseline = min(self.baseline_jaw_samples)
                if hasattr(self, 'jaw_open_baseline'):
                    self.jaw_open_baseline = jaw_baseline
            
            self.calibration_needed = False
            print("Lightweight calibration completed!")
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
    
    def detect_cheek_puff_stereo(self, landmarks_primary, landmarks_secondary, w_primary, h_primary, w_secondary, h_secondary):
        """ステレオカメラによる高精度頬膨らみ検出"""
        try:
            # プライマリカメラの頬ポイント
            left_cheek_p = landmarks_primary[234]
            right_cheek_p = landmarks_primary[454]
            
            # セカンダリカメラの頬ポイント
            left_cheek_s = landmarks_secondary[234]
            right_cheek_s = landmarks_secondary[454]
            
            # ステレオ視による3D座標計算
            # 視差計算（2台のカメラ間での位置差）
            left_disparity = abs((left_cheek_p.x * w_primary) - (left_cheek_s.x * w_secondary))
            right_disparity = abs((right_cheek_p.x * w_primary) - (right_cheek_s.x * w_secondary))
            
            # 基準視差を保存
            if not hasattr(self, 'base_cheek_disparity'):
                self.base_cheek_disparity = {
                    'left': left_disparity,
                    'right': right_disparity
                }
            
            # 膨らみによる視差変化を検出
            left_depth_change = (left_disparity - self.base_cheek_disparity['left']) / (self.base_cheek_disparity['left'] + 1e-6)
            right_depth_change = (right_disparity - self.base_cheek_disparity['right']) / (self.base_cheek_disparity['right'] + 1e-6)
            
            # 膨らみ強度計算（前方への突出）
            left_puff = max(0, min(1.0, left_depth_change * 20.0))
            right_puff = max(0, min(1.0, right_depth_change * 20.0))
            
            return left_puff, right_puff
            
        except (IndexError, AttributeError):
            # フォールバック：単一カメラでの検出
            return self.detect_cheek_puff_optical_flow_single(landmarks_primary, w_primary, h_primary)

    def detect_cheek_puff_optical_flow_single(self, landmarks, w, h):
        """単一カメラ用頬膨らみ検出（大幅改良版）"""
        try:
            # より多くの頬ランドマークを使用
            left_cheek_main = landmarks[234]   # 左頬の中心
            right_cheek_main = landmarks[454]  # 右頬の中心
            
            # 顔の基準線（鼻筋）
            nose_bridge_top = landmarks[6]     # 鼻筋上部
            nose_bridge_bottom = landmarks[2]  # 鼻筋下部
            face_center_line_x = (nose_bridge_top.x + nose_bridge_bottom.x) * w / 2
            
            # 詳細な頬ポイント（輪郭を含む）
            left_cheek_detailed = [landmarks[i] for i in [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 234, 227, 137]]
            right_cheek_detailed = [landmarks[i] for i in [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376, 454, 447, 366]]
            
            # 頬の幅（顔の外側への膨らみ）を測定
            left_cheek_outer_x = min([p.x * w for p in left_cheek_detailed])  # 最も左のポイント
            right_cheek_outer_x = max([p.x * w for p in right_cheek_detailed]) # 最も右のポイント
            
            # 基準幅を設定
            if not hasattr(self, 'base_cheek_width'):
                self.base_cheek_width = {
                    'left_distance': abs(face_center_line_x - left_cheek_outer_x),
                    'right_distance': abs(right_cheek_outer_x - face_center_line_x),
                    'total_width': abs(right_cheek_outer_x - left_cheek_outer_x)
                }
            
            # 現在の頬の外側への膨らみを計算
            current_left_distance = abs(face_center_line_x - left_cheek_outer_x)
            current_right_distance = abs(right_cheek_outer_x - face_center_line_x)
            
            # 膨らみ率を計算
            left_expansion_ratio = (current_left_distance - self.base_cheek_width['left_distance']) / (self.base_cheek_width['left_distance'] + 1e-6)
            right_expansion_ratio = (current_right_distance - self.base_cheek_width['right_distance']) / (self.base_cheek_width['right_distance'] + 1e-6)
            
            # 頬の面積による膨らみ検出も併用
            left_area = self.calculate_cheek_area(left_cheek_detailed, w, h)
            right_area = self.calculate_cheek_area(right_cheek_detailed, w, h)
            
            if not hasattr(self, 'base_cheek_areas'):
                self.base_cheek_areas = {
                    'left': left_area,
                    'right': right_area
                }
            
            # 面積変化
            left_area_expansion = (left_area - self.base_cheek_areas['left']) / (self.base_cheek_areas['left'] + 1e-6)
            right_area_expansion = (right_area - self.base_cheek_areas['right']) / (self.base_cheek_areas['right'] + 1e-6)
            
            # 幅と面積の両方を考慮した膨らみ検出
            left_puff_width = max(0, left_expansion_ratio * 25.0)   # 幅による検出
            right_puff_width = max(0, right_expansion_ratio * 25.0)
            
            left_puff_area = max(0, left_area_expansion * 20.0)     # 面積による検出
            right_puff_area = max(0, right_area_expansion * 20.0)
            
            # より強い方を採用
            left_puff = min(1.0, max(left_puff_width, left_puff_area))
            right_puff = min(1.0, max(right_puff_width, right_puff_area))
            
            # デバッグ情報を保存
            self.cheek_debug_info = {
                'left_expansion_ratio': left_expansion_ratio,
                'right_expansion_ratio': right_expansion_ratio,
                'left_area_expansion': left_area_expansion,
                'right_area_expansion': right_area_expansion,
                'left_puff_final': left_puff,
                'right_puff_final': right_puff,
                'face_center_x': face_center_line_x
            }
            
            return left_puff, right_puff
            
        except (IndexError, AttributeError, ZeroDivisionError):
            return 0, 0
    
    def calculate_cheek_area(self, cheek_points, w, h):
        """頬の面積を計算"""
        if len(cheek_points) < 3:
            return 0
        
        # 2D座標に変換
        points = np.array([(p.x * w, p.y * h) for p in cheek_points], dtype=np.float32)
        
        # 凸包を計算して面積取得
        hull = cv2.convexHull(points)
        area = cv2.contourArea(hull)
        
        return area
    
    def detect_tongue_stereo(self, landmarks_primary, landmarks_secondary, w, h, mouth_very_open, mouth_center_x, mouth_width, mouth_height, precise_mouth_height, get_rotated):
        """ステレオカメラによる高精度舌検出"""
        if not mouth_very_open:
            return {'x_displacement': 0, 'protrusion': 0, 'up': 0, 'down': 0}
        
        try:
            # プライマリとセカンダリの舌ポイント
            tongue_tip_p = landmarks_primary[17]
            tongue_tip_s = landmarks_secondary[17]
            
            # ステレオ視による舌の3D位置推定
            tongue_disparity = abs((tongue_tip_p.x * w) - (tongue_tip_s.x * w))
            
            # 基準視差
            if not hasattr(self, 'base_tongue_disparity'):
                self.base_tongue_disparity = tongue_disparity
            
            # 舌の前後移動（突出）
            depth_change = (tongue_disparity - self.base_tongue_disparity) / (self.base_tongue_disparity + 1e-6)
            tongue_protrusion = max(0, min(1.0, depth_change * 25.0))
            
            # 従来の2D検出も併用
            single_result = self.detect_tongue_single(landmarks_primary, w, h, mouth_very_open, mouth_center_x, mouth_width, mouth_height, precise_mouth_height, get_rotated)
            
            # ステレオと2Dの結果を合成
            return {
                'x_displacement': single_result['x_displacement'],
                'protrusion': max(tongue_protrusion, single_result['protrusion']),
                'up': single_result['up'],
                'down': single_result['down']
            }
            
        except (IndexError, AttributeError):
            # フォールバック
            return self.detect_tongue_single(landmarks_primary, w, h, mouth_very_open, mouth_center_x, mouth_width, mouth_height, precise_mouth_height, get_rotated)
    
    def detect_tongue_single(self, landmarks, w, h, mouth_very_open, mouth_center_x, mouth_width, mouth_height, precise_mouth_height, get_rotated):
        """単一カメラでの舌検出（大幅改良版）"""
        # 口の開き具合をより詳細にチェック
        mouth_aspect_ratio = precise_mouth_height / (mouth_width + 1e-6)
        mouth_open_enough = mouth_aspect_ratio > 0.06  # より低い閾値
        
        if not mouth_open_enough:
            return {'x_displacement': 0, 'protrusion': 0, 'up': 0, 'down': 0}
        
        # 舌関連のより多くのランドマーク
        try:
            tongue_tip = get_rotated(17)  # 舌先
            lower_lip_center = get_rotated(14)  # 下唇中央
            upper_lip_center = get_rotated(13)  # 上唇中央
            
            # 口の内側の基準点
            mouth_left_inner = get_rotated(78)   # 口の左内側
            mouth_right_inner = get_rotated(308) # 口の右内側
            
            # 舌の中央位置計算
            mouth_inner_center_x = (mouth_left_inner[0] + mouth_right_inner[0]) / 2
            mouth_inner_center_y = (upper_lip_center[1] + lower_lip_center[1]) / 2
            
            # 舌の左右移動（改良版）
            tongue_x_offset = (tongue_tip[0] - mouth_inner_center_x) / (mouth_width + 1e-6)
            
            # より小さなデッドゾーンと高感度
            tongue_deadzone = 0.03  # 3%まで削減
            
            if abs(tongue_x_offset) > tongue_deadzone:
                # 左右の感度を大幅アップ
                tongue_x_displacement = (tongue_x_offset - math.copysign(tongue_deadzone, tongue_x_offset)) * 12.0
            else:
                tongue_x_displacement = 0
                
            # 舌の突出度（大幅改良）
            # 舌先と下唇の距離をより正確に
            lip_baseline_y = lower_lip_center[1]
            tongue_y = tongue_tip[1]
            
            # 舌が口から出ているかの判定（より精密）
            lip_to_tongue_distance = (tongue_y - lip_baseline_y) / (precise_mouth_height + 1e-6)
            
            # 突出検出の感度を上げる
            tongue_protrusion = 0
            if lip_to_tongue_distance > 0.05:  # 5%以上下に出ていれば検出
                tongue_protrusion = min(1.0, lip_to_tongue_distance * 15.0)
            
            # 舌の上下動作（大幅改良）
            # 舌が上に動く場合
            upper_baseline_y = upper_lip_center[1]
            tongue_up_distance = (upper_baseline_y - tongue_y) / (precise_mouth_height + 1e-6)
            
            tongue_up = 0
            if tongue_up_distance > 0.1:  # 上唇に近づく
                tongue_up = min(1.0, (tongue_up_distance - 0.1) * 10.0)
            
            # 舌が下に動く場合（突出とは別）
            tongue_down_distance = (tongue_y - mouth_inner_center_y) / (precise_mouth_height + 1e-6)
            
            tongue_down = 0
            if tongue_down_distance > 0.15:  # 口の中央より下
                tongue_down = min(1.0, (tongue_down_distance - 0.15) * 8.0)
            
            # デバッグ情報を保存
            self.tongue_debug_info = {
                'mouth_open': mouth_open_enough,
                'mouth_ratio': mouth_aspect_ratio,
                'x_offset': tongue_x_offset,
                'protrusion_raw': lip_to_tongue_distance,
                'up_distance': tongue_up_distance,
                'down_distance': tongue_down_distance,
                'tongue_pos': [tongue_tip[0], tongue_tip[1]],
                'mouth_center': [mouth_inner_center_x, mouth_inner_center_y]
            }
            
            return {
                'x_displacement': min(max(tongue_x_displacement, -1), 1),
                'protrusion': min(max(tongue_protrusion, 0), 1),
                'up': min(max(tongue_up, 0), 1),
                'down': min(max(tongue_down, 0), 1)
            }
            
        except (IndexError, AttributeError):
            return {'x_displacement': 0, 'protrusion': 0, 'up': 0, 'down': 0}
    
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

    def calculate_raw_params_lightweight(self, landmarks, w, h):
        """軽量化：JawOpenと基本的な口の動きのみ計算"""
        def get_xy(idx):
            return landmarks[idx].x * w, landmarks[idx].y * h
        
        # 超至近距離・顎下からの画角に特化したJawOpen計算

        # 重要なランドマークのみ使用（軽量化）
        upper_lip = get_xy(13)      # 上唇中央
        lower_lip = get_xy(14)      # 下唇中央
        left_corner = get_xy(61)    # 左口角
        right_corner = get_xy(291)  # 右口角
        chin = get_xy(18)           # 顎先

        # 口の基本寸法
        mouth_width = abs(right_corner[0] - left_corner[0])

        # 唇の距離（純粋な縦方向のみ）
        lip_distance = abs(upper_lip[1] - lower_lip[1])

        # 顎先から上唇までの距離を正規化
        chin_to_upper_lip = abs(chin[1] - upper_lip[1])
        jaw_open_ratio = lip_distance / (chin_to_upper_lip + 1e-6)

        # キャリブレーション中は基準値を記録
        if self.calibration_needed:
            if not hasattr(self, 'jaw_baseline_buffer'):
                self.jaw_baseline_buffer = []

            self.jaw_baseline_buffer.append(jaw_open_ratio)

            # キャリブレーション完了時に基準値を確定
            if self.calibration_frames >= 60:
                self.jaw_open_baseline = np.mean(self.jaw_baseline_buffer)
                self.jaw_baseline_buffer = []

        # 基準値が設定されていない場合は現在値を使用
        if not hasattr(self, 'jaw_open_baseline'):
            self.jaw_open_baseline = jaw_open_ratio

        # JawOpen計算（デッドゾーン付き）
        jaw_open_change = jaw_open_ratio - self.jaw_open_baseline

        # デッドゾーン: 小さな変化は無視
        jaw_deadzone = 0.015  # 少し大きめに設定
        if abs(jaw_open_change) < jaw_deadzone:
            jaw_open_change = 0
        else:
            jaw_open_change = jaw_open_change - math.copysign(jaw_deadzone, jaw_open_change)

        # 口角が上がっている時は、誤検出を防ぐため感度を下げる
        corner_height_avg = (left_corner[1] + right_corner[1]) / 2
        mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2

        # 口角が口の中心より上にある場合（笑顔など）
        if corner_height_avg < mouth_center_y:
            # 感度を50%に減らす
            jaw_open_change *= 0.5

        jaw_open = max(0, min(1.0, jaw_open_change * 15.0))
        
        # 顎の左右移動（簡素化）
        face_center_x = (left_corner[0] + right_corner[0]) / 2
        jaw_offset = (chin[0] - face_center_x) / (mouth_width + 1e-6)
        
        # デッドゾーン適用
        jaw_deadzone = 0.02
        if abs(jaw_offset) > jaw_deadzone:
            jaw_displacement = (jaw_offset - math.copysign(jaw_deadzone, jaw_offset)) * 8.0
        else:
            jaw_displacement = 0
        
        # 口角の動き（キャリブレーションベース）
        # mouth_center_yは上で既に計算済み

        # キャリブレーション中は基準値を記録
        if self.calibration_needed:
            if not hasattr(self, 'corner_baseline_buffer'):
                self.corner_baseline_buffer = {'left': [], 'right': []}

            self.corner_baseline_buffer['left'].append(left_corner[1])
            self.corner_baseline_buffer['right'].append(right_corner[1])

            # キャリブレーション完了時に基準値を確定
            if self.calibration_frames >= 60:
                self.baseline_left_corner_y = np.mean(self.corner_baseline_buffer['left'])
                self.baseline_right_corner_y = np.mean(self.corner_baseline_buffer['right'])
                self.corner_baseline_buffer = {'left': [], 'right': []}

        # 基準値が設定されていない場合は現在値を使用
        if not hasattr(self, 'baseline_left_corner_y'):
            self.baseline_left_corner_y = left_corner[1]
            self.baseline_right_corner_y = right_corner[1]

        # 基準値からの変化量を計算
        left_corner_change = (self.baseline_left_corner_y - left_corner[1])
        right_corner_change = (self.baseline_right_corner_y - right_corner[1])

        # デッドゾーン: 小さな変化は無視（ノイズ対策）
        corner_deadzone = mouth_width * 0.02  # 口の幅の2%

        if abs(left_corner_change) < corner_deadzone:
            left_corner_change = 0
        else:
            left_corner_change = left_corner_change - math.copysign(corner_deadzone, left_corner_change)

        if abs(right_corner_change) < corner_deadzone:
            right_corner_change = 0
        else:
            right_corner_change = right_corner_change - math.copysign(corner_deadzone, right_corner_change)

        # 口の幅で正規化
        left_corner_pull = max(0, min(1, left_corner_change / (mouth_width * 0.08 + 1e-6)))
        right_corner_pull = max(0, min(1, right_corner_change / (mouth_width * 0.08 + 1e-6)))
        
        # 口すぼめ（簡素化）
        inner_width = mouth_width * 0.7  # 簡易計算
        pucker = max(0, (mouth_width - inner_width) / mouth_width * 2.0)
        
        return {
            "JawOpen": min(max(jaw_open, 0), 1),
            "JawRight": min(max(jaw_displacement, 0), 1) if jaw_displacement > 0 else 0,
            "JawLeft": min(max(-jaw_displacement, 0), 1) if jaw_displacement < 0 else 0,
            "MouthCornerPullRight": min(max(right_corner_pull, 0), 1),
            "MouthCornerPullLeft": min(max(left_corner_pull, 0), 1),
            "MouthPucker": min(max(pucker, 0), 1),
            # 不要な機能は全て0
            "JawForward": 0,
            "CheekPuffRight": 0,
            "CheekPuffLeft": 0,
            "TongueOut": 0,
            "TongueUp": 0,
            "TongueDown": 0,
            "TongueRight": 0,
            "TongueLeft": 0,
            "NoseSneerLeft": 0,
            "NoseSneerRight": 0
        }
        
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
            # === 顎の左右動作の改善（感度差解消） ===
            chin_point = landmarks[18]
            face_center_x = (landmarks[234].x + landmarks[454].x) * w / 2
            face_width = abs(landmarks[234].x - landmarks[454].x) * w
            
            # 顎の横方向偏移（正面時は0になるように）
            jaw_offset = (chin_point.x * w - face_center_x) / (face_width + 1e-6)
            
            # さらに小さなデッドゾーンで感度向上
            jaw_deadzone = 0.015  # 1.5%のデッドゾーン（さらに削減）
            
            if abs(jaw_offset) > jaw_deadzone:
                # 左右の感度をさらに強化（特に右側）
                if jaw_offset > 0:  # 右方向
                    jaw_displacement_x_raw = (jaw_offset - jaw_deadzone) * 8.0  # 右側感度大幅アップ
                else:  # 左方向
                    jaw_displacement_x_raw = (jaw_offset + jaw_deadzone) * 7.0  # 左側も感度アップ
            else:
                jaw_displacement_x_raw = 0
            
            # === 顎の前後動作の改善（距離依存解消） ===
            chin_point = landmarks[18]
            upper_lip_center = landmarks[13]
            
            # 顎から上唇への距離を顔幅で正規化（距離依存解消）
            chin_to_lip_distance_raw = abs(chin_point.y - upper_lip_center.y) * h
            chin_to_lip_distance_normalized = chin_to_lip_distance_raw / (face_width + 1e-6)
            
            # 正規化された基準距離を設定
            if not hasattr(self, 'base_chin_lip_ratio'):
                self.base_chin_lip_ratio = chin_to_lip_distance_normalized
            
            # 正規化された距離の変化率で前後動作を検出
            ratio_change = (chin_to_lip_distance_normalized - self.base_chin_lip_ratio) / (self.base_chin_lip_ratio + 1e-6)
            
            # 顎が前に出ると比率が小さくなるので、マイナス値を前進として扱う
            jaw_forward_raw = max(0, -ratio_change * 10.0)  # 感度調整
            
        else:
            # 3D姿勢推定が失敗した場合のフォールバック
            chin_point = get_rotated(18)
            face_center_x = (landmarks[234].x + landmarks[454].x) * w / 2
            face_width = abs(landmarks[234].x - landmarks[454].x) * w
            
            # フォールバック時も同じ感度調整を適用
            jaw_offset_from_center = (chin_point[0] - face_center_x) / (face_width + 1e-6)
            jaw_deadzone = 0.015
            
            if abs(jaw_offset_from_center) > jaw_deadzone:
                if jaw_offset_from_center > 0:  # 右方向
                    jaw_displacement_x_raw = (jaw_offset_from_center - jaw_deadzone) * 8.0
                else:  # 左方向
                    jaw_displacement_x_raw = (jaw_offset_from_center + jaw_deadzone) * 7.0
            else:
                jaw_displacement_x_raw = 0
            
            # フォールバック時も正規化された距離検出
            chin_point_landmark = landmarks[18]
            upper_lip_center = landmarks[13]
            chin_to_lip_distance_raw = abs(chin_point_landmark.y - upper_lip_center.y) * h
            chin_to_lip_distance_normalized = chin_to_lip_distance_raw / (face_width + 1e-6)
            
            if not hasattr(self, 'base_chin_lip_ratio'):
                self.base_chin_lip_ratio = chin_to_lip_distance_normalized
            
            ratio_change = (chin_to_lip_distance_normalized - self.base_chin_lip_ratio) / (self.base_chin_lip_ratio + 1e-6)
            jaw_forward_raw = max(0, -ratio_change * 10.0)
        
        # （jaw_open_factorは上で既に計算済み）
        
        # === 光学フローによる高精度頬膨らみ検出 ===
        # グレースケール変換
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # ステレオカメラ対応の頬膨らみ検出
        if self.dual_camera and hasattr(self, 'secondary_landmarks'):
            left_cheek_expansion, right_cheek_expansion = self.detect_cheek_puff_stereo(
                landmarks, self.secondary_landmarks, w, h, w, h)
        else:
            left_cheek_expansion, right_cheek_expansion = self.detect_cheek_puff_optical_flow_single(landmarks, w, h)
        
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
        
        # パラメータの計算と正規化（舌・頬機能を無効化）
        params = {
            "JawOpen": min(max(jaw_open_factor, 0), 1),  # 口の開閉 - 動作確認済み
            "JawRight": min(max(jaw_displacement_x_raw, 0), 1),  # 顎右 - 動作確認済み
            "JawLeft": min(max(-jaw_displacement_x_raw, 0), 1),  # 顎左 - 動作確認済み
            "JawForward": 0,  # 無効化（動作しないため削除）
            "MouthCornerPullRight": min(max(right_corner_pull * 3, 0), 1),  # 右口角 - 動作確認済み
            "MouthCornerPullLeft": min(max(left_corner_pull * 3, 0), 1),  # 左口角 - 動作確認済み
            "MouthPucker": min(max(pucker_intensity, 0), 1),  # 口すぼめ - 動作確認済み
            # 頬・舌機能を無効化
            "CheekPuffRight": 0,  # 無効化
            "CheekPuffLeft": 0,   # 無効化
            "TongueOut": 0,       # 無効化
            "TongueUp": 0,        # 無効化
            "TongueDown": 0,      # 無効化
            "TongueRight": 0,     # 無効化
            "TongueLeft": 0       # 無効化
        }
        
        # スケール情報を保存（デバッグ用）
        self.last_scale_info = {
            'factor': scale_factor,
            'scale': face_scale
        }
        
        # デバッグ用の口と頬のパラメータを保存
        self.last_mouth_params = {
            'jaw_open': params.get('JawOpen', 0),
            'jaw_forward': params.get('JawForward', 0),
            'jaw_forward_raw': jaw_forward_raw,
            'chin_lip_ratio': chin_to_lip_distance_normalized if 'chin_to_lip_distance_normalized' in locals() else 0,
            'chin_lip_base_ratio': getattr(self, 'base_chin_lip_ratio', 0),
            'jaw_offset': jaw_offset if 'jaw_offset' in locals() else 0,
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
        """目のパラメータ - 動作しないため無効化"""
        # 目のトラッキングが不安定なため、全て無効にする
        return {
            "EyesX": 0,
            "EyesY": 0, 
            "LeftEyeLid": 0,
            "RightEyeLid": 0,
            "EyesWiden": 0,
            "EyeSquintLeft": 0,
            "EyeSquintRight": 0,
            "EyeWideLeft": 0,
            "EyeWideRight": 0,
            "BrowInnerUpLeft": 0,
            "BrowInnerUpRight": 0,
            "BrowLowererLeft": 0,
            "BrowLowererRight": 0,
            "BrowOuterUpLeft": 0,
            "BrowOuterUpRight": 0
        }

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

    def process_dual_frame(self, frame_primary, frame_secondary=None):
        """2台カメラフレームを処理してOSCデータを送信"""
        # プライマリカメラの処理
        rgb_primary = cv2.cvtColor(frame_primary, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_primary.shape
        
        results_primary = self.face_mesh_primary.process(rgb_primary)
        
        # セカンダリカメラの処理（利用可能な場合）
        results_secondary = None
        if self.dual_camera and frame_secondary is not None:
            rgb_secondary = cv2.cvtColor(frame_secondary, cv2.COLOR_BGR2RGB)
            results_secondary = self.face_mesh_secondary.process(rgb_secondary)
            if results_secondary.multi_face_landmarks:
                self.secondary_landmarks = results_secondary.multi_face_landmarks[0].landmark
        
        return self.process_landmarks(frame_primary, results_primary, w, h)
    
    def process_frame(self, frame):
        """単一カメラ用のフレーム処理（VRゴーグル対応：顔下半分専用）"""
        h, w, _ = frame.shape

        # カメラが上下逆さなので反転（VRゴーグル装着時）
        frame_flipped = cv2.flip(frame, -1)  # 上下左右反転

        # VRゴーグル対応：画像を上に拡張して顔全体に見せかける
        # MediaPipeは顔全体が必要なため、画像の上部を黒で埋めて拡張
        expanded_height = int(h * 2.5)  # より大きく拡張
        expanded_frame = np.zeros((expanded_height, w, 3), dtype=np.uint8)
        # 元の画像を下部に配置
        y_offset = expanded_height - h
        expanded_frame[y_offset:, :] = frame_flipped

        # 画像前処理：検出率向上のため
        # 1. ノイズ除去
        frame_processed = cv2.bilateralFilter(expanded_frame, 5, 50, 50)

        # 2. CLAHE（コントラスト制限適応ヒストグラム平坦化）
        yuv = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        frame_enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # MediaPipe用にRGBに変換
        rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)

        # MediaPipe処理
        results = self.face_mesh.process(rgb)

        # ランドマークのY座標を調整（元の画像座標系に戻す）
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                for landmark in landmarks.landmark:
                    # Y座標を元の画像座標系に戻す
                    landmark.y = (landmark.y * expanded_height - y_offset) / h

        # 反転後のフレームで描画
        return self.process_landmarks(frame_flipped, results, w, h)
    
    def draw_buttons(self, frame):
        """画面上部にボタンを描画"""
        h, w, _ = frame.shape
        
        for button_name, button in self.buttons.items():
            x, y, width, height = button['x'], button['y'], button['width'], button['height']
            text = button['text']
            
            # ボタンの背景
            color = (50, 150, 50) if button_name == 'calibrate' else (100, 100, 100)
            if hasattr(self, 'button_hover') and self.button_hover == button_name:
                color = (80, 180, 80) if button_name == 'calibrate' else (130, 130, 130)
            
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
            
            # ボタンのテキスト
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x + (width - text_size[0]) // 2
            text_y = y + (height + text_size[1]) // 2
            
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def check_button_click(self, x, y):
        """ボタンクリックの判定"""
        for button_name, button in self.buttons.items():
            if (button['x'] <= x <= button['x'] + button['width'] and
                button['y'] <= y <= button['y'] + button['height']):
                return button_name
        return None
    
    def process_landmarks(self, frame, results, w, h):
        """ランドマーク処理の共通ロジック"""
        
        # ボタンを描画
        frame = self.draw_buttons(frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # RGB変換（キャリブレーション用）
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
            
            # 顔スケール計算（全体で共有）
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            face_width = abs(left_cheek.x - right_cheek.x) * w
            face_height = abs(landmarks[10].y - landmarks[152].y) * h
            face_scale = (face_width + face_height) / 2
            
            # パラメータ計算（軽量版）
            raw_params = self.calculate_raw_params_lightweight(landmarks, w, h)
            
            # 不要な機能は削除（軽量化）
            
            # 軽量化：基準値処理を簡素化
            calibrated_params = raw_params
            
            # スムージング適用
            smoothed_params = self.smooth_params(calibrated_params)
            
            # OSC送信とUI表示 - 動作するパラメータのみ
            h, w, _ = frame.shape
            left_col_x = 10
            right_col_x = w // 2 + 10
            left_y = 60
            right_y = 60
            
            # 無効化されたパラメータリスト（軽量化：大部分を無効化）
            disabled_params = {
                'JawForward',  # 顎前方
                'CheekPuffRight', 'CheekPuffLeft',  # 頬膨らみ
                'TongueOut', 'TongueUp', 'TongueDown', 'TongueRight', 'TongueLeft',  # 舌
                'EyesX', 'EyesY', 'LeftEyeLid', 'RightEyeLid', 'EyesWiden',  # 目
                'EyeSquintLeft', 'EyeSquintRight', 'EyeWideLeft', 'EyeWideRight',  # 目
                'BrowInnerUpLeft', 'BrowInnerUpRight', 'BrowLowererLeft', 'BrowLowererRight',  # 眉毛
                'BrowOuterUpLeft', 'BrowOuterUpRight',  # 眉毛
                'NoseSneerLeft', 'NoseSneerRight'  # 鼻（軽量化のため無効化）
            }
            
            # パラメータを2列に分けて表示（動作するもののみ）
            param_count = 0
            for key, val in smoothed_params.items():
                if key in PARAMS and key not in disabled_params:  # 動作するパラメータのみ送信
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
            
            # システム状態表示（上部中央）
            cv2.putText(frame, f"VR Face Tracking: Lightweight Mode (Jaw Focus)", (w//2 - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 軽量化：最小限のデバッグ表示
            # 主要な口のポイントのみ表示
            mouth_points = []
            for idx in [61, 291, 13, 14, 18]:  # 左口角、右口角、上唇、下唇、顎先
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                mouth_points.append((x, y))
            
            # 有効なパラメータのみデバッグ表示（青い文字）
            blue_color = (255, 0, 0)  # BGRなので青は(255,0,0)
            
            # 動作確認済みのパラメータのみ表示
            if hasattr(self, 'last_mouth_params'):
                mouth_params = self.last_mouth_params
                cv2.putText(frame, f"WORKING PARAMETERS:", (10, h - 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue_color, 2)
                cv2.putText(frame, f"Jaw Open: {mouth_params['jaw_open']:.3f}", (10, h - 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Jaw L/R: {mouth_params.get('jaw_left', 0):.3f}/{mouth_params.get('jaw_right', 0):.3f}", (10, h - 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Corner Pull L/R: {mouth_params['corner_left']:.3f}/{mouth_params['corner_right']:.3f}", (10, h - 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Mouth Pucker: {mouth_params.get('pucker', 0):.3f}", (10, h - 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue_color, 1)
                cv2.putText(frame, f"Jaw Offset: {mouth_params.get('jaw_offset', 0):.3f} (L/R sensitivity: 7.0/8.0)", (10, h - 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, blue_color, 1)
            
            # 舌のデバッグ情報を表示
            if hasattr(self, 'tongue_debug_info'):
                tongue_info = self.tongue_debug_info
                debug_y = h - 220
                cv2.putText(frame, f"TONGUE DEBUG:", (w//2, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Open: {tongue_info['mouth_open']} Ratio: {tongue_info['mouth_ratio']:.3f}", (w//2, debug_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(frame, f"X: {tongue_info['x_offset']:.3f} Prot: {tongue_info['protrusion_raw']:.3f}", (w//2, debug_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(frame, f"Up: {tongue_info['up_distance']:.3f} Down: {tongue_info['down_distance']:.3f}", (w//2, debug_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 頬のデバッグ情報を表示
            if hasattr(self, 'cheek_debug_info'):
                cheek_info = self.cheek_debug_info
                debug_y = h - 140
                cv2.putText(frame, f"CHEEK DEBUG:", (w//2, debug_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"L Exp: {cheek_info['left_expansion_ratio']:.3f} R Exp: {cheek_info['right_expansion_ratio']:.3f}", (w//2, debug_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(frame, f"L Puff: {cheek_info['left_puff_final']:.3f} R Puff: {cheek_info['right_puff_final']:.3f}", (w//2, debug_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # カメラ使用状況の表示
            camera_info = "Dual Camera Active" if (self.dual_camera and hasattr(self, 'secondary_landmarks')) else "Single Camera Mode"
            cv2.putText(frame, f"Camera: {camera_info}", (10, h - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 口の輪郭線を描画
            if len(mouth_points) >= 4:
                cv2.line(frame, mouth_points[0], mouth_points[1], (255, 0, 0), 1)  # 口角線
                cv2.line(frame, mouth_points[2], mouth_points[3], (255, 0, 0), 1)  # 縦線
        
        else:
            # 顔検出失敗時のメッセージ
            h, w, _ = frame.shape
            frame = self.put_japanese_text(frame, "顔が検出できません", (10, 60), self.font_large, (0, 0, 255))
            frame = self.put_japanese_text(frame, "対処:", (10, h - 100), self.font, (0, 255, 255))
            frame = self.put_japanese_text(frame, "1. カメラを口周辺（鼻下〜顎）に向けてください", (10, h - 70), self.font_small, (255, 255, 255))
            frame = self.put_japanese_text(frame, "2. 照明を明るくしてください", (10, h - 45), self.font_small, (255, 255, 255))

        return frame

    def extract_mouth_roi(self, frame, landmarks, w, h, padding=50):
        """
        口周辺のROI（関心領域）を抽出
        VRゴーグル着用時は目を含まず、口周辺のみを処理
        """
        try:
            # 口周辺の主要ポイント
            mouth_points = []
            # 口角
            mouth_points.append((landmarks[61].x * w, landmarks[61].y * h))   # 左口角
            mouth_points.append((landmarks[291].x * w, landmarks[291].y * h)) # 右口角
            # 上下唇
            mouth_points.append((landmarks[0].x * w, landmarks[0].y * h))     # 上唇上部
            mouth_points.append((landmarks[17].x * w, landmarks[17].y * h))   # 下唇下部
            # 顎
            mouth_points.append((landmarks[18].x * w, landmarks[18].y * h))   # 顎先

            # バウンディングボックスを計算
            x_coords = [p[0] for p in mouth_points]
            y_coords = [p[1] for p in mouth_points]

            min_x = max(0, int(min(x_coords) - padding))
            max_x = min(w, int(max(x_coords) + padding))
            min_y = max(0, int(min(y_coords) - padding))
            max_y = min(h, int(max(y_coords) + padding))

            # ROIを抽出
            roi = frame[min_y:max_y, min_x:max_x]

            return roi, (min_x, min_y, max_x, max_y)

        except (IndexError, AttributeError):
            return frame, (0, 0, w, h)

class CameraSelector:
    """カメラ選択とコントロール用のGUIクラス"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VR Facial Tracking - Camera Selector")
        self.root.geometry("400x200")

        self.selected_camera = None
        self.is_running = False
        self.tracker = None
        self.cap = None
        self.tracking_thread = None  # スレッド管理用
        self.cleanup_done = False  # クリーンアップ完了フラグ（重複防止）

        # 利用可能なカメラを検出
        self.available_cameras = detect_available_cameras()

        # UI構築
        self.build_ui()

        # クリーンアップハンドラーを登録
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # GUIウィンドウを閉じる時のハンドラー（最重要）
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """GUIウィンドウを閉じる時の処理"""
        print("\nウィンドウを閉じています...")

        # トラッキングを停止
        self.is_running = False

        # クリーンアップ
        self.cleanup()

        # GUIを破棄
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

        # プロセスを完全に終了
        print("プロセスを終了します")
        sys.exit(0)

    def build_ui(self):
        """UIを構築"""
        # タイトル
        title_label = tk.Label(self.root, text="VR Facial Tracking", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # カメラ選択フレーム
        camera_frame = tk.Frame(self.root)
        camera_frame.pack(pady=10)

        tk.Label(camera_frame, text="カメラを選択:").pack(side=tk.LEFT, padx=5)

        # カメラ選択ドロップダウン
        self.camera_var = tk.StringVar()
        if self.available_cameras:
            self.camera_var.set(f"Camera {self.available_cameras[0]}")

        camera_options = [f"Camera {i}" for i in self.available_cameras]
        self.camera_dropdown = ttk.Combobox(camera_frame, textvariable=self.camera_var,
                                           values=camera_options, state="readonly", width=15)
        self.camera_dropdown.pack(side=tk.LEFT, padx=5)

        # スタート/ストップボタンフレーム
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        self.start_button = tk.Button(button_frame, text="スタート", command=self.start_tracking,
                                      bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                      width=10, height=2)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(button_frame, text="ストップ", command=self.stop_tracking,
                                     bg="#f44336", fg="white", font=("Arial", 12, "bold"),
                                     width=10, height=2, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        # 緊急停止ボタン
        self.emergency_button = tk.Button(button_frame, text="緊急停止", command=self.emergency_stop,
                                         bg="#ff0000", fg="white", font=("Arial", 10, "bold"),
                                         width=8, height=2)
        self.emergency_button.pack(side=tk.LEFT, padx=10)

        # ステータスラベル
        self.status_label = tk.Label(self.root, text="準備完了", fg="blue")
        self.status_label.pack(pady=10)

    def start_tracking(self):
        """トラッキングを開始"""
        if not self.available_cameras:
            self.status_label.config(text="カメラが見つかりません", fg="red")
            return

        # 選択されたカメラのインデックスを取得
        camera_text = self.camera_var.get()
        camera_index = int(camera_text.split()[1])

        self.selected_camera = camera_index
        self.is_running = True

        # ボタンの状態を更新
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.camera_dropdown.config(state=tk.DISABLED)

        self.status_label.config(text=f"Camera {camera_index} で実行中...", fg="green")

        # トラッキングを別スレッドで開始（非デーモン）
        self.tracking_thread = threading.Thread(target=self.run_tracking, daemon=False)
        self.tracking_thread.start()

    def stop_tracking(self):
        """トラッキングを停止"""
        print("\n停止ボタンが押されました")
        self.is_running = False

        # クリーンアップを実行（スレッド終了を待ってから）
        if self.tracking_thread is not None and self.tracking_thread.is_alive():
            print("スレッドの終了を待機中...")
            self.tracking_thread.join(timeout=3.0)

        # リソース解放
        self.cleanup()

        # ボタンの状態を更新
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.camera_dropdown.config(state="readonly")
        self.status_label.config(text="停止しました", fg="blue")

    def emergency_stop(self):
        """緊急停止: すべてのカメラリソースを即座に解放"""
        print("\n=== 緊急停止 ===")

        # 共通のクリーンアップ処理を呼び出す
        self.cleanup()

        # UIを更新
        try:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.camera_dropdown.config(state="readonly")
            self.status_label.config(text="緊急停止しました", fg="red")
        except:
            pass

    def cleanup(self):
        """アプリケーション終了時のクリーンアップ（1回のみ実行）"""
        # 重複実行を防止
        if self.cleanup_done:
            return

        self.cleanup_done = True
        print("\n=== クリーンアップ開始 ===")

        # トラッキングを停止
        self.is_running = False

        # OSCクライアントをクリーンアップ（最初に実行）
        if self.tracker is not None and hasattr(self.tracker, 'osc_client'):
            try:
                print("1. OSCパラメータをリセット...")
                for param_name in PARAMS.keys():
                    self.tracker.osc_client.send_message(PARAMS[param_name], 0.0)
            except Exception as e:
                print(f"   OSCリセットエラー: {e}")

        # MediaPipeリソースを明示的に解放
        if self.tracker is not None and hasattr(self.tracker, 'face_mesh'):
            try:
                print("2. MediaPipe解放中...")
                self.tracker.face_mesh.close()
            except Exception as e:
                print(f"   MediaPipe解放エラー: {e}")

        # OpenCVウィンドウを閉じる（カメラ解放の前）
        try:
            print("3. OpenCVウィンドウを閉じる...")
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"   ウィンドウクローズエラー: {e}")

        # カメラを解放（最も重要・1回だけ）
        if self.cap is not None:
            try:
                print("4. カメラ解放中...")
                # capをローカル変数に移動してから解放
                cap_to_release = self.cap
                self.cap = None  # 先にNoneにして二重解放を防ぐ
                cap_to_release.release()
                del cap_to_release
                print("   カメラ解放完了")
            except Exception as e:
                print(f"   カメラ解放エラー: {e}")
                self.cap = None

        # 最後にtrackerをNoneに設定
        self.tracker = None

        print("=== クリーンアップ完了 ===\n")

    def signal_handler(self, sig, frame):
        """シグナルハンドラー（Ctrl+C対応）"""
        print("\n\nCtrl+Cが検出されました。安全に終了します...")

        # クリーンアップ
        self.cleanup()

        # GUIを閉じる
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

        print("終了しました")
        sys.exit(0)

    def run_tracking(self):
        """トラッキングループを実行"""
        try:
            # トラッカー初期化
            self.tracker = FacialTracker(dual_camera=False)

            # カメラを開く
            self.cap = cv2.VideoCapture(self.selected_camera, cv2.CAP_DSHOW)

            if not self.cap.isOpened():
                # カメラが開けなかった場合、capを解放してからreturn
                cap_failed = self.cap
                self.cap = None
                cap_failed.release()
                del cap_failed
                self.root.after(0, lambda: self.status_label.config(text="カメラを開けませんでした", fg="red"))
                self.is_running = False
                return

            # カメラ設定（超低負荷）
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 640→320（さらに軽量化）
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 480→240
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)

            window_name = 'VR Facial Tracking'
            cv2.namedWindow(window_name)

            # マウスコールバックを設定（ボタンクリック用）
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    button_clicked = self.tracker.check_button_click(x, y)
                    if button_clicked == 'calibrate':
                        self.tracker.calibration_needed = True
                        self.tracker.calibration_frames = 0
                        self.tracker.baseline = {}
                        # 全バッファをクリア
                        if hasattr(self.tracker, 'jaw_baseline_buffer'):
                            delattr(self.tracker, 'jaw_baseline_buffer')
                        if hasattr(self.tracker, 'corner_baseline_buffer'):
                            delattr(self.tracker, 'corner_baseline_buffer')
                        if hasattr(self.tracker, 'jaw_open_baseline'):
                            delattr(self.tracker, 'jaw_open_baseline')
                        if hasattr(self.tracker, 'baseline_left_corner_y'):
                            delattr(self.tracker, 'baseline_left_corner_y')
                            delattr(self.tracker, 'baseline_right_corner_y')
                        print("キャリブレーション開始...")
                    elif button_clicked == 'reset':
                        self.tracker.calibration_needed = True
                        self.tracker.calibration_frames = 0
                        self.tracker.baseline = {}
                        # 全バッファをクリア
                        if hasattr(self.tracker, 'jaw_baseline_buffer'):
                            delattr(self.tracker, 'jaw_baseline_buffer')
                        if hasattr(self.tracker, 'corner_baseline_buffer'):
                            delattr(self.tracker, 'corner_baseline_buffer')
                        if hasattr(self.tracker, 'jaw_open_baseline'):
                            delattr(self.tracker, 'jaw_open_baseline')
                        if hasattr(self.tracker, 'baseline_left_corner_y'):
                            delattr(self.tracker, 'baseline_left_corner_y')
                            delattr(self.tracker, 'baseline_right_corner_y')
                        print("リセット...")
                    elif button_clicked == 'switch_camera':
                        print("カメラ切り替えは起動時のGUIから行ってください")

            cv2.setMouseCallback(window_name, mouse_callback)

            frame_count = 0
            import time
            frame_time_target = 1.0 / 30.0  # 30fps制限

            while self.is_running:
                frame_start = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    break

                # 水平反転（鏡像）
                frame = cv2.flip(frame, 1)

                # トラッキング処理（毎フレーム実行）
                frame = self.tracker.process_frame(frame)

                # コントロール用のテキストを追加
                h, w, _ = frame.shape
                cv2.putText(frame, "'q': Quit | 'c': Calibrate | 'r': Reset | 'ESC': Emergency Stop", (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(window_name, frame)

                # FPS制限（CPU負荷軽減）
                frame_elapsed = time.time() - frame_start
                sleep_time = max(0, frame_time_target - frame_elapsed)
                key = cv2.waitKey(max(1, int(sleep_time * 1000))) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                elif key == 27:  # ESCキー
                    print("\n⚠⚠⚠ ESCキーで緊急停止 ⚠⚠⚠")
                    self.is_running = False
                    # リソース解放はcleanup()で行う（二重解放を防ぐ）
                    self.root.after(0, lambda: self.status_label.config(text="緊急停止しました", fg="red"))
                    break
                elif key == ord('r'):
                    self.tracker.calibration_needed = True
                    self.tracker.calibration_frames = 0
                    self.tracker.baseline = {}
                    # 全バッファをクリア
                    if hasattr(self.tracker, 'jaw_baseline_buffer'):
                        delattr(self.tracker, 'jaw_baseline_buffer')
                    if hasattr(self.tracker, 'corner_baseline_buffer'):
                        delattr(self.tracker, 'corner_baseline_buffer')
                    if hasattr(self.tracker, 'jaw_open_baseline'):
                        delattr(self.tracker, 'jaw_open_baseline')
                    if hasattr(self.tracker, 'baseline_left_corner_y'):
                        delattr(self.tracker, 'baseline_left_corner_y')
                        delattr(self.tracker, 'baseline_right_corner_y')
                elif key == ord('c'):
                    self.tracker.calibration_needed = True
                    self.tracker.calibration_frames = 0
                    self.tracker.baseline = {}
                    # 全バッファをクリア
                    if hasattr(self.tracker, 'jaw_baseline_buffer'):
                        delattr(self.tracker, 'jaw_baseline_buffer')
                    if hasattr(self.tracker, 'corner_baseline_buffer'):
                        delattr(self.tracker, 'corner_baseline_buffer')
                    if hasattr(self.tracker, 'jaw_open_baseline'):
                        delattr(self.tracker, 'jaw_open_baseline')
                    if hasattr(self.tracker, 'baseline_left_corner_y'):
                        delattr(self.tracker, 'baseline_left_corner_y')
                        delattr(self.tracker, 'baseline_right_corner_y')

        finally:
            # トラッキングスレッド終了
            # 重要: リソース解放は一切しない（cleanup()で行う）
            print("\nトラッキングスレッド終了")

            # UIのみ更新（リソース解放はcleanup()に任せる）
            try:
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
                self.root.after(0, lambda: self.camera_dropdown.config(state="readonly"))
                self.root.after(0, lambda: self.status_label.config(text="停止しました", fg="blue"))
            except:
                pass

    def run(self):
        """GUIのメインループを開始"""
        self.root.mainloop()

def detect_available_cameras():
    """
    安全なカメラ検出（Varjo競合対策版）

    重要: カメラを一切開かない
    デバイス名のリストだけを取得
    """
    print("\nカメラ検出中（デバイスを開きません）...")

    # pygrabberが必須
    if not PYGRABBER_AVAILABLE:
        print("=" * 60)
        print("エラー: pygrabberが利用できません")
        print("=" * 60)
        print("以下のコマンドを実行してインストールしてください:")
        print("  pip install pygrabber")
        print("")
        print("pygrabberなしではWindowsデバイスマネージャーの")
        print("安全性を保証できないため、起動を中止します。")
        print("=" * 60)
        return []

    # pygrabberでデバイスリストを取得（カメラを一切開かない）
    try:
        graph = FilterGraph()
        devices = graph.get_input_devices()

        available_cameras = []
        varjo_keywords = ['varjo', 'eye', 'tracking', 'gaze']

        print("\n利用可能なカメラデバイス:")
        for i, device_name in enumerate(devices):
            # Varjoカメラをスキップ
            is_varjo = any(keyword in device_name.lower() for keyword in varjo_keywords)

            if is_varjo:
                print(f"  [{i}] {device_name} - Varjo（スキップ）")
                continue

            # カメラを開かずにリストに追加
            available_cameras.append(i)
            print(f"  [{i}] {device_name}")

        print(f"\n検出完了: {len(available_cameras)} 個のカメラ")
        print("注意: カメラは「スタート」ボタンを押すまで開きません\n")

        return available_cameras

    except Exception as e:
        print(f"エラー: カメラ検出に失敗 - {e}")
        return []

def main():
    """VR Facial Tracking - 起動"""
    print("VR Facial Tracking 起動中...")

    # GUIを起動
    app = CameraSelector()
    app.run()

if __name__ == "__main__":
    main()