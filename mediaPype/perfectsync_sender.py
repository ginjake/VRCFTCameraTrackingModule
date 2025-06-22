import cv2
import mediapipe as mp
from pythonosc import udp_client

# OSC設定
OSC_IP = "127.0.0.10"
OSC_PORT = 9011

PARAMS = {
    "A": "/avatar/parameters/Voice_A",
    "I": "/avatar/parameters/Voice_I",
    "U": "/avatar/parameters/Voice_U",
    "E": "/avatar/parameters/Voice_E",
    "O": "/avatar/parameters/Voice_O",
}

osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def mouth_params(lm, w, h):
    # 口の中心上下（13, 14）開き（A音）/横（78, 308）でI, U, E, O近似
    open_v = ((lm[13].y - lm[14].y) * h)
    wide = ((lm[78].x - lm[308].x) * w)
    # ざっくり正規化
    A = min(max((abs(open_v) - 5) / 20, 0), 1)
    I = min(max((abs(wide) - 35) / 20, 0), 1)
    U = min(max((abs(open_v) - 2) / 10, 0), 1)
    E = min(max((abs(wide) - 20) / 15, 0), 1)
    O = min(max((abs(open_v) + abs(wide) - 35) / 20, 0), 1)
    return {"A": A, "I": I, "U": U, "E": E, "O": O}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape
    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        mouth = mouth_params(lm, w, h)
        y = 30
        for key, val in mouth.items():
            # OSC送信
            osc_client.send_message(PARAMS[key], float(val))
            # 画面に値表示
            cv2.putText(frame, f"{key}: {val:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            y += 30
    cv2.imshow('PerfectSync OSC Demo', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESCで終了
        break

cap.release()
cv2.destroyAllWindows()