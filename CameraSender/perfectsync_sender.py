import cv2
import mediapipe as mp
from pythonosc import udp_client

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

osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)


def facial_params(lm, w, h):
    jaw_open = abs((lm[13].y - lm[14].y) * h)
    jaw_right = (lm[152].x - lm[10].x) * w
    jaw_left = (lm[10].x - lm[152].x) * w
    jaw_forward = (lm[152].z - lm[10].z) * w
    mouth_corner_right = abs((lm[61].x - lm[291].x) * w)
    mouth_corner_left = abs((lm[291].x - lm[61].x) * w)
    mouth_pucker = abs((lm[78].x - lm[308].x) * w)
    cheek_puff_right = abs((lm[425].x - lm[436].x) * w)
    cheek_puff_left = abs((lm[205].x - lm[216].x) * w)
    tongue_out = abs((lm[17].y - lm[14].y) * h)
    tongue_up = (lm[17].y - lm[13].y) * h
    tongue_down = (lm[13].y - lm[17].y) * h
    tongue_right = (lm[17].x - lm[13].x) * w
    tongue_left = (lm[13].x - lm[17].x) * w

    return {
        "JawOpen": min(max((jaw_open - 5) / 15, 0), 1),
        "JawRight": min(max(jaw_right / 10, 0), 1),
        "JawLeft": min(max(jaw_left / 10, 0), 1),
        "JawForward": min(max(jaw_forward * 50, 0), 1),
        "MouthCornerPullRight": min(max((mouth_corner_right - 5) / 15, 0), 1),
        "MouthCornerPullLeft": min(max((mouth_corner_left - 5) / 15, 0), 1),
        "MouthPucker": min(max((mouth_pucker - 5) / 20, 0), 1),
        "CheekPuffRight": min(max((cheek_puff_right - 1) / 10, 0), 1),
        "CheekPuffLeft": min(max((cheek_puff_left - 1) / 10, 0), 1),
        "TongueOut": min(max((tongue_out - 2) / 10, 0), 1),
        "TongueUp": min(max(tongue_up / 10, 0), 1),
        "TongueDown": min(max(tongue_down / 10, 0), 1),
        "TongueRight": min(max(tongue_right / 10, 0), 1),
        "TongueLeft": min(max(tongue_left / 10, 0), 1)
    }


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        params = facial_params(lm, w, h)
        y = 30
        for key, val in params.items():
            osc_client.send_message(PARAMS[key], float(val))
            cv2.putText(frame, f"{key}: {val:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 25

    cv2.imshow('VRChat Facial OSC Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
