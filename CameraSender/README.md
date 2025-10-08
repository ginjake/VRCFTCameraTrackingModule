# VR Facial Tracking (MediaPipe版 - perfectsync_sender)

VRゴーグル装着時の口周辺の表情をトラッキングし、VRChat等へOSCで送信するツールです。
顔の下半分（鼻、口、顎）のみをカメラに映すことで動作します。

## 起動方法

1.  ターミナルを開き、以下のコマンドで仮想環境を有効にします。
    ```powerShell
    vr_mediapipe_env\Scripts\activate
    ```
2.  次に、以下のコマンドでトラッキング用のGUIを起動します。
    ```powerShell
    python perfectsync_sender.py
    ```
3.  表示されたウィンドウで、使用するカメラを選択し「スタート」ボタンを押してください。

## 操作方法

プレビューウィンドウを選択した状態で、以下のキー操作が可能です。

-   `q` または `ESC`: トラッキングを終了します。
-   `c` または `r`: 表情のキャリブレーションをリセットします。
-   ウィンドウ上のボタンでもキャリブレーションが可能です。

## 技術情報

-   **トラッキングライブラリ:** MediaPipe FaceMesh
-   **OSC送信先:** 127.0.0.1:9011
-   **対応パラメータ:** JawOpen, JawRight, JawLeft, MouthCornerPullRight, MouthCornerPullLeft, MouthPucker など (スクリプト内の `PARAMS` 辞書を参照)

## 依存関係のインストール

初回のみ、以下のコマンドを実行して必要なライブラリをインストールしてください。

```powerShell
vr_mediapipe_env\Scripts\activate
pip install -r requirements.txt
```