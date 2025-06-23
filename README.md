# CameraFacialTrackingModule 導入手順

## 1. VRCFaceTracking（VRCFT）のインストール

[VRCFaceTracking (Steam)](https://store.steampowered.com/app/3329480/VRCFaceTracking/?l=japanese) からVRCFaceTrackingをインストール。

---

## 2. CameraFacialTrackingModuleのビルド

1. `CameraFacialTrackingModule.sln` を Visual Studio 2022 で開く。
2. `CameraFacialTrackingModule.cs` のポート番号（例：`udp = new UdpClient(9011);`）を必要に応じて変更。
3. ビルドして `CameraFacialTrackingModule.dll` を生成。
4. 生成した `CameraFacialTrackingModule.dll` を `%APPDATA%\VRCFaceTracking\CustomLibs` に移動。

---

## 3. 必要なPythonパッケージのインストールとスクリプトの実行

CameraSender内の `perfectsync_sender.py` のIPとポートをいい感じに編集する。  
PowerShellなどで以下のコマンドを実行。

```powershell
pip install opencv-python mediapipe python-osc
```

次に、cameraSender ディレクトリ内で以下を実行。

```
powershell
python perfectsync_sender.py
```