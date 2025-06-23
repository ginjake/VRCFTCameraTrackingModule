using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using VRCFaceTracking;
using VRCFaceTracking.Core;
using VRCFaceTracking.Core.Library;
using VRCFaceTracking.Core.Params.Expressions;

namespace CameraFacialTrackingModule
{
    public class CameraFacialTrackingModule : ExtTrackingModule
    {
        // パラメータ受信用（アイトラッキング・眉毛・鼻パラメータを追加）
        private readonly float[] values = new float[32];

        private UdpClient udp;
        private IPEndPoint ep;

        // 対応表情の順番を定義（perfectsync_sender.pyと同じ順序）
        private static readonly string[] ParamKeys = new[]
        {
            "JawOpen",
            "JawRight",
            "JawLeft",
            "JawForward",
            "MouthCornerPullRight",
            "MouthCornerPullLeft",
            "MouthPucker",
            "CheekPuffRight",
            "CheekPuffLeft",
            "TongueOut",
            "TongueUp",
            "TongueDown",
            "TongueRight",
            "TongueLeft",
            // アイトラッキングパラメータ
            "EyesX",
            "EyesY",
            "LeftEyeLid",
            "RightEyeLid",
            "EyesWiden",
            "EyeSquintRight",
            "EyeSquintLeft",
            "EyeWideLeft",
            "EyeWideRight",
            // 眉毛パラメータ
            "BrowInnerUpLeft",
            "BrowInnerUpRight",
            "BrowLowererLeft",
            "BrowLowererRight",
            "BrowOuterUpLeft",
            "BrowOuterUpRight",
            // 鼻パラメータ
            "NoseSneerLeft",
            "NoseSneerRight"
        };

        public override (bool SupportsEye, bool SupportsExpression) Supported => (true, true);

        public override (bool eyeSuccess, bool expressionSuccess) Initialize(bool eyeAvailable, bool expressionAvailable)
        {
            ModuleInformation.Name = "CameraFacialTrackingModule";
            udp = new UdpClient(9011);
            ep = new IPEndPoint(IPAddress.Any, 0);
            udp.BeginReceive(OnUdp, null);

            // Example of an embedded image stream being referenced as a stream
            var stream = GetType().Assembly.GetManifestResourceStream("CameraFacialTrackingModule.Assets.cameraFacialTrackingModule.png");


            // Setting the stream to be referenced by VRCFaceTracking.
            ModuleInformation.StaticImages =
                stream != null ? new List<Stream> { stream } : ModuleInformation.StaticImages;

            return (true, true);
        }

        public override void Update()
        {
            // 口・顔の表情パラメータ
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawOpen].Weight = values[0];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawRight].Weight = values[1];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawLeft].Weight = values[2];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawForward].Weight = values[3];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.MouthCornerPullRight].Weight = values[4];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.MouthCornerPullLeft].Weight = values[5];
            // UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.MouthPucker].Weight = values[6]; // MouthPuckerは未定義のためコメントアウト
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.CheekPuffRight].Weight = values[7];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.CheekPuffLeft].Weight = values[8];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueOut].Weight = values[9];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueUp].Weight = values[10];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueDown].Weight = values[11];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueRight].Weight = values[12];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueLeft].Weight = values[13];
            
            // アイトラッキングパラメータ
            UnifiedTracking.Data.Eye.Left.Gaze.x = values[14]; // EyesX
            UnifiedTracking.Data.Eye.Left.Gaze.y = values[15]; // EyesY
            UnifiedTracking.Data.Eye.Right.Gaze.x = values[14]; // EyesX
            UnifiedTracking.Data.Eye.Right.Gaze.y = values[15]; // EyesY
            // LeftEyeLidとRightEyeLidは既に0-1で閉じた度合いなので、Opennessには1から引く
            UnifiedTracking.Data.Eye.Left.Openness = 1.0f - values[16]; // LeftEyeLid (0=開, 1=閉)
            UnifiedTracking.Data.Eye.Right.Openness = 1.0f - values[17]; // RightEyeLid (0=開, 1=閉)
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeWideLeft].Weight = values[21];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeWideRight].Weight = values[22];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeSquintLeft].Weight = values[20];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.EyeSquintRight].Weight = values[19];
            
            // 眉毛パラメータ
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.BrowInnerUpLeft].Weight = values[23];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.BrowInnerUpRight].Weight = values[24];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.BrowLowererLeft].Weight = values[25];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.BrowLowererRight].Weight = values[26];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.BrowOuterUpLeft].Weight = values[27];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.BrowOuterUpRight].Weight = values[28];
            
            // 鼻パラメータ
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.NoseSneerLeft].Weight = values[29];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.NoseSneerRight].Weight = values[30];

            Thread.Sleep(4);
        }

        public override void Teardown()
        {
            udp?.Close();
        }

        private void OnUdp(IAsyncResult ar)
        {
            try
            {
                var bytes = udp.EndReceive(ar, ref ep);
                ParseOsc(bytes);
                udp.BeginReceive(OnUdp, null);
            }
            catch { }
        }

        private void ParseOsc(byte[] bytes)
        {
            string addr = ParseOscAddress(bytes);
            float? val = ParseOscFloat(bytes);
            if (addr == null || val == null) return;

            for (int i = 0; i < ParamKeys.Length; i++)
            {
                if (addr == $"/avatar/parameters/{ParamKeys[i]}")
                    values[i] = Math.Clamp(val.Value, 0f, 1f);
            }
        }

        private string ParseOscAddress(byte[] bytes)
        {
            int i = 0;
            while (i < bytes.Length && bytes[i] != 0) i++;
            return System.Text.Encoding.ASCII.GetString(bytes, 0, i);
        }

        private float? ParseOscFloat(byte[] bytes)
        {
            int pos = Array.IndexOf(bytes, (byte)',');
            if (pos < 0 || pos + 2 >= bytes.Length) return null;
            if (bytes[pos + 1] != (byte)'f') return null;
            int floatStart = (pos + 4);
            if (floatStart + 4 > bytes.Length) return null;
            byte[] floatBytes = new byte[4];
            Array.Copy(bytes, floatStart, floatBytes, 0, 4);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(floatBytes);
            return BitConverter.ToSingle(floatBytes, 0);
        }
    }
}
