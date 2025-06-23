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
        // パラメータ受信用
        private readonly float[] values = new float[14];

        private UdpClient udp;
        private IPEndPoint ep;

        // 対応表情の順番を定義
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
            "TongueLeft"
        };

        public override (bool SupportsEye, bool SupportsExpression) Supported => (false, true);

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
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawOpen].Weight = values[0];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawRight].Weight = values[1];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawLeft].Weight = values[2];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawForward].Weight = values[3];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.MouthCornerPullRight].Weight = values[4];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.MouthCornerPullLeft].Weight = values[5];
//            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.MouthPucker].Weight = values[6];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.CheekPuffRight].Weight = values[7];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.CheekPuffLeft].Weight = values[8];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueOut].Weight = values[9];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueUp].Weight = values[10];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueDown].Weight = values[11];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueRight].Weight = values[12];
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.TongueLeft].Weight = values[13];

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
