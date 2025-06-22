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
        private float jawA = 0f;
        private float jawI = 0f;
        private float jawU = 0f;
        private float jawE = 0f;
        private float jawO = 0f;

        private UdpClient udp;
        private IPEndPoint ep;

        public override (bool SupportsEye, bool SupportsExpression) Supported => (false, true);

        public override (bool eyeSuccess, bool expressionSuccess) Initialize(bool eyeAvailable, bool expressionAvailable)
        {
            ModuleInformation.Name = "CameraFacialTrackingModule";
            udp = new UdpClient(9011);
            ep = new IPEndPoint(IPAddress.Any, 0);
            udp.BeginReceive(OnUdp, null);
            return (true, true);
        }

        public override void Update()
        {
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawOpen].Weight = jawA;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawClench].Weight = jawI;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawForward].Weight = jawU;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawMandibleRaise].Weight = jawE;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawBackward].Weight = jawO;

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

            if (addr == "/avatar/parameters/Voice_A") jawA = Math.Clamp(val.Value, 0f, 1f);
            if (addr == "/avatar/parameters/Voice_I") jawI = Math.Clamp(val.Value, 0f, 1f);
            if (addr == "/avatar/parameters/Voice_U") jawU = Math.Clamp(val.Value, 0f, 1f);
            if (addr == "/avatar/parameters/Voice_E") jawE = Math.Clamp(val.Value, 0f, 1f);
            if (addr == "/avatar/parameters/Voice_O") jawO = Math.Clamp(val.Value, 0f, 1f);
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
