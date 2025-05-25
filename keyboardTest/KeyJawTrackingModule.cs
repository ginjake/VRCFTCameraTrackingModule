using System;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using VRCFaceTracking;
using VRCFaceTracking.Core;
using VRCFaceTracking.Core.Library;
using VRCFaceTracking.Core.Params.Expressions;

namespace KeyJawTrackingModule
{
    public class KeyJawTrackingModule : ExtTrackingModule
    {
        [DllImport("user32.dll")]
        private static extern short GetAsyncKeyState(int vKey);
        private const int VK_O = 0x4F; // 'O'
        private const int VK_P = 0x50; // 'P'
        private const int VK_U = 0x55; // 'U'
        private const int VK_I = 0x49; // 'I'

        // 現在の顎の開き具合を保持
        private float currentJaw = 0f;
        // U/Iキーで増減する量
        private const float Step = 0.02f;

        public override (bool SupportsEye, bool SupportsExpression) Supported => (false, true);

        public override (bool eyeSuccess, bool expressionSuccess) Initialize(bool eyeAvailable, bool expressionAvailable)
        {
            ModuleInformation.Name = "KeyJawTrackingModule";
            //var stream = GetType().Assembly.GetManifestResourceStream("keyboardTest.Assets.test.png");
            //ModuleInformation.StaticImages = stream != null ? new List<Stream> { stream } : ModuleInformation.StaticImages;

            return (true, true);

        }

        public override void Update()
        {
            if (Status != ModuleState.Active)
            {
                Thread.Sleep(5);
                //return;
            }

            // キー状態を取得
            bool downO = (GetAsyncKeyState(VK_O) & 0x8000) != 0;
            bool downP = (GetAsyncKeyState(VK_P) & 0x8000) != 0;
            bool downU = (GetAsyncKeyState(VK_U) & 0x8000) != 0;
            bool downI = (GetAsyncKeyState(VK_I) & 0x8000) != 0;

            // O/P が優先：即時開閉
            if (downO)
            {
                currentJaw = 1.0f;
            }
            else if (downP)
            {
                currentJaw = 0.0f;
            }
            else
            {
                // U/I で徐々に増減
                if (downU)
                {
                    currentJaw += Step;
                }
                else if (downI)
                {
                    currentJaw -= Step;
                }
                // 範囲を [0,1] にクランプ
                currentJaw = Math.Clamp(currentJaw, 0f, 1f);
            }

            // JawOpen に反映
            /*
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawOpen].Weight = currentJaw;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawRight].Weight = 1.0f;
            UnifiedTracking.Data.Eye.Left.Openness = 0.3f ;
            */

            UnifiedTracking.Data.Eye.Left.Gaze.x = currentJaw;
            UnifiedTracking.Data.Eye.Left.Gaze.y = currentJaw;
            UnifiedTracking.Data.Eye.Right.Gaze.x = currentJaw;
            UnifiedTracking.Data.Eye.Right.Gaze.y = currentJaw;

            // Eye Openness
            UnifiedTracking.Data.Eye.Left.Openness = 0.3f;
            UnifiedTracking.Data.Eye.Right.Openness = currentJaw;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawOpen].Weight = currentJaw;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.MouthClosed].Weight = 1.0f-currentJaw;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.JawRight].Weight = 0;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.CheekPuffLeft].Weight = 1.0f;
            UnifiedTracking.Data.Shapes[(int)UnifiedExpressions.CheekSquintLeft].Weight = 1.0f;



            Thread.Sleep(4);
        }

        public override void Teardown()
        {
            // 特になし
        }
    }
}