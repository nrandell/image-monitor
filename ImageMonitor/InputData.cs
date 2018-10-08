using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;

namespace ImageMonitor
{
    public class InputData : IDisposable
    {
        public TFTensor Tensor { get; }
        public Mat[] TensorMats { get; }

        public InputData(int imageSize, int batchSize)
        {
            var singleImageSize = imageSize * imageSize * 3;
            Tensor = new TFTensor(TFDataType.UInt8, new long[] { batchSize, imageSize, imageSize, 3 }, batchSize * singleImageSize);

            var ptr = Tensor.Data;
            var mats = new Mat[batchSize];
            for (var i = 0; i < batchSize; i++)
            {
                mats[i] = new Mat(imageSize, imageSize, MatType.CV_8UC3, ptr);
                ptr += singleImageSize;
            }
            TensorMats = mats;
        }

        public void Dispose()
        {
            for (var i = 0; i < TensorMats.Length; i++)
            {
                TensorMats[i].Dispose();
            }
            Tensor.Dispose();
        }
    }
}
