using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorFlow;

namespace Utilities
{
    public static class ImageUtilities
    {
        public static async Task<(TFTensor, Exception[])> CreateTensorFromImageFiles(IReadOnlyList<string> fileNames, TFDataType destinationDataType, int size, CancellationToken ct)
        {
            Exception[] exceptions = null;

            var readingTasks = fileNames.Select(fileName => File.ReadAllBytesAsync(fileName, ct)).ToList();
            try
            {
                await Task.WhenAll(readingTasks).ConfigureAwait(false);
            }
            catch (Exception)
            {
                exceptions = new Exception[fileNames.Count];
#pragma warning disable ERP022 // Catching everything considered harmful.
            }
#pragma warning restore ERP022 // Catching everything considered harmful.

            using (var graph = new TFGraph())
            {
                var batchIndex = 0;
                var batchTensors = new TFTensor[fileNames.Count];
                var batchInputs = new TFOutput[fileNames.Count];
                var expandeds = new TFOutput[fileNames.Count];

                for (var i = 0; i < fileNames.Count; i++)
                {
                    var readingTask = readingTasks[i];
                    if (readingTask.IsFaulted)
                    {
                        exceptions[i] = readingTask.Exception;
                    }
                    else
                    {
                        var contents = readingTask.Result;
                        batchTensors[batchIndex] = TFTensor.CreateString(contents);
                        var batchInput = graph.Placeholder(TFDataType.String);
                        batchInputs[batchIndex] = batchInput;
                        var decoded = graph.DecodeJpeg(contents: batchInput, channels: 3, operName: $"Decode_jpeg_image_{batchIndex}");
                        expandeds[batchIndex] = graph.ExpandDims(input: decoded, dim: graph.Const(0), operName: $"Expand_dimensions_{batchIndex}");

                        batchIndex++;
                    }
                }

                if (batchIndex != fileNames.Count)
                {
                    Array.Resize(ref batchTensors, batchIndex);
                    Array.Resize(ref batchInputs, batchIndex);
                    Array.Resize(ref expandeds, batchIndex);
                }

                var expandedBatch = graph.Concat(concat_dim: graph.Const(0), values: expandeds, operName: "Batch_images");
                var resized = graph.ResizeBilinear(images: expandedBatch, size: graph.Const(new[] { size, size }), operName: "Resize_images");
                var output = graph.Cast(x: resized, DstT: destinationDataType, operName: "Cast_results");

                using (var session = new TFSession(graph))
                {
                    var normalized = session.Run(
                        inputs: batchInputs,
                        inputValues: batchTensors,
                        outputs: new[] { output }
                    );
                    return (normalized[0], exceptions);
                }
            }
        }

        public static async Task<(TFTensor, TFOutput)> CreateTensorFromImageFile(string fileName, TFDataType destinationDataType, int size, CancellationToken ct)
        {
            var contents = await File.ReadAllBytesAsync(fileName, ct).ConfigureAwait(false);
            var tensor = TFTensor.CreateString(contents);

            using (var graph = CreateGraphToNormalizeImage(out var input, out var output, destinationDataType, size, 0))
            using (var session = new TFSession(graph))
            {
                var normalized = session.Run(
                    inputs: new[] { input },
                    inputValues: new[] { tensor },
                    outputs: new[] { output });
                return (normalized[0], output);
            }
        }

        public static TFGraph CreateGraphToNormalizeImage(out TFOutput input, out TFOutput output, TFDataType destinationDataType, int size, int batchIndex)
        {
            var graph = new TFGraph();
            input = graph.Placeholder(TFDataType.String);
            var decoded = graph.DecodeJpeg(contents: input, channels: 3);
            var expanded = graph.ExpandDims(input: decoded, dim: graph.Const(batchIndex));
            var targetSize = graph.Const(new int[] { size, size }, "resize");
            var resized = graph.ResizeBilinear(expanded, size: targetSize);
            output = graph.Cast(resized, destinationDataType);

            return graph;
        }
    }
}
