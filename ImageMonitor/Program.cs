using OpenCvSharp;
using Serilog;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorFlow;

namespace ImageMonitor
{
    internal static class Program
    {
        //private const string ModelDirectory = @"c:\users\nick\downloads\faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28";
        //private const string CatalogFile = @"c:\users\nick\Downloads\tensor\oid_bbox_trainable_label_map.pbtxt";
        // private const int ImageSize = 600;
        // private const double MinScore = 0.5;

        // private const string ModelDirectory = @"c:\users\nick\downloads\tensor\ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03";
        // private const string CatalogFile = @"c:\users\nick\Downloads\tensor\mscoco_label_map.pbtxt";
        // private const int ImageSize = 224;

        //private const string ModelDirectory = @"c:\users\nick\downloads\tensor\ssd_mobilenet_v2_coco_2018_03_29";
        //private const string CatalogFile = @"c:\users\nick\Downloads\tensor\mscoco_label_map.pbtxt";
        private const string ModelDirectory = ".";
        private const string CatalogFile = "label_map.pbtxt";
        private const int ImageSize = 300;

        private const int BatchSize = 30;

        private const double MinScore = 0.0;

        public static async Task Main(string[] args)
        {
            Log.Logger = new LoggerConfiguration()
                .WriteTo.Console()
                .WriteTo.File(@"c:\temp\imgs.txt")
                .CreateLogger();
            Log.Information("Starting up");
            using (var cts = new CancellationTokenSource())
            {
                var cancelled = false;
                Console.CancelKeyPress += (_, ev) =>
                {
                    if (!cancelled)
                    {
                        ev.Cancel = true;
                        cancelled = true;
                    }
                    cts.Cancel();
                };

                try
                {
                    await RunAsync(args, cts.Token);
                }
                catch (OperationCanceledException)
                {
                    Log.Debug("Cancelled");
                }
                catch (Exception ex)
                {
                    Log.Fatal(ex, "Error running: {Exception}", ex.Message);
                }
            }
            if (Debugger.IsAttached)
            {
                Debugger.Break();
            }
            Log.Information("Shutting down");
        }

        private static async Task RunAsync(string[] args, CancellationToken ct)
        {
            var loading = new GraphLoading(Path.Combine(ModelDirectory, "frozen_inference_graph.pb"));
            var catalog = new Catalog(CatalogFile);

            await catalog.LoadAsync(ct);
            var graph = await loading.LoadGraphAsync(ct);

            using (var session = new TFSession(graph))
            using (var input = new InputData(ImageSize, BatchSize))
            {

                foreach (var directoryName in args)
                {
                    var imageFiles = Directory.EnumerateFiles(directoryName, "*.jpg", SearchOption.AllDirectories);
                    var batch = new List<string>(BatchSize);
                    foreach (var imageFile in imageFiles)
                    {
                        batch.Add(Path.Combine(directoryName, imageFile));
                        if (batch.Count == BatchSize)
                        {
                            await TryProcessImage(catalog, graph, session, input, batch, ct);
                            batch.Clear();
                        }
                    }
                    if (batch.Count > 0)
                    {
                        await TryProcessImage(catalog, graph, session, input, batch, ct);
                    }
                }
            }
        }

        private static async Task TryProcessImage(Catalog catalog, TFGraph graph, TFSession session, InputData input,
            IReadOnlyList<string> imageFiles, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                await ProcessBatch(catalog, graph, session, input, imageFiles, ct);
            }
            catch (Exception ex)
            {
                Log.Warning(ex, "Error processing images {Image}: {Exception}", imageFiles, ex.Message);
            }
        }

        private static async Task TryLoadImageData(string imageFileName, Mat tensorMat, CancellationToken ct)
        {
            var imageData = await File.ReadAllBytesAsync(imageFileName, ct);
            using (var imageMat = Mat.FromImageData(imageData))
            using (var resized = imageMat.Resize(new Size(ImageSize, ImageSize), interpolation: InterpolationFlags.Linear))
            using (var colored = resized.CvtColor(ColorConversionCodes.BGR2RGB))
            {
                colored.ConvertTo(tensorMat, MatType.CV_8UC3);
            }
        }


        private static async Task ProcessBatch(Catalog catalog, TFGraph graph, TFSession session, InputData input, IReadOnlyList<string> imageFiles, CancellationToken ct)
        {
            var tasks = new Task[imageFiles.Count];
            for (var i = 0; i < tasks.Length; i++)
            {
                tasks[i] = TryLoadImageData(imageFiles[i], input.TensorMats[i], ct);
            }
            await Task.WhenAll(tasks);
            var loadedFiles = new List<string>(imageFiles.Count);

            var runner = session.GetRunner();
            var tensorIndex = loadedFiles.Count;
            runner
                .AddInput(graph["image_tensor"][tensorIndex], input.Tensor)
                .Fetch(
                    graph["detection_boxes"][tensorIndex],
                    graph["detection_scores"][tensorIndex],
                    graph["detection_classes"][tensorIndex],
                    graph["num_detections"][tensorIndex]
                );

            for (var i = 0; i < tasks.Length; i++)
            {
                if (tasks[i].IsCompletedSuccessfully)
                {
                    loadedFiles.Add(imageFiles[i]);
                }
            }
            if (loadedFiles.Count > 0)
            {
                var output = runner.Run();
                ProcessResults(catalog, output, loadedFiles);
            }

            //using (var imageMat = Mat.FromImageData(imageData))
            //using (var resized = imageMat.Resize(new Size(ImageSize, ImageSize), interpolation: InterpolationFlags.Linear))
            //using (var colored = resized.CvtColor(ColorConversionCodes.BGR2RGB))
            //{
            //    colored.ConvertTo(tensorMat, MatType.CV_8UC3);

            //    var runner = session.GetRunner();
            //    runner
            //        .AddInput(graph["image_tensor"][0], tensor)
            //        .Fetch(
            //            graph["detection_boxes"][0],
            //            graph["detection_scores"][0],
            //            graph["detection_classes"][0],
            //            graph["num_detections"][0]
            //        );

            //    var output = runner.Run();
            //    ProcessResults(catalog, output, imageFileName);
            //}
        }

        private static void ProcessResults(Catalog catalog, TFTensor[] output, IReadOnlyList<string> imageFileNames)
        {
            var boxes = (float[,,])output[0].GetValue(jagged: false);
            var scores = (float[,])output[1].GetValue(jagged: false);
            var classes = (float[,])output[2].GetValue(jagged: false);
            var num = (float[])output[3].GetValue(jagged: false);

            var x = boxes.GetLength(0);
            var y = boxes.GetLength(1);

            for (var fileIndex = 0; fileIndex < x; fileIndex++)
            {
                var found = false;
                for (var detectionIndex = 0; detectionIndex < y; detectionIndex++)
                {
                    var score = scores[fileIndex, detectionIndex];
                    if (score > MinScore)
                    {
                        found = true;
                        var classValue = Convert.ToInt32(classes[fileIndex, detectionIndex]);

                        if (catalog.TryLookup(classValue, out var item))
                        {
                            Log.Information("{File}: Found {Label} of {Score} at {Detection}", imageFileNames[fileIndex], item.DisplayName, score, detectionIndex);
                        }
                        else
                        {
                            Log.Information("{File}: Found {Id} (no label) of {Score} at {Detection}", imageFileNames[fileIndex], classValue, score, detectionIndex);
                        }
                    }
                }
                if (!found)
                {
                    Log.Information("{File} found nothing", imageFileNames[fileIndex]);
                }
            }
        }
    }
}
