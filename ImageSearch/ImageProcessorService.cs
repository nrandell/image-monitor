using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorFlow;
using Utilities;

namespace ImageSearch
{
    public class ImageProcessorService : IHostedService, IDisposable
    {
        public ILogger Logger { get; }
        public IOptions<ModelConfig> ModelConfig { get; }
        public IApplicationLifetime Lifetime { get; }

        public ImageProcessorService(ILogger<ImageProcessorService> logger, IOptions<ModelConfig> modelConfig, IApplicationLifetime lifetime)
        {
            Logger = logger;
            ModelConfig = modelConfig;
            Lifetime = lifetime;
        }

        private Task _runner;

        public Task StartAsync(CancellationToken cancellationToken)
        {
            var modelConfig = ModelConfig.Value;
            if (string.IsNullOrWhiteSpace(modelConfig.OutputFileName))
            {
                throw new ArgumentException("No output file name");
            }
            if ((modelConfig.ImageDirectories == null) || (modelConfig.ImageDirectories.Length == 0))
            {
                throw new ArgumentException("No image directories");
            }
            if (modelConfig.ImageSize == 0)
            {
                throw new ArgumentException("No image size");
            }
            if (modelConfig.BatchSize == 0)
            {
                throw new ArgumentException("No batch size");
            }
            _runner = Task.Factory.StartNew(RunAsync, TaskCreationOptions.LongRunning).Unwrap();
            return Task.CompletedTask;
        }

        private async Task RunAsync()
        {
            var modelConfig = ModelConfig.Value;
            var cancellationToken = Lifetime.ApplicationStopped;

            var loading = new GraphLoading(modelConfig.ModelToUse);
            var catalog = new Catalog(modelConfig.LabelsToUse);
            await catalog.LoadAsync(cancellationToken);

            using (var resultBuilder = new ResultBuilder(modelConfig.OutputFileName, catalog))
            using (var graph = await loading.LoadGraphAsync(cancellationToken))
            using (var session = new TFSession(graph))
            {
                var batch = new List<string>(modelConfig.BatchSize);

                foreach (var imageDirectory in modelConfig.ImageDirectories)
                {
                    foreach (var inputFile in Directory.EnumerateFiles(imageDirectory, "*.jpg", SearchOption.AllDirectories))
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        batch.Add(Path.Combine(imageDirectory, inputFile));
                        if (batch.Count == batch.Capacity)
                        {
                            await ProcessBatch(resultBuilder, graph, session, batch, cancellationToken);
                            batch.Clear();
                        }
                    }
                }
            }
            Logger.LogInformation("All images processed");
            Lifetime.StopApplication();
        }

        private async Task ProcessBatch(ResultBuilder resultBuilder, TFGraph graph, TFSession session, IReadOnlyList<string> imageFileNames, CancellationToken cancellationToken)
        {
            var (tensor, exceptions) = await ImageUtilities.CreateTensorFromImageFiles(imageFileNames, TFDataType.UInt8, ModelConfig.Value.ImageSize, cancellationToken);

            IReadOnlyList<string> usedFileNames;
            if (exceptions == null)
            {
                usedFileNames = imageFileNames;
            }
            else
            {
                var used = new string[imageFileNames.Count];
                var index = 0;
                for (var i = 0; i < imageFileNames.Count; i++)
                {
                    var imageFileName = imageFileNames[i];
                    var ex = exceptions[i];
                    if (ex == null)
                    {
                        used[index++] = imageFileName;
                    }
                    else
                    {
                        Logger.LogWarning(ex, "Error reading {ImageFileName}: {Exception}", imageFileName, ex.Message);
                    }
                }
                Array.Resize(ref used, index);
                usedFileNames = used;
            }

            var runner = session.GetRunner();
            runner.AddInput(graph["image_tensor"][0], tensor)
                .Fetch(
                    graph["detection_boxes"][0],
                    graph["detection_scores"][0],
                    graph["detection_classes"][0],
                    graph["num_detections"][0]);
            var output = runner.Run();

            await resultBuilder.WriteResultsAsync(usedFileNames, output, Logger, cancellationToken);
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation("Stopping");
            return Task.CompletedTask;
        }

        public void Dispose()
        {
        }
    }
}
