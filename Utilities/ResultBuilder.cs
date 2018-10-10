using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorFlow;

namespace Utilities
{
    public class ResultBuilder : IDisposable
    {
        private static readonly JsonSerializerSettings _jsonSerializerSettings =
            new JsonSerializerSettings
            {
                Formatting = Formatting.Indented,
                ContractResolver = new CamelCasePropertyNamesContractResolver()
            };

        private static readonly JsonSerializer _jsonSerializer = JsonSerializer.Create(_jsonSerializerSettings);

        private readonly JsonTextWriter _writer;
        public Catalog Catalog { get; }

        public ResultBuilder(string outputFileName, Catalog catalog)
        {
            Catalog = catalog;
            var stream = new FileStream(outputFileName, FileMode.Create, FileAccess.Write, FileShare.Read, 8192, true);
            var streamWriter = new StreamWriter(stream);
            _writer = new JsonTextWriter(streamWriter)
            {
                AutoCompleteOnClose = true,
                CloseOutput = true,
                Formatting = Formatting.Indented,
            };
            _writer.WriteStartArray();
        }

        public void Dispose()
        {
            _writer.Close();
        }

        public async Task WriteResultsAsync(IReadOnlyList<string> imageFileNames, TFTensor[] output, ILogger logger, CancellationToken ct)
        {
            var boxes = (float[,,])output[0].GetValue(jagged: false);
            var scores = (float[,])output[1].GetValue(jagged: false);
            var classes = (float[,])output[2].GetValue(jagged: false);
            var num = (float[])output[3].GetValue(jagged: false);

            var fileIndex = 0;
            foreach (var imageFileName in imageFileNames)
            {
                var numberOfDetections = (int)num[fileIndex];
                var detections = new Detection[numberOfDetections];
                for (var i = 0; i < numberOfDetections; i++)
                {
                    var score = scores[fileIndex, i];
                    var id = (int)classes[fileIndex, i];
                    string label = "unknown";
                    if (Catalog.TryLookup(id, out var item))
                    {
                        label = item.DisplayName;
                    }

                    var detection = new Detection
                    {
                        Score = score,
                        Id = id,
                        Label = label,
                        Top = boxes[fileIndex, i, 0],
                        Left = boxes[fileIndex, i, 1],
                        Bottom = boxes[fileIndex, i, 2],
                        Right = boxes[fileIndex, i, 3]
                    };
                    detections[i] = detection;
                }
                var result = new Results { Detections = detections, FileName = imageFileName };
                var jo = JObject.FromObject(result, _jsonSerializer);
                await jo.WriteToAsync(_writer, ct).ConfigureAwait(false);

                var labels = string.Join(',', detections.Select(d => d.Label));
                logger.LogInformation("{ImageFileName} = {Labels}", imageFileName, labels);
                fileIndex++;
            }
        }
    }
}
