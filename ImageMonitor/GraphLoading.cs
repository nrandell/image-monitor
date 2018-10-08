using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorFlow;

namespace ImageMonitor
{
    public class GraphLoading
    {
        public string FrozenInferenceGraphFileName { get; }

        public GraphLoading(string frozenInferenceGraphFileName)
        {
            FrozenInferenceGraphFileName = frozenInferenceGraphFileName;
        }

        public async Task<TFGraph> LoadGraphAsync(CancellationToken ct)
        {
            var model = await File.ReadAllBytesAsync(FrozenInferenceGraphFileName, ct);
            var graph = new TFGraph();
            graph.Import(model);
            return graph;
        }
    }
}
