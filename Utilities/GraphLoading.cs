using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorFlow;

namespace Utilities
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
            var model = await File.ReadAllBytesAsync(FrozenInferenceGraphFileName, ct).ConfigureAwait(false);
            var graph = new TFGraph();
            graph.Import(model);
            return graph;
        }
    }
}
