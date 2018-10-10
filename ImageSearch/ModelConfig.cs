namespace ImageSearch
{
    public class ModelConfig
    {
        public string ModelToUse { get; set; }
        public string LabelsToUse { get; set; }
        public int ImageSize { get; set; }
        public string[] ImageDirectories { get; set; }
        public string OutputFileName { get; set; }
        public int BatchSize { get; set; }
    }
}
