using System;
using System.Collections.Generic;
using System.Text;

namespace Utilities
{
    public class Detection
    {
        public string Label { get; set; }
        public int Id { get; set; }
        public float Score { get; set; }
        public float Left { get; set; }
        public float Top { get; set; }
        public float Right { get; set; }
        public float Bottom { get; set; }
    }
}
