using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace Utilities
{
    public class Catalog
    {
        // regexes with different new line symbols
        private const string CATALOG_ITEM_PATTERN = @"item {{{0}  name: ""(?<name>.*)""{0}  id: (?<id>\d+){0}  display_name: ""(?<displayName>.*)""{0}}}";
        private static readonly string CATALOG_ITEM_PATTERN_ENV = string.Format(CultureInfo.InvariantCulture, CATALOG_ITEM_PATTERN, Environment.NewLine);
        private static readonly string CATALOG_ITEM_PATTERN_UNIX = string.Format(CultureInfo.InvariantCulture, CATALOG_ITEM_PATTERN, "\n");

        public string FileName { get; }

        private readonly Dictionary<int, CatalogItem> _items = new Dictionary<int, CatalogItem>();

        public Catalog(string fileName)
        {
            FileName = fileName;
        }

        public async Task LoadAsync(CancellationToken ct)
        {
            var text = await File.ReadAllTextAsync(FileName, ct).ConfigureAwait(false);
            var items = _items;
            if (!string.IsNullOrWhiteSpace(text))
            {
                var regex = new Regex(CATALOG_ITEM_PATTERN_ENV);
                var matches = regex.Matches(text);
                if (matches.Count == 0)
                {
                    regex = new Regex(CATALOG_ITEM_PATTERN_UNIX);
                    matches = regex.Matches(text);
                }

                foreach (Match match in matches)
                {
                    var name = match.Groups[1].Value;
                    var id = int.Parse(match.Groups[2].Value);
                    var displayName = match.Groups[3].Value;

                    var item = new CatalogItem()
                    {
                        Id = id,
                        Name = name,
                        DisplayName = displayName
                    };
                    items.Add(item.Id, item);
                }
            }
        }

        public bool TryLookup(int key, out CatalogItem item) => _items.TryGetValue(key, out item);
    }
}