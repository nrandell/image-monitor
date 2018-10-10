using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ImageSearch
{
    internal class PrintTextToConsoleService : IHostedService, IDisposable
    {
        private ILogger Logger { get; }
        private IOptions<AppConfig> AppConfig { get; }
        private Timer _timer;

        public PrintTextToConsoleService(ILogger<PrintTextToConsoleService> logger, IOptions<AppConfig> appConfig)
        {
            Logger = logger;
            AppConfig = appConfig;
        }

        public void Dispose()
        {
            _timer?.Dispose();
        }

        public Task StartAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation("Starting...");
            _timer = new Timer(DoWork, null, TimeSpan.Zero, TimeSpan.FromSeconds(5));
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation("Stopping");
            _timer?.Change(Timeout.Infinite, 0);
            return Task.CompletedTask;
        }

        private void DoWork(object state)
        {
            Logger.LogInformation($"Background work with text: {AppConfig.Value.TextToPrint}");
        }
    }
}
