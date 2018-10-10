using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Serilog;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace ImageSearch
{
    internal static class Program
    {
        private static IConfigurationBuilder BuildConfiguration(IConfigurationBuilder builder, string prefix, string[] args)
        {
            builder
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .AddJsonFile($"appsettings.{Environment.GetEnvironmentVariable(prefix + "ENVIRONMENT") ?? "Production"}.json", optional: true, reloadOnChange: true)
                .AddEnvironmentVariables(prefix);

            if (args != null)
            {
                builder.AddCommandLine(args);
            }
            return builder;
        }

        public static async Task Main(string[] args)
        {
            const string prefix = "IMAGESEARCH_";
            var configuration = BuildConfiguration(new ConfigurationBuilder(), prefix, args).Build();

            Log.Logger = new LoggerConfiguration()
                .ReadFrom.Configuration(configuration)
                .CreateLogger();

            try
            {
                Log.Information("Starting up");

                var builder = new HostBuilder()
                    .ConfigureAppConfiguration(configurationBuilder => BuildConfiguration(configurationBuilder, prefix, args))
                    .UseSerilog()
                    .ConfigureServices((hostContext, services) =>
                    {
                        services.AddOptions();
                        services.Configure<AppConfig>(hostContext.Configuration.GetSection("AppConfig"));
                        services.Configure<ModelConfig>(hostContext.Configuration.GetSection("ModelConfig"));

                        //services.AddSingleton<IHostedService, PrintTextToConsoleService>();
                        services.AddSingleton<IHostedService, ImageProcessorService>();
                    });

                await builder.RunConsoleAsync();
            }
            catch (Exception ex)
            {
                Log.Fatal(ex, "Host terminated unexpectedly: {Exception}", ex.Message);
            }
            finally
            {
                Log.CloseAndFlush();
            }
            if (Debugger.IsAttached)
            {
                Debugger.Break();
            }
        }
    }
}
