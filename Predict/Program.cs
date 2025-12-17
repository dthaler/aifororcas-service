// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT

using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.Logging;

// Simple config holder matching keys used by the Python code.
// The loader below supports a simple YAML subset (flat key: value pairs) or JSON.
sealed class Config
{
    public string? ModelType { get; init; }
    public string? ModelPath { get; init; }
    public double? ModelLocalThreshold { get; init; }
    public double? ModelGlobalThreshold { get; init; }
    public string? ModelName { get; init; }

    public bool UploadToAzure { get; init; }
    public bool DeleteLocalWavs { get; init; }
    public string? LogResults { get; init; }

    public string? HlsStreamType { get; init; }
    public int HlsPollingInterval { get; init; }
    public string? HlsHydrophoneId { get; init; }

    public string? HlsStartTimePst { get; init; }
    public string? HlsEndTimePst { get; init; }

    // any additional keys can be read from Extra if needed
    public Dictionary<string, object>? Extra { get; init; }
}

public class Program
{
    static Config LoadConfig(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        var text = File.ReadAllText(path);

        if (ext == ".json")
        {
            var doc = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(text);
            return MapFromDictionary(doc);
        }

        // try a tiny YAML-ish parser for simple flat mappings (key: value).
        // This intentionally only supports the simple config shapes typically used by this project.
        var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        using (var sr = new StringReader(text))
        {
            string? line;
            while ((line = sr.ReadLine()) != null)
            {
                line = line.Trim();
                if (string.IsNullOrEmpty(line) || line.StartsWith("#"))
                    continue;

                var idx = line.IndexOf(':');
                if (idx <= 0)
                    continue;

                var key = line.Substring(0, idx).Trim();
                var value = line.Substring(idx + 1).Trim();

                // remove surrounding quotes if present
                if (value.StartsWith("\"") && value.EndsWith("\"") || value.StartsWith("'") && value.EndsWith("'"))
                {
                    value = value[1..^1];
                }

                dict[key] = value;
            }
        }

        return MapFromStringDictionary(dict);
    }

    static Config MapFromDictionary(Dictionary<string, JsonElement>? d)
    {
        d ??= new();
        string? GetString(string k) => d.TryGetValue(k, out var e) && e.ValueKind != JsonValueKind.Null ? e.ToString().Trim('"') : null;
        double? GetDouble(string k) => d.TryGetValue(k, out var e) && e.TryGetDouble(out var dv) ? dv : (double?)null;
        int GetInt(string k, int fallback = 0) => d.TryGetValue(k, out var e) && e.TryGetInt32(out var iv) ? iv : fallback;
        bool GetBool(string k) => d.TryGetValue(k, out var e) && e.ValueKind == JsonValueKind.True;

        return new Config
        {
            ModelType = GetString("model_type"),
            ModelPath = GetString("model_path"),
            ModelLocalThreshold = GetDouble("model_local_threshold"),
            ModelGlobalThreshold = GetDouble("model_global_threshold"),
            ModelName = GetString("model_name"),
            UploadToAzure = GetBool("upload_to_azure"),
            DeleteLocalWavs = GetBool("delete_local_wavs"),
            LogResults = GetString("log_results"),
            HlsStreamType = GetString("hls_stream_type"),
            HlsPollingInterval = GetInt("hls_polling_interval"),
            HlsHydrophoneId = GetString("hls_hydrophone_id"),
            HlsStartTimePst = GetString("hls_start_time_pst"),
            HlsEndTimePst = GetString("hls_end_time_pst"),
            Extra = null
        };
    }

    static Config MapFromStringDictionary(Dictionary<string, string> d)
    {
        string? GetString(string k) => d.TryGetValue(k, out var v) ? v : null;
        double? GetDouble(string k) => d.TryGetValue(k, out var v) && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var dv) ? dv : (double?)null;
        int GetInt(string k, int fallback = 0) => d.TryGetValue(k, out var v) && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out var iv) ? iv : fallback;
        bool GetBool(string k) => d.TryGetValue(k, out var v) && (v.Equals("true", StringComparison.OrdinalIgnoreCase) || v == "1" || v.Equals("yes", StringComparison.OrdinalIgnoreCase));

        // keep any leftovers
        var used = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
    {
        "model_type","model_path","model_local_threshold","model_global_threshold","model_name",
        "upload_to_azure","delete_local_wavs","log_results",
        "hls_stream_type","hls_polling_interval","hls_hydrophone_id","hls_start_time_pst","hls_end_time_pst"
    };
        var extra = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        foreach (var kv in d)
        {
            if (!used.Contains(kv.Key))
                extra[kv.Key] = kv.Value;
        }

        return new Config
        {
            ModelType = GetString("model_type"),
            ModelPath = GetString("model_path"),
            ModelLocalThreshold = GetDouble("model_local_threshold"),
            ModelGlobalThreshold = GetDouble("model_global_threshold"),
            ModelName = GetString("model_name"),
            UploadToAzure = GetBool("upload_to_azure"),
            DeleteLocalWavs = GetBool("delete_local_wavs"),
            LogResults = GetString("log_results"),
            HlsStreamType = GetString("hls_stream_type"),
            HlsPollingInterval = GetInt("hls_polling_interval", 60),
            HlsHydrophoneId = GetString("hls_hydrophone_id"),
            HlsStartTimePst = GetString("hls_start_time_pst"),
            HlsEndTimePst = GetString("hls_end_time_pst"),
            Extra = extra
        };
    }

    static void PrintUsage()
    {
        Console.WriteLine("Usage:");
        Console.WriteLine("  --config <filename>        Path to configuration file (YAML or JSON)");
        Console.WriteLine("  --config=<filename>        Same as above");
        Console.WriteLine("  --max_iterations <integer> Maximum number of iterations");
        Console.WriteLine("  --max_iterations=<integer> Same as above");
        Console.WriteLine("  -h, --help                 Show this help and exit");
    }

    static bool TryTakeValue(string arg, string prefix, string[] args, ref int i, out string? value)
    {
        value = null;
        if (arg.StartsWith(prefix + "=", StringComparison.Ordinal))
        {
            value = arg.Substring(prefix.Length + 1);
            return true;
        }

        // value should be next token
        if (i + 1 < args.Length && !args[i + 1].StartsWith("-"))
        {
            i++;
            value = args[i];
            return true;
        }

        return false;
    }

    public static async Task<int> Main(string[] args)
    {
        string? configFile = null;
        int? maxIterations = null;
        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];

            if (a == "-h" || a == "--help")
            {
                PrintUsage();
                return 1;
            }

            if (a.StartsWith("--config", StringComparison.Ordinal))
            {
                if (!TryTakeValue(a, "--config", args, ref i, out var val) || string.IsNullOrWhiteSpace(val))
                {
                    Console.Error.WriteLine("Error: '--config' requires a filename argument.");
                    PrintUsage();
                    Environment.Exit(1);
                }
                configFile = val;
                continue;
            }

            if (a.StartsWith("--max_iterations", StringComparison.Ordinal))
            {
                if (!TryTakeValue(a, "--max_iterations", args, ref i, out var val) || string.IsNullOrWhiteSpace(val))
                {
                    Console.Error.WriteLine("Error: '--max_iterations' requires an integer argument.");
                    PrintUsage();
                    return 1;
                }

                if (!int.TryParse(val, out int parsed) || parsed < 0)
                {
                    Console.Error.WriteLine("Error: '--max_iterations' value must be a non-negative integer.");
                    Environment.Exit(1);
                }

                maxIterations = parsed;
                continue;
            }

            Console.Error.WriteLine($"Warning: Unrecognized argument '{a}'");
        }

        if (string.IsNullOrWhiteSpace(configFile))
        {
            Console.Error.WriteLine("Error: configuration file must be specified with --config <file>");
            PrintUsage();
            Environment.Exit(1);
        }

        if (!File.Exists(configFile))
        {
            Console.Error.WriteLine($"Error: configuration file '{configFile}' does not exist.");
            Environment.Exit(1);
        }


        Config config;
        try
        {
            config = LoadConfig(configFile);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to load config: {ex.Message}");
            return 1;
        }

        // Set up logging. If Application Insights connection string is present, print it.
        // Hooking up an Application Insights provider requires adding the appropriate package.
        // Here we provide a Console logger plus an extensible ILoggerFactory.
        var aiConnectionString = Environment.GetEnvironmentVariable("INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING");
        Console.WriteLine("INSTRUMENTATION KEY: " + aiConnectionString);

        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddSimpleConsole(options =>
            {
                options.SingleLine = true;
                options.TimestampFormat = "yyyy-MM-dd HH:mm:ss ";
            });

            // If the Application Insights provider is available and a connection string is set,
            // the provider could be added here. That requires adding a NuGet package such as
            // Microsoft.Extensions.Logging.ApplicationInsights or Azure Monitor exporters.
            // We intentionally do not require that package in this file.
        });
        var logger = loggerFactory.CreateLogger("Predict");

        // Model instantiation (assumes the concrete model classes exist in the project).
        // The project already includes an Inferencer; adapt names if necessary.
        IWhalecallClassificationModel? whalecallClassificationModel = null;
        try
        {
#if false
            if (string.Equals(config.ModelType, "AudioSet", StringComparison.OrdinalIgnoreCase))
            {
                // OrcaDetectionModel(modelPath, threshold=model_local_threshold, min_num_positive_calls_threshold=model_global_threshold)
                whalecallClassificationModel = Activator.CreateInstance(Type.GetType("OrcaDetectionModel, Inferencer") ?? Type.GetType("OrcaDetectionModel"), config.ModelPath, config.ModelLocalThreshold ?? 0.0, config.ModelGlobalThreshold ?? 0.0);
            }
            else
#endif
            if (string.Equals(config.ModelType, "FastAI", StringComparison.OrdinalIgnoreCase))
            {
                whalecallClassificationModel = new FastAIModel(
                    config.ModelPath ?? "./model",
                    config.ModelName ?? "model.onnx",
                    config.ModelLocalThreshold ?? 0.0,
                    config.ModelGlobalThreshold ?? 0.0);
            }
            else
            {
                throw new InvalidOperationException("model_type should be one of AudioSet / FastAI");
            }
        }
        catch (Exception ex)
        {
            logger.LogWarning("Model creation failed (types may be in another assembly). Exception: {Message}", ex.Message);
            // leave model null — downstream code should guard against null (or replace with your own model).
        }

        // Azure clients
        object? blobServiceClient = null;
        object? cosmosClient = null;
        if (config.UploadToAzure)
        {
            var connectStr = Environment.GetEnvironmentVariable("AZURE_STORAGE_CONNECTION_STRING");
            if (string.IsNullOrWhiteSpace(connectStr))
            {
                logger.LogError("upload_to_azure is true in config but AZURE_STORAGE_CONNECTION_STRING is not set.");
                Environment.Exit(1);
            }

            try
            {
                // Attempt to create BlobServiceClient if Azure.Storage.Blobs is referenced.
                var blobType = Type.GetType("Azure.Storage.Blobs.BlobServiceClient, Azure.Storage.Blobs");
                if (blobType != null)
                    blobServiceClient = Activator.CreateInstance(blobType, connectStr);

                var cosmosEndpoint = "https://aifororcasmetadatastore.documents.azure.com:443/";
                var cosmosKey = Environment.GetEnvironmentVariable("AZURE_COSMOSDB_PRIMARY_KEY");
                var cosmosType = Type.GetType("Azure.Cosmos.CosmosClient, Azure.Cosmos");
                if (cosmosType != null && !string.IsNullOrWhiteSpace(cosmosKey))
                    cosmosClient = Activator.CreateInstance(cosmosType, cosmosEndpoint, cosmosKey);
            }
            catch (Exception ex)
            {
                logger.LogWarning("Failed to create Azure clients: {Message}", ex.Message);
            }
        }

        // Create wav directory if missing
        var localDir = "wav_dir";
        if (!Directory.Exists(localDir))
        {
            Directory.CreateDirectory(localDir);
        }

        // Instantiate HLS stream
        IHlsStream? hlsStream = null;
        try
        {
            if (string.Equals(config.HlsStreamType, "LiveHLS", StringComparison.OrdinalIgnoreCase))
            {
                var url = "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/" + config.HlsHydrophoneId;
                hlsStream = new HlsStream(url, config.HlsPollingInterval, localDir);
            }
            else if (string.Equals(config.HlsStreamType, "DateRangeHLS", StringComparison.OrdinalIgnoreCase))
            {
                if (string.IsNullOrWhiteSpace(config.HlsStartTimePst) || string.IsNullOrWhiteSpace(config.HlsEndTimePst))
                {
                    throw new InvalidOperationException("hls_start_time_pst and hls_end_time_pst must be set for DateRangeHLS");
                }

                // Parse PST local times and convert to Unix seconds (UTC)
                const string fmt = "yyyy-MM-dd HH:mm";
                var startDt = DateTime.ParseExact(config.HlsStartTimePst!, fmt, CultureInfo.InvariantCulture, DateTimeStyles.None);
                var endDt = DateTime.ParseExact(config.HlsEndTimePst!, fmt, CultureInfo.InvariantCulture, DateTimeStyles.None);

                // On Windows use "Pacific Standard Time", on Linux the zone id may be "US/Pacific".
                TimeZoneInfo pstZone = null!;
                try
                {
                    pstZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
                }
                catch
                {
                    pstZone = TimeZoneInfo.FindSystemTimeZoneById("US/Pacific");
                }

                var startOffset = TimeZoneInfo.ConvertTimeToUtc(startDt, pstZone);
                var endOffset = TimeZoneInfo.ConvertTimeToUtc(endDt, pstZone);
                var startUnix = new DateTimeOffset(startOffset).ToUnixTimeSeconds();
                var endUnix = new DateTimeOffset(endOffset).ToUnixTimeSeconds();

                var url = "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/" + config.HlsHydrophoneId;
                hlsStream = new DateRangeHlsStream(url, config.HlsPollingInterval, startUnix, endUnix, localDir, false);
            }
            else
            {
                throw new InvalidOperationException("hls_stream_type should be one of LiveHLS or DateRangeHLS");
            }
        }
        catch (Exception ex)
        {
            logger.LogError("Failed to instantiate HLS stream: {Message}", ex.Message);
            Environment.Exit(1);
        }

        // We want clips that end a few seconds ago to allow for upload time
        var currentClipEndTime = DateTime.UtcNow - TimeSpan.FromSeconds(10);

        // Loop over stream
        int iterationCount = 0;
        while (true)
        {
            // Try to call hlsStream.IsStreamOver() via reflection
            bool isOver = false;
            if (hlsStream != null)
            {
                isOver = hlsStream.IsStreamOver();
            }
            else
            {
                logger.LogWarning("HLS stream is not instantiated. Breaking loop to avoid infinite run.");
                break;
            }

            if (isOver)
            {
                break;
            }

            if (maxIterations.HasValue && iterationCount >= maxIterations.Value)
            {
                break;
            }
            iterationCount++;

            var nextClipResult = hlsStream.GetNextClip(currentClipEndTime);
            if (nextClipResult.clipPath == null)
            {
                logger.LogWarning("GetNextClip returned null result.");
                break;
            }

            string? clipPath = nextClipResult.clipPath;
            DateTime startTimestamp = DateTime.MinValue;
            if (!string.IsNullOrWhiteSpace(nextClipResult.startTimestamp) &&
                DateTime.TryParse(nextClipResult.startTimestamp, null, DateTimeStyles.RoundtripKind, out var parsedStart))
            {
                startTimestamp = parsedStart;
            }
            DateTime newClipEnd = nextClipResult.newCurrentClipEndTime;

            // If clipPath is empty or null, skip processing for this iteration
            if (!string.IsNullOrWhiteSpace(clipPath))
            {
                // Assume a SpectrogramVisualizer.WriteSpectrogram method exists
                string? spectrogramPath = null;
                try
                {
                    spectrogramPath = SpectrogramVisualizer.WriteSpectrogram(clipPath);
                }
                catch (Exception ex)
                {
                    logger.LogWarning("Failed to generate spectrogram: {Message}", ex.Message);
                }

                // Predict via model if available
                IDictionary<string, object>? predictionResults = null;
                if (whalecallClassificationModel != null)
                {
                    try
                    {
                        IDictionary<string, object> raw = whalecallClassificationModel.Predict(clipPath);
                        // Try to convert to IDictionary<string, object>
                        if (raw is IDictionary<string, object> dict)
                        {
                            predictionResults = dict;
                        }
                        else
                        {
                            // attempt reflection-based mapping of properties
                            var prs = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
                            foreach (var p in raw.GetType().GetProperties())
                            {
                                prs[p.Name] = p.GetValue(raw)!;
                            }
                            predictionResults = prs;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogWarning("Prediction failed: {Message}", ex.Message);
                    }
                }

                if (predictionResults != null)
                {
                    Console.WriteLine("\nlocal_confidences: {0}\n", predictionResults.TryGetValue("local_confidences", out var lc) ? lc : "(n/a)");
                    Console.WriteLine("local_predictions: {0}\n", predictionResults.TryGetValue("local_predictions", out var lp) ? lp : "(n/a)");
                    Console.WriteLine("global_confidence: {0}\n", predictionResults.TryGetValue("global_confidence", out var gc) ? gc : "(n/a)");
                    Console.WriteLine("global_prediction: {0}", predictionResults.TryGetValue("global_prediction", out var gp) ? gp : "(n/a)");

                    if (predictionResults.TryGetValue("global_prediction", out var gpv) && Convert.ToInt32(gpv) == 1)
                    {
                        Console.WriteLine("FOUND!!!!");

                        // logging to app insights (or console)
                        logger.LogInformation("Orca Found: Hydrophone ID={HydrophoneId}", config.HlsHydrophoneId);

                        if (config.UploadToAzure && blobServiceClient != null)
                        {
                            try
                            {
                                // upload audio
                                var audioBlobClientMethod = blobServiceClient.GetType().GetMethod("GetBlobClient", new[] { typeof(string), typeof(string) }) ?? blobServiceClient.GetType().GetMethod("GetBlobClient", new[] { typeof(string) });
                                // The exact APIs vary; here we attempt a common pattern if Azure SDK is present.
                                var audioName = Path.GetFileName(clipPath);
                                // Fallback behavior: do nothing if SDK not available.
                                Console.WriteLine("Uploaded audio to Azure Storage (placeholder)");

                                var spectrogramName = Path.GetFileName(spectrogramPath ?? "");
                                Console.WriteLine("Uploaded spectrogram to Azure Storage (placeholder)");
                            }
                            catch (Exception ex)
                            {
                                logger.LogWarning("Azure upload failed: {Message}", ex.Message);
                            }
                        }

                        // Insert metadata into CosmosDB if configured - placeholder behavior if SDK not present
                        if (config.UploadToAzure && cosmosClient != null)
                        {
                            try
                            {
                                // assemble metadata and create item in Cosmos DB
                                Console.WriteLine("Added metadata to Azure CosmosDB (placeholder)");
                            }
                            catch (Exception ex)
                            {
                                logger.LogWarning("Failed to add metadata to CosmosDB: {Message}", ex.Message);
                            }
                        }
                    }

                    // delete local files if configured
                    if (config.DeleteLocalWavs)
                    {
                        try
                        {
                            if (File.Exists(clipPath))
                            {
                                File.Delete(clipPath);
                            }
                            if (!string.IsNullOrWhiteSpace(spectrogramPath) && File.Exists(spectrogramPath))
                            {
                                File.Delete(spectrogramPath);
                            }
                        }
                        catch (Exception ex)
                        {
                            logger.LogWarning("Failed to delete local files: {Message}", ex.Message);
                        }
                    }

                    // log results locally if configured and not uploading to Azure
                    if (!string.IsNullOrWhiteSpace(config.LogResults) && !config.UploadToAzure)
                    {
                        try
                        {
                            Directory.CreateDirectory(config.LogResults);
                            predictionResults["model_type"] = config.ModelType ?? "";
                            predictionResults["model_name"] = config.ModelName ?? "";
                            predictionResults["model_path"] = config.ModelPath ?? "";
                            var jsonName = Path.GetFileName(clipPath).Replace(".wav", ".json", StringComparison.OrdinalIgnoreCase);
                            var outPath = Path.Combine(config.LogResults, jsonName);
                            File.WriteAllText(outPath, JsonSerializer.Serialize(predictionResults));
                        }
                        catch (Exception ex)
                        {
                            logger.LogWarning("Failed to write log results: {Message}", ex.Message);
                        }
                    }
                }
            }

            // advance currentClipEndTime by polling interval
            currentClipEndTime = currentClipEndTime + TimeSpan.FromSeconds(config.HlsPollingInterval);
            // small delay to avoid tight loop if GetNextClip returned immediately
            await Task.Delay(100);
        }

        Console.WriteLine("Processing finished.");
        return 0;
    }
}