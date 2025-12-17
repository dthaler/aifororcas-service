// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Diagnostics;
using System.Threading;

public class HlsStream : IHlsStream
{
    private readonly string _streamBase;
    private readonly int _pollingInterval;
    private readonly string _wavDir;
    private readonly int _audioOffset;
    private readonly string _s3Bucket;
    private readonly string _hydrophoneId;
    private readonly HttpClient _http;

    public HlsStream(string streamBase, int pollingInterval, string wavDir)
        : this(streamBase, pollingInterval, wavDir, 2)
    {
    }

    public HlsStream(string streamBase, int pollingInterval, string wavDir, int audioOffset = 2)
    {
        _streamBase = streamBase ?? throw new ArgumentNullException(nameof(streamBase));
        _pollingInterval = pollingInterval;
        _wavDir = wavDir ?? throw new ArgumentNullException(nameof(wavDir));
        _audioOffset = audioOffset;

        var afterPrefix = _streamBase.Replace("https://s3-us-west-2.amazonaws.com/", "");
        var tokens = afterPrefix.Split(new[] { '/' }, StringSplitOptions.RemoveEmptyEntries);
        _s3Bucket = tokens.Length > 0 ? tokens[0] : string.Empty;
        _hydrophoneId = tokens.Length > 1 ? tokens[1] : string.Empty;

        _http = new HttpClient();
        _http.Timeout = TimeSpan.FromSeconds(30);
    }

    public string? GetLatestFolderTime()
    {
        try
        {
            var latestUrl = $"{_streamBase.TrimEnd('/')}/latest.txt";
            var resp = _http.GetStringAsync(latestUrl).Result;
            return resp?.Trim();
        }
        catch (Exception ex) when (ex is HttpRequestException || ex is TaskCanceledException)
        {
            Console.WriteLine($"Failed to fetch latest.txt: {ex.Message}");
            return null;
        }
    }

    public (string? clipPath, string? startTimestamp, DateTime newCurrentClipEndTime) GetNextClip(DateTime currentClipEndTime)
    {
        try
        {
            var now = DateTime.UtcNow;
            var timeToSleep = (currentClipEndTime - now).TotalSeconds + 10;
            if (timeToSleep > 0)
            {
                Thread.Sleep(TimeSpan.FromSeconds(timeToSleep));
            }

            Console.WriteLine($"Listening to location {_streamBase}");
            var streamId = GetLatestFolderTime();
            if (streamId == null)
                return (null, null, currentClipEndTime);

            var streamUrl = $"{_streamBase.TrimEnd('/')}/hls/{streamId}/live.m3u8";

            string playlist;
            try
            {
                playlist = _http.GetStringAsync(streamUrl).Result;
            }
            catch (Exception ex) when (ex is HttpRequestException || ex is TaskCanceledException)
            {
                Console.WriteLine($".m3u8 file does not exist or failed to fetch: {ex.Message}");
                return (null, null, currentClipEndTime);
            }

            var lines = playlist.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                                .Select(l => l.Trim()).ToList();
            var segments = new List<(double duration, string uri)>();
            for (int i = 0; i < lines.Count; i++)
            {
                var l = lines[i];
                if (l.StartsWith("#EXTINF:", StringComparison.OrdinalIgnoreCase))
                {
                    var durPart = l.Substring(8).Trim().TrimEnd(',');
                    if (double.TryParse(durPart, NumberStyles.Float, CultureInfo.InvariantCulture, out var d))
                    {
                        if (i + 1 < lines.Count)
                        {
                            var next = lines[i + 1];
                            if (!next.StartsWith("#"))
                            {
                                segments.Add((d, next));
                            }
                        }
                    }
                }
            }

            if (segments.Count == 0)
            {
                Console.WriteLine("No segments in playlist.");
                return (null, null, currentClipEndTime);
            }

            int numTotalSegments = segments.Count;
            var targetDuration = segments.Average(s => s.duration);
            if (targetDuration <= 0)
            {
                Console.WriteLine("Invalid target duration from playlist.");
                return (null, null, currentClipEndTime);
            }

            var numSegmentsInWavDuration = (int)Math.Ceiling(_pollingInterval / targetDuration);

            if (!long.TryParse(streamId, out var streamFolderUnix))
            {
                Console.WriteLine("stream id is not a unix epoch integer, aborting");
                return (null, null, currentClipEndTime);
            }

            var currentClipEndUnix = new DateTimeOffset(currentClipEndTime).ToUnixTimeSeconds();
            var timeSinceFolderStart = (double)(currentClipEndUnix - streamFolderUnix);

            timeSinceFolderStart -= _audioOffset;

            if (timeSinceFolderStart < _pollingInterval + 20)
            {
                Console.WriteLine("not enough data for a 1 minute clip + 20 second buffer");
                return (null, null, currentClipEndTime);
            }

            var minNumTotalSegmentsRequired = (int)Math.Ceiling(timeSinceFolderStart / targetDuration);
            var segmentStartIndex = minNumTotalSegmentsRequired - numSegmentsInWavDuration;
            var segmentEndIndex = segmentStartIndex + numSegmentsInWavDuration;

            double endSeconds = segmentEndIndex * targetDuration + streamFolderUnix + _audioOffset;
            var endUtc = DateTimeOffset.FromUnixTimeSeconds((long)Math.Round(endSeconds)).UtcDateTime;
            var newCurrentClipEndTime = endUtc;

            if (segmentEndIndex > numTotalSegments)
                return (null, null, newCurrentClipEndTime);

            var tmpPath = Path.Combine(Path.GetTempPath(), "hls_tmp_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tmpPath);

            var baseUri = streamUrl.Substring(0, streamUrl.LastIndexOf('/') + 1);

            var downloadedFiles = new List<string>();
            for (int i = segmentStartIndex; i < segmentEndIndex; i++)
            {
                if (i < 0 || i >= segments.Count)
                    continue;

                var seg = segments[i];
                var fileName = seg.uri;
                string audioUrl;
                if (Uri.IsWellFormedUriString(fileName, UriKind.Absolute))
                {
                    audioUrl = fileName;
                }
                else
                {
                    audioUrl = baseUri + fileName;
                }

                try
                {
                    var bytes = _http.GetByteArrayAsync(audioUrl).Result;
                    var outPath = Path.Combine(tmpPath, fileName.Replace("/", "_"));
                    File.WriteAllBytes(outPath, bytes);
                    downloadedFiles.Add(outPath);
                }
                catch (Exception)
                {
                    Console.WriteLine("Skipping " + audioUrl + " : error.");
                }
            }

            var currentClipStartTime = newCurrentClipEndTime - TimeSpan.FromSeconds(_pollingInterval);
            var clipStartIso = currentClipStartTime.ToString("o", CultureInfo.InvariantCulture);

            var (clipName, _) = GetReadableClipName(_hydrophoneId, currentClipStartTime);

            var hlsFile = clipName + ".ts";
            var audioFile = clipName + ".wav";
            var wavFilePath = Path.Combine(_wavDir, audioFile);
            var hlsFilePath = Path.Combine(tmpPath, hlsFile);

            using (var outFs = File.Create(hlsFilePath))
            {
                foreach (var segPath in downloadedFiles)
                {
                    try
                    {
                        using var inFs = File.OpenRead(segPath);
                        inFs.CopyTo(outFs);
                    }
                    catch { }
                }
            }

            try
            {
                RunFfmpeg(hlsFilePath, wavFilePath);
            }
            catch (Exception e)
            {
                Console.WriteLine("FFmpeg command failed: " + e.Message);
                try { Directory.Delete(tmpPath, true); } catch { }
                throw;
            }

            try { Directory.Delete(tmpPath, true); } catch { }

            return (wavFilePath, clipStartIso + "Z", newCurrentClipEndTime);
        }
        catch (Exception ex)
        {
            Console.WriteLine("GetNextClip unexpected failure: " + ex.Message);
            return (null, null, currentClipEndTime);
        }
    }

    public bool IsStreamOver()
    {
        return false;
    }

    private static (string clipName, DateTime pacificDate) GetReadableClipName(string hydrophoneId, DateTime clipTimeUtc)
    {
        var utc = DateTime.SpecifyKind(clipTimeUtc, DateTimeKind.Utc);

        TimeZoneInfo pstZone = null!;
        try
        {
            pstZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time");
        }
        catch
        {
            try
            {
                pstZone = TimeZoneInfo.FindSystemTimeZoneById("US/Pacific");
            }
            catch
            {
                pstZone = TimeZoneInfo.Utc;
            }
        }

        var pacific = TimeZoneInfo.ConvertTimeFromUtc(utc, pstZone);
        var dateFormat = "yyyy_MM_dd_HH_mm_ss_zzz";
        var clipname = pacific.ToString(dateFormat, CultureInfo.InvariantCulture).Replace(":", "-");
        return ($"{hydrophoneId}_{clipname}", pacific);
    }

    private static void RunFfmpeg(string inputTs, string outputWav)
    {
        if (!File.Exists(inputTs))
            throw new FileNotFoundException("Input TS file not found", inputTs);

        Directory.CreateDirectory(Path.GetDirectoryName(outputWav) ?? ".");

        var psi = new ProcessStartInfo
        {
            FileName = "ffmpeg",
            Arguments = $"-y -i \"{inputTs}\" \"{outputWav}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var proc = Process.Start(psi);
        if (proc == null)
            throw new InvalidOperationException("Failed to start ffmpeg process.");

        string stdout = proc.StandardOutput.ReadToEnd();
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();

        if (proc.ExitCode != 0)
        {
            throw new InvalidOperationException($"ffmpeg exited with code {proc.ExitCode}. stderr: {stderr}");
        }
    }
}