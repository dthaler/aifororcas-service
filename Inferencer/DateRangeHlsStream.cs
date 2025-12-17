// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Diagnostics;

public class DateRangeHlsStream : IHlsStream
{
    private readonly string _streamBase;
    private readonly int _pollingIntervalInSeconds;
    private long _startUnixTime;
    private readonly long _endUnixTime;
    private readonly string _wavDir;
    private readonly bool _overwriteOutput;
    private readonly bool _realTime;
    private bool _isEndOfStream;
    private readonly int _audioOffset;

    private readonly List<long> _validFolders;
    private int _currentFolderIndex;
    private long _currentClipStartTime;
    private readonly HttpClient _http;

    public DateRangeHlsStream(string streamBase, int pollingInterval, long startUnixTime, long endUnixTime, string wavDir, bool overwriteOutput)
        : this(streamBase, pollingInterval, startUnixTime, endUnixTime, wavDir, overwriteOutput, false, 2)
    { }

    public DateRangeHlsStream(string streamBase, int pollingInterval, long startUnixTime, long endUnixTime, string wavDir, bool overwriteOutput = false, bool realTime = false, int audioOffset = 2)
    {
        _streamBase = streamBase ?? throw new ArgumentNullException(nameof(streamBase));
        _pollingIntervalInSeconds = pollingInterval;
        _startUnixTime = startUnixTime;
        _endUnixTime = endUnixTime;
        _wavDir = wavDir ?? throw new ArgumentNullException(nameof(wavDir));
        _overwriteOutput = overwriteOutput;
        _realTime = realTime;
        _isEndOfStream = false;
        _audioOffset = audioOffset;

        Directory.CreateDirectory(_wavDir);

        // Conservative folder list based on interval; original code queried S3 for available folders.
        _validFolders = new List<long>();
        for (long t = _startUnixTime; t <= _endUnixTime; t += Math.Max(1, _pollingIntervalInSeconds))
            _validFolders.Add(t);

        _currentFolderIndex = 0;
        _currentClipStartTime = _startUnixTime;

        _http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
    }

    public (string? clipPath, string? startTimestamp, DateTime newCurrentClipEndTime) GetNextClip(DateTime currentClipEndTime)
    {
        try
        {
            if (_currentFolderIndex >= _validFolders.Count)
                return (null, null, currentClipEndTime);

            var currentFolder = _validFolders[_currentFolderIndex];

            var streamUrl = $"{_streamBase.TrimEnd('/')}/hls/{currentFolder}/live.m3u8";

            string playlist;
            try
            {
                playlist = _http.GetStringAsync(streamUrl).Result;
            }
            catch (Exception ex) when (ex is HttpRequestException || ex is TaskCanceledException)
            {
                _currentFolderIndex++;
                if (_currentFolderIndex >= _validFolders.Count)
                    return (null, null, currentClipEndTime);
                _currentClipStartTime = _validFolders[_currentFolderIndex];
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
                                segments.Add((d, next));
                        }
                    }
                }
            }

            if (segments.Count == 0)
            {
                _currentFolderIndex++;
                if (_currentFolderIndex >= _validFolders.Count) return (null, null, currentClipEndTime);
                _currentClipStartTime = _validFolders[_currentFolderIndex];
                return (null, null, currentClipEndTime);
            }

            int numTotalSegments = segments.Count;
            var targetDuration = segments.Average(s => s.duration);
            if (targetDuration <= 0) return (null, null, currentClipEndTime);

            var numSegmentsInWavDuration = (int)Math.Ceiling(_pollingIntervalInSeconds / targetDuration);

            var timeSinceFolderStart = (_currentClipStartTime - currentFolder) - _audioOffset;
            var segmentStartIndex = (int)Math.Ceiling(timeSinceFolderStart / targetDuration);
            var segmentEndIndex = segmentStartIndex + numSegmentsInWavDuration;

            if (segmentEndIndex > numTotalSegments)
            {
                _currentFolderIndex++;
                if (_currentFolderIndex >= _validFolders.Count)
                    return (null, null, currentClipEndTime);
                _currentClipStartTime = _validFolders[_currentFolderIndex];
                return (null, null, currentClipEndTime);
            }

            _currentClipStartTime = AddIntervalToUnixTime(_currentClipStartTime, _pollingIntervalInSeconds);

            var tmpPath = Path.Combine(Path.GetTempPath(), "drhls_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tmpPath);

            var baseUri = streamUrl.Substring(0, streamUrl.LastIndexOf('/') + 1);
            var downloadedFiles = new List<string>();
            for (int i = segmentStartIndex; i < segmentEndIndex; i++)
            {
                if (i < 0 || i >= segments.Count) continue;
                var seg = segments[i];
                var fileName = seg.uri;
                string audioUrl = Uri.IsWellFormedUriString(fileName, UriKind.Absolute) ? fileName : baseUri + fileName;
                try
                {
                    var bytes = _http.GetByteArrayAsync(audioUrl).Result;
                    var outPath = Path.Combine(tmpPath, fileName.Replace("/", "_"));
                    File.WriteAllBytes(outPath, bytes);
                    downloadedFiles.Add(outPath);
                }
                catch
                {
                    Console.WriteLine("Skipping " + audioUrl + " : error.");
                }
            }

            var clipName = GetReadableClipName(_streamBase, currentFolder, _currentClipStartTime).clipName;
            var hlsFile = Path.Combine(tmpPath, clipName + ".ts");
            using (var outFs = File.Create(hlsFile))
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

            var audioFile = clipName + ".wav";
            var wavFilePath = Path.Combine(_wavDir, audioFile);

            try
            {
                RunFfmpeg(hlsFile, wavFilePath, _overwriteOutput);
            }
            catch (Exception)
            {
                try { Directory.Delete(tmpPath, true); } catch { }
                throw;
            }

            try { Directory.Delete(tmpPath, true); } catch { }

            string clipStartIso;
            var clipStartDt = DateTimeOffset.FromUnixTimeSeconds(currentFolder).UtcDateTime;
            clipStartIso = clipStartDt.ToString("o", CultureInfo.InvariantCulture) + "Z";

            var newCurrentClipEndTime = DateTimeOffset.FromUnixTimeSeconds(_currentClipStartTime).UtcDateTime;
            return (wavFilePath, clipStartIso, newCurrentClipEndTime);
        }
        catch (Exception ex)
        {
            Console.WriteLine("GetNextClip unexpected failure: " + ex.Message);
            return (null, null, currentClipEndTime);
        }
    }

    public bool IsStreamOver()
    {
        return _currentClipStartTime >= _endUnixTime;
    }

    private static (string clipName, DateTime pacificDate) GetReadableClipName(string streamBase, long folderUnix, long clipUnix)
    {
        var clipTimeUtc = DateTimeOffset.FromUnixTimeSeconds(clipUnix).UtcDateTime;
        DateTime utc = DateTime.SpecifyKind(clipTimeUtc, DateTimeKind.Utc);

        TimeZoneInfo pstZone;
        try { pstZone = TimeZoneInfo.FindSystemTimeZoneById("Pacific Standard Time"); }
        catch
        {
            try { pstZone = TimeZoneInfo.FindSystemTimeZoneById("US/Pacific"); }
            catch { pstZone = TimeZoneInfo.Utc; }
        }

        var pacific = TimeZoneInfo.ConvertTimeFromUtc(utc, pstZone);
        var dateFormat = "yyyy_MM_dd_HH_mm_ss_zzz";
        var clipname = pacific.ToString(dateFormat, CultureInfo.InvariantCulture).Replace(":", "-");

        var afterPrefix = streamBase.Replace("https://s3-us-west-2.amazonaws.com/", "");
        var tokens = afterPrefix.Split(new[] { '/' }, StringSplitOptions.RemoveEmptyEntries);
        var hydrophoneId = tokens.Length > 1 ? tokens[1] : (tokens.Length > 0 ? tokens[0] : "unknown");
        return ($"{hydrophoneId}_{clipname}", pacific);
    }

    private static long AddIntervalToUnixTime(long unix, int intervalSeconds) => unix + intervalSeconds;

    private static void RunFfmpeg(string inputTs, string outputWav, bool overwrite)
    {
        if (!File.Exists(inputTs)) throw new FileNotFoundException("Input TS file not found", inputTs);
        Directory.CreateDirectory(Path.GetDirectoryName(outputWav) ?? ".");

        var args = overwrite ? $"-y -i \"{inputTs}\" \"{outputWav}\"" : $"-n -i \"{inputTs}\" \"{outputWav}\"";
        var psi = new ProcessStartInfo
        {
            FileName = "ffmpeg",
            Arguments = args,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var proc = Process.Start(psi);
        if (proc == null) throw new InvalidOperationException("Failed to start ffmpeg process.");
        string stdout = proc.StandardOutput.ReadToEnd();
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        if (proc.ExitCode != 0) throw new InvalidOperationException($"ffmpeg exited with code {proc.ExitCode}. stderr: {stderr}");
    }
}
