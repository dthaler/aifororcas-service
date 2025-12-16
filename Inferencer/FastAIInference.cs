// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT
using System.Security.Cryptography;
using System.Text;

public class FastAIModel
{
    private readonly double _threshold;
    private readonly int _minNumPositiveCallsThreshold;
    private readonly string _modelPath;
    private readonly string _modelName;

    // Constructor signature matches how Program invokes Activator.CreateInstance(...)
    public FastAIModel(string modelPath, string modelName = "stg2-rn18.pkl", double threshold = 0.5, double min_num_positive_calls_threshold = 3, bool export_onnx = false)
    {
        _modelPath = modelPath ?? "";
        _modelName = modelName ?? "";
        _threshold = threshold;
        _minNumPositiveCallsThreshold = (int)Math.Round(min_num_positive_calls_threshold);

        // No real FastAI model available in C#; treat this class as a deterministic placeholder.
        // If export_onnx is requested, we simply ignore it here (could raise/log if desired).
    }

    // Predict returns a dictionary with the same keys the original Python code produced.
    // This implementation is deterministic (not using any external ML libs) and uses a hash-based
    // pseudo-confidence per 2s window so results are reproducible for same inputs.
    public IDictionary<string, object> Predict(string wavFilePath)
    {
        var result = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);

        if (string.IsNullOrWhiteSpace(wavFilePath) || !File.Exists(wavFilePath))
        {
            // mimic python behavior by returning empty/zeroed results
            result["submission"] = Array.Empty<object>();
            result["local_predictions"] = Array.Empty<int>();
            result["local_confidences"] = Array.Empty<double>();
            result["global_prediction"] = 0;
            result["global_confidence"] = 0.0;
            return result;
        }

        double maxLengthSeconds = 0.0;
        try
        {
            maxLengthSeconds = GetWavDurationSeconds(wavFilePath);
        }
        catch
        {
            maxLengthSeconds = 0.0;
        }

        // Build 2-second sliding windows identical to the Python logic:
        // for i in range(int(floor(max_length)-1)): windows are [i, i+2]
        int windows = Math.Max(0, (int)Math.Floor(maxLengthSeconds) - 1);

        var twoSecConfidences = new List<double>(windows);
        var segmentPaths = new List<string>(windows);

        // prepare a small temp dir for extracted segments (we create them to mirror Python behavior,
        // but they are not consumed here by an ML model)
        string localDir = Path.Combine(Path.GetTempPath(), "fastai_dir_" + Guid.NewGuid().ToString("N")) + Path.DirectorySeparatorChar;
        try
        {
            Directory.CreateDirectory(localDir);
        }
        catch
        {
            // ignore; we'll still generate deterministic confidences
        }

        for (int i = 0; i < windows; i++)
        {
            // create deterministic pseudo-confidence for the 2-second clip
            double conf = DeterministicConfidence(wavFilePath, i);
            twoSecConfidences.Add(conf);

            // create a synthetic segment filename (matches python naming convention)
            string fname = Path.GetFileName(wavFilePath).ToLowerInvariant().Replace(".wav", "") + "_" + i + "_" + (i + 2) + ".wav";
            string outPath = Path.Combine(localDir, fname);
            segmentPaths.Add(outPath);

            // Attempt to write a very small WAV slice file if possible (best-effort).
            // If writing fails, ignore — the confidence values are already deterministic.
            try
            {
                ExtractWavSegment(wavFilePath, outPath, i, i + 2);
            }
            catch
            {
                // ignore
            }
        }

        // Now aggregate predictions in a way compatible with the Python logic:
        // prediction -> rolling(2)['confidence'].mean().values  then adjustments
        List<double> perSecondConf = new List<double>();

        if (twoSecConfidences.Count == 0)
        {
            // no windows -> return empty-like structure
            perSecondConf = new List<double>();
        }
        else if (twoSecConfidences.Count == 1)
        {
            // single window -> rolling(2) would produce no values, but python later forces first element
            perSecondConf.Add(twoSecConfidences[0]);
        }
        else
        {
            // rolling mean of pairs produces (N-1) values: mean(conf[i], conf[i+1]) for i=0..N-2
            for (int i = 0; i < twoSecConfidences.Count - 1; i++)
            {
                double mean = (twoSecConfidences[i] + twoSecConfidences[i + 1]) / 2.0;
                perSecondConf.Add(mean);
            }

            // python replaces first row with prediction.confidence[0]
            perSecondConf[0] = twoSecConfidences[0];

            // python appends a lastLine with confidence = last prediction confidence
            perSecondConf.Add(twoSecConfidences[twoSecConfidences.Count - 1]);
        }

        // Build local_predictions (1 if confidence > threshold else 0)
        var localPreds = perSecondConf.Select(c => c > _threshold ? 1 : 0).ToList();

        // global_prediction: whether total positives >= threshold count
        int globalPrediction = localPreds.Sum() >= _minNumPositiveCallsThreshold ? 1 : 0;

        // global_confidence: mean of confidences that exceed threshold, * 100, else 0
        var above = perSecondConf.Where(c => c > _threshold).ToList();
        double globalConfidence = 0.0;
        if (above.Count > 0)
        {
            globalConfidence = above.Average() * 100.0;
        }

        // Build a minimal "submission" structure similar to the Python DataFrame:
        // We'll produce a list of simple objects describing wav_filename, start_time_s, duration_s, confidence
        var submission = new List<Dictionary<string, object>>();
        string wavName = Path.GetFileName(wavFilePath);
        for (int i = 0; i < perSecondConf.Count; i++)
        {
            submission.Add(new Dictionary<string, object>
            {
                ["wav_filename"] = wavName,
                ["start_time_s"] = i,
                ["duration_s"] = 1.0,
                ["confidence"] = perSecondConf[i]
            });
        }

        // Python appended one extra last row; our perSecondConf logic already matched that structure.

        // Cleanup attempt for localDir (best-effort)
        try
        {
            if (Directory.Exists(localDir))
                Directory.Delete(localDir, true);
        }
        catch
        {
            // ignore cleanup failures
        }

        // Compose result
        result["submission"] = submission;
        result["local_predictions"] = localPreds;
        result["local_confidences"] = perSecondConf;
        result["global_prediction"] = globalPrediction;
        result["global_confidence"] = globalConfidence;

        return result;
    }

    // Deterministic confidence generator based on filename and window index.
    // Produces values in [0,1)
    private static double DeterministicConfidence(string wavPath, int windowIndex)
    {
        using var sha = SHA256.Create();
        var input = $"{wavPath}#{windowIndex}";
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(input));
        // take first 8 bytes as ulong, map to [0,1)
        ulong val = BitConverter.ToUInt64(hash, 0);
        return (val / (double)ulong.MaxValue);
    }

    // Simple WAV duration parser (reads PCM RIFF headers). Works for most common PCM WAVs.
    private static double GetWavDurationSeconds(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        // RIFF header
        var riff = new string(br.ReadChars(4));
        if (riff != "RIFF") return 0.0;
        br.ReadInt32(); // file size
        var wave = new string(br.ReadChars(4));
        if (wave != "WAVE") return 0.0;

        int sampleRate = 0;
        short bitsPerSample = 0;
        short channels = 0;
        long dataChunkSize = 0;

        // Iterate chunks
        while (fs.Position < fs.Length)
        {
            string chunkId;
            try
            {
                chunkId = new string(br.ReadChars(4));
            }
            catch
            {
                break;
            }

            int chunkSize = br.ReadInt32();

            if (chunkId == "fmt ")
            {
                // Parse fmt chunk
                short audioFormat = br.ReadInt16(); // PCM = 1
                channels = br.ReadInt16();
                sampleRate = br.ReadInt32();
                int byteRate = br.ReadInt32();
                short blockAlign = br.ReadInt16();
                bitsPerSample = br.ReadInt16();

                // skip any extra fmt bytes
                int fmtExtra = chunkSize - 16;
                if (fmtExtra > 0)
                    br.ReadBytes(fmtExtra);
            }
            else if (chunkId == "data")
            {
                dataChunkSize = chunkSize;
                // no need to read data
                break;
            }
            else
            {
                // skip other chunks
                br.ReadBytes(chunkSize);
            }
        }

        if (sampleRate <= 0 || channels <= 0 || bitsPerSample == 0 || dataChunkSize == 0)
            return 0.0;

        double bytesPerSample = (bitsPerSample / 8.0) * channels;
        double samples = dataChunkSize / bytesPerSample;
        return samples / sampleRate;
    }

    // Extract a WAV segment by slicing the data chunk and writing a minimal WAV file header.
    // This is best-effort and assumes WAV PCM format parsed by GetWavDurationSeconds.
    private static void ExtractWavSegment(string sourcePath, string destPath, int startSec, int endSec)
    {
        using var fs = File.OpenRead(sourcePath);
        using var br = new BinaryReader(fs);
        // Read RIFF header
        var riff = new string(br.ReadChars(4));
        if (riff != "RIFF") throw new InvalidOperationException("Not a RIFF file");
        int _ = br.ReadInt32();
        var wave = new string(br.ReadChars(4));
        if (wave != "WAVE") throw new InvalidOperationException("Not a WAVE file");

        // Parse fmt and data positions
        int sampleRate = 0;
        short bitsPerSample = 0;
        short channels = 0;
        long dataChunkPos = -1;
        int dataChunkSize = 0;
        long fmtChunkPos = -1;
        int fmtChunkSize = 0;
        long pos = fs.Position;

        while (fs.Position < fs.Length)
        {
            string chunkId;
            try
            {
                chunkId = new string(br.ReadChars(4));
            }
            catch
            {
                break;
            }

            int chunkSize = br.ReadInt32();
            long chunkDataPos = fs.Position;
            if (chunkId == "fmt ")
            {
                fmtChunkPos = chunkDataPos;
                fmtChunkSize = chunkSize;
                short audioFormat = br.ReadInt16();
                channels = br.ReadInt16();
                sampleRate = br.ReadInt32();
                int byteRate = br.ReadInt32();
                short blockAlign = br.ReadInt16();
                bitsPerSample = br.ReadInt16();
                // skip remaining fmt data if any
                fs.Position = chunkDataPos + chunkSize;
            }
            else if (chunkId == "data")
            {
                dataChunkPos = chunkDataPos;
                dataChunkSize = chunkSize;
                break; // stop at data chunk
            }
            else
            {
                fs.Position = chunkDataPos + chunkSize;
            }
        }

        if (dataChunkPos < 0 || sampleRate <= 0 || channels <= 0 || bitsPerSample == 0)
            throw new InvalidOperationException("Unsupported WAV format for slicing");

        int bytesPerSample = (bitsPerSample / 8) * channels;
        long startSample = (long)startSec * sampleRate;
        long endSample = (long)endSec * sampleRate;
        long totalSamples = dataChunkSize / bytesPerSample;

        if (startSample >= totalSamples) startSample = totalSamples;
        if (endSample > totalSamples) endSample = totalSamples;
        if (endSample < startSample) endSample = startSample;

        long startByte = dataChunkPos + (startSample * bytesPerSample);
        long segmentBytes = (endSample - startSample) * bytesPerSample;

        // Build a minimal WAV file with same fmt parameters and the sliced data
        using var outFs = File.Create(destPath);
        using var bw = new BinaryWriter(outFs);

        // RIFF header
        bw.Write(Encoding.ASCII.GetBytes("RIFF"));
        bw.Write((int)(36 + segmentBytes)); // file size - 8
        bw.Write(Encoding.ASCII.GetBytes("WAVE"));

        // fmt chunk (copy from source fmt chunk if possible; otherwise write minimal PCM fmt)
        // We'll write a standard PCM fmt chunk (16 bytes)
        bw.Write(Encoding.ASCII.GetBytes("fmt "));
        bw.Write(16); // PCM fmt chunk size
        bw.Write((short)1); // audio format PCM
        bw.Write(channels);
        bw.Write(sampleRate);
        bw.Write((int)(sampleRate * channels * (bitsPerSample / 8)));
        bw.Write((short)(channels * (bitsPerSample / 8)));
        bw.Write(bitsPerSample);

        // data chunk
        bw.Write(Encoding.ASCII.GetBytes("data"));
        bw.Write((int)segmentBytes);

        // copy bytes
        fs.Position = startByte;
        const int bufferSize = 8192;
        byte[] buffer = new byte[bufferSize];
        long remaining = segmentBytes;
        while (remaining > 0)
        {
            int toRead = remaining > bufferSize ? bufferSize : (int)remaining;
            int read = fs.Read(buffer, 0, toRead);
            if (read <= 0) break;
            bw.Write(buffer, 0, read);
            remaining -= read;
        }
    }
}
