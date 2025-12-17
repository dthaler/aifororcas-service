// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT

// The implementation below uses NAudio for WAV reading/resampling and MathNet.Numerics for FFT.
using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using System.Numerics;
using System.Text;

public class FastAIModel : IWhalecallClassificationModel
{
    private readonly double _threshold;
    private readonly int _minNumPositiveCallsThreshold;
    private readonly InferenceSession _session;

    // Spectrogram config constants matching the Python snippet.
    private const int TargetSampleRate = 20000;    // config.resample_to
    private const int NFFT = 2560;                // n_fft
    private const int HopLength = 256;            // hop_length
    private const int NMels = 256;                // n_mels
    private const double Fmin = 0.0;
    private const double Fmax = 10000.0;
    private const double TopDb = 100.0;
    private const int DurationMs = 4000;          // config.duration (ms)
    private const bool Downmix = true;

    // We will produce a 256x256 input for ONNX (channels=1).
    private const int ModelHeight = NMels;
    private const int ModelWidth = 256;

    public FastAIModel(string modelPath, string modelName = "stg2-rn18.pkl", double threshold = 0.5, double min_num_positive_calls_threshold = 3)
    {
        string fullPath = Path.Combine(modelPath, modelName);
        _session = new InferenceSession(fullPath);
        _threshold = threshold;
        _minNumPositiveCallsThreshold = (int)Math.Round(min_num_positive_calls_threshold);
    }

    // Generate local predictions using a .WAV file.
    public IDictionary<string, object> Predict(string wavFilePath)
    {
        var result = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);

        if (string.IsNullOrWhiteSpace(wavFilePath) || !File.Exists(wavFilePath))
        {
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

        // Prepare a small temp dir for extracted segments.
        string localDir = Path.Combine(Path.GetTempPath(), "fastai_dir_" + Guid.NewGuid().ToString("N")) + Path.DirectorySeparatorChar;
        try
        {
            Directory.CreateDirectory(localDir);
        }
        catch
        {
            // Fallthrough.
        }

        for (int i = 0; i < windows; i++)
        {
            // Create synthetic segment filename (matches python naming convention).
            string fname = Path.GetFileName(wavFilePath).ToLowerInvariant().Replace(".wav", "") + "_" + i + "_" + (i + 2) + ".wav";
            string outPath = Path.Combine(localDir, fname);
            segmentPaths.Add(outPath);

            // Attempt to write a very small WAV slice file if possible (best-effort).
            try
            {
                ExtractWavSegment(wavFilePath, outPath, i, i + 2);
            }
            catch
            {
                // ignore; continue to attempt inference (may fall back).
            }

            double conf = 0.0;
            try
            {
                // Build mel-spectrogram matching Python config and produce a [1,1,256,256] tensor.
                var samples = LoadAndResampleMono(outPath, TargetSampleRate);

                // Ensure length matches DurationMs (padding/truncating).
                int targetSamples = (int)((DurationMs / 1000.0) * TargetSampleRate);
                var segmentSamples = EnsureLength(samples, targetSamples);

                // Compute mel spectrogram (NMels x frames).
                var mel = ComputeMelSpectrogram(segmentSamples, TargetSampleRate, NFFT, HopLength, NMels, Fmin, Fmax);

                // Convert to dB and clip top_db.
                var melDb = PowerToDb(mel);
                var clipped = ClipTopDb(melDb, TopDb);

                // Resize or pad to ModelWidth frames (time axis).
                var resized = ResizeTimeAxis(clipped, ModelWidth);

                // Normalize to roughly [-1,1] or [0,1] depending on model expectation.
                // Here we scale dB to [0,1].
                var normalized = NormalizeTo01(resized);

                // Create tensor [1,1,256,256].
                var inputData = new float[1 * 1 * ModelHeight * ModelWidth];
                for (int r = 0; r < ModelHeight; r++)
                {
                    for (int c = 0; c < ModelWidth; c++)
                    {
                        inputData[r * ModelWidth + c] = (float)normalized[r][c];
                    }
                }
                var tensor = new DenseTensor<float>(inputData, new int[] { 1, 1, ModelHeight, ModelWidth });

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", tensor) };

                using var outputs = _session.Run(inputs);
                var first = outputs.FirstOrDefault();
                if (first != null)
                {
                    try
                    {
                        var outTensor = first.AsTensor<float>();
                        float outVal = outTensor.First();
                        conf = Math.Max(0.0, Math.Min(1.0, outVal));
                    }
                    catch
                    {
                        conf = 0.0;
                    }
                }
                else
                {
                    conf = 0.0;
                }
            }
            catch
            {
                conf = 0.0;
            }

            twoSecConfidences.Add(conf);
        }

        // Aggregate (same as previous logic).
        List<double> perSecondConf = BuildPerSecondConf(twoSecConfidences);

        var localPreds = perSecondConf.Select(c => c > _threshold ? 1 : 0).ToList();
        int globalPrediction = localPreds.Sum() >= _minNumPositiveCallsThreshold ? 1 : 0;
        var above = perSecondConf.Where(c => c > _threshold).ToList();
        double globalConfidence = above.Count > 0 ? above.Average() * 100.0 : 0.0;

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

        try { if (Directory.Exists(localDir)) Directory.Delete(localDir, true); } catch { /* ignore */ }

        result["submission"] = submission;
        result["local_predictions"] = localPreds;
        result["local_confidences"] = perSecondConf;
        result["global_prediction"] = globalPrediction;
        result["global_confidence"] = globalConfidence;

        return result;
    }

    private static List<double> BuildPerSecondConf(List<double> twoSecConfidences)
    {
        var perSecondConf = new List<double>();
        if (twoSecConfidences.Count == 0) return perSecondConf;
        if (twoSecConfidences.Count == 1) { perSecondConf.Add(twoSecConfidences[0]); return perSecondConf; }

        for (int i = 0; i < twoSecConfidences.Count - 1; i++)
        {
            perSecondConf.Add((twoSecConfidences[i] + twoSecConfidences[i + 1]) / 2.0);
        }

        perSecondConf[0] = twoSecConfidences[0];
        perSecondConf.Add(twoSecConfidences[twoSecConfidences.Count - 1]);
        return perSecondConf;
    }

    private static float[] EnsureLength(float[] samples, int targetSamples)
    {
        if (samples.Length == targetSamples)
        {
            return samples;
        }
        var outArr = new float[targetSamples];
        if (samples.Length >= targetSamples)
        {
            Array.Copy(samples, 0, outArr, 0, targetSamples);
            return outArr;
        }

        // Pad center with zeros.
        Array.Copy(samples, 0, outArr, 0, samples.Length);
        for (int i = samples.Length; i < targetSamples; i++)
        {
            outArr[i] = 0f;
        }
        return outArr;
    }

    private static float[] LoadAndResampleMono(string path, int targetSampleRate)
    {
        if (!File.Exists(path))
        {
            return Array.Empty<float>();
        }

        using var reader = new AudioFileReader(path); // yields float samples, stereo->interleaved if stereo
        ISampleProvider sampleProvider = reader;
        if (Downmix && reader.WaveFormat.Channels > 1)
        {
            sampleProvider = new StereoToMonoSampleProvider(reader) { LeftVolume = 0.5f, RightVolume = 0.5f };
        }

        if (reader.WaveFormat.SampleRate != targetSampleRate)
        {
            var resampler = new WdlResamplingSampleProvider(sampleProvider, targetSampleRate);
            sampleProvider = resampler;
        }

        // Read all samples.
        var samples = new List<float>();
        float[] buffer = new float[8192];
        int read;
        while ((read = sampleProvider.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < read; i++)
            {
                samples.Add(buffer[i]);
            }
        }
        return samples.ToArray();
    }

    private static double[][] ComputeMelSpectrogram(float[] samples, int sampleRate, int nfft, int hopLength, int n_mels, double fmin, double fmax)
    {
        int fftSize = nfft;
        int winLength = nfft;
        int hop = hopLength;

        int frameCount = Math.Max(1, 1 + ((samples.Length - winLength) / hop));
        if (samples.Length < winLength)
        {
            // Pad.
            var padded = new float[winLength];
            Array.Copy(samples, padded, Math.Min(samples.Length, winLength));
            samples = padded;
            frameCount = 1;
        }

        // Precompute window.
        var window = Window.Hann(winLength);

        // FFT bins.
        int bins = fftSize / 2 + 1;

        // Create mel filterbank.
        var melFilter = MelFilterBank(n_mels, fftSize, sampleRate, fmin, fmax);

        // Power spectrogram (n_mels x frames).
        var melSpectrogram = CreateJaggedDoubleArray(n_mels, frameCount);

        // Temporary arrays.
        var fftBuffer = new Complex[fftSize];

        for (int frame = 0; frame < frameCount; frame++)
        {
            int offset = frame * hop;

            // Fill fftBuffer real part with windowed samples, zero pad remainder.
            for (int j = 0; j < fftSize; j++)
            {
                double v = 0.0;
                int idx = offset + j;
                if (idx < samples.Length)
                {
                    v = samples[idx];
                }
                fftBuffer[j] = new Complex(v * window[j], 0.0);
            }

            // Perform FFT.
            Fourier.Forward(fftBuffer, FourierOptions.Matlab);

            // Compute magnitude^2 (power) for bins.
            var power = new double[bins];
            for (int b = 0; b < bins; b++)
            {
                var c = fftBuffer[b];
                power[b] = (c.Real * c.Real + c.Imaginary * c.Imaginary);
            }

            // Apply mel filters.
            for (int m = 0; m < n_mels; m++)
            {
                double sum = 0.0;
                var filter = melFilter[m];
                for (int b = 0; b < bins; b++)
                {
                    sum += power[b] * filter[b];
                }
                melSpectrogram[m][frame] = sum;
            }
        }

        return melSpectrogram;
    }

    // Create mel filter bank (n_mels x bins).
    private static double[][] MelFilterBank(int n_mels, int n_fft, int sampleRate, double fmin, double fmax)
    {
        int bins = n_fft / 2 + 1;
        double[] fftFreqs = new double[bins];
        for (int i = 0; i < bins; i++)
        {
            fftFreqs[i] = (double)i * sampleRate / n_fft;
        }

        double minMel = HertzToMel(fmin);
        double maxMel = HertzToMel(fmax);
        double[] mels = new double[n_mels + 2];
        for (int i = 0; i < mels.Length; i++)
        {
            mels[i] = minMel + (maxMel - minMel) * i / (n_mels + 1);
        }
        double[] hz = mels.Select(HertzFromMel).ToArray();

        var filters = new double[n_mels][];
        for (int m = 0; m < n_mels; m++)
        {
            filters[m] = new double[bins];
            double f_m_left = hz[m];
            double f_m = hz[m + 1];
            double f_m_right = hz[m + 2];

            for (int k = 0; k < bins; k++)
            {
                double f = fftFreqs[k];
                double weight = 0.0;
                if (f >= f_m_left && f <= f_m)
                {
                    weight = (f - f_m_left) / (f_m - f_m_left);
                }
                else if (f >= f_m && f <= f_m_right)
                {
                    weight = (f_m_right - f) / (f_m_right - f_m);
                }
                filters[m][k] = weight;
            }
        }
        return filters;
    }

    private static double HertzToMel(double hz) => 2595.0 * Math.Log10(1.0 + hz / 700.0);
    private static double HertzFromMel(double mel) => 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);

    private static double[][] CreateJaggedDoubleArray(int rows, int cols)
    {
        var a = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            a[i] = new double[cols];
        }
        return a;
    }

    private static double[][] PowerToDb(double[][] power)
    {
        int rows = power.Length;
        int cols = power[0].Length;
        var outArr = CreateJaggedDoubleArray(rows, cols);
        double amin = 1e-10;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                double val = power[r][c];
                outArr[r][c] = 10.0 * Math.Log10(Math.Max(amin, val));
            }
        }
        return outArr;
    }

    private static double[][] ClipTopDb(double[][] S_db, double top_db)
    {
        int rows = S_db.Length;
        int cols = S_db[0].Length;
        var result = CreateJaggedDoubleArray(rows, cols);
        double max = double.NegativeInfinity;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (S_db[r][c] > max) max = S_db[r][c];
            }
        }

        double minAllowed = max - top_db;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result[r][c] = Math.Max(S_db[r][c], minAllowed);
            }
        }
        return result;
    }

    private static double[][] ResizeTimeAxis(double[][] mel, int targetWidth)
    {
        int rows = mel.Length;
        int cols = mel[0].Length;
        var outArr = CreateJaggedDoubleArray(rows, targetWidth);

        if (cols == targetWidth)
        {
            for (int r = 0; r < rows; r++)
            {
                Array.Copy(mel[r], outArr[r], cols);
            }
            return outArr;
        }
        if (cols < targetWidth)
        {
            // Pad with last column.
            for (int r = 0; r < rows; r++)
            {
                int c = 0;
                for (; c < cols; c++)
                {
                    outArr[r][c] = mel[r][c];
                }
                for (; c < targetWidth; c++)
                {
                    outArr[r][c] = mel[r][cols - 1];
                }
            }
            return outArr;
        }

        // cols > targetWidth: downsample by averaging groups.
        double scale = cols / (double)targetWidth;
        for (int r = 0; r < rows; r++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                double start = x * scale;
                double end = (x + 1) * scale;
                int iStart = (int)Math.Floor(start);
                int iEnd = (int)Math.Min(cols - 1, Math.Floor(end));
                if (iStart > iEnd)
                {
                    outArr[r][x] = mel[r][Math.Min(iStart, cols - 1)];
                    continue;
                }
                double sum = 0.0;
                int count = 0;
                for (int k = iStart; k <= iEnd; k++)
                {
                    sum += mel[r][k]; count++;
                }
                outArr[r][x] = count > 0 ? sum / count : mel[r][iStart];
            }
        }
        return outArr;
    }

    private static double[][] NormalizeTo01(double[][] arr)
    {
        int rows = arr.Length;
        int cols = arr[0].Length;
        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var v = arr[r][c];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        var outArr = CreateJaggedDoubleArray(rows, cols);
        double range = max - min;
        if (range <= 0.0)
        {
            range = 1.0;
        }
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                outArr[r][c] = (arr[r][c] - min) / range;
            }
        }
        return outArr;
    }

    // WAV duration helper (unchanged)
    private static double GetWavDurationSeconds(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        var riff = new string(br.ReadChars(4));
        if (riff != "RIFF")
        {
            return 0.0;
        }
        br.ReadInt32();
        var wave = new string(br.ReadChars(4));
        if (wave != "WAVE")
        {
            return 0.0;
        }

        int sampleRate = 0;
        short bitsPerSample = 0;
        short channels = 0;
        long dataChunkSize = 0;

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
                short audioFormat = br.ReadInt16();
                channels = br.ReadInt16();
                sampleRate = br.ReadInt32();
                int byteRate = br.ReadInt32();
                short blockAlign = br.ReadInt16();
                bitsPerSample = br.ReadInt16();
                int fmtExtra = chunkSize - 16;
                if (fmtExtra > 0)
                {
                    br.ReadBytes(fmtExtra);
                }
            }
            else if (chunkId == "data")
            {
                dataChunkSize = chunkSize;
                break;
            }
            else
            {
                br.ReadBytes(chunkSize);
            }
        }

        if (sampleRate <= 0 || channels <= 0 || bitsPerSample == 0 || dataChunkSize == 0) return 0.0;

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
        if (riff != "RIFF")
        {
            throw new InvalidOperationException("Not a RIFF file");
        }
        int _ = br.ReadInt32();
        var wave = new string(br.ReadChars(4));
        if (wave != "WAVE")
        {
            throw new InvalidOperationException("Not a WAVE file");
        }

        // Parse fmt and data positions.
        int sampleRate = 0;
        short bitsPerSample = 0;
        short channels = 0;
        long dataChunkPos = -1;
        int dataChunkSize = 0;

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
        {
            throw new InvalidOperationException("Unsupported WAV format for slicing");
        }

        int bytesPerSample = (bitsPerSample / 8) * channels;
        long startSample = (long)startSec * sampleRate;
        long endSample = (long)endSec * sampleRate;
        long totalSamples = dataChunkSize / bytesPerSample;

        if (startSample >= totalSamples) startSample = totalSamples;
        if (endSample > totalSamples) endSample = totalSamples;
        if (endSample < startSample) endSample = startSample;

        long startByte = dataChunkPos + (startSample * bytesPerSample);
        long segmentBytes = (endSample - startSample) * bytesPerSample;

        // Build a minimal WAV file with same fmt parameters and the sliced data.
        using var outFs = File.Create(destPath);
        using var bw = new BinaryWriter(outFs);

        // RIFF header.
        bw.Write(Encoding.ASCII.GetBytes("RIFF"));
        bw.Write((int)(36 + segmentBytes)); // file size - 8
        bw.Write(Encoding.ASCII.GetBytes("WAVE"));

        // fmt chunk (copy from source fmt chunk if possible; otherwise write minimal PCM fmt).
        // We'll write a standard PCM fmt chunk (16 bytes).
        bw.Write(Encoding.ASCII.GetBytes("fmt "));
        bw.Write(16); // PCM fmt chunk size
        bw.Write((short)1); // audio format PCM
        bw.Write(channels);
        bw.Write(sampleRate);
        bw.Write((int)(sampleRate * channels * (bitsPerSample / 8)));
        bw.Write((short)(channels * (bitsPerSample / 8)));
        bw.Write(bitsPerSample);

        // Data chunk.
        bw.Write(Encoding.ASCII.GetBytes("data"));
        bw.Write((int)segmentBytes);

        // Copy bytes.
        fs.Position = startByte;
        const int bufferSize = 8192;
        byte[] buffer = new byte[bufferSize];
        long remaining = segmentBytes;
        while (remaining > 0)
        {
            int toRead = remaining > bufferSize ? bufferSize : (int)remaining;
            int read = fs.Read(buffer, 0, toRead);
            if (read <= 0)
            {
                break;
            }
            bw.Write(buffer, 0, read);
            remaining -= read;
        }
    }
}
