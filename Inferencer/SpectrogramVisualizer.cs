// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;

public static class SpectrogramVisualizer
{
    /// <summary>
    /// Create a spectrogram image for the given WAV file.
    /// Tries to use ffmpeg showspectrumpic filter if ffmpeg is available.
    /// Falls back to a small placeholder file if not.
    /// Returns the path to the generated PNG (existing file).
    /// </summary>
    public static string WriteSpectrogram(string wavFilePath)
    {
        if (string.IsNullOrWhiteSpace(wavFilePath))
            throw new ArgumentNullException(nameof(wavFilePath));

        var directory = Path.GetDirectoryName(wavFilePath) ?? ".";
        var basename = Path.GetFileNameWithoutExtension(wavFilePath);
        var outputPath = Path.Combine(directory, basename + ".png");

        // Try ffmpeg showspectrumpic (creates a spectrogram image)
        try
        {
            // size 1280x480 (two halves of 640x480 in original python)
            var args = $"-y -i \"{wavFilePath}\" -lavfi showspectrumpic=s=1280x480:legend=0 \"{outputPath}\"";
            RunFfmpeg(args);
            if (File.Exists(outputPath))
                return outputPath;
        }
        catch
        {
            // fallthrough to placeholder
        }

        // Fallback: create a simple placeholder PNG (1280x480) with text
        try
        {
            using var bmp = new Bitmap(1280, 480);
            using var g = Graphics.FromImage(bmp);
            g.Clear(Color.DarkSlateGray);
            var font = new Font(FontFamily.GenericSansSerif, 20, FontStyle.Bold);
            var brush = Brushes.White;
            var text = "Spectrogram unavailable";
            var textSize = g.MeasureString(text, font);
            g.DrawString(text, font, brush, (bmp.Width - textSize.Width) / 2, (bmp.Height - textSize.Height) / 2);
            bmp.Save(outputPath, ImageFormat.Png);
            return outputPath;
        }
        catch
        {
            // As a last resort, write a textual placeholder file so callers still get a path.
            File.WriteAllText(outputPath, "spectrogram placeholder");
            return outputPath;
        }
    }

    /// <summary>
    /// Annotate an existing spectrogram image with rectangles for positive predictions and confidence text.
    /// Expects data to contain "local_predictions" and "local_confidences" (enumerables).
    /// If annotation fails it will try best-effort and leave original image intact.
    /// </summary>
    public static void WriteAnnotationsOnSpectrogram(string wavFilePath, string wavTimestamp, IDictionary<string, object> data, string specOutputPath)
    {
        if (string.IsNullOrWhiteSpace(specOutputPath))
            throw new ArgumentNullException(nameof(specOutputPath));

        // Validate/normalize data arrays
        var localPredictions = ExtractIntListFromData(data, "local_predictions");
        var localConfidences = ExtractDoubleListFromData(data, "local_confidences");

        if (localPredictions == null || localConfidences == null || localPredictions.Count == 0)
        {
            // Nothing to annotate; still ensure timestamp overlay if possible
            TryOverlayTimestamp(specOutputPath, wavTimestamp);
            return;
        }

        // Try to annotate using System.Drawing (best-effort; may not be supported on all platforms)
        try
        {
            using var image = Image.FromFile(specOutputPath);
            using var bmp = new Bitmap(image);
            using var g = Graphics.FromImage(bmp);

            int width = bmp.Width;
            int height = bmp.Height;

            int num = localPredictions.Count;
            int annotationWidth = Math.Max(1, (int)Math.Floor(width / (double)num));

            var rectPen = new Pen(Color.White, 2);
            var font = new Font(FontFamily.GenericSansSerif, 12, FontStyle.Bold);
            var brush = Brushes.White;

            for (int i = 0; i < num; i++)
            {
                if (localPredictions[i] == 1)
                {
                    var x1 = i * annotationWidth;
                    var rect = new Rectangle(x1, 20, annotationWidth, Math.Max(1, height - 40));
                    g.DrawRectangle(rectPen, rect);

                    string confText = i < localConfidences.Count ? (localConfidences[i].ToString(CultureInfo.InvariantCulture)) : "n/a";
                    g.DrawString(confText, font, brush, x1 + 5, height / 2f - 10);
                }
            }

            // draw timestamp top-left
            if (!string.IsNullOrWhiteSpace(wavTimestamp))
            {
                g.DrawString(wavTimestamp, font, brush, 5, 2);
            }

            // Save back
            bmp.Save(specOutputPath, ImageFormat.Png);
            return;
        }
        catch
        {
            // If annotation via System.Drawing fails, fallback to writing a small sidecar .ann.txt file with annotations
            try
            {
                var annPath = Path.ChangeExtension(specOutputPath, ".ann.txt");
                using var sw = new StreamWriter(annPath, false);
                sw.WriteLine("timestamp: " + wavTimestamp);
                sw.WriteLine("predictions:");
                for (int i = 0; i < localPredictions.Count; i++)
                {
                    var conf = i < localConfidences.Count ? localConfidences[i].ToString(CultureInfo.InvariantCulture) : "n/a";
                    sw.WriteLine($"{i}: pred={localPredictions[i]}, conf={conf}");
                }
            }
            catch
            {
                // ignore
            }
        }
    }

    // Helpers

    private static List<int>? ExtractIntListFromData(IDictionary<string, object> data, string key)
    {
        if (data == null || !data.TryGetValue(key, out var val) || val == null)
            return null;

        try
        {
            // Prefer strongly-typed enumerables first, then fall back to object sequences
            if (val is IEnumerable<int> ei)
                return ei.ToList();

            if (val is int singleInt)
                return new List<int> { singleInt };

            if (val is IEnumerable<object> eo)
                return eo.Select(o => Convert.ToInt32(o)).ToList();

            if (val is object[] arr)
                return arr.Select(o => Convert.ToInt32(o)).ToList();

            return null;
        }
        catch
        {
            return null;
        }
    }

    private static List<double>? ExtractDoubleListFromData(IDictionary<string, object> data, string key)
    {
        if (data == null || !data.TryGetValue(key, out var val) || val == null)
            return null;

        try
        {
            if (val is IEnumerable<double> ed)
                return ed.ToList();

            if (val is double singleDouble)
                return new List<double> { singleDouble };

            if (val is IEnumerable<object> eo)
                return eo.Select(o => Convert.ToDouble(o)).ToList();

            if (val is object[] arr)
                return arr.Select(o => Convert.ToDouble(o)).ToList();

            return null;
        }
        catch
        {
            return null;
        }
    }

    private static void TryOverlayTimestamp(string specOutputPath, string wavTimestamp)
    {
        if (string.IsNullOrWhiteSpace(wavTimestamp) || !File.Exists(specOutputPath))
            return;

        try
        {
            using var image = Image.FromFile(specOutputPath);
            using var bmp = new Bitmap(image);
            using var g = Graphics.FromImage(bmp);
            var font = new Font(FontFamily.GenericSansSerif, 12, FontStyle.Bold);
            g.DrawString(wavTimestamp, font, Brushes.White, 5, 2);
            bmp.Save(specOutputPath, ImageFormat.Png);
        }
        catch
        {
            // ignore failures
        }
    }

    private static void RunFfmpeg(string arguments)
    {
        var psi = new ProcessStartInfo
        {
            FileName = "ffmpeg",
            Arguments = arguments,
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
            throw new InvalidOperationException($"ffmpeg exited with code {proc.ExitCode}. stderr: {stderr}");
    }
}