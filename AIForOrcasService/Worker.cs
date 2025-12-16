// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AIForOrcasService
{
    public class Worker : BackgroundService
    {
        private readonly ILogger<Worker> _logger;
        private readonly InferenceSession _session;

        public Worker(ILogger<Worker> logger)
        {
            _logger = logger;

            var curdir = Directory.GetCurrentDirectory();

            // Load ONNX model once at startup
            _session = new InferenceSession("model.onnx");
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                // Example: create dummy input (replace with hydrophone data)
                var inputData = new float[] { 1.0f, 2.0f, 3.0f };
                var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 3 });

                var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

                // Run inference
                using var results = _session.Run(inputs);

                foreach (var result in results)
                {
                    var outputTensor = result.AsTensor<float>();
                    _logger.LogInformation($"Output: {string.Join(", ", outputTensor)}");
                }

                // Wait before next cycle
                await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
            }
        }

        public override void Dispose()
        {
            _session.Dispose();
            base.Dispose();
        }
    }
}
