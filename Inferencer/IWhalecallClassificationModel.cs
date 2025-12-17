// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT
using System;
using System.Collections.Generic;

public interface IWhalecallClassificationModel
{
    /// <summary>
    /// Run prediction for the provided WAV file path.
    /// Must return a dictionary-like object with keys:
    ///  - "submission" (object/list)
    ///  - "local_predictions" (list/array)
    ///  - "local_confidences" (list/array)
    ///  - "global_prediction" (int)
    ///  - "global_confidence" (double)
    /// </summary>
    IDictionary<string, object> Predict(string wavFilePath);
}