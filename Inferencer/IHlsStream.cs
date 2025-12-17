// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT
using System;

public interface IHlsStream
{
    /// <summary>
    /// Retrieve the next clip for processing.
    /// Returns a tuple: (clipPath, startTimestampIsoString, newCurrentClipEndTime).
    /// If no clip is available, clipPath should be null.
    /// </summary>
    (string? clipPath, string? startTimestamp, DateTime newCurrentClipEndTime) GetNextClip(DateTime currentClipEndTime);

    /// <summary>
    /// Indicates whether the stream is finished (true) or still producing data (false).
    /// </summary>
    bool IsStreamOver();
}