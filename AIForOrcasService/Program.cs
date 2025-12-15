// Copyright (c) AIForOrcas Service contributors
// SPDX-License-Identifier: MIT
using AIForOrcasService;

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddHostedService<Worker>();

var host = builder.Build();
host.Run();
