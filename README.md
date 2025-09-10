#cambench

A tiny C++/OpenCV benchmarking & A/B validation tool for simple video pipelines.
It can:

run a fixed image pipeline (resize → blur → Sobel),

time each stage per frame and write CSV metrics,

save processed frames,

run A/B validation between two runs with PSNR/SSIM thresholds and produce a markdown report.

Built with C++20, OpenCV, and yaml-cpp.
