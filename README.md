# cambench

A tiny **C++20 / OpenCV / yaml-cpp** benchmarking & A/B validation tool for simple video pipelines.

## ✨ Features
- Run a fixed image pipeline:  
  `resize → blur → Sobel`
- Time each stage per frame and write **CSV metrics**
- Save processed frames for inspection
- Run **A/B validation** between two runs:
  - Compare outputs with **PSNR** and **SSIM** thresholds
  - Generate a markdown **report** with summary stats

## 🛠️ Built With
- **C++20**
- **OpenCV**
- **yaml-cpp**
