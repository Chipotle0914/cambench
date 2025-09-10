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


### 0) Install prerequisites (Ubuntu/WSL)
```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev libyaml-cpp-dev
```
### 1) Build (Release)
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j
./build/bin/cambench --help 
```

### 2) Run pipeline A (smaller blur kernel → sharper edges)
```
./bin/cambench run --config ../configs/a_fast.yaml
```

### 3) Run pipeline B (larger blur kernel → smoother edges)
```
./bin/cambench run --config ../configs/b_quality.yaml
```

### 4) Validate A vs B (PSNR/SSIM thresholds)
```
./build/bin/cambench validate \
  --ref ./out/run_b \
  --test ./out/run_a \
  --out ./out/validate_ab \
  --psnr-min 30 \
  --ssim-min 0.95
```
### 5) One-liner without YAML
```
./build/bin/cambench run \
  --input ./data/2_ppl_running.mp4 \
  --out ./out/run_cli \
  --frames 300 --warmup 20 --width 640 --height 360 --save \
  --blur-ksize 5
```
### 6) Clean outputs and re-run
```
rm -rf ./out/*
```

