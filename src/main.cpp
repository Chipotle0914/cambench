#include <iostream>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm> 
#include <numeric>
#include <cmath>
#include <fstream>
#include <tuple>
#include "cambench/version.hpp"
#include "cambench/metrics.hpp"
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;


static void usage() {
    std::cout <<
      "cambench " << cambench::kVersion << "\n\n"
      "Usage:\n"
      "  cambench capture --input <video.mp4> --out <folder> [--frames N]\n"
      "  cambench run     --config <cfg.yaml>\n"
      "  cambench run     --input <video.mp4> --out <folder> [--frames N] [--width W --height H] [--save] [--warmup N] [--blur-ksize K]\n" 
      "  cambench validate --ref <folder> --test <folder> [--out <folder>] [--psnr-min X] [--ssim-min Y]\n";

}


static std::string getArg(const std::vector<std::string>& args, const std::string& key, const std::string& def="") {
    for (size_t i=0; i+1<args.size(); ++i)
        if (args[i] == key) return args[i+1];
    return def;
}


static bool hasFlag(const std::vector<std::string>& args, const std::string& key) {
    for (auto& a : args) if (a == key) return true;
    return false;
}


static double psnr(const cv::Mat& I1, const cv::Mat& I2) {
    cv::Mat diff; cv::absdiff(I1, I2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    cv::Scalar s = cv::sum(diff);
    double sse = s[0] + s[1] + s[2];
    if (sse <= 1e-10) return 100.0;
    double mse = sse / (double)(I1.total() * I1.channels());
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}


static double ssim(const cv::Mat& i1, const cv::Mat& i2) {
    cv::Mat I1, I2; 
    if (i1.channels() == 3) { cv::cvtColor(i1, I1, cv::COLOR_BGR2GRAY); } else { I1 = i1; }
    if (i2.channels() == 3) { cv::cvtColor(i2, I2, cv::COLOR_BGR2GRAY); } else { I2 = i2; }
    I1.convertTo(I1, CV_32F); I2.convertTo(I2, CV_32F);

    const double C1 = 6.5025, C2 = 58.5225;
    cv::Mat mu1, mu2; 
    cv::GaussianBlur(I1, mu1, cv::Size(11,11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11,11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12, t1, t2, t3;
    cv::GaussianBlur(I1.mul(I1), sigma1_2, cv::Size(11,11), 1.5);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2.mul(I2), sigma2_2, cv::Size(11,11), 1.5);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1.mul(I2), sigma12,  cv::Size(11,11), 1.5);
    sigma12 -= mu1_mu2;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = (mu1_2 + mu2_2 + C1).mul(sigma1_2 + sigma2_2 + C2);

    cv::Mat ssim_map = (t1.mul(t2)).mul(1.0 / t3);
    cv::Scalar mssim = cv::mean(ssim_map);
    return mssim[0];
}


int cmd_validate(const std::vector<std::string>& args) {
    std::string refDir  = getArg(args, "--ref");
    std::string testDir = getArg(args, "--test");
    std::string outDir  = getArg(args, "--out", "out/validate");
    double psnrMin = 30.0;
    double ssimMin = 0.95;
    try { psnrMin = std::stod(getArg(args, "--psnr-min", "30.0")); } catch (...) {}
    try { ssimMin = std::stod(getArg(args, "--ssim-min", "0.95")); } catch (...) {}

    if (refDir.empty() || testDir.empty()) {
        std::cerr << "Missing --ref or --test folder\n";
        return 2;
    }
    std::error_code ec; fs::create_directories(outDir, ec);

    std::vector<std::string> files;
    for (auto& p : fs::directory_iterator(refDir)) {
        if (!p.is_regular_file()) continue;
        auto name = p.path().filename().string();
        if (name.rfind("proc_", 0) == 0 && p.path().extension() == ".png") {
            if (fs::exists(fs::path(testDir) / name)) files.push_back(name);
        }
    }
    if (files.empty()) {
        std::cerr << "No matching proc_*.png files in both folders.\n";
        return 3;
    }
    std::sort(files.begin(), files.end());

    cambench::CsvWriter csv(outDir + "/metrics.csv", "file,psnr,ssim,pass");
    std::vector<double> psnrs, ssims;
    int passCount = 0;

    for (auto& name : files) {
        cv::Mat ref = cv::imread((fs::path(refDir)/name).string(), cv::IMREAD_COLOR);
        cv::Mat tst = cv::imread((fs::path(testDir)/name).string(), cv::IMREAD_COLOR);
        if (ref.empty() || tst.empty() || ref.size() != tst.size()) {
            csv.row(name, -1, -1, 0);
            continue;
        }
        double p = psnr(ref, tst);
        double s = ssim(ref, tst);
        bool ok = (p >= psnrMin) && (s >= ssimMin);
        csv.row(name, p, s, ok ? 1 : 0);
        psnrs.push_back(p);
        ssims.push_back(s);
        if (ok) ++passCount;
    }

    auto summarize = [](std::vector<double> v) {
        if (v.empty()) return std::tuple<double,double,double>(0.0,0.0,0.0);
        std::sort(v.begin(), v.end());
        double avg = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double med = v[v.size()/2];
        double mn  = v.front();
        return std::tuple<double,double,double>(avg, med, mn);
    };
    auto [psnrAvg, psnrMed, psnrMinSeen] = summarize(psnrs);
    auto [ssimAvg, ssimMed, ssimMinSeen] = summarize(ssims);


    {
        std::ofstream md(outDir + "/report.md");
        md << "# Validate Report\n\n";
        md << "**Ref:** " << refDir << "\n\n";
        md << "**Test:** " << testDir << "\n\n";
        md << "**Thresholds:** PSNR ≥ " << psnrMin << ", SSIM ≥ " << ssimMin << "\n\n";
        md << "Frames checked: " << files.size() << "\n";
        md << "Pass frames: " << passCount << " / " << files.size() << "\n\n";
        md << "## Summary (over frames)\n";
        md << "- PSNR: avg " << psnrAvg << ", median " << psnrMed << ", min " << psnrMinSeen << "\n";
        md << "- SSIM: avg " << ssimAvg << ", median " << ssimMed << ", min " << ssimMinSeen << "\n";
    }

    std::cout << "Validated " << files.size() << " frames. "
              << "PSNR(avg/med/min): " << psnrAvg << "/" << psnrMed << "/" << psnrMinSeen << ", "
              << "SSIM(avg/med/min): " << ssimAvg << "/" << ssimMed << "/" << ssimMinSeen << "\n"
              << "Report: " << (outDir + "/report.md") << "\n";
    return 0;
}



int cmd_run(const std::vector<std::string>& args) {
    int warmup = 10;
    try { warmup = std::stoi(getArg(args, "--warmup", "10")); } catch (...) {}
    std::string input  = getArg(args, "--input");
    std::string outdir = getArg(args, "--out", "out/run");
    int frames = 0;
    try { frames = std::stoi(getArg(args, "--frames", "0")); } catch (...) {}
    int W = 0, H = 0;
    try { W = std::stoi(getArg(args, "--width",  "640")); } catch (...) {}
    try { H = std::stoi(getArg(args, "--height", "360")); } catch (...) {}
    int blurK = 3;
    try { blurK = std::stoi(getArg(args, "--blur-ksize", "3")); } catch (...) {}
    if (blurK % 2 == 0) ++blurK;      
    blurK = std::max(1, blurK);
    bool save_imgs = hasFlag(args, "--save");

    if (input.empty()) { std::cerr << "Missing --input <video.mp4>\n"; return 2; }
    std::error_code ec; fs::create_directories(outdir, ec);

    cv::VideoCapture cap(input);
    if (!cap.isOpened()) {
        cap.open(input, cv::CAP_FFMPEG);
        if (!cap.isOpened()) cap.open(input, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open: " << input << "\n";
            return 3;
        }
    }

    cambench::CsvWriter csv(outdir + "/metrics.csv",
        "frame,read_ms,resize_ms,blur_ms,sobel_ms,total_ms,out_w,out_h");

    std::vector<double> totals_ms;
    cambench::Timer t; t.start();
    int i = 0, processed = 0;

    cv::Mat src, resized, blurred, gray, sobelx, sobely, sobel;
    while (true) {
        if (frames > 0 && i >= frames) break;

        double t0 = t.ms();
        if (!cap.read(src) || src.empty()) break;
        double t_read = t.ms() - t0;

        double t1 = t.ms();
        if (W > 0 && H > 0) cv::resize(src, resized, cv::Size(W, H));
        else                resized = src;
        double t_resize = t.ms() - t1;

        double t2 = t.ms();
        cv::GaussianBlur(resized, blurred, cv::Size(blurK, blurK), 0);
        double t_blur = t.ms() - t2;

        double t3 = t.ms();
        cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);
        cv::Sobel(gray, sobelx, CV_16S, 1, 0, 3);
        cv::Sobel(gray, sobely, CV_16S, 0, 1, 3);
        cv::Mat absx, absy; cv::convertScaleAbs(sobelx, absx); cv::convertScaleAbs(sobely, absy);
        cv::addWeighted(absx, 0.5, absy, 0.5, 0, sobel);
        double t_sobel = t.ms() - t3;

        double total_ms = t_read + t_resize + t_blur + t_sobel;
        ++processed; ++i;

        if (i-1 >= warmup) {
            totals_ms.push_back(total_ms);
        }

        if (save_imgs) {
            std::ostringstream name;
            name << outdir << "/proc_" << std::setw(6) << std::setfill('0') << i-1 << ".png";
            cv::imwrite(name.str(), sobel);
        }

        csv.row(i-1, t_read, t_resize, t_blur, t_sobel, total_ms, resized.cols, resized.rows);
    }

    auto summarize = [](std::vector<double> v) {
        if (v.empty()) return std::tuple<double,double,double>(0.0, 0.0, 0.0);
        std::sort(v.begin(), v.end());
        double avg = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double med = v[v.size() / 2];
        size_t p95_idx = std::min(v.size() - 1, static_cast<size_t>(std::lround(v.size() * 0.95)));
        double p95 = v[p95_idx];
        return std::tuple<double,double,double>(avg, med, p95);
    };

    auto [avg_ms, med_ms, p95_ms] = summarize(totals_ms);
    double fps_avg = (avg_ms > 0.0) ? 1000.0 / avg_ms : 0.0;
    double fps_med = (med_ms > 0.0) ? 1000.0 / med_ms : 0.0;
    double fps_p95 = (p95_ms > 0.0) ? 1000.0 / p95_ms : 0.0;

    int effective_warmup = std::max(0, processed - static_cast<int>(totals_ms.size()));
    std::cout
        << "Processed " << processed << " frames (warmup requested " << warmup
        << ", effective " << effective_warmup << ", measured " << totals_ms.size()
        << ") to " << outdir << "\n"
        << "  avg  : " << avg_ms  << " ms  (" << fps_avg << " FPS)\n"
        << "  median: " << med_ms  << " ms  (" << fps_med << " FPS)\n"
        << "  p95  : " << p95_ms  << " ms  (" << fps_p95 << " FPS)\n";
}


int cmd_run_config(const std::string& cfgPath) {
    YAML::Node cfg = YAML::LoadFile(cfgPath);


    std::string input  = cfg["input"]  ? cfg["input"].as<std::string>()  : "";
    std::string outdir = cfg["out"]    ? cfg["out"].as<std::string>()    : "out/run";
    int frames         = cfg["frames"] ? cfg["frames"].as<int>()         : 0;
    int warmup         = cfg["warmup"] ? cfg["warmup"].as<int>()         : 10;


    int W = 0, H = 0;
    int ksize = 3;
    bool save = cfg["save"] ? cfg["save"].as<bool>() : false;


    YAML::Node stages = cfg["stages"];
    if (stages && stages.IsSequence()) {
        for (const auto& st : stages) { 
            if (!st["op"]) continue;
            std::string op = st["op"].as<std::string>();

            if (op == "resize") {
                if (st["width"])  W = st["width"].as<int>();
                if (st["height"]) H = st["height"].as<int>();
            } else if (op == "gaussian_blur") {
                if (st["ksize"])  ksize = std::max(1, st["ksize"].as<int>());
                if ((ksize % 2) == 0) ++ksize; 
            }
        }
    }


    std::vector<std::string> args;
    if (!input.empty())  { args.push_back("--input");  args.push_back(input); }
    if (!outdir.empty()) { args.push_back("--out");    args.push_back(outdir); }
    args.push_back("--frames");  args.push_back(std::to_string(frames));
    args.push_back("--warmup");  args.push_back(std::to_string(warmup));
    if (W > 0 && H > 0) {
        args.push_back("--width");  args.push_back(std::to_string(W));
        args.push_back("--height"); args.push_back(std::to_string(H));
    }

    if (save) args.push_back("--save");

    if (ksize > 0) {
        args.push_back("--blur-ksize");
        args.push_back(std::to_string(ksize));
    }

    return cmd_run(args);

}


int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 0; }

    std::string sub = argv[1];
    std::vector<std::string> args;
    for (int i=2; i<argc; ++i) args.emplace_back(argv[i]);

    if (sub == "--help" || sub == "-h") { usage(); return 0; }

    if (sub == "run") {
        std::string cfg = getArg(args, "--config");
        if (!cfg.empty()) return cmd_run_config(cfg);
        return cmd_run(args);
    }
    if (sub == "validate") {
        return cmd_validate(args);
    }
    if (sub != "capture") {
        std::cerr << "Unknown subcommand: " << sub << "\n";
        usage();
        return 1;
    }


    std::string input = getArg(args, "--input");
    std::string outdir = getArg(args, "--out", "out/cap");
    int frames = 0;
    try { frames = std::stoi(getArg(args, "--frames", "0")); } catch (...) {}

    if (input.empty()) { std::cerr << "Missing --input <video.mp4>\n"; return 2; }
    fs::create_directories(outdir);

    cv::VideoCapture cap(input);
    if (!cap.isOpened()) { std::cerr << "Failed to open: " << input << "\n"; return 3; }

    cambench::CsvWriter csv(outdir + "/capture_metrics.csv", "frame,ms,width,height");
    cambench::Timer t; t.start();

    cv::Mat frame;
    int i = 0, saved = 0;
    double total_ms = 0.0;

    while (true) {
        if (frames > 0 && i >= frames) break;
        if (!cap.read(frame) || frame.empty()) break;

        double t0 = t.ms();
        std::ostringstream name;
        name << outdir << "/frame_" << std::setw(6) << std::setfill('0') << i << ".png";
        cv::imwrite(name.str(), frame);
        double dt = t.ms() - t0;

        csv.row(i, dt, frame.cols, frame.rows);
        total_ms += dt; ++saved; ++i;
    }

    double avg_ms = (saved > 0) ? total_ms / saved : 0.0;
    double fps = (avg_ms > 0.0) ? 1000.0 / avg_ms : 0.0;

    std::cout << "Saved " << saved << " frames to " << outdir
              << " | avg save time: " << avg_ms << " ms (" << fps << " FPS)\n";
    return 0;
}
