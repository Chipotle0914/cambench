#pragma once
#include <chrono>
#include <fstream>
#include <string>


namespace cambench {

struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double ms() const {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  }
};

class CsvWriter {
  std::ofstream ofs_;
public:
  CsvWriter(const std::string& path, const std::string& header) {
    bool exists = static_cast<bool>(std::ifstream(path));
    ofs_.open(path, std::ios::app);
    if (!exists) ofs_ << header << "\n";
  }
  template<typename T, typename... Ts>
  void row(const T& first, const Ts&... rest) {
    ofs_ << first;
    ((ofs_ << ',' << rest), ...);
    ofs_ << "\n";
  }
};

} 
