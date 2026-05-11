#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct SynthOptions {
  std::string voice;
  std::optional<std::filesystem::path> prompt_wav_path;
  int seed = 1234;
  int max_new_frames = -1;
};

class MossTtsEngine {
 public:
  explicit MossTtsEngine(std::filesystem::path model_dir, int thread_count = 4);
  MossTtsEngine(const MossTtsEngine&) = delete;
  MossTtsEngine& operator=(const MossTtsEngine&) = delete;
  ~MossTtsEngine();

  std::vector<uint8_t> synthesize_wav(const std::string& text, const SynthOptions& opt);

  std::string voices_json() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  mutable std::mutex mutex_;
};
