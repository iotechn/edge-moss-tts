#include "engine.hpp"

#include <httplib.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

static int usage(const char* argv0) {
  std::cerr << "用法:\n  " << argv0 << " --model-dir <models目录> [--host 0.0.0.0] [--port 18083] [--threads 4]\n";
  return 2;
}

static std::filesystem::path getenv_path(const char* key, std::filesystem::path fallback) {
  const char* v = std::getenv(key);
  if (!v || !v[0]) return fallback;
  return std::filesystem::path(v);
}

int main(int argc, char** argv) {
  std::string host = "127.0.0.1";
  int port = 18083;
  int threads = 4;
  std::filesystem::path model_dir;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](void) -> char* {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    try {
      if (a == "--help" || a == "-h") return usage(argv[0]);
      if (a == "--model-dir") model_dir = need();
      else if (a == "--host") host = need();
      else if (a == "--port") port = std::stoi(need());
      else if (a == "--threads") threads = std::stoi(need());
      else throw std::runtime_error("unknown arg: " + a);
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      return usage(argv[0]);
    }
  }

  if (model_dir.empty()) model_dir = getenv_path("MOSS_TTS_MODEL_DIR", std::filesystem::path("./models"));

  std::cout << "加载模型目录: " << std::filesystem::weakly_canonical(model_dir).string() << "\n";
  MossTtsEngine engine(model_dir, threads);

  httplib::Server svr;

  svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.status = 200;
    res.set_content("{\"ok\":true}", "application/json; charset=utf-8");
  });

  svr.Get("/voices", [&](const httplib::Request&, httplib::Response& res) {
    res.status = 200;
    res.set_content(engine.voices_json(), "application/json; charset=utf-8");
  });

  svr.Post("/tts", [&](const httplib::Request& req, httplib::Response& res) {
    try {
      SynthOptions opt;
      opt.voice = req.get_param_value("voice");
      const std::string text = req.get_param_value("text");
      if (text.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"missing text\"}", "application/json; charset=utf-8");
        return;
      }
      if (req.has_param("seed")) opt.seed = std::stoi(req.get_param_value("seed"));
      if (req.has_param("max_new_frames")) opt.max_new_frames = std::stoi(req.get_param_value("max_new_frames"));

      const std::string prompt_path = req.get_param_value("prompt_wav");
      if (!prompt_path.empty()) opt.prompt_wav_path = prompt_path;

      const auto t0 = std::chrono::steady_clock::now();
      auto wav_bytes = engine.synthesize_wav(text, opt);
      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      res.status = 200;
      res.set_header("X-Generate-Time-Ms", std::to_string(ms));
      res.set_content(reinterpret_cast<const char*>(wav_bytes.data()), wav_bytes.size(), "audio/wav");
    } catch (const std::exception& e) {
      res.status = 500;
      std::string body = std::string("{\"error\":\"") + e.what() + "\"}";
      res.set_content(body, "application/json; charset=utf-8");
    }
  });

  std::cout << "监听 http://" << host << ":" << port << "\n";
  std::cout << "  GET  /health\n";
  std::cout << "  GET  /voices\n";
  std::cout << "  POST /tts  (x-www-form-urlencoded: text=...&voice=Junhao&prompt_wav=/abs/path.wav)\n";

  if (!svr.listen(host.c_str(), port)) {
    std::cerr << "监听失败（端口占用或地址无效）\n";
    return 1;
  }
  return 0;
}
