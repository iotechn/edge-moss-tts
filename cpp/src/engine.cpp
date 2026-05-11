#include "engine.hpp"

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.hpp>

#include <sentencepiece_processor.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

using json = nlohmann::json;

static std::vector<Ort::Value> session_run_all(Ort::Session& session, const char* const* input_names,
                                               const Ort::Value* inputs, size_t input_count) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t output_count = session.GetOutputCount();
  std::vector<std::string> output_owned_names;
  output_owned_names.reserve(output_count);
  std::vector<const char*> output_name_ptrs;
  output_name_ptrs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    auto alloc_name = session.GetOutputNameAllocated(i, allocator);
    output_owned_names.emplace_back(alloc_name.get());
  }
  for (const auto& name : output_owned_names) output_name_ptrs.push_back(name.c_str());
  return session.Run(Ort::RunOptions{nullptr}, input_names, inputs, input_count, output_name_ptrs.data(),
                     output_name_ptrs.size());
}

static constexpr const char* kSentenceEnd = ".!?。！？；;";
static constexpr const char* kClause = ",，、；;：:";
static constexpr const char* kClosing = "\"'\"')]}）】》」』";

static bool char_in_set(uint32_t cp, const char* utf8_set) {
  std::string s = [&]() {
    std::string out;
    out.reserve(4);
    if (cp <= 0x7f) {
      out.push_back(static_cast<char>(cp));
      return out;
    }
    if (cp <= 0x7ff) {
      out.push_back(static_cast<char>(0xc0 | ((cp >> 6) & 0x1f)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
      return out;
    }
    if (cp <= 0xffff) {
      out.push_back(static_cast<char>(0xe0 | ((cp >> 12) & 0x0f)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
      return out;
    }
    out.push_back(static_cast<char>(0xf0 | ((cp >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    return out;
  }();
  return std::strstr(utf8_set, s.c_str()) != nullptr;
}

static bool utf8_next(const std::string& text, size_t& i, uint32_t& cp) {
  if (i >= text.size()) return false;
  unsigned char c = static_cast<unsigned char>(text[i]);
  if (c < 0x80) {
    cp = c;
    i += 1;
    return true;
  }
  auto need = [&](unsigned char ch) -> bool {
    return i + ch <= text.size();
  };
  if ((c >> 5) == 0x6 && need(2)) {
    cp = ((c & 0x1f) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3f);
    i += 2;
    return true;
  }
  if ((c >> 4) == 0xe && need(3)) {
    cp = ((c & 0x0f) << 12) | ((static_cast<unsigned char>(text[i + 1]) & 0x3f) << 6) |
         (static_cast<unsigned char>(text[i + 2]) & 0x3f);
    i += 3;
    return true;
  }
  if ((c >> 3) == 0x1e && need(4)) {
    cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(text[i + 1]) & 0x3f) << 12) |
         ((static_cast<unsigned char>(text[i + 2]) & 0x3f) << 6) |
         (static_cast<unsigned char>(text[i + 3]) & 0x3f);
    i += 4;
    return true;
  }
  cp = c;
  i += 1;
  return true;
}

static bool contains_cjk(const std::string& text) {
  size_t i = 0;
  uint32_t cp = 0;
  while (utf8_next(text, i, cp)) {
    if ((cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
        (cp >= 0x3040 && cp <= 0x30ff) || (cp >= 0xac00 && cp <= 0xd7af)) {
      return true;
    }
  }
  return false;
}

static bool ends_with_sentence_punct(const std::string& text) {
  size_t i = 0;
  uint32_t cp = 0;
  uint32_t last_cp = 0;
  while (utf8_next(text, i, cp)) last_cp = cp;
  if (last_cp == 0) return false;
  return char_in_set(last_cp, kSentenceEnd);
}

static std::string strip_copy(std::string s) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

static std::string collapse_spaces(std::string s) {
  std::string out;
  out.reserve(s.size());
  bool prev_space = false;
  for (unsigned char ch : s) {
    bool sp = std::isspace(ch);
    if (sp) {
      if (!prev_space) out.push_back(' ');
      prev_space = true;
    } else {
      out.push_back(static_cast<char>(ch));
      prev_space = false;
    }
  }
  return strip_copy(out);
}

static std::string prepare_text_for_sentence_chunking(std::string text) {
  std::string normalized = collapse_spaces(strip_copy(text));
  if (normalized.empty()) throw std::runtime_error("Text prompt cannot be empty.");
  for (char& ch : normalized) {
    if (ch == '\r' || ch == '\n') ch = ' ';
  }
  normalized = collapse_spaces(normalized);
  if (contains_cjk(normalized)) {
    if (!ends_with_sentence_punct(normalized)) normalized += "。";
    return normalized;
  }
  if (!normalized.empty() && std::islower(static_cast<unsigned char>(normalized[0]))) {
    normalized[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(normalized[0])));
  }
  if (!normalized.empty() && std::isalnum(static_cast<unsigned char>(normalized.back()))) normalized += ".";
  int word_count = 0;
  std::istringstream iss(normalized);
  std::string w;
  while (iss >> w)
    if (!w.empty()) word_count++;
  if (word_count < 5) normalized = " " + normalized;
  return normalized;
}

static bool is_sentence_end_cp(uint32_t cp) { return char_in_set(cp, kSentenceEnd); }
static bool is_clause_cp(uint32_t cp) { return char_in_set(cp, kClause); }
static bool is_closing_cp(uint32_t cp) { return char_in_set(cp, kClosing); }

static std::vector<std::string> split_by_predicate(const std::string& text,
                                                   const std::function<bool(uint32_t)>& is_boundary) {
  std::vector<std::string> sentences;
  std::string current;
  size_t i = 0;
  while (i < text.size()) {
    size_t start = i;
    uint32_t cp = 0;
    if (!utf8_next(text, i, cp)) break;
    current.append(text.begin() + static_cast<std::ptrdiff_t>(start),
                   text.begin() + static_cast<std::ptrdiff_t>(i));
    if (is_boundary(cp)) {
      size_t lookahead = i;
      while (lookahead < text.size()) {
        size_t la0 = lookahead;
        uint32_t cp2 = 0;
        if (!utf8_next(text, lookahead, cp2)) break;
        if (!is_closing_cp(cp2)) {
          lookahead = la0;
          break;
        }
        current.append(text.begin() + static_cast<std::ptrdiff_t>(la0),
                       text.begin() + static_cast<std::ptrdiff_t>(lookahead));
      }
      auto piece = strip_copy(current);
      if (!piece.empty()) sentences.push_back(piece);
      current.clear();
      while (lookahead < text.size() && std::isspace(static_cast<unsigned char>(text[lookahead]))) lookahead++;
      i = lookahead;
      continue;
    }
  }
  auto tail = strip_copy(current);
  if (!tail.empty()) sentences.push_back(tail);
  return sentences;
}

static std::vector<std::string> split_sentence_end(const std::string& text) {
  return split_by_predicate(text, is_sentence_end_cp);
}

static std::vector<std::string> split_clause(const std::string& text) {
  return split_by_predicate(text, is_clause_cp);
}

static std::string join_parts(const std::string& left, const std::string& right) {
  if (left.empty()) return right;
  if (right.empty()) return left;
  if (contains_cjk(left) || contains_cjk(right)) return left + right;
  return left + " " + right;
}

static json load_json_file(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open json: " + path.string());
  json j;
  in >> j;
  return j;
}

static std::filesystem::path find_manifest(const std::filesystem::path& model_dir) {
  const std::vector<std::string> candidates = {
      "browser_poc_manifest.json",
      "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
      "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
  };
  for (const auto& rel : candidates) {
    auto p = model_dir / rel;
    if (std::filesystem::exists(p)) return std::filesystem::canonical(p);
  }
  throw std::runtime_error("browser_poc_manifest.json not found under " + model_dir.string());
}

static std::filesystem::path resolve_manifest_relative(const std::filesystem::path& manifest_dir,
                                                       const std::filesystem::path& relative_path) {
  static const std::unordered_map<std::string, std::string> aliases = {
      {"MOSS-TTS-Nano-ONNX-CPU", "MOSS-TTS-Nano-100M-ONNX"},
      {"MOSS-Audio-Tokenizer-Nano-ONNX-CPU", "MOSS-Audio-Tokenizer-Nano-ONNX"},
  };
  auto rel = relative_path.lexically_normal().generic_string();
  auto candidate = manifest_dir / relative_path;
  if (std::filesystem::exists(candidate)) return std::filesystem::canonical(candidate);
  std::string rel_text = rel;
  for (const auto& kv : aliases) {
    const std::string needle = "/" + kv.first + "/";
    if (rel_text.find(needle) == std::string::npos) continue;
    std::string rewritten = rel_text;
    size_t pos = 0;
    while ((pos = rewritten.find(kv.first, pos)) != std::string::npos) {
      rewritten.replace(pos, kv.first.size(), kv.second);
      pos += kv.second.size();
    }
    auto p2 = manifest_dir / rewritten;
    if (std::filesystem::exists(p2)) return std::filesystem::canonical(p2);
  }
  return std::filesystem::canonical(candidate);
}

static float clamp01(float x) { return std::max(0.0f, std::min(0.99999994f, x)); }

struct WavPcmFloatStereo {
  int sample_rate = 48000;
  int channels = 2;
  std::vector<float> planar_ct;  // [c0 samples..., c1 samples...]
};

static WavPcmFloatStereo decode_wav_file(const std::filesystem::path& path);

static std::vector<float> linear_resample_planar(const float* channel, int samples, int src_sr, int dst_sr) {
  if (src_sr == dst_sr) return std::vector<float>(channel, channel + samples);
  double ratio = static_cast<double>(dst_sr) / static_cast<double>(src_sr);
  int out_len = std::max(1, static_cast<int>(std::floor(samples * ratio)));
  std::vector<float> out(static_cast<size_t>(out_len));
  for (int i = 0; i < out_len; ++i) {
    double src_pos = static_cast<double>(i) / ratio;
    int i0 = static_cast<int>(std::floor(src_pos));
    int i1 = std::min(i0 + 1, samples - 1);
    double t = src_pos - static_cast<double>(i0);
    double v0 = channel[i0];
    double v1 = channel[i1];
    out[static_cast<size_t>(i)] = static_cast<float>(v0 * (1.0 - t) + v1 * t);
  }
  return out;
}

static WavPcmFloatStereo load_prompt_waveform(const std::filesystem::path& wav_path, int target_sr, int target_ch) {
  auto wav = decode_wav_file(wav_path);
  int cur_samples = wav.channels > 0 ? static_cast<int>(wav.planar_ct.size()) / wav.channels : 0;
  std::vector<std::vector<float>> planes;
  planes.reserve(static_cast<size_t>(target_ch));

  if (wav.channels == 1 && target_ch > 1) {
    const float* ch0 = wav.planar_ct.data();
    auto res = linear_resample_planar(ch0, cur_samples, wav.sample_rate, target_sr);
    planes.assign(static_cast<size_t>(target_ch), res);
  } else if (wav.channels > 1 && target_ch == 1) {
    std::vector<float> mono;
    for (int oc = 0; oc < wav.channels; ++oc) {
      const float* och = wav.planar_ct.data() + static_cast<size_t>(oc) * static_cast<size_t>(cur_samples);
      auto rch = linear_resample_planar(och, cur_samples, wav.sample_rate, target_sr);
      if (mono.empty()) mono.assign(rch.size(), 0.f);
      for (size_t i = 0; i < mono.size(); ++i) mono[i] += rch[i];
    }
    float inv = 1.f / static_cast<float>(wav.channels);
    for (float& v : mono) v *= inv;
    planes.push_back(std::move(mono));
  } else {
    if (wav.channels != target_ch) {
      throw std::runtime_error("Unsupported channel conversion: " + std::to_string(wav.channels) + " -> " +
                               std::to_string(target_ch));
    }
    for (int c = 0; c < wav.channels; ++c) {
      const float* ch = wav.planar_ct.data() + static_cast<size_t>(c) * static_cast<size_t>(cur_samples);
      planes.push_back(linear_resample_planar(ch, cur_samples, wav.sample_rate, target_sr));
    }
  }

  int samples_per_ch = planes.empty() ? 0 : static_cast<int>(planes[0].size());
  std::vector<float> planar(static_cast<size_t>(target_ch) * static_cast<size_t>(samples_per_ch));
  for (int c = 0; c < target_ch; ++c) {
    std::copy(planes[static_cast<size_t>(c)].begin(), planes[static_cast<size_t>(c)].end(),
              planar.begin() + static_cast<size_t>(c) * static_cast<size_t>(samples_per_ch));
  }
  wav.planar_ct = std::move(planar);
  wav.sample_rate = target_sr;
  wav.channels = target_ch;
  return wav;
}

static std::vector<uint8_t> encode_wav_pcm16_stereo(const std::vector<float>& interleaved_lr, int sample_rate,
                                                    int channels) {
  std::vector<uint8_t> bytes;
  bytes.resize(44 + interleaved_lr.size() * 2);
  auto w32 = [&](size_t off, uint32_t v) {
    bytes[off] = static_cast<uint8_t>(v & 0xff);
    bytes[off + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
    bytes[off + 2] = static_cast<uint8_t>((v >> 16) & 0xff);
    bytes[off + 3] = static_cast<uint8_t>((v >> 24) & 0xff);
  };
  auto w16 = [&](size_t off, uint16_t v) {
    bytes[off] = static_cast<uint8_t>(v & 0xff);
    bytes[off + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
  };
  uint32_t data_bytes = static_cast<uint32_t>(interleaved_lr.size() * 2);
  bytes[0] = 'R';
  bytes[1] = 'I';
  bytes[2] = 'F';
  bytes[3] = 'F';
  w32(4, 36 + data_bytes);
  bytes[8] = 'W';
  bytes[9] = 'A';
  bytes[10] = 'V';
  bytes[11] = 'E';
  bytes[12] = 'f';
  bytes[13] = 'm';
  bytes[14] = 't';
  bytes[15] = ' ';
  w32(16, 16);
  w16(20, 1);
  w16(22, static_cast<uint16_t>(channels));
  w32(24, static_cast<uint32_t>(sample_rate));
  w32(28, static_cast<uint32_t>(sample_rate * channels * 2));
  w16(32, static_cast<uint16_t>(channels * 2));
  w16(34, 16);
  bytes[36] = 'd';
  bytes[37] = 'a';
  bytes[38] = 't';
  bytes[39] = 'a';
  w32(40, data_bytes);
  size_t pcm_off = 44;
  for (float f : interleaved_lr) {
    float c = std::max(-1.f, std::min(1.f, f));
    int16_t s = static_cast<int16_t>(std::lround(c * 32767.0f));
    w16(pcm_off, static_cast<uint16_t>(s));
    pcm_off += 2;
  }
  return bytes;
}

static std::vector<float> merge_planar_to_interleaved(const std::vector<float>& planar, int channels, int samples) {
  std::vector<float> interleaved(static_cast<size_t>(samples) * static_cast<size_t>(channels));
  for (int t = 0; t < samples; ++t) {
    for (int c = 0; c < channels; ++c) {
      interleaved[static_cast<size_t>(t) * static_cast<size_t>(channels) + static_cast<size_t>(c)] =
          planar[static_cast<size_t>(c) * static_cast<size_t>(samples) + static_cast<size_t>(t)];
    }
  }
  return interleaved;
}

static std::vector<float> tensor_to_vec_float(const Ort::Value& v) {
  auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(std::max<int64_t>(0, d));
  const float* p = v.GetTensorData<float>();
  return std::vector<float>(p, p + n);
}

static std::vector<int32_t> tensor_to_vec_int32(const Ort::Value& v) {
  auto shape = v.GetTensorTypeAndShapeInfo().GetShape();
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(std::max<int64_t>(0, d));
  const int32_t* p = v.GetTensorData<int32_t>();
  return std::vector<int32_t>(p, p + n);
}

static int64_t read_scalar_int64_tensor(const Ort::Value& v) {
  auto t = v.GetTensorTypeAndShapeInfo().GetElementType();
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return v.GetTensorData<int64_t>()[0];
  }
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return static_cast<int64_t>(v.GetTensorData<int32_t>()[0]);
  }
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return v.GetTensorData<bool>()[0] ? 1 : 0;
  }
  throw std::runtime_error("unsupported scalar tensor type");
}

// CreateTensor 不拷贝数据；返回的 Ort::Value 在 Run 完成前，data 指向的 vector 必须保持有效。
static Ort::Value tensor_float_ref(const std::vector<float>& data, const std::vector<int64_t>& shape) {
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor<float>(mem, const_cast<float*>(data.data()), data.size(), shape.data(), shape.size());
}

static Ort::Value tensor_int_mixed(std::vector<int32_t>& data32, std::vector<int64_t>& data64_out,
                                   ONNXTensorElementDataType dt, const std::vector<int64_t>& shape) {
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return Ort::Value::CreateTensor<int32_t>(mem, data32.data(), data32.size(), shape.data(), shape.size());
  }
  if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    data64_out.resize(data32.size());
    for (size_t i = 0; i < data32.size(); ++i) {
      data64_out[i] = static_cast<int64_t>(data32[i]);
    }
    return Ort::Value::CreateTensor<int64_t>(mem, data64_out.data(), data64_out.size(), shape.data(), shape.size());
  }
  throw std::runtime_error("ONNX int tensor: expected int32 or int64 element type");
}

static ONNXTensorElementDataType input_elem_type(Ort::Session& session, const char* target_name) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t n = session.GetInputCount();
  for (size_t i = 0; i < n; ++i) {
    auto name = session.GetInputNameAllocated(i, allocator);
    if (std::strcmp(name.get(), target_name) != 0) continue;
    Ort::TypeInfo ti = session.GetInputTypeInfo(i);
    return ti.GetTensorTypeAndShapeInfo().GetElementType();
  }
  throw std::runtime_error(std::string("missing ONNX input: ") + target_name);
}

static std::vector<int32_t> tensor_flat_to_int32_codes(const Ort::Value& v) {
  auto info = v.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  ONNXTensorElementDataType et = info.GetElementType();
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(std::max<int64_t>(0, d));
  std::vector<int32_t> out(n);
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t* p = v.GetTensorData<int32_t>();
    std::copy(p, p + n, out.begin());
    return out;
  }
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t* p = v.GetTensorData<int64_t>();
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(p[i]);
    return out;
  }
  throw std::runtime_error("codec tensor: expected int32 or int64 codes");
}

static Ort::Value tensor_from_float_buffer(const std::vector<float>& data, const std::vector<int64_t>& shape) {
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor<float>(mem, const_cast<float*>(data.data()), data.size(), shape.data(),
                                         shape.size());
}

static std::vector<float> extract_last_global_hidden(const Ort::Value& hidden_tensor) {
  auto shape = hidden_tensor.GetTensorTypeAndShapeInfo().GetShape();
  const float* data = hidden_tensor.GetTensorData<float>();
  if (shape.size() == 3) {
    if (shape[0] != 1) throw std::runtime_error("global_hidden: expected batch 1");
    int64_t seq = shape[1];
    int64_t hs = shape[2];
    const float* last = data + (seq - 1) * hs;
    return std::vector<float>(last, last + hs);
  }
  if (shape.size() == 2) {
    int64_t d0 = shape[0];
    int64_t d1 = shape[1];
    if (d0 == 1) return std::vector<float>(data, data + d1);
    const float* last = data + (d0 - 1) * d1;
    return std::vector<float>(last, last + d1);
  }
  throw std::runtime_error("global_hidden: unexpected rank");
}

static std::vector<float> as_row_global_hidden(const std::vector<float>& hidden_vec) {
  int64_t hs = static_cast<int64_t>(hidden_vec.size());
  std::vector<float> row = hidden_vec;
  return row;
}

struct WavPcmFloatStereo decode_wav_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open wav: " + path.string());
  char riff[12];
  in.read(riff, 12);
  if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(riff + 8, "WAVE", 4) != 0)
    throw std::runtime_error("invalid wav");
  uint16_t audio_format = 0;
  uint16_t num_channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  bool fmt_ok = false;
  while (in) {
    char cid[4];
    uint32_t csize = 0;
    in.read(cid, 4);
    if (!in) break;
    in.read(reinterpret_cast<char*>(&csize), 4);
    if (!in) break;
    if (std::strncmp(cid, "fmt ", 4) == 0) {
      std::vector<char> chunk(csize);
      in.read(chunk.data(), csize);
      if (csize < 16) throw std::runtime_error("bad fmt chunk");
      audio_format = *reinterpret_cast<uint16_t*>(chunk.data());
      num_channels = *reinterpret_cast<uint16_t*>(chunk.data() + 2);
      sample_rate = *reinterpret_cast<uint32_t*>(chunk.data() + 4);
      bits_per_sample = *reinterpret_cast<uint16_t*>(chunk.data() + 14);
      fmt_ok = true;
    } else if (std::strncmp(cid, "data", 4) == 0) {
      if (!fmt_ok) throw std::runtime_error("wav fmt missing");
      std::vector<char> pcm(csize);
      in.read(pcm.data(), csize);
      int samples = static_cast<int>(csize / (num_channels * (bits_per_sample / 8)));
      WavPcmFloatStereo out;
      out.sample_rate = static_cast<int>(sample_rate);
      out.channels = num_channels;
      out.planar_ct.resize(static_cast<size_t>(num_channels) * static_cast<size_t>(samples));
      if (audio_format != 1 && audio_format != 3) {
        throw std::runtime_error("wav: only PCM integer or float supported");
      }
      if (audio_format == 1 && bits_per_sample == 16) {
        const int16_t* s = reinterpret_cast<const int16_t*>(pcm.data());
        for (int t = 0; t < samples; ++t) {
          for (int c = 0; c < num_channels; ++c) {
            float v = static_cast<float>(s[t * num_channels + c]) / 32768.f;
            out.planar_ct[static_cast<size_t>(c) * static_cast<size_t>(samples) + static_cast<size_t>(t)] = v;
          }
        }
      } else if (audio_format == 3 && bits_per_sample == 32) {
        const float* s = reinterpret_cast<const float*>(pcm.data());
        for (int t = 0; t < samples; ++t) {
          for (int c = 0; c < num_channels; ++c) {
            out.planar_ct[static_cast<size_t>(c) * static_cast<size_t>(samples) + static_cast<size_t>(t)] =
                s[t * num_channels + c];
          }
        }
      } else {
        throw std::runtime_error("wav: unsupported bits/sample combo");
      }
      return out;
    } else {
      in.seekg(csize, std::ios::cur);
    }
  }
  throw std::runtime_error("wav: data chunk missing");
}

}  // namespace

struct MossTtsEngine::Impl {
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "moss_tts"};
  Ort::SessionOptions session_options_;

  std::filesystem::path manifest_dir_;
  json manifest_;
  json tts_meta_;
  json codec_meta_;

  std::unique_ptr<Ort::Session> sess_prefill_;
  std::unique_ptr<Ort::Session> sess_decode_;
  std::unique_ptr<Ort::Session> sess_local_fixed_;
  std::unique_ptr<Ort::Session> sess_codec_encode_;
  std::unique_ptr<Ort::Session> sess_codec_decode_;

  sentencepiece::SentencePieceProcessor sp_;

  std::vector<const char*> prefill_in_names_;
  std::vector<const char*> decode_in_names_;
  std::vector<const char*> fixed_in_names_;

  std::vector<std::string> prefill_in_name_strings_;
  std::vector<std::string> decode_in_name_strings_;
  std::vector<std::string> fixed_in_name_strings_;

  int thread_count_ = 4;

  ONNXTensorElementDataType prefill_input_ids_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  ONNXTensorElementDataType prefill_attention_mask_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  ONNXTensorElementDataType decode_input_ids_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  ONNXTensorElementDataType decode_past_valid_lengths_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  ONNXTensorElementDataType fixed_repetition_mask_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  ONNXTensorElementDataType codec_input_lengths_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  ONNXTensorElementDataType codec_decode_audio_codes_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  ONNXTensorElementDataType codec_decode_audio_code_lengths_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

  Impl(std::filesystem::path model_dir, int threads) : thread_count_(std::max(1, threads)) {
    session_options_.SetIntraOpNumThreads(thread_count_);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    auto manifest_path = find_manifest(model_dir);
    manifest_dir_ = manifest_path.parent_path();
    manifest_ = load_json_file(manifest_path);

    auto tts_meta_path =
        resolve_manifest_relative(manifest_dir_, manifest_["model_files"]["tts_meta"].get<std::string>());
    auto codec_meta_path =
        resolve_manifest_relative(manifest_dir_, manifest_["model_files"]["codec_meta"].get<std::string>());
    tts_meta_ = load_json_file(tts_meta_path);
    codec_meta_ = load_json_file(codec_meta_path);

    auto tokenizer_path =
        resolve_manifest_relative(manifest_dir_, manifest_["model_files"]["tokenizer_model"].get<std::string>());
    const auto status = sp_.Load(tokenizer_path.string());
    if (!status.ok()) throw std::runtime_error("sentencepiece load failed: " + status.ToString());

    auto tts_dir = tts_meta_path.parent_path();
    auto codec_dir = codec_meta_path.parent_path();

    auto mk_session = [&](const std::filesystem::path& p) {
      return std::make_unique<Ort::Session>(env_, p.string().c_str(), session_options_);
    };

    sess_prefill_ = mk_session(tts_dir / tts_meta_["files"]["prefill"].get<std::string>());
    sess_decode_ = mk_session(tts_dir / tts_meta_["files"]["decode_step"].get<std::string>());
    if (!tts_meta_["files"].contains("local_fixed_sampled_frame")) {
      throw std::runtime_error("This server expects local_fixed_sampled_frame ONNX (default upstream manifest).");
    }
    sess_local_fixed_ = mk_session(tts_dir / tts_meta_["files"]["local_fixed_sampled_frame"].get<std::string>());
    sess_codec_encode_ = mk_session(codec_dir / codec_meta_["files"]["encode"].get<std::string>());
    sess_codec_decode_ = mk_session(codec_dir / codec_meta_["files"]["decode_full"].get<std::string>());

    prefill_in_name_strings_ = {"input_ids", "attention_mask"};
    for (const auto& s : prefill_in_name_strings_) prefill_in_names_.push_back(s.c_str());

    for (const auto& item : tts_meta_["onnx"]["decode_input_names"]) {
      decode_in_name_strings_.push_back(item.get<std::string>());
    }
    for (const auto& s : decode_in_name_strings_) decode_in_names_.push_back(s.c_str());

    for (const auto& item : tts_meta_["onnx"]["local_fixed_sampled_frame_input_names"]) {
      fixed_in_name_strings_.push_back(item.get<std::string>());
    }
    for (const auto& s : fixed_in_name_strings_) fixed_in_names_.push_back(s.c_str());

    codec_in_name_strings_ = {"waveform", "input_lengths"};
    for (const auto& s : codec_in_name_strings_) codec_in_names_.push_back(s.c_str());

    prefill_input_ids_type_ = input_elem_type(*sess_prefill_, "input_ids");
    prefill_attention_mask_type_ = input_elem_type(*sess_prefill_, "attention_mask");
    decode_input_ids_type_ = input_elem_type(*sess_decode_, "input_ids");
    decode_past_valid_lengths_type_ = input_elem_type(*sess_decode_, "past_valid_lengths");
    fixed_repetition_mask_type_ = input_elem_type(*sess_local_fixed_, "repetition_seen_mask");
    codec_input_lengths_type_ = input_elem_type(*sess_codec_encode_, "input_lengths");
    codec_decode_audio_codes_type_ = input_elem_type(*sess_codec_decode_, "audio_codes");
    codec_decode_audio_code_lengths_type_ = input_elem_type(*sess_codec_decode_, "audio_code_lengths");
  }

  std::vector<int> encode_text(const std::string& text) const {
    std::vector<int> ids;
    sp_.Encode(text, &ids);
    return ids;
  }

  int count_tokens(const std::string& text) const { return static_cast<int>(encode_text(text).size()); }

  std::vector<std::string> split_text_by_token_budget(std::string remaining, int max_tokens) const {
    std::vector<std::string> pieces;
    remaining = strip_copy(remaining);
    while (!remaining.empty()) {
      if (count_tokens(remaining) <= max_tokens) {
        pieces.push_back(remaining);
        break;
      }
      int low = 1;
      int high = static_cast<int>(remaining.size());
      int best = 1;
      while (low <= high) {
        int mid = (low + high) / 2;
        auto cand = strip_copy(remaining.substr(0, static_cast<size_t>(mid)));
        if (cand.empty()) {
          low = mid + 1;
          continue;
        }
        if (count_tokens(cand) <= max_tokens) {
          best = mid;
          low = mid + 1;
        } else {
          high = mid - 1;
        }
      }
      int cut_index = best;
      auto prefix = remaining.substr(0, static_cast<size_t>(best));
      const std::string preferred_chars = std::string(kClause) + std::string(kSentenceEnd) + " ";
      int preferred_index = -1;
      int scan_min = std::max(-1, static_cast<int>(prefix.size()) - 25);
      for (int idx = static_cast<int>(prefix.size()) - 1; idx > scan_min; --idx) {
        unsigned char ch = static_cast<unsigned char>(prefix[static_cast<size_t>(idx)]);
        if (preferred_chars.find(static_cast<char>(ch)) != std::string::npos) {
          preferred_index = idx + 1;
          break;
        }
      }
      if (preferred_index > 0) cut_index = preferred_index;
      auto piece = strip_copy(remaining.substr(0, static_cast<size_t>(cut_index)));
      if (piece.empty()) {
        piece = strip_copy(remaining.substr(0, static_cast<size_t>(best)));
        cut_index = best;
      }
      pieces.push_back(piece);
      remaining = strip_copy(remaining.substr(static_cast<size_t>(cut_index)));
    }
    return pieces;
  }

  std::vector<std::string> split_voice_clone_text(const std::string& text, int max_tokens) const {
    std::string normalized = strip_copy(text);
    if (normalized.empty()) return {};
    int safe_max = std::max(1, max_tokens);
    std::string prepared = prepare_text_for_sentence_chunking(normalized);
    auto sentence_candidates = split_sentence_end(prepared);
    if (sentence_candidates.empty()) sentence_candidates = {strip_copy(prepared)};
    struct Slice {
      int tok_count = 0;
      std::string s;
    };
    std::vector<Slice> slices;
    for (const auto& sentence_text : sentence_candidates) {
      auto ns = strip_copy(sentence_text);
      if (ns.empty()) continue;
      int tc = count_tokens(ns);
      if (tc <= safe_max) {
        slices.push_back({tc, ns});
        continue;
      }
      auto clauses = split_clause(ns);
      if (clauses.size() <= 1) clauses = {ns};
      for (const auto& clause_text : clauses) {
        auto nc = strip_copy(clause_text);
        if (nc.empty()) continue;
        int ctc = count_tokens(nc);
        if (ctc <= safe_max) {
          slices.push_back({ctc, nc});
          continue;
        }
        for (const auto& piece : split_text_by_token_budget(nc, safe_max)) {
          auto np = strip_copy(piece);
          if (!np.empty()) slices.push_back({count_tokens(np), np});
        }
      }
    }
    std::vector<std::string> chunks;
    std::string current;
    int current_tc = 0;
    for (const auto& sl : slices) {
      if (current.empty()) {
        current = sl.s;
        current_tc = sl.tok_count;
        continue;
      }
      if (current_tc + sl.tok_count > safe_max) {
        chunks.push_back(strip_copy(current));
        current = sl.s;
        current_tc = sl.tok_count;
      } else {
        current = join_parts(current, sl.s);
        current_tc = count_tokens(current);
      }
    }
    if (!strip_copy(current).empty()) chunks.push_back(strip_copy(current));
    if (chunks.size() > 1) return chunks;
    return {normalized};
  }

  double pause_seconds_for_chunk(const std::string& chunk) const {
    int words = 0;
    std::istringstream iss(chunk);
    std::string w;
    while (iss >> w)
      if (!w.empty()) words++;
    return words <= 4 ? 0.40 : 0.24;
  }

  std::vector<std::vector<int>> encode_reference_audio(const std::filesystem::path& wav_path) {
    int target_sr = codec_meta_["codec_config"]["sample_rate"].get<int>();
    int target_ch = codec_meta_["codec_config"]["channels"].get<int>();
    auto wav = load_prompt_waveform(wav_path, target_sr, target_ch);
    int samples = wav.channels > 0 ? static_cast<int>(wav.planar_ct.size()) / wav.channels : 0;
    std::vector<float> nchw(static_cast<size_t>(1) * static_cast<size_t>(wav.channels) * static_cast<size_t>(samples));
    for (int c = 0; c < wav.channels; ++c) {
      std::copy(wav.planar_ct.begin() + static_cast<size_t>(c) * static_cast<size_t>(samples),
                wav.planar_ct.begin() + static_cast<size_t>(c + 1) * static_cast<size_t>(samples),
                nchw.begin() + static_cast<size_t>(c) * static_cast<size_t>(samples));
    }
    std::vector<int64_t> wf_shape = {1, wav.channels, samples};
    std::vector<int32_t> lens = {samples};
    std::vector<int64_t> len_i64_scratch;

    std::vector<Ort::Value> inputs;
    inputs.push_back(tensor_float_ref(nchw, wf_shape));
    inputs.push_back(tensor_int_mixed(lens, len_i64_scratch, codec_input_lengths_type_, {1}));

    auto outputs = session_run_all(*sess_codec_encode_, codec_in_names_.data(), inputs.data(), inputs.size());
    std::vector<int32_t> audio_codes = tensor_flat_to_int32_codes(outputs[0]);
    std::vector<int32_t> code_lens = tensor_flat_to_int32_codes(outputs[1]);
    int code_len = code_lens.empty() ? 0 : code_lens[0];
    int num_q = codec_meta_["codec_config"]["num_quantizers"].get<int>();
    std::vector<std::vector<int>> frames;
    frames.reserve(static_cast<size_t>(code_len));
    for (int t = 0; t < code_len; ++t) {
      std::vector<int> row(static_cast<size_t>(num_q));
      for (int q = 0; q < num_q; ++q) {
        row[q] = audio_codes[static_cast<size_t>(t) * static_cast<size_t>(num_q) + static_cast<size_t>(q)];
      }
      frames.push_back(std::move(row));
    }
    return frames;
  }

  std::vector<std::string> codec_in_name_strings_;
  std::vector<const char*> codec_in_names_;

  std::vector<std::vector<int>> resolve_prompt_codes(const SynthOptions& opt) {
    if (opt.prompt_wav_path.has_value()) {
      return encode_reference_audio(*opt.prompt_wav_path);
    }
    std::string voice = opt.voice;
    if (voice.empty()) voice = manifest_["builtin_voices"][0]["voice"].get<std::string>();
    for (const auto& row : manifest_["builtin_voices"]) {
      if (row["voice"].get<std::string>() == voice) {
        std::vector<std::vector<int>> codes;
        for (const auto& fr : row["prompt_audio_codes"]) {
          std::vector<int> r;
          for (const auto& v : fr) r.push_back(v.get<int>());
          codes.push_back(std::move(r));
        }
        return codes;
      }
    }
    throw std::runtime_error("Built-in voice not found: " + voice);
  }

  json build_voice_clone_rows(const std::vector<std::vector<int>>& prompt_audio_codes,
                              const std::vector<int>& text_token_ids) const {
    const int n_vq = manifest_["tts_config"]["n_vq"].get<int>();
    const int audio_pad = manifest_["tts_config"]["audio_pad_token_id"].get<int>();
    const int audio_user_slot = manifest_["tts_config"]["audio_user_slot_token_id"].get<int>();

    auto build_text_rows = [&](const std::vector<int>& ids) {
      std::vector<std::vector<int>> rows;
      int row_width = n_vq + 1;
      for (int tid : ids) {
        std::vector<int> row(static_cast<size_t>(row_width), audio_pad);
        row[0] = tid;
        rows.push_back(std::move(row));
      }
      return rows;
    };

    auto build_audio_prefix_rows = [&](const std::vector<std::vector<int>>& codes) {
      std::vector<std::vector<int>> rows;
      int row_width = n_vq + 1;
      for (const auto& code_row : codes) {
        std::vector<int> row(static_cast<size_t>(row_width), audio_pad);
        row[0] = audio_user_slot;
        for (int i = 0; i < std::min(static_cast<int>(code_row.size()), n_vq); ++i) row[i + 1] = code_row[i];
        rows.push_back(std::move(row));
      }
      return rows;
    };

    std::vector<int> prefix_text_ids;
    for (const auto& v : manifest_["prompt_templates"]["user_prompt_prefix_token_ids"]) prefix_text_ids.push_back(v);
    prefix_text_ids.push_back(manifest_["tts_config"]["audio_start_token_id"].get<int>());

    std::vector<int> suffix_text_ids;
    suffix_text_ids.push_back(manifest_["tts_config"]["audio_end_token_id"].get<int>());
    for (const auto& v : manifest_["prompt_templates"]["user_prompt_after_reference_token_ids"])
      suffix_text_ids.push_back(v.get<int>());
    for (int v : text_token_ids) suffix_text_ids.push_back(v);
    for (const auto& v : manifest_["prompt_templates"]["assistant_prompt_prefix_token_ids"])
      suffix_text_ids.push_back(v.get<int>());
    suffix_text_ids.push_back(manifest_["tts_config"]["audio_start_token_id"].get<int>());

    std::vector<std::vector<int>> rows;
    auto a = build_text_rows(prefix_text_ids);
    auto b = build_audio_prefix_rows(prompt_audio_codes);
    auto c = build_text_rows(suffix_text_ids);
    rows.insert(rows.end(), a.begin(), a.end());
    rows.insert(rows.end(), b.begin(), b.end());
    rows.insert(rows.end(), c.begin(), c.end());

    std::vector<int> mask_row(static_cast<int>(rows.size()), 1);
    json out;
    out["inputIds"] = rows;
    out["attentionMask"] = std::vector<std::vector<int>>{mask_row};
    return out;
  }

  std::vector<float> decode_full_audio(const std::vector<std::vector<int>>& frames) const {
    if (frames.empty()) return {};
    int num_q = codec_meta_["codec_config"]["num_quantizers"].get<int>();
    int T = static_cast<int>(frames.size());
    std::vector<int32_t> codes_flat(static_cast<size_t>(1) * static_cast<size_t>(T) * static_cast<size_t>(num_q));
    for (int t = 0; t < T; ++t) {
      for (int q = 0; q < num_q; ++q) {
        int v = q < static_cast<int>(frames[t].size()) ? frames[t][q] : 0;
        codes_flat[static_cast<size_t>(t) * static_cast<size_t>(num_q) + static_cast<size_t>(q)] = v;
      }
    }
    std::vector<int32_t> lens = {T};
    std::vector<int64_t> codes_i64_scratch;
    std::vector<int64_t> lens_i64_scratch;

    std::vector<Ort::Value> ins;
    ins.push_back(tensor_int_mixed(codes_flat, codes_i64_scratch, codec_decode_audio_codes_type_, {1, T, num_q}));
    ins.push_back(tensor_int_mixed(lens, lens_i64_scratch, codec_decode_audio_code_lengths_type_, {1}));
    std::vector<const char*> names = {"audio_codes", "audio_code_lengths"};
    auto outs = session_run_all(*sess_codec_decode_, names.data(), ins.data(), ins.size());
    std::vector<float> audio = tensor_to_vec_float(outs[0]);
    int64_t audio_len = read_scalar_int64_tensor(outs[1]);
    int ch = codec_meta_["codec_config"]["channels"].get<int>();
    int samples = static_cast<int>(audio_len);
    std::vector<float> planar(static_cast<size_t>(ch) * static_cast<size_t>(samples));
    for (int c = 0; c < ch; ++c) {
      std::copy(audio.begin() + static_cast<size_t>(c) * static_cast<size_t>(samples),
                audio.begin() + static_cast<size_t>(c + 1) * static_cast<size_t>(samples),
                planar.begin() + static_cast<size_t>(c) * static_cast<size_t>(samples));
    }
    return merge_planar_to_interleaved(planar, ch, samples);
  }

  std::pair<bool, std::vector<int>> run_local_fixed_frame(Ort::Session& sess, const std::vector<float>& global_hidden_row,
                                                          const std::vector<std::unordered_set<int>>& prev_sets,
                                                          int n_vq, int codebook_size,
                                                          ONNXTensorElementDataType repetition_mask_elem_type,
                                                          std::mt19937_64& rng) const {
    int64_t hidden_size = static_cast<int64_t>(global_hidden_row.size());
    auto gh_tensor = tensor_float_ref(global_hidden_row, {1, hidden_size});

    std::vector<int32_t> mask(static_cast<size_t>(1) * static_cast<size_t>(n_vq) * static_cast<size_t>(codebook_size),
                              0);
    for (int c = 0; c < n_vq; ++c) {
      for (int tid : prev_sets[static_cast<size_t>(c)]) {
        if (tid >= 0 && tid < codebook_size) {
          mask[static_cast<size_t>(c) * static_cast<size_t>(codebook_size) + static_cast<size_t>(tid)] = 1;
        }
      }
    }
    std::vector<int64_t> mask_i64_scratch;
    auto mask_tensor =
        tensor_int_mixed(mask, mask_i64_scratch, repetition_mask_elem_type, {1, n_vq, codebook_size});

    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<float> au = {clamp01(dist(rng))};
    auto au_tensor = tensor_float_ref(au, {1});
    std::vector<float> aru(static_cast<size_t>(n_vq));
    for (int i = 0; i < n_vq; ++i) aru[static_cast<size_t>(i)] = clamp01(dist(rng));
    auto aru_tensor = tensor_float_ref(aru, {1, n_vq});

    std::vector<Ort::Value> ins;
    ins.push_back(std::move(gh_tensor));
    ins.push_back(std::move(mask_tensor));
    ins.push_back(std::move(au_tensor));
    ins.push_back(std::move(aru_tensor));

    auto outs = session_run_all(sess, fixed_in_names_.data(), ins.data(), ins.size());
    bool should_continue = read_scalar_int64_tensor(outs[0]) != 0;
    std::vector<int> frame(n_vq, 0);
    auto et = outs[1].GetTensorTypeAndShapeInfo().GetElementType();
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      auto frame_i32 = tensor_to_vec_int32(outs[1]);
      for (int i = 0; i < n_vq && i < static_cast<int>(frame_i32.size()); ++i) frame[i] = frame_i32[static_cast<size_t>(i)];
    } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      auto shape = outs[1].GetTensorTypeAndShapeInfo().GetShape();
      size_t n = 1;
      for (auto d : shape) n *= static_cast<size_t>(std::max<int64_t>(0, d));
      const int64_t* p = outs[1].GetTensorData<int64_t>();
      for (int i = 0; i < n_vq && i < static_cast<int>(n); ++i) frame[i] = static_cast<int>(p[i]);
    } else {
      throw std::runtime_error("frame_token_ids: unsupported dtype");
    }
    return {should_continue, frame};
  }

  std::vector<std::vector<int>> generate_audio_frames(json request_rows, json generation_defaults,
                                                      std::mt19937_64& rng) {
    const int n_vq = manifest_["tts_config"]["n_vq"].get<int>();
    const int row_width = n_vq + 1;
    const int audio_pad = manifest_["tts_config"]["audio_pad_token_id"].get<int>();
    const int assistant_slot = manifest_["tts_config"]["audio_assistant_slot_token_id"].get<int>();

    const auto& rows = request_rows["inputIds"].get<std::vector<std::vector<int>>>();
    const auto& mask_row = request_rows["attentionMask"][0].get<std::vector<int>>();
    int seq_len = static_cast<int>(rows.size());

    std::vector<int32_t> input_ids_flat(static_cast<size_t>(1) * static_cast<size_t>(seq_len) *
                                        static_cast<size_t>(row_width));
    for (int t = 0; t < seq_len; ++t) {
      for (int j = 0; j < row_width; ++j) {
        input_ids_flat[static_cast<size_t>(t) * static_cast<size_t>(row_width) + static_cast<size_t>(j)] =
            rows[static_cast<size_t>(t)][static_cast<size_t>(j)];
      }
    }
    std::vector<int32_t> mask_flat(static_cast<size_t>(1) * static_cast<size_t>(seq_len));
    for (int t = 0; t < seq_len; ++t) mask_flat[static_cast<size_t>(t)] = mask_row[static_cast<size_t>(t)];

    std::vector<int64_t> input_ids_i64_scratch;
    std::vector<int64_t> mask_flat_i64_scratch;

    std::vector<Ort::Value> prefill_inputs;
    prefill_inputs.push_back(
        tensor_int_mixed(input_ids_flat, input_ids_i64_scratch, prefill_input_ids_type_, {1, seq_len, row_width}));
    prefill_inputs.push_back(tensor_int_mixed(mask_flat, mask_flat_i64_scratch, prefill_attention_mask_type_, {1, seq_len}));

    auto prefill_outputs = session_run_all(*sess_prefill_, prefill_in_names_.data(), prefill_inputs.data(), 2);

    std::vector<std::string> prefill_out_names;
    for (const auto& item : tts_meta_["onnx"]["prefill_output_names"]) prefill_out_names.push_back(item.get<std::string>());

    std::unordered_map<std::string, Ort::Value> past_tensors;
    std::vector<float> global_hidden_vec;

    for (size_t i = 0; i < prefill_out_names.size(); ++i) {
      const auto& name = prefill_out_names[i];
      if (name == "global_hidden") {
        global_hidden_vec = extract_last_global_hidden(prefill_outputs[i]);
      } else {
        std::string past_name = name;
        if (past_name.rfind("present_", 0) == 0) past_name.replace(0, 8, "past_");
        past_tensors.emplace(past_name, std::move(prefill_outputs[i]));
      }
    }

    int past_valid_length = 0;
    for (int v : mask_row) past_valid_length += v;

    std::vector<std::vector<int>> generated;
    std::vector<std::unordered_set<int>> prev_sets(static_cast<size_t>(n_vq));

    const int max_frames = generation_defaults["max_new_frames"].get<int>();
    const int codebook_size = tts_meta_["model_config"]["audio_codebook_sizes"][0].get<int>();
    auto sample_mode = generation_defaults["sample_mode"].get<std::string>();
    const bool use_fixed_frame =
        sess_local_fixed_ && sample_mode == std::string("fixed") && tts_meta_["files"].contains("local_fixed_sampled_frame");
    if (!use_fixed_frame) {
      throw std::runtime_error("Only sample_mode=fixed with local_fixed_sampled_frame is supported by this C++ server.");
    }

    for (int step = 0; step < max_frames; ++step) {
      auto gh_row = as_row_global_hidden(global_hidden_vec);
      auto [should_continue, frame] = run_local_fixed_frame(*sess_local_fixed_, gh_row, prev_sets, n_vq, codebook_size,
                                                            fixed_repetition_mask_type_, rng);
      if (!should_continue) break;
      for (int c = 0; c < n_vq; ++c) prev_sets[static_cast<size_t>(c)].insert(frame[static_cast<size_t>(c)]);
      generated.push_back(frame);

      std::vector<int32_t> next_row(static_cast<size_t>(row_width), audio_pad);
      next_row[0] = assistant_slot;
      for (int i = 0; i < n_vq; ++i) next_row[static_cast<size_t>(i + 1)] = frame[static_cast<size_t>(i)];

      std::vector<Ort::Value> decode_inputs;
      decode_inputs.reserve(decode_in_name_strings_.size());

      std::vector<int64_t> next_row_i64_scratch;
      std::vector<int64_t> pvl_i64_scratch;
      decode_inputs.push_back(
          tensor_int_mixed(next_row, next_row_i64_scratch, decode_input_ids_type_, {1, 1, row_width}));

      std::vector<int32_t> pvl = {past_valid_length};
      decode_inputs.push_back(tensor_int_mixed(pvl, pvl_i64_scratch, decode_past_valid_lengths_type_, {1}));

      std::vector<std::vector<float>> past_float_copies;
      past_float_copies.reserve(decode_in_name_strings_.size());
      for (size_t i = 2; i < decode_in_name_strings_.size(); ++i) {
        const auto& in_name = decode_in_name_strings_[i];
        auto it = past_tensors.find(in_name);
        if (it == past_tensors.end()) throw std::runtime_error("missing past tensor: " + in_name);
        past_float_copies.push_back(tensor_to_vec_float(it->second));
        auto shape = it->second.GetTensorTypeAndShapeInfo().GetShape();
        decode_inputs.push_back(tensor_from_float_buffer(past_float_copies.back(), shape));
      }

      auto decode_outputs =
          session_run_all(*sess_decode_, decode_in_names_.data(), decode_inputs.data(), decode_inputs.size());

      std::vector<std::string> decode_out_names;
      for (const auto& item : tts_meta_["onnx"]["decode_output_names"]) decode_out_names.push_back(item.get<std::string>());

      past_tensors.clear();
      for (size_t i = 0; i < decode_out_names.size(); ++i) {
        const auto& name = decode_out_names[i];
        if (name == "global_hidden") {
          global_hidden_vec = extract_last_global_hidden(decode_outputs[i]);
        } else {
          std::string past_name = name;
          if (past_name.rfind("present_", 0) == 0) past_name.replace(0, 8, "past_");
          past_tensors.emplace(past_name, std::move(decode_outputs[i]));
        }
      }
      past_valid_length += 1;
    }

    return generated;
  }

  std::string prepare_text_simple(const std::string& text) const {
    auto s = collapse_spaces(strip_copy(text));
    if (s.empty()) throw std::runtime_error("empty text");
    return s;
  }

  std::vector<uint8_t> synthesize_wav(const std::string& text, const SynthOptions& opt) {
    json gen_defaults = manifest_["generation_defaults"];
    if (opt.max_new_frames > 0) gen_defaults["max_new_frames"] = opt.max_new_frames;

    std::mt19937_64 rng(static_cast<uint64_t>(opt.seed));

    auto prompt_codes = resolve_prompt_codes(opt);
    std::string prepared = prepare_text_simple(text);
    auto chunks = split_voice_clone_text(prepared, 75);

    int sample_rate = codec_meta_["codec_config"]["sample_rate"].get<int>();
    int channels = codec_meta_["codec_config"]["channels"].get<int>();

    std::vector<float> interleaved_all;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      const auto& chunk = chunks[ci];
      auto text_ids = encode_text(chunk);
      auto req = build_voice_clone_rows(prompt_codes, text_ids);
      auto frames = generate_audio_frames(req, gen_defaults, rng);
      auto pcm = decode_full_audio(frames);
      interleaved_all.insert(interleaved_all.end(), pcm.begin(), pcm.end());
      if (ci + 1 < chunks.size()) {
        int pause_samples = static_cast<int>(std::llround(pause_seconds_for_chunk(chunk) * sample_rate));
        interleaved_all.resize(interleaved_all.size() + static_cast<size_t>(pause_samples) * static_cast<size_t>(channels),
                               0.f);
      }
    }
    int samples = channels > 0 ? static_cast<int>(interleaved_all.size()) / channels : 0;
    return encode_wav_pcm16_stereo(interleaved_all, sample_rate, channels);
  }

  std::string voices_json() const {
    json out = json::array();
    for (const auto& row : manifest_["builtin_voices"]) {
      json item;
      item["voice"] = row["voice"];
      item["display_name"] = row["display_name"];
      item["group"] = row["group"];
      out.push_back(std::move(item));
    }
    return out.dump();
  }
};

MossTtsEngine::MossTtsEngine(std::filesystem::path model_dir, int thread_count)
    : impl_(std::make_unique<Impl>(std::move(model_dir), thread_count)) {}

MossTtsEngine::~MossTtsEngine() = default;

std::vector<uint8_t> MossTtsEngine::synthesize_wav(const std::string& text, const SynthOptions& opt) {
  std::lock_guard<std::mutex> lock(mutex_);
  return impl_->synthesize_wav(text, opt);
}

std::string MossTtsEngine::voices_json() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return impl_->voices_json();
}
