// The JSON subset of vpreplay's index.json: a parser into a tree (objects keep their key order) and string escaping.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace vpjson {

struct Value {
  enum Type { Null, Bool, Number, String, Array, Object } type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::string> keys;  // Object: the keys in order, vals[i] their values (a vector of pairs needs Value complete)
  std::vector<Value> vals;

  const Value* get(const char* key) const {
    for (size_t i = 0; i < keys.size(); ++i)
      if (keys[i] == key) return &vals[i];
    return nullptr;
  }
  std::string text(const char* key, const std::string& def = {}) const {
    const Value* v = get(key);
    return v && v->type == String ? v->str : def;
  }
  double number(const char* key, double def = 0) const {
    const Value* v = get(key);
    return v && v->type == Number ? v->num : def;
  }
};

inline std::string escape(const std::string& s) {
  std::string o;
  for (const char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          o += b;
        } else {
          o += c;
        }
    }
  }
  return o;
}

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}
  bool parse(Value& out) {
    ws();
    if (!value(out)) return false;
    ws();
    return p_ == s_.size();
  }

 private:
  void ws() {
    while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\n' || s_[p_] == '\t' || s_[p_] == '\r')) ++p_;
  }
  bool lit(const char* w) {
    size_t n = 0;
    while (w[n]) ++n;
    if (s_.compare(p_, n, w) != 0) return false;
    p_ += n;
    return true;
  }
  // Four hex digits at p_ (not consumed here); false when any is not one.
  bool hex4(uint32_t& out) const {
    if (p_ + 4 > s_.size()) return false;
    out = 0;
    for (size_t i = 0; i < 4; ++i) {
      const char h = s_[p_ + i];
      const int d = h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;
      if (d < 0) return false;
      out = out << 4 | uint32_t(d);
    }
    return true;
  }
  static void utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += char(cp);
    else if (cp < 0x800) o += char(0xC0 | cp >> 6), o += char(0x80 | (cp & 0x3F));
    else if (cp < 0x10000) o += char(0xE0 | cp >> 12), o += char(0x80 | ((cp >> 6) & 0x3F)), o += char(0x80 | (cp & 0x3F));
    else o += char(0xF0 | cp >> 18), o += char(0x80 | ((cp >> 12) & 0x3F)), o += char(0x80 | ((cp >> 6) & 0x3F)), o += char(0x80 | (cp & 0x3F));
  }
  bool string(std::string& o) {
    if (p_ >= s_.size() || s_[p_] != '"') return false;
    ++p_;
    while (p_ < s_.size() && s_[p_] != '"') {
      char c = s_[p_++];
      if (c == '\\' && p_ < s_.size()) {
        c = s_[p_++];
        switch (c) {
          case 'n': c = '\n'; break;
          case 't': c = '\t'; break;
          case 'r': c = '\r'; break;
          case 'b': c = '\b'; break;
          case 'f': c = '\f'; break;
          case 'u': {  // \uXXXX as UTF-8, a surrogate pair as one code point
            uint32_t cp = 0;
            if (!hex4(cp)) return false;
            p_ += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && p_ + 6 <= s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') {
              const size_t keep = p_;
              uint32_t lo = 0;
              p_ += 2;
              if (hex4(lo) && lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00), p_ += 4;
              else p_ = keep;
            }
            utf8(o, cp);
            continue;
          }
          default: break;  // \" \\ \/
        }
      }
      o += c;
    }
    if (p_ >= s_.size()) return false;
    ++p_;
    return true;
  }
  // Nesting is bounded: a hostile file of nothing but '[' must not overflow the stack (index.json nests 3 levels).
  bool value(Value& v) {
    if (depth_ >= kMaxDepth) return false;
    ++depth_;
    const bool ok = valueAt(v);
    --depth_;
    return ok;
  }
  bool valueAt(Value& v) {
    if (p_ >= s_.size()) return false;
    const char c = s_[p_];
    if (c == '{') {
      v.type = Value::Object;
      ++p_, ws();
      if (p_ < s_.size() && s_[p_] == '}') return ++p_, true;
      for (;;) {
        std::string k;
        ws();
        if (!string(k)) return false;
        ws();
        if (p_ >= s_.size() || s_[p_++] != ':') return false;
        ws();
        v.keys.push_back(std::move(k));
        v.vals.emplace_back();
        if (!value(v.vals.back())) return false;
        ws();
        if (p_ >= s_.size()) return false;
        if (s_[p_] == ',') {
          ++p_;
          continue;
        }
        return s_[p_++] == '}';
      }
    }
    if (c == '[') {
      v.type = Value::Array;
      ++p_, ws();
      if (p_ < s_.size() && s_[p_] == ']') return ++p_, true;
      for (;;) {
        ws();
        v.arr.emplace_back();
        if (!value(v.arr.back())) return false;
        ws();
        if (p_ >= s_.size()) return false;
        if (s_[p_] == ',') {
          ++p_;
          continue;
        }
        return s_[p_++] == ']';
      }
    }
    if (c == '"') return v.type = Value::String, string(v.str);
    if (lit("true")) return v.type = Value::Bool, v.b = true, true;
    if (lit("false")) return v.type = Value::Bool, v.b = false, true;
    if (lit("null")) return v.type = Value::Null, true;
    if (c != '-' && (c < '0' || c > '9')) return false;  // strtod would also take "nan", "inf" and hex floats
    char* end = nullptr;
    v.num = std::strtod(s_.c_str() + p_, &end);
    if (end == s_.c_str() + p_) return false;
    v.type = Value::Number;
    p_ = size_t(end - s_.c_str());
    return true;
  }
  static constexpr int kMaxDepth = 64;
  const std::string& s_;
  size_t p_ = 0;
  int depth_ = 0;
};

}  // namespace vpjson
