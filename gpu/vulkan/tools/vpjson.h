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
          case 'u':
            if (p_ + 4 > s_.size()) return false;
            c = char(std::strtol(s_.substr(p_, 4).c_str(), nullptr, 16));
            p_ += 4;
            break;
          default: break;
        }
      }
      o += c;
    }
    if (p_ >= s_.size()) return false;
    ++p_;
    return true;
  }
  bool value(Value& v) {
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
    char* end = nullptr;
    v.num = std::strtod(s_.c_str() + p_, &end);
    if (end == s_.c_str() + p_) return false;
    v.type = Value::Number;
    p_ = size_t(end - s_.c_str());
    return true;
  }
  const std::string& s_;
  size_t p_ = 0;
};

}  // namespace vpjson
