// vpimgdiff A_DIR B_DIR [--max-diff N] [--min-psnr DB] [--all] : compares two vpreplay dump directories.
#include <cstdlib>
#include <cstring>

#include "vpimgdiff.h"

int main(int argc, char** argv) {
  vpimgdiff::Options opt;
  std::vector<std::string> dirs;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--max-diff") && i + 1 < argc) opt.maxDiff = uint32_t(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--min-psnr") && i + 1 < argc) opt.minPsnr = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--all")) opt.all = true;
    else dirs.push_back(argv[i]);
  }
  if (dirs.size() != 2) {
    std::fprintf(stderr, "usage: vpimgdiff A_DIR B_DIR [--max-diff N (8-bit units, default 0)] [--min-psnr DB] [--all]\n");
    return 2;
  }
  return vpimgdiff::compareDirs(dirs[0], dirs[1], opt, stdout);
}
