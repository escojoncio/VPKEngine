#!/usr/bin/env bash
# compare.sh: replays a VPCAP capture on lavapipe (reference) and on VPVulkan/Metal, and reports the first divergence.
#   compare.sh <f.vpcap> [OUT]                      both legs and the diff into OUT (default replay_out/<name>): ref/ vpvk/ report.txt
#   compare.sh lvp  <f.vpcap> <DIR>                 reference leg only: lavapipe dumps (PNG + index.json) into DIR
#   compare.sh vpvk <f.vpcap> <DIR>                 VPVulkan leg only (pixels need Metal: the visionOS simulator build)
#   compare.sh diff <REF_DIR> <VPVK_DIR> [REPORT]   vpimgdiff + report; divergent PNGs of both sides go to <VPVK_DIR>.diverged/
# Env: VPREPLAY, VPIMGDIFF  gpu/vulkan build outputs.  LVP_ICD  lavapipe manifest (default /usr/share/vulkan/icd.d/lvp_icd*.json).
#      VPVK_LIB  libVPVulkan (default: next to VPREPLAY).  RUNNER  command prefix of the vpvk leg (gpu/metal/ci/simctl-run.sh).
#      MIN_PSNR  dB a dump must reach (default 40).  DUMP_AT  draw|pass (default draw).
# Exit: 0 match, 1 diverges (also: no dump in common, so nothing compared), 2 usage or replay error.
set -eu
: "${MIN_PSNR:=40}" "${DUMP_AT:=draw}"

die() { echo "compare.sh: $*" >&2; exit 2; }
usage() { sed -n '2,10p' "$0" | cut -c3- >&2; exit 2; }
abs() { local p=${1%/}; case $p in /*) echo "$p" ;; *) echo "$PWD/$p" ;; esac; }

# leg lvp|vpvk <cap> <dir>
leg() {
  local kind=$1 cap dir icd lib runner="" f rc=0
  cap=$(abs "$2"); dir=$(abs "$3")
  [ -f "$cap" ] || die "$cap: no such file"
  [ -n "${VPREPLAY:-}" ] || die "VPREPLAY is not set"
  if [ "$kind" = lvp ]; then
    icd=${LVP_ICD:-}
    if [ -z "$icd" ]; then
      for f in /usr/share/vulkan/icd.d/lvp_icd*.json; do [ -f "$f" ] && icd=$f && break; done
    fi
    [ -n "$icd" ] || die "lavapipe not found (package mesa-vulkan-drivers); set LVP_ICD"
    set -- --icd system --icd-file "$icd" --strict
  else
    lib=${VPVK_LIB:-}
    if [ -z "$lib" ]; then
      for f in "$(dirname "$VPREPLAY")/libVPVulkan.so" "$(dirname "$VPREPLAY")/libVPVulkan.dylib"; do [ -f "$f" ] && lib=$f && break; done
    fi
    [ -n "$lib" ] || die "libVPVulkan not found; set VPVK_LIB"
    runner=${RUNNER:-}
    set -- --icd vpvk --lib "$(abs "$lib")" --strict
  fi
  rm -rf "$dir"; mkdir -p "$dir"
  # The driver logs a lot on stdout: all of it goes to replay.log next to the dumps.
  $runner "$(abs "$VPREPLAY")" "$cap" "$@" --dump "$dir" --dump-at "$DUMP_AT" > "$dir/replay.log" 2>&1 || rc=$?
  grep -E '^vpreplay: (calls=|capture )' "$dir/replay.log" | sed "s|^|$kind: |" || true
  if [ "$rc" -ne 0 ] || [ ! -f "$dir/index.json" ]; then
    tail -n 20 "$dir/replay.log" >&2
    die "$kind replay of $(basename "$cap") failed (exit $rc, ${dir}/replay.log)"
  fi
}

# diff_dirs <ref> <vpvk> [report]
diff_dirs() {
  local ref vp rep div out rc=0 f
  ref=$(abs "$1"); vp=$(abs "$2"); rep=$(abs "${3:-$2.report.txt}"); div=$vp.diverged
  [ -n "${VPIMGDIFF:-}" ] || die "VPIMGDIFF is not set"
  rm -rf "$div"
  # PSNR is the only criterion: --max-diff 255 disables the per-channel limit (the default, 0, accepts no difference at all).
  out=$("$VPIMGDIFF" "$ref" "$vp" --max-diff 255 --min-psnr "$MIN_PSNR" --all) || rc=$?
  [ "$rc" -le 1 ] || { echo "$out" >&2; die "vpimgdiff failed (exit $rc)"; }
  {
    echo "replay comparison $(basename "$vp"): A = $ref (lavapipe), B = $vp (VPVulkan); each dump must reach $MIN_PSNR dB"
    for f in "$ref/replay.log" "$vp/replay.log"; do [ -f "$f" ] && grep -H -E '^vpreplay: (calls=|capture )' "$f" || true; done
    echo "$out"
    if [ "$rc" -eq 0 ]; then
      echo "RESULT: match"
    else
      echo "RESULT: DIVERGES. $(echo "$out" | grep -m1 -E '^(FIRST DIVERGENCE|NOTHING COMPARED)' || echo 'extra dumps in B only')"
    fi
  } > "$rep"
  if [ "$rc" -ne 0 ]; then
    mkdir -p "$div"
    echo "$out" | sed -n -E 's/^(FIRST DIVERGENCE|divergence) ([^:]*):.*/\2/p' | while read -r f; do
      cp "$ref/$f" "$div/ref_$f" 2>/dev/null || true
      cp "$vp/$f" "$div/vpvk_$f" 2>/dev/null || true
    done
  fi
  cat "$rep"
  return "$rc"
}

case ${1:-} in
  lvp|vpvk) [ $# -eq 3 ] || usage; leg "$1" "$2" "$3" ;;
  diff) [ $# -ge 3 ] && [ $# -le 4 ] || usage; diff_dirs "$2" "$3" "${4:-}" ;;
  ""|-h|--help) usage ;;
  *)
    [ $# -le 2 ] || usage
    name=$(basename "$1" .vpcap)
    out=${2:-replay_out/$name}
    leg lvp "$1" "$out/ref"
    leg vpvk "$1" "$out/vpvk"
    diff_dirs "$out/ref" "$out/vpvk" "$out/report.txt"
    ;;
esac
