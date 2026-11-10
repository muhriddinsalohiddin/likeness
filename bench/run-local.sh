#!/bin/sh
# Full benchmark battery without Docker. Usage: ./run.sh            (all CPUs)
#                                               THREADS=4 ./run.sh  (cap ORT threads)
D="$(cd "$(dirname "$0")" && pwd)"
T="${THREADS:-$(nproc)}"
B="$D/facebench -lib $D/lib/libonnxruntime.so -models $D/models -spin=false"
echo "##### host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | nproc=$(nproc) | $(grep -m1 -o -w -E 'avx512_vnni' /proc/cpuinfo || echo no-vnni)"
$B -threads "$T" -sections decode,pre,search
for t in 1 2 4 "$T"; do $B -threads "$t" -iters 60 -sections models 2>&1 | grep -v -E '^(facebench|$)' | sed "s/^== Model.*/== Model inference, ORT threads=$t ==/"; done
for c in det_500m@320+w600k_r50 det_500m@320+w600k_mbf yunet_2023mar+sface_2021dec; do
  $B -threads "$T" -iters 150 -sections e2e,conc -combos "$c" -conc-combo "$c" -conc 1,2,3,4,8 2>&1 | grep -v -E '^(facebench|$)'
done
