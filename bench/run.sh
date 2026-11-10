#!/bin/sh
# Full benchmark battery. Override with: docker run ... -e THREADS=4
T="${THREADS:-$(nproc)}"
B="/app/facebench -lib /app/lib/libonnxruntime.so -models /app/models -spin=false"
echo "##### host: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | nproc=$(nproc) | flags: $(grep -m1 -o -w -E 'avx2|avx512f|avx512_vnni' /proc/cpuinfo | sort -u | tr '\n' ' ')"
$B -threads "$T" -sections decode,pre,search
for t in 1 2 4 "$T"; do $B -threads "$t" -iters 60 -sections models 2>&1 | grep -v -E '^(facebench|$)' | sed "s/^== Model.*/== Model inference, ORT threads=$t ==/"; done
for c in det_500m@320+w600k_r50 det_500m@320+w600k_mbf yunet_2023mar+sface_2021dec; do
  $B -threads "$T" -iters 150 -sections e2e,conc -combos "$c" -conc-combo "$c" -conc 1,2,3,4,8 2>&1 | grep -v -E '^(facebench|$)'
done
