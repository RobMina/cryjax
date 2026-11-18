#!/usr/bin/env bash
# Build the CRY shim: download the CRY 1.7 release (checksum-verified), compile CRY + cry_shim.cc into one shared
# library. Needs bash, curl, sha256sum and a C++17 compiler. Prints the two variables cryjax reads.
#   bash build-shim.sh OUTDIR
set -euo pipefail
out=${1:?usage: build-shim.sh OUTDIR}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
cd "$out"
if [ ! -f cry_v1.7.tar.gz ]; then
  curl --fail --location --retry 3 --max-time 180 https://nuclear.llnl.gov/simulation/cry_v1.7.tar.gz -o cry_v1.7.tar.gz
fi
echo 'dcee2428f81cba113f82e0c7c42f4d85bff4b8530e5ab5c82c059bed3e570c20  cry_v1.7.tar.gz' | sha256sum -c -
rm -rf cry_v1.7 && tar -xzf cry_v1.7.tar.gz
( cd cry_v1.7/data
  echo 'b5227d8c1d38ab249a3b238456a67a60c9a9a5b6f91f748562f4a517390dca75  cosmics_0.data' | sha256sum -c -
  echo '86cc99fb87fad93896d307fdbe8a90c63855e02c788bfb1eafee77c4c8254e95  cosmics_2100.data' | sha256sum -c -
  echo '8d583055d1c38a941c1e84a7843337ef4e00bce7d95df7dbb71f2947e2aec9e8  cosmics_11300.data' | sha256sum -c - )
${CXX:-g++} -std=c++17 -O2 -fPIC -shared -pthread -Wl,-z,defs -Icry_v1.7/src \
  "$here/cry_shim.cc" cry_v1.7/src/CRY*.cc -o libcryshim.so
echo "export CRYDATAPATH=$out/cry_v1.7/data"
echo "export CRYJAX_SHIM=$out/libcryshim.so"
