#!/bin/sh
# Build Edax 4.6 for Apple Silicon into third_party/edax (git-ignored).
# Edax is GPL-3.0 and used only as an external opponent: https://github.com/abulmo/edax-reversi
set -eu

root="$(cd "$(dirname "$0")/.." && pwd)"
work="$root/third_party"
mkdir -p "$work/edax/data" "$work/edax-release"

if [ ! -d "$work/edax-src" ]; then
    git clone --quiet --depth 1 --branch v4.6 https://github.com/abulmo/edax-reversi.git "$work/edax-src"
fi

# macOS rejects aligned_alloc sizes that are not a multiple of the alignment.
sed -i '' \
    's|aligned_alloc(32, (size + HASH_N_WAY + 1) \* sizeof (Hash))|aligned_alloc(32, adjust_size(32, (size + HASH_N_WAY + 1) * sizeof (Hash)))|' \
    "$work/edax-src/src/hash.c"

mkdir -p "$work/edax-src/bin"
make -C "$work/edax-src/src" build ARCH=armv8-a OS=osx CC=clang >/dev/null 2>&1
cp "$work/edax-src/bin/mEdax-armv8-a" "$work/edax/edax"

# The evaluation weights are only shipped inside the release archives.
archive="$work/edax-release/edax-4.6-linux-x86.tar.gz"
if [ ! -f "$archive" ]; then
    curl --silent --show-error --fail --location --output "$archive" \
        https://github.com/abulmo/edax-reversi/releases/download/v4.6/edax-4.6-linux-x86.tar.gz
fi
tar xzf "$archive" -C "$work/edax-release" data/eval.dat
cp "$work/edax-release/data/eval.dat" "$work/edax/data/eval.dat"

echo "built $work/edax/edax"
