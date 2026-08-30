#!/usr/bin/env bash
#
# Fetch the HELMET image archives from OSF node 4pwj8 (P4-U5).
#
# The seven part_N.zip archives total ~28.9 GB. Files are addressed by their stable
# OSF GUIDs, and every download is checked against the byte size the OSF API reports
# before it is accepted.
#
# Deliberately NO curl -C/resume: OSF's redirect target does not reliably honour
# Range requests, and a resumed part_6 came back 55 MB LARGER than the source. A
# clean re-fetch is the only safe retry, and the size check below is what caught the
# corruption in the first place.
#
# The licence and access gates in registry/datasets/helmet-myanmar.yaml must be
# resolved before running this (dataset-policy §14). The three small files -- the
# readme, data_split.csv and annotation.zip -- are fetched separately per the
# runbook in README.md.
#
# Usage:  bash experiments/helmet_cnn_vit/fetch_helmet.sh [destination]
#         (default destination: <repo>/data/raw/helmet-myanmar)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST="${1:-$REPO_ROOT/data/raw/helmet-myanmar}"

mkdir -p "$DEST/image"
cd "$DEST" || { echo "cannot enter $DEST" >&2; exit 1; }

declare -A G=( [part_1]=452nq [part_2]=muzgb [part_3]=9dmw3 [part_4]=39ynb [part_5]=v3rch [part_6]=jyg9s [part_7]=6h5ka )
declare -A S=( [part_1]=4693924414 [part_2]=4763012557 [part_3]=4487756257 [part_4]=4729671413 [part_5]=4744737081 [part_6]=4349041754 [part_7]=1092842740 )

status=0
for p in part_1 part_2 part_3 part_4 part_5 part_6 part_7; do
  f="image/$p.zip"
  if [ -f "$f" ] && [ "$(stat -c%s "$f")" = "${S[$p]}" ]; then echo "[skip] $p already complete"; continue; fi
  rm -f "$f"
  echo "[fetch] $p -> ${G[$p]}  ($(( ${S[$p]} / 1048576 )) MiB)"
  curl -sS -L --retry 5 --retry-delay 5 --retry-connrefused -o "$f" "https://osf.io/download/${G[$p]}/"
  got=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$got" = "${S[$p]}" ]; then
    echo "[ok]   $p $got bytes"
  else
    echo "[BAD]  $p expected ${S[$p]} got $got" >&2
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "[done] one or more parts are incomplete; re-run to re-fetch them" >&2
else
  echo "[done] all seven parts present and size-verified"
fi
exit "$status"
