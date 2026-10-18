#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
fetch() {
  local name="$1" url="$2" commit="$3" target="$root/third_party/$1"
  if [[ ! -d "$target/.git" ]]; then
    git clone --no-checkout "$url" "$target"
  fi
  if ! git -C "$target" cat-file -e "$commit^{commit}"; then
    git -C "$target" fetch origin "$commit"
  fi
  [[ "$(git -C "$target" remote get-url origin)" == "$url" ]] || { echo "Unexpected origin: $name" >&2; exit 1; }
  git -C "$target" checkout --detach "$commit"
  [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]]
  shopt -s nullglob
  local patch
  for patch in "$root/third_party/patches/$name/"*.patch; do
    if git -C "$target" apply --reverse --check "$patch"; then
      echo "Already applied: $patch"
    else
      git -C "$target" apply --check "$patch"
      git -C "$target" apply "$patch"
    fi
  done
}
fetch loss-of-plasticity https://github.com/shibhansh/loss-of-plasticity.git a6b79580d85f3025bdb601566d3627c5f489f13b
fetch PyCIL https://github.com/LAMDA-CL/PyCIL.git f3509b8ca3f20660ce4aa13f19d5283de81b4b35
fetch RepDistiller https://github.com/HobbitLong/RepDistiller.git b84f547c5db6a35318d4671d7d5c4de74c822403
fetch upgd https://github.com/mohmdelsayed/upgd.git b75e90ad4b09c28971ac9dbb902a8fd86709b28c
fetch C-Flat https://github.com/WanNaa/C-Flat.git f57a557523136834bdc28a4902cde9842859b2a8
fetch warm_start https://github.com/JordanAsh/warm_start.git 896176b2cf89f259bcde8e241a8987e6019741d0
fetch dopamine https://github.com/google/dopamine.git 5873f5494ee0c2d7c016d0ab2ad530354fec59d0
