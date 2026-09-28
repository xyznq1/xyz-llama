#!/usr/bin/env bash
# xyz-serve.sh [MODEL] -- Ternary Bonsai 2 27B + our xyz v1.2 drafter: 160k context in about 8 GB of GPU memory, OpenAI API on :8080.
# No MODEL given and none in models/? The first start downloads it (Hugging Face, cached after that), and our drafter too.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
model="${1:-$root/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf}"
drafter="${XYZ_DRAFTER:-$root/models/xyz-v1.2-drafter.gguf}"
server="${XYZ_SERVER:-$root/build/bin/llama-server}"
drafter_url=https://github.com/xyznq1/xyz-drafter/releases/download/v1.2/xyz-v1.2-drafter.gguf

m=(-m "$model")
[ $# -gt 0 ] || [ -f "$model" ] || m=(-hf prism-ml/Ternary-Bonsai-2-27B-gguf -hff Ternary-Bonsai-2-27B-PTQ1_0.gguf)
if [ ! -f "$drafter" ]; then
    echo "Downloading our drafter from $drafter_url"
    curl -fL --create-dirs -o "$drafter.part" "$drafter_url"
    mv "$drafter.part" "$drafter"
fi

exec "$server" "${m[@]}" -md "$drafter" --no-mmproj -ngl 999 -c 163840 -fa on -ctk xyzkv2 -ctv xyzkv2 -ctkd q4_0 -ctvd q4_0 \
    --spec-type draft-xyz --spec-draft-n-max 4 --spec-coupled --spec-rejection --parallel 1
