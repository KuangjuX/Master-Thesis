#!/bin/bash
# ============================================================
# AffineGraph IR Evaluation: Lines of Code Counter
# ============================================================
# This script helps you collect LoC data from various frameworks.
#
# Usage:
#   1. Clone the repos listed below into a common directory
#   2. Run: bash scripts/count_loc.sh <repos_dir>
#
# Example:
#   mkdir -p ~/repos/eval && cd ~/repos/eval
#   git clone https://github.com/TiledTensor/TiledLower.git
#   git clone https://github.com/Dao-AILab/flash-attention.git
#   git clone https://github.com/NVIDIA/cutlass.git
#   git clone https://github.com/triton-lang/triton.git
#   git clone https://github.com/HazyResearch/ThunderKittens.git
#   git clone https://github.com/microsoft/TileLang.git
#   bash ~/Papers/Master-Thesis/scripts/count_loc.sh ~/repos/eval
# ============================================================

set -e

REPOS_DIR="${1:-.}"
OUTPUT_FILE="$(dirname "$0")/../data/loc_results.csv"
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "framework,algorithm,file,total_lines,code_lines,comment_lines,blank_lines" > "$OUTPUT_FILE"

# Helper: count non-blank, non-comment lines for a file
count_code_lines() {
    local file="$1"
    local ext="${file##*.}"
    
    local total=$(wc -l < "$file" 2>/dev/null || echo 0)
    local blank=$(grep -c '^\s*$' "$file" 2>/dev/null || echo 0)
    local comments=0
    
    case "$ext" in
        cu|cuh|cpp|h|hpp)
            # C/C++/CUDA: count // and /* */ comments (simplified)
            comments=$(grep -c '^\s*//' "$file" 2>/dev/null || echo 0)
            ;;
        py)
            # Python: count # comments
            comments=$(grep -c '^\s*#' "$file" 2>/dev/null || echo 0)
            ;;
        rs)
            # Rust: count // comments
            comments=$(grep -c '^\s*//' "$file" 2>/dev/null || echo 0)
            ;;
    esac
    
    local code=$((total - blank - comments))
    echo "$total,$code,$comments,$blank"
}

echo "============================================================"
echo "  AffineGraph IR Evaluation: LoC Counter"
echo "============================================================"
echo ""

# ============================================================
# 1. TiledLower (AffineGraph IR)
# ============================================================
TILED_LOWER="$REPOS_DIR/TiledLower"
if [ -d "$TILED_LOWER" ]; then
    echo "[AffineGraph] Scanning $TILED_LOWER/examples/ ..."
    for f in "$TILED_LOWER"/examples/*.rs; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        algo="unknown"
        case "$basename" in
            *gemm*|*GEMM*) algo="GEMM" ;;
            *attention*|*flash*) algo="FlashAttention-2" ;;
            *softmax*) algo="Fused Softmax" ;;
            *norm*) algo="LayerNorm" ;;
            *) algo="$basename" ;;
        esac
        stats=$(count_code_lines "$f")
        echo "AffineGraph,$algo,$basename,$stats" >> "$OUTPUT_FILE"
        echo "  [AffineGraph] $basename -> $stats"
    done
else
    echo "[WARN] TiledLower not found at $TILED_LOWER"
fi

# ============================================================
# 2. FlashAttention (CUDA)
# ============================================================
FA_DIR="$REPOS_DIR/flash-attention"
if [ -d "$FA_DIR" ]; then
    echo ""
    echo "[CUDA/FlashAttention] Scanning $FA_DIR/csrc/ ..."
    # Count the core kernel files
    for f in "$FA_DIR"/csrc/flash_attn/*.cu "$FA_DIR"/csrc/flash_attn/*.cuh \
             "$FA_DIR"/hopper/*.cu "$FA_DIR"/hopper/*.cuh; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        stats=$(count_code_lines "$f")
        echo "CUDA,FlashAttention-2,$basename,$stats" >> "$OUTPUT_FILE"
        echo "  [CUDA] $basename -> $stats"
    done
else
    echo "[WARN] flash-attention not found at $FA_DIR"
fi

# ============================================================
# 3. CUTLASS / CuTe
# ============================================================
CUTLASS_DIR="$REPOS_DIR/cutlass"
if [ -d "$CUTLASS_DIR" ]; then
    echo ""
    echo "[CuTe/CUTLASS] Scanning examples ..."
    # GEMM examples
    for f in "$CUTLASS_DIR"/examples/cute/tutorial/*.cu; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        stats=$(count_code_lines "$f")
        echo "CuTe,Tutorial,$basename,$stats" >> "$OUTPUT_FILE"
        echo "  [CuTe] $basename -> $stats"
    done
else
    echo "[WARN] cutlass not found at $CUTLASS_DIR"
fi

# ============================================================
# 4. ThunderKittens
# ============================================================
TK_DIR="$REPOS_DIR/ThunderKittens"
if [ -d "$TK_DIR" ]; then
    echo ""
    echo "[ThunderKittens] Scanning kernels ..."
    for f in "$TK_DIR"/kernels/attn/*.cu "$TK_DIR"/kernels/attn/*.cuh \
             "$TK_DIR"/kernels/gemm/*.cu "$TK_DIR"/kernels/gemm/*.cuh; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        algo="unknown"
        case "$basename" in
            *gemm*|*GEMM*) algo="GEMM" ;;
            *attn*|*attention*) algo="FlashAttention-2" ;;
            *) algo="$basename" ;;
        esac
        stats=$(count_code_lines "$f")
        echo "ThunderKittens,$algo,$basename,$stats" >> "$OUTPUT_FILE"
        echo "  [TK] $basename -> $stats"
    done
else
    echo "[WARN] ThunderKittens not found at $TK_DIR"
fi

# ============================================================
# 5. Triton
# ============================================================
TRITON_DIR="$REPOS_DIR/triton"
if [ -d "$TRITON_DIR" ]; then
    echo ""
    echo "[Triton] Scanning tutorials ..."
    for f in "$TRITON_DIR"/python/tutorials/*.py; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        algo="unknown"
        case "$basename" in
            *gemm*|*matmul*) algo="GEMM" ;;
            *attention*|*flash*) algo="FlashAttention-2" ;;
            *softmax*) algo="Fused Softmax" ;;
            *norm*) algo="LayerNorm" ;;
            *) algo="$basename" ;;
        esac
        stats=$(count_code_lines "$f")
        echo "Triton,$algo,$basename,$stats" >> "$OUTPUT_FILE"
        echo "  [Triton] $basename -> $stats"
    done
else
    echo "[WARN] triton not found at $TRITON_DIR"
fi

# ============================================================
# 6. TileLang
# ============================================================
TILELANG_DIR="$REPOS_DIR/TileLang"
if [ -d "$TILELANG_DIR" ]; then
    echo ""
    echo "[TileLang] Scanning examples ..."
    for f in "$TILELANG_DIR"/examples/*.py "$TILELANG_DIR"/testing/*.py; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        algo="unknown"
        case "$basename" in
            *gemm*|*matmul*) algo="GEMM" ;;
            *attention*|*flash*) algo="FlashAttention-2" ;;
            *softmax*) algo="Fused Softmax" ;;
            *norm*) algo="LayerNorm" ;;
            *) algo="$basename" ;;
        esac
        stats=$(count_code_lines "$f")
        echo "TileLang,$algo,$basename,$stats" >> "$OUTPUT_FILE"
        echo "  [TileLang] $basename -> $stats"
    done
else
    echo "[WARN] TileLang not found at $TILELANG_DIR"
fi

echo ""
echo "============================================================"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review the CSV and identify the kernel files for each algorithm"
echo "  2. For multi-file kernels, sum the relevant files"
echo "  3. Update scripts/generate_ir_evaluation.py with real data"
echo "  4. Re-run: python3 scripts/generate_ir_evaluation.py"
