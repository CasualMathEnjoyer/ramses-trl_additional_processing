#!/bin/bash
# Script to apply custom vocabulary size patch to ramses-trl submodule
# This patch adds support for different vocabulary sizes in the encoder/decoder

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/ramses-trl-vocab-size-patch.patch"
SUBMODULE_DIR="${SCRIPT_DIR}/ramses-trl"

echo "Applying vocabulary size patch to ramses-trl submodule..."

# Check if patch file exists
if [ ! -f "$PATCH_FILE" ]; then
    echo "Error: Patch file not found at $PATCH_FILE"
    exit 1
fi

# Check if submodule directory exists
if [ ! -d "$SUBMODULE_DIR" ]; then
    echo "Error: Submodule directory not found at $SUBMODULE_DIR"
    echo "Please initialize the submodule first with: git submodule update --init"
    exit 1
fi

# Navigate to submodule directory
cd "$SUBMODULE_DIR"

# Check if we're in a git repository (submodules have .git as a file)
if [ ! -e ".git" ]; then
    echo "Error: Not a git repository. Please initialize the submodule first."
    exit 1
fi

# Check if patch is already applied
if git apply --check "$PATCH_FILE" 2>/dev/null; then
    echo "Patch can be applied. Applying now..."
    git apply "$PATCH_FILE"
    echo "✓ Patch applied successfully!"
    echo ""
    echo "Note: These changes are uncommitted in the submodule."
    echo "They will persist as long as you don't reset the submodule."
elif git diff --quiet python/translit_lib/encoder_decoder.py 2>/dev/null; then
    # Check if the changes are already present (maybe manually applied)
    if git diff python/translit_lib/encoder_decoder.py | grep -q "decoder_vocab_size"; then
        echo "✓ Patch appears to already be applied (changes detected in encoder_decoder.py)"
    else
        echo "Warning: Patch cannot be applied cleanly."
        echo "The file may have been modified or the patch is already applied."
        echo "Current status:"
        git status --short python/translit_lib/encoder_decoder.py || true
    fi
else
    echo "✓ Changes already present in working directory"
    git status --short python/translit_lib/encoder_decoder.py || true
fi

cd "$SCRIPT_DIR"
echo ""
echo "Setup complete!"

