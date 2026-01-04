# RAMSES-TRL Additional Processing

This repository contains additional processing scripts and tools built around the [RAMSES-TRL](https://gitlab2.cnam.fr/rosmorse/ramses-trl) project.

## Repository Structure

- `ramses-trl/` - Git submodule containing the original RAMSES-TRL repository
- Additional scripts and tools (to be added)

## Setup

This repository uses a git submodule. To clone with the submodule automatically:

```bash
git clone --recurse-submodules <repository-url>
```

If you've already cloned without submodules, initialize them:

```bash
git submodule update --init --recursive
```

To update the submodule to the latest commit: `git submodule update --remote ramses-trl`

### Applying Custom Vocabulary Size Patch

This repository includes a custom patch that adds support for different vocabulary sizes in the encoder/decoder. This is required for some models that use non-standard vocabulary sizes.

After cloning or updating the submodule, apply the patch:

```bash
./apply_vocab_patch.sh
```

The patch adds the following features to `ramses-trl`:
- Support for custom `decoder_vocab_size` parameter (default: 256)
- Support for custom `decoder_char_vocab` for character mapping
- Improved handling of BOS/EOS tokens

**Note:** The patch creates uncommitted changes in the submodule. These changes will persist as long as you don't reset the submodule. If you update the submodule, you'll need to reapply the patch.

## Original Repository

The original RAMSES-TRL repository is maintained at: https://gitlab2.cnam.fr/rosmorse/ramses-trl

