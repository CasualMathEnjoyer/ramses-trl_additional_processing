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

## Original Repository

The original RAMSES-TRL repository is maintained at: https://gitlab2.cnam.fr/rosmorse/ramses-trl

