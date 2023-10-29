# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

As long as the mayor version is `0`, the API is not stable and may change with minor versions.

The minor version will be incremented if any of the following changes:
- The HTTP API
- The C API
- The inputs for the neural networks

This means you can reuse the same neural networks between for example 0.2.0 and 0.2.1, but not between 0.1.0 and 0.2.0.

## [Unreleased]

- `added` Rudimentary 2-ply evaluation of positions.
- `added` Batch inference of neural networks.
- `changed` Improved selection of positions for rollouts via self play.
- `changed` Different neural networks for _contact_ and _race_.
- `changed` Rollout data is now stored with GnuBG position IDs.

## 0.1.0 - 2023-10-17

Initial release of `wildbg`.

- `added` Move generation.
- `added` Finding positions for rollouts via self play.
- `added` Rollouts with fixed number of 1296 games for generating training data.
- `added` Training of new neural networks with `PyTorch`.
- `added` Inference of existing neural networks with [`tract`](https://github.com/sonos/tract).
- `added` Comparison of neural networks by playing against each other.
- `added` Implementation of the GnuBG position ID.
- `added` HTTP API for moves and cubes.
- `added` Simple C API for best move in 1-pointer.

### Thanks for contributions
- [@bungogood](https://github.com/bungogood)
- [@carsten-wenderdel](https://github.com/carsten-wenderdel)
- [@oradwastaken](https://github.com/oradwastaken)
