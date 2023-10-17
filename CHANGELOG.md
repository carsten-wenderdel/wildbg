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

## 0.1.0 - 2023-10-17

Initial release of `wildbg`.

### Added

- Move generation.
- Finding positions for rollouts via self play.
- Rollouts with fixed number of 1296 games for generating training data.
- Training of new neural networks with `PyTorch`.
- Inference of existing neural networks with [`tract`](https://github.com/sonos/tract).
- Comparison of neural networks by playing against each other.
- Implementation of the GnuBG position ID.
- HTTP API for moves and cubes.
- Simple C API for best move in 1-pointer.

### Thanks for contributions

- @bungogood
- @carsten-wenderdel
- @oradwastaken
