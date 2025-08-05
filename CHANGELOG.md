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

## Unreleased

### Changed

- Default neural nets are now compiled into the executable.

### Internal / Training

- Use `CrossEntropyLoss` as PyTorch Optimizer and add `softmax` only after training.
- Reverted from `Hardsigmoid` back to `ReLU`.
- Rollouts and the training process are now deterministic.

## 0.2.0 - 2023-11-26

### Added

- The C API now supports raw evaluation of positions.
- Big speed up by batch inference of neural networks.

### Changed

- The C API doesn't need to reload the neural nets for every call.
- Different neural networks for _contact_ and _race_.

### Internal / Training

- Documentation for `engine` and the training process.
- Use `L1Loss` instead of `MSELoss` as loss function during supervised training.
- Use `AdamW` instead of `SGD` as PyTorch Optimizer.
- Use `Hardsigmoid` instead of `ReLU` for hidden layers.
- Rollout data is now stored with GnuBG position IDs.
- Improved selection of positions for rollouts via self play.

## 0.1.0 - 2023-10-17

Initial release of `wildbg`.

Thanks for their contributions:

- [@bungogood](https://github.com/bungogood) ([#10](https://github.com/carsten-wenderdel/wildbg/pull/10), [11](https://github.com/carsten-wenderdel/wildbg/pull/11))
- [@oradwastaken](https://github.com/oradwastaken) ([#7](https://github.com/carsten-wenderdel/wildbg/pull/7), [#9](https://github.com/carsten-wenderdel/wildbg/pull/9))

### Added

- Simple C API for best move in 1-pointer.
- HTTP API for moves and cubes.
- Inference of existing neural networks with [`tract`](https://github.com/sonos/tract).
- Move generation.

### Internal / Training

- Implementation of the GnuBG position ID.
- Comparison of neural networks by playing against each other.
- Training of new neural networks with `PyTorch`.
- Rollouts with fixed number of 1296 games for generating training data.
- Finding positions for rollouts via self play.
