# wildbg

`wildbg` is a backgammon engine based on neural networks. Currently, it's in a pre-alpha phase.

## Try it out

#### Online

You can access the API and see yourself how `wildbg` would move: https://wildbg.shuttleapp.rs/swagger-ui/

An example for the starting position and rolling 3 and 1: https://wildbg.shuttleapp.rs/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2

#### Locally

Install Rust on your machine and then execute `cargo run` or `cargo run --release`.
A web server will be started which you can access via http://localhost:8080/swagger-ui/

## Goals

1. Provide source code and documentation to train neural nets from zero to super human strength.
2. Implement logic to evaluate all kind of backgammon positions: cubeless and cubeful equities, multi-ply evaluation, rollouts, etc.
3. Make the backgammon engine accessible via an easy-to-use HTTP/json API.

A graphical user interface (GUI) is not part of this project.

## Current state

#### Topic 1: Neural nets

The training process consists of three steps, which are repeated in a loop:
1. Find lots of positions for a later rollout. Currently, rather random self play is used; later we want to make sure to find all kind of positions, including backgames.
2. Roll out these positions. Currently, only 1-ply rollouts are possible.
3. Train neural networks based on the rollout data. Currently, a single net with several hidden layers is supported; later different nets for different game phases are planned.

Already implemented is:
* Roll out a certain position 1296 times, multithreaded.
* 202 neural net inputs similar to TD-Gammon, representing the raw board.
* Find 100,000 random positions through self play for later rollouts.
* Train a single neural net with one hidden layer via PyTorch and save the result as an ONNX file. The net has six outputs for winning/losing 1, 2 or 3 points.
* Inference of that neural net in Rust via [tract](https://github.com/sonos/tract).

An older, weaker neural network is committed to this repository.  You can find the latest training progress and networks here: https://github.com/carsten-wenderdel/wildbg-training

#### Topic 2: Backgammon logic
Currently only cubeless equities and moves are implemented. Cubes and cubeful equities are missing.

#### Topic 3: HTTP/json API

Getting the best move is already implemented: https://wildbg.shuttleapp.rs/swagger-ui/

### Installation of python environment

Make an editable install by targeting the training directory:
``` pip install -e "training/[dev]"```

## Contributing

Help is more than welcome! There are some smaller tasks but also bigger ones, see https://github.com/carsten-wenderdel/wildbg/issues.
Currently, most needed is:
- Documentation of backgammon metrics / neural net inputs: https://github.com/carsten-wenderdel/wildbg/issues/4
- Implementation of a bearoff database: https://github.com/carsten-wenderdel/wildbg/issues/1

## License

Licensed under either of

* Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
