# wildbg

`wildbg` is a backgammon engine based on neural networks. Currently, it's in alpha stage.

As of November 2023, it reaches an ELO rating of roughly 1800 and an error rate mEMG of roughly 7.5 when being analyzed with GnuBG.

For discussions, please join the Discord Server [Computer Backgammon](https://discord.gg/BcU9AzunGx) ![Discord Shield](https://discordapp.com/api/guilds/1159408833999945798/widget.png?style=shield).

## Try it out

#### FIBS

Thanks to [@oysteio](https://github.com/oysteijo) you can play against `wildbg` on the backgammon server [FIBS](http://www.fibs.com). As FIBS client I recommend [JavaFibs](http://www.fibs.com/javafibs/).

#### Online API

You can access the API and see yourself how `wildbg` would move: https://wildbg.shuttleapp.rs/swagger-ui/

An example for the starting position and rolling 3 and 1: https://wildbg.shuttleapp.rs/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2

#### Locally

Install Rust on your machine and then execute `cargo run` or `cargo run --release`.
A web server will be started which you can access via http://localhost:8080/swagger-ui/

Beware that the networks committed to this repository are very small networks just for demonstration purposes.
You can find the latest training progress and networks here: https://github.com/carsten-wenderdel/wildbg-training

## Goals

1. Provide source code and documentation to train neural nets from zero to super human strength.
2. Implement logic to evaluate all kind of backgammon positions: cubeless and cubeful equities, multi-ply evaluation, rollouts, etc.
3. Make the backgammon engine accessible via an easy-to-use HTTP JSON API.

A graphical user interface (GUI) is not part of this project.

## Training process

The training process consists of three steps, which are repeated in a loop:
1. Find lots of positions (at least 100,000) through self-play for a later rollout.
2. Roll out these positions. Currently, only 1-ply rollouts are possible.
3. Train neural networks based on the rollout data. There are two different networks for `contact` and `race` positions. This third step is the only one done in Python, everything else is implemented in Rust.

## Documentation

#### For users (bots and GUIs)
- HTTP API: https://wildbg.shuttleapp.rs/swagger-ui/
- C API: [docs/user/wildbg-c.md](docs/user/wildbg-c.md)

#### For contributors
- Code structure: [docs/dev/architecture.md](docs/dev/architecture.md)

Also see the [CHANGELOG](CHANGELOG.md) for a list of changes.

## Contributing

Help is more than welcome! There are some smaller tasks but also bigger ones, see https://github.com/carsten-wenderdel/wildbg/issues.
Currently, most needed is:
- Documentation of backgammon metrics / neural net inputs: https://github.com/carsten-wenderdel/wildbg/issues/4
- Proper cube handling: https://github.com/carsten-wenderdel/wildbg/issues/17

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
