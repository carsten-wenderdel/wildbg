# wildbg

`wildbg` is a backgammon engine based on neural networks. Currently, it's in alpha stage.

As of January 2024, it reaches an error rate of roughly 5.9 for 1-pointers when being analyzed with GnuBG 2-ply.

For discussions, please join the Discord Server [Computer Backgammon](https://discord.gg/BcU9AzunGx) ![Discord Shield](https://discordapp.com/api/guilds/1159408833999945798/widget.png?style=shield).

## Goals

1. Provide source code and documentation to train neural nets from zero to super human strength.
2. Implement logic to evaluate all kind of backgammon positions: cubeless and cubeful equities, multi-ply evaluation, rollouts, etc.
3. Make the backgammon engine accessible via an easy-to-use HTTP JSON API.

A graphical user interface (GUI) is not part of this project.

## Try it out

### Bots on Backgammon Servers

#### bgammon.org

Thanks to [@tslocum](https://github.com/tslocum) you can play against `BOT_wildbg` on his new backgammon server https://bgammon.org.
No need to download a client or register.

The source code of the bot can be found [here](https://code.rocket9labs.com/tslocum/bgammon-wildbg-bot).
There are also [winning statistics](https://bgammon.org/stats-wildbg/) available.

#### OpenGammon

On [OpenGammon.org](https://beta.opengammon.com/home/) you can play against `WildBG`.

#### FIBS

Thanks to [@oysteijo](https://github.com/oysteijo) you can play against `wildbg` on the backgammon server [FIBS](http://www.fibs.com). As FIBS client I recommend [JavaFibs](http://www.fibs.com/javafibs/).

### HTTP API

#### Online

You can access the API and see yourself how `wildbg` would move: https://wildbg.shuttleapp.rs/swagger-ui/

An example for the starting position and rolling 3 and 1: https://wildbg.shuttleapp.rs/move?die1=3&die2=1&p24=2&p19=-5&p17=-3&p13=5&p12=-5&p8=3&p6=5&p1=-2

#### Locally

Install Rust on your machine and then execute `cargo run` or `cargo run --release`.
A web server will be started which you can access via http://localhost:8080/swagger-ui/

Beware that the networks committed to this repository are very small networks just for demonstration purposes.
You can find the latest training progress and networks here: https://github.com/carsten-wenderdel/wildbg-training

## Documentation

#### For users (bots and GUIs)
- HTTP API: https://wildbg.shuttleapp.rs/swagger-ui/
- C API: [docs/user/wildbg-c.md](docs/user/wildbg-c.md)

#### For contributors
- Code structure: [docs/dev/architecture.md](docs/dev/architecture.md)
- Engine: [docs/dev/engine.md](docs/dev/engine.md)
- Training process: [docs/dev/training.md](docs/dev/training.md)

Also see the [CHANGELOG](CHANGELOG.md) for a list of changes.

## Acknowledgments

This project is inspired and influenced by other backgammon engines:

- [TD-Gammon](https://bkgm.com/articles/authors.html#tesauro_gerald) by Gerald Tesauro brought the idea of using neural networks to backgammon
- [GnuBG](https://www.gnu.org/software/gnubg/) - The strongest open source backgammon engine

Thanks to JetBrains for providing a free license for their IDEs via their [Open Source Support Program](https://jb.gg/OpenSourceSupport).

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
