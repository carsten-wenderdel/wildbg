# Engine

This document describes how `engine` works internally.
The embedded code reflects the state of the repository in November 2023.

For an overview over the whole project, see [docs/dev/architecture.md](./architecture.md).

## Overview: Board game

When a computer plays a board game, it typically follows this pattern:
1. For a given position (and in case of backgammon, a pair of dice), generate all possible moves. Then generate all positions following these moves.
2. Evaluate the value of each of these positions and pick the position that's worst for the opponent.

So the computer doesn't compare moves, but positions. The evaluation function typically returns a floating point number. The higher the number, the better the position.

In case of backgammon, we often use cubeless money play evaluation.
This means: If the game is guaranteed to be lost with 1 point, its value is `-1.0`.
If the game is guaranteed to be won with a gammon, its value is `2.0`.
If the game is guaranteed to be lost with a backgammon, its value is `-3.0`.
If both players have identical chances, its value is `0.0`.
For more complicated situations, this value can be anything between `-3.0` and `3.0`.

`wildbg` uses the output of a neural net for this evaluation.

## `Position` and move generation

The central struct is `Position`, which encodes the state of the board. `engine` doesn't know anything about cubes.

https://github.com/carsten-wenderdel/wildbg/blob/d5c7280a60a52cb61c92af78018fb811cf3dd223/crates/engine/src/position.rs#L94-L105

An important function is the following. It calculates all possible moves for a given position and a pair of dice.
`engine` doesn't have a data type for `Move`, so we already return the resulting positions.

https://github.com/carsten-wenderdel/wildbg/blob/d5c7280a60a52cb61c92af78018fb811cf3dd223/crates/engine/src/position.rs#L184-L185

## Evaluators

The most important trait in `wildbg` is `Evaluator`.
For a given position in cubeless money game, it returns the probabilities to win/lose normal/gammon/backgammon.

https://github.com/carsten-wenderdel/wildbg/blob/d5c7280a60a52cb61c92af78018fb811cf3dd223/crates/engine/src/evaluator.rs#L31

Examples of implementations are
- `GameOverEvaluator` for positions where the game is over and the probabilities are known.
- `OnnxEvaluator` which uses a neural net in `ONNX` format to calculate the probabilities.
- `CompositeEvaluator` which consists of a `GameOverEvaluator` and other evaluators for different game phases. It decides which evaluator to use based on the position.
- `RolloutEvaluator` which uses another evaluator to do rollouts.

## Inputs generation

There are different inputs for `race` and `contact` positions.
Each position type needs a struct implementing the trait `InputsGen` with this function:

https://github.com/carsten-wenderdel/wildbg/blob/d5c7280a60a52cb61c92af78018fb811cf3dd223/crates/engine/src/inputs.rs#L10

Each pip usually translates to 4 inputs per player. For `race` positions we don't encode the `bar` and the `24` point for each player, as there never will be a checker.

For the encoding, `wildbg` was inspired by TD-Gammon and GnuBG. See also https://stackoverflow.com/questions/32428237/board-encoding-in-tesauros-td-gammon

`wildbg` doesn't use any expert features as introduced by Hans Berliner in his famous backgammon program _BKG 9.8_.
That is a TODO for the future.

## Neural net inference

All the neural nets are stored in the `ONNX` format, for inference we rely on the open source library [`tract`](https://github.com/sonos/tract).
The benefit of this approach is that everything is done in Rust, no Python installation is needed.
At some point `wildbg` should be able to run on smartphones or maybe via WebAssembly in the browser.

The `ONNX` format was chosen out of convenience, it might be replaced by something else (NNEF, CoreML) in the future.
It's also possible that we might use other libraries for inference, replacing or complementing `tract`.

To allow batch evaluation of several positions at once, `OnnxEvaluator` implements `BatchEvaluator`:

https://github.com/carsten-wenderdel/wildbg/blob/d5c7280a60a52cb61c92af78018fb811cf3dd223/crates/engine/src/onnx.rs#L30

## Best move

Each `evaluator` has several convenience functions with default implementations. One example is the `best_position_by_equity`
which is used during rollouts:

https://github.com/carsten-wenderdel/wildbg/blob/d5c7280a60a52cb61c92af78018fb811cf3dd223/crates/engine/src/evaluator.rs#L50

Those functions only work for cubeless money games. In money games, for example a gammon is twice as valuable as winning normal.
In a 1-pointer however a gammon has the same worth as a normal win. In those cases a custom `value` function is needed:

https://github.com/carsten-wenderdel/wildbg/blob/d5c7280a60a52cb61c92af78018fb811cf3dd223/crates/engine/src/evaluator.rs#L57-L60
