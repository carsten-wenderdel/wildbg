# Training

The training code is split into two different parts, a Rust part (folder [`crates/coach`](../../crates/coach/src/bin/)) and a Python part (folder [`training`](../../training/src/)).

No TD-training or reinforcement learning is done. Instead, we completely rely on rollouts (done in Rust) and subsequent supervised learning (Python/PyTorch).

## Overview

The training process consists of three steps, which are repeated in a loop:
1. Find lots of positions (at least 100,000) through self-play for a later rollout.
2. Roll out these positions. Currently, only 1-ply rollouts are possible, 1296 games per position.
3. Train neural networks based on the rollout data. There are two different networks for `contact` and `race` positions. This third step is the only one done in Python.

This way we generate several generations of neural nets, each generation serving as a stepping stone for the next one.

The following sections describe how to generate new training and how to train new nets.

As the source code for all this is not meant to be run "during production", its code quality is not as high as that of `engine` and there are less unit tests.
Using it requires manual editing of source code, which is described in the next sections.

## HowTo`generate-training-data`

This sections describes the creation of new training data.

- The networks committed in [`neural-nets`](../../neural-nets) are small weak nets. Replace them with the latest nets from https://github.com/carsten-wenderdel/wildbg-training.
- Edit [`generate-training-data.rs`](../../crates/coach/src/bin/generate-training-data.rs) and chose the desired game **phase** (`contact` or `race`) and
the **amount** of positions for rollout.
- Execute `cargo run -r -p coach --bin generate-training-data`. This will take many hours.

##  HowTo`training`

This section describes the creation of new nets via supervised learning.

- Decide whether you want to create a new `contact` or `race` network. Let's say, we choose `contact`.
- Download the latest training data from https://github.com/carsten-wenderdel/wildbg-training.
Look for a file `contact.csv` in the `data` folder. It also might make sense to download multiple `contact.csv` files, the Python code can
deal with one or several files. These files contain position IDs in the GnuBG format along with game outcome probabilities.
- Store those files in the `training-data` folder.
- Edit the file [`convert-to-inputs.rs`](../../crates/coach/src/bin/convert-to-inputs.rs) and make sure that the filenames are correct.
- Run `cargo run -p coach --bin convert-to-inputs`.
This reads the downloaded CSV file and creates a new CSV file with inputs and outputs for PyTorch.
If you want to try different inputs, you have to program that in Rust ([inputs.rs](../../crates/engine/src/inputs.rs)).
- Edit the file [`train-on-rollout-data.py`](../../training/src/train-on-rollout-data.py). Make sure the correct model is
defined, it should be something like `mode = "contact"`.
- You might want to edit various hyperparameters. Number of epochs, optimizer and loss function should be ok, but maybe you find better ones.
In any case you should try various learning rates, they have a big impact on the quality of the net.
- Go to the folder `training` and execute `./src/train-on-rollout-data.py` - this will create several new nets in the `training-data` folder. It should take only a few minutes.

### Compare neural nets
Before deciding which new neural net is the best, you should compare it to the current best net. This is done by letting two evaluators play against each other.
To have a baseline, copy existing onnx files from https://github.com/carsten-wenderdel/wildbg-training to the folder `neural-networks`. Those committed to this repository are small and weaker.
- Check the quality of the net: 

#### Compare just two evaluators
- Edit [`compare-evaluators.rs`](../../crates/coach/src/bin/compare-evaluators.rs) and pick different nets you want to compare.
- Execute `cargo run -r -p coach --bin compare-evaluators`. This starts two evaluators with different nets playing against each other.
After several ten thousand games a difference in equity should be visible. This helps to pick the strongest net.

#### Compare all neural nets in the `training-data` folder
- Edit [`benchmark-evaluators.rs`](../../crates/coach/src/bin/compare-evaluators.rs) and and pick the number of games that should be played per comparison. Even with 300,000 games the results can easily fluctuate by 0.04 equity points.
- Execute `cargo run -r -p coach --bin benchmark-evaluators`. After having results, you might want to repeat this with less onnx files in the `training-data` folder and more games.
