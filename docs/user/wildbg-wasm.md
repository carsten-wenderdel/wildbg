# Crate `wildbg-wasm`

This crate contains WebAssembly (Wasm) bindings to run `wildbg` entirely client-side in web browsers or JavaScript/TypeScript runtimes.

In contrast to the [`web`](../../crates/web/src/) HTTP API, the Wasm engine executes locally with zero server requests, using the pure-Rust Tract neural network evaluator and embedded neural network weights.

---

## Building the WebAssembly Package

### Prerequisites

- [Rust](https://www.rust-lang.org/) (edition 2024)
- WebAssembly target:

  ```shell
  rustup target add wasm32-unknown-unknown
  ```

- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

  ```shell
  cargo install wasm-pack
  ```

### Build Command

From the root of the repository:

```shell
wasm-pack build crates/wildbg-wasm --target web --release --out-dir ../../pkg
```

This generates:

- `pkg/wildbg_wasm.js`: JavaScript glue code and loader
- `pkg/wildbg_wasm_bg.wasm`: Compiled WebAssembly binary with embedded neural network models
- `pkg/wildbg_wasm.d.ts`: TypeScript type definitions

---

## JavaScript / TypeScript Usage

### 1. Initialization

```javascript
import init, { Wildbg, starting_position } from './pkg/wildbg_wasm.js';

// Initialize the Wasm module
await init();

// Instantiate the engine
const engine = new Wildbg();

// Get the starting 26-point board position
const board = starting_position();
```

### 2. Position Array Convention

Positions are passed as a 26-element array (e.g. `Int8Array` or `number[]`):

- Points `1..24`: Board points from the active player's perspective (the player whose turn it is to roll or move).
- Index `25`: Active player's bar.
- Index `0`: Opponent's bar.
- Positive numbers: Checkers belonging to the active player.
- Negative numbers: Checkers belonging to the opponent.

> **Note on `AnalyzedMove.position`**: In returned move candidates, the resulting position array is also formatted from the perspective of the player who just made the move (positive checkers = player who moved). Before the opponent takes their turn, the position's perspective should be flipped (i.e. reversed and negated, or via `sides_switched()`).

### 3. Move Analysis

Analyze all legal moves for a given roll:

```javascript
// analyze(pips, die1, die2, isOnePointer)
const analysis = engine.analyze(board, 3, 1, false);

console.log('Phase:', analysis.phase); // "contact" | "race" | "game-over"
console.log('Best move:', analysis.moves[0]);
// moves[0].play: Array of { from: number, to: number }
// moves[0].position: Resulting 26-element position from moving player's view
// moves[0].equity: Cubeless equity
// moves[0].score: Ranking score
// moves[0].probabilities: { win, win_gammon, win_backgammon, lose_gammon, lose_backgammon }
```

### 4. Position Evaluation & Cube Decisions

```javascript
// Evaluate a single position
const evaluation = engine.evaluate(board);
console.log('Win probability:', evaluation.probabilities.win);

// Doubling cube decision (Janowski formulas)
const cube = engine.cube_info(board);
console.log('Should double:', cube.should_double);
console.log('Should take:', cube.should_take);
console.log('No-double equity:', cube.equity_no_double);
console.log('Double/take equity:', cube.equity_double_take);

// Check if game is completed
const status = engine.result(board); // "ongoing" | "win" | "win-gammon" | "loss" | ...
```

---

## Performance Optimizations

### Web Workers

Because multi-ply evaluation analyzes candidate responses across all 21 dice rolls, it is recommended to run `Wildbg` inside a [Web Worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API) so long-running evaluations do not block the browser's UI thread.

### WebAssembly SIMD (128-bit)

`tract-linalg` includes optimized WebAssembly SIMD kernels that can be enabled by setting the `simd128` target feature during the build. This provides a 2×–4× speedup on all modern browsers supporting Wasm SIMD:

```shell
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build crates/wildbg-wasm --target web --release --out-dir ../../pkg
```
