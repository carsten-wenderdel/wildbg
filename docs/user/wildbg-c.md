# Crate `wildbg-c`

This crate contains a small C library to access some functionality of `wildbg`.

In contrast to the [`web`](../../crates/web/src/) API this library offers fewer features and requires more manual work to set up. Only use it if you have existing C code and no other way to connect that to `wildbg`.

You can see the API in the header file: [`crates/wildbg-c/wildbgh.h`](../../crates/wildbg-c/wildbg.h).

### How to use this library from your C code

Execute the following from the project's root folder.

#### 1. Build the library
```
cargo build --package wildbg-c --lib --release
```
#### 2. Copy the library to your C project
The shared library file created may be called `wildbg.dll` or `wildbg.so`, depending on your operating system.
```shell
cp target/release/libwildbg.a $YOUR_C_PROJECT_FOLDER
```
#### 3. Copy the library header
Replace `$YOUR_C_PROJECT_FOLDER` with the correct path):
```shell
cp crates/wildbg-c/wildbg.h $YOUR_C_PROJECT_FOLDER
```
#### 4. Copy the neural nets
```shell
cp -r neural-nets $YOUR_C_PROJECT_FOLDER
```
#### 5. Include the header file in your C file and use `wildbg` from there.
```c
#include <stdio.h>
#include "wildbg.h"

int main() {
  // Initialize engine:
  Wildbg *wildbg = wildbg_new();
  
  // Define position and print winning probability:
  int starting[] = {0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,};
  CProbabilities p = probabilities(wildbg, &starting);
  printf("The estimated probability to win is %.2f percent.\n", 100 * p.win);
  
  // Specify 1 pointer game type:
  BgConfig config = { .x_away = 1, .o_away = 1 };
  
  // Find and print best move:
  CMove move = best_move(wildbg, &starting, 3, 1, &config);
  printf("The computer would make the following moves:\n")
  for (int i = 0; i < move.detail_count; i ++){
    printf("\tfrom %d to %d.\n", move.details[i].from, move.details[i].to);
  }
  
  // Deconstruct the engine and free the memory:
  wildbg_free(wildbg);  
}
```
#### 6. Link the library into your binary
```shell
cd $YOUR_C_PROJECT_FOLDER
gcc main.c libwildbg.a
```