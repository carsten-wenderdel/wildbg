# Crate `wildbg-c`

This crate contains a small C library to access some functionality of `wildbg`.

In contrast to the [`web`](../web/) API this library contains less features and needs more manual work to set up. Only use it if you have existing C code and no other way to connect that to `wildbg`.

You can see the API in the header file: [`crates/wildbg-c/wildbgh.h`](../../crates/wildbg-c/wildbg.h).

### How to use this library from your C code

Execute the following from the project's root folder:

1. Build the library:
```
cargo build --package wildbg-c --lib --release
```

2. Copy the library to your C project: (the file might be called `wildbg.dll` or `wildbg.so` depending on your operating system)
```
cp target/release/libwildbg.a $YOUR_C_PROJECT_FOLDER
```

3. Copy the library header (Replace `$YOUR_C_PROJECT_FOLDER` with the correct path):
```
cp crates/wildbg-c/wildbg.h $YOUR_C_PROJECT_FOLDER
```

4. Copy the neural nets:
```
cp -r neural-nets $YOUR_C_PROJECT_FOLDER
```

5. Include the header file in your C file and use `wildbg` from there. Example:
```c
#include <stdio.h>
#include "wildbg.h"

int main() {
  // Initialize engine:
  Wildbg *wildbg = wildbg_new();
  
  // Specify 1 pointer:
  BgConfig config = { .x_away = 1, .o_away = 1 };
  
  // Define position and find best move:
  int starting[] = {0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0,};
  CMove move = best_move(wildbg, &starting, 3, 1, &config); 
  printf("The computer would move from %d to %d and from %d to %d.\n", move.detail1.from, move.detail1.to, move.detail2.from, move.detail2.to);
  
  // Deconstruct the engine and free the memory:
  wildbg_free(wildbg);  
}
```

6. Don't forget linking the library file into your binary. Example:
```
cd $YOUR_C_PROJECT_FOLDER
gcc main.c libwildbg.a
```
