#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * If the move is not possible, both `from` and `to` will contain `-1`.
 *
 * If the move is possible, `from` is an integer between 25 and 1,
 * `to` is an integer between 24 and 0.
 * `from - to` is then at least 1 and at most 6.
 */
typedef struct CMoveDetail {
  int from;
  int to;
} CMoveDetail;

/**
 * When no move is possible, all member variables in all details will be `-1`.
 *
 * If only one checker can be moved once, `detail1` will contain this information,
 * `detail2`, `detail3` and `detail4` will contain `-1` for both `from` and `to`.
 *
 * If the same checker is moved twice, this is encoded in two details.
 */
typedef struct CMove {
  struct CMoveDetail detail1;
  struct CMoveDetail detail2;
  struct CMoveDetail detail3;
  struct CMoveDetail detail4;
} CMove;

/**
 * Returns the best move for the given position.
 *
 * The player on turn always moves from pip 24 to pip 1.
 * The array `pips` contains the player's bar in index 25, the opponent's bar in index 0.
 * Checkers of the player on turn are encoded with positive integers, the opponent's checkers with negative integers.
 */
struct CMove best_move_1ptr(const int (*pips)[26],
                            unsigned int die1,
                            unsigned int die2);
