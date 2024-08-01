#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Wildbg Wildbg;

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
 * When no move is possible, detail_count will be 0.
 *
 * If only a single move is possible, `details[0]` will contain this information.
 * `detail_count` will contain a value between 0 and 4.
 *
 * If the same checker is moved twice, this is encoded in two details.
 */
typedef struct CMove {
  struct CMoveDetail details[4];
  int detail_count;
} CMove;

/**
 * Configuration needed for the evaluation of positions.
 *
 * Currently only 1 pointers and money game are supported.
 * In the future `BgConfig` can also include information about Crawford, cube possession, strength of the engine and so on.
 */
typedef struct BgConfig {
  /**
   * Number of points the player on turn needs to finish the match. Zero indicates money game.
   */
  unsigned int x_away;
  /**
   * Number of points the opponent needs to finish the match. Zero indicates money game.
   */
  unsigned int o_away;
} BgConfig;

typedef struct CProbabilities {
  /**
   * Cubeless probability to win the game. This includes gammons and backgammons.
   */
  float win;
  /**
   * Probability to win gammon or backgammon.
   */
  float win_g;
  /**
   * Probability to win backgammon.
   */
  float win_bg;
  /**
   * Probability to lose gammon or backgammon.
   */
  float lose_g;
  /**
   * Probability to lose backgammon.
   */
  float lose_bg;
} CProbabilities;

/**
 * Loads the neural nets into memory and returns a pointer to the API.
 * Returns `NULL` if the neural nets cannot be found.
 *
 * To free the memory after usage, call `wildbg_free`.
 */
struct Wildbg *wildbg_new(void);

/**
 * # Safety
 *
 * Frees the memory of the argument.
 * Don't call it with a NULL pointer. Don't call it more than once for the same `Wildbg` pointer.
 */
void wildbg_free(struct Wildbg *ptr);

/**
 * Returns the best move for the given position.
 *
 * The player on turn always moves from pip 24 to pip 1.
 * The array `pips` contains the player's bar in index 25, the opponent's bar in index 0.
 * Checkers of the player on turn are encoded with positive integers, the opponent's checkers with negative integers.
 */
struct CMove best_move(const struct Wildbg *wildbg,
                       const int (*pips)[26],
                       unsigned int die1,
                       unsigned int die2,
                       const struct BgConfig *config);

/**
 * Returns cubeless money game probabilities for a certain position.
 * If an illegal position is encountered, all probabilities will be zero.
 *
 * The player on turn always moves from pip 24 to pip 1.
 * The array `pips` contains the player's bar in index 25, the opponent's bar in index 0.
 * Checkers of the player on turn are encoded with positive integers, the opponent's checkers with negative integers.
 */
struct CProbabilities probabilities(const struct Wildbg *wildbg,
                                    const int (*pips)[26]);
