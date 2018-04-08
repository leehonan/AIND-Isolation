# only modules allowed are random, numpy, scipy, sklearn, itertools, math, heapq, collections, array, copy, and operator
import numpy as np
from random import shuffle
from math import exp

# ====================================================================================================================
# CONSTANTS
# ====================================================================================================================

MAX_POSSIBLE_MOVES = 8.0
MOVE_OFFSETS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]

# ====================================================================================================================
# GLOBALS
# ====================================================================================================================

move_eval_dict = None
knights_tour_matrix = None
corners = None
corner_attacks = None
stats = None

# ====================================================================================================================
# EVALUATION DICT & BITBOARD FUNCTIONS
# ====================================================================================================================

def board_as_bitboard(game_board, board_width):
    '''
    returns board as numpy array of ints as used in Board._board_state's 1d list
    reshapes and then transposes from columnar ordering in _board_state to row ordering

    '''
    return np.transpose(np.reshape(np.array(game_board, int), (-1, board_width)))


def get_eval_dict_key(board_as_bit_board, game):
    '''
    Get key to use with the heuristic evaluation cache.

    Key format is: <player_to_move>,<board_as_int>,<p1_pos>,<p2_pos>

    '''
    player_to_move = 1 if game._active_player == game._player_1 else 2
    p1_pos = -1 if game._board_state[-1] is None else game._board_state[-1]  # None if yet to move
    p2_pos = -1 if game._board_state[-2] is None else game._board_state[-2]  # None if yet to move

    # get int value for flattened binary representation of board (little-endian)
    binary_array = board_as_bit_board.ravel()
    board_int = binary_array.dot(1 << np.arange(binary_array.size)[::-1])

    # return key, using % for speed...
    return '%s,%s,%s,%s' % (player_to_move, board_int, p1_pos, p2_pos)


def put_eval_dict_item(eval_dict, game, eval_result):
    '''
    Caches evaluation of game state.  eval_result is a dict of floats.
    Will overwrite if key exists - no checking.
    '''
    # get bitboard for game
    board_as_bit_board = board_as_bitboard(game._board_state[:-3], game.width)
    # get key and update cache
    eval_dict[get_eval_dict_key(board_as_bit_board, game)] = eval_result


def find_eval_dict_item(eval_dict, game, with_transformations=True):
    '''
    Tries to get an evaluation from key matching board state; checking rotated and flipped vertical,
    horizontal, diagonal equivalents if with_transformations is true.
    Some redundancy versus other functions for performance.

    The cache can record successful hits.

    '''
    # get bitboard for game
    bit_board = board_as_bitboard(game._board_state[:-3], game.width)

    # try and retrieve with 'regular' key
    result = eval_dict.get(get_eval_dict_key(bit_board, game))
    if result is not None:
        # eval_dict['hits'] += 1
        return result

    # no hits, if with_transformations iterate through bitboard rotate and flip
    # transformations, looking for matching key

    if with_transformations:

        #   rotate 90
        bit_board_90 = np.rot90(bit_board)
        result = eval_dict.get(get_eval_dict_key(bit_board_90, game))
        if result is not None:
            # eval_dict['hits'] += 1
            return result

        #   rotate 180
        bit_board_180 = np.rot90(bit_board_90)
        result = eval_dict.get(get_eval_dict_key(bit_board_180, game))
        if result is not None:
            # eval_dict['hits'] += 1
            return result

        #   rotate 270
        bit_board_270 = np.rot90(bit_board_180)
        result = eval_dict.get(get_eval_dict_key(bit_board_270, game))
        if result is not None:
            # eval_dict['hits'] += 1
            return result

        #   flip horizontal
        bit_board_flip_h = np.fliplr(bit_board)
        result = eval_dict.get(get_eval_dict_key(bit_board_flip_h, game))
        if result is not None:
            # eval_dict['hits'] += 1
            return result

        #   flip vertical
        bit_board_flip_v = np.flipud(bit_board)
        result = eval_dict.get(get_eval_dict_key(bit_board_flip_v, game))
        if result is not None:
            # eval_dict['hits'] += 1
            return result

        #   flip diagonal on left-top\right-bottom
        bit_board_flip_d_ltrb = np.rot90(bit_board_flip_h)
        result = eval_dict.get(get_eval_dict_key(bit_board_flip_d_ltrb, game))
        if result is not None:
            # eval_dict['hits'] += 1
            return result

        #   flip diagonal on left-bottom/right-top
        bit_board_flip_d_lbrt = np.rot90(bit_board_flip_v)
        result = eval_dict.get(get_eval_dict_key(bit_board_flip_d_lbrt, game))
        if result is not None:
            # eval_dict['hits'] += 1
            return result

    # Return None if no entry
    return None


# ====================================================================================================================
# EVALUATION FUNCTIONS
# ====================================================================================================================

def get_open_board_moves_from_pos(game, pos):
    '''
    Get possible moves from a position, ignoring whether cells are occupied

    7*7 layout is...
       cols   ->   0      1      2      3      4      5      6
       rows   ->   0123456012345601234560123456012345601234560123456
       on/off ->   0000001010100011001010110010010101001010101010100
    '''
    if pos is None:
        return game.get_blank_spaces()

    row, col = pos
    valid_moves = [(row + row_dir, col + col_dir) for row_dir, col_dir in MOVE_OFFSETS
                   if (row + row_dir in range(game.height) and col + col_dir in range(game.width))]

    return valid_moves


def get_legal_moves_from_pos(game, pos):
    '''
    Get list of possible *legal* moves from a position in the game state.  Some redundancy versus get_open_board_moves_from_pos
    for performance.

    '''
    if pos is None:
        return game.get_blank_spaces()

    row, col = pos
    legal_moves = [(row + row_dir, col + col_dir) for row_dir, col_dir in MOVE_OFFSETS
                             if (row + row_dir in range(game.height) and col + col_dir in range(game.width))
                                and game._board_state[(row + row_dir) + (col + col_dir) * game.height] == 0]
    shuffle(legal_moves)

    return legal_moves


def paths_from_pos(game, pos, at_level=0, to_level=0, max_paths=MAX_POSSIBLE_MOVES, timer_player_obj=None, time_limit=None):
    '''
    Get a capped count of unique paths to the level/depth specified.
    E.g. for an board with max_paths of 2 and to_level of 3 specified, could include:
            2 moves at depth 2
            1 move at depth 3 from expanding all of depth 2 but finding that only 1 path reached depth 3

    This would return 1 - only one path was found to depth 3.

    Implemented as a depth-first search, recursing using at_level and to_level.

    '''

    if timer_player_obj is not None and timer_player_obj.time_left() < time_limit:
        raise SearchTimeout()

    row, col = pos
    legal_moves = [(row + row_dir, col + col_dir) for row_dir, col_dir in MOVE_OFFSETS
                             if (row + row_dir in range(game.height) and col + col_dir in range(game.width))
                                and game._board_state[(row + row_dir) + (col + col_dir) * game.height] == 0]

    moves_here = len(legal_moves)

    path_count = 0

    # terminate and return if no paths
    if moves_here == 0:
        return path_count

    if at_level >= to_level:
        path_count = min(moves_here, max_paths)
        return path_count
    else:
        # recurse
        try:
            # Sort moves by move count on an empty board
            legal_moves.sort(key=lambda p_move: knights_tour_matrix[p_move], reverse=True)
            for move in legal_moves:
                # unique paths!
                m_path_count = paths_from_pos(game=game, pos=move, at_level=at_level + 1,
                                                    to_level=to_level, max_paths=max_paths,
                                                    timer_player_obj=timer_player_obj,
                                                    time_limit=time_limit)

                if m_path_count >= 1:
                    path_count += 1

                if path_count >= max_paths:
                    break

        except SearchTimeout:
            # assumedly another timeout will occur higher in the call stack when the 'real' time limit is reached
            return path_count

    return path_count


# ====================================================================================================================
# MISC. FUNCTIONS
# ====================================================================================================================

def init_globals(game):
    '''
    Initialise global vars.  Used instead of class variables to get past Udacity submission tests.
    '''
    global corners
    global corner_attacks
    global knights_tour_matrix

    corners = [(0, 0), (game.width - 1, 0), (0, game.height - 1), (game.width - 1, game.height - 1)]

    corner_attacks = [[(1, 2), (2, 1)],
                      [(1, game.width - 3), (2, game.width - 2)],
                      [(game.height - 3, 1), (game.height - 2, 2)],
                      [(game.height - 3, game.width - 2), (game.height - 2, game.width - 3)]]

    # Build knight's tour matrix if it doesn't exist (number of moves each square allows on an open board)
    knights_tour_matrix = np.zeros((game.height, game.width))
    for x in range(game.width):
        for y in range(game.height):
            knights_tour_matrix[x, y] = len(get_open_board_moves_from_pos(game, (x, y)))

    global move_eval_dict
    if move_eval_dict is None:
        move_eval_dict = {'hits': 0}
    else:
        move_eval_dict['hits'] = 0

    global stats
    stats = {'max_search_depth': 0, 'score_gen': {}}


# ====================================================================================================================
# Custom Score Generators
#
# Use a combination of evaluation functions and evaluation dictionary to implement different strategies.
#
# ====================================================================================================================

def custom_score_generator(game, player, progress, use_eval_dict=False, use_transformations=True, get_stats=False,
                           # opening_book_weight=100.0,
                           kt_weight=100.0,
                           kt_progress_measured_to=50.0, late_centrality_weight=100.0,
                           player_safe_moves_weight=100.0, player_safe_search_depth=0, player_safe_search_paths=0,
                           opp_isol_moves_weight=100.0, opp_isol_search_depth=0, opp_isol_search_paths=0):
    '''
    Parameterised scoring engine, where different strategies can be employed by passing different parameters.

    Results can be cached and retrieved for equivalent rotated/flipped boards.

    Implemented as a long function to minimise expensive function calls.  Some redundancy vs Board class for speed.

    Uses global variables because Udacity's submission unit tests thwart AlphaBetaPlayer with custom init and
    object properties.
    '''


    # get player and opponent
    player_num = 1 if player == game._player_1 else 2
    player_location = None if game._board_state[-player_num] is None \
                            else (game._board_state[-player_num] % game.height,
                                    game._board_state[-player_num] // game.width)
    opposition = game.get_opponent(player)
    opposition_num = 1 if opposition == game._player_1 else 2
    opposition_location = None if game._board_state[-opposition_num] is None \
                            else (game._board_state[-opposition_num] % game.height,
                                    game._board_state[-opposition_num] // game.width)

    # get legal moves for each player, once - faster than Board.get_legal_moves()
    player_moves = get_legal_moves_from_pos(game, player_location)
    opp_moves = get_legal_moves_from_pos(game, opposition_location)

    num_player_moves = len(player_moves)
    num_opp_moves = len(opp_moves)

    # return if player has won
    if player == game._inactive_player and num_opp_moves == 0:
        return float('inf')

    # return if player has lost
    if player == game._active_player and num_player_moves == 0:
        return float('-inf')

    # DEBUG_PRINT
    # print(game.to_string())

    # Try and get scores from cache, return weighted score if successful
    global move_eval_dict
    if use_eval_dict and move_eval_dict is not None:
        eval_dict_item = find_eval_dict_item(move_eval_dict, game, with_transformations=use_transformations)
        if eval_dict_item is not None:
            final_score = ( #(eval_dict_item['opening_book_score'] * opening_book_weight) +
                                (eval_dict_item['kt_score'] * kt_weight) +
                                (eval_dict_item['late_centrality_score'] * late_centrality_weight) +
                                (eval_dict_item['player_safe_moves_score'] * player_safe_moves_weight) +
                                (eval_dict_item['opp_isol_moves_score'] * opp_isol_moves_weight))

            # DEBUG_PRINT
            # print('            {} ({:.2f}%): P@{} OBS={} KTS={} L_CENT={} PM_SAFE={} OM_ISOL={} => {}'.format(
            #     game.move_count, progress, player_location, eval_dict_item['opening_book_score'], eval_dict_item['kt_score'], eval_dict_item['late_centrality_score'],
            #     eval_dict_item['player_safe_moves_score'], eval_dict_item['opp_isol_moves_score'], final_score
            # ))

            return final_score

    global corners
    global corner_attacks
    global knights_tour_matrix

    # Pre-calc fast position evaluators are initialised by AlphaBetaPlayer.check_reset() - should not need
    if corners is None or corner_attacks is None or knights_tour_matrix is None:
        init_globals(game)


    #   REMOVED! tournament initialises start position at random....
    # ==================================================================================================================
    # OPENING BOOK                                                                                      [~aggressive]
    #
    # Maximise our moves through the game by moving to a corner if we are the first to move (P1), in
    # line with the 'Knight's tour' problem.  If we are P2 and P1 is in a corner we attack to thwart their tour,
    # otherwise we move to a corner that P1 is not threatening.
    #
    # Achieved by setting a score based on conformance to these conditions at start of game, regardless of how far game
    # has progressed.
    #
    # ==================================================================================================================
    #
    # opening_book_score = 0.5
    #
    # if player_num == 1 and player_open in corners:
    #     # move to corner if P1
    #     opening_book_score = 1.0
    # elif player_num == 2 and game.move_count == 2:
    #     # opposition not in a corner, player in safe corner
    #     if player_open in corners and opposition_location not in corners and \
    #             opposition_location not in corner_attacks[corners.index(player_location)]:
    #         opening_book_score = 1.0
    #     # opposition in a corner, player attacking it
    #     elif opposition_location in corners and player_open in corner_attacks[corners.index(opp_open)]:
    #         opening_book_score = 1.0
    #     elif player_open in corners:
    #         # move to corner if P2 and blind to P1 opening move
    #         opening_book_score = 1.0


    # ==================================================================================================================
    # ALIGNMENT WITH KNIGHT'S TOUR, CENTRALITY                                                          [defensive]
    #
    # Measure the value of cells the player has visited in terms of alignment with a Knight's tour in the earlier
    # stages of the game.  In later stages, may prefer centrality.
    #
    # ==================================================================================================================

    # reward for reaching as many knight's tour squares as possible, including in earlier stages of game
    kt_score = 0.5
    if kt_progress_measured_to > 0.1 and progress <= kt_progress_measured_to:
        kt_score = (MAX_POSSIBLE_MOVES - knights_tour_matrix[player_location]) / (MAX_POSSIBLE_MOVES - 2)

    # reward centrality later in game, post rewarding knight's tour
    # set to neutral as may not have got to late game
    late_centrality_score = 0.5
    if kt_progress_measured_to < 99.9 and progress > kt_progress_measured_to:
        late_centrality_score = (knights_tour_matrix[player_location] - 2) / (MAX_POSSIBLE_MOVES - 2)

    # ==================================================================================================================
    # MOVE LOOKAHEAD
    #
    # Measure the following:
    #       * Safe player moves - a n-level lookahead for move availability                             [defensive]
    #       * Proportion of opposition moves that will isolate opposition (inverse of above)            [aggressive]
    #
    # ==================================================================================================================

    # get legal moves with recursive look ahead to prune moves into isolation
    # look ahead is exponential cost per root move/ply (~ 1/1 = 20uS, 1/2 = 106uS, 1/3 = 600uS).

    player_safe_moves_score = 0.0
    opp_isol_moves_score = 0.0

    if game.move_count > 2:
        TIME_BUFFER = 10
        player_safe_search_depth = max(1, player_safe_search_depth)
        if num_player_moves > 0 and player_safe_search_depth > 1:
            # explore player moves to depth 1+n (where player_moves gives exploration to n=1)
            paths_to_max_depth = 0
            for move in player_moves:
                # search result is number of paths found at depth specified, ignoring any that fall short
                m_paths = paths_from_pos(game=game, pos=move, max_paths=player_safe_search_paths, at_level=2, to_level=player_safe_search_depth,
                                                timer_player_obj=player, time_limit=player.TIMER_THRESHOLD - TIME_BUFFER)
                if m_paths > 0:
                    paths_to_max_depth += 1
                if paths_to_max_depth >= player_safe_search_paths:
                    break

            player_safe_moves_score = paths_to_max_depth * player_safe_search_depth**2

        # add score for L1, calculate
        player_safe_moves_score += float(num_player_moves)
        player_safe_moves_score = player_safe_moves_score / (MAX_POSSIBLE_MOVES + (player_safe_search_paths * player_safe_search_depth**2))

        opp_isol_search_depth = max(1, opp_isol_search_depth)
        if num_opp_moves > 0 and opp_isol_search_depth > 1:
            paths_to_max_depth = 0
            for move in opp_moves:
                # search result is number of paths found at depth specified, ignoring any that fall short
                m_paths = paths_from_pos(game=game, pos=move, max_paths=opp_isol_search_paths, at_level=2, to_level=opp_isol_search_depth,
                                                timer_player_obj=player, time_limit=player.TIMER_THRESHOLD - TIME_BUFFER)
                if m_paths > 0:
                    paths_to_max_depth += 1
                if paths_to_max_depth >= opp_isol_search_paths:
                    break

            opp_isol_moves_score = paths_to_max_depth * opp_isol_search_depth**2

        # add score for L1, calculate
        opp_isol_moves_score += float(num_opp_moves)
        opp_isol_moves_score = 1 - (opp_isol_moves_score / (MAX_POSSIBLE_MOVES + (opp_isol_search_paths * opp_isol_search_depth**2)))


    # ==================================================================================================================

    # Calculate final score, applying weightings
    final_score = (#(opening_book_score * opening_book_weight) +
                        (kt_score * kt_weight) +
                        (late_centrality_score * late_centrality_weight) +
                        (player_safe_moves_score * player_safe_moves_weight) +
                        (opp_isol_moves_score * opp_isol_moves_weight))

    if use_eval_dict and move_eval_dict is not None:
        new_eval_dict_item = {
            # 'opening_book_score': opening_book_score,
            'kt_score': kt_score,
            'late_centrality_score': late_centrality_score,
            'player_safe_moves_score': player_safe_moves_score,
            'opp_isol_moves_score': opp_isol_moves_score
        }
        put_eval_dict_item(move_eval_dict, game, new_eval_dict_item)

    # DEBUG_PRINT
    # print('            {} ({:.2f}%): P@{} OBS={} KTS={} L_CENT={} PM_SAFE={} OM_ISOL={} => {}'.format(
    #     game.move_count, progress, player_location, opening_book_score, kt_score, late_centrality_score,
    #     player_safe_moves_score, opp_isol_moves_score, final_score
    # ))

    # TODO: COMMENT OUT BEFORE SUBMITTING
    # if get_stats:
    #     global stats
    #     from time import time
    #     stats['score_gen']['P' + str(player_num) + '_' + str(int(time() * 1000)) + '_' + str(randint(1, 99))] = {
    #                 'prog': progress,
    #                 'moves': game.move_count,
    #                 'p_num': player_num,
    #                 'p_loc': player_location,
    #                 'o_loc': opposition_location,
    #                 'score': final_score,
    #                 'kt_meas_to': kt_progress_measured_to,
    #                 'kt_score': kt_score,
    #                 'kt_weight': kt_weight,
    #                 'obook_score': opening_book_score,
    #                 'obook_weight': opening_book_weight,
    #                 'cent_score': late_centrality_score,
    #                 'cent_score_weight': late_centrality_weight,
    #                 'player_safe_moves_score': player_safe_moves_score,
    #                 'player_safe_moves_weight': player_safe_moves_weight,
    #                 'opp_isol_moves_score': opp_isol_moves_score,
    #                 'opp_isol_moves_weight': opp_isol_moves_weight
    #             }

    return final_score


# DEBUG_PROFILE
# from line_profiler import LineProfiler
#
# def do_profile(follow=[]):
#     def inner(func):
#         def profiled_func(*args, **kwargs):
#             try:
#                 profiler = LineProfiler()
#                 profiler.add_function(func)
#                 for f in follow:
#                     profiler.add_function(f)
#                 profiler.enable_by_count()
#                 return func(*args, **kwargs)
#             finally:
#                 profiler.print_stats()
#         return profiled_func
#     return inner
#
# @do_profile(follow=[paths_from_pos])


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    """

    '''
    =========
    STRATEGY:
    =========
    Defensive player with shallow beam search of own moves (looking to maximise 
    opportunities to reach states with one or more moves at depth 2), with weight toward 
    centrality after 30% of the game.  Could play better with deeper beam search of own 
    position (and opponent's for defensive purposes), but this would require a longer timeout.  
    Maximising the number of own moves at depth 2 is a reasonable proxy.
    
    Cache is disabled throughout game - doesn't seem to help, lookup misses are probably stealing 
    from opportunity to deepen search.
    '''

    # Cache.  Accumulates across games.  Best used with higher search depths, longer timeouts.
    # If specified, USE_TRANSFORMATIONS will check for matches by rotating and flipping the board.
    USE_EVAL_DICT = False
    USE_TRANSFORMATIONS = False

    # Captures stats to global dict.
    GET_STATS = False

    # Get rough progress to endgame (0 to 100) based on the number of empty cells remaining.
    # Pessimistic (tends to over-estimate by about 20%), scales with board size
    progress = 100.0 * (exp(game.move_count / (game.width * game.height)) - 1.0)

    OPP_ISOL_SEARCH_PATHS =             1           # ignored if search depth < 2
    OPP_ISOL_SEARCH_DEPTH =             1
    PLAYER_SAFE_SEARCH_PATHS =          1           # ignored if search depth < 2
    PLAYER_SAFE_SEARCH_DEPTH =          2
    OPP_ISOL_MOVES_WEIGHT =             100.0
    PLAYER_SAFE_MOVES_WEIGHT =          300.0
    KT_WEIGHT =                         50.0
    LATE_CENTRALITY_WEIGHT =            80.0
    KT_PROGRESS_MEASURED_TO =           30.0


    return custom_score_generator(game, player, progress, use_eval_dict=USE_EVAL_DICT, use_transformations=USE_TRANSFORMATIONS, get_stats=GET_STATS,
                                    # opening_book_weight=OPENING_BOOK_WEIGHT,
                                    kt_weight=KT_WEIGHT, kt_progress_measured_to=KT_PROGRESS_MEASURED_TO,
                                    late_centrality_weight=LATE_CENTRALITY_WEIGHT,
                                    player_safe_moves_weight=PLAYER_SAFE_MOVES_WEIGHT,
                                    player_safe_search_depth=PLAYER_SAFE_SEARCH_DEPTH,
                                    player_safe_search_paths=PLAYER_SAFE_SEARCH_PATHS,
                                    opp_isol_moves_weight=OPP_ISOL_MOVES_WEIGHT,
                                    opp_isol_search_depth=OPP_ISOL_SEARCH_DEPTH,
                                    opp_isol_search_paths=OPP_ISOL_SEARCH_PATHS)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    '''
    =========
    STRATEGY:
    =========
    Staged player: starts aggressively, but becomes increasingly defensive.  Uses a shallow beam
    search of own or opponent's moves depending on game stage.  Ignores alignment with knight's
    tour path or centrality.

    Could play better with deeper beam search of own position (and opponent's for defensive
    purposes), but this would require a longer timeout.
    Maximising the number of moves searched at depth 2 is a reasonable proxy.

    Cache is disabled throughout game - doesn't seem to help, lookup misses are probably stealing
    from opportunity to deepen search.
    '''

    # Cache.  Accumulates across games.  Best used with higher search depths, longer timeouts.
    # If specified, USE_TRANSFORMATIONS will check for matches by rotating and flipping the board.
    USE_EVAL_DICT = False
    USE_TRANSFORMATIONS = False

    # Captures stats to global dict.
    GET_STATS = False

    # Get rough progress to endgame (0 to 100) based on the number of empty cells remaining.
    # Pessimistic (tends to over-estimate by about 20%), scales with board size
    progress = 100.0 * (exp(game.move_count / (game.width * game.height)) - 1.0)

    if progress <= 20.0:
        '''
        ==============================================================================================================
        EARLY GAME
        ...
        ==============================================================================================================
        '''

        OPP_ISOL_SEARCH_PATHS =             1           # ignored if search depth < 2
        OPP_ISOL_SEARCH_DEPTH =             1
        PLAYER_SAFE_SEARCH_PATHS =          1           # ignored if search depth < 2
        PLAYER_SAFE_SEARCH_DEPTH =          2
        OPP_ISOL_MOVES_WEIGHT =             250.0
        PLAYER_SAFE_MOVES_WEIGHT =          150.0
        KT_WEIGHT =                         0.0
        LATE_CENTRALITY_WEIGHT =            0.0
        KT_PROGRESS_MEASURED_TO =           0.0

    elif progress <= 80.0:
        '''
        ==============================================================================================================
        MID GAME
        ...
        ==============================================================================================================
        '''

        OPP_ISOL_SEARCH_PATHS =             1           # ignored if search depth < 2
        OPP_ISOL_SEARCH_DEPTH =             1
        PLAYER_SAFE_SEARCH_PATHS =          2           # ignored if search depth < 2
        PLAYER_SAFE_SEARCH_DEPTH =          2
        OPP_ISOL_MOVES_WEIGHT =             150.0
        PLAYER_SAFE_MOVES_WEIGHT =          250.0
        KT_WEIGHT =                         0.0
        LATE_CENTRALITY_WEIGHT =            0.0
        KT_PROGRESS_MEASURED_TO =           0.0

    else:
        '''
        ==============================================================================================================
        END GAME
        ...
        ==============================================================================================================
        '''
        USE_EVAL_DICT = False

        OPP_ISOL_SEARCH_PATHS =             1           # ignored if search depth < 2
        OPP_ISOL_SEARCH_DEPTH =             2
        PLAYER_SAFE_SEARCH_PATHS =          2           # ignored if search depth < 2
        PLAYER_SAFE_SEARCH_DEPTH =          2
        OPP_ISOL_MOVES_WEIGHT =             100.0
        PLAYER_SAFE_MOVES_WEIGHT =          300.0
        KT_WEIGHT =                         0.0
        LATE_CENTRALITY_WEIGHT =            0.0
        KT_PROGRESS_MEASURED_TO =           0.0




    return custom_score_generator(game, player, progress, use_eval_dict=USE_EVAL_DICT, use_transformations=USE_TRANSFORMATIONS, get_stats=GET_STATS,
                                    # opening_book_weight=OPENING_BOOK_WEIGHT,
                                    kt_weight=KT_WEIGHT, kt_progress_measured_to=KT_PROGRESS_MEASURED_TO,
                                    late_centrality_weight=LATE_CENTRALITY_WEIGHT,
                                    player_safe_moves_weight=PLAYER_SAFE_MOVES_WEIGHT,
                                    player_safe_search_depth=PLAYER_SAFE_SEARCH_DEPTH,
                                    player_safe_search_paths=PLAYER_SAFE_SEARCH_PATHS,
                                    opp_isol_moves_weight=OPP_ISOL_MOVES_WEIGHT,
                                    opp_isol_search_depth=OPP_ISOL_SEARCH_DEPTH,
                                    opp_isol_search_paths=OPP_ISOL_SEARCH_PATHS)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    
    '''
    =========
    STRATEGY:
    =========
    Aggressive player with shallow beam search of opponent's moves (looking to minimise 
    opportunities to reach states with one or more moves at depth 2), with weight toward 
    centrality after 30% of the game.  Could play better with deeper beam search of opponents 
    position (and own for defensive purposes), but this would require a longer timeout.  Minimising 
    the number of opponent moves at depth 2 is a reasonable proxy.
    
    Cache is disabled throughout game - doesn't seem to help, lookup misses are probably stealing 
    from opportunity to deepen search.
    '''

    # Cache.  Accumulates across games.  Best used with higher search depths, longer timeouts.
    # If specified, USE_TRANSFORMATIONS will check for matches by rotating and flipping the board.
    USE_EVAL_DICT = False
    USE_TRANSFORMATIONS = False

    # Captures stats to global dict.
    GET_STATS = False

    # Get rough progress to endgame (0 to 100) based on the number of empty cells remaining.
    # Pessimistic (tends to over-estimate by about 20%), scales with board size
    progress = 100.0 * (exp(game.move_count / (game.width * game.height)) - 1.0)

    OPP_ISOL_SEARCH_PATHS =             1           # ignored if search depth < 2
    OPP_ISOL_SEARCH_DEPTH =             2
    PLAYER_SAFE_SEARCH_PATHS =          1           # ignored if search depth < 2
    PLAYER_SAFE_SEARCH_DEPTH =          1
    OPP_ISOL_MOVES_WEIGHT =             300.0
    PLAYER_SAFE_MOVES_WEIGHT =          100.0
    KT_WEIGHT =                         50.0
    LATE_CENTRALITY_WEIGHT =            80.0
    KT_PROGRESS_MEASURED_TO =           30.0


    return custom_score_generator(game, player, progress, use_eval_dict=USE_EVAL_DICT, use_transformations=USE_TRANSFORMATIONS, get_stats=GET_STATS,
                                    # opening_book_weight=OPENING_BOOK_WEIGHT,
                                    kt_weight=KT_WEIGHT, kt_progress_measured_to=KT_PROGRESS_MEASURED_TO,
                                    late_centrality_weight=LATE_CENTRALITY_WEIGHT,
                                    player_safe_moves_weight=PLAYER_SAFE_MOVES_WEIGHT,
                                    player_safe_search_depth=PLAYER_SAFE_SEARCH_DEPTH,
                                    player_safe_search_paths=PLAYER_SAFE_SEARCH_PATHS,
                                    opp_isol_moves_weight=OPP_ISOL_MOVES_WEIGHT,
                                    opp_isol_search_depth=OPP_ISOL_SEARCH_DEPTH,
                                    opp_isol_search_paths=OPP_ISOL_SEARCH_PATHS)

# ====================================================================================================================
# Players
# ====================================================================================================================

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.0):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        return self.minimax(game, self.search_depth)


    def __is_terminal(self, game, depth):
        # terminal if no legal moves remain or we are at depth_limit.
        self.__check_time()
        return len(game.get_legal_moves()) == 0 or depth <= 0


    def __check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


    def __min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        self.__check_time()
        val = float('inf')

        # check if out of moves or at depth limit
        if self.__is_terminal(game, depth):
            val = self.score(game, self)
        else:
            for move in game.get_legal_moves():
                # depth decremented by 1 on each call
                val = min(val, self.__max_value(game.forecast_move(move), depth - 1))

        return val


    def __max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        self.__check_time()
        val = float('-inf')

        # check if out of moves or at depth limit
        if self.__is_terminal(game, depth):
            val = self.score(game, self)
        else:
            for move in game.get_legal_moves():
                # depth decremented by 1 on each call
                val = max(val, self.__min_value(game.forecast_move(move), depth - 1))

        return val


    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        self.__check_time()

        legal_moves = game.get_legal_moves()

        if len(legal_moves) == 0:
            return (-1, -1)
        else:
            # ensure we return something
            best_move = legal_moves[0]
            best_move_score = float('-inf')
            try:
                for move in legal_moves:
                    # search with depth limit decreasing as each ply is visited
                    move_score = self.__min_value(game.forecast_move(move), depth - 1)
                    if move_score > best_move_score:
                        best_move_score = move_score
                        best_move = move

            except SearchTimeout:
                pass

            return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def check_reset(self, game):
        '''
        Uses global variables because Udacity's submission unit tests thwart AlphaBetaPlayer with custom init
        and object properties.

        '''
        # check if new game - will not work if moves pre-initialised
        if game.get_player_location(self) is None:
            init_globals(game)


    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # check if new game, reset if it is
        self.check_reset(game)
        best_move = (-1, -1)
        depth = 1

        try:
            while True:
                # each completed iteration will be more accurate than previous, so update best move with each iteration's
                # root node (max) result
                best_move = self.alphabeta(game=game, depth=depth)

                # return best move if no result
                if best_move == (-1, -1):
                    # DEBUG_PRINT
                    # print('D={} MOVING TO=> {} from TERM\n'.format(depth, best_move))
                    return best_move

                # global stats
                # stats['max_search_depth'] = max(depth, stats['max_search_depth'])
                depth += 1


        except SearchTimeout:
            # DEBUG_PRINT
            # print('D={} MOVING TO=> {}\n\n\n'.format(stats['max_search_depth'], best_move))
            pass

        return best_move


    def __max_play(self, game, depth, alpha, beta):
        '''
        A-B max evaluation of player's moves.  Returns tuple of highest score and best move.

        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        score = float('-inf')
        player_moves = get_legal_moves_from_pos(game, game.get_player_location(game.active_player))

        # terminate if out of moves, or at depth limit
        if depth <= 0 or len(player_moves) == 0:
            score = self.score(game, self)
        else:
            for move in player_moves:
                # get result of call to a-b-min for next ply
                # DEBUG_PRINT
                # print('\n        MAX/MIN ({} of {})...'.format(i, len(player_moves)))
                score = max(score, self.__min_play(game=game.forecast_move(move), depth=depth - 1, alpha=alpha, beta=beta))

                # prune min branch if score is higher than or equal to beta
                if score >= beta:
                    # DEBUG_PRINT
                    # print('             PRUNED!')
                    return score

                # update alpha (highest in max path) to highest_score if score > alpha
                alpha = max(alpha, score)

        return score


    def __min_play(self, game, depth, alpha, beta):
        '''
        A-B min evaluation of player's moves.  Returns lowest score.
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        score = float('inf')
        player_moves = get_legal_moves_from_pos(game, game.get_player_location(game.active_player))

        # terminate if out of moves or at depth limit
        if depth <= 0 or len(player_moves) == 0:
            score = self.score(game, self)
        else:
            for move in player_moves:
                # get result of call to a-b-max for next ply
                # DEBUG_PRINT
                # print('\n        MIN/MAX ({} of {})...'.format(i, len(player_moves)))
                score = min(score, self.__max_play(game=game.forecast_move(move), depth=depth - 1, alpha=alpha, beta=beta))

                # prune max branch if score is less than or equal to alpha
                if score <= alpha:
                    # DEBUG_PRINT
                    # print('             PRUNED!')
                    return score

                # update beta (lowest in min path) to score if score < beta
                beta = min(beta, score)

        return score


    def alphabeta(self, game, depth, alpha=float('-inf'), beta=float('inf')):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.


        Implements max function, but returning the best move rather than a value

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float('-inf')

        player_moves = get_legal_moves_from_pos(game, game.get_player_location(game.active_player))

        if len(player_moves) == 0:
            # DEBUG_PRINT
            # print('D={} BEST MOVE=> {}, TERM, maxD={}'.format(depth, best_move, stats['max_search_depth']))
            return (-1, -1)
        else:
            # initialise best move so we don't return a forfeit
            best_move = player_moves[0]

        # Order move evaluation by highest number of legal moves first
        player_moves.sort(key=lambda p_move: len(get_legal_moves_from_pos(game, p_move)), reverse=True)


        for move in player_moves:
            score = self.__min_play(game=game.forecast_move(move), depth=depth - 1, alpha=alpha, beta=beta)

            # prune min branch if score is higher than or equal to beta
            if score >= beta:
                return move

            # otherwise update best score and move
            if score > best_score:
                best_move = move
                best_score = score

            # update alpha (highest in max path) to highest_score if score > alpha
            alpha = max(alpha, best_score)

        # DEBUG_PRINT
        # if best_score not in [float('inf'), float('-inf')]:
            # print('    D={} BEST MOVE=> {}, score={}, maxD={}\n'.format(depth, best_move, best_score, stats['max_search_depth']))

        return best_move

