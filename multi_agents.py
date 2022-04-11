import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        score = successor_game_state.score
        num_of_zeros = len(np.where(board == 0)[0])
        board_mean = np.mean(board[board != 0])
        # adding constant to avoid negative numbers:
        return 10000 + score - 100 * (action==Action.UP) + 3 * board_mean + 10 * num_of_zeros


def dist_to_corner(pos_x, pos_y):
    min_distance_to_lt = pos_x + pos_y
    min_distance_to_lb = 3 - pos_x + pos_y
    min_distance_to_rt = 3 - pos_y + pos_x
    min_distance_to_rb = 6 - pos_x - pos_y
    return min(min_distance_to_lt, min_distance_to_lb, min_distance_to_rt, min_distance_to_rb)


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class Node:
    def __init__(self, value, state, depth=0, parent=None, successors=None):
        self.value = value
        self.state = state
        self.depth = depth
        self.parent = parent
        self.successors = successors if successors is not None else []


def get_value(state, depth, evaluation_function, alpha=None, beta=None, player=0):  # player = 0 --> want to maiximize. player = 1 --> want to minimize
    if depth == 0:
        return evaluation_function(state), Action.STOP, state
    best_val = -1 if player == 0 else float('inf')
    best_action = None
    best_state = None
    actions = state.get_legal_actions(player)
    if len(actions) == 0:
        return evaluation_function(state), Action.STOP, state
    for action in actions:
        successor = state.generate_successor(player, action)
        new_depth = depth - player  # if player==1 then we decrease depth
        s_val, s_action, s_state = get_value(successor, new_depth, evaluation_function, alpha, beta, 1 - player)
        if player == 0:
            if alpha is not None:
                if s_val >= beta:
                    return float('inf'), Action.STOP, state
                if s_val > alpha:
                    alpha = s_val
            update_condition = s_val > best_val
        else:
            if beta is not None:
                if s_val <= alpha:
                    return -1, Action.STOP, state
                if s_val < beta:
                    beta = s_val
            update_condition = s_val < best_val
        if update_condition:
            best_val = s_val
            best_action = action
            best_state = s_state
    return best_val, best_action, best_state

#
# def get_value(state, depth, evaluation_function, player=0):  # player = 0 --> want to maiximize. player = 1 --> want to minimize
#     if depth == 0:
#         return evaluation_function(state), Action.STOP, state
#     best_val = -1 if player == 0 else float('inf')
#     best_action = None
#     best_state = None
#     actions = state.get_legal_actions(player)
#     if len(actions) == 0:
#         return evaluation_function(state), Action.STOP, state
#     for action in actions:
#         successor = state.generate_successor(player, action)
#         new_depth = depth - player  # if player==1 then we decrease depth
#         s_val, s_action, s_state = get_value(successor, new_depth, evaluation_function, 1 - player)
#         condition = s_val > best_val if player == 0 else s_val < best_val
#         if condition:
#             best_val = s_val
#             best_action = action
#             best_state = s_state
#     return best_val, best_action, best_state


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        val, action, state = get_value(game_state, self.depth, self.evaluation_function)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        val, action, state = get_value(game_state, self.depth, self.evaluation_function, alpha=-1, beta=float('inf'))
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        val, action, state = self.get_value(game_state, self.depth)
        return action

    def get_value(self, state, depth, player=0):
        if depth == 0:
            return self.evaluation_function(state), Action.STOP, state
        best_val = -1 if player == 0 else float('inf')
        best_action = None
        best_state = None
        sum_vals_player1 = 0
        actions = state.get_legal_actions(player)
        if len(actions) == 0:
            return self.evaluation_function(state), Action.STOP, state
        for action in actions:
            successor = state.generate_successor(player, action)
            new_depth = depth - player  # if player==1 then we decrease depth
            s_val, s_action, s_state = self.get_value(successor, new_depth, 1 - player)
            if player == 0:
                if s_val > best_val:
                    best_val = s_val
                    best_action = action
                    best_state = s_state
            else: #player == 1
                sum_vals_player1 += s_val
        return_val = best_val if player == 0 else (sum_vals_player1 / len(actions))
        return return_val, best_action, best_state


def row_diff(state):
    row_diffs = state[3, :]

def large_num_in_row(row):
    good_nums = [0, 2, 4]
    return int(row[0] not in good_nums) + int(row[1] not in good_nums) + int(row[2] not in good_nums) + int(row[3] not in good_nums)


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did> TODO
    """
    # Useful information you can extract from a GameState (game_state.py)

    # successor_game_state = current_game_state.generate_successor(action=action)
    board = current_game_state.board
    max_tile = current_game_state.max_tile
    score = current_game_state.score
    num_of_zeros = len(np.where(board == 0)[0])
    board_mean = np.mean(board[board != 0])
    is_terminal = 0
    if len(current_game_state.get_legal_actions(0)) == 0 or len(current_game_state.get_legal_actions(1)) == 0:
        is_terminal = 1
    # adding constant to avoid negative numbers:
    is_large_number_down = large_num_in_row(current_game_state.board[3, :])
    # return 10000 + score - 100 * np.sum(current_game_state.board[0, :]) + 3 * board_mean + 10 * num_of_zeros

    return 100000 + score - 100 * is_large_number_down + 4 * np.sum(current_game_state.board[0, :]) + 2 * np.sum(current_game_state.board[1, :]) + np.sum(current_game_state.board[2, :])

    #* (np.sum(current_game_state.board) - current_game_state.board[0, 0])
    # - np.sum(current_game_state.board[1:3, 1:3])
    #+ 100 * num_of_zeros
    # return max(0, score + (score / 16) * num_of_zeros) - current_game_state.max_tile * is_terminal
    # return score + (score/16)*num_of_zeros
    # #
    # best_action_score = mcts(current_game_state, 0.02)
    # return best_action_score if best_action_score != None else 0


# Abbreviation
better = better_evaluation_function
