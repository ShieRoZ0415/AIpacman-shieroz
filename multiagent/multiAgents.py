# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        min_food_dist = min([manhattanDistance(newPos, food) for food in newFood.asList()] + [float('inf')])
        score += 1.0 / (min_food_dist + 1)

        for ghost in newGhostStates:
            dist = manhattanDistance(newPos, ghost.getPosition())
            if ghost.scaredTimer > 0:
                score += 10.0 / (dist + 1)
            else:
                score -= 5.0 / (dist + 1)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def value(self,state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        if agentIndex == 0:  # Pacman's turn (maximizing player)
            return max(self.value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in state.getLegalActions(agentIndex))
        else:  # Ghosts' turn (minimizing players)
            return min(self.value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in state.getLegalActions(agentIndex))

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        best_score = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.value(successor, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def value(self, state, depth, agentIndex, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        if agentIndex == 0:  # Pacman's turn (maximizing palyer)
            v = float('-inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, self.value(successor, nextDepth, nextAgent, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        else:  # Ghosts' turn (minimizing players)
            v = float('inf')
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, self.value(successor, nextDepth, nextAgent, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha, beta = float('-inf'), float('inf')
        best_score = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.value(successor, self.depth, 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def value(self,state, depth, agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        if nextAgent == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        if agentIndex == 0:  # Pacman's turn (maximizing player)
            return max(self.value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in state.getLegalActions(agentIndex))
        else:  # Ghosts' turn (minimizing players)
            actions = state.getLegalActions(agentIndex)
            values = [self.value(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) for action in actions]
            return sum(values) / len(values)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        best_score = float('-inf')
        best_action = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.value(successor, self.depth, 1)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    1. Winning states receive positive infinity, while losing states receive negative infinity。
    2. A penalty proportional to the number of remaining food pellets is applied to encourage Pacman to finish the maze.
    3. A reciprocal-distance reward is added for the nearest food pellet, encouraging Pacman to move toward reachable food.
    4. Scared ghosts provide a large reciprocal-distance reward, encouraging Pacman to chase and eat them while they are vulnerable.
    5. Active ghosts produce a reciprocal-distance penalty, encouraging Pacman to keep a safe distance from dangerous ghosts.
    6. A collision with an active ghost is treated as an immediately losing state.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    
    score -= 5 * len(foodList)

    if foodList:
        min_food_dist = min([manhattanDistance(pos, food) for food in foodList])
        score += 1.0 / (min_food_dist + 1)

    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            score += 100.0 / (dist + 1)
        else:
            if dist > 0:
                score -= 10.0 / (dist + 1)
            else:
                score = float('-inf') 

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
