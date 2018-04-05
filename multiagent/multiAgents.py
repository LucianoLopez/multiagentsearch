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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        pacPosition = successorGameState.getPacmanPosition()
        foodList = currentGameState.getFood().asList()

        if action == 'Stop':
            return -float("inf")
        for ghost in newGhostStates:
            if samePosition(ghost.getPosition(), tuple(pacPosition)) and ghost.scaredTimer == 0:
                return -float("inf")

        foodDists = [manhattanDistance(food, pacPosition) for food in foodList]
        return max(foodDists)

def samePosition(position1, position2):
    """
    Returns whether position1 and position2 are the same
    """
    return position1 == position2

def manhattanDistance(position1, position2):
        xPos = -1 * abs(position1[0] - position2[0])
        yPos = -1 * abs(position1[1] - position2[1])
        return xPos + yPos

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        def evaluator(gameState):
            return self.evaluationFunction(gameState)

        def value(gameState, agentIndex, depth):
            if agentIndex % gameState.getNumAgents() == 0 and agentIndex != 0:
                agentIndex = 0
                depth += 1
            if depth == self.depth:
                return evaluator(gameState)
            if agentIndex is not 0:
                return min_value(gameState, agentIndex, depth)
            else:
                return max_value(gameState, agentIndex, depth)

        def min_value(gameState, agentIndex, depth):
            v = (float("inf"), "Unknown")
            actionList = gameState.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            if gameState.isWin() or gameState.isLose():
                return evaluator(gameState)
            for action in actionList:
                successorState = gameState.generateSuccessor(agentIndex, action)
                evaluation = value(successorState, nextAgent, depth)
                if type(evaluation) is tuple:
                    evaluation = evaluation[0]
                vNew = min(v[0], evaluation)
                if vNew is not v[0]:
                    v = (vNew, action)
                    # v = min(v, min_value(successorState, nextAgent, depth, desiredDepth))
            return v

        def max_value(gameState, agentIndex, depth):
            v = (-float("inf"), "Unknown")
            actionList = gameState.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            if gameState.isWin() or gameState.isLose():
                return evaluator(gameState)
            for action in actionList:
                successorState = gameState.generateSuccessor(agentIndex, action)
                evaluation = value(successorState, nextAgent, depth)
                if type(evaluation) is tuple:
                    evaluation = evaluation[0]
                vNew = max(v[0], evaluation)
                if vNew is not v[0]:
                    v = (vNew, action)
                    # v = max(v, min_value(successorState, nextAgent, depth, desiredDepth))
            return v
        return value(gameState, 0, 0)[1]




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def evaluator(gameState):
            return self.evaluationFunction(gameState)

        def value(gameState, agentIndex, depth, alpha, beta):
            if agentIndex % gameState.getNumAgents() == 0 and agentIndex != 0:
                agentIndex = 0
                depth += 1
            if depth == self.depth:
                return evaluator(gameState)
            if agentIndex is not 0:
                return min_value(gameState, agentIndex, depth, alpha, beta)
            else:
                return max_value(gameState, agentIndex, depth, alpha, beta)

        def min_value(gameState, agentIndex, depth, alpha, beta):
            v = (float("inf"), "Move")
            actionList = gameState.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            if gameState.isWin() or gameState.isLose():
                return evaluator(gameState)
            for action in actionList:
                successorState = gameState.generateSuccessor(agentIndex, action)
                evaluation = value(successorState, nextAgent, depth, alpha, beta)
                if type(evaluation) is tuple: 
                    evaluation = evaluation[0]
                vNew = min(v[0], evaluation)
                if vNew is not v[0]:
                    v = (vNew, action)
                if v[0] < alpha:
                    return v
                beta = min(beta, v[0])
                    # v = min(v, min_value(successorState, nextAgent, depth, desiredDepth))
            return v

        def max_value(gameState, agentIndex, depth, alpha, beta):
            v = (-float("inf"), "Move")
            actionList = gameState.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            if gameState.isWin() or gameState.isLose():
                return evaluator(gameState)
            for action in actionList:
                successorState = gameState.generateSuccessor(agentIndex, action)
                evaluation = value(successorState, nextAgent, depth, alpha, beta)
                if type(evaluation) is tuple:
                    evaluation = evaluation[0]
                vNew = max(v[0], evaluation)
                if vNew is not v[0]:
                    v = (vNew, action)
                if v[0] > beta:
                    return v
                alpha = max(alpha, v[0])
                    # v = max(v, min_value(successorState, nextAgent, depth, desiredDepth))
            return v
        alpha = -float("inf")
        beta = float("inf")
        return value(gameState, 0, 0, alpha, beta)[1]
        # def evaluator(gameState):
        #     return self.evaluationFunction(gameState)
        #
        # def min_value(gameState, agentIndex, depth, desiredDepth, alpha, beta):
        #     v = (float("inf"), "Unknown")
        #     actionList = gameState.getLegalActions(agentIndex)
        #     numAgents = gameState.getNumAgents()
        #     nextAgent = agentIndex + 1
        #     if nextAgent % numAgents == 0:
        #         nextAgent = 0
        #     for action in actionList:
        #         successorState = gameState.generateSuccessor(agentIndex, action)
        #         if successorState.isWin() or successorState.isLose() or (nextAgent == 0 and desiredDepth == depth):
        #             evaluation = evaluator(successorState)
        #             vNew = min(v[0], evaluation)
        #             if vNew is not v[0]:
        #                 v = (vNew, action)
        #         elif nextAgent == 0:
        #             evaluation = max_value(successorState, nextAgent, depth + 1, desiredDepth, alpha, beta)
        #             if type(evaluation) is tuple:
        #                 evaluation = evaluation[0]
        #             vNew = min(v[0], evaluation)
        #             if vNew is not v[0]:
        #                 v = (vNew, action)
        #             if agentIndex - 1 == 0:
        #                 if v[0] > beta:
        #                     return v
        #                 alpha = max(alpha, v[0])
        #             else:
        #                 if v[0] < alpha:
        #                     return v
        #                 beta = min(beta, v[0])
        #             # if v < alpha:
        #             #     return v
        #             # beta = min(beta, v[0])
        #         else:
        #             evaluation = min_value(successorState, nextAgent, depth, desiredDepth, alpha, beta)
        #             if type(evaluation) is tuple:
        #                 evaluation = evaluation[0]
        #             vNew = min(v[0], evaluation)
        #             if vNew is not v[0]:
        #                 v = (vNew, action)
        #             if agentIndex - 1 == 0:
        #                 if v[0] > beta:
        #                     return v
        #                 alpha = max(alpha, v[0])
        #             else:
        #                 if v[0] < alpha:
        #                     return v
        #                 beta = min(beta, v[0])
        #     return v
        #
        # def max_value(gameState, agentIndex, depth, desiredDepth, alpha, beta):
        #     v = (-float("inf"), "Unknown")
        #     actionList = gameState.getLegalActions(agentIndex)
        #     numAgents = gameState.getNumAgents()
        #     nextAgent = agentIndex + 1
        #     if nextAgent % numAgents == 0:
        #         nextAgent = 0
        #     for action in actionList:
        #         successorState = gameState.generateSuccessor(agentIndex, action)
        #         if successorState.isWin() or successorState.isLose():
        #             evaluation = evaluator(successorState, agentIndex)
        #             vNew = max(v[0], evaluation)
        #             if vNew is not v[0]:
        #                 v = (vNew, action)
        #         else:
        #             evaluation = min_value(successorState, nextAgent, depth, desiredDepth, alpha, beta)
        #             if type(evaluation) is tuple:
        #                 evaluation = evaluation[0]
        #             vNew = max(v[0], evaluation)
        #             if vNew is not v[0]:
        #                 v = (vNew, action)
        #             if v[0] < alpha:
        #                 return v
        #             beta = min(beta, v[0])
        #     return v
        #
        # agentIndex = 0
        # alpha = -float("inf")
        # beta = float("inf")
        # if gameState.isWin() or gameState.isLose():
        #     return evaluator(gameState, agentIndex)
        # if agentIndex == 0:
        #     agentIndex += 1
        #     return max_value(gameState, 0, 1, self.depth, alpha, beta)[1]
        # else:
        #     agent = agentIndex
        #     agentIndex += 1
        #     return min_value(gameState, agent, 1, self.depth, alpha, beta)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def evaluator(gameState):
            return self.evaluationFunction(gameState)
        def value(gameState, agentIndex, depth):
            if agentIndex % gameState.getNumAgents() == 0 and agentIndex != 0:
                agentIndex = 0
                depth += 1
            if depth == self.depth:
                return evaluator(gameState)
            if agentIndex != 0:
                return exp_value(gameState, agentIndex, depth)
            else:
                return max_value(gameState, agentIndex, depth)
        def exp_value(gameState, agentIndex, depth):
            v = (0, "action")
            actionList = gameState.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            if gameState.isWin() or gameState.isLose():
                return evaluator(gameState)
            length = len(actionList)
            p = 1.0 / length
            for action in actionList:
                successorState = gameState.generateSuccessor(agentIndex, action)
                evaluation = value(successorState, nextAgent, depth)
                if type(evaluation) is tuple:
                    evaluation = evaluation[0]
                vWithProb = v[0] + p * evaluation
                # print(vWithProb)
                v = (vWithProb, action)
            return v
        def max_value(gameState, agentIndex, depth):
            v = (-float("inf"), "action")
            actionList = gameState.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            if gameState.isWin() or gameState.isLose():
                return evaluator(gameState)
            for action in actionList:
                successorState = gameState.generateSuccessor(agentIndex, action)
                evaluation = value(successorState, nextAgent, depth)
                if type(evaluation) is tuple:
                    evaluation = evaluation[0]
                vNew = max(v[0], evaluation)
                if vNew is not v[0]:
                    v = (vNew, action)
            return v
        return value(gameState, 0, 0)[1]
            # def evaluator(gameState):
            #     return self.evaluationFunction(gameState)
            #
            # def value(gameState, agentIndex, depth):
            #     if agentIndex % gameState.getNumAgents() == 0 and agentIndex != 0:
            #         agentIndex = 0
            #         depth += 1
            #     if depth == self.depth:
            #         return evaluator(gameState)
            #     if agentIndex is not 0:
            #         return min_value(gameState, agentIndex, depth)
            #     else:
            #         return max_value(gameState, agentIndex, depth)
            #
            # def min_value(gameState, agentIndex, depth):
            #     v = (float("inf"), "Unknown")
            #     actionList = gameState.getLegalActions(agentIndex)
            #     nextAgent = agentIndex + 1
            #     if gameState.isWin() or gameState.isLose():
            #         return evaluator(gameState)
            #     for action in actionList:
            #         successorState = gameState.generateSuccessor(agentIndex, action)
            #         evaluation = value(successorState, nextAgent, depth)
            #         if type(evaluation) is tuple:
            #             evaluation = evaluation[0]
            #         vNew = min(v[0], evaluation)
            #         if vNew is not v[0]:
            #             v = (vNew, action)
            #             # v = min(v, min_value(successorState, nextAgent, depth, desiredDepth))
            #     return v
            #
            # def max_value(gameState, agentIndex, depth):
            #     v = (-float("inf"), "Unknown")
            #     actionList = gameState.getLegalActions(agentIndex)
            #     nextAgent = agentIndex + 1
            #     if gameState.isWin() or gameState.isLose():
            #         return evaluator(gameState)
            #     for action in actionList:
            #         successorState = gameState.generateSuccessor(agentIndex, action)
            #         evaluation = value(successorState, nextAgent, depth)
            #         if type(evaluation) is tuple:
            #             evaluation = evaluation[0]
            #         vNew = max(v[0], evaluation)
            #         if vNew is not v[0]:
            #             v = (vNew, action)
            #             # v = max(v, min_value(successorState, nextAgent, depth, desiredDepth))
            #     return v
            #
            # return value(gameState, 0, 0)[1]
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Will use a similar tactic as in #1 where I'll use the manhattan distance, but instead of actions
      we will evaluate states prioritizing food and ghosts.
    """
    "*** YOUR CODE HERE ***"
    numScaredGhosts = 0
    # foodDists = [manhattanDistance(food, pacPosition) for food in foodList]
    ghostDists = []
    foodList = currentGameState.getFood().asList()
    foodDist = []
    ghostStates = currentGameState.getGhostStates()
    currentPos = tuple(currentGameState.getPacmanPosition())
    for ghost in ghostStates:
        numScaredGhosts += 1
        ghostDists.append(manhattanDistance(ghost.getPosition(), currentPos))
    for food in foodList:
        foodDist.append(manhattanDistance(food, currentPos))
    # foodDists = [manhattanDistance(food, currentPos) for food in foodList] ## not sure why this isn't better than list comp
    if not foodDist:
        foodDist.append(0)
    ghost = max(ghostDists)
    if ghost != 0:
        ghost = 1.0/ghost
    return max(foodDist) + ghost + currentGameState.getScore()

def samePosition(position1, position2):
        """
        Returns whether position1 and position2 are the same
        """
        return position1 == position2

def manhattanDistance(position1, position2):
        xPos = -1 * abs(position1[0] - position2[0])
        yPos = -1 * abs(position1[1] - position2[1])
        return xPos + yPos


# Abbreviation
better = betterEvaluationFunction

