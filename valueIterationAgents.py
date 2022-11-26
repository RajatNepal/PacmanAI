# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        #make sure we iterate the specified amount of times
        for i in range(self.iterations):
            states = self.mdp.getStates()
            counter = util.Counter()
            for state in states:
                #for each state, go through all actions and extract max Q value
                maxQ = -1 * float('inf')
                for action in self.mdp.getPossibleActions(state):
                    Qvalue = self.computeQValueFromValues(state, action)
                    if Qvalue > maxQ:
                        maxQ = Qvalue
                #store the maxQ as long as there is one, if there isnt, just store 0
                if maxQ == -1 *float('inf'):
                    counter[state] = 0
                else:
                    counter[state] = maxQ
            self.values = counter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        Qvalue = 0

        #with the transaction, get all the next states, and the prob of those next states
        # add them up and make sure to discount along the way
        for stateProbPair in transitions:
            nextState = stateProbPair[0]
            prob = stateProbPair[1]
            reward = self.mdp.getReward(state, action, nextState)
            Qvalue += prob * (reward + self.discount * self.values[nextState]) 
        return Qvalue 

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        #simply finding the action that gives the max q value and returning that action
        actions = self.mdp.getPossibleActions(state)
        bestAction = None
        maxQ = -1 * float('inf')
        for action in actions: 
           Qvalue = self.computeQValueFromValues(state, action)

           if Qvalue > maxQ:
                maxQ = Qvalue
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        #ake sure to iterate. we know which state we are on based on the modulus operator
        #since we go through each state, we can just undex the iteration, and mod by the #states to get which state we on corrently
        for i in range(self.iterations):
            index = i %len(states)
            state = states[index]

            #as long as we arent at terminal state, keep going
            if not self.mdp.isTerminal(state):
                maxQ = -float('inf')

                #for each action, find the max q value and store it
                for action in self.mdp.getPossibleActions(state):
                    Qvalue = self.computeQValueFromValues(state, action)

                    if Qvalue > maxQ:
                        maxQ = Qvalue
                self.values[state] = maxQ

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #Compute predecessors of all states.
        previous = dict()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for stateProbPair in transitions:
                        nextState = stateProbPair[0]
                        if nextState in previous.keys():
                            previous[nextState].add(state)
                        else:
                            previous[nextState] = {state}
        #Initialize an empty priority queue.
        pq = util.PriorityQueue()


        #For each non-terminal state:
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                maxQ = -1 * float('inf')
                #Find the absolute value of the difference between the current value of s in self.values 
                # and the highest Q-value across all possible actions from s 
                # (this represents what the value should be); call this number diff. 
                # Do NOT update self.values[s] in this step

                for action in self.mdp.getPossibleActions(state):
                    Qvalue = self.computeQValueFromValues(state, action)
                    if Qvalue > maxQ:
                        maxQ = Qvalue
                diff = abs(maxQ - self.values[state])

                #Push s into the priority queue with priority -diff (note that this is negative). 
                #We use a negative because the priority queue is a min heap, 
                # but we want to prioritize updating states that have a higher error.
                pq.push(state, -diff)
                
        #For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):
            #If the priority queue is empty, then terminate.
            if pq.isEmpty():
                break

            #pop and update
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                maxQ = -1 * float('inf')

                for action in self.mdp.getPossibleActions(state):
                    Qvalue = self.computeQValueFromValues(state, action)
                    if Qvalue > maxQ:
                        maxQ = Qvalue
                self.values[state] = maxQ

            #For each predecessor p of s, do
            for p in previous[state]:
                maxQ = -1 * float('inf')
                #Find the absolute value of the difference between the current value of p in self.values 
                #and the highest Q-value across all possible actions from p 
                #(this represents what the value should be); call this number diff. 
                #Do NOT update self.values[p] in this step.
                for action in self.mdp.getPossibleActions(p):
                    Qvalue = self.computeQValueFromValues(p, action)
                    if Qvalue > maxQ:
                        maxQ = Qvalue
                diff = abs(maxQ - self.values[p])
                
                if diff > self.theta:
                    pq.update(p, -diff)