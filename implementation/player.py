from typing import List
import numpy as np
from scipy.special import softmax

class Player:
    """
    A classic interactive player that will be asked which card he wants to play
    in the console
    """

    def __init__(self, name: str,n_cards: int=5) -> None:
        super().__init__() 
        self.name_: str = name
        self.n_cards: int = n_cards
        self.card_list_: List[int] = [i for i in range(1, 3*self.n_cards+1)] ##
        self.score_: int = 0

    def __repr__(self) -> str:
        return f"{self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        print("Card for this turn is:", point_card)
        while True:
            print(f"{self.name_}' cards available", self.card_list_)
            card: int = int(input(f"{self.name_}'s card: "))
            if card in self.card_list_:
                break
            else:
                print(f"{card} is not available")
        self.card_list_.remove(card)
        return card

    def get(self, point_card: int) -> None:
        self.score_ += point_card

    def dont_get(self, point_card: int) -> None:
        pass

    def no_one_get(self, point_card: int) -> None:
        pass

    def win(self, n_player) -> None:
        pass

    def lose(self, n_player) -> None:
        pass

    def end_game(self) -> None:
        self.score_ = 0
        self.card_list_ = [i for i in range(1, 3*self.n_cards+1)] ##


class NNPlayer(Player): 
    """
    Player taking its decision based on neural network outputs
    for probability of cards to be picked based on the
    card to be won and the turn.
    """

    def __init__(self, name: str,n_cards:int, weights=None, learning_rate=0.5, n_player=2) -> None:
        super().__init__(name,n_cards)
        if weights is None:
            self.A_ = np.ones((self.n_cards*3, 2))
            self.b_ = np.zeros((self.n_cards*3, 1))
        else:
            self.A_ = weights["A"]
            self.b_ = weights["b"]
        self.loss_ = np.zeros((self.n_cards*3, 1))
        self.learning_rate_ = learning_rate
        self.n_player_ = n_player
        self.t_ = 1
        self.x_list_ = np.zeros((self.n_cards*3, 2))
        self.sigma_gradient_ = np.zeros((self.n_cards*3, self.n_cards*3))

    def __repr__(self) -> str:
        return f"NN Player {self.name_}"

    def update_weights(self, lr):
        self.A_ = self.A_ + lr * self.loss_ * self.sigma_gradient_ @ self.x_list_
        self.b_ = self.b_ + lr * self.loss_ * self.sigma_gradient_.sum(axis=1).reshape(
            -1, 1
        )

    def play(self, point_card: int, turn: int = 0) -> int:
        self.turn_ = turn
        x = np.array([point_card, turn]).reshape(-1, 1)
        probabilities = softmax(self.A_ @ x + self.b_).reshape(-1)
        idx_available = np.array(self.card_list_) - 1
        probabilities_available = probabilities[idx_available]
        proba_sum = probabilities_available.sum()
        if proba_sum == 0:
            probabilities_available += 1
            proba_sum = probabilities_available.sum()
        probabilities_available /= proba_sum
        card: int = int(np.random.choice(self.card_list_, p=probabilities_available))
        self.card_list_.remove(card)
        self.x_list_[card - 1] = x.flatten()
        prod_grad = -np.ones(self.n_cards*3) * probabilities[card - 1]
        prod_grad[card - 1] += 1
        self.sigma_gradient_[card - 1] = probabilities * prod_grad
        return card

    def get(self, point_card: int) -> None:
        self.score_ += point_card
        self.loss_[point_card, 0] = point_card

    def dont_get(self, point_card: int) -> None:
        self.loss_[point_card, 0] = -point_card

    def no_one_get(self, point_card: int) -> None:
        self.loss_[point_card, 0] = -point_card / self.n_player_

    def end_game(self) -> None:
        self.update_weights(self.learning_rate_)
        self.t_ += 1
        self.x_list_ = np.zeros((self.n_cards*3, 2))
        self.sigma_gradient_ = np.zeros((self.n_cards*3, self.n_cards*3))
        return super().end_game()


class RandomPlayer(Player):
    """
    Plays random card.
    """

    def __repr__(self) -> str:
        return f"RandomPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        card: int = int(np.random.choice(self.card_list_))
        self.card_list_.remove(card)
        return card


class MaxPlayer(Player):
    """
    Plays the max card available.
    """

    def __repr__(self) -> str:
        return f"MaxPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        card = max(self.card_list_)
        self.card_list_.remove(card)
        return card


class MinPlayer(Player):
    """
    Plays the min card available.
    """

    def __repr__(self) -> str:
        return f"MinPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        card = min(self.card_list_)
        self.card_list_.remove(card)
        return card


class BonusCravingPlayer(Player):
    """
    plays with high probability a high card to gain high bonuses, and plays randomly for the rest
    """

    def __repr__(self) -> str:
        return f"BonusCravingPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card >= self.n_cards:
            card: int = int(np.random.choice(self.card_list_[-len(self.card_list_)//3:]))   #choose a random card among his best cards
        else:
            card: int = int(np.random.choice(self.card_list_))
        self.card_list_.remove(card)
        return card


class MalusAdverse(Player):
    """
    plays with high probability high cards for maluses, and plays randomly for the rest
    """
    
    def __repr__(self) -> str:
        return f"MalusAdverse {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card <= 0:
            card: int = int(np.random.choice(self.card_list_[-len(self.card_list_)//3:]))   #choose a random card among his best cards
        else:
            card: int = int(np.random.choice(self.card_list_))
        self.card_list_.remove(card)
        return card


class GetRidOfBadCards(Player):
    """
    plays with high probability low cards on small maluses or bonuses and randomly for the rest
    """

    def __repr__(self) -> str:
        return f"GetRidOfBadCards {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card >= -self.n_cards//2 and point_card <= self.n_cards:
            card: int = int(np.random.choice(self.card_list_[-len(self.card_list_)//3:]))   #choose a random card among his best cards
        else:
            card: int = int(np.random.choice(self.card_list_))
        self.card_list_.remove(card)
        return card

class MCPlayer(Player):
    """
    Implements the Monte-Carlo Reinforcement Learning method
    with epsilon-greedy approach
    """
    def __init__(self, name: str,n_cards:int, **kwargs) -> None:
        super().__init__(name,n_cards)
        self.game_result = []
        self.train(**kwargs)

    def __repr__(self) -> str:
        return f"MCPlayer {self.name_}"
    
    def train(self, eps_start=1.0, eps_decay=.99999,eps_min=0.3):
        """
        Initialize training of the MCPlayer with given epsilon parameters:
        - Set attribute is_training to True
        - Initialize epsilon trajectory
        - Initialize MC parameters
        """
        self.is_training = True

        self.episode = []

        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.eps_max = eps_start

        # Initializing the MC learning parameters

        # Listing every state of the state space
        current_card_state_space = [i for i in range(-self.n_cards,2*self.n_cards+1) if i!=0]

        start_cards = [i for i in range(1,3*self.n_cards+1)]
        hand_state_space = [[]]
        for e in start_cards:
            hand_state_space += [sub + [e] for sub in hand_state_space]
        hand_state_space = hand_state_space[1:] #drop empty hand

        # Initializing memory of the returns and count at each state
        self.returns = {
        (current_card, tuple(hand)): np.zeros(len(hand))
        for current_card in current_card_state_space for hand in hand_state_space}

        self.count = {
            (current_card, tuple(hand)): np.zeros(len(hand))
        for current_card in current_card_state_space for hand in hand_state_space}

        # Initializing Q estimate
        self.Q = {
            (current_card, tuple(hand)): np.zeros(len(hand))
        for current_card in current_card_state_space for hand in hand_state_space}

    def get_probs(self, Q_s: np.array, epsilon: float, nA: int):
        """
        Returns the epsilon-greedy weights for sampling next action

        Parameters
        ----------
        Q_s : np.array
            Current Q estimate for each action available at current state
        epsilon : float
            epsilon value for epsilon greedy search
        nA : int
            Number of actions available for current state
        """
        policy_s = np.zeros(nA) + epsilon/nA
        a_star = Q_s.argmax()
        policy_s[a_star] += 1 - epsilon 
        return policy_s

    def update_Q(self, reward, episode):
        """
        Update Q after observing a full episode and its associated reward
        """
        for state, i_action in episode:
            self.returns[state][i_action] += reward
            self.count[state][i_action] += 1
            self.Q[state][i_action] = self.returns[state][i_action]/self.count[state][i_action]
    
    def define_policy_from_Q(self, Q):
        """
        Define policy based on an estimation of Q(action|state) = Q[state][action]
        """
        policy = {state: state[1][Q_s.argmax()] for state, Q_s in Q.items()}
        return policy

    def eval(self, Q_estimate=None, policy=None):
        """
        Stop the training of the player and define the policy
        it will follow for future games based on:
        - Q_estimate if one is provided
        - policy if one is provided
        - self.Q if neither of the previous arguments are provided
        """
        self.is_training = False
        if Q_estimate is not None:
            # Define self.policy based on Q_estimate
            self.policy = self.define_policy_from_Q(Q_estimate)
        elif policy is not None:
            # Define self.policy based on policy
            self.policy = policy
        else:
            # Define self.policy based on self.Q
            self.policy = self.define_policy_from_Q(self.Q)
    
    def play(self, point_card: int, turn: int = 0) -> int:
        state = (point_card, tuple(self.card_list_))
        if self.is_training:
            # Retrives Q estimate for current state
            Q_s = self.Q[state]
            # Draw a card according to epsilon-greedy strategy
            nA = len(self.card_list_)
            card: int = int(np.random.choice(self.card_list_, p=self.get_probs(Q_s, self.epsilon, nA)))
            i_card = self.card_list_.index(card)
            self.episode.append((state, i_card))
        else:
            card = self.policy[state]

        self.card_list_.remove(card)
        return card

    def win(self, n_player) -> None:
        self.reward = 1 - 1/n_player

    def lose(self, n_player) -> None:
        self.reward = - 1/n_player
    
    def end_game(self):
        """
        Updates the Q estimation of the player, according to MC learning
        if the player is training
        """
        # Save reward
        self.game_result.append(self.reward)
        if self.is_training:
            # update Q
            self.update_Q(self.reward, self.episode)
            # Initialize episode
            self.episode = []

            self.epsilon *=self.eps_decay
                
            self.reward = 0
        # Initialize player cards
        return super().end_game()


class SARSAPlayer(MCPlayer, Player):
    """
    Inherits from MCPlayer as the only changes occur in updates of Q
    """
    def __init__(self, name: str,n_cards:int, alpha: float = 0.01, gamma: float = 0.9, **kwargs) -> None:
        super().__init__(name,n_cards)
        self.game_result = []
        self.alpha = alpha
        self.gamma = gamma
        self.reward = 0
        self.train(**kwargs)

    def __repr__(self):
        return f"SARSAPlayer {self.name_}"

    def update_Q_sarsa(self):
        """
        Q update function for SARSA training
        Updates are made only when two sequence of the episode have already been made
        as otherwise we cannot observe both current and next state
        """
        if len(self.episode) >= 2:
            state, action = self.episode[-2]
            next_state, next_action = self.episode[-1]
            self.Q[state][action] = (
                self.Q[state][action] 
                + self.alpha * 
                (self.reward + self.gamma*self.Q[next_state][next_action] - self.Q[state][action])
            )
    
    def get(self, point_card: int) -> None:
        """
        Update Q only if the state is not terminal
        Because in terminal state, one should know the reward
        before updating Q
        The final update is done during end_game method
        """
        self.score_ += point_card
        if len(self.card_list_) > 0:
            self.update_Q_sarsa()


    def dont_get(self, point_card: int) -> None:
        if len(self.card_list_) > 0:
            self.update_Q_sarsa()

    def no_one_get(self, point_card: int) -> None:
        if len(self.card_list_) > 0:
            self.update_Q_sarsa()

    def end_game(self):
        # Save reward
        self.game_result.append(self.reward)
        if self.is_training:
            # update Q
            self.update_Q_sarsa()
            # Initialize episode
            self.episode = []
            self.epsilon *= self.eps_decay
            self.reward = 0
        # Initialize player cards
        self.score_ = 0
        self.card_list_ = [i for i in range(1, 3*self.n_cards+1)]
    

class Robot(Player):
    """
    plays a given policy
    """
    def __init__(self, name: str,n_cards:int, policy=None, play_random=False) -> None:
        super().__init__(name,n_cards)
        self.policy = policy
        self.random = play_random

    def __repr__(self) -> str:
        return f"Robot {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        state = (point_card,tuple(self.card_list_))
        if self.random == False:
            chosen_card = self.policy[state]
        else:
            chosen_card = np.random.choice(self.card_list_, p=self.policy[state])
        self.card_list_.remove(chosen_card)
        return chosen_card


class ParamPlayer(Player):
    
    def __init__(self, name: str,n_cards:int,theta_shape:int,alpha:int) -> None:
        super().__init__(name,n_cards)
        self.theta_shape = theta_shape
        self.theta = np.ones(theta_shape)
        self.n_cards = n_cards
        self.alpha = alpha
        self.episode = []   
        self.game_result = []

        # Listing every state of the state space
        current_card_state_space = [i for i in range(-self.n_cards,2*self.n_cards+1) if i!=0]

        start_cards = [i for i in range(1,3*self.n_cards+1)]
        hand_state_space = [[]]
        for e in start_cards:
            hand_state_space += [sub + [e] for sub in hand_state_space]
        hand_state_space = hand_state_space[1:] #drop empty hand

        self.hand_state_space = hand_state_space

        # Initializing policy
        self.policy = {
        (current_card, tuple(hand)): np.concatenate([np.ones(1),np.zeros(len(hand)-1)])
        for current_card in current_card_state_space for hand in hand_state_space}

    def __repr__(self) -> str:
        return f"ParamPlayer {self.name_}"    

    def play(self, point_card: int, turn: int = 0) -> int:
        state = (point_card,tuple(self.card_list_))
        self.policy[state] = self.softmax(state)
        chosen_card = int(np.random.choice(self.card_list_, p=self.policy[state]))
        self.episode.append((state,  chosen_card))
        self.card_list_.remove(chosen_card)
        return chosen_card

    def win(self, n_player) -> None:
        self.reward = 1

    def lose(self, n_player) -> None:
        self.reward = - 1

    def action_state_vector(self,state,action):
        '''
        function that returns vector x(s,a) in state s, taking action a (as an array)
        '''
        action_vector = np.zeros(3*self.n_cards)
        #dummy variable with action
        action_vector[action-1] = 1

        state_vector1 = np.zeros(3*self.n_cards)
        #dummy variable with deck card
        if state[0] > 0: # there is no "0 card"
            state_vector1[state[0]+self.n_cards-1] = 1
        else:
            state_vector1[state[0]+self.n_cards] = 1
        
        state_vector2 = np.zeros(len(self.hand_state_space))
        #dummy variable with hand card
        hand = list(state[1])
        state_vector2[self.hand_state_space.index(hand)] = 1
        
        return np.concatenate([action_vector,state_vector1,state_vector2])

    def softmax(self, state):
        '''
        Compute the softmax policy.

        INPUT:
            - theta is the vector of weight parameters 
            - state is the current state of the game
    
        OUTPUT:
            - array of probabilities of the soft-max policy
    
        Careful: action_state_vector and theta must be of the same dimension
        '''
        scalar_products = np.zeros(len(self.card_list_))

        for i,act in enumerate(self.card_list_):
            scalar_products[i] = np.dot(self.theta, self.action_state_vector(state,act))

        return np.exp(scalar_products)/np.sum(np.exp(scalar_products))


    def expectation(self,state):
        '''
        Function that returns the expectation of x(state,A) under a given soft-max policy

        INPUT:
            - state is the current state of the game
            - theta is the current soft-max parameter (must be an array)
            - policy is the current set of soft-max probabilities (must be an array)
    
        OUTPUT:
            - array of probabilities of the soft-max policy
    
        Careful: action_state_vector and theta must be of the same dimension
        '''
        E = np.zeros(self.theta_shape)
        probas = self.softmax(state)
        for i_card, card in enumerate(self.card_list_):
            E += action_state_vector(state,card) * probas[i_card]
        return E


    def gradient_softmax(self, state, action):
        '''
        Compute the gradient associated to the log softmax policy 
        for a given couple (action,state)

        INPUT:
            - state is the current state of the game
            - action is the performed action
            - policy is the current set of soft-max probabilities (must be an array)
    
        OUTPUT:
            - array of probabilities of the soft-max policy
    
        Careful: action_state_vector and theta must be of the same dimension
        '''
        return self.action_state_vector(state,action) - self.expectation(state)

    def update_theta(self):
        grad_sum = 0
        for state,action in self.episode:
            grad_sum += self.alpha*self.gradient_softmax(state,action)*self.reward
        self.theta = self.theta + grad_sum


    def end_game(self):
        """
        Updates theta
        """
        # Save reward
        self.game_result.append(self.reward)

        self.update_theta()

        # Initialize episode
        self.episode = []
        self.reward = 0

        #self.epsilon *=self.eps_decay
            
        self.reward = 0
        # Initialize player cards
        return super().end_game()



































### Players for k-bandits

class BestAllocationPlayer(Player):
    """
    plays the BAS strategy
    """

    def __repr__(self) -> str:
        return f"BestAllocationPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card > self.n_cards:
            card: int = point_card + self.n_cards
        else:
            if 2*abs(point_card) in self.card_list_:
                card: int = 2*abs(point_card)
            else:
                card: int = 2*abs(point_card)-1
        self.card_list_.remove(card)
        return card


### for n_cards = 5

class BAS1Player(Player):
    """
    plays the BAS1 strategy
    """

    def __repr__(self) -> str:
        return f"BAS1Player {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card > self.n_cards:
            if point_card == 10:
                card: int = 13
            if point_card == 9:
                card: int = 15
            if point_card == 8:
                card: int = 14
            if point_card == 7:
                card: int = 12
            if point_card == 6:
                card: int = 11
        else:
            if 2*abs(point_card) in self.card_list_:
                card: int = 2*abs(point_card)
            else:
                card: int = 2*abs(point_card)-1
        self.card_list_.remove(card)
        return card


class BAS2Player(Player):
    """
    plays the BAS2 strategy
    """

    def __repr__(self) -> str:
        return f"BAS2Player {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card > self.n_cards:
            if point_card == 10:
                card: int = 14
            if point_card == 9:
                card: int = 15
            if point_card == 8:
                card: int = 13
            if point_card == 7:
                card: int = 12
            if point_card == 6:
                card: int = 11
        else:
            if 2*abs(point_card) in self.card_list_:
                card: int = 2*abs(point_card)
            else:
                card: int = 2*abs(point_card)-1
        self.card_list_.remove(card)
        return card

class BAS2bisPlayer(Player):
    """
    plays the BAS2bis strategy
    """

    def __repr__(self) -> str:
        return f"BAS2bisPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card > self.n_cards:
            if point_card == 10:
                card: int = 14
            if point_card == 9:
                card: int = 11
            if point_card == 8:
                card: int = 15
            if point_card == 7:
                card: int = 13
            if point_card == 6:
                card: int = 12
        else:
            if 2*abs(point_card) in self.card_list_:
                card: int = 2*abs(point_card)
            else:
                card: int = 2*abs(point_card)-1
        self.card_list_.remove(card)
        return card

class BASfusionPlayer(Player):
    """
    plays the BASfusion strategy
    """
    def __init__(self, name: str,n_cards:int) -> None:
        super().__init__(name,n_cards)
        self.strat = 0

    def __repr__(self) -> str:
        return f"BASfusionPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        
        if turn == 0: #choose a new strat at the beginning of a game
            self.strat = np.random.randint(3)
        
        if self.strat==0: #BAS
            if point_card > self.n_cards:
                card: int = point_card + self.n_cards
            else:
                if 2*abs(point_card) in self.card_list_:
                    card: int = 2*abs(point_card)
                else:
                    card: int = 2*abs(point_card)-1

        if self.strat==1: #BAS1

            if point_card > self.n_cards:
                if point_card == 10:
                    card: int = 13
                if point_card == 9:
                    card: int = 15
                if point_card == 8:
                    card: int = 14
                if point_card == 7:
                    card: int = 12
                if point_card == 6:
                    card: int = 11
            else:
                if 2*abs(point_card) in self.card_list_:
                    card: int = 2*abs(point_card)
                else:
                    card: int = 2*abs(point_card)-1

        if self.strat==2: #BAS2
            if point_card > self.n_cards:
                if point_card == 10:
                    card: int = 14
                if point_card == 9:
                    card: int = 15
                if point_card == 8:
                    card: int = 13
                if point_card == 7:
                    card: int = 12
                if point_card == 6:
                    card: int = 11
            else:
                if 2*abs(point_card) in self.card_list_:
                    card: int = 2*abs(point_card)
                else:
                    card: int = 2*abs(point_card)-1

        self.card_list_.remove(card)
        return card


### for n_cards = 3


class threeBAS1Player(Player):
    """
    plays the BAS1 strategy with n_cards = 3
    """

    def __repr__(self) -> str:
        return f"3BAS1Player {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card > self.n_cards:
            if point_card == 6:
                card: int = 7
            if point_card == 5:
                card: int = 9
            if point_card == 4:
                card: int = 8
        else:
            if 2*abs(point_card) in self.card_list_:
                card: int = 2*abs(point_card)
            else:
                card: int = 2*abs(point_card)-1
        self.card_list_.remove(card)
        return card


class threeBAS2Player(Player):
    """
    plays the BAS2 strategy with n_cards = 3
    """

    def __repr__(self) -> str:
        return f"3BAS2Player {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        if point_card > self.n_cards:
            if point_card == 6:
                card: int = 8
            if point_card == 5:
                card: int = 7
            if point_card == 4:
                card: int = 9
        else:
            if 2*abs(point_card) in self.card_list_:
                card: int = 2*abs(point_card)
            else:
                card: int = 2*abs(point_card)-1
        self.card_list_.remove(card)
        return card


class threeBASfusionPlayer(Player):
    """
    plays the BASfusion strategy with n_cards = 3
    """
    def __init__(self, name: str,n_cards:int) -> None:
        super().__init__(name,n_cards)
        self.strat = 0

    def __repr__(self) -> str:
        return f"BAS fusion {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        
        if turn == 0: #choose a new strat at the beginning of a game
            self.strat = np.random.randint(3)
        
        if self.strat==0: #BAS
            if point_card > self.n_cards:
                card: int = point_card + self.n_cards
            else:
                if 2*abs(point_card) in self.card_list_:
                    card: int = 2*abs(point_card)
                else:
                    card: int = 2*abs(point_card)-1

        if self.strat==1: #BAS1

            if point_card > self.n_cards:
                if point_card == 6:
                    card: int = 7
                if point_card == 5:
                    card: int = 9
                if point_card == 4:
                    card: int = 8
        
            else:
                if 2*abs(point_card) in self.card_list_:
                    card: int = 2*abs(point_card)
                else:
                    card: int = 2*abs(point_card)-1

        if self.strat==2: #BAS2
            if point_card > self.n_cards:
                if point_card == 6:
                    card: int = 8
                if point_card == 5:
                    card: int = 7
                if point_card == 4:
                    card: int = 9
            else:
                if 2*abs(point_card) in self.card_list_:
                    card: int = 2*abs(point_card)
                else:
                    card: int = 2*abs(point_card)-1

        self.card_list_.remove(card)
        return card