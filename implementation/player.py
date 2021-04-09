from typing import List
import numpy as np
from scipy.special import softmax


class Player:
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name_: str = name
        self.card_list_: List[int] = [i for i in range(1, 16)]
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

    def win(self) -> None:
        pass

    def lose(self) -> None:
        pass

    def end_game(self) -> None:
        self.score_ = 0
        self.card_list_ = [i for i in range(1, 16)]


class NNPlayer(Player):
    def __init__(self, name: str, weights=None, learning_rate=0.5, n_player=2,) -> None:
        super().__init__(name)
        if weights is None:
            self.A_ = np.ones((15, 2))
            self.b_ = np.zeros((15, 1))
        else:
            self.A_ = weights["A"]
            self.b_ = weights["b"]
        self.loss_ = np.zeros((15, 1))
        self.learning_rate_ = learning_rate
        self.n_player_ = n_player
        self.t_ = 1
        self.x_list_ = np.zeros((15, 2))
        self.sigma_gradient_ = np.zeros((15, 15))

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
        prod_grad = -np.ones(15) * probabilities[card - 1]
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
        self.x_list_ = np.zeros((15, 2))
        self.sigma_gradient_ = np.zeros((15, 15))
        return super().end_game()


class MaxPlayer(Player):
    def __repr__(self) -> str:
        return f"MaxPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        card = max(self.card_list_)
        self.card_list_.remove(card)
        return card


class RandomPlayer(Player):
    def __repr__(self) -> str:
        return f"RandomPlayer {self.name_}"

    def play(self, point_card: int, turn: int = 0) -> int:
        card: int = int(np.random.choice(self.card_list_))
        self.card_list_.remove(card)
        return card
