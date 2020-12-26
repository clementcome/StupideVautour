from typing import List
import numpy as np


class Player:
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name_: str = name
        self.card_list_: List[int] = [i for i in range(1, 16)]
        self.score_: int = 0

    def __repr__(self) -> str:
        return f"{self.name_}"

    def play(self, point_card: int) -> int:
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


class MaxPlayer(Player):
    def __repr__(self) -> str:
        return f"MaxPlayer {self.name_}"

    def play(self, point_card: int) -> int:
        card = max(self.card_list_)
        self.card_list_.remove(card)
        return card


class RandomPlayer(Player):
    def __repr__(self) -> str:
        return f"RandomPlayer {self.name_}"

    def play(self, point_card: int) -> int:
        card: int = int(np.random.choice(self.card_list_))
        self.card_list_.remove(card)
        return card
