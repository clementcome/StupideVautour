import numpy as np


class Game:
    def __init__(self, n_player, random=False, verbose=True) -> None:
        super().__init__()
        self.n_player_ = n_player
        self.verbose_ = verbose
        self.card_list_ = list(range(-5, 0)) + list(range(1, 11))
        np.random.shuffle(self.card_list_)
        self.player_list_ = [RandomPlayer(i) for i in range(n_player)]
        self.player_score_list_ = [0 for _ in range(n_player)]
        if not (random):
            for i in range(n_player):
                player_name = input(f"Name of player {i}: ")
                if player_name != "":
                    if player_name == "max":
                        self.player_list_[i] = MaxPlayer(i)
                    else:
                        self.player_list_[i] = Player(player_name)
        if self.verbose_:
            print("Game ready with players:", self.player_list_)
            print("and cards:", self.card_list_)
            print("---")

    def bonus(self, card_list, card):
        while True:
            m = max(card_list)
            if m == 0:
                if self.verbose_:
                    print("No one won this card")
                    print("---")
                return None
            elif sum(1 if card == m else 0 for card in card_list) == 1:
                player_idx = card_list.index(m)
                player = self.player_list_[player_idx]
                self.player_score_list_[player_idx] += card
                if self.verbose_:
                    print(f"{player} won this card! :-)")
                    print("---")
                return player
            else:
                card_list = [card if card != m else 0 for card in card_list]

    def malus(self, card_list, card):
        while True:
            m = min(card_list)
            if m == 16:
                if self.verbose_:
                    print("No one won this card")
                    print("---")
                return None
            elif sum(1 if card == m else 0 for card in card_list) == 1:
                player_idx = card_list.index(m)
                player = self.player_list_[player_idx]
                self.player_score_list_[player_idx] += card
                if self.verbose_:
                    print(f"{player} won this card.. :-(")
                    print("---")
                return player
            else:
                card_list = [card if card != m else 16 for card in card_list]

    def turn(self, card):
        if self.verbose_:
            print("Card for this turn is:", card)
        player_card_list = [player.play(card) for player in self.player_list_]
        if self.verbose_:
            print("Cards played:")
            for player, player_card in zip(self.player_list_, player_card_list):
                print(f"{player}: {player_card}")
        if card > 0:
            return self.bonus(player_card_list, card)
        else:
            return self.malus(player_card_list, card)

    def play(self):
        for card in self.card_list_:
            self.turn(card)
        if self.verbose_:
            print("Final scores:")
            for player, score in zip(self.player_list_, self.player_score_list_):
                print(f"{player}'s score:", score)


class Player:
    def __init__(self, name) -> None:
        super().__init__()
        self.name_ = name
        self.card_list_ = [i for i in range(1, 16)]

    def __repr__(self) -> str:
        return f"{self.name_}"

    def play(self, point_card):
        print("Card for this turn is:", point_card)
        while True:
            print(f"{self.name_}' cards available", self.card_list_)
            card = int(input(f"{self.name_}'s card: "))
            if card in self.card_list_:
                break
            else:
                print(f"{card} is not available")
        self.card_list_.remove(card)
        return card


class MaxPlayer(Player):
    def __repr__(self) -> str:
        return f"MaxPlayer {self.name_}"

    def play(self, point_card):
        card = max(self.card_list_)
        self.card_list_.remove(card)
        return card


class RandomPlayer(Player):
    def __init__(self, name) -> None:
        super().__init__(name)

    def __repr__(self) -> str:
        return f"RandomPlayer {self.name_}"

    def play(self, point_card):
        card = np.random.choice(self.card_list_)
        self.card_list_.remove(card)
        return card


if __name__ == "__main__":
    game = Game(2)
    game.play()
