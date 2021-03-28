import numpy as np
from time import time
from tqdm import tqdm
from typing import List, Dict, Union, Tuple
import json
import pickle

from player import Player, RandomPlayer, MaxPlayer, NNPlayer


class Game:
    def __init__(
        self, n_player: int, n_game: int = 1, random: bool = False, verbose: bool = True
    ) -> None:
        super().__init__()
        self.n_player_ = n_player
        self.n_game_ = n_game
        self.verbose_ = verbose
        self.card_list_ = list(range(-5, 0)) + list(range(1, 11))
        np.random.shuffle(self.card_list_)
        self.player_list_: List[Player] = [RandomPlayer(i) for i in range(n_player)]
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

    def bonus(self, card_list: List[int], card: int) -> Union[Player, None]:
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

    def malus(self, card_list: List[int], card: int) -> Union[Player, None]:
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

    def turn(self, card: int, turn: int) -> Tuple[Player, List[int]]:
        if self.verbose_:
            print("Card for this turn is:", card)
        player_card_list = [player.play(card, turn) for player in self.player_list_]
        if self.verbose_:
            print("Cards played:")
            for player, player_card in zip(self.player_list_, player_card_list):
                print(f"{player}: {player_card}")
        if card > 0:
            getter = self.bonus(player_card_list, card)
        else:
            getter = self.malus(player_card_list, card)

        if getter is None:
            for player in self.player_list_:
                player.no_one_get(card)
        else:
            for player in self.player_list_:
                if player == getter:
                    player.get(card)
                else:
                    player.dont_get(card)

        return getter, player_card_list

    def play(self) -> Tuple[List[Player], List[Dict], List[Player]]:
        start_time = time()
        game = []
        for i, card in enumerate(self.card_list_):
            turn_winner, player_card_list = self.turn(card, i)
            game.append(
                {
                    "card": card,
                    "player_card_list": player_card_list,
                    "turn_winner": str(turn_winner),
                }
            )
        winner_score = max(self.player_score_list_)
        winner_list = []
        if self.verbose_:
            print("Final scores:")
            for player, score in zip(self.player_list_, self.player_score_list_):
                if score == winner_score:
                    print(f"{player}'s score: {score} | Winner")
                else:
                    print(f"{player}'s score:", score)
            print(f"Game took {time() - start_time}s")
        for player, score in zip(self.player_list_, self.player_score_list_):
            if score == winner_score:
                player.win()
                winner_list.append(player)
            else:
                player.lose()
            player.end_game()
        self.card_list_ = list(range(-5, 0)) + list(range(1, 11))
        np.random.shuffle(self.card_list_)
        self.n_game_ = self.n_game_ - 1
        self.player_score_list_ = [0 for _ in range(self.n_player_)]
        return self.player_list_, game, winner_list


# Playing several games
if __name__ == "__main__":
    n_player = 2
    n_game = 1
    game = Game(n_player, n_game=n_game, random=True, verbose=True)
    nn_player_1 = NNPlayer("Tristan", learning_rate=0.6)
    player_2 = Player("Eva")
    # game.player_list_[0] = nn_player_1
    game.player_list_[1] = player_2
    game_summary_list = []
    winner_list_list = []
    for _ in tqdm(range(n_game)):
        player_list, game_summary, winner_list = game.play()
        game_summary_list.append(game_summary)
        winner_list_list.append(list(map(str, winner_list)))
    # print(player_list)
    # print(game_summary_list)
    # print(winner_list_list)
    with open(f"summary_{n_player}p_{n_game}g_nn_1.json", "w") as f:
        json.dump(
            {
                "player_list": list(map(str, player_list)),
                "game_summary_list": game_summary_list,
                "winner_list_list": winner_list_list,
            },
            f,
            indent=4,
        )
    # Saving the weights of neural network player
    # with open(f"weights.pkl", "wb") as f:
    #     pickle.dump({"A": nn_player_1.A_, "b": nn_player_1.b_}, f)

