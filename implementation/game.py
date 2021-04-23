import numpy as np
from time import time
from tqdm import tqdm
from typing import List, Dict, Union, Tuple
import json
import pickle

from player import Player, RandomPlayer, MaxPlayer, MinPlayer, NNPlayer, BonusCravingPlayer, MalusAdverse, GetRidOfBadCards, Robot, MCPlayer


class Game:
    """
    Implements Stupide Vautour game with the following main functions:
    - `bonus` and `malus` will deal with every player's card to determine
    which one got the card
    - `turn` will allow every player to play at one turn
    - `play` will perform all the turns that contribute to a full game
    For the original game see: https://www.gigamic.com/jeu/stupide-vautour
    """

    def __init__(
        self, n_player: int, n_cards: int = 5, n_game: int = 1, random: bool = False, verbose: bool = True ##
    ) -> None:
        """
        Initialize the game

        Parameters
        ----------
        n_cards : int
            Number of the deck's malus cards so that the game has n_cards of malus cards and 3*n_cards of bonus cards. 
            Standard game is played with n_cards = 5.
        n_player : int
            Number of players in the game
        n_game : int, optional
            Number of game to perform
            Not used right now, by default 1
        random : bool, optional
            True -> Initialize all the players with RandomPlayers
            False -> Ask for names of each player:
                - No name provided: the player will stay random
                - "max" provided as the name: the player will be a MaxPlayer
                - Another name is provided: A normal player 
                    (asks for card in the console)
            , by default False
        verbose : bool, optional
            Whether to print or not messages during evolution
            of the game, by default True
        """
        super().__init__()
        self.n_cards = n_cards ## I want to use it in Player class
        self.n_player_ = n_player
        self.n_game_ = n_game
        self.verbose_ = verbose
        self.card_list_ = list(range(-self.n_cards, 0)) + list(range(1, 2*self.n_cards + 1)) ##
        np.random.shuffle(self.card_list_)
        self.player_list_: List[Player] = [RandomPlayer(i, self.n_cards) for i in range(n_player)]
        self.player_score_list_ = [0 for _ in range(n_player)]
        if not (random):
            for i in range(n_player):
                player_name = input(f"Name of player {i}: ")
                if player_name != "":
                    if player_name == "max":
                        self.player_list_[i] = MaxPlayer(i,self.n_cards)
                    if player_name == 'min':
                        self.player_list_[i] = MinPlayer(i,self.n_cards)
                    if player_name == 'NN':
                        self.player_list_[i] = NNPlayer(i,self.n_cards, learning_rate=0.5, n_player=self.n_player_)
                    if player_name == 'bonus craving':
                        self.player_list_[i] = BonusCravingPlayer(i,self.n_cards)
                    if player_name == 'malus adverse':
                        self.player_list_[i] = MalusAdverse(i,self.n_cards)
                    if player_name == 'get rid of bad cards':
                        self.player_list_[i] = GetRidOfBadCards(i,self.n_cards)
                    if player_name == 'MC':
                        self.player_list_[i] = MCPlayer(i,self.n_cards,[1.0, 0.99999, 0.05])
                    else:
                        self.player_list_[i] = Player(player_name,self.n_cards)
        if self.verbose_:
            print("Game ready with players:", self.player_list_)
            print("and cards:", self.card_list_)
            print("---")

    def bonus(self, card_list: List[int], card: int) -> Union[Player, None]:
        """
        Handles the outcome of a turn when the card is positive
        Needs to handle the tie cases

        Parameters
        ----------
        card_list : List[int]
            List of the cards provided the players
        card : int
            Card to be won at the turn

        Returns
        -------
        Union[Player, None]
            Returns the player that won the card if there is one
            else returns None
        """
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
                card_list = [card if card != m else 0 for card in card_list] # ties are broken by taking next highest value card

    def malus(self, card_list: List[int], card: int) -> Union[Player, None]:
        """
        Handles the outcome of a turn when the card is negative
        Needs to handle the tie cases

        Parameters
        ----------
        card_list : List[int]
            List of the cards provided the players
        card : int
            Card to be won at the turn

        Returns
        -------
        Union[Player, None]
            Returns the player that won the card if there is one
            else returns None
        """
        while True:
            m = min(card_list)
            upper = self.n_cards * 3 + 1 ##
            if m == upper: # this is impossible for the first loop, while m == 16 was if we extend the game with more than 16 cards in hand ##
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
                card_list = [card if card != m else upper for card in card_list] ##

    def turn(self, card: int, turn: int) -> Tuple[Player, List[int]]:
        """
        Performs a turn of the game:
        - Retrieves the car played by each player
        - Determines the winner of turn
        - Inform each player of the outcome of the turn

        Parameters
        ----------
        card : int
            Card picked for this turn
        turn : int
            Number of the turn to be played

        Returns
        -------
        Tuple[Player, List[int]]
            Returns the player that one this turn and 
            the cards played by each player
        """
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
        """
        Performs all the turns needed to do a whole game

        Returns
        -------
        Tuple[List[Player], List[Dict], List[Player]]
            A list of the players that took part to the game,
            the record of the whole game,
            the list of all winners at each turn
        """
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
                player.win(self.n_player_)
                winner_list.append(player)
            else:
                player.lose(self.n_player_)
            player.end_game()
        self.card_list_ = list(range(-self.n_cards, 0)) + list(range(1, 2*self.n_cards + 1)) ##
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
    with open(f"data/game_summary/summary_{n_player}p_{n_game}g_.json", "w+") as f:
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
    # with open(f"../data/saved_model/weights.pkl", "wb") as f:
    #     pickle.dump({"A": nn_player_1.A_, "b": nn_player_1.b_}, f)

