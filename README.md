# Stupid Python

Implementation of "Stupide Vautour" and artificial players.

The project is divided in several parts:

- `implementation/game.py` defines the game rules and flow, an example of its use is given at the end of the file
- `implementation/player.py` defines the players that can be used to play the game
- `implementation/analyse.py` contains several functions to process the results of the games played

## Simple explanation of game rules

Stupide Vautour is a turn by turn game.

1. Every player is given 15 cards from 1 to 15. A player can see only its cards and can play each of them only once.
2. A deck of 5 malus cards (-5 to -1) and 10 bonus cards (1 to 10) is shuffled.
3. For each card in the deck:
   1. Turn over the bonus/malus card
   2. Each player chooses a card from its hand that he want to play during this turn and put it face down so no one sees it.
   3. When every player has put down its card, they can turn it face up simultaneously.
   4. - If the card is a malus the player that played the smallest card gets it.
      - If the card is a bonus the player that played the highest card gets it.
      - In case of a tie, the card goes to the next smallest (or highest) card.
4. The final score of each player is computed with the sum of bonuses minus the sum of maluses.
5. The winner is the player with the highest final score

The real rules have other slight details that are not considered here.

## Modelisation of the game for reinforcement learning

Several aspects can be considered when modeling this game:

- State: what is the state of the game for a given player ?
  - Should one consider the card of the agent in the state?
  - Should one use the cards previously played by the other players?
- ...
