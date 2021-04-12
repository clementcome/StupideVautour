import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_game(path: str, return_player: bool = False) -> pd.DataFrame:
    with open(path) as f:
        dic_content = json.load(f)
    player_list = dic_content["player_list"]
    game_summary_list = dic_content["game_summary_list"]
    winner_list_list = dic_content["winner_list_list"]
    del dic_content
    print(f"Number of games: {len(game_summary_list)}")
    game_list = []
    game_winner_list = []
    turn_list = []
    point_card_list = []
    card_list = []
    player_idx_list = []
    win_list = []
    for i, (game_summary, winner_list) in enumerate(
        zip(game_summary_list, winner_list_list)
    ):
        for k, turn in enumerate(game_summary):
            j_win = -1
            turn_winner = turn["turn_winner"]
            if turn_winner != "None":
                j_win = player_list.index(turn_winner)
            for j, player_card in enumerate(turn["player_card_list"]):
                game_list.append(i)
                game_winner_list.append(
                    list(map(lambda winner: player_list.index(winner), winner_list))
                )
                turn_list.append(k)
                point_card_list.append(turn["card"])
                card_list.append(player_card)
                player_idx_list.append(j)
                win_list.append(1 if j_win == j else 0)
    if return_player:
        return (
            player_list,
            pd.DataFrame(
                {
                    "game": game_list,
                    "turn": turn_list,
                    "point_card": point_card_list,
                    "card": card_list,
                    "player_idx": player_idx_list,
                    "win": win_list,
                    "game_winner": game_winner_list,
                }
            ),
        )
    return pd.DataFrame(
        {
            "game": game_list,
            "turn": turn_list,
            "point_card": point_card_list,
            "card": card_list,
            "player_idx": player_idx_list,
            "win": win_list,
            "game_winner": game_winner_list,
        }
    )


def plot_hist_win_by_card(df_game: pd.DataFrame):
    df_win = df_game[df_game["win"] == 1]
    fig, axes = plt.subplots(3, 5, figsize=(16, 8))
    for i, value in enumerate(np.unique(df_win["point_card"])):
        ax = axes[i // 5, i % 5]
        df_value = df_win[df_win["point_card"] == value]["card"]
        df_value.hist(ax=ax, bins=14)
        ax.set_title(value)
    plt.plot()


def plot_hist_winner_by_card(df_game: pd.DataFrame):
    df_win = df_game[
        df_game.apply(lambda row: row["player_idx"] in row["game_winner"], axis=1)
    ]
    fig, axes = plt.subplots(3, 5, figsize=(16, 8))
    for i, value in enumerate(np.unique(df_win["point_card"])):
        ax = axes[i // 5, i % 5]
        df_value = df_win[df_win["point_card"] == value]["card"]
        df_value.hist(ax=ax, bins=15)
        ax.set_title(value)
    plt.plot()

def plot_win_over_time(df_game: pd.DataFrame, window=100) -> pd.DataFrame:
    print("Number of games in df_game:", df_game["game"].nunique())
    df_nn_clean = df_game[df_game["game_winner"].apply(len) == 1]
    print("Number of games with a unique winner:", df_nn_clean["game"].nunique())
    df_nn_clean["game_winner"] = df_nn_clean["game_winner"].apply(lambda x: x[0])
    df_nn_win = pd.DataFrame(df_nn_clean.groupby("game")["game_winner"].first())
    for player in df_nn_win["game_winner"].value_counts().index:
        df_nn_win[f"{player}_won"] = df_nn_win["game_winner"] == player
    (df_nn_win.drop(columns=["game_winner"]).rolling(window, min_periods=1).sum()).plot()
    return df_nn_win


if __name__ == "__main__":
    df = read_game("summary_2p_1000g_random_1.json")
    print(df.head())
