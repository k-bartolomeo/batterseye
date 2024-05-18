import os
import random
from itertools import combinations
from collections import defaultdict

import numpy as np
import numpy.typing as npt

import pandas as pd
import tensorflow as tf

from tqdm import tqdm


class ImgLoader:
    """Loads images from given directory

    Uses provided DataFrame to match images to the batters and pitchers
    in them. Depending on value provided to `kind` argument during 
    initialization, creates a dictionary mapping batter IDs to images 
    of the batters, a dictionary mapping pitcher IDs to images of the 
    pitchers, or both. Crops images upon loading to ensure that players
    are central focus of the image.

    Attributes
    ----------
    batters : dict[str, list]
        Dictionary mapping batter ID to a list of images of that batter.
    pitchers : dict[str, list]
        Dictionary mapping pitcher ID to a list of images of that pitcher.
    matchups : pd.DataFrame
        pandas DataFrame containing play ID and matchup information.
    batter_id_dict : dict[str, str] | None
        Dictionary mapping play ID to batter ID if value provided to 
        `kind` argument during initialization is either 'batter' or 
        'both'. Otherwise, `None`.
    pitcher_id_dict : dict[str, str] | None
        Dictionary mapping play ID to pitcher ID if value provided to 
        `kind` argument during initialization is either 'pitcher' or 
        'both'. Otherwise, `None`.
    kind : str
        Type of images returned by image loader.
    
    Parameters
    ----------
    matchups
        pandas DataFrame containing play ID and matchup information.
    kind
        Type of images returned by image loader.
    """
    def __init__(self, matchups: pd.DataFrame, kind: str = 'both'):
        self.batters = defaultdict(list)
        self.pitchers = defaultdict(list)
        self.matchups = matchups
        self.batter_id_dict = (
            self._init_batter_id_dict(matchups) if kind != 'pitcher' else None
        )
        self.pitcher_id_dict = (
            self._init_pitcher_id_dict(matchups) if kind != 'batter' else None
        )
        self.kind = kind
        if self.kind == 'pitcher':
            self._load_fn = self._load_pitcher_img
        elif self.kind == 'batter':
            self._load_fn = self._load_batter_img
        else:
            self._load_fn = self._load_both_imgs

    @staticmethod
    def _init_batter_id_dict(matchups: pd.DataFrame) -> dict:
        """Initializes dictionary mapping play IDs to batter IDs"""
        return dict(matchups[['play_id', 'batter_id']].values)
    
    @staticmethod
    def _init_pitcher_id_dict(matchups: pd.DataFrame) -> dict:
        """Initializes dictionary mapping play IDs to pitcher IDs"""
        return dict(matchups[['play_id', 'pitcher_id']].values)

    def _load_batter_img(self, img: npt.NDArray, play_id: str) -> None:
        """Adds image to dictionary mapping batters to their images"""
        batter_id = self.batter_id_dict[play_id]
        self.batters[batter_id].append(img)

    def _load_pitcher_img(self, img: npt.NDArray, play_id: str) -> None:
        """Adds image to dictionary mapping pitchers to their images"""
        pitcher_id = self.pitcher_id_dict[play_id]
        self.pitchers[pitcher_id].append(img)

    def _load_both_imgs(self, img: npt.NDArray, play_id: str) -> None:
        """Adds images to both batter and pitcher dictionaries"""
        self._load_batter_img(img, play_id)
        self._load_pitcher_img(img, play_id)

    def load(
        self, data_dir: str, crop_rate: float = 0.6
    ) -> defaultdict | tuple[defaultdict, defaultdict]:
        """Loads images from given directory
        
        Creates dictionaries of batter images, pitcher images, or
        both depending on value supplied to `kind` argument during
        initialization. Images are generally wide-angled, so they 
        are cropped to the center of the image. This allows
        the batter and the pitcher to take up more of the image
        while removing unneeded visual information like the field
        and the crowd.

        Parameters
        ----------
        data_dir
            Path to directory storing .npy files with image data.
        crop_rate
            How much the image should be cropped after it is loaded.

        Returns
        -------
        defaultdict | tuple[defaultdict, defaultdict]
            Dictionary mapping batter IDs to images of that batter,
            dictionary mapping pitcher IDs to images of that pitcher,
            or both.
        """
        for fname in tqdm(os.listdir(data_dir)):
            path = os.path.join(data_dir, fname)
            try:
                img = np.load(path)[..., [2, 1, 0]]
                img = tf.image.central_crop(img, crop_rate)
                play_id = fname.rstrip('.npy')
                self._load_fn(img=img, play_id=play_id)
            except:
                continue

        if self.kind == 'batter':
            return self.batters
        elif self.kind == 'pitcher':
            return self.pitchers
        else:
            return self.batters, self.pitchers
        
    def split_ids(self) -> list[set]:
        """Splits batter and pitcher IDs into sets by handedness"""
        player_ids = [None, None, None, None]

        if self.kind in {'batter', 'both'}:
            player_ids[0] = set(self.matchups[self.matchups.bats == 'L'].batter_id)
            player_ids[1] = set(self.matchups[self.matchups.bats == 'R'].batter_id)
        elif self.kind in {'pitcher', 'both'}:
            player_ids[2] = set(self.matchups[self.matchups.throws == 'L'].pitcher_id)
            player_ids[3] = set(self.matchups[self.matchups.throws == 'R'].pitcher_id)

        return [x for x in player_ids if x is not None]
        
    def __len__(self):
        return len(self.batters), len(self.pitchers)
    
    @property
    def pitchers(self) -> set:
        """Gets set of pitcher IDs"""
        return set(self.pitchers)
    
    @property
    def batters(self) -> set:
        """Gets set of batter IDs"""
        return set(self.batters)
    

def build_siamese_dataset(
    players: dict[str, list],
    lefty: set[str],
    righty: set[str],
    same_factor: int = 3,
    n_diff: int = 500
):
    """Builds dataset for training and evaluating Siamese network
    
    Creates pairs of images of players with same handedness. Since 
    number of image pairs that can be created where the player is 
    the same will generally be fewer than the number of image pairs
    that can be created where the players are different, the pairs
    where the player is the same are oversampled using the provided
    `same_factor`. For image pairs with different players, each 
    player will be included in at least `n_diff` such pairs.


    Parameters
    ----------
    players
        Dictionary mapping player IDs to images of the player.
    lefty
        Set of player IDs for left-handed players.
    righty
        Set of player IDs for right-handed players.
    same_factor
        Scaling factor for number of image pairs of same player 
        to include in dataset.
    n_diff
        Number of image pairs of different players to create
        for a given player.
    """
    pairs = []
    labels = []

    all_players = set(players)
    for player in tqdm(all_players):
        if len(players[player]) < 2:
            continue
        same_pairs = list(combinations(players[player], 2))
        same_pairs = same_pairs * same_factor

        player_set = lefty if player in lefty else righty
        diff_players = list(player_set - set([player]))
        for _ in range(n_diff):
            if len(same_pairs) > 0:
                same_player = same_pairs.pop()
                pairs.append(same_player)
                labels.append(0)

            player_img = random.choice(players[player])
            diff_player = random.choice(diff_players)
            diff_player_img = random.choice(players[diff_player])
            pairs.append((player_img, diff_player_img))
            labels.append(1)

    records = list(zip(pairs, labels))
    random.shuffle(records)
    pairs = [x[0] for x in records]
    labels = [x[1] for x in records]

    return pairs, labels


def split(
    pairs: list[tuple], labels: list[int], val: float, test: float
) -> tuple[list, list, list, list, list, list]:
    """Splits data into train, val, and test sets"""
    train = 1 - val - test
    train_val = train + val
    train_split = int(len(pairs) * train)
    train_val_split = int(len(pairs) * train_val)

    X_train = pairs[:train_split]
    X_val = pairs[train_split:train_val_split]
    X_test = pairs[train_val_split:]

    y_train = labels[:train_split]
    y_val = labels[train_split:train_val_split]
    y_test = labels[train_val_split:]

    return X_train, y_train, X_val, y_val, X_test, y_test
