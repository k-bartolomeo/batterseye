import os
from collections import defaultdict

import numpy as np
import numpy.typing as npt

import pandas as pd
import statsapi as sa
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
        self._batters = defaultdict(list)
        self._pitchers = defaultdict(list)
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
        self._batters[batter_id].append(img)

    def _load_pitcher_img(self, img: npt.NDArray, play_id: str) -> None:
        """Adds image to dictionary mapping pitchers to their images"""
        pitcher_id = self.pitcher_id_dict[play_id]
        self._pitchers[pitcher_id].append(img)

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
            return self._batters
        elif self.kind == 'pitcher':
            return self._pitchers
        else:
            return self._batters, self._pitchers
        
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
        return len(self._batters), len(self._pitchers)
    
    @property
    def pitchers(self) -> set:
        """Gets set of pitcher IDs"""
        return set(self._pitchers)
    
    @property
    def batters(self) -> set:
        """Gets set of batter IDs"""
        return set(self._batters)
    

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


def get_player_team(player_id: int) -> list[int]:
    """Retrieve given player's teams during 2023 MLB season"""
    query = {
        'personIds': player_id,
        'season': 2023,
        'hydrate': 'stats(group=[pitching],type=season,season=2023)'
    }
    player = sa.get('people', query)
    player_splits = player['people'][0]['stats'][0]['splits']
    team_ids = [x['team']['id'] for x in player_splits if 'team' in x]
    return team_ids


def get_teammates(player_team_map: dict) -> dict:
    """Gets dictionary mapping player IDs to teammates' IDs"""
    teammates = {}
    for player, teams in tqdm(player_team_map.items()):
        same_team = []
        for team in teams:
            t = [
                k for k in player_team_map
                if team in player_team_map[k] and k != player
            ]
            same_team.extend(t)
        teammates[player] = same_team

    return teammates


def hash_tensor(tensor: tf.Tensor) -> str:
    """Gets hash value for 3D image tensor"""
    h = str(hash(tensor.numpy()[0].tobytes()))
    return h


def hash_img_pair(img_pair: list[tf.Tensor]) -> str:
    """Gets hash value for pair of 3D image tensors
    
    Gets hash value for each image in pair individually. Converts
    hash values to strings and then joins them together, returning
    a single string.

    Parameters
    ----------
    img_pair
        Pair of images for which hash value is derived.

    Returns
    -------
    str
        Hash value for given image pair.
    """
    h = ''.join([hash_tensor(img) for img in img_pair])
    return h


def get_img_pair_hashes(img_pairs: list[tuple]) -> list[str]:
    """Gets list of hashes for given image pairs"""
    hashes = [hash_img_pair(img_pair) for img_pair in tqdm(img_pairs)]
    return hashes


def write_img_hashes(hashes: list[str], path: str) -> None:
    """Writes list of hash values to text file"""
    lines = [f'{h}\n' for h in hashes]
    with open(path, 'w') as f:
        f.writelines(lines)
    f.close()


def load_img_hashes(path: str) -> set:
    """Loads list of hash values from given text file"""
    with open(path, 'r') as f:
        hashes = set(h.rstrip('\n') for h in f.readlines())
    f.close()
    return hashes
