import random
from itertools import combinations

import tensorflow as tf
from tqdm import tqdm

from .utils import hash_img_pair
from .generator import PairGenerator


def make_tf_dataset(
    pairs: list, 
    labels: list,
    output_types: tuple,
    output_shapes: tuple,
    batch_size: int,
    repeats: int,
    training: bool = False,
    return_gen: bool = False,
) -> tf.data.Dataset:
    """Builds TensorFlow dataset from given data

    Creates instance of PairGenerator using provided image pairs and 
    labels. Generator needed because size of dataset too big for inputs
    and outputs to be saved in memory as a pair of NumPy arrays. Arrays
    for individual images can be loaded into memory as NumPy arrays and 
    stored within a list comfortably.

    Parameters
    ----------
    pairs
        List of image pairs stored as NumPy arrays.
    labels
        List of boolean values denoting whether images in pairs are 
        of same or different players. 0 indicates same player, and 1
        indicates different players.
    output_types
        Two-item tuple denoting dtypes of values returned by dataset.
        First item of tuple is dictionary with keys 'x1' and 'x2' and
        values that are TensorFlow dtypes. Second item of tuple is 
        TensorFlow dtype.
    output_shapes
        Two-item tuple denoting shapes of values returned by dataset.
        First item of tuple is dictionary with keys 'x1' and 'x2' and 
        values for (height, width, channels) of arrays returned. Second
        item of tuple is either `None` or a TensorFlow dtype.
    batch_size
        Batch size for the dataset.
    repeats
        Number of times to repeat dataset. Usually set to number of 
        epochs for which model is trained. Used to ensure  that 
        dataset is repeated sufficiently such that TensorFlow does 
        not return an error suggesting the generator has run out of 
        data during training.
    training
        Boolean indicating whether or not the dataset is a training 
        or test dataset. Value passed to PairGenerator initialization.
    return_gen
        Boolean indicating whether or not PairGenerator should be 
        returned along with dataset.

    Returns
    -------
    tf.data.Dataset
        TensorFlow dataset for given image pairs and labels. Batched
        and repeated using given parameters.
    """
    gen = PairGenerator(pairs=pairs, labels=labels, training=training)
    dataset = tf.data.Dataset.from_generator(
        gen, output_types=output_types, output_shapes=output_shapes
    )
    dataset = dataset.batch(batch_size).repeat(repeats)
    if return_gen:
        return dataset, gen
    return dataset


def build_siamese_dataset(
    players: dict[str, list],
    lefty: set[str],
    righty: set[str],
    teammates: dict[str, list],
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
        # If there are only 1 or 2 images in dataset for given 
        # player, skip because there isn't enough useful info
        # for training model.
        if len(players[player]) < 2:
            continue
        same_pairs = list(combinations(players[player], 2))
        same_pairs = same_pairs * same_factor

        # Subset used for comparison constrained to player with
        # same handedness. Ensures model is capable of detecting 
        # smaller differences, as opposed  to just obvious ones, 
        # like left-handed vs. right-handed. 
        player_set = lefty if player in lefty else righty
        diff_players = list(player_set - set([player]))
        same_team = teammates[player]
        remaining_diff = n_diff

        # First create pairings with teammates to ensure that model
        # can distinguish between more than just different uniform
        # colors and styles.
        for teammate in same_team:
            teammate_samples = min(5, len(players[teammate]))
            teammate_imgs = random.sample(players[teammate], k=teammate_samples)
            for teammate_img in teammate_imgs:
                player_img = random.choice(players[player])
                pairs.append((player_img, teammate_img))
                labels.append(1)
                remaining_diff -= 1

        # After pairings with teammates created, create pairings
        # between random players and pairings of same player.
        for _ in range(remaining_diff):
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


def build_comp_set(
    player: int,
    test_img: tf.Tensor, 
    comps: list[tf.Tensor],
    comp_size: int,
    hashes: list[str],
    other_player: int = None,
) -> tuple[list, list]:
    """Builds mini dataset of comparisons for given player

    Given a test image for which the player ID is assumed to be 
    unknown, creates a set of pairings of this image and other 
    images for which the player ID is assumed to be known. Comparison
    images in pairings may be of same player in test image or may 
    be of different player.

    Parameters
    ----------
    player
        Player ID for which set of comparisons is built.
    test_img
        Baseline image for player with given player ID against
        which other images with known labels is compared.
    comps
        List of images to compare against given test image.
    comp_size
        Number of comparisons to include in mini dataset.
    hashes
        List of image pair hashes for image pairs used during
        training. Ensures that none of the same pairings appear
        in the comparison set.
    other_player
        Optional player ID for a second player if the comparison
        set being constructed is for different players instead
        of for the same player.

    Returns
    -------
    tuple[list, list]
        Two-item tuple, where first item is list of image pairs and
        second item is list of player ID pairs.
    """
    imgs = []
    labels = []
    player_pair = (player, player)
    if other_player is not None:
        player_pair = (other_player, player)

    n_comps = len(comps)
    comp_count = 0
    comp_idx = 0
    while comp_count < comp_size and comp_idx < n_comps:
        comp = comps[comp_idx]
        comp_idx += 1
        pair_hash = hash_img_pair([test_img, comp])
        reverse_pair_hash = hash_img_pair([comp, test_img])

        # Ensures that image pair was not seen during training.
        if pair_hash in hashes or reverse_pair_hash in hashes:
            continue

        imgs.append((comp, test_img))
        labels.append(player_pair)
        comp_count += 1

    return imgs, labels


def get_test_and_comps(imgs: list[tf.Tensor]) -> tuple[tf.Tensor, list[tf.Tensor]]:
    """Randomly splits images into test image and comparisons"""
    imgs = random.sample(imgs, k=len(imgs))
    test_img = imgs[0]
    comps = imgs[1:]
    return test_img, comps


def build_classification_set(
    players: dict[int, list[tf.Tensor]],
    player_set: set,
    train_hashes: set,
    comp_size: int,
) -> tuple[list, list]:
    """Builds test set used for classifying images

    For each player in the dataset, one image is randomly selected to
    be the test image for that player. Assumption is that this image 
    has been collected and needs to be assigned to one of the players 
    in the dataset. For all players in the dataset, a set of no more 
    than `comp_size` labelled images is randomly selected. 
    
    Each image in set is paired with the test image, and known labels 
    in set are paired with what is assumed to be the unknown label. 
    Comparison sets for each player in dataset are aggregated as a list 
    of image pairs and label pairs and returned such that they can be
    fed to the trained model, which will then assign a probability that
    the images in each pair are of the same player. To classifiy a given
    test image, the known label in the pair producing the highest 
    probability of similarity is assigned to the test image.

    Parameters
    ----------
    players
        Dictionary mapping player IDs to player images.
    player_set
        Set of all player IDs.
    train_hashes
        Set of hash values for image pairs used to train model.
    comp_size
        Number of images randomly selected from each player in 
        dataset against which test images will be compared.

    Returns
    -------
    tuple[list, list]
        Two-item tuple, where first item is list of image pairs and
        second item is list of player ID pairs.
    """
    player_imgs = []
    player_labels = []

    for player, imgs in tqdm(players.items()):
        if len(imgs) < 2:
            continue

        # Gets test image for given player and set of labelled
        # images for same player against which test image can
        # be compared.
        test_img, comps = get_test_and_comps(imgs)
        imgs, labels = build_comp_set(
            player=player,
            test_img=test_img,
            comps=comps,
            comp_size=comp_size, 
            hashes=train_hashes
        )
        player_imgs.extend(imgs)
        player_labels.extend(labels)

        # For each other player in dataset, gets set of labelled
        # images against which test image can be compared.
        other_players = player_set - set([player])
        for other_player in other_players:
            other_imgs = players[other_player]
            random.shuffle(other_imgs)
            imgs, labels = build_comp_set(
                player=player,
                test_img=test_img,
                comps=other_imgs,
                comp_size=comp_size,
                hashes=train_hashes,
                other_player=other_player
            )
            player_imgs.extend(imgs)
            player_labels.extend(labels)

    records = list(zip(player_imgs, player_labels))
    random.shuffle(records)
    player_imgs = [x[0] for x in records]
    player_labels = [x[1] for x in records]

    return player_imgs, player_labels
