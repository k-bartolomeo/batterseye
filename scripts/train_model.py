import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from sklearn.metrics import classification_report

from ..batterseye.data.utils import (
    ImgLoader,
    get_player_team,
    get_teammates,
    split,
    load_img_hashes,
    hash_img_pair,
    write_img_hashes
)
from ..batterseye.data.dataset import (
    make_tf_dataset, build_siamese_dataset, build_classification_set
)
from ..batterseye.data.generator import PairGenerator
from ..batterseye.models.siamese import SiameseNetwork
from ..batterseye.models.loss import contrastive_loss
from ..batterseye.utils import evalaute


if __name__ == '__main__':
    """Trains Siamese network for player pair classification"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-kind', default='pitcher')
    parser.add_argument('--img-dir', default='./data')
    parser.add_argument('--val-split', default=0.2)
    parser.add_argument('--test_split', default=0.2)
    parser.add_argument('--train-hash-path', default='./hashes/train_hashes.txt')
    parser.add_argument('--batch-size', default=16)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--input-dim', default=3)
    parser.add_argument('--output-dim', default=32)
    parser.add_argument('--layer-type', default='residual')
    parser.add_argument('--blocks', default=2)
    parser.add_argument('--filters', default=32)
    parser.add_argument('--kernel-size', default=3)
    parser.add_argument('--strides', default=1)
    parser.add_argument('--margin', default=1)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--steps-per-epoch', default=20_000)
    parser.add_argument('--validation-steps', default=5_000)
    parser.add_argument('--weight-path', default='./weights/siamese.weights.h5')
    args = parser.parse_args()

    print('Loading gameplay matchup data')
    matchups = pd.read_csv('../data/matchups.csv')
    pitcher_ids = matchups.pitcher_id.unique()
    pitcher_team_map = {
        pitcher_id: get_player_team(pitcher_id) for pitcher_id in tqdm(pitcher_ids)
    }
    teammates = get_teammates(pitcher_team_map)
    matchups['pitcher_team'] = matchups.pitcher_id.map(pitcher_team_map)
    print('Matchup data loaded and processed')

    print('Loading images')
    img_loader = ImgLoader(matchups=matchups, kind=args.img_kind)
    players = img_loader.load(data_dir=args.img_dir)
    lefties, righties = img_loader.split_ids()
    print('Images loaded')
    print('Building dataset for Siamese network')
    pairs, labels = build_siamese_dataset(
        players=players,
        lefty=lefties,
        righty=righties,
        teammates=teammates
    )
    print('Splitting dataset into training, validation, and test sets')
    X_train, y_train, X_val, y_val, X_test, y_test = split(
        pairs, labels, val=args.val_split, test=args.test_split
    )
    print(f'Training set with {len(X_train):,} examples created')
    print(f'Validation set with {len(X_val):,} examples created')
    print(f'Test set with {len(X_test):,} examples created')

    print('Hashing training image pairs')
    train_hashes = [hash_img_pair(x) for x in X_train]
    write_img_hashes(hashes=train_hashes, path=args.train_hash_path)
    print(f'Train image pair hashes written to {args.train_hash_path}')

    output_types = ({'x1': tf.float32, 'x2': tf.float32}, tf.int32)
    output_shapes = ({'x1': (216, 216, 3), 'x2': (216, 216, 3)}, ())
    train_ds = make_tf_dataset(
        pairs=X_train,
        labels=y_train,
        output_types=output_types,
        output_shapes=output_shapes,
        batch_size=args.batch_size,
        repeats=args.epochs,
        training=True
    )
    val_ds = make_tf_dataset(
        pairs=X_val,
        labels=y_val,
        output_types=output_types,
        output_shapes=output_shapes,
        batch_size=args.batch_size,
        repeats=args.epochs,
        training=False
    )
    test_ds = make_tf_dataset(
        pairs=X_test,
        labels=y_test,
        output_types=output_types,
        output_shapes=output_shapes,
        batch_size=args.batch_size,
        repeats=1,
        training=False
    )

    print('Building Siamese network with given parameters')
    siamese = SiameseNetwork(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        layer_type=args.layer_type,
        n_blocks=args.blocks,
        filters=args.filters,
        kernel_size=args.kernel_size,
        strides=args.strides
    )
    siamese.compile(
        loss=contrastive_loss(margin=args.margin),
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        metrics=['accuracy']
    )
    print('Fitting Siamese network on training set')
    history = siamese.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=val_ds,
        validation_steps=args.validation_steps
    )
    siamese.save_weights(args.weight_path)
    print('Training completed')
    print(f'Trained model weights saved to {args.weight_path}')

    print('Generating predictions for test set')
    y_pred = siamese.predict(test_ds).flatten()
    y_pred = np.where(y_pred < 0.5, 0, 1)

    print('Performance of trained model on test set:')
    print(classification_report(y_true=y_test, y_pred=y_pred))
