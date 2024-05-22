import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from ..batterseye.utils import evaluate
from ..batterseye.models.siamese import SiameseNetwork
from ..batterseye.data.utils import load_img_hashes, ImgLoader
from ..batterseye.data.dataset import build_classification_set, make_tf_dataset


def run_eval(
    mode: str, preds: np.array, cls_labels: list[tuple[int, int]], top_k: int
) -> None:
    """Runs evaluation and prints classification report"""
    acc_type = 'Accuracy' if top_k == 1 else f'Top-{top_k} Accuracy'
    if mode == 'atleast':
        mode_msg = 'at least'
    else:
        mode_msg = 'exactly'

    accs = {}
    for i in range(1, 6):
        correct, chances, acc, _ = evaluate(
            predictions=preds, labels=cls_labels, mode='atleast', k=i, top_k=1
        )
        accs[i] = acc

    print('\n', '=' * 50)
    print(f'{acc_type} of model with {mode_msg} k comparison images')
    for i, acc in sorted(accs.items(), key=lambda x: x[0]):
        print(f'{i} comparison image(s): {acc*100:.2f}% ({correct:,}/{chances:,})')
    print('=' * 50, '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups', default='./data/matchups.csv')
    parser.add_argument('--train-hashes', default='./hashes/train_hashes.txt')
    parser.add_argument('--data', default='./data/images')
    parser.add_argument('--img-kind', default='pitcher')
    parser.add_argument('--batch-size', default=16)
    parser.add_argument('--input-dim', default=3)
    parser.add_argument('--output-dim', default=32)
    parser.add_argument('--layer-type', default='residual')
    parser.add_argument('--blocks', default=2)
    parser.add_argument('--filters', default=32)
    parser.add_argument('--kernel-size', default=3)
    parser.add_argument('--strides', default=1)
    parser.add_argument('--weights', default='./weights/siamese.weights.h5')
    args = parser.parse_args()

    matchups = pd.read_csv(args.matchups)
    img_loader = ImgLoader(matchups=matchups, kind=args.img_kind)
    pitchers = img_loader.load(data_dir=args.data_dir)

    siamese = SiameseNetwork(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        layer_type=args.layer_type,
        n_blocks=args.blocks,
        filters=args.filters,
        kernel_size=args.kernel_size,
        strides=args.strides
    )
    siamese.load_weights(args.weight_path)

    
    train_hashes = load_img_hashes(args.train_hashes)
    cls_imgs, cls_labels = build_classification_set(
        players=pitchers,
        player_set=img_loader.pitchers,
        train_hashes=train_hashes,
        comp_size=5,
    )

    output_types = ({'x1': tf.float32, 'x2': tf.float32}, tf.int32)
    output_shapes = ({'x1': (216, 216, 3), 'x2': (216, 216, 3)}, ())
    img_ds, img_gen = make_tf_dataset(
        pairs=cls_imgs,
        labels=[0] * len(cls_imgs),
        output_types=output_types,
        output_shapes=output_shapes,
        batch_size=64,
        repeats=1,
        training=False,
        return_gen=True
    )
    preds = siamese.predict(img_ds, steps=img_gen.steps)

    # At least k comparison images accuracy
    run_eval(mode='atleast', preds=preds, cls_labels=cls_labels, top_k=1)

    # At least k comparison images top-5 accuracy
    run_eval(mode='atleast', preds=preds, cls_labels=cls_labels, top_k=5)

    # Exactly k comparison images accuracy
    run_eval(mode='exactly', preds=preds, cls_labels=cls_labels, top_k=1)

    # Exactly k comparison images top-5 accuracy
    run_eval(mode='exactly', preds=preds, cls_labels=cls_labels, top_k=5)
