import argparse

from ..batterseye.scraper.images import write_images
from ..batterseye.scraper.matchups import get_game_ids, get_matchups, matchups_to_df


if __name__ == '__main__':
    """Builds dataset of images and matchup metadata"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--team-ids', default='147,146,121,111')
    parser.add_argument('--matchup-path', default='./data/matchups.dsv')
    parser.add_argument('--img-dir', default='./data/images')
    parser.add_argument('--img-size', default=360)
    parser.add_argument('--threads', default=50)
    args = parser.parse_args()

    team_ids = args.team_ids.split(',')
    game_ids = get_game_ids(team_ids=team_ids)
    matchups = get_matchups(game_ids=game_ids)
    matchup_df = matchups_to_df(matchups=matchups)
    matchup_df.to_csv(args.matchup_path, index=False)
    write_images(
        matchups=matchups,
        output_dir=args.img_dir,
        img_size=(args.img_size, args.img_size),
        n_threads=args.threads
    )