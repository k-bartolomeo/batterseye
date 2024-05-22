import random

import pandas as pd
import mlbstatsapi as sa

from tqdm import tqdm


def get_game_ids(team_ids: list) -> list:
    """Gets game IDs from 2023 season for given teams"""
    game_ids = []
    for team_id in team_ids:
        schedule = sa.schedule(
            start_date='04/01/2023', end_date='11/01/2023', team=team_id
        )
        team_game_ids = [x['game_id'] for x in schedule]
        game_ids.extend(team_game_ids)
    return game_ids


def get_batter(play: dict) -> dict:
    """Gets batter information from given play"""
    matchup_info = play['matchup']
    batter_info = matchup_info['batter']
    return {
        'player_id': batter_info['id'],
        'name': batter_info['fullName'],
        'bats': matchup_info['batSide']['code']
    }


def get_pitcher(play: dict) -> dict:
    """Gets pitcher information from given play"""
    matchup_info = play['matchup']
    pitcher_info = matchup_info['pitcher']
    return {
        'player_id': pitcher_info['id'],
        'name': pitcher_info['fullName'],
        'throws': matchup_info['pitchHand']['code']
    }


def get_matchup(play: dict) -> dict:
    """Wrapper for getting batter and pitcher information"""
    batter = get_batter(play)
    pitcher = get_pitcher(play)
    return {
        'batter': batter,
        'pitcher': pitcher
    }


def pick_play_id(play: dict) -> dict:
    """Randomly selects play ID from given play

    MLB Stats API defines each at-bat as a 'play'. Within
    the 'play', there are a number of pitches, each of which is
    assigned its own 'play ID'.

    Parameters
    ----------
    play
        Dictionary containing at-bat data for a single at-bat.
    """
    play_event_ids = [
        event.get('playId', None) for event in play['playEvents']
    ]
    chosen_event = random.choice([
        event for event in play_event_ids if event is not None
    ])
    return {'play_id': chosen_event}


def get_metadata(play: dict) -> dict:
    """Wrapper for getting matchup and play ID data"""
    matchup = get_matchup(play)
    play_id = pick_play_id(play)
    matchup.update(play_id)
    return matchup


def get_all_metadata(play: dict) -> list[dict]:
    """Gets lists of metadata for all sub-events in given play"""
    matchup = get_matchup(play)
    play_ids = [event.get('playId', None) for event in play['playEvents']]

    # The `|` operator updates the dictionary containing 
    # play ID information with matchup information.
    return [
        {'play_id': play_id} | matchup for play_id in play_ids
    ]


def get_matchups(game_ids: list[int]) -> list[dict]:
    """Gets matchup metadata for all plays in given games
    
    Uses given list of game IDs to retrieve play-by-play (at-bat-level)
    data for each game. Extracts list of plays from play-by-play data
    and then retrieves dictionaries of metadata for each play and its
    sub-events.

    Parameters
    ----------
    game_ids
        List of game IDs for games from which matchup and play ID data
        is downloaded.

    Returns
    -------
    list[dict]
        List of dictionaries containing metadata for all plays in all
        given games.
    """
    matchups = []
    for game_id in tqdm(game_ids):
        play_by_play = sa.get('game_playByPlay', {'gamePk': game_id})
        all_plays = play_by_play['allPlays']
        for play in all_plays:
            matchups.extend(get_all_metadata(play))

    return matchups


def matchups_to_df(matchups: list[dict]) -> pd.DataFrame:
    """Converts metadata list to pandas DataFrame"""
    def reformat_matchup(m: dict) -> dict:
        """Reformats matchup structure for pandas DataFrame"""
        batter_id = m['batter']['player_id']
        batter_name = m['batter']['name']
        pitcher_id = m['pitcher']['player_id']
        pitcher_name = m['pitcher']['name']
        return {
            'play_id': m['play_id'],
            'batter_id': batter_id,
            'batter_name': batter_name,
            'bats': m['batter']['bats'],
            'pitcher_id': pitcher_id,
            'pitcher_name': pitcher_name,
            'throws': m['pitcher']['throws']
        }
    
    reformatted = [reformat_matchup(m) for m in matchups]
    df = pd.DataFrame(reformatted)
    df = df.drop_duplicates().reset_index(drop=True)
    return df