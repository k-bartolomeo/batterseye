import os
import time
import random
import requests
import threading

import cv2
import numpy as np
import numpy.typing as npt

from tqdm import tqdm
from bs4 import BeautifulSoup


def write_images(
    matchups: list[dict], 
    output_dir: str,
    img_size: tuple[int, int] = (360, 360),
    n_threads: int = 50
) -> None:
    """Retrieves images for given matchups
    
    Baseball Savant website has webpage dedicated to each play from
    each MLB game. URL for play webpage is standardized, with only
    changing component's being the play ID for the given play. Link
    to MP4 embedded within webpage exists within

    Parameters
    ----------
    matchups
        List of dictionaries with play metadata, including play IDs.
    output_dir
        Directory where images are saved.
    img_size
        Height and width of image array outputs.
    n_threads
        Number of threads running at a single time during download.
    """
    semaphore = threading.Semaphore(n_threads)
    count = 0
    total = len(matchups)

    def get_mp4_url(video_url: str) -> str:
        """Retrieves MP4 download URL from given webpage

        Retrieves HTML from webpage in which video of play is
        embedded. Parses HTML using BeautifulSoup and extracts
        <source> tag containing URL from which MP4 can be read.

        Parameters
        ----------
        video_url
            URL to Baseball Savant page with embedded MP4.

        Returns
        -------
        str
            MP4 URL from which play video can be read.
        """
        r = requests.get(video_url)
        soup = BeautifulSoup(r.content, 'html.parser')
        video_links = [
            source
            for source in soup.find_all('source')
            if hasattr(source, 'type') and source['type'] == 'video/mp4'
        ]
        assert len(video_links) == 1
        return video_links[0]['src']

    def get_random_image(
        mp4_url: str, img_size: tuple[int, int]
    ) -> npt.NDArray | None:
        """Gets random image from video
        
        Uses cv2 to read video directly from MP4 URL. Selects random
        index from range 0 to 100, restricting frames that can be 
        selected to the first hundred where the pitcher and batter
        are in clear view.

        Parameters
        ----------
        mp4_url
            URL to MP4 of play from which still iamge is retrieved.

        img_size
            Height and width of the returned image.

        Returns
        -------
        npt.NDArray | None
            Array of pixel values with shape (img_size[0], img_size[1], 3)
            if image can be retrieved from MP4 URL. If image cannot be
            retrieved, returns `None`.
        """
        vidcap = cv2.VideoCapture(mp4_url)
        success, image = vidcap.read()
        image = cv2.resize(image, img_size)
        idx = random.choice(range(100))
        if idx == 0:
            return image
        count = 1

        while success:
            success, image = vidcap.read()

            # If `count` matches the randomly chosen index,
            # resize and return the image at that index.
            if idx == count:
                image = cv2.resize(image, img_size)
                return image
            count += 1

        return None
    
    def download_image(
        matchup: dict, img_size: tuple[int, int]
    ) -> npt.NDArray | None:
        """Wrapper for downloading random image from given matchup"""
        base = 'https://baseballsavant.mlb.com/sporty-videos?playId={:}'
        video_url = base.format(matchup['play_id'])
        mp4_url = get_mp4_url(video_url)
        img = get_random_image(mp4_url, img_size=img_size)
        return img
    
    def write_image(
        matchup: dict, output_dir: str, img_size: tuple[int, int]
    ) -> None:
        """Writes downloaded image array to .npy file"""
        with semaphore:
            img = download_image(matchup, img_size=img_size)
            fname = f"{matchup['play_id']}.npy"
            path = os.path.join(output_dir, fname)
            np.save(path, img)

    for matchup in tqdm(matchups):
        # `waiting` flag used to limit the number of
        # threads kicked off because of limited resources
        waiting = True
        while waiting is True:
            if threading.active_count() < 1_000:
                t = threading.Thread(
                    target=write_image, args=(matchup, output_dir, img_size)
                )
                t.start()
                waiting = False
                count += 1
            else:
                time.sleep(1)

    # wait until images from all matchups have been downloaded
    while count < total:
        time.sleep(1)
