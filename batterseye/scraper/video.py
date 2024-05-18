import os
import requests
import threading

from moviepy.editor import VideoFileClip


def get_play_id(play_event: dict) -> str | None:
    """Retrieves play ID from play event dictionary

    Originally, project intended to model the exit velocity of the 
    baseball off the batter's bat. As such, only videos of balls in
    play were downloaded.

    Parameters
    ----------
    play_event
        Dictionary containing play event details. Play event defined
        as an individual instance of everything that happened during
        a given at-bat.

    Returns
    -------
    str
        Either the playId if the ball was put into play during the 
        given play event, or `None` if the ball was not put into
        play.    
    """
    if 'details' in play_event:
        if 'isInPlay' in play_event['details']:
            if play_event['details']['isInPlay'] is True:
                if 'playId' in play_event:
                    return play_event['playId']
    return None


def process_video(path: str) -> None:
    """Postprocesses downloaded video
    
    Trims downloaded clip to first 4.5 seconds of clip, since that
    is the time frame of interest. Cuts resolution of video in half
    to save on storage, and also removes audio from clip.

    Parameters
    ----------
    path
        Path to downloaded raw MP4 file.
    """
    success = False
    clip = VideoFileClip(path)
    no_audio = clip.without_audio()
    trimmed = no_audio.subclip(0, 4.5)
    resized = trimmed.resize(newsize=0.5)
    pardir, fname = os.path.split(path)
    fname = fname.rstrip('.mp4')
    fname = f'{fname}_edit.mp4'
    out_path = os.path.join(pardir, fname)
    try:
        resized.write_videofile(out_path, logger=None, verbose=False)
        success = True
    except Exception as e:
        print(e)

    if success:
        os.remove(path)


def get_video_threaded(
    play_id: str, video_url: str, download_dir: str, semaphore: threading.Semaphore
) -> None:
    """Downloads video via multi-threading"""
    with semaphore:
        get_video(play_id=play_id, video_url=video_url, download_dir=download_dir)


def get_video(play_id: str, video_url: str, download_dir: str) -> None:
    """Downloads video from given URL

    Requests video from given video URL and downloads to given 
    directory as MP4 file. Postprocesses video using `process_video`
    function.

    Parameters
    ----------
    play_id
        Play ID for the video being downloaded.
    video_url
        URL at which MP4 is stored online.
    download_dir
        Directory to which raw MP4 is downloaded.
    semaphore
        threading.Semaphore object limiting the number of downloads 
        taking place at once.
    """
    path = os.path.join(download_dir, f'{play_id}.mp4')
    edit_path = os.path.join(download_dir, f'{play_id}_edit.mp4')
    current_files = set(os.listdir(download_dir))
    if path not in current_files and edit_path not in current_files:
        r = requests.get(video_url, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1_024**2):
                if chunk:
                    f.write(chunk)
        f.close()

    process_video(path)
