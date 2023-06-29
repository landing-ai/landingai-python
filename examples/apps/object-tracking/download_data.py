import argparse
from pathlib import Path
from typing import Callable, List

import cv2
import m3u8
import numpy.typing as npt
import requests

M3U8_URL = (
    "https://live.hdontap.com/hls/hosb1/sunset-static_swellmagenet.stream/playlist.m3u8"
)
TS_URL = (
    "https://edge06.nginx.hdontap.com/hosb1/sunset-static_swellmagenet.stream/media_"
)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--out_dir", type=str, default="data")
arg_parser.add_argument("--video_file", type=str, default="vid1")
args = arg_parser.parse_args()


def get_latest_ts_file(out_path: str) -> None:
    if Path(out_path).suffix != ".ts":
        raise ValueError(f"Must be a .ts file, got {out_path}")
    r = requests.get(M3U8_URL)
    m3u8_r = m3u8.loads(r.text)
    playlist_uri = m3u8_r.data["playlists"][0]["uri"]
    r = requests.get(playlist_uri)
    m3u8_r = m3u8.loads(r.text)
    media_sequence = m3u8_r.data["media_sequence"]
    ts_file = requests.get(TS_URL + str(media_sequence) + ".ts")
    with open(out_path, "wb") as f:
        f.write(ts_file.content)


def get_frames(video_file: str, skip_frame: int = 5) -> List[npt.NDArray]:
    cap = cv2.VideoCapture(video_file)
    frames = []
    i = 0
    read, frame = cap.read()
    while read:
        if i % skip_frame == 0 and frame is not None:
            frames.append(frame)
        read, frame = cap.read()
        i += 1
    return frames


def write_frames(video_file: str, out_dir: str) -> List[str]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    frames = get_frames(video_file)
    video_file_name = Path(video_file).name
    output_files = []
    for i, frame in enumerate(frames):
        file_name = str(out_dir_p / video_file_name) + ".frame" + str(i) + ".jpg"
        cv2.imwrite(file_name, frame)
        output_files.append(file_name)

    return output_files


def crop_data(
    input_files: List[str], crop: Callable[[npt.NDArray], npt.NDArray]
) -> None:
    for f in input_files:
        img = cv2.imread(f)
        img = crop(img)
        cv2.imwrite(f, img)


if __name__ == "__main__":
    video_file = str((Path(args.out_dir) / Path(args.video_file)).with_suffix(".ts"))
    get_latest_ts_file(video_file)
    files = write_frames(video_file, args.out_dir)
    crop_data(files, lambda x: x[600:800, 1300:1700])
