import cv2
import time
import numpy as np
import numpy.typing as npt

from typing import Optional
from tqdm import tqdm
from landingai.predict import Predictor


CROP_BOX = (1300, 600, 1700, 800)


def crop(image: npt.NDArray, box: tuple[int, ...] = CROP_BOX) -> npt.NDArray:
    return image[box[1] : box[3], box[0] : box[2]]


def plot_boxes_on_image(
    image: npt.NDArray, boxes: list[tuple[int, ...]]
) -> npt.NDArray:
    for box in boxes:
        image = cv2.rectangle(image, box[0:2], box[2:4], (0, 255, 0), 2)
    return image


def plot_boxes_on_image_with_ids(
    image: npt.NDArray,
    boxes: list[tuple[int, ...]],
    ids: dict[int, int],
    crop_box: Optional[tuple[int, ...]] = None,
) -> npt.NDArray:
    for i, box in enumerate(boxes):
        if crop_box is not None:
            box = (
                box[0] + crop_box[0],
                box[1] + crop_box[1],
                box[2] + crop_box[0],
                box[3] + crop_box[1],
            )
        image = cv2.rectangle(image, box[0:2], box[2:4], (0, 255, 0), 2)
        image = cv2.putText(
            image,
            str(ids.get(i, None)),
            box[0:2],
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def get_preds(
    frames: list[npt.NDArray], predictor: Predictor
) -> list[list[tuple[int, ...]]]:
    bboxes = []
    for frame in tqdm(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = crop(frame)
        pred = predictor.predict(frame)
        bboxes.append([list(p.bboxes) + [p.score] for p in pred])
    return bboxes


def match_dets_with_prev(
    prev_detections: list[tuple[int, ...]],
    detections: list[tuple[int, ...]],
) -> tuple[dict[int, int], set[int]]:
    iou = bbox_ious(np.array(prev_detections)[:, :4], np.array(detections)[:, :4])
    prev_det_to_det = iou.argmax(axis=1)
    matches = {}
    unmatched = []
    for prev_det_idx, det_idx in enumerate(prev_det_to_det):
        if iou[prev_det_idx, det_idx] != 0.0:
            matches[prev_det_idx] = det_idx
    unmatched = set([i for i in range(len(detections))]) - set(prev_det_to_det)
    return matches, unmatched


def track_iou(
    bboxes: list[list[tuple[int, ...]]],
    display: bool = False,
    frames: Optional[list[npt.NDArray]] = None,
) -> tuple[dict[int, list[tuple[int, ...]]], list[dict[int, int]]]:
    all_idx_to_track = []
    idx_to_track = {}
    tracks = {}
    max_id = 0
    if frames is None:
        frames = [np.zeros((1,)) for _ in range(len(bboxes))]
    for i, (frame, boxes) in enumerate(zip(frames, bboxes)):
        next_idx_to_track = {}
        if i == 0:
            for j in range(len(boxes)):
                idx_to_track[j] = j
                tracks[j] = [boxes[j]]
                max_id = j
        else:
            matches, unmatched = match_dets_with_prev(bboxes[i - 1], boxes)
            for idx in matches:
                next_idx_to_track[matches[idx]] = idx_to_track[idx]
                tracks[idx_to_track[idx]].append(boxes[matches[idx]])
            for idx in unmatched:
                max_id += 1
                next_idx_to_track[idx] = max_id
                tracks[max_id] = [boxes[idx]]
            idx_to_track = next_idx_to_track

        all_idx_to_track.append(idx_to_track)
        if display:
            frame = crop(frame)
            cv2.imshow(
                "video", plot_boxes_on_image_with_ids(frame, boxes, next_idx_to_track)
            )
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
            time.sleep(0.5)

    return tracks, all_idx_to_track


def filter_parked_cars(
    tracks: dict[int, list[tuple[int, ...]]], total_frames: int
) -> tuple[dict[int, list[tuple[int, ...]]], int]:
    tracks_ = tracks.copy()
    n = 0
    for track_id in tracks:
        if (
            len(tracks[track_id]) > total_frames * 0.8
            and abs(tracks[track_id][0][1] - tracks[track_id][-1][1]) < 50
        ):
            tracks_.pop(track_id)
            n += 1
    return tracks_, n


def filter_spurious_preds(
    tracks: dict[int, list[tuple[int, ...]]],
) -> tuple[dict[int, list[tuple[int, ...]]], int]:
    tracks_ = tracks.copy()
    n = 0
    for track_id in tracks:
        if len(tracks[track_id]) == 1:
            tracks_.pop(track_id)
            n += 1
    return tracks_, n


def get_northbound_southbound(
    tracks: dict[int, list[tuple[int, ...]]]
) -> tuple[list[int], list[int]]:
    northbound = []
    southbound = []
    for track_id in tracks:
        boxes = tracks[track_id]
        if boxes[0][1] < boxes[-1][1]:
            southbound.append(track_id)
        else:
            northbound.append(track_id)
    return northbound, southbound


def write_video(
    frames: list[npt.NDArray],
    bboxes: list[list[tuple[int, ...]]],
    idx_to_track: list[dict[int, int]],
    video_file: str,
) -> None:
    writer = cv2.VideoWriter(
        video_file, cv2.VideoWriter_fourcc(*"avc1"), 3, frames[0].shape[:2][::-1]
    )
    for i, (frame, boxes) in enumerate(zip(frames, bboxes)):
        # frame = crop(frame)
        writer.write(
            plot_boxes_on_image_with_ids(frame, boxes, idx_to_track[i], CROP_BOX)
        )
    writer.release()
