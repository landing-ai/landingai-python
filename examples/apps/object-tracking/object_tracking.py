import cv2
import time
import numpy as np
import numpy.typing as npt

from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
from landingai.predict import Predictor


CROP_BOX = (1300, 600, 1700, 800)


def bbox_ious(a, b):
    # based off of torchvision's box_iou https://pytorch.org/vision/main/generated/torchvision.ops.box_iou.html
    def box_area(a):
        return (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])

    def inter_union(a, b):
        area1 = box_area(a)
        area2 = box_area(b)

        lt = np.maximum(a[:, None, :2], b[:, :2])  # [N, M, 2]
        rb = np.minimum(a[:, None, 2:], b[:, 2:])  # [N, M, 2]

        wh = (rb - lt).clip(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter
        return inter, union

    inter, union = inter_union(a, b)
    iou = inter / union
    return iou


def crop(image: npt.NDArray, box: Tuple[int, ...] = CROP_BOX) -> npt.NDArray:
    return image[box[1] : box[3], box[0] : box[2]]


def plot_boxes_on_image(
    image: npt.NDArray, boxes: List[Tuple[int, ...]]
) -> npt.NDArray:
    for box in boxes:
        image = cv2.rectangle(image, box[0:2], box[2:4], (0, 255, 0), 2)
    return image


def plot_boxes_on_image_with_ids(
    image: npt.NDArray,
    boxes: List[Tuple[int, ...]],
    ids: Dict[int, int],
    crop_box: Optional[Tuple[int, ...]] = None,
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
    frames: List[npt.NDArray], predictor: Predictor
) -> List[List[Tuple[int, ...]]]:
    bboxes = []
    for frame in tqdm(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = crop(frame)
        pred = predictor.predict(frame)
        bboxes.append([List(p.bboxes) + [p.score] for p in pred])
    return bboxes


def match_dets_with_prev(
    prev_detections: List[Tuple[int, ...]],
    detections: List[Tuple[int, ...]],
) -> Tuple[Dict[int, int], Set[int]]:
    iou = bbox_ious(np.array(prev_detections)[:, :4], np.array(detections)[:, :4])
    prev_det_to_det = iou.argmax(axis=1)
    matches = {}
    unmatched = []
    for prev_det_idx, det_idx in enumerate(prev_det_to_det):
        if iou[prev_det_idx, det_idx] != 0.0:
            matches[prev_det_idx] = det_idx
    unmatched = set([i for i in range(len(detections))]) - set(matches.values())
    return matches, unmatched


def track_iou(
    bboxes: List[List[Tuple[int, ...]]],
    display: bool = False,
    frames: Optional[List[npt.NDArray]] = None,
) -> Tuple[Dict[int, List[Tuple[int, ...]]], List[Dict[int, int]]]:
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
    tracks: Dict[int, List[Tuple[int, ...]]], total_frames: int
) -> Tuple[Dict[int, List[Tuple[int, ...]]], int]:
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
    tracks: Dict[int, List[Tuple[int, ...]]],
) -> Tuple[Dict[int, List[Tuple[int, ...]]], int]:
    tracks_ = tracks.copy()
    n = 0
    for track_id in tracks:
        if len(tracks[track_id]) == 1:
            tracks_.pop(track_id)
            n += 1
    return tracks_, n


def get_northbound_southbound(
    tracks: Dict[int, List[Tuple[int, ...]]]
) -> Tuple[List[int], List[int]]:
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
    frames: List[npt.NDArray],
    bboxes: List[List[Tuple[int, ...]]],
    idx_to_track: List[Dict[int, int]],
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
