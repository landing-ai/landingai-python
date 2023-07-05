import numpy as np
import PIL.Image

from landingai import visualize
from landingai.common import ObjectDetectionPrediction, OcrPrediction


def test_overlay_bboxes():
    img = PIL.Image.open("tests/data/images/cereal1.jpeg")
    preds = [
        ObjectDetectionPrediction(
            id="1",
            label_index=0,
            label_name="screw",
            score=0.623112,
            bboxes=(432, 1035, 651, 1203),
        ),
        ObjectDetectionPrediction(
            id="2",
            label_index=0,
            label_name="screw",
            score=0.892,
            bboxes=(1519, 1414, 1993, 1800),
        ),
        ObjectDetectionPrediction(
            id="3",
            label_index=0,
            label_name="screw",
            score=0.7,
            bboxes=(948, 1592, 1121, 1797),
        ),
    ]

    img = visualize.overlay_predictions(
        [preds[0]], img, {"bbox_style": "default", "draw_label": False}
    )
    img = visualize.overlay_predictions([preds[1]], img, {"bbox_style": "FLAG",},)
    result_img = visualize.overlay_predictions(
        [preds[2]], img, {"bbox_style": "t-label"}
    )
    assert result_img.size == (2048, 2048)
    expected = PIL.Image.open("tests/data/images/expected_bbox_overlay.png")
    diff = PIL.ImageChops.difference(result_img, expected)
    assert diff.getbbox() is None, "Expected and actual images should be the same"


def test_overlay_ocr_predition():
    img = PIL.Image.open("tests/data/images/ocr_test.png")
    json_preds = [
        {
            "score": 0.861878514289856,
            "text": "公司名称",
            "location": [[99, 19], [366, 19], [366, 75], [99, 75]],
        },
        {
            "score": 0.9647054672241211,
            "text": "业务方向",
            "location": [[577, 21], [835, 21], [835, 77], [577, 77]],
        },
        {
            "score": 0.8744988441467285,
            "text": "Anysphere",
            "location": [[10, 108], [256, 108], [256, 157], [10, 157]],
        },
        {
            "score": 0.9655991792678833,
            "text": "AI工具",
            "location": [[625, 105], [785, 105], [785, 155], [625, 155]],
        },
        {
            "score": 0.9070963263511658,
            "text": "Atomic Semi",
            "location": [[10, 181], [310, 181], [310, 226], [10, 226]],
        },
        {
            "score": 0.7428176999092102,
            "text": "芯片",
            "location": [[652, 173], [763, 176], [761, 232], [650, 229]],
        },
        {
            "score": 0.9924920201301575,
            "text": "Cursor",
            "location": [[10, 258], [160, 258], [160, 302], [10, 302]],
        },
        {
            "score": 0.906830906867981,
            "text": "代码编辑",
            "location": [[597, 253], [817, 253], [817, 302], [597, 302]],
        },
        {
            "score": 0.922673225402832,
            "text": "Diagram",
            "location": [[10, 329], [207, 332], [206, 383], [9, 379]],
        },
        {
            "score": 0.9909729361534119,
            "text": "设计",
            "location": [[649, 323], [767, 321], [769, 378], [651, 381]],
        },
        {
            "score": 0.9989902973175049,
            "text": "Harvey",
            "location": [[6, 403], [172, 407], [171, 457], [5, 452]],
        },
        {
            "score": 0.9266898036003113,
            "text": "AI法律顾问",
            "location": [[575, 402], [835, 402], [835, 448], [575, 448]],
        },
        {
            "score": 0.9164015054702759,
            "text": "Kick",
            "location": [[8, 476], [105, 476], [105, 525], [8, 525]],
        },
        {
            "score": 0.9329562187194824,
            "text": "会计软件",
            "location": [[597, 474], [817, 474], [817, 523], [597, 523]],
        },
        {
            "score": 0.9220211505889893,
            "text": "Milo",
            "location": [[6, 551], [112, 551], [112, 600], [6, 600]],
        },
        {
            "score": 0.9562898278236389,
            "text": "家长虚拟助理",
            "location": [[545, 550], [870, 550], [870, 595], [545, 595]],
        },
        {
            "score": 0.9696773886680603,
            "text": "qqbot.dev",
            "location": [[10, 630], [247, 627], [248, 672], [11, 676]],
        },
        {
            "score": 0.971221923828125,
            "text": "开发者工具",
            "location": [[570, 624], [842, 624], [842, 669], [570, 669]],
        },
        {
            "score": 0.998053789138794,
            "text": "EdgeDB",
            "location": [[8, 700], [195, 697], [196, 748], [9, 750]],
        },
        {
            "score": 0.9924381971359253,
            "text": "开源数据库",
            "location": [[571, 698], [842, 698], [842, 742], [571, 742]],
        },
        {
            "score": 0.9033893346786499,
            "text": "Mem Labs",
            "location": [[10, 776], [250, 773], [251, 816], [10, 818]],
        },
        {
            "score": 0.9257971048355103,
            "text": "笔记应用",
            "location": [[598, 771], [816, 771], [816, 819], [598, 819]],
        },
        {
            "score": 0.8411136865615845,
            "text": "Speak",
            "location": [[7, 848], [154, 844], [156, 896], [8, 899]],
        },
        {
            "score": 0.9975636601448059,
            "text": "英语学习",
            "location": [[599, 842], [814, 845], [814, 894], [599, 892]],
        },
        {
            "score": 0.9817236661911011,
            "text": "Descript",
            "location": [[9, 920], [202, 920], [202, 967], [9, 967]],
        },
        {
            "score": 0.9847884178161621,
            "text": "音视频编辑",
            "location": [[571, 919], [916, 911], [917, 963], [572, 970]],
        },
    ]
    preds = [OcrPrediction(**pred) for pred in json_preds]
    result_img = visualize.overlay_predictions(preds, img)
    expected = PIL.Image.open("tests/data/images/expected_ocr_overlay.png")
    result_img_np = np.asarray(result_img)
    expected_np = np.asarray(expected)
    diff_cnt = np.count_nonzero(result_img_np - expected_np)
    assert (
        diff_cnt / np.prod(result_img_np.shape) < 0.01
    ), f"The percentage of different pixels ({diff_cnt / np.prod(result_img_np.shape)}) is greater than 1%."
