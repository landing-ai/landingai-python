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
    img = visualize.overlay_predictions(
        [preds[1]],
        img,
        {
            "bbox_style": "FLAG",
        },
    )
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
            "text": "\\u516c\\u53f8\\u540d\\u79f0",
            "text_location": [[99, 19], [366, 19], [366, 75], [99, 75]],
        },
        {
            "score": 0.9647054672241211,
            "text": "\\u4e1a\\u52a1\\u65b9\\u5411",
            "text_location": [[577, 21], [835, 21], [835, 77], [577, 77]],
        },
        {
            "score": 0.8744988441467285,
            "text": "Anysphere",
            "text_location": [[10, 108], [256, 108], [256, 157], [10, 157]],
        },
        {
            "score": 0.9655991792678833,
            "text": "AI\\u5de5\\u5177",
            "text_location": [[625, 105], [785, 105], [785, 155], [625, 155]],
        },
        {
            "score": 0.9070963263511658,
            "text": "Atomic Semi",
            "text_location": [[10, 181], [310, 181], [310, 226], [10, 226]],
        },
        {
            "score": 0.7428176999092102,
            "text": "\\u82af\\u7247",
            "text_location": [[652, 173], [763, 176], [761, 232], [650, 229]],
        },
        {
            "score": 0.9924920201301575,
            "text": "Cursor",
            "text_location": [[10, 258], [160, 258], [160, 302], [10, 302]],
        },
        {
            "score": 0.906830906867981,
            "text": "\\u4ee3\\u7801\\u7f16\\u8f91",
            "text_location": [[597, 253], [817, 253], [817, 302], [597, 302]],
        },
        {
            "score": 0.922673225402832,
            "text": "Diagram",
            "text_location": [[10, 329], [207, 332], [206, 383], [9, 379]],
        },
        {
            "score": 0.9909729361534119,
            "text": "\\u8bbe\\u8ba1",
            "text_location": [[649, 323], [767, 321], [769, 378], [651, 381]],
        },
        {
            "score": 0.9989902973175049,
            "text": "Harvey",
            "text_location": [[6, 403], [172, 407], [171, 457], [5, 452]],
        },
        {
            "score": 0.9266898036003113,
            "text": "AI\\u6cd5\\u5f8b\\u987e\\u95ee",
            "text_location": [[575, 402], [835, 402], [835, 448], [575, 448]],
        },
        {
            "score": 0.9164015054702759,
            "text": "Kick",
            "text_location": [[8, 476], [105, 476], [105, 525], [8, 525]],
        },
        {
            "score": 0.9329562187194824,
            "text": "\\u4f1a\\u8ba1\\u8f6f\\u4ef6",
            "text_location": [[597, 474], [817, 474], [817, 523], [597, 523]],
        },
        {
            "score": 0.9220211505889893,
            "text": "Milo",
            "text_location": [[6, 551], [112, 551], [112, 600], [6, 600]],
        },
        {
            "score": 0.9562898278236389,
            "text": "\\u5bb6\\u957f\\u865a\\u62df\\u52a9\\u7406",
            "text_location": [[545, 550], [870, 550], [870, 595], [545, 595]],
        },
        {
            "score": 0.9696773886680603,
            "text": "qqbot.dev",
            "text_location": [[10, 630], [247, 627], [248, 672], [11, 676]],
        },
        {
            "score": 0.971221923828125,
            "text": "\\u5f00\\u53d1\\u8005\\u5de5\\u5177",
            "text_location": [[570, 624], [842, 624], [842, 669], [570, 669]],
        },
        {
            "score": 0.998053789138794,
            "text": "EdgeDB",
            "text_location": [[8, 700], [195, 697], [196, 748], [9, 750]],
        },
        {
            "score": 0.9924381971359253,
            "text": "\\u5f00\\u6e90\\u6570\\u636e\\u5e93",
            "text_location": [[571, 698], [842, 698], [842, 742], [571, 742]],
        },
        {
            "score": 0.9033893346786499,
            "text": "Mem Labs",
            "text_location": [[10, 776], [250, 773], [251, 816], [10, 818]],
        },
        {
            "score": 0.9257971048355103,
            "text": "\\u7b14\\u8bb0\\u5e94\\u7528",
            "text_location": [[598, 771], [816, 771], [816, 819], [598, 819]],
        },
        {
            "score": 0.8411136865615845,
            "text": "Speak",
            "text_location": [[7, 848], [154, 844], [156, 896], [8, 899]],
        },
        {
            "score": 0.9975636601448059,
            "text": "\\u82f1\\u8bed\\u5b66\\u4e60",
            "text_location": [[599, 842], [814, 845], [814, 894], [599, 892]],
        },
        {
            "score": 0.9817236661911011,
            "text": "Descript",
            "text_location": [[9, 920], [202, 920], [202, 967], [9, 967]],
        },
        {
            "score": 0.9847884178161621,
            "text": "\\u97f3\\u89c6\\u9891\\u7f16\\u8f91\\u5b50\\u4f4d",
            "text_location": [[571, 919], [916, 911], [917, 963], [572, 970]],
        },
    ]
    preds = [OcrPrediction(**pred) for pred in json_preds]
    result_img = visualize.overlay_predictions(preds, img)
    expected = PIL.Image.open("tests/data/images/expected_ocr_overlay.png")
    diff = PIL.ImageChops.difference(result_img, expected)
    assert diff.getbbox() is None, "Expected and actual images should be the same"
