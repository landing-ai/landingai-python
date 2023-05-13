import numpy as np

from landingai.common import SegmentationPrediction, decode_bitmap_rle


def test_decode_bitmap_rle():
    encoded_mask = "2N3Z2N5Z"
    encoding_map = {"Z": 0, "N": 1}
    decoded_mask = decode_bitmap_rle(encoded_mask, encoding_map)
    assert decoded_mask == [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_segmentation_prediction_get():
    encoded_mask = "2N3Z2N5Z"
    encoding_map = {"Z": 0, "N": 1}
    label_index = 3
    prediction = SegmentationPrediction(
        id="123",
        label_index=label_index,
        label_name="class1",
        score=0.5,
        encoded_mask=encoded_mask,
        encoding_map=encoding_map,
        mask_shape=(3, 4),
    )
    expected = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]).reshape((3, 4))
    np.testing.assert_array_almost_equal(prediction.decoded_boolean_mask, expected)
    np.testing.assert_array_almost_equal(
        prediction.decoded_index_mask, expected * label_index
    )
