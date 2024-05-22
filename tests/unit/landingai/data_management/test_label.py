import responses

from landingai.data_management.label import Label


@responses.activate
def test_get_label_map():
    responses._add_from_file(file_path="tests/data/responses/test_get_label_map.yaml")
    project_id = 34243219343364
    api_key = "land_sk_12345"
    client = Label(project_id, api_key)
    res = client.get_label_map()
    assert res == {
        "0": "ok",
        "1": "num_plate",
        "2": "number_plate",
    }


@responses.activate
def test_post_label_map():
    responses._add_from_file(file_path="tests/data/responses/test_get_label_map.yaml")
    project_id = 34243219343364
    api_key = "land_sk_12345"
    client = Label(project_id, api_key)
    label_map = {"1": {"name": "num_plate"}, "2": {"name": "number_plate"}}
    res = client.create_label_map(label_map)
    assert res == {
        "0": "ok",
        "1": "num_plate",
        "2": "number_plate",
    }
