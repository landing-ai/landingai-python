## Data management

The `landingai` library provides a set of APIs to manage dataset medias (e.g. images) and their metadata.
This section explains how to use the associated Python APIs to:

-   Upload new (single) media with metadata
-   Upload all medias from folder with metadata
-   Set metadata for existing medias


### Introduction to Metadata

Metadata is additional information you can attach to a media (e.g. image). Every media can be associated with multiple metadata. Each metadata is a key-value pair associated with this media, where key is the name of this metadata and value is a string that represents this information.
For example, when you upload an image to LandingLens, you can add metadata like the country where the image was created, the timestamp when the image was created, etc.

Metadata is useful when you need to manage hundreds or thousands of medias on LandingLens or you need to collaborate with other team members to label this dataset.
For example, you can use metadata to group certain type of medias together (e.g. images taken last week), then change their the split key, or create a [labeling task](https://support.landing.ai/landinglens/docs/agreement-based-labeling#send-labeling-tasks) out of those medias.

You can use the `landingai.data_management.metadata.Metadata` API to manage metadata.

**Prerequisite**: if this is the first time you update a metadata, you need to register the metadata key on LandingLens first through the web UI.

Below screenshot shows you how to access the Metadata Management UI.

![the Metadata Management UI](assets/Metadata_Management_UI.png)

### Code Example of Metadata Management

```python
from landingai.data_management import Metadata

# Provide API Key and project ID
YOUR_API_KEY = "land_sk_12345"
YOUR_PROJECT_ID = 1190
metadata_client = Metadata(YOUR_PROJECT_ID, YOUR_API_KEY)
# Set three metadata ('timestamp', 'country' and 'labeler') for media with id 123 and 124. 
metadata_client.update(media_id=[123, 124], timestamp=12345, country="us", labeler="tom")
# Output:
# {
#    project_id: 1190,
#    metadata:   {"country": "us", "timestamp": "12345", labeler="tom"},
#    media_ids:  [123, 124]
# }
```

### Update split key for medias on LandingLens

When managing hundreds or thousands of images on the platform, it's handy to manage (add/update/remove) split key programmatically. You an use the `update_split_key()` function in `landingai.data_management.media.Media` to achieve it.

**Example**

```python
>>> client = Media(project_id, api_key)
>>> client.update_split_key(media_ids=[1001, 1002], split_key="test")  # assign split key 'test' for media ids 1001 and 1002
>>> client.update_split_key(media_ids=[1001, 1002], split_key="")    # remove split key for media ids 1001 and 1002
```

**Split Keys**

Valid split keys are "train", "dev", or "test" (case insensitive).
You can also remove a split key from a media by assigning a "" split, then the media will be under the "unassigned" split.

**Media ID**

To update split key, you need to provide a list of media ids. The media ids can be found from the browser UI or using the `ls()` function in `landingai.data_management.media.Media`.

Example:

```python
>>> media_client.ls()
>>> { medias: [{'id': 4358352, 'mediaType': 'image', 'srcType': 'user', 'srcName': 'Michal', 'properties': {'width': 258 'height': 176}, 'name': 'n01443537_501.JPEG', 'uploadTime': '2020-09-15T22:29:01.338Z', 'metadata': {'split': 'train' 'source': 'prod'}, 'media_status': 'raw'}, ...], num_requested: 1000, count: 300, offset: 0, limit: 1000 }
```

### Upload medias to LandingLens

You can use the `landingai.data_management.media.Media` API to upload medias to a specific project or list what medias are available in that project on LandingLens.

In addition to upload medias, the upload API supports a few nice features:
1. Assign a split ('train'/'dev'/'test') to the media(s). An empty string '' represents Unassigned and is the default.
2. Upload labels along with the media. The suported label files are:
    a. Pascal VOC xml file for object detection project
    b. A segmentation mask file for segmentation project
    c. A classification name (string) for classificaiton project
3. Attach additional metadata (key-value pairs) to the medias.

See [here](https://support.landing.ai/landinglens/docs/uploading#upload-images-with-split-and-label-information) for more information.


### Upload Segmentation Masks

You can use the `upload()` function to upload an image and its label together. When you upload a segmentation mask, the function requires a `seg_defect_map` parameter. This parameter points to a json file that maps the pixel values to class names. To get this map, you can use the `landingai.data_management.label.Label` API. See below code as an example.

```python
>>> client = Label(project_id, api_key)
>>> client.get_label_map()
>>> {'0': 'ok', '1': 'cat', '2': 'dog'}  # then write this map to a json file locally
```

### File Upload Limitations

**Supported Media File Types**

The following media file types are supported by LandingLens: "jpg", "jpeg", "png", "bmp"

In addition, the Python upload API supports uploading `tiff` image file. But the `upload()` API will automatically convert `tiff` to a `png` file and then upload the converted `png` image to the platform.

### Code Example of Media Management

Below code example shows you how to list existing medias in a project, and upload a single media or a folder of medias.

```python
from landingai.data_management import Media

# Provide API Key and project ID
YOUR_API_KEY = "land_sk_12345"
YOUR_PROJECT_ID = 1190

# Lists all medias with metadata from a project
media_client = Media(YOUR_PROJECT_ID, YOUR_API_KEY)
media_client.ls()
# Output:
# { medias: [{'id': 4358352, 'mediaType': 'image', 'srcType': 'user', 'srcName': 'Michal', 'properties': {'width': 258, 'height': 176}, 'name': 'n01443537_501.JPEG', 'uploadTime': '2020-09-15T22:29:01.338Z', 'metadata': {'split': 'train', 'source': 'prod'}, 'media_status': 'raw'}, ...],
# num_requested: 1000,
# count: 300,
# offset: 0,
# limit: 1000 }

# paginate with a custom page size
media_client.ls(offset=0, limit=100)

# any metadata can be used to filter the medias (server side filtering)
media_client.ls(split="train")

# upload a single media along with the label file, and assign it to the 'dev' split
media_client.upload("/Users/tom/Downloads/image.jpg", split="dev", object_detection_xml="/Users/tom/Downloads/image.xml")

# upload a folder of images and assign them to the 'train' split
media_client.upload("/Users/tom/Downloads/images", split="train")
```
