## Data Management

The `landingai` library provides a set of APIs to manage dataset images and their metadata.
This section explains how to use the associated Python APIs to:

-   Upload new (single) images with metadata
-   Upload all images from a folder with metadata
-   Set metadata for images already in LandingLens


### Introduction to Metadata

Metadata is additional information you can attach to an image. Every image can be associated with multiple metadata. Each metadata is a key-value pair associated with an image, where key is the metadata name and value is a string that represents the information. For example, when you upload an image to LandingLens, you can add metadata like the country where the image was created, the timestamp when the image was created, etc.

Metadata is useful when you need to manage hundreds or thousands of images in LandingLens or you need to collaborate with other team members to label datasets. For example, you can metadata to group certain types of images together (ex: images taken last week), then change their [split key](https://support.landing.ai/docs/datasets-and-splits) or create a [labeling task](https://support.landing.ai/landinglens/docs/agreement-based-labeling#send-labeling-tasks) for those images.

Use the `landingai.data_management.metadata.Metadata` API to manage metadata.

**Prerequisite**: You must create a metadata key in the LandingLens UI before you can update it (assign a value to it) via the API. Each metadata key is project-specific.

The following screenshot shows how to acess the Manage Metadata module.

![the Metadata Management UI](images/Metadata_Management_UI.png)

### Code Example of Metadata Management
The following code snippet shows how to assign values to the Timestamp, Country, and Labeler metadata keys. 

```python
from landingai.data_management import Metadata

# Provide the API key and project ID
YOUR_API_KEY = "land_sk_12345"
YOUR_PROJECT_ID = 1190
metadata_client = Metadata(YOUR_PROJECT_ID, YOUR_API_KEY)
# Set three metadata values ('timestamp', 'country' and 'labeler') for images with IDs 123 and 124 
metadata_client.update(media_id=[123, 124], timestamp=12345, country="us", labeler="tom")
# Output:
# {
#    project_id: 1190,
#    metadata:   {"country": "us", "timestamp": "12345", labeler="tom"},
#    media_ids:  [123, 124]
# }
```

### Update Split Key for Images

When managing hundreds or thousands of images on the platform, it can be more efficient to manage (add/update/remove) the [split key](https://support.landing.ai/docs/datasets-and-splits) programmatically. Use the `update_split_key()` function in `landingai.data_management.media.Media` to manage the the split value for images.

**Example**

```python
>>> client = Media(project_id, api_key)
>>> client.update_split_key(media_ids=[1001, 1002], split_key="test")  # assign split key 'test' for images with IDs 1001 and 1002
>>> client.update_split_key(media_ids=[1001, 1002], split_key="")    # remove split key for images with IDs 1001 and 1002
```

**Split Keys**

Valid split keys are "train", "dev", or "test" (case insensitive).
To remove a split key from an image, assign the split value as "". After that, the image split will be  "unassigned".

**Media ID**

To update the split key, you need to provide a list of `media ids` (image IDs). The image IDs can be found in the LandingLens UI or by using the `ls()` function in `landingai.data_management.media.Media`.

Example:

```python
>>> media_client.ls()
>>> { medias: [{'id': 4358352, 'mediaType': 'image', 'srcType': 'user', 'srcName': 'Michal', 'properties': {'width': 258 'height': 176}, 'name': 'n01443537_501.JPEG', 'uploadTime': '2020-09-15T22:29:01.338Z', 'metadata': {'split': 'train' 'source': 'prod'}, 'media_status': 'raw'}, ...], num_requested: 1000, count: 300, offset: 0, limit: 1000 }
```

### Upload Images to LandingLens

Use the `landingai.data_management.media.Media` API to upload images to a specific project or list the images already in a specific project.

In addition to uploading images, the upload API supports the following features:
1. Assign a split ('train'/'dev'/'test') to images. An empty string '' represents Unassigned and is the default.
2. Upload labels along with images. The suported label files are:
    * [Pascal VOC XML files](https://support.landing.ai/docs/upload-labeled-images-od) for Object Detection projects.
    * [Segmentation mask files](https://support.landing.ai/docs/upload-labeled-images-seg) for Segmentation projects.
    * A classification name (string) for Classificaiton projects.
3. Attach additional metadata (key-value pairs) to images.

for more information, go [here](https://support.landing.ai/landinglens/docs/uploading#upload-images-with-split-and-label-information).


### Upload Segmentation Masks

Use the `upload()` function to upload an image and its segmentation mask (labels) together. When you upload a segmentation mask, the function requires a `seg_defect_map` parameter. This parameter points to a JSON file that maps the pixel values to class names. To get this map, use the `landingai.data_management.label.Label` API. 

The following code snippet shows how to upload an image and its segmentation mask. 

```python
>>> client = Label(project_id, api_key)
>>> client.get_label_map()
>>> {'0': 'ok', '1': 'cat', '2': 'dog'}  # then write this map to a JSON file locally
```

### File Upload Limitations

**Supported Image File Types**

LandingLens supports the following image file types: `bmp`, `jpeg`, `jpg`, `png`.

Additionally, the Python `upload()` API supports uploading `tiff` image files. However, the `upload()` API will automatically convert the `tiff` to a `png` file and, then upload the `png` image to LandingLens.

### Code Example of Image Management

The following code snippet shows how to list images currently in a project, upload a single image, and upload a folder of images. 

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

# if you have created a metadata in the platform, for example "cloudy" (this is case sensitive), you can also upload a value for that metadata
media_client.upload("/Users/tom/Downloads/images/image1.png", metadata_dict={"cloudy": "true"})
```
