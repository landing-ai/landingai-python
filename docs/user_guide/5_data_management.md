## Data management

The `landingai` library provides a set of APIs to manage dataset medias (e.g. images) and their metadata.
This section explains how to use the associated Python APIs to:

-   Upload new (single) Media with metadata
-   Upload ALL medias from Folder with metadata
-   Set Metadata for existing Medias


### Introduction to Metadata

Metadata is additional information you can attach to a media (e.g. image). Every media can be associated with multiple metadata. Each metadata is a key-value pair associated with this media, where key is the name of this metadata and value is a string that represents this information.
For example, when you upload an image to LandingLens, you can add metadata like the country where the image was created, the timestamp when the image was created, etc.

Metadata is useful when you need to manage hundreds or thousands of medias on LandingLens or you need to collaborate with other team members to label this dataset.
For example, you can use metadata to group certain type of medias together (e.g. images taken last week), then change their the split key, or create a [labeling task](https://support.landing.ai/landinglens/docs/agreement-based-labeling#send-labeling-tasks) out of those medias.

### Code Example of Metadata Management

**Prerequisite**: if this is the first time you update a metadata, you need to register the metadata key on LandingLens first through the web UI.

Below screenshot shows you how to access the Metadata Management UI.

![the Metadata Management UI](https://drive.google.com/uc?export=view&id=1DCWtfaUApXlqmvIAl6MSjlM4httlxoQV)

```python
from landingai.data_management import Metadata

# Provide API Key and project ID
YOUR_API_KEY = "land_sk_12345"
YOUR_PROJECT_ID = 1190
metadata_client = Metadata(YOUR_API_KEY, YOUR_PROJECT_ID)
# Set three metadata ('timestamp', 'country' and 'labeler') for media with id 123 and 124. 
metadata_client.update(media_id=[123, 124], timestamp=12345, country="us", labeler="tom")
# Output:
# {
#    project_id: 1190,
#    metadata:   {"country": "us", "timestamp": "12345", labeler="tom"},
#    media_ids:  [123, 124]
# }
```

### Code Example of Media Management

Below code example shows you how to list existing medias in a project, and upload a single media or a folder of medias.

```python
from landingai.data_management import Media

# Provide API Key and project ID
YOUR_API_KEY = "land_sk_12345"
YOUR_PROJECT_ID = 1190

# Lists all medias with metadata from a project
media_client = Media(YOUR_API_KEY, YOUR_PROJECT_ID)
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

# upload a single media
media_client.upload("/Users/tom/Downloads//image.jpg")

# upload a folder of images
media_client.upload("/Users/tom/Downloads/images")
```
