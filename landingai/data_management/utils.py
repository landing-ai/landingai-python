import json
import pprint
from enum import Enum
from typing import Any, Dict, cast


def metadata_to_ids(
    input_metadata: Dict[str, Any], metadata_mapping: Dict[str, Any]
) -> Dict[str, Any]:
    validate_metadata(input_metadata, metadata_mapping)
    return {
        metadata_mapping[key][0]: val
        for key, val in input_metadata.items()
        if key in metadata_mapping
    }


def ids_to_metadata(
    metadata_ids: Dict[str, Any], id_to_metadata: Dict[int, str]
) -> Dict[str, Any]:
    return {
        id_to_metadata[int(key)]: val
        for key, val in metadata_ids.items()
        if int(key) in id_to_metadata
    }


def to_camel_case(snake_str: str) -> str:
    """Convert a snake case string to camel case"""
    words = snake_str.split("_")
    return words[0] + "".join(word.title() for word in words[1:])


def validate_metadata(
    input_metadata: Dict[str, Any], metadata_mapping: Dict[str, Any]
) -> None:
    """Validate the input metadata against the metadata mapping. Raise ValueError if any metadata keys are not available."""
    not_allowed = set(input_metadata.keys()) - set(metadata_mapping.keys())
    # TODO: Validate also values and maybe types. Or shouldn't it be the job of the server?
    if len(not_allowed) > 0:
        raise ValueError(
            f"""Not allowed fields: {not_allowed}.
Available fields are {metadata_mapping.keys()}.
If you want to add new fields, please add it to the associated project on the LandingLens platform."""
        )


def obj_to_dict(obj: object) -> Dict[str, Any]:
    """Convert an object to a json dictionary with camel case keys"""
    json_body = json.dumps(obj, cls=Encoder)
    return cast(Dict[str, Any], json.loads(json_body))


def obj_to_params(obj: object) -> Dict[str, Any]:
    """Convert an object to query parameters in dict format where the dict keys are in camel case."""
    return {
        to_camel_case(k): v if isinstance(v, list) else json.dumps(v, cls=Encoder)
        for k, v in obj.__dict__.items()
    }


class Encoder(json.JSONEncoder):
    """JSON encoder that converts all keys to camel case"""

    def default(self, obj: object) -> Any:
        if isinstance(obj, dict):
            return {to_camel_case(k): v for k, v in obj.items()}
        if isinstance(obj, Enum):
            return obj._name_
        return {to_camel_case(k): v for k, v in obj.__dict__.items()}


class PrettyPrintable:
    """A mix-in class that enables its subclass to be serialized into pretty printed string"""

    def to_str(self) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.__dict__)

    def __repr__(self) -> str:
        """For `print` and `pprint`"""
        return self.to_str()
