from enum import Enum
from pydantic import BaseModel, root_validator, validator


class AutoSplitOptions(str, Enum):
    all_labeled = "all-labeled"
    without_split = "without-split"


class SplitPercentages(BaseModel):
    train: int
    dev: int
    test: int

    @validator("train", "dev", always=True)
    def must_contain_value(cls, v):
        if v is None or v == 0:
            raise ValueError('"train" and "dev" should be defined and cannot be zero')
        return v

    @root_validator(pre=True)
    def check_split_match(cls, values):
        train = values.get("train")
        dev = values.get("dev")
        test = values.get("test")
        splits_sum = train + dev + test
        if splits_sum != 100:
            raise ValueError(
                f"train + dev + test values should add up to 100 but they add up to {splits_sum}"
            )
        return values
