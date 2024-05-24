from enum import Enum
from pydantic import BaseModel, root_validator, validator
from typing import Any, Dict, Optional


class AutoSplitOptions(str, Enum):
    all_labeled = "all-labeled"
    without_split = "without-split"


class SplitPercentages(BaseModel):
    train: int
    dev: int
    test: int

    @validator("train", "dev", always=True)
    def must_contain_value(cls, v: int) -> Any:
        if v is None or v == 0:
            raise ValueError('"train" and "dev" should be defined and cannot be zero')
        return v

    @root_validator(pre=True)
    def check_split_match(cls, values: Dict[str, Optional[int]]) -> Any:
        train = values.get("train") or 0
        dev = values.get("dev") or 0
        test = values.get("test") or 0
        splits_sum = train + dev + test
        if splits_sum != 100:
            raise ValueError(
                f"train + dev + test values should add up to 100 but they add up to {splits_sum}"
            )
        return values
