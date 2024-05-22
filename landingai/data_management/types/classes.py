from typing import Optional
from pydantic import BaseModel


class ClassMap(BaseModel):
    name: str
    color: Optional[str] = None
    description: str = ""
