from typing import List

from pydantic import BaseModel


class DataItem(BaseModel):
    client_id: str
    summary: str
    language: str
    category: str
    url: str

class DataPayload(BaseModel):
    items: List[DataItem]