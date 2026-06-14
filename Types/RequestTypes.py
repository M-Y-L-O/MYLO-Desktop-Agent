from pydantic import BaseModel

class FilePathRequest(BaseModel):
    filepath:str