from langchain.tools import Tool
import json
import os
from typing import List

def read_file(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()

def write_file(file_path: str, content: str) -> str:
    with open(file_path, 'w') as f:
        f.write(content)
    return f"Content written to {file_path}"

file_read_tool = Tool(
    name="Read File",
    func=read_file,
    description="Read a file's content"
)

write_to_file_tool = Tool(
    name="Write to File",
    func=write_file,
    description="Write content to a file"
)

def list_directory(directory: str) -> List[str]:
    return os.listdir(directory)

directory_read_tool = Tool(
    name="List Directory",
    func=list_directory,
    description="List files in a directory"
)