import os
import math

def get_file_size(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "does not exist"

    size_bytes = os.path.getsize(file_path)
    if size_bytes == 0:
        return "0 B"

    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"
