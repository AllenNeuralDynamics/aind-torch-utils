"""
Utility functions
"""

import json
import os
from typing import Optional
from urllib.parse import urlparse

import boto3

def load_json(data_path: str, keyname: Optional[str] = ".zattrs") -> dict:
    """
    Loads a JSON file from either a local path or S3.

    Parameters
    ----------
    data_path: str
        Path to the data. Can be:
        - Local file path (e.g., "/path/to/file.json" or "/path/to/zarr/dataset")
        - S3 path (e.g., "s3://bucket-name/path/to/zarr/dataset")
    keyname: Optional[str]
        Name of the JSON file to load. Default is ".zattrs".
        - For local paths: this is appended to data_path
        - For S3 paths: this is appended to the S3 key

    Returns
    -------
    dict
        Dictionary with the JSON data

    Raises
    ------
    FileNotFoundError
        If the local file doesn't exist
    Exception
        If there's an error accessing S3 or parsing JSON
    """

    # Check if it's an S3 path
    if data_path.startswith("s3://"):
        return _load_json_from_s3(data_path, keyname)
    else:
        return _load_json_from_local(data_path, keyname)


def _load_json_from_s3(s3_data_path: str, keyname: str) -> dict:
    """
    Load JSON from S3.
    """
    parsed = urlparse(s3_data_path)
    bucket_name = parsed.netloc
    key = f"{parsed.path.lstrip('/')}/{keyname}"

    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket_name, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        return data
    except Exception as e:
        raise Exception(f"Error loading JSON from S3 path '{s3_data_path}/{keyname}': {str(e)}")


def _load_json_from_local(local_path: str, keyname: str) -> dict:
    """
    Load JSON from local file system.
    """
    # Construct the full file path
    if keyname:
        full_path = os.path.join(local_path, keyname)
    else:
        full_path = local_path

    # Handle case where the path already includes the filename
    if not os.path.exists(full_path) and os.path.exists(local_path):
        # If the constructed path doesn't exist but the original path does,
        # assume the original path is the complete file path
        full_path = local_path

    try:
        with open(full_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at path: {full_path}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing JSON file '{full_path}': {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading JSON file '{full_path}': {str(e)}")
