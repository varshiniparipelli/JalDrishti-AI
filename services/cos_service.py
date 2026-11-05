"""
JalDrishti-AI — IBM Cloud Object Storage Service
Upload / download dataset files to/from IBM COS Lite bucket.
"""

import os
import ibm_boto3
from ibm_botocore.client import Config, ClientError
from dotenv import load_dotenv

load_dotenv()

COS_API_KEY       = os.getenv("IBM_COS_API_KEY")
COS_INSTANCE_ID   = os.getenv("IBM_COS_INSTANCE_ID")
COS_ENDPOINT      = os.getenv("IBM_COS_ENDPOINT")
COS_AUTH_ENDPOINT = os.getenv("IBM_COS_AUTH_ENDPOINT")
BUCKET_NAME       = os.getenv("IBM_COS_BUCKET_NAME", "jaldrishti-data-bucket")


def _get_client():
    return ibm_boto3.client(
        "s3",
        ibm_api_key_id=COS_API_KEY,
        ibm_service_instance_id=COS_INSTANCE_ID,
        ibm_auth_endpoint=COS_AUTH_ENDPOINT,
        config=Config(signature_version="oauth"),
        endpoint_url=COS_ENDPOINT,
    )


def ensure_bucket_exists():
    """Create the bucket if it does not already exist."""
    client = _get_client()
    try:
        client.head_bucket(Bucket=BUCKET_NAME)
        print(f"[COS] Bucket '{BUCKET_NAME}' already exists.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            client.create_bucket(Bucket=BUCKET_NAME)
            print(f"[COS] Bucket '{BUCKET_NAME}' created.")
        else:
            raise


def upload_file(local_path: str, object_name: str = None):
    """Upload a local file to IBM COS."""
    if object_name is None:
        object_name = os.path.basename(local_path)
    client = _get_client()
    try:
        client.upload_file(local_path, BUCKET_NAME, object_name)
        print(f"[COS] Uploaded '{local_path}' → '{BUCKET_NAME}/{object_name}'")
        return True
    except ClientError as e:
        print(f"[COS] Upload failed: {e}")
        return False


def download_file(object_name: str, local_path: str):
    """Download a file from IBM COS to a local path."""
    client = _get_client()
    try:
        client.download_file(BUCKET_NAME, object_name, local_path)
        print(f"[COS] Downloaded '{BUCKET_NAME}/{object_name}' → '{local_path}'")
        return True
    except ClientError as e:
        print(f"[COS] Download failed: {e}")
        return False


def list_objects():
    """List all objects in the COS bucket."""
    client = _get_client()
    try:
        resp = client.list_objects_v2(Bucket=BUCKET_NAME)
        items = [obj["Key"] for obj in resp.get("Contents", [])]
        return items
    except ClientError as e:
        print(f"[COS] List objects failed: {e}")
        return []
