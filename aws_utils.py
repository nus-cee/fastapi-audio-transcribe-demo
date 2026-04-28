import os

import boto3
from botocore.config import Config as BotoConfig

AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", "betekk-audio-to-text-27042026")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            config=BotoConfig(signature_version="s3v4"),
        )
    return _s3_client


def upload_text_to_s3(
    text: str,
    object_key: str,
    content_type: str = "text/plain; charset=utf-8",
) -> str:
    """
    Upload text content to S3 and return the public URL.

    Args:
        text:         The text content to upload.
        object_key:   The S3 object key (e.g. "transcripts/2024/file.txt").
        content_type: MIME type for the uploaded object.

    Returns:
        Public URL: https://{bucket}.s3.{region}.amazonaws.com/{object_key}
    """
    client = _get_s3_client()
    client.put_object(
        Bucket=AWS_BUCKET_NAME,
        Key=object_key,
        Body=text.encode("utf-8"),
        ContentType=content_type,
        ACL="public-read",
    )
    return f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_key}"
