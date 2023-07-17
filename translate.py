import boto3
import os
from dotenv import load_dotenv, find_dotenv



def translate_video(filename="default"):
    if filename == "default":
        return "Error"
    load_dotenv(find_dotenv())
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_ACCESS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("S3_REGION"))
    try:
        video = s3_client.get_object(
            Bucket="gyhibo-databucket",
            Key=filename + ".webm"
        )
        
        print("VIDEO PROCESSING")
        """
        여기서 모델 활용
        """
    except:
        return "Error"
    return "translated_word"

if __name__ == "__main__":
    translate_video("83ed2fad-0bdc-4e48-ae89-f676ea5ca540")