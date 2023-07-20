import boto3
import os
from dotenv import load_dotenv, find_dotenv
from preprocessing_ray import preprocessing


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
        s3_client.download_file(
            "gyhibo-databucket",
            filename + ".webm",
            filename + ".webm")
        
        print("VIDEO PROCESSING")
        # 전처리 파트 시작
        preprocessing(filename)
        os.remove(filename + ".webm")
        """
        여기서 모델 활용
        """
    except:
        return "Error"
    return "translated_word"

if __name__ == "__main__":
    translate_video("bf2c632e-6641-4ff7-a905-7f8d9a64ea80")