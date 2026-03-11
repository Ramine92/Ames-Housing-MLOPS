import boto3
import os
import pathlib

s3 = boto3.client(
    's3',
    endpoint_url='https://dagshub.com/Ramine92/Ames-Housing-MLOPS.s3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

pathlib.Path('ml/models/artifacts').mkdir(parents=True, exist_ok=True)

s3.download_file(
    'dvc',
    'files/md5/42/4bd5d78150d116cccc9aa86f22e09c',
    'ml/models/artifacts/RandomForestRegressor_v1.pkl'
)

print("Model downloaded successfully!")
