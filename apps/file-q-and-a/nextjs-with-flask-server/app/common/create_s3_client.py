import boto3
        
def CreateS3Client():
    return boto3.client(
        's3',
        aws_access_key_id='AKIARXGEVPFIHRAQWOL5',
        aws_secret_access_key='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',
        region_name='us-east-1',
    )