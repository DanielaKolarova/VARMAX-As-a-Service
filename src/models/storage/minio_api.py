import boto3, botocore
import io
import pickle


# declare constants 
BUCKET_PREFIX = 'statistical_models'
ACCESS_KEY = "V87unISJLLVd5KIL8gBk",
SECRET_KEY = "kqOpsc8c29UPZUXadtYAH4Im3i2TCLqLYODDjTTS"
BUCKET_NAME = 'models'


def store_model(name, version, model):
    """
        Stores the model in a minio bucket suffixed by the specified version:
        
        name - model name
        version - model version 
        model - model object
    """
    s3client = boto3.client('s3', 
        endpoint_url='http://localhost:9100/',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
    )
    
    bytes_file = pickle.dumps(model)

    object_name = _build_object_name(name, version)
    
    # Idempotant operation that will create the bucket if it does not exist
    try:
        s3client.create_bucket(Bucket=BUCKET_NAME)
    except botocore.exceptions.ClientError as e:
        print(f'Bucket already created: {BUCKET_NAME}')

    #places file in the bucket
    s3client.put_object(
                Bucket=BUCKET_NAME,
                Key=object_name,
                Body=io.BytesIO(bytes_file)
            )
    
        
def load_model(name, version):
    """
        Loads the model stored in a minio bucket suffixed by the specified version:
        
        name - model name
        version - model version 
    """
    s3client = boto3.client('s3', 
        endpoint_url='http://localhost:9100/',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
    )

    object_name = _build_object_name(name, version)

    response = s3client.get_object(Bucket=BUCKET_NAME, Key=object_name)
    body = response['Body'].read()
    model = pickle.loads(body)   
        
    return model

def _build_object_name(name, version):
    return ''.join([name, '/', version, '/', 'model.pkl'])