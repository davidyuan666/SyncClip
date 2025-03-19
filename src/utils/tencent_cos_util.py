from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import logging
import os


class COSOperationsUtil:
    def __init__(self):
        self.region = os.getenv('COS_REGION')
        self.bucket_name = os.getenv('COS_BUCKET_NAME')
        self.secret_id = os.getenv('COS_SECRET_ID')
        self.secret_key = os.getenv('COS_SECRET_KEY')

        self.config = CosConfig(
            Region=self.region,
            SecretId=self.secret_id,
            SecretKey=self.secret_key,
            Token=None,
            Scheme='https'
        )
        self.client = CosS3Client(self.config)
        
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)

    def create_bucket(self):
        try:
            response = self.client.create_bucket(
                Bucket=self.bucket_name
            )
            print(response)
            self.logger.info(f"Bucket created successfully: {self.bucket_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating bucket: {e}")
            return False

    def list_buckets(self):
        try:
            response = self.client.list_buckets()
            print(response)
            buckets = response['Buckets']['Bucket']
            self.logger.info(f"Buckets listed successfully. Count: {len(buckets)}")
            return buckets
        except Exception as e:
            self.logger.error(f"Error listing buckets: {e}")
            return []

    def upload_file(self, local_file_path, cos_file_path):
        try:
            # Get the file size
            file_size = os.path.getsize(local_file_path)
            
            # Calculate an appropriate part size (minimum 5MB, maximum 5GB)
            part_size = max(5, min(file_size // 10000, 5120))
            
            response = self.client.upload_file(
                Bucket=self.bucket_name,
                LocalFilePath=local_file_path,
                Key=cos_file_path,
                PartSize=part_size,
                MAXThread=10,
                EnableMD5=True
            )
            
            self.logger.info(f"File uploaded successfully: {cos_file_path}")
            return response
        except Exception as e:
            self.logger.error(f"Error uploading file: {e}")
            return None

    def list_objects(self, prefix=''):
        try:
            response = self.client.list_objects(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            objects = [content['Key'] for content in response.get('Contents', [])]
            self.logger.info(f"Objects listed successfully. Count: {len(objects)}")
            return objects
        except Exception as e:
            self.logger.error(f"Error listing objects: {e}")
            return []

    def download_file(self, cos_file_path, local_file_path):
        try:
            response = self.client.download_file(
                Bucket=self.bucket_name,
                Key=cos_file_path,
                DestFilePath=local_file_path
            )
            self.logger.info(f"File downloaded successfully: {local_file_path} and response: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return None

    def delete_object(self, cos_file_path):
        try:
            response = self.client.delete_object(
                Bucket=self.bucket_name,
                Key=cos_file_path
            )
            self.logger.info(f"Object deleted successfully: {cos_file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting object: {e}")
            return False


    def get_file_url(self, cos_file_path, expires=None):
        """
        获取文件的访问URL
        
        Args:
            bucket (str): 存储桶名称
            cos_file_path (str): 对象在 COS 中的路径
            expires (int, optional): URL 的有效期，单位为秒，默认为永久
            
        Returns:
            str: 文件的访问URL，失败返回None
        """
        try:
            if not cos_file_path:
                self.logger.error("COS file path cannot be empty")
                return None
                
            # 检查文件是否存在
            try:
                self.client.head_object(
                    Bucket=self.bucket_name,
                    Key=cos_file_path
                )
            except Exception as e:
                self.logger.error(f"File does not exist in COS: {cos_file_path}")
                return None
            
            if expires:
                # 生成预签名URL（临时访问）
                url = self.client.get_presigned_url(
                    Method='GET',
                    Bucket=self.bucket_name,
                    Key=cos_file_path,
                    Expired=expires
                )
            else:
                # 使用实例变量 self.region 而不是 config.region
                url = f'https://{self.bucket_name}.cos.{self.region}.myqcloud.com/{cos_file_path}'
                
            self.logger.info(f"Generated URL for file: {cos_file_path}")
            return url
            
        except Exception as e:
            self.logger.error(f"Error generating URL for file {cos_file_path}: {str(e)}")
            return None
            
    def check_file_exists(self, cos_file_path):
        """
        检查文件是否存在于 COS 中
        
        Args:
            bucket (str): 存储桶名称
            cos_file_path (str): 对象在 COS 中的路径
            
        Returns:
            bool: 文件存在返回True，否则返回False
        """
        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=cos_file_path
            )
            return True
        except Exception as e:
            self.logger.debug(f"File does not exist in COS: {cos_file_path}")
            return False