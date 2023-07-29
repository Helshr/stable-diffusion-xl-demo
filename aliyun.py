import os
import oss2


class MyAliyun:
    def __init__(self):
        OSS = {
            "Host": os.environ.get("OSS_HOST"),
            "Endpoint": os.environ.get("OSS_ENDPOINT"),
            "AccessKey_ID": os.environ.get("OSS_ACCESS_KEY_ID"),
            "AccessKey_Secret": os.environ.get("OSS_ACCESS_PASSWORD"),
            "Bucket_Name": os.environ.get("OSS_BUCKET"),
            "ALLOWED_EXTENSIONS": {"png", "jpg", "mp4", "gif", "jpge", "ttf", "otf", "mp3", "wav"},
        }
        auth = oss2.Auth(os.environ.get("OSS_ACCESS_KEY_ID"), os.environ.get("OSS_ACCESS_PASSWORD"))
        bucket = oss2.Bucket(auth, os.environ.get("OSS_STATIC_URI"), os.environ.get("OSS_STATIC"))
        self.bucket = bucket

    def web_insert_aliyun_file(self, remote_file, file_name):
        result = self.bucket.put_object_from_file(remote_file, file_name)
        return result.status
