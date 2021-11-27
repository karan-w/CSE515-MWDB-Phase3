import os
import datetime
import json

class Output:
    def __init__(self) -> None:
        pass

    def create_timestamp_folder(self, output_folder_path):
        timestamp_folder_path = os.path.join(output_folder_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(timestamp_folder_path)
        return timestamp_folder_path
    
    def save_dict_as_json_file(self, dictionary, json_file_path):
        json_file = open(json_file_path, "w")
        json_string = json.dumps(dictionary, default=lambda obj: obj.__dict__, indent=4)
        json_file.write(json_string)
        json_file.close()
