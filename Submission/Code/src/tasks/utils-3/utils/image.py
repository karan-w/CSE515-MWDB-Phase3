import json 

class Image:
    def __init__(self, image_filename, image_matrix, subject_id, image_id, image_type, image_filepath):
        self.filename = image_filename
        self.matrix = image_matrix
        self.subject_id = subject_id
        self.image_id = image_id
        self.image_type = image_type
        self.filepath = image_filepath

    def __str__(self) -> str:
        string_representation = f'Image \nImage filename - {self.filename}\nMatrix\n{self.matrix}\nSubject ID - {self.subject_id}\nImage ID - {self.image_id}\nImage Type - {self.image_type}\n Image Filepath - {self.filepath}' 
        return string_representation
    
    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)