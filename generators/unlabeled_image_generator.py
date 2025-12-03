import os
from generators.patient_image_generator import PatientImageGenerator

class UnlabeledImageGenerator(PatientImageGenerator):
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.patient_data_path = f"data/trackrad2025_unlabeled_training_data/{patient_id}"

        if not os.path.exists(self.patient_data_path):
            raise FileNotFoundError(f"Patient data not found for patient ID {patient_id}")

        self.patient_images_path = f"{self.patient_data_path}/images"
        self.patient_metadata = dict()

        with open(f"{self.patient_data_path}/b-field-strength.json", 'r') as f:
            self.patient_metadata['b_field_strength'] = float(f.read())
        with open(f"{self.patient_data_path}/frame-rate.json", 'r') as f:
            self.patient_metadata['frame_rate'] = float(f.read())
        with open(f"{self.patient_data_path}/scanned-region.json", 'r') as f:
            self.patient_metadata['scanned_region'] = f.read()

        self._read_images()

    def _read_images(self):
        self.patient_images = dict()
        for filename in os.listdir(self.patient_images_path):
            if filename.endswith('.mha'):
                print(f"reading file {filename}...")
                image = self._read_aquisition(f"{self.patient_images_path}/{filename}")
                filename = filename.split('.')[0]
                self.patient_images[filename] = image

    def generate_images(self):
        for (index, (filename, volume)) in enumerate(self.patient_images.items()):
            output_file_name_prefix = f"{filename}_"
            for index, frame in enumerate(volume):
                output_dir = f"images/{self.patient_id}/images/{filename}"
                output_file_name = f"{output_file_name_prefix}{index + 1:04d}"
                print(f"generating frame: {output_file_name}")
                self._convert_frame(frame, output_dir, output_file_name)

    def __str__(self):
        images = {k: v.shape for k, v in self.patient_images.items()}
        return f"""
Labeled patient analyzer:
    patient_id: {self.patient_id}
    patient_data_path: {self.patient_data_path}
    patient_images_path: {self.patient_images_path}
    metadata:
        b_field_strength: {self.patient_metadata['b_field_strength']}
        frame_rate: {self.patient_metadata['frame_rate']}
        scanned_region: {self.patient_metadata['scanned_region']}
    images:
        {images}
        """
