import os
import numpy as np
from generators.patient_image_generator import PatientImageGenerator
from PIL import Image

class LabeledImageGenerator(PatientImageGenerator):
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.patient_data_path = f"data/trackrad2025_labeled_training_data/{patient_id}"

        if not os.path.exists(self.patient_data_path):
            raise FileNotFoundError(f"Patient data not found for patient ID {patient_id}")

        self.patient_images_path = f"{self.patient_data_path}/images"
        self.patient_labels_path = f"{self.patient_data_path}/targets"
        self.patient_metadata = dict()

        with open(f"{self.patient_data_path}/b-field-strength.json", 'r') as f:
            self.patient_metadata['b_field_strength'] = float(f.read())
        with open(f"{self.patient_data_path}/frame-rate.json", 'r') as f:
            self.patient_metadata['frame_rate'] = float(f.read())
        with open(f"{self.patient_data_path}/scanned-region.json", 'r') as f:
            self.patient_metadata['scanned_region'] = f.read()

        self._read_images()
        self._read_labels()

    def _read_images(self):
        self.patient_images = dict()
        for filename in os.listdir(self.patient_images_path):
            if filename.endswith('.mha'):
                print(f"reading file {filename}...")
                image = self._read_aquisition(f"{self.patient_images_path}/{filename}")
                filename = filename.split('.')[0]
                self.patient_images[filename] = image

    def _read_labels(self):
        self.patient_labels = dict()
        for filename in os.listdir(self.patient_labels_path):
            if filename.endswith('.mha'):
                print(f"reading file {filename}...")
                image = self._read_aquisition(f"{self.patient_labels_path}/{filename}")
                filename = filename.split('.')[0]
                self.patient_labels[filename] = (image)

    def generate_images(self):
        for (index, (filename, volume)) in enumerate(self.patient_images.items()):
            output_file_name_prefix = f"{filename}_"
            for index, frame in enumerate(volume):
                output_dir = f"images/{self.patient_id}/images"
                output_file_name = f"{output_file_name_prefix}{index + 1:04d}"
                print(f"generating frame: {output_file_name}")
                self._convert_frame(frame, output_dir, output_file_name)

    def generate_labels(self):
        for (index, (filename, volume)) in enumerate(self.patient_labels.items()):
            output_file_name_prefix = f"{filename}_"
            for index, frame in enumerate(volume):
                output_dir = f"images/{self.patient_id}/labels/"
                output_file_name = f"{output_file_name_prefix}{index + 1:04d}"
                print(f"generating label: {output_file_name}")
                self._convert_frame(frame, output_dir, output_file_name)

    def generate_labeled_frames(self, alpha: float = 0.5, overlay_color=(0, 0, 128)):
        if not hasattr(self, 'patient_images') or not hasattr(self, 'patient_labels'):
            raise RuntimeError("Image and label volumes must be read before generating labeled frames.")

        for (filename, image_volume) in self.patient_images.items():
            label_file_name = filename.split('_')
            label_file_name = f"{label_file_name[0]}_{label_file_name[1]}_labels"
            label_volume = self.patient_labels.get(label_file_name)
            if label_volume is None:
                print(f"no matching labels found for {filename}, skipping labeled frames for this series")
                continue

            output_file_name_prefix = f"{filename}_labeled_"
            # iterate over paired frames; zip stops at the shortest volume if sizes mismatch
            for idx, (img_frame, lbl_frame) in enumerate(zip(image_volume, label_volume)):
                output_dir = f"images/{self.patient_id}/labeled/{filename}"
                output_file_name = f"{output_file_name_prefix}{idx + 1:04d}"
                print(f"generating labeled frame: {output_file_name}")

                # normalize the image to uint8 grayscale
                img_norm = self._normalize_frame(img_frame)  # uint8, 2D (H, W)

                # prepare RGB base image as float for blending
                base_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1).astype(np.float32)

                # create boolean mask where label is present
                mask = (lbl_frame > 0)

                if mask.any():
                    # create overlay image (same shape) filled with overlay_color
                    overlay = np.zeros_like(base_rgb, dtype=np.float32)
                    overlay[..., 0] = overlay_color[0]
                    overlay[..., 1] = overlay_color[1]
                    overlay[..., 2] = overlay_color[2]

                    a = float(alpha)
                    base_rgb[mask] = (1.0 - a) * base_rgb[mask] + a * overlay[mask]

                # clip and convert back to uint8
                out_arr = np.clip(base_rgb, 0, 255).astype(np.uint8)
                out_image = Image.fromarray(out_arr)

                os.makedirs(output_dir, exist_ok=True)
                out_image.save(os.path.join(output_dir, f"{output_file_name}.png"))

    def __str__(self):
        images = {k: v.shape for k, v in self.patient_images.items()}
        labels = {k: v.shape for k, v in self.patient_labels.items()}
        return f"""
Labeled patient analyzer:
    patient_id: {self.patient_id}
    patient_data_path: {self.patient_data_path}
    patient_images_path: {self.patient_images_path}
    patient_labels_path: {self.patient_labels_path}
    metadata:
        b_field_strength: {self.patient_metadata['b_field_strength']}
        frame_rate: {self.patient_metadata['frame_rate']}
        scanned_region: {self.patient_metadata['scanned_region']}
    images:
        {images}
    labels:
        {labels}
        """
