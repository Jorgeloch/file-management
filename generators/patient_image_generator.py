import os
import SimpleITK as sitk
import numpy as np
from PIL import Image

class PatientImageGenerator():
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        maxVal = frame.max()
        minVal = frame.min()
        normalized = np.zeros(frame.shape, dtype=np.uint8)
        if maxVal > minVal:
            normalized = ((frame - minVal) / (maxVal - minVal) * 255.0).astype(np.uint8)
        return normalized

    def _convert_frame(self, frame: np.ndarray, output_dir: str, output_file_name: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        frameNormalized = self._normalize_frame(frame)
        image = Image.fromarray(frameNormalized)
        image.save(os.path.join(output_dir, f"{output_file_name}.png"))

    def _read_aquisition(self, image_path: str) -> np.ndarray:
        # read image
        image: sitk.Image = sitk.ReadImage(image_path)
        return np.transpose(sitk.GetArrayFromImage(image), (2, 0, 1))
