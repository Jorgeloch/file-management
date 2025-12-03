import os
from generators.labeled_image_generator import LabeledImageGenerator
from generators.unlabeled_image_generator import UnlabeledImageGenerator


# Reads the labeled data and generates the respective images
for patient_id in os.listdir("data/trackrad2025_labeled_training_data/"):
    analyzer = LabeledImageGenerator(patient_id)
    analyzer.generate_images()
    analyzer.generate_labels()
    analyzer.generate_labeled_frames()

# Reads the unlabeled data and generates the respective images
for patient_id in os.listdir("data/trackrad2025_unlabeled_training_data/"):
    analyzer = UnlabeledImageGenerator(patient_id)
    analyzer.generate_images()
