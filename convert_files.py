import SimpleITK as sitk
import numpy as np
import os
from PIL import Image

def readImage(filePath: str) -> np.ndarray:
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"The file {filePath} does not exist.")
    image: sitk.Image = sitk.ReadImage(filePath)
    return sitk.GetArrayFromImage(image)

def normalizeFrame(frame: np.ndarray) -> np.ndarray:
    maxVal = frame.max()
    minVal = frame.min()

    frameNormalized = np.zeros(frame.shape, dtype=np.uint8)

    if maxVal > minVal:
        frameNormalized = ((frame - minVal) / (maxVal - minVal) * 255.0).astype(np.uint8)

    return frameNormalized


def convertToPNG(frame: np.ndarray, outputDir: str, outputFileName: str) -> None:
    os.makedirs(outputDir, exist_ok=True)

    frameNormalized = normalizeFrame(frame)
    image = Image.fromarray(frameNormalized)

    image.save(os.path.join(outputDir, f"{outputFileName}.png"))

def getListOfFiles(directory: str) -> list[str]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def getListOfDirectories(baseDirectory: str) -> list[str]:
    if not os.path.exists(baseDirectory):
        raise FileNotFoundError(f"The directory {baseDirectory} does not exist.")
    return [d for d in os.listdir(baseDirectory) if os.path.isdir(os.path.join(baseDirectory, d))]

def generateFiles(patientID:str, fileType: str) -> None: 
    baseDirectory = f"./data/trackrad2025_labeled_training_data/{patientID}/{fileType}"
    fileList = getListOfFiles(baseDirectory)

    for fileName in fileList:
        print(f"Processing file: {fileName}")
        filePath = os.path.join(baseDirectory, fileName)
        frameArray = readImage(filePath)

        for i, frame in enumerate(frameArray):
            print(f"Processing frame {i + 1}...")
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
            outputFileName = f"frame_{i + 1:03d}"
            convertToPNG(frame, f"./{fileType}/{patientID}/{fileName}", outputFileName)
            print(f"Converted frame {i + 1} to PNG.")


listOfPatients = getListOfDirectories("./data/trackrad2025_labeled_training_data/")
for patientID in listOfPatients:
    print(f"Processing patient: {patientID}")
    generateFiles(patientID, "images")
    generateFiles(patientID, "targets")
