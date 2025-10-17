import os
import cv2
try:
    from mtcnn_ort import MTCNN
except:
    pip install mtcnn-onnxruntime
    from mtcnn_ort import MTCNN
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent.futures


def crop_and_align(image, result):
    print(image.shape)
    # Find the face with highest confidence
    max_confidence_face = max(result, key=lambda x: x['confidence'])
    box = max_confidence_face['box']
    keypoints = max_confidence_face['keypoints']

    # Crop the face
    cropped = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

    # Align the face
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.arctan(dy / dx) * 180. / np.pi
    M = cv2.getRotationMatrix2D(
        (cropped.shape[1] / 2, cropped.shape[0] / 2), angle, 1)
    aligned = cv2.warpAffine(cropped, M, (cropped.shape[1], cropped.shape[0]))

    # Resize to 224x224
    resized = cv2.resize(aligned, (224, 224))

    return resized

def process_image(detector, src_path, dst_path):
    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Read and process the image
    image = cv2.cvtColor(cv2.imread(src_path), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    if result:
        processed = crop_and_align(image, result)
        cv2.imwrite(dst_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))


def process_directory(detector, src_dir, dst_dir, num_threads):
    # Collect all the tasks
    # 读取所有图片
    tasks = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.png'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(
                    dst_dir, os.path.relpath(src_path, src_dir))
                tasks.append((src_path, dst_path))

    # Process the tasks with a thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(
            process_image, detector, src_path, dst_path) for src_path, dst_path in tasks]

        # Display progress with tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass


detector = MTCNN()
src_dir = 'MAFW/frames'
dst_dir = 'MAFW/frames_crop_aligned'
process_directory(detector, src_dir, dst_dir, num_threads=4)
