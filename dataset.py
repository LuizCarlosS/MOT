import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image

@dataclass
class FrameResults:
    ids: np.ndarray
    detections: np.ndarray  # List of detections. Expected to be in the shape (x, y, w, h)
    frame_id: int
    frame: np.ndarray

    def get_frame_with_bbox(self):
        drawn_frame = self.frame.copy()
        for det, id in zip(self.detections, self.ids):
            x, y, w, h = det
            drawn_frame = cv2.rectangle(drawn_frame, (x, y), (x + w, y + h), (36, 255, 12), 1)
            cv2.putText(drawn_frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return drawn_frame


class MOT17Dataset(Dataset):
    def __init__(self, data_dir, sequence_length):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.sequences = self.load_sequences()

    def load_sequences(self):
        sequences = []
        sequence_dirs = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        for sequence_dir in sequence_dirs:
            sequence_path = os.path.join(self.data_dir, sequence_dir)
            annotation_file = os.path.join(sequence_path, 'gt', 'gt.txt')
            df = pd.read_csv(annotation_file, delimiter=',', header=None)
            df.columns = ['frame_id', 'object_id', 'xmin', 'ymin', 'width', 'height', 'conf', 'class', 'visibility']
            sequences.append(df)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        num_frames = len(sequence)

        # Randomly select a frame index
        frame_index = torch.randint(0, num_frames, (1,))
        frame_data = sequence.iloc[frame_index.item()]

        # Determine the start index for previous frames
        start_index = max(0, frame_index.item() - self.sequence_length + 1)
        previous_frames = sequence.iloc[start_index:frame_index.item()]

        # Repeat information for the oldest known frames if necessary
        if start_index == 0:
            num_repeats = self.sequence_length - len(previous_frames)
            previous_frames = pd.concat([previous_frames] * num_repeats)

        # Process the data and create tensors
        frames_tensor = torch.tensor(previous_frames[['xmin', 'ymin', 'width', 'height']].values, dtype=torch.float32)
        current_frame_tensor = torch.tensor(frame_data[['xmin', 'ymin', 'width', 'height']].values, dtype=torch.float32)

        # Load and process the images
        frames_images = self.load_images(sequence, previous_frames['frame_id'])
        current_frame_image = self.load_image(sequence, frame_data['frame_id'])

        return frames_tensor, current_frame_tensor, frames_images, current_frame_image

    def load_images(self, sequence, frame_ids):
        image_dir = os.path.join(self.data_dir, sequence.iloc[0]['frame_id'], 'img1')
        images = []
        for frame_id in frame_ids:
            image_path = os.path.join(image_dir, f"{str(frame_id).zfill(6)}.jpg")
            image = Image.open(image_path)
            images.append(image)
        return images

    def load_image(self, sequence, frame_id):
        image_dir = os.path.join(self.data_dir, sequence.iloc[0]['frame_id'], 'img1')
        image_path = os.path.join(image_dir, f"{str(frame_id).zfill(6)}.jpg")
        image = Image.open(image_path)
        return image