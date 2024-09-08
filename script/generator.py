import numpy as np
import cv2

def image_mask_generator(image_paths, mask_paths, batch_size):
    num_training_samples = len(image_paths)
    while True:
        for i in range(0, num_training_samples, batch_size):
            batch_training_image_paths = image_paths[i:i + batch_size]
            batch_training_mask_paths = mask_paths[i:i + batch_size]

            batch_training_images = []
            batch_training_masks = []

            for image_path, mask_path in zip(batch_training_image_paths, batch_training_mask_paths):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0  # Normalize to [0, 1]
                batch_training_images.append(image)

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = mask / 255.0  # Normalize to [0, 1]
                batch_training_masks.append(mask)

            batch_training_images = np.array(batch_training_images)
            batch_training_masks = np.array(batch_training_masks)

            yield batch_training_images, batch_training_masks

        