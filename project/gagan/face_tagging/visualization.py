import os

import matplotlib.pyplot as plt


def visualize(matched_pairs: dict, images_path: str):
    """
    Visualizes all the images using matplotlib

    Args:
        matched_pairs (dict): A dictionary containing grouped images.
        images_path (str): The path where images are present
    """
    if matched_pairs == []:
        raise ValueError("Empty List")
    
    n_rows = len(matched_pairs)

    for group, images in matched_pairs.items():
        fig = plt.figure(figsize=(8, 6))
        fig.subplots_adjust(hspace=0.5)
        for idx, image in enumerate(images):
            img = plt.imread(os.path.join(images_path, image))
            ax = fig.add_subplot(n_rows, idx+3, idx+1)
            ax.title.set_text(group)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.imshow(img)
