import pandas as pd
import json
from PIL import Image

class MemeDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Load JSON and TSV files
        self.memes_data_df = pd.read_csv(f"{data_dir}/memes_data.tsv", sep='\t')

    def load_image(self, image_path):
        """
        Load an image from the specified path.
        """
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def get_captioned_data(self):
        """
        Return a DataFrame containing image paths and their corresponding captions.
        """
        return self.memes_data_df[["image_path", "caption"]]
