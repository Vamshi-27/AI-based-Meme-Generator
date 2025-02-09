import torch
import clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image

class ClipGPT2Model:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        
        # Load CLIP Model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load GPT-2 Model
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)

    def get_image_features(self, image):
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        return image_features

    def generate_caption(self, image):
        image_features = self.get_image_features(image)

        # Convert features to text prompt
        text_input = "Meme description: "
        input_ids = self.gpt2_tokenizer.encode(text_input, return_tensors="pt").to(self.device)

        # Generate text caption
        with torch.no_grad():
            output_ids = self.gpt2_model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.gpt2_tokenizer.eos_token_id,
            )
        caption = self.gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
