# Import Libraries
import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to caption the image
def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL image and convert to RGB
    raw_image = Image.fromarray(input_image).convert("RGB")

    # Process the image
    inputs = processor(images=raw_image, return_tensors='pt')

    # Generate the image caption
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode the generated tokens into text
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn = caption_image,
    inputs = gr.Image(),
    outputs = 'text',
    title = 'Image Captioning',
    description = "Simple web app for generating image captions"
)

iface.launch()
