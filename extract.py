import argparse
import os
import pandas as pd
from PIL import Image
import torch
from ram.models import ram
from ram import inference_ram as inference
from ram import get_transform
from lavis.models import load_model
from lavis.models import load_model_and_preprocess



# Function to process and infer tags for multiple images
def process_images(images_dir, model,model1,vis_processors, transform, device):
    image_files = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    data = []

    for file in image_files:
        file_path = os.path.join(images_dir, file)
        image = transform(Image.open(file_path)).unsqueeze(0).to(device)
        raw_image = Image.open(file_path).convert("RGB")
        res = inference(image, model)
        images_blip = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        print(images_blip.shape)
        # Generate caption for the image
        captions = model1.generate({"image": images_blip})
        tags = res[0]  # Assuming the tags are in the first element
        # Extract other metadata as needed
        data.append({'Filename': file, 'Tags': tags, 'Captions': captions,'image_path': file_path })

    return pd.DataFrame(data)

# Parser setup and argument parsing
parser = argparse.ArgumentParser(description='Tag2Text inference for tagging and captioning')
parser.add_argument('--image-dir',
                    metavar='DIR',
                    help='path to dataset',
                    default='/content/drive/My Drive/ImageSample')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')

if __name__ == "__main__":
    args = parser.parse_args()

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform(image_size=args.image_size)
    model = ram(pretrained=args.pretrained, image_size=args.image_size, vit='swin_l')
    model.eval()
    model = model.to(device)
    # Load the BLIP image captioning model
    model1, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

    # Process images in the specified directory
    df = process_images(args.image_dir, model, model1 ,vis_processors,transform, device)

    # Display the DataFrame
    

    # Optionally, save the DataFrame to a CSV file
    df.to_csv('image_tags.csv', index=False)
