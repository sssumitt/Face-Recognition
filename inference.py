import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import ConvolutionalNetwork

# Configuration
CHECKPOINT_PATH = 'model/model.pth'  
IMG1_PATH       = 'img1.jpg'                   # First image path
IMG2_PATH       = 'img2.jpg'                   # Second image path
DEVICE          = 'cpu'                        # or 'cuda' if available


def load_model(checkpoint_path: str, device: str = DEVICE):
    """Loads the model from a state_dict .pth file."""
    # Initialize model architecture
    model = ConvolutionalNetwork(
        num_classes=540,
        s=32.0,
        m=0.3,
        lr=1e-3
    )
    # Load state_dict
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess(image_path: str, device: str = DEVICE):
    """Reads and preprocesses an image into a normalized tensor."""
    tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = tf(img).unsqueeze(0).to(device)
    return tensor


def compute_similarity(model, img1_tensor: torch.Tensor, img2_tensor: torch.Tensor):
    """Computes cosine similarity between two face embeddings."""
    with torch.no_grad():
        emb1 = model.encode(img1_tensor)
        emb2 = model.encode(img2_tensor)
        sim = F.cosine_similarity(emb1, emb2)
    return sim.item()


def main():
    # Load model and images
    model = load_model(CHECKPOINT_PATH)
    img1 = preprocess(IMG1_PATH)
    img2 = preprocess(IMG2_PATH)

    # Compute similarity
    similarity = compute_similarity(model, img1, img2)

    threshold = 0.37

    if(similarity >= 0.37):
        print("Similar Image")
    else:
        print("different Image")
    print(similarity)
        

if __name__ == '__main__':
    main()
