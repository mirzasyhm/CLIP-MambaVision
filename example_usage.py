import torch
from clip import load, tokenize
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Load the models
model_original, preprocess_original = load(name='CLIP_Original')
model_original.to('cuda' if torch.cuda.is_available() else 'cpu')

model_mamba, preprocess_mamba = load(name='CLIP_MambaVision_T')
model_mamba.to('cuda' if torch.cuda.is_available() else 'cpu')

# Download and load a sample image from CIFAR-10
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to match CLIP's expected input
    transforms.ToTensor()
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

sample_image, label = testset[0]  # Change index for different images
sample_image_pil = transforms.ToPILImage()(sample_image)

# Display the image
plt.imshow(sample_image_pil)
plt.axis('off')
plt.title(f'Label: {testset.classes[label]}')
plt.show()

# Prepare Inputs
device = "cuda" if torch.cuda.is_available() else "cpu"
image_input_original = preprocess_original(sample_image_pil).unsqueeze(0).to(device)
image_input_mamba = preprocess_mamba(sample_image_pil).unsqueeze(0).to(device)

texts = ["A photo of a cat.", "A photo of a dog.", "A photo of a bird."]
text_input = tokenize(texts).to(device)

# Forward Pass - Original Encoder
with torch.no_grad():
    logits_per_image_original, logits_per_text_original = model_original(image_input_original, text_input)
    probs_original = logits_per_image_original.softmax(dim=-1).cpu().numpy()

# Forward Pass - MambaVision Encoder
with torch.no_grad():
    logits_per_image_mamba, logits_per_text_mamba = model_mamba(image_input_mamba, text_input)
    probs_mamba = logits_per_image_mamba.softmax(dim=-1).cpu().numpy()

# Display Results
print("Original Encoder Probabilities:", probs_original)
print("MambaVision Encoder Probabilities:", probs_mamba)

# Visualization
labels = texts
x = range(len(labels))
width = 0.35  # Width of the bars

plt.figure(figsize=(10,6))
plt.bar(x, probs_original[0], width, label='Original Encoder', color='blue')
plt.bar([p + width for p in x], probs_mamba[0], width, label='MambaVision Encoder', color='red')

plt.xlabel('Labels')
plt.ylabel('Probability')
plt.title('Comparison of Encoder Probabilities')
plt.xticks([p + width/2 for p in x], labels)
plt.legend()
plt.show()
