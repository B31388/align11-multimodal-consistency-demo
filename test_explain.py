import yaml, torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from model.tmcl import load_tmcl_from_hub
from model.explain import explain_pair
print("Starting...")
config = yaml.safe_load(open("config.yaml"))
device = torch.device("cpu")
print("Loading model...")
model = load_tmcl_from_hub(repo_id=config["model"]["repo_id"], filename=config["model"]["filename"], token=None, device=device, freeze=True, attn_implementation="eager")
print("Model loaded")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
img = Image.new("RGB", (224,224), color=(100,80,50))
print("Running explain_pair...")
result = explain_pair(image_path=img, document="A black mountain bike near a lake", model=model, tokenizer=tokenizer, transform=transform, threshold=0.70, device=device, show_plot=False)
print("similarity:", result.get("similarity"))
print("vit_attention_map is None:", result.get("vit_attention_map") is None)
print("top_tokens:", result.get("top_tokens")[:5] if result.get("top_tokens") else None)
print("Done")

