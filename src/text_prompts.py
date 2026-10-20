import os
import torch

try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


UCF_CLASSES = [
    "normal", "abuse", "arrest", "arson", "assault", "burglary",
    "explosion", "fighting", "roadaccidents", "robbery", "shooting",
    "shoplifting", "stealing", "vandalism"
]

UCF_PROMPTS = [
    "a surveillance video showing normal daily activity",
    "a surveillance video showing abuse",
    "a surveillance video showing arrest",
    "a surveillance video showing arson and fire",
    "a surveillance video showing physical assault",
    "a surveillance video showing burglary and break-in",
    "a surveillance video showing explosion",
    "a surveillance video showing street fighting",
    "a surveillance video showing road accident and car crash",
    "a surveillance video showing robbery and armed theft",
    "a surveillance video showing gun shooting",
    "a surveillance video showing shoplifting",
    "a surveillance video showing stealing",
    "a surveillance video showing vandalism and destruction"
]

XD_CLASSES = [
    "normal", "fighting", "shooting", "riot", "abuse", "car accident", "explosion"
]

XD_PROMPTS = [
    "a video showing normal peaceful scenes",
    "a video showing fighting and physical brawl",
    "a video showing gun shooting and firearms",
    "a video showing public riot and protest violence",
    "a video showing violence and abuse",
    "a video showing car accident and vehicle crash",
    "a video showing explosion and blast"
]

LABEL_MAP_UCF = {
    'Normal': 'normal',
    'Abuse': 'abuse',
    'Arrest': 'arrest',
    'Arson': 'arson',
    'Assault': 'assault',
    'Burglary': 'burglary',
    'Explosion': 'explosion',
    'Fighting': 'fighting',
    'RoadAccidents': 'roadaccidents',
    'Robbery': 'robbery',
    'Shooting': 'shooting',
    'Shoplifting': 'shoplifting',
    'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}

LABEL_MAP_XD = {
    'A': 'normal',
    'B1': 'fighting',
    'B2': 'shooting',
    'B4': 'riot',
    'B5': 'abuse',
    'B6': 'car accident',
    'G': 'explosion'
}


def get_class_index(label_str, dataset='ucf'):
    """Maps label string to class index (0 is normal)."""
    if dataset == 'ucf':
        label_clean = str(label_str).strip()
        mapped = LABEL_MAP_UCF.get(label_clean, 'normal')
        if mapped in UCF_CLASSES:
            return UCF_CLASSES.index(mapped)
        return 0
    else:
        label_clean = str(label_str).strip()
        mapped = LABEL_MAP_XD.get(label_clean, 'normal')
        if mapped in XD_CLASSES:
            return XD_CLASSES.index(mapped)
        return 0


def load_or_encode_text_features(device, dataset='ucf', clip_model_name="ViT-B/16"):
    """
    Returns normalized CLIP text features of shape (C, 512).
    Caches encoded tensor locally for fast reloading.
    """
    cache_path = f"list/{dataset}_clip_text_features.pt"
    if os.path.exists(cache_path):
        features = torch.load(cache_path, map_location=device)
        return features

    prompts = UCF_PROMPTS if dataset == 'ucf' else XD_PROMPTS

    if not HAS_CLIP:
        # Fallback to random orthogonal features if clip library is not installed
        print("[Warning] CLIP library not installed. Generating dummy text embeddings.")
        C = len(prompts)
        features = torch.randn(C, 512, device=device)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features

    model, _ = clip.load(clip_model_name, device=device)
    model.eval()
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(text_features.cpu(), cache_path)
    return text_features.to(device)
