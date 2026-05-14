from .nano_seed import NanoSeedEdit
from .seedream_text import SeedreamTextToImage

NODE_CLASS_MAPPINGS = {
    "NanoSeedEdit": NanoSeedEdit,
    "SeedreamTextToImage": SeedreamTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoSeedEdit": "NanoSeed",
    "SeedreamTextToImage": "Seedream Text to Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
