from utils.models.clip import AltClip, AlignClip, FlavaClip, ViTOpenAIClip

CONFIG = {
    "openai": {
        "models": [
            ["RN50x64", "openai"],
            ["ViT-B-16", "openai"],
            ["ViT-B-32", "openai"],
            ["ViT-B-32", "laion400m_e31"],
            ["ViT-B-32", "laion2b_s34b_b79k"],
            ["ViT-L-14", "openai"],
            ["ViT-L-14-336", "openai"],
            ["ViT-L-14", "laion2b_s32b_b82k"],
            ["ViT-H-14", "laion2b_s32b_b79k"],
            ["ViT-g-14", "laion2b_s34b_b88k"],
            ["xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"],
            ["xlm-roberta-large-ViT-H-14", "frozen_laion5b_s13b_b90k"],
            ["coca_ViT-L-14", "mscoco_finetuned_laion2b_s13b_b90k"],
        ],
        "handler": ViTOpenAIClip,
    },
    "alt": {
        "models": [["BAAI/AltCLIP"]],
        "handler": AltClip,
    },
    "align": {
        "models": [["kakaobrain/align-base"]],
        "handler": AlignClip
    },
    "align": {
        "models": [["facebook/flava-full"]],
        "handler": FlavaClip
    }
}
