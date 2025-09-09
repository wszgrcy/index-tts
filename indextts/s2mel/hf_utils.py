import os
from huggingface_hub import hf_hub_download


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename="config.yml"):
    os.makedirs("./checkpoints", exist_ok=True)
    model_path = os.path.join(os.environ.get(
        'MODEL_DIR'), repo_id, model_filename,)
    if config_filename is None:
        return model_path
    config_path = os.path.join(os.environ.get(
        'MODEL_DIR'), repo_id, config_filename,)

    return model_path, config_path