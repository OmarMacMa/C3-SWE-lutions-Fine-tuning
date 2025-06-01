# modal_stub.py

import modal
from modal import App, Image

app = modal.App("devstral-finetuning")
#volume = modal.SharedVolume().persist("devstral-vol")
#volume = modal.Volume.from_name("my-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("unsloth", "xformers", "datasets", "trl", "bitsandbytes", "accelerate", "torch", "triton")
    .env({"HF_TOKEN": ""})
)

image = image.add_local_dir(".", "/root").add_local_file("modal_finetuning.py", "/root/modal_finetuning.py")
#@app.function(image=image, gpu="A100-80GB", timeout=3600, volumes={"/root/data":   volume})
@app.function(
    image=image, 
    gpu="A100-80GB", 
    timeout=3600,
    )
def train(): 
    import os
    print("Listing files in /root:")
    for file in os.listdir("/root"):
        print(" -", file)   

    import sys
    sys.path.append("/root")
    #import shutil
    #shutil.copy("/root/data/training_lite.json", ".")
    import modal_finetuning  # this runs your `train.py`

if __name__ == "__main__":
    app.deploy()
    #train.call()
