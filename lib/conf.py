import os
import docker

client = docker.from_env()
docker_utils_image = 'utils'
web_interface_port = 7860
web_dir_expire = 72 # hours

model_root = os.path.abspath("./models")
audiobooks_dir = os.path.abspath("./audiobooks")
process_dir = os.path.abspath("./tmp")