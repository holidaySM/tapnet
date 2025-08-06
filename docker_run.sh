docker run -itd \
  -v /home/sm/PycharmProjects/tapnet:/projects/tapnet \
  -v /home/sm/Datasets:/app/datasets \
  --ipc=host \
  --gpus all \
  -p 30000:22 nvcr.io/nvidia/jax:23.08-py3 \
  /bin/bash