# GPU image; works on CPU too (slower).  docker build -t avsum . && docker run --gpus all -v $PWD/data:/data avsum /data/in.mp4 /data/out.json
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python weights/download_weights.py
ENV HF_HUB_OFFLINE=1
ENTRYPOINT ["./run.sh"]
