# SynLoc YOLO26 on RunPod

## 1. Build and push locally

```bash
docker build -t YOUR_DOCKERHUB_USER/synloc-yolo26:1.0 .
docker push YOUR_DOCKERHUB_USER/synloc-yolo26:1.0
```

For an Apple Silicon Mac, build the Linux AMD64 image used by normal RunPod GPU machines:

```bash
docker buildx build \
  --platform linux/amd64 \
  -t YOUR_DOCKERHUB_USER/synloc-yolo26:1.0 \
  --push .
```

## 2. Create the RunPod Pod

Use the pushed image and attach persistent storage at `/workspace`.
The default container command is `idle`, which keeps the Pod alive for Web
Terminal or SSH access.

Recommended environment variables:

```text
SYNLOC_VERSION=fullhd
SYNLOC_SPLITS=train,valid
EPOCHS=100
IMGSZ=1280
BATCH=2
WORKERS=8
RUN_NAME=yolo26n_fullhd
```

Use `SYNLOC_SPLITS=train,valid,test,challenge` only when all four splits are
needed. YOLO training itself only needs `train,valid`.

## 3. Run interactively in the Pod terminal

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

/app/run_pipeline.sh download
/app/run_pipeline.sh convert
/app/run_pipeline.sh train
```

Or run everything:

```bash
/app/run_pipeline.sh all
```

## 4. Automatic startup

Set the RunPod container command to one of:

```text
pipeline
```

or:

```text
download
convert
train
```

`pipeline` downloads, extracts, converts, and trains sequentially.

## Persistent outputs

```text
/workspace/data/SoccerNet/SpiideoSynLoc
/workspace/data/SoccerNet/SpiideoSynLoc_yolo
/workspace/runs/synloc
```

All large data and training outputs remain under `/workspace`, not inside the
container filesystem.
