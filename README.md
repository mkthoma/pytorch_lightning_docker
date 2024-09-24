# Lightning Hydra Project Template

## Requirements

Build the Docker image:

```
docker build -t dogbreed-classification .
```

To run training:
```
docker run --gpus all -v /path/to/your/data:/app/data dogbreed-classification
```

To run evaluation:
```
docker run --gpus all -v /path/to/your/data:/app/data dogbreed-classification python eval.py
```

To run inference:
```
docker run --gpus all -v /path/to/your/input/images:/app/input -v /path/to/your/output:/app/outpu
```
