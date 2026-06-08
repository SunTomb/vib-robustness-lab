# Lightweight Deployment Notes

## Recommended deployment shape

Deploy only the artifact viewer:

- FastAPI backend reads precomputed `artifacts/full`.
- React/Vite frontend is built once and served statically or through a small web server.
- No GPU, training loop, raw dataset download, or checkpoint-heavy workflow is required on the cloud server.

## Not recommended on the small cloud server

- Full PyTorch/CUDA installation
- Full MNIST/Fashion-MNIST experiment sweeps
- Raw dataset/cache transfer unless explicitly needed
- Docker images that include GPU training dependencies

## Suggested production commands

Build frontend locally or on a capable machine:

```bash
npm --prefix frontend run build
```

Run backend against copied artifacts:

```bash
VIB_ARTIFACT_DIR=artifacts/full python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Artifact policy

If deploying remotely, copy only:

- `backend/`
- `frontend/dist/`
- `artifacts/full/`
- minimal Python dependency files

Do not copy raw datasets, local virtual environments, `node_modules`, logs unless needed for audit, or experiment checkpoints.
