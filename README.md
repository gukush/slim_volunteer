# Volunteer Compute

Minimal volunteer computing system: Node.js server + browser clients (WebGPU).

## Quickstart

```bash
./generate-certs.sh
docker compose up --build
```

Start at least one WebGPU client:

```bash
google-chrome --enable-gpu --enable-unsafe-webgpu --use-vulkan=swiftshader   --enable-features=Vulkan --disable-gpu-sandbox --no-sandbox --ignore-certificate-errors   --new-window --start-maximized --enable-logging --v=stderr   "https://localhost:3000/?mode=headless&workerId=test1&log=debug"
```

## REST

- `GET /strategies`
- `POST /tasks` (multipart)
  - fields: `strategyId`, `K` (int), `label` (string), `config` (JSON), `inputArgs` (JSON)
  - files:
    - **block-matmul**: `A.bin` (N x K), `B.bin` (K x M) as Float32 row-major
    - **ecm-stage1** (demo): one JSON file `numbers.json` containing `[1,2,3,...]`
- `POST /tasks/:id/start`
- `GET /tasks` / `GET /tasks/:id`
- `DELETE /tasks/:id`

## Strategies

### block-matmul (WebGPU)
### ecm-stage1 (WebGPU demo)
### distributed-sort (WebGPU)
