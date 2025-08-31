# Volunteer Compute (Fresh)

Minimal, readable volunteer computing system: Node.js server + browser clients (WebGPU).

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
- Config: `{ "N": 512, "K": 512, "M": 512, "tileSize": 128 }`
- Output: `output.bin` (Float32 row-major N x M) in task dir

### ecm-stage1 (WebGPU demo)
- Demo harness using your provided WGSL if available; otherwise uses a simple transform kernel.
- Input: JSON list of `u32` values; output: transformed chunks JSON for inspection.

## Notes
- Executor + kernels are sent **once** per task (`task:init`). Chunks then carry only data + meta.
- Redundancy `K` and checksum verification implemented on the server.
- Timings CSV are written to `server/storage/timing/`.

