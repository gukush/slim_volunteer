import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../lib/logger.js';

export const id = 'ecm-stage1';
export const name = 'ECM Stage 1 (WebGPU demo)';
export const framework = 'webgpu';

export function getClientExecutorInfo(config){
  return {
    framework,
    path: 'executors/webgpu-ecm-stage1.client.js',
    kernels: ['kernels/ecm_stage1_webgpu_compute.wgsl'],
    schema: { output: 'Uint32Array' },
  };
}

// Input: a JSON file with an array of 32-bit integers to "process" (demo)
// Config: { chunkSize: 65536 }
export function buildChunker({ taskId, taskDir, K, config, inputFiles }){
  const jsonFile = inputFiles.find(f=>/\.json$/.test(f.originalName));
  if(!jsonFile) throw new Error('ECM demo expects a JSON file of integers');
  const arr = JSON.parse(fs.readFileSync(jsonFile.path, 'utf-8'));
  const total = arr.length;
  const CS = config.chunkSize || 65536;

  return {
    async *stream(){
      for(let offset=0; offset<total; offset+=CS){
        const part = new Uint32Array(arr.slice(offset, Math.min(total, offset+CS)));
        const payload = { data: part.buffer.slice(0), dims: { n: part.length } };
        const meta = { offset, n: part.length };
        yield { id: uuidv4(), payload, meta, tCreate: Date.now() };
      }
      logger.info('ECM stage 1 demo chunker done');
    }
  };
}

export function buildAssembler({ taskId, taskDir, config }){
  const results = [];
  const outPath = path.join(taskDir, 'ecm_stage1_output.json');
  return {
    integrate({ chunkId, result, meta }){
      const out = Array.from(new Uint32Array(result));
      results.push({ meta, out });
    },
    finalize(){
      fs.writeFileSync(outPath, JSON.stringify({ chunks: results }, null, 2));
      return { outPath, chunks: results.length };
    }
  };
}
