// gen-matrix.mjs
import fs from 'fs';
import path from 'path';

function lcg(seed){ // deterministic pseudo-random [0,1)
  let x = BigInt(seed >>> 0) || 1n;
  const a = 1664525n, c = 1013904223n, m = (1n<<32n);
  return ()=>{ x = (a*x + c) % m; return Number(x) / 2**32; };
}

// fills row-major float32 matrix R x C
async function writeMatrix({ rows, cols, out, seed=1, fn='rand' }){
  await fs.promises.mkdir(path.dirname(out), { recursive: true });
  const fd = fs.openSync(out, 'w');
  const rowBytes = cols*4;
  const buf = Buffer.allocUnsafe(rowBytes);
  const rand = lcg(seed);
  const f32 = new Float32Array(buf.buffer, buf.byteOffset, cols);

  for (let r=0; r<rows; r++){
    for (let c=0; c<cols; c++){
      if (fn === 'rand') f32[c] = rand()*2-1; // [-1,1)
      else if (fn === 'sin') f32[c] = Math.sin((r+1)*0.01) + Math.cos((c+1)*0.02);
      else f32[c] = 1; // fallback
    }
    fs.writeSync(fd, buf, 0, rowBytes, r*rowBytes);
    if ((r & 1023) === 0) process.stdout.write(`\rwriting ${out} row ${r+1}/${rows}`);
  }
  fs.closeSync(fd);
  process.stdout.write(`\r${out} done (${rows}x${cols})\n`);
}

const args = Object.fromEntries(process.argv.slice(2).map(s=>{
  const [k,v] = s.split('=');
  return [k.replace(/^--/, ''), v ?? true];
}));
await writeMatrix({
  rows: Number(args.rows),
  cols: Number(args.cols),
  out: args.out,
  seed: Number(args.seed ?? 1),
  fn: args.fn || 'rand'
});
