import crypto from 'crypto';
import fs from 'fs';
export function sha256(buf){ return crypto.createHash('sha256').update(buf).digest('hex'); }
export function ensureDir(p){ if(!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true }); }
export function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }
export function now(){ return Date.now(); }
export function writeJSON(p, obj){ fs.writeFileSync(p, JSON.stringify(obj, null, 2)); }
