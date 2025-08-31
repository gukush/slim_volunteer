const level = process.env.LOG_LEVEL || 'info';
const levels = ['error','warn','info','debug','trace'];
const idx = levels.indexOf(level);
function log(kind, ...args){ if(levels.indexOf(kind) <= idx) console[kind==='trace'?'debug':kind](new Date().toISOString(), kind.toUpperCase(), ...args); }
export const logger = {
  error: (...a)=>log('error',...a),
  warn: (...a)=>log('warn',...a),
  info: (...a)=>log('info',...a),
  debug: (...a)=>log('debug',...a),
  trace: (...a)=>log('trace',...a),
};
