// server/lib/osMetrics.js
import fs from 'fs';
import path from 'path';
import pidusage from 'pidusage';
import si from 'systeminformation';
import { ensureDir } from './utils.js';
import { CSVWriter } from './metrics.js';

/**
 * Samples OS + process usage while a task is running and writes a CSV time-series per task.
 * - Per-process: cpu%, rss (MB), heapUsed (MB), io read/write (KB) [Linux]
 * - System: cpu%, mem used/free (MB), disk usage for a mount, network rx/tx totals + kbps
 */
export class OSUsageTracker {
  constructor({
    taskId,
    outDir,
    pidList = [process.pid],
    intervalMs = 1000,
    mountPoint = '/',
    ifaceAllow = /./,                           // allow all by default
    ifaceBlock = /^(lo|docker|veth|br-|kube|tun|tap)/, // skip common virtuals
  }){
    this.taskId = taskId;
    this.outDir = outDir;
    this.pidList = pidList;
    this.intervalMs = intervalMs;
    this.mountPoint = mountPoint;
    this.ifaceAllow = ifaceAllow;
    this.ifaceBlock = ifaceBlock;

    ensureDir(outDir);
    this.seriesPath = path.join(outDir, 'os_metrics.csv');
    this.series = new CSVWriter(this.seriesPath, [
      'timestamp_iso','rel_ms','task_id','sample_idx',
      'proc_cpu_pct','proc_rss_mb','proc_heap_used_mb',
      'proc_io_read_kb','proc_io_write_kb',
      'sys_cpu_pct',
      'sys_mem_used_mb','sys_mem_free_mb',
      'fs_used_pct','fs_used_gb','fs_free_gb',
      'net_rx_kbps','net_tx_kbps','net_rx_total_mb','net_tx_total_mb'
    ]);

    this.summaryPath = path.join(path.dirname(outDir), 'task_resource_summaries.csv');
    this.timer = null;
    this.sampleIdx = 0;
    this.startTs = null;
    this.prevNet = null; // {rx, tx, ts}
    this.agg = {
      samples: 0,
      procCpuSum: 0, procCpuPeak: 0,
      procRssSum: 0, procRssPeak: 0,
      netRxBytesStart: 0, netTxBytesStart: 0,
      netRxBytesEnd: 0, netTxBytesEnd: 0,
      netRxKbpsPeak: 0, netTxKbpsPeak: 0,
      ioReadBytes: 0, ioWriteBytes: 0,
      fsUsedPctStart: null, fsUsedPctEnd: null,
    };
  }

  async start(){
    this.startTs = Date.now();
    const fsSnap = await this.#fsUsage();
    this.agg.fsUsedPctStart = fsSnap?.usedPct ?? null;
    const net = await this.#netTotals();
    this.prevNet = net ? { ...net, ts: this.startTs } : null;
    if(net){ this.agg.netRxBytesStart = net.rx; this.agg.netTxBytesStart = net.tx; }

    await this.#sample(); // first sample immediately
    this.timer = setInterval(() => { this.#sample().catch(()=>{}); }, this.intervalMs);
  }

  stop(status='completed'){
    if(this.timer){ clearInterval(this.timer); this.timer = null; }
    const end = Date.now();
    const dur = end - this.startTs;
    const avgCpu = this.agg.samples ? this.agg.procCpuSum / this.agg.samples : 0;
    const avgRss = this.agg.samples ? this.agg.procRssSum / this.agg.samples : 0;
    const netRxMB = (this.agg.netRxBytesEnd - this.agg.netRxBytesStart) / (1024*1024);
    const netTxMB = (this.agg.netTxBytesEnd - this.agg.netTxBytesStart) / (1024*1024);
    const needHeader = !fs.existsSync(this.summaryPath);

    if(needHeader){
      fs.writeFileSync(
        this.summaryPath,
        'taskId,start,end,duration_ms,status,samples,' +
        'proc_cpu_avg,proc_cpu_peak,proc_rss_avg_mb,proc_rss_peak_mb,' +
        'proc_io_read_kb_total,proc_io_write_kb_total,' +
        'net_rx_mb_total,net_tx_mb_total,net_rx_kbps_peak,net_tx_kbps_peak,' +
        'fs_used_pct_start,fs_used_pct_end\n'
      );
    }

    const row = [
      this.taskId, this.startTs, end, dur, status, this.agg.samples,
      avgCpu.toFixed(2), this.agg.procCpuPeak.toFixed(2),
      avgRss.toFixed(2), this.agg.procRssPeak.toFixed(2),
      (this.agg.ioReadBytes/1024).toFixed(0), (this.agg.ioWriteBytes/1024).toFixed(0),
      netRxMB.toFixed(2), netTxMB.toFixed(2),
      this.agg.netRxKbpsPeak.toFixed(2), this.agg.netTxKbpsPeak.toFixed(2),
      this.agg.fsUsedPctStart ?? '', this.agg.fsUsedPctEnd ?? ''
    ];
    fs.appendFileSync(this.summaryPath, row.join(',') + '\n');
  }

  async #sample(){
    const ts = Date.now();
    const iso = new Date(ts).toISOString();
    const rel = ts - this.startTs;

    const [proc, sysCpu, mem, fsu, net] = await Promise.all([
      this.#procUsage(),
      si.currentLoad().then(x => x.currentload).catch(()=>null),
      si.mem().catch(()=>null),
      this.#fsUsage(),
      this.#netTotals(),
    ]);

    // network rates
    let rxKbps = '', txKbps = '';
    if(this.prevNet && net){
      const dtSec = (ts - this.prevNet.ts)/1000;
      if(dtSec > 0){
        rxKbps = ((net.rx - this.prevNet.rx) / 1024) / dtSec;
        txKbps = ((net.tx - this.prevNet.tx) / 1024) / dtSec;
        this.agg.netRxKbpsPeak = Math.max(this.agg.netRxKbpsPeak, rxKbps || 0);
        this.agg.netTxKbpsPeak = Math.max(this.agg.netTxKbpsPeak, txKbps || 0);
      }
    }
    this.prevNet = net ? { ...net, ts } : this.prevNet;

    // aggregates
    if(proc){
      this.agg.samples++;
      this.agg.procCpuSum += (proc.cpuPct || 0);
      this.agg.procCpuPeak = Math.max(this.agg.procCpuPeak, proc.cpuPct || 0);
      this.agg.procRssSum += (proc.rssMB || 0);
      this.agg.procRssPeak = Math.max(this.agg.procRssPeak, proc.rssMB || 0);
      this.agg.ioReadBytes += (proc.ioRead || 0);
      this.agg.ioWriteBytes += (proc.ioWrite || 0);
    }

    this.series.append([
      iso, rel, this.taskId, this.sampleIdx++,
      proc?.cpuPct?.toFixed?.(2) ?? '',
      proc?.rssMB?.toFixed?.(2) ?? '',
      proc?.heapUsedMB?.toFixed?.(2) ?? '',
      proc?.ioRead ? (proc.ioRead/1024).toFixed(0) : '',
      proc?.ioWrite ? (proc.ioWrite/1024).toFixed(0) : '',
      sysCpu?.toFixed?.(2) ?? '',
      mem ? (mem.active/1024/1024).toFixed(2) : '',
      mem ? (mem.free/1024/1024).toFixed(2) : '',
      fsu?.usedPct?.toFixed?.(2) ?? '',
      fsu ? (fsu.usedGB).toFixed(2) : '',
      fsu ? (fsu.freeGB).toFixed(2) : '',
      (rxKbps!=='' ? rxKbps.toFixed(2) : ''),
      (txKbps!=='' ? txKbps.toFixed(2) : ''),
      net ? (net.rx/1024/1024).toFixed(2) : '',
      net ? (net.tx/1024/1024).toFixed(2) : '',
    ]);

    if(net){ this.agg.netRxBytesEnd = net.rx; this.agg.netTxBytesEnd = net.tx; }
    if(fsu){ this.agg.fsUsedPctEnd = fsu.usedPct; }
  }

  async #procUsage(){
    // Sum multiple PIDs if provided
    const results = await Promise.all(this.pidList.map(pid => pidusage(pid).catch(()=>null)));
    const valid = results.filter(Boolean);
    if(valid.length === 0) return null;
    const sum = valid.reduce((a,b)=>({
      cpu: (a.cpu||0)+(b.cpu||0),
      memory: (a.memory||0)+(b.memory||0),
      io: {
        read_bytes: (a.io?.read_bytes||0)+(b.io?.read_bytes||0),
        write_bytes: (a.io?.write_bytes||0)+(b.io?.write_bytes||0),
      }
    }), {});
    const pu = process.memoryUsage();
    return {
      cpuPct: sum.cpu,                 // %
      rssMB: sum.memory/1024/1024,     // bytes -> MB
      heapUsedMB: pu.heapUsed/1024/1024,
      ioRead: sum.io.read_bytes,
      ioWrite: sum.io.write_bytes,
    };
  }

  async #fsUsage(){
    try{
      const sizes = await si.fsSize();
      const m = sizes.find(s => s.mount === this.mountPoint) || sizes[0];
      const usedPct = m.use; // percent used
      const freeGB = m.size ? (m.size - m.used)/1024/1024/1024 : 0;
      const usedGB = m.used/1024/1024/1024;
      return { usedPct, freeGB, usedGB };
    }catch{ return null; }
  }

  async #netTotals(){
    try{
      const stats = await si.networkStats();
      let rx=0, tx=0;
      for(const s of stats){
        const name = s.iface || '';
        if(this.ifaceBlock.test(name)) continue;
        if(!this.ifaceAllow.test(name)) continue;
        rx += s.rx_bytes || 0;
        tx += s.tx_bytes || 0;
      }
      return { rx, tx };
    }catch{ return null; }
  }
}
