import fs from 'fs';
import path from 'path';
import { ensureDir } from './utils.js';

export class CSVWriter{
  constructor(filePath, header){
    this.filePath = filePath;
    ensureDir(path.dirname(filePath));
    if(!fs.existsSync(filePath)){
      fs.writeFileSync(filePath, header.join(',') + '\n');
    }
  }
  append(row){
    const line = row.map(v => (v===undefined||v===null) ? '' : String(v).replaceAll(',', ';'));
    fs.appendFileSync(this.filePath, line.join(',') + '\n');
  }
}

export class TaskTimers{
  constructor(taskId, dir){
    this.taskId = taskId;
    this.start = Date.now();
    this.dir = dir;
    this.roundTrip = new CSVWriter(path.join(dir, `chunk_timing_${taskId}.csv`),
      ['chunkId','replica','t_chunk_create','t_sent','t_client_recv','t_client_done','t_server_recv','t_assembled','duration_ms']);
  }
  chunkRow(data){
    const {chunkId, replica, tCreate, tSent, tClientRecv, tClientDone, tServerRecv, tAssembled} = data;
    const duration = (tAssembled||tServerRecv||Date.now()) - (tCreate||this.start);
    this.roundTrip.append([chunkId, replica, tCreate, tSent, tClientRecv, tClientDone, tServerRecv, tAssembled, duration]);
  }
  endSummary(outFile, status='completed'){
    const end = Date.now();
    const dur = end - this.start;
    const header = !fs.existsSync(outFile);
    const line = [this.taskId, this.start, end, dur, status];
    if(header) fs.writeFileSync(outFile, 'taskId,start,end,duration_ms,status\n');
    fs.appendFileSync(outFile, line.join(',') + '\n');
  }
}
