#!/usr/bin/env node
// Simple native client that connects to the server via WebSocket and executes binaries

import WebSocket from 'ws';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = Object.fromEntries(process.argv.slice(2).map(s => {
  const m = s.match(/^--([^=]+)=(.*)$/);
  return m ? [m[1], m[2]] : [s.replace(/^--/, ''), true];
}));

const serverHost = args.host || '127.0.0.1';
const serverPort = args.port || '3001';
const clientId = args.clientId || `native_${Date.now()}`;
const capacity = parseInt(args.capacity || '1', 10);

const wsUrl = `ws://${serverHost}:${serverPort}/ws-native`;

console.log(`🔌 Connecting to server: ${wsUrl}`);
console.log(`🆔 Client ID: ${clientId}`);
console.log(`⚡ Capacity: ${capacity}`);

const ws = new WebSocket(wsUrl);

ws.on('open', () => {
  console.log('✅ Connected to server');
  
  // Send client join message
  const joinMessage = {
    type: 'client:join',
    data: {
      clientId,
      capacity,
      frameworks: ['native-opencl', 'native-cuda', 'native-vulkan'],
      capabilities: ['binary_execution']
    }
  };
  
  console.log('📤 Sending join message:', joinMessage);
  ws.send(JSON.stringify(joinMessage));
});

ws.on('message', async (data) => {
  try {
    const message = JSON.parse(data.toString());
    console.log('📥 Received message:', message.type);
    
    switch (message.type) {
      case 'client:join:ack':
        console.log('✅ Server acknowledged client join');
        break;
        
      case 'workload:new':
        console.log('🎯 Received workload:', message.data.taskId);
        await handleWorkload(message.data);
        break;
        
      case 'workload:ready':
        console.log('✅ Workload ready:', message.data.taskId);
        break;
        
      case 'chunk:assign':
        console.log('📦 Received chunk assignment:', message.data.chunkId);
        await handleChunk(message.data);
        break;
        
      case 'workload:error':
        console.error('❌ Workload error:', message.data.message);
        break;
        
      default:
        console.log('❓ Unknown message type:', message.type);
    }
  } catch (error) {
    console.error('❌ Error processing message:', error);
  }
});

ws.on('close', (code, reason) => {
  console.log(`🔌 Connection closed: ${code} - ${reason}`);
  process.exit(0);
});

ws.on('error', (error) => {
  console.error('❌ WebSocket error:', error);
  process.exit(1);
});

async function handleWorkload(workload) {
  console.log('🎯 Handling workload:', workload.taskId);
  console.log('📋 Framework:', workload.framework);
  console.log('📋 Schema:', workload.schema);
  
  if (workload.artifacts && workload.artifacts.length > 0) {
    console.log('📦 Artifacts received:');
    for (const artifact of workload.artifacts) {
      console.log(`  - ${artifact.name} (${artifact.type}, ${artifact.size} bytes)`);
      
      if (artifact.type === 'binary' && artifact.bytes) {
        // Save binary artifact to disk
        const binaryPath = path.join(__dirname, artifact.name);
        const binaryData = Buffer.from(artifact.bytes, 'base64');
        fs.writeFileSync(binaryPath, binaryData);
        fs.chmodSync(binaryPath, 0o755); // Make executable
        console.log(`  ✅ Saved binary: ${binaryPath}`);
      }
    }
  }
}

async function handleChunk(chunk) {
  const { taskId, chunkId, payload, meta } = chunk;
  console.log(`📦 Processing chunk ${chunkId} for task ${taskId}`);
  
  try {
    // For now, we'll just simulate processing
    // In a real implementation, this would execute the binary with the chunk data
    console.log('🔧 Chunk metadata:', meta);
    console.log('📊 Payload keys:', Object.keys(payload));
    
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Send result back to server
    const result = {
      type: 'workload:chunk_done_enhanced',
      data: {
        taskId,
        chunkId,
        status: 'ok',
        result: new ArrayBuffer(1024), // Dummy result
        checksum: 'dummy_checksum',
        timings: {
          tClientRecv: Date.now(),
          tClientDone: Date.now()
        }
      }
    };
    
    console.log('📤 Sending chunk result');
    ws.send(JSON.stringify(result));
    
  } catch (error) {
    console.error('❌ Error processing chunk:', error);
    
    // Send error back to server
    const errorResult = {
      type: 'workload:chunk_error',
      data: {
        taskId,
        chunkId,
        error: error.message
      }
    };
    
    ws.send(JSON.stringify(errorResult));
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down native client...');
  ws.close();
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down native client...');
  ws.close();
  process.exit(0);
});

