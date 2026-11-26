// Pure DSP/transport unit checks; real hardware is tested separately.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let Processor;const posted=[];
class Base {constructor(){this.port={postMessage:e=>posted.push(e),onmessage:null};}}
vm.runInNewContext(fs.readFileSync(require('node:path').join(__dirname,'../web/live-worklet.js'),'utf8'),{AudioWorkletProcessor:Base,sampleRate:48000,Float32Array,Math,registerProcessor:(_,p)=>Processor=p});
const p=new Processor(),send=e=>p.port.onmessage({data:e});
send({type:'reply_start',epoch:1});send({type:'audio',epoch:1,chunk:1,rate:24000,samples:new Float32Array(2400).fill(.1)});send({type:'reply_done',epoch:1});
let rendered=0;
for(let n=0;n<375;n++){const out=new Float32Array(128);p.process([[new Float32Array(128)]],[[out]]);for(const x of out)if(x>.09)rendered++;}
assert.equal(rendered,4800);assert.equal(posted.filter(e=>e.type==='capture').length,50);
assert.equal(posted.filter(e=>e.type==='ack').length,1);
assert(posted.findIndex(e=>e.type==='playback_start')<posted.findIndex(e=>e.type==='playback_end'));
send({type:'cancel',next_epoch:2});send({type:'audio',epoch:1,chunk:2,rate:16000,samples:new Float32Array(640).fill(1)});
const out=new Float32Array(128);p.process([[new Float32Array(128)]],[[out]]);assert(out.every(x=>x===0));
send({type:'audio',epoch:2,chunk:3,rate:16000,samples:new Float32Array(16000*61)});assert(posted.some(e=>e.type==='error'));
send({type:'reply_start',epoch:3});send({type:'audio',epoch:3,chunk:4,rate:16000,samples:new Float32Array(16000).fill(.4)});
for(let i=0;i<30;i++)p.process([[new Float32Array(128).fill(.02)]],[[new Float32Array(128)]]);
assert(posted.some(e=>e.type==='playback_suppressed'&&e.epoch===3));assert.equal(p.gain,.25);
console.log(JSON.stringify({status:'passed',checks:6,scope:'resampling, capture pacing, acknowledgement order, stale audio rejection, output bound',physical_hardware:false}));
