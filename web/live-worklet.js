class DuplexAudio extends AudioWorkletProcessor {
  constructor(){
    super();this.epoch=0;this.queue=[];this.done=false;this.ended=true;this.started=false;this.gain=1;this.voiceFrames=0;this.quietFrames=0;this.localDucked=false;this.serverDucked=false;
    this.capture=new Float32Array(320);this.at=0;this.phase=0;this.frames=0;this.outputFrames=0;this.overlapFrames=0;this.outputActive=false;this.duckNotify=false;
    this.port.onmessage=({data:e})=>{
      if(e.type==='cancel'&&e.next_epoch>=this.epoch){this.epoch=e.next_epoch;this.queue=[];this.done=false;this.ended=true;this.started=false;}
      if(e.type==='reply_start'&&e.epoch>=this.epoch){this.epoch=e.epoch;this.queue=[];this.done=false;this.ended=false;this.started=false;this.gain=1;this.voiceFrames=0;this.localDucked=false;this.serverDucked=false;}
      if(e.type==='gain'){if(e.server)this.serverDucked=e.value<1;this.gain=e.value;if(e.value<1&&this.started&&!this.ended)this.duckNotify=true;}
      if(e.type==='audio'&&e.epoch===this.epoch){
        if(this.queue.reduce((n,c)=>n+(c.samples.length-c.at)/c.rate,0)+e.samples.length/e.rate>60){this.port.postMessage({type:'error',message:'Playback queue exceeded 60 seconds'});return;}
        this.queue.push({...e,at:0});
      }
      if(e.type==='reply_done'&&e.epoch===this.epoch)this.done=true;
    };
  }
  process(inputs,outputs){
    const input=inputs[0]?.[0],out=outputs[0]?.[0];if(!out)return true;
    if(this.duckNotify){this.port.postMessage({type:'playback_suppressed',epoch:this.epoch,gain:this.gain});this.duckNotify=false;}
    for(let i=0;i<out.length;i++){
      let value=0;
      while(this.queue.length&&this.queue[0].at>=this.queue[0].samples.length){const c=this.queue.shift();this.port.postMessage({type:'ack',epoch:this.epoch,chunk:c.chunk});}
      if(this.queue.length){
        const c=this.queue[0];if(!this.started){this.started=true;this.port.postMessage({type:'playback_start',epoch:this.epoch,chunk:c.chunk});}
        const at=Math.floor(c.at),fraction=c.at-at;
        value=(c.samples[at]*(1-fraction)+(c.samples[at+1]??c.samples[at])*fraction)*this.gain;c.at+=c.rate/sampleRate;
      }
      out[i]=value;if(Math.abs(value)>.001)this.outputActive=true;
      this.phase+=16000/sampleRate;
      if(this.phase>=1){
        this.phase-=1;this.capture[this.at++]=Math.max(-1,Math.min(1,input?.[i]??0));
        if(this.at===320){
          this.frames++;let energy=0;for(const x of this.capture)energy+=x*x;const rms=Math.sqrt(energy/320);
          if(this.outputActive){this.outputFrames++;if(rms>.004)this.overlapFrames++;}
          this.voiceFrames=rms>.004?this.voiceFrames+1:0;this.quietFrames=rms<.002?this.quietFrames+1:0;
          if(this.started&&!this.ended&&this.queue.length&&this.voiceFrames>=3&&!this.localDucked&&!this.serverDucked){this.gain=.25;this.localDucked=true;this.port.postMessage({type:'playback_suppressed',epoch:this.epoch,gain:.25});}
          if(this.localDucked&&!this.serverDucked&&this.quietFrames>=50){this.gain=1;this.localDucked=false;}

          this.port.postMessage({type:'capture',pcm:this.capture.buffer,rms,frames:this.frames,output_frames:this.outputFrames,overlap_frames:this.overlapFrames},[this.capture.buffer]);
          this.capture=new Float32Array(320);this.at=0;this.outputActive=false;
        }
      }
    }
    while(this.queue.length&&this.queue[0].at>=this.queue[0].samples.length){const c=this.queue.shift();this.port.postMessage({type:'ack',epoch:this.epoch,chunk:c.chunk});}
    if(this.done&&!this.queue.length&&!this.ended){this.ended=true;this.port.postMessage({type:'playback_end',epoch:this.epoch});}
    return true;
  }
}
registerProcessor('duplex-audio',DuplexAudio);
