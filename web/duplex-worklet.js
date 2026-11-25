class Capture extends AudioWorkletProcessor {
  constructor(){super();this.frame=new Float32Array(1920);this.offset=0;}
  process(inputs){
    const samples=inputs[0]?.[0];
    if(samples)for(const sample of samples){
      this.frame[this.offset++]=sample;
      if(this.offset===1920){this.port.postMessage(this.frame.buffer,[this.frame.buffer]);this.frame=new Float32Array(1920);this.offset=0;}
    }
    return true;
  }
}
registerProcessor('duplex-capture',Capture);
