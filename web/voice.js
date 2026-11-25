/* Actual microphone -> PCM WAV -> native LFM model -> playable generated WAV. */
(() => {
  const el=id=>document.getElementById(id);
  const session=crypto.randomUUID();
  let recording=null,busy=false,ready=false,resetNext=true;
  const players=[],urls=[];
  const state=(text,error=false)=>{el('voice-state').textContent=text;el('voice-state').classList.toggle('bad',error);};
  function controls(){el('record').disabled=!ready||busy;el('voice-file').disabled=!ready||busy||!!recording;el('voice-new').disabled=!ready||busy||!!recording;el('voice-load').disabled=busy||!!recording;}
  async function api(path,body,type='application/json'){
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':type,'X-STS-Session':session},body});
    const data=await response.json();if(!response.ok)throw Error(data.error||'Local voice request failed');return data;
  }
  const stopPlayback=()=>{players.forEach(p=>p.pause());el('voice-stop').disabled=true;};
  function message(label,text){const article=document.createElement('article');article.className='voice-message';const name=document.createElement('strong');name.textContent=label;const p=document.createElement('p');p.textContent=text;article.append(name,p);el('voice-messages').append(article);return article;}
  async function send(blob,fromRecording=false){
    if(busy&&!fromRecording)return;busy=true;controls();stopPlayback();state('Listening to your audio and generating speech locally…');
    const started=performance.now();message('Your voice',`${(blob.size/48000).toFixed(1)} seconds of PCM audio`);
    try{
      const result=await api('/api/voice/turn'+(resetNext?'?reset=1':''),blob,'audio/wav');resetNext=false;
      window.dispatchEvent(new CustomEvent('sts-voice-reply',{detail:result}));
      const article=message('LFM2.5-Audio',result.text||'Spoken response');
      const bytes=Uint8Array.from(atob(result.audio_base64),c=>c.charCodeAt(0));
      const url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));urls.push(url);
      const player=document.createElement('audio');player.controls=true;player.src=url;players.push(player);article.append(player);
      const detail=document.createElement('p');detail.className='caption';detail.textContent=`Turn ${result.turn} · ${result.wall_seconds.toFixed(1)} s local inference · ${result.output.duration_seconds.toFixed(1)} s spoken reply${result.truncated?' · reply reached the length limit':''}`;article.append(detail);
      player.onplay=()=>{el('voice-stop').disabled=false;};player.onended=()=>{el('voice-stop').disabled=true;};
      try{await player.play();state('Reply ready. Record your next message when playback finishes.');}catch{state('Reply ready. Press play to hear it.');}
    }catch(error){state(error.message,true);}finally{busy=false;controls();}
  }
  async function wav(chunks,rate){
    const length=chunks.reduce((n,x)=>n+x.length,0);if(length<rate*.2)throw Error('Record at least a short sentence.');
    const merged=new Float32Array(length);let at=0;for(const chunk of chunks){merged.set(chunk,at);at+=chunk.length;}
    const offline=new OfflineAudioContext(1,Math.floor(length*24000/rate),24000);
    const source=offline.createBufferSource();source.buffer=offline.createBuffer(1,length,rate);source.buffer.copyToChannel(merged,0);source.connect(offline.destination);source.start();
    const rendered=await offline.startRendering(),samples=rendered.getChannelData(0),buffer=new ArrayBuffer(44+samples.length*2),v=new DataView(buffer);
    const str=(offset,text)=>{for(let i=0;i<text.length;i++)v.setUint8(offset+i,text.charCodeAt(i));};
    str(0,'RIFF');v.setUint32(4,36+samples.length*2,true);str(8,'WAVE');str(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,24000,true);v.setUint32(28,48000,true);v.setUint16(32,2,true);v.setUint16(34,16,true);str(36,'data');v.setUint32(40,samples.length*2,true);
    samples.forEach((x,i)=>v.setInt16(44+i*2,Math.max(-32768,Math.min(32767,Math.round(x*32767))),true));
    return new Blob([buffer],{type:'audio/wav'});
  }
  async function finish(){
    const take=recording;if(!take)return;recording=null;busy=true;controls();clearTimeout(take.timer);take.node.disconnect();take.source.disconnect();take.stream.getTracks().forEach(t=>t.stop());await take.context.close();el('record').textContent='Record voice';el('mic-level').value=0;
    try{await send(await wav(take.chunks,take.rate),true);}catch(error){state(error.message,true);busy=false;controls();}
  }
  el('record').onclick=async()=>{
    if(recording){await finish();return;}
    let stream,context;
    busy=true;controls();
    try{
      stopPlayback();
      stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true}});
      context=new AudioContext();await context.resume();const source=context.createMediaStreamSource(stream),node=context.createScriptProcessor(4096,1,1),chunks=[];
      node.onaudioprocess=e=>{const input=e.inputBuffer.getChannelData(0);chunks.push(new Float32Array(input));let energy=0;for(const x of input)energy+=x*x;el('mic-level').value=Math.min(1,Math.sqrt(energy/input.length)*8);};
      source.connect(node);node.connect(context.destination);recording={stream,context,source,node,chunks,rate:context.sampleRate,timer:setTimeout(finish,19000)};
      el('record').textContent='Stop and send';state('Recording on this Mac. Speak, then press Stop and send.');
    }catch(error){if(stream)stream.getTracks().forEach(t=>t.stop());if(context)await context.close();state(error.name==='NotAllowedError'?'Microphone permission is off. Allow it in the browser, or upload a WAV.':error.message,true);}finally{busy=false;controls();}
  };
  el('voice-load').onclick=async()=>{busy=true;controls();state('Loading the local speech model…');try{await api('/api/voice/load','{}');ready=true;el('inference-status').textContent='Native STS ready · English · turn-by-turn';state('Model ready. Record your voice or upload a WAV.');el('voice-load').textContent='Model loaded';}catch(e){state(e.message,true);}finally{busy=false;controls();}};
  el('voice-new').onclick=async()=>{stopPlayback();try{await api('/api/voice/reset','{}');resetNext=true;el('voice-messages').replaceChildren();urls.splice(0).forEach(u=>URL.revokeObjectURL(u));players.length=0;state('New conversation. Record a message.');}catch(e){state(e.message,true);}};
  el('voice-stop').onclick=stopPlayback;
  el('voice-file').onchange=async e=>{const file=e.target.files[0];if(file)await send(file);e.target.value='';};
  window.addEventListener('beforeunload',()=>{if(recording)recording.stream.getTracks().forEach(t=>t.stop());urls.forEach(u=>URL.revokeObjectURL(u));});
  fetch('/api/voice/status').then(r=>r.json()).then(s=>{ready=s.ready;el('inference-status').textContent=s.ready?'Native STS ready · English · turn-by-turn':s.assets_ready?'Native model installed · click Load voice model':'Native model assets are not installed';controls();if(ready)state('Model ready. Record your voice or upload a WAV.');}).catch(()=>state('Local speech service unavailable',true));
})();
