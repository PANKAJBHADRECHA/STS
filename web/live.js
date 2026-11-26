const $=id=>document.getElementById(id);
let ws=null,activeEpoch=-1,playing=false,lastError='',audioContext=null,audioNode=null,micStream=null,engineReady=false,audioReady=null;
window.liveMetrics={captureFrames:0,outputFrames:0,captureDuringPlayback:0,cancellations:[],replies:[],events:[],audioStarts:[],audioStats:[],deviceStatusEvents:0,synthesisStarvationFrames:0};
const assistant=new Map();
function send(e){if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify(e));}
function status(title,detail){$('status').textContent=title;if(detail)$('detail').textContent=detail;}
function row(role,text){$('messages').querySelector('.empty')?.remove();const item=document.createElement('div');item.className='message '+role;const label=document.createElement('small');label.textContent=role==='user'?'YOU':'ASSISTANT';const body=document.createElement('span');body.textContent=text;item.append(label,body);$('messages').append(item);$('messages').scrollTop=$('messages').scrollHeight;return body;}
function eventMetric(e){liveMetrics.events.push({type:e.type,epoch:e.epoch,at_ms:performance.now()});if(liveMetrics.events.length>2000)liveMetrics.events.shift();}
function playbackStart(e){if(e.epoch!==activeEpoch)return;playing=true;$('speaking').textContent='● Speaking';$('orb').classList.add('active');status('Speaking','You can interrupt — I’m still listening.');if(!liveMetrics.audioStarts.some(x=>x.epoch===e.epoch))liveMetrics.audioStarts.push({epoch:e.epoch,at_ms:performance.now()});}
function playbackEnd(e){if(e.epoch!==activeEpoch)return;playing=false;$('speaking').textContent='○ Quiet';$('listening').textContent='● Listening';$('orb').classList.remove('active');status('Listening','You can continue or ask something new.');}
async function stopBrowserAudio(){
  const stream=micStream,context=audioContext;micStream=null;audioContext=null;audioNode=null;engineReady=false;
  stream?.getTracks().forEach(t=>t.stop());if(context&&context.state!=='closed')await context.close();
  if(stream)liveMetrics.microphoneReleased=stream.getTracks().every(t=>t.readyState==='ended');
}
async function finish(message='Conversation ended'){
  const old=ws;ws=null;old?.close();await stopBrowserAudio();playing=false;$('start').disabled=false;for(const id of ['stop','interrupt','reset','text','send'])$(id).disabled=true;
  $('listening').textContent='● Microphone off';$('speaking').textContent='○ Quiet';$('orb').classList.remove('active');$('level').value=0;status(lastError||message,lastError?'Check setup and microphone permission, then start again.':'Start again whenever you’re ready.');
}
async function prepareBrowserAudio(socket){
  liveMetrics.audioStage='resume';const context=audioContext;await context.resume();
  liveMetrics.audioStage='microphone';
  const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
  if(socket!==ws){stream.getTracks().forEach(t=>t.stop());await context.close();return;}
  micStream=stream;liveMetrics.audioStage='worklet module';await context.audioWorklet.addModule('/live-worklet.js');if(socket!==ws)return;
  liveMetrics.audioStage='worklet node';const node=new AudioWorkletNode(context,'duplex-audio',{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1]});audioNode=node;
  liveMetrics.audioStage='connect microphone';const source=context.createMediaStreamSource(stream);source.connect(node);node.connect(context.destination);
  const track=stream.getAudioTracks()[0],settings=track.getSettings();liveMetrics.captureSettings=Object.fromEntries(["echoCancellation","noiseSuppression","autoGainControl","sampleRate","channelCount","latency"].filter(k=>k in settings).map(k=>[k,settings[k]]));liveMetrics.microphoneReleased=false;liveMetrics.audioContextRate=context.sampleRate;
  liveMetrics.audioStage='device inventory';const devices=await navigator.mediaDevices.enumerateDevices();liveMetrics.devices={input:track.label,output:devices.find(d=>d.kind==='audiooutput'&&d.deviceId==='default')?.label||'macOS default output'};
  node.port.onmessage=({data:e})=>{
    if(socket!==ws)return;
    if(e.type==='capture'){
      liveMetrics.captureFrames=e.frames;liveMetrics.outputFrames=e.output_frames;liveMetrics.captureDuringPlayback=e.overlap_frames;$('level').value=e.rms;
      if(engineReady&&socket.readyState===WebSocket.OPEN){if(socket.bufferedAmount>16000*4*2){lastError='Audio transport fell behind; start again';liveMetrics.lastError=lastError;send({type:'stop'});stopBrowserAudio();}else socket.send(e.pcm);}return;
    }
    eventMetric(e);
    if(e.type==='playback_suppressed'){status('Listening','Go ahead — I’m listening.');$('listening').textContent='● Hearing you';}
    if(e.type==='ack')send(e);
    if(e.type==='playback_start')playbackStart(e);
    if(e.type==='playback_end'){send({type:'played',epoch:e.epoch});playbackEnd(e);}
    if(e.type==='error'){lastError=e.message;liveMetrics.lastError=e.message;send({type:'stop'});}
  };
}
$('start').onclick=()=>{
  lastError='';engineReady=false;audioContext=new AudioContext({latencyHint:'interactive'});audioContext.resume();$('start').disabled=true;$('stop').disabled=false;status('Warming up…','Preparing the local voice models and audio.');
  const socket=new WebSocket(`ws://${location.hostname}:${Number(location.port)+1}`);socket.binaryType='arraybuffer';ws=socket;
  // AudioContext is unlocked in the click handler; capture is only enabled
  // for a browser-audio ready message after the models finish loading.
  audioReady=null;
  socket.onclose=e=>{if(ws===socket)finish(e.reason||'Connection closed');};socket.onerror=()=>{if(ws===socket)finish('Could not connect to the local engine');};
  socket.onmessage=async event=>{
    if(socket!==ws)return;
    if(typeof event.data!=='string'){
      const view=new DataView(event.data),epoch=view.getUint32(0),chunk=view.getUint32(4),rate=view.getUint32(8);const samples=new Float32Array(event.data.slice(12));
      audioNode?.port.postMessage({type:'audio',epoch,chunk,rate,samples},[samples.buffer]);return;
    }
    const e=JSON.parse(event.data);eventMetric(e);
    if(e.type==='audio_mode'&&e.audio==='browser'){
      audioReady=prepareBrowserAudio(socket).catch(error=>{lastError=liveMetrics.audioStage+': '+error.name+': '+error.message;console.error(error);liveMetrics.lastError=lastError;send({type:'stop'});throw error;});audioReady.catch(()=>{});
    }
    if(e.type==='stopped'){liveMetrics.engineMetrics=e.metrics;if(!micStream)liveMetrics.microphoneReleased=e.microphone_released;await finish();}
    if(e.type==='ready'){
      if(e.browser_audio){try{await audioReady;}catch{return;}if(socket!==ws)return;engineReady=true;}
      else liveMetrics.devices=e.devices;
      liveMetrics.configuration={model:e.model,asr:e.asr,voice:e.voice,audio:e.browser_audio?'browser':'native'};liveMetrics.speakersMuted=e.speakers_muted;
      $('devices').textContent=`Mic: ${liveMetrics.devices.input} · Output: ${liveMetrics.devices.output}`;for(const id of ['interrupt','reset','text','send'])$(id).disabled=false;
      $('listening').textContent='● Listening';status('Listening',e.speakers_muted?'The selected output is muted in macOS. Unmute to hear replies.':'Speak naturally. The microphone stays active while I speak.');
    }
    if(e.type==='capture'){liveMetrics.captureFrames=e.frames;liveMetrics.outputFrames=e.output_frames;liveMetrics.captureDuringPlayback=e.overlap_frames;liveMetrics.deviceStatusEvents=e.device_status_events;liveMetrics.synthesisStarvationFrames=e.synthesis_starvation_frames;}
    if(e.type==='audio_stats'){liveMetrics.audioStats.push(e);if(liveMetrics.audioStats.length>600)liveMetrics.audioStats.shift();if(!audioNode)$('level').value=e.raw_rms;}
    if(e.type==='speech_start'){audioNode?.port.postMessage({type:'gain',value:.25,server:true});status('Listening','Go ahead — I’m listening.');$('listening').textContent='● Hearing you';}
    if(e.type==='duck_restore')audioNode?.port.postMessage({type:'gain',value:1,server:true});
    if(e.type==='partial')$('partial').textContent=e.text;
    if(e.type==='user'){row('user',e.text);$('partial').textContent='';status('Thinking','Preparing a short reply.');}
    if(e.type==='reply_start'){activeEpoch=e.epoch;audioNode?.port.postMessage(e);assistant.set(e.epoch,row('assistant','…'));}
    if(e.type==='playback_start')playbackStart(e);
    if(e.type==='playback_end'||e.type==='listening')playbackEnd(e);
    if(e.type==='reply_done'){audioNode?.port.postMessage(e);liveMetrics.replies.push(e);const body=assistant.get(e.epoch);if(body)body.textContent=e.text;$('metrics').textContent=`First audio ${Math.round((e.first_audio_s||0)*1000)} ms after recognition · all inference is local`;}
    if(e.type==='cancel'){activeEpoch=e.next_epoch;audioNode?.port.postMessage(e);if(playing){liveMetrics.cancellations.push({at_ms:performance.now(),stopped_ms:null});playing=false;$('speaking').textContent='○ Quiet';$('orb').classList.remove('active');const body=assistant.get(e.epoch);if(body){const note=document.createElement('em');note.textContent=' · interrupted';body.parentElement.append(note);}}}
    if(e.type==='reset'){$('messages').replaceChildren();assistant.clear();$('partial').textContent='';status('Listening','A new conversation.');}
    if(e.type==='error'){lastError=e.message;liveMetrics.lastError=e.message;status('Conversation stopped',e.message);send({type:'stop'});}
  };
};
$('stop').onclick=()=>{stopBrowserAudio();send({type:'stop'});status('Ending…','Releasing the microphone and speaker.');};$('interrupt').onclick=()=>send({type:'interrupt'});$('reset').onclick=()=>send({type:'reset'});
$('text-form').onsubmit=e=>{e.preventDefault();const text=$('text').value.trim();if(text){send({type:'text',text});$('text').value='';}};
window.addEventListener('pagehide',()=>{micStream?.getTracks().forEach(t=>t.stop());ws?.close();});
