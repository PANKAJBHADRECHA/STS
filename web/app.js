const $ = id => document.getElementById(id);
const names = {baseline:'Baseline',task_memory:'Task memory',validated_execution:'Validated execution',playback_recovery:'Playback recovery',combined:'Combined'};
const familyNames = {argument_corrections:'Argument corrections',tool_failures:'Tool failures',interruptions:'Interruptions',delayed_feedback:'Delayed feedback',long_sessions:'Long sessions'};
const esc = value => String(value).replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
async function request(url, data) {
  const response = await fetch(url, data === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Request failed');
  return result;
}
function status(text, error=false) { $('status').textContent=text; $('status').classList.toggle('bad',error); }
function render(report) {
  const winner=report.selection.winner;
  if (winner && names[winner] && report.generator_version === '1.1.0') $('variant').value=winner;
  const best=report.summaries.find(s=>s.variant===winner);
  $('results').innerHTML=`<div class="metrics"><div class="metric"><span>Selected controller</span><strong>${esc(names[winner]||'No eligible variant')}</strong></div><div class="metric"><span>Paired controller runs</span><strong>${report.run_count.toLocaleString()}</strong></div><div class="metric"><span>Gain over baseline</span><strong>${(report.selection.gain_over_baseline*100).toFixed(1)} pp</strong></div></div><div class="table-scroll"><table><caption class="caption">Synthetic held-out tasks · ${report.scenario_count} cases per variant · seeds ${report.seeds.join(', ')} · no audio inference</caption><thead><tr><th>Architecture</th><th>Task completion</th><th>Argument accuracy</th><th>Duplicate actions</th><th>Unsupported claims</th><th>Controller p95</th></tr></thead><tbody>${report.summaries.map(s=>`<tr class="${s.variant===winner?'winner':''}"><td class="variant-name">${esc(names[s.variant])}${s.variant===winner?'<span class="tag">SELECTED</span>':''}</td><td>${(s.completion_rate*100).toFixed(1)}% <span class="caption">${s.passed}/${s.total}</span><progress max="100" value="${s.completion_rate*100}" aria-label="Task completion percent"></progress></td><td>${s.argument_accuracy===null?'—':(s.argument_accuracy*100).toFixed(1)+'%'}</td><td class="${s.duplicate_actions?'bad':''}">${s.duplicate_actions}</td><td class="${s.unsupported_claims?'bad':''}">${s.unsupported_claims}</td><td>${s.controller_runtime_p95_ms.toFixed(2)} ms</td></tr>`).join('')}</tbody></table></div>${best?`<h3>Selected controller by failure family</h3><div class="family-grid">${Object.entries(best.by_family).map(([key,value])=>`<div><p>${esc(familyNames[key])}</p><strong>${value.passed}/${value.total}</strong></div>`).join('')}</div>`:''}<p class="caption">Manifest SHA-256: ${esc(report.scenario_manifest_sha256.slice(0,20))}… · ${esc(new Date(report.generated_at).toLocaleString())}</p>`;
}
async function busy(button, action) {button.disabled=true;try{await action();}catch(e){status(e.message,true);}finally{button.disabled=false;}}
$('run').onclick=()=>busy($('run'),async()=>{status('Running every variant against all five scenario families…');render(await request('/api/evaluate',{}));status('Evaluation complete. Results and failure traces saved locally.');});
$('replay-run').onclick=()=>busy($('replay-run'),async()=>{
  const result=await request('/api/replay',{family:$('family').value,variant:$('variant').value,index:Number($('case-index').value)});
  const el=$('replay-result');el.innerHTML=`<div class="metrics"><div class="metric"><span>Task result</span><strong>${result.passed?'Passed':'Failed'}</strong></div><div class="metric"><span>Final appointment</span><strong>${esc(result.final_state.slot)}</strong><span>${esc(result.final_state.status)}</span></div><div class="metric"><span>Timeline</span><strong>${(result.duration_ms/1000).toFixed(1)} s</strong><span>Simulated time</span></div></div><p class="caption">${esc(result.scenario)} · ${result.trace.length} events · ${result.duplicate_actions} duplicate actions</p><div class="trace"></div>`;
  const trace=el.querySelector('.trace');
  for(const item of result.trace) {const row=document.createElement('div');row.className='trace-item'; const time=document.createElement('time');time.textContent=`${item.at_ms} ms`;const body=document.createElement('div');const title=document.createElement('strong');title.textContent=item.kind;const pre=document.createElement('pre');const {at_ms,kind,...payload}=item;pre.textContent=JSON.stringify(payload,null,2);body.append(title);if(Object.keys(payload).length)body.append(pre);row.append(time,body);trace.append(row);}
  status('Replay complete. Playback events are simulated acknowledgements.');
});
let viewRequest = 0;
document.querySelectorAll('nav button').forEach(button=>button.onclick=async()=>{
  const token=++viewRequest;
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b===button));
  document.querySelectorAll('.view').forEach(v=>v.hidden=true);
  const name=button.dataset.view;
  if(['voice','experiment','replay'].includes(name)){$(name).hidden=false;return;}
  $('document').hidden=false;$('document-title').textContent=button.textContent;$('document-body').textContent='Loading document…';
  try {const doc=await request('/api/document?name='+encodeURIComponent(name));if(token===viewRequest)$('document-body').textContent=doc.text;} catch(e) {if(token===viewRequest)status(e.message,true);}
});
async function init() {
  try{render(await request('/api/report'));status('Saved evaluation loaded. Rerun or inspect any scenario.');}catch(e){status('Ready. Run all experiments to generate a comparison.');}
}
init();
