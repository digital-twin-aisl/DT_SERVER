// SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
// SPDX-License-Identifier: LGPL-2.1-or-later
const $ = id => document.getElementById(id);
const states = {running:'동작 중', starting:'준비 중', validating:'검증 중', degraded:'일부 장애', failed:'실패', stopped:'중지됨', stopping:'중지 중', completed:'완료', interrupted:'관리 서비스 중단'};
let regions = [], selected = null, busy = false, refreshing = false;
const enc = encodeURIComponent;
function node(tag, text, cls) { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; }
function badge(state, text) { return node('span', text || states[state] || state, 'badge ' + state); }
function button(text, action, cls='subtle') { const n = node('button', text, cls); n.onclick = () => perform(action); return n; }
function error(message) { $('error').hidden = !message; $('error').textContent = message || ''; }
async function api(path, method='GET', body) {
  const response = await fetch('/api/v1/control' + path, {method, headers:{'Content-Type':'application/json'}, body:body === undefined ? undefined : JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
async function perform(action) { if (busy) return; busy = true; error(''); try { await action(); await refresh(); } catch(e) { error(e.message); } finally { busy = false; } }
function path(suffix='') { if (!selected) throw new Error('구역을 선택하세요.'); return '/regions/' + enc(selected) + suffix; }
async function logs(edge) { const result = await api(path('/logs') + (edge ? '?edge_id=' + enc(edge) : '')); $('log-text').textContent = result.text || '아직 로그가 없습니다.'; $('log-dialog').showModal(); }
function render(region) {
  $('region-name').textContent = region.name; $('region-id').textContent = region.region_id;
  $('region-state').replaceWith(Object.assign(badge(region.state), {id:'region-state'}));
  const age = region.last_scene_age_seconds;
  $('region-detail').textContent = `${region.edges.length}개 엣지 · ${region.edges.reduce((n,e)=>n+e.cameras.length,0)}개 카메라 · ${age == null ? '아직 장면을 받지 못했습니다.' : `마지막 장면 ${age.toFixed(1)}초 전`} ${region.recording ? '· 기록 중' : ''} ${region.operation === 'replay' ? '· 기록 재생' : ''}`;
  $('runtime-error').textContent = region.error || region.worker?.error || region.recording_status?.error || (region.recording_status?.dropped ? `기록 대기열에서 ${region.recording_status.dropped}개 장면이 누락되었습니다.` : '');
  for (const id of ['start','stop','view','logs','settings']) $(id).disabled = false;
  $('edges').replaceChildren(...region.edges.map(edge => {
    const card = node('article', null, 'edge');
    const title = node('div', null, 'section-title'); title.append(node('h2', edge.edge_id), badge(edge.online ? 'online':'offline',edge.online?'장치 연결됨':'장치 연결 끊김'));
    const description = node('p', `${edge.approved ? '승인됨':'승인 대기'} · ${edge.data_fresh ? '관측 수신 중':'새 관측 없음'} · ${states[edge.inference?.stage] || edge.inference?.stage || '추론 대기'}`, 'muted');
    const table = node('table'); const head = node('tr'); ['카메라','연결','보정 정보'].forEach(t=>head.append(node('th',t))); table.append(head);
    for (const camera of edge.cameras) { const row = node('tr'), status = camera.status || {}, cal = status.calibration || {}; row.append(node('td',camera.camera_id),node('td',status.exists == null ? '미확인':status.ping?'연결됨':'연결 안 됨'),node('td',cal.intrinsic && cal.extrinsic ? '있음':'확인 필요')); table.append(row); }
    const actions = node('div', null, 'toolbar');
    actions.append(button('로그', ()=>logs(edge.edge_id)),button('카메라 보정', ()=>{const form=$('calibration-form'); form.elements.edge_id.value=edge.edge_id; $('calibration-dialog').showModal();}),button(edge.approved?'승인 해제':'승인',()=>api('/edges/'+enc(edge.edge_id)+'/'+(edge.approved?'revoke':'approve'),'POST',{})));
    card.append(title,description,table,actions);
    if(edge.error || edge.inference?.error) card.append(node('p',edge.error || edge.inference.error,'error-text'));
    return card;
  }));
}
async function refresh() {
  if (refreshing) return; refreshing = true;
  try {
    const [next, devices] = await Promise.all([api('/regions'), api('/edges')]); regions = next;
    $('connection').textContent = '관리 서비스 연결됨';
    if (!regions.some(r=>r.region_id===selected)) selected=regions[0]?.region_id || null;
    $('regions').replaceChildren(...regions.map(r=>{ const b=button(r.name,async()=>{selected=r.region_id; render(r); await loadRecordings();},r.region_id===selected?'selected':''); return b; }));
    const region=regions.find(r=>r.region_id===selected); if(region) render(region);
    else { $('region-name').textContent='등록된 구역이 없습니다.'; $('edges').replaceChildren(); for(const id of ['start','stop','view','logs','settings']) $(id).disabled=true; }
    $('discovered').replaceChildren(...devices.map(d=>{const row=node('div',null,'device'); row.append(node('strong',d.display_name || d.edge_id),node('small',`${d.edge_id} · ${d.online?'연결됨':'연결 끊김'} · ${d.approved?'승인됨':'승인 대기'}`),button(d.approved?'승인 해제':'승인',()=>api('/edges/'+enc(d.edge_id)+'/'+(d.approved?'revoke':'approve'),'POST',{}))); return row; }));
    if (!devices.length) $('discovered').textContent='아직 발견된 장치가 없습니다.';
  } catch(e) { $('connection').textContent='관리 서비스 연결 끊김'; $('region-state').replaceWith(Object.assign(badge('unknown','상태 확인 불가'),{id:'region-state'})); $('region-detail').textContent='관리 서비스의 응답이 없어 현재 동작 상태를 확인할 수 없습니다.'; error(e.message); for(const id of ['start','stop','view','logs','settings']) $(id).disabled=true; }
  finally { refreshing=false; }
}
async function loadRecordings() {
  if(!selected)return;
  const regionId=selected, base='/regions/'+enc(regionId);
  const data=await api(base+'/recordings');
  if(selected!==regionId)return;
  $('recordings').replaceChildren(...data.map(record=>{
    const row=node('div',null,'record-row'), description=node('div');
    description.append(node('strong',`${record.operation} · ${states[record.status] || record.status}`),node('small',new Date(record.started_at*1000).toLocaleString()+' · '+record.run_id));
    const actions=node('div',null,'toolbar');
    if(record.scene_bytes>0){
      const preview=node('a','뷰어 재생');preview.href='/viewer?region='+enc(regionId)+'&recording='+enc(record.run_id);preview.target='_blank';preview.rel='noopener';preview.title='서버 실행을 바꾸지 않고 새 뷰어에서만 재생';
      const replay=button('구역 재생',()=>api(base+'/replay','POST',{recording_id:record.run_id}));replay.title='기존 관리 서비스 재생: 구역의 공유 장면 토픽으로 송출';
      const link=node('a','다운로드');link.href='/api/v1/control'+base+'/recordings/'+enc(record.run_id)+'/download';
      actions.append(node('small',`${(record.scene_bytes/1024/1024).toFixed(2)} MB`),preview,replay,link);
    }
    row.append(description,actions);return row;
  }));
  if(!data.length)$('recordings').textContent='아직 실행한 작업이 없습니다.';
}
$('start').onclick=()=>perform(async()=>{await api(path('/start'),'POST',{record:$('record').checked});await loadRecordings();});
$('stop').onclick=()=>perform(()=>api(path('/stop'),'POST',{}));
$('logs').onclick=()=>perform(()=>logs());
$('view').onclick=()=>{$('viewer-panel').hidden=false;$('viewer').src='/viewer?region='+enc(selected);$('viewer-panel').scrollIntoView({behavior:'smooth'});};
$('close-view').onclick=()=>{$('viewer-panel').hidden=true;$('viewer').src='about:blank';};
$('refresh-recordings').onclick=()=>perform(loadRecordings);
for(const close of document.querySelectorAll('[data-close]'))close.onclick=()=>close.closest('dialog').close();
$('calibration-form').onsubmit=e=>{e.preventDefault();perform(async()=>{const data=Object.fromEntries(new FormData(e.target));await api(path('/calibrate'),'POST',data);$('calibration-dialog').close();await loadRecordings();});};
function editRegion(existing) {const f=$('region-form');f.reset(); f.elements.region_id.readOnly=Boolean(existing);for(const key of ['region_id','name','deployment','server_profile','scene_topic'])f.elements[key].value=existing?.[key] || ''; $('remove-region').hidden=!existing;$('region-dialog').showModal();}
$('register-region').onclick=()=>editRegion(null);
$('settings').onclick=()=>editRegion(regions.find(r=>r.region_id===selected));
$('region-form').onsubmit=e=>{e.preventDefault();perform(async()=>{const body=Object.fromEntries(new FormData(e.target)),id=body.region_id;delete body.region_id;if(!body.scene_topic)delete body.scene_topic;await api('/regions/'+enc(id),'PUT',body);selected=id;$('region-dialog').close();});};
$('remove-region').onclick=()=>perform(async()=>{await api('/regions/'+enc($('region-form').elements.region_id.value),'DELETE');$('region-dialog').close();});
await refresh();await perform(loadRecordings);setInterval(refresh,3000);
