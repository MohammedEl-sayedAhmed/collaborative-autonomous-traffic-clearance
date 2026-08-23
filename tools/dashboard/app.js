const PALETTE = ["#4fd1c5","#e0b64a","#e06c6c","#7aa2f7","#9ece6a","#bb9af7","#ff9e64","#2ac3de","#f7768e","#73daca"];
const OUTCOME = {0:"in progress",1:"max time",2:"ambulance goal",3:"agent goal",4:"sim died"};
const OUTCOME_COLOR = {1:"#e0b64a",2:"#46c37b",3:"#4fd1c5",4:"#e06c6c",0:"#9aa7bd"};

// Single source of truth: `selected` decides what is shown. No competing "focus".
const state = {runs:[], data:{}, selected:new Set(), colors:{}, first:true};
const chartView = {};   // per-chart zoom domain {x0,x1,y0,y1}; undefined = auto-fit
const chartGeom = {};   // per-chart last-draw geometry, for pixel<->data mapping
const chartUndo = {};   // per-chart view-history stacks (matplotlib-style Back)
let interacting = false; // pause polling while the user is dragging/zooming a chart

// matplotlib-style navigation: every view change pushes the prior view so Back can restore it
function pushView(id, v){ (chartUndo[id] || (chartUndo[id] = [])).push(chartView[id]); chartView[id] = v; syncNav(); drawAll(); }
function backView(id){ const s = chartUndo[id]; if(s && s.length){ chartView[id] = s.pop(); syncNav(); drawAll(); } }
function homeView(id){ if(chartView[id] !== undefined) pushView(id, undefined); }
function canBack(id){ return !!(chartUndo[id] && chartUndo[id].length); }
// enable/disable the per-chart Back buttons based on whether history exists
function syncNav(){ document.querySelectorAll("[data-back]").forEach(b=>{ b.disabled = !canBack(b.getAttribute("data-back")); }); }

const $ = s => document.querySelector(s);
const fmt = (v,d=2) => (v==null||isNaN(v)) ? "—" : (+v).toFixed(d);
const shortSha = s => (s&&s!=="unknown") ? s.slice(0,7) : "—";
function escapeHtml(s){ return (s||"").replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function rem(){ return parseFloat(getComputedStyle(document.documentElement).fontSize) || 15; }  // px per rem (fluid)

function colorFor(id){
  if(!(id in state.colors)) state.colors[id] = PALETTE[Object.keys(state.colors).length % PALETTE.length];
  return state.colors[id];
}
function ageSecs(t){ return t? (Date.now()/1000 - t) : null; }
function statusBadge(r){
  if(r.mode==="test") return `<span class="badge b-test">test</span>`;
  const age = ageSecs(r.heartbeat);
  if(r.status==="running" && age!=null && age<15) return `<span class="badge b-live">LIVE</span>`;
  if(r.status==="running") return `<span class="badge b-stale">stalled</span>`;
  return `<span class="badge b-done">done</span>`;
}
async function jget(u){ const r = await fetch(u,{cache:"no-store"}); return r.json(); }

function refresh(){ renderRunList(); renderTiles(); renderTable(); drawAll(); pollSelected(); }
function setSelected(id, on){ if(on) state.selected.add(id); else state.selected.delete(id); refresh(); }
function toggleSelected(id){ setSelected(id, !state.selected.has(id)); }

async function pollRuns(){
  if($("#pauseChk").checked || interacting) return;
  let payload;
  try{ payload = await jget("/api/runs"); }catch(e){ $("#pollDot").style.background="var(--bad)"; $("#pollDot").className="dot"; return; }
  $("#pollDot").style.background=""; $("#pollDot").className="dot pulse";
  state.runs = payload.runs || [];
  $("#runsDir").textContent = payload.runs_dir || "";
  $("#runCount").textContent = "("+state.runs.length+")";
  if(state.first && state.runs.length){
    const live = state.runs.filter(r=>r.live);
    (live.length?live:[state.runs[state.runs.length-1]]).forEach(r=>state.selected.add(r.run_id));
    state.first=false;
  }
  renderRunList();
  await pollSelected();
  renderTiles(); renderTable();
  $("#pollInfo").textContent = "updated "+new Date().toLocaleTimeString();
}

async function pollSelected(){
  const ids = [...state.selected];
  await Promise.all(ids.map(async id=>{ try{ state.data[id] = await jget("/api/run?id="+encodeURIComponent(id)); }catch(e){} }));
  // re-render everything that depends on the fetched per-run data, so tiles/table
  // don't lag the charts when a newly-selected run's data arrives.
  drawAll(); renderTiles(); renderTable();
}

function renderRunList(){
  const el = $("#runlist");
  if(!state.runs.length){ el.innerHTML = `<div class="empty">No runs yet.<br/>Start training with <code>./run.sh rl</code><br/>or seed demos with <code>./run.sh dashboard-demo</code>.</div>`; return; }
  el.innerHTML = "";
  state.runs.forEach(r=>{
    const sel = state.selected.has(r.run_id);
    const row = document.createElement("div");
    row.className = "runrow"+(sel?" sel":"");
    row.setAttribute("data-run", r.run_id);
    row.innerHTML = `
      <input type="checkbox" class="chk" tabindex="-1" ${sel?"checked":""}/>
      <span class="sw" style="background:${colorFor(r.run_id)};${sel?'':'opacity:.28'}"></span>
      <span class="meta">
        <div class="lbl">${escapeHtml(r.label)}${statusBadge(r)}</div>
        <div class="sub">${shortSha(r.git_sha)} · ${r.num_episodes} ep · final ${fmt(r.final_reward)}</div>
      </span>`;
    row.addEventListener("click", ()=> toggleSelected(r.run_id));
    el.appendChild(row);
  });
}

function selectedRuns(){
  return [...state.selected]
    .map(id=>({id, summary:state.runs.find(r=>r.run_id===id), detail:state.data[id]}))
    .filter(x=>x.summary && x.detail && x.detail.episodes);
}

function renderTiles(){
  const runs = selectedRuns();
  const el = $("#tiles"); el.innerHTML="";
  const liveRun = runs.find(r=>r.detail.status && r.detail.status.state==="running");
  if(liveRun){
    const s = liveRun.detail.status, age = ageSecs(s.time);
    el.appendChild(tile("● Live: "+escapeHtml(liveRun.summary.label),
      `ep ${s.episode} · step ${s.step}`, `cum ${fmt(s.cum_reward)} · ε ${fmt(s.epsilon,3)}${age!=null?" · "+Math.round(age)+"s ago":""}`));
  }
  if(runs.length===1){
    const rw = runs[0].detail.episodes.map(e=>e.cum_reward);
    el.appendChild(tile("Episodes", runs[0].detail.episodes.length, ""));
    el.appendChild(tile("Final reward", fmt(rw[rw.length-1]), ""));
    el.appendChild(tile("Best reward", fmt(rw.length?Math.max(...rw):null), ""));
    el.appendChild(tile("Mean reward", fmt(rw.length?rw.reduce((a,b)=>a+b,0)/rw.length:null), ""));
    el.appendChild(tile("Success", fmt(100*successRate(runs[0].detail.episodes),0)+"%", "amb. reached goal"));
  } else if(runs.length>1){
    el.appendChild(tile("Runs compared", runs.length, ""));
    el.appendChild(tile("Total episodes", runs.reduce((a,r)=>a+r.detail.episodes.length,0), ""));
  }
}
function tile(k,v,sub){ const d=document.createElement("div"); d.className="tile";
  d.innerHTML=`<div class="k">${k}</div><div class="v">${v} ${sub?`<small>${sub}</small>`:""}</div>`; return d; }
function successRate(eps){ return eps.length? eps.filter(e=>e.outcome===2).length/eps.length : 0; }

function movingAvg(vals,w){
  if(w<=1) return vals.slice();
  const out=[]; const h=Math.floor(w/2);
  for(let i=0;i<vals.length;i++){ let s=0,c=0;
    for(let j=Math.max(0,i-h);j<=Math.min(vals.length-1,i+h);j++){s+=vals[j];c++;} out.push(s/c); }
  return out;
}

let maximized = null;     // id of the chart currently shown maximized, or null
let CHARTS = {};          // registry so the maximize view can re-render the same chart

function drawAll(){
  const runs = selectedRuns();
  const smooth = $("#smoothChk").checked ? +$("#smoothWin").value : 1;
  const lineOf = (pick, opts) => ({ type:"line", opts: opts||{},
    series: runs.map(r=>({label:r.summary.label, color:colorFor(r.id), points:pick(r)})) });
  CHARTS = {
    chartReward: Object.assign(lineOf(r=>{ const ys=movingAvg(r.detail.episodes.map(e=>e.cum_reward),smooth);
      return r.detail.episodes.map((e,i)=>[e.episode, ys[i]]); }, {}), {title:"Cumulative reward per episode"}),
    chartEps: Object.assign(lineOf(r=>r.detail.episodes.map(e=>[e.episode,e.epsilon]), {ymin:0}), {title:"Epsilon (exploration) per episode"}),
    chartSteps: Object.assign(lineOf(r=>r.detail.episodes.map(e=>[e.episode,e.num_steps]), {ymin:0}), {title:"Steps per episode"}),
    chartOutcomes: {type:"outcome", runs, title:"Episode outcomes"},
  };
  lineChart($("#chartReward"), CHARTS.chartReward.series, CHARTS.chartReward.opts);
  legend($("#legReward"), CHARTS.chartReward.series);
  lineChart($("#chartEps"), CHARTS.chartEps.series, CHARTS.chartEps.opts);
  lineChart($("#chartSteps"), CHARTS.chartSteps.series, CHARTS.chartSteps.opts);
  outcomeChart($("#chartOutcomes"), runs);
  if(maximized) renderMaximized();
}

function renderMaximized(){
  const c = CHARTS[maximized]; if(!c) return;
  $("#chartBigTitle").textContent = c.title;
  const big = $("#chartBig");
  if(c.type==="outcome"){ legend($("#legBig"), []); $("#legBig").style.display="none"; outcomeChart(big, c.runs); }
  else { $("#legBig").style.display=""; legend($("#legBig"), c.series); lineChart(big, c.series, c.opts); }
}
function openMaximized(id){ maximized=id; chartView["chartBig"]=undefined; chartUndo["chartBig"]=[]; $("#chartModal").classList.add("open"); renderMaximized(); syncNav(); }
function closeMaximized(){ maximized=null; $("#chartModal").classList.remove("open"); }
function legend(el, series){ el.innerHTML = series.map(s=>`<span><span class="swatch" style="background:${s.color}"></span>${escapeHtml(s.label)}</span>`).join(""); }

function lineChart(svg, series, opts={}){
  const id=svg.id;
  const W=svg.clientWidth||560, H=svg.clientHeight||220, FS=(rem()*0.74), P={l:rem()*3.2,r:rem()*1,t:rem()*.8,b:rem()*1.9};
  const pts = series.flatMap(s=>s.points).filter(p=>p[1]!=null && !isNaN(p[1]));
  if(!pts.length){ svg.innerHTML=`<text x="${W/2}" y="${H/2}" fill="#9aa7bd" text-anchor="middle" font-size="${rem()*.85}">no data yet</text>`; chartGeom[id]=null; return; }
  let xmin,xmax,ymin,ymax;
  const view = chartView[id];
  if(view){ xmin=view.x0; xmax=view.x1; ymin=view.y0; ymax=view.y1; }
  else {
    const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    xmin=Math.min(...xs); xmax=Math.max(...xs);
    ymin=(opts.ymin!=null?opts.ymin:Math.min(...ys)); ymax=Math.max(...ys);
    if(xmax===xmin) xmax=xmin+1; if(ymax===ymin) ymax=ymin+1;
    const pad=(ymax-ymin)*0.08; ymin-=(opts.ymin!=null?0:pad); ymax+=pad;
  }
  const plotW=W-P.l-P.r, plotH=H-P.t-P.b;
  const X=x=>P.l+(x-xmin)/(xmax-xmin)*plotW;
  const Y=y=>H-P.b-(y-ymin)/(ymax-ymin)*plotH;
  chartGeom[id]={W,H,P,xmin,xmax,ymin,ymax,plotW,plotH,zoomed:!!view};
  let g=`<g font-size="${FS.toFixed(1)}" fill="#9aa7bd">`;
  for(let i=0;i<=4;i++){ const yv=ymin+(ymax-ymin)*i/4, y=Y(yv);
    g+=`<line x1="${P.l}" y1="${y.toFixed(1)}" x2="${W-P.r}" y2="${y.toFixed(1)}" stroke="#232e47"/>`;
    g+=`<text x="${(P.l-FS*.5).toFixed(1)}" y="${(y+FS*.34).toFixed(1)}" text-anchor="end">${fmtAxis(yv)}</text>`; }
  const nT=Math.min(6,Math.max(1,Math.round(xmax-xmin)));
  for(let i=0;i<=nT;i++){ const xv=xmin+(xmax-xmin)*i/nT, x=X(xv);
    g+=`<text x="${x.toFixed(1)}" y="${(H-FS*.6).toFixed(1)}" text-anchor="middle">${Math.round(xv)}</text>`; }
  g+=`</g>`;
  const clip="clip_"+id;
  let out=`<defs><clipPath id="${clip}"><rect x="${P.l}" y="${P.t}" width="${plotW}" height="${plotH}"/></clipPath></defs>`+g;
  out+=`<g clip-path="url(#${clip})">`;
  series.forEach(s=>{
    const P2=s.points.filter(p=>p[1]!=null&&!isNaN(p[1]));
    if(!P2.length) return;
    const d=P2.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+" "+Y(p[1]).toFixed(1)).join(" ");
    out+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.2"/>`;
    const last=P2[P2.length-1];
    out+=`<circle cx="${X(last[0]).toFixed(1)}" cy="${Y(last[1]).toFixed(1)}" r="3.4" fill="${s.color}"/>`;
  });
  out+=`</g>`;
  if(view){ out+=`<text x="${(W-P.r).toFixed(1)}" y="${(P.t+FS).toFixed(1)}" text-anchor="end" font-size="${(FS).toFixed(1)}" fill="#4fd1c5">zoomed — double-click to reset</text>`; }
  svg.innerHTML=out;
  svg.__series=series;
  if(!svg.__interactive) attachChartInteraction(svg);
}

// matplotlib-style interaction: hover tooltip, scroll-zoom, drag box-zoom, double-click reset
function attachChartInteraction(svg){
  svg.__interactive=true; svg.style.cursor="crosshair";
  const id=svg.id;
  const rel=e=>{ const r=svg.getBoundingClientRect(); return [(e.clientX-r.left)*(svg.clientWidth/r.width),
                                                             (e.clientY-r.top)*(svg.clientHeight/r.height)]; };
  const dataX=px=>{ const G=chartGeom[id]; return G? G.xmin+(px-G.P.l)/G.plotW*(G.xmax-G.xmin):0; };
  let drag=null;
  svg.addEventListener("mousedown",e=>{ if(e.button!==0) return; drag={x:rel(e)[0]}; interacting=true; hideTip(); e.preventDefault(); });
  svg.addEventListener("mousemove",e=>{
    const G=chartGeom[id]; if(!G) return; const [px]=rel(e);
    if(drag){
      let sel=svg.querySelector("#__sel");
      if(!sel){ sel=document.createElementNS("http://www.w3.org/2000/svg","rect"); sel.setAttribute("id","__sel");
        sel.setAttribute("fill","rgba(79,209,197,0.15)"); sel.setAttribute("stroke","#4fd1c5"); sel.setAttribute("stroke-dasharray","3 3"); svg.appendChild(sel); }
      sel.setAttribute("x",Math.min(drag.x,px)); sel.setAttribute("y",G.P.t);
      sel.setAttribute("width",Math.abs(px-drag.x)); sel.setAttribute("height",G.plotH);
    } else showTip(svg,e,px);
  });
  window.addEventListener("mouseup",e=>{
    if(!drag) return; const G=chartGeom[id]; const [px]=rel(e);
    const sel=svg.querySelector("#__sel"); if(sel) sel.remove();
    if(G && Math.abs(px-drag.x)>6){ const a=dataX(Math.min(px,drag.x)), b=dataX(Math.max(px,drag.x));
      pushView(id,{x0:a,x1:b,y0:G.ymin,y1:G.ymax}); }
    drag=null; interacting=false;
  });
  svg.addEventListener("wheel",e=>{ const G=chartGeom[id]; if(!G) return; e.preventDefault();
    const cx=dataX(rel(e)[0]); const f=e.deltaY<0?0.8:1.25;
    pushView(id,{x0:cx-(cx-G.xmin)*f, x1:cx+(G.xmax-cx)*f, y0:G.ymin, y1:G.ymax});
  },{passive:false});
  svg.addEventListener("dblclick",()=>homeView(id));
  svg.addEventListener("mouseleave",()=>hideTip());
}

function showTip(svg,e,px){
  const G=chartGeom[svg.id], series=svg.__series; if(!G||!series) return;
  const ep=Math.round(G.xmin+(px-G.P.l)/G.plotW*(G.xmax-G.xmin));
  let rows="";
  series.forEach(s=>{ let best=null;
    s.points.forEach(p=>{ if(p[1]==null||isNaN(p[1])) return; if(best===null||Math.abs(p[0]-ep)<Math.abs(best[0]-ep)) best=p; });
    if(best) rows+=`<div><span class="ttsw" style="background:${s.color}"></span>${escapeHtml(s.label)}: <b>${(+best[1]).toFixed(2)}</b></div>`;
  });
  if(!rows) return;
  const tt=$("#tt"); tt.innerHTML=`<div class="tth">episode ${ep}</div>${rows}`;
  tt.style.display="block"; tt.style.left=(e.clientX+14)+"px"; tt.style.top=(e.clientY+14)+"px";
}
function hideTip(){ const tt=$("#tt"); if(tt) tt.style.display="none"; }
function fmtAxis(v){ const a=Math.abs(v); if(a>=1000) return (v/1000).toFixed(1)+"k"; if(a<1&&a>0) return v.toFixed(2); return v.toFixed(a<10?1:0); }

function outcomeChart(svg, runs){
  const W=svg.clientWidth||560,H=svg.clientHeight||220,FS=rem()*0.74,P={l:rem()*1,r:rem()*1,t:rem()*1.7,b:rem()*3};
  if(!runs.length){ svg.innerHTML=`<text x="${W/2}" y="${H/2}" fill="#9aa7bd" text-anchor="middle" font-size="${rem()*.85}">no data yet</text>`; return; }
  const keys=[1,2,3,4,0];
  const bw=Math.min(rem()*4.6,(W-P.l-P.r)/runs.length-rem());
  const maxN=Math.max(1,...runs.map(r=>r.detail.episodes.length));
  let g="";
  runs.forEach((r,ri)=>{
    const counts={}; keys.forEach(k=>counts[k]=0);
    r.detail.episodes.forEach(e=>{counts[e.outcome]=(counts[e.outcome]||0)+1;});
    const cx=P.l+(ri+0.5)*((W-P.l-P.r)/runs.length);
    let y=H-P.b;
    keys.forEach(k=>{ const h=(counts[k]/maxN)*(H-P.t-P.b);
      if(h>0){ g+=`<rect x="${(cx-bw/2).toFixed(1)}" y="${(y-h).toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" fill="${OUTCOME_COLOR[k]}"/>`; y-=h; } });
    g+=`<text x="${cx.toFixed(1)}" y="${(H-P.b+FS*1.4).toFixed(1)}" fill="#e6ecf5" font-size="${FS.toFixed(1)}" text-anchor="middle">${escapeHtml(r.summary.label).slice(0,16)}</text>`;
    g+=`<text x="${cx.toFixed(1)}" y="${(H-P.b+FS*2.7).toFixed(1)}" fill="#9aa7bd" font-size="${(FS*.9).toFixed(1)}" text-anchor="middle">${r.detail.episodes.length} ep</text>`;
  });
  let lx=P.l; g+=`<g font-size="${(FS*.92).toFixed(1)}" fill="#9aa7bd">`;
  [2,3,1,4].forEach(k=>{ g+=`<rect x="${lx.toFixed(1)}" y="1" width="${(FS*.8).toFixed(1)}" height="${(FS*.8).toFixed(1)}" fill="${OUTCOME_COLOR[k]}"/><text x="${(lx+FS).toFixed(1)}" y="${(FS*.85).toFixed(1)}">${OUTCOME[k]}</text>`; lx+=Math.min(W/4,rem()*8.5); });
  g+=`</g>`;
  svg.innerHTML=g;
}

function renderTable(){
  const rows = [...state.selected].map(id=>state.runs.find(r=>r.run_id===id)).filter(Boolean);
  const el=$("#cmpTable");
  if(!rows.length){ el.innerHTML=`<div class="empty">Select one or more runs to compare.</div>`; return; }
  el.innerHTML = `<table><thead><tr>
    <th>Run</th><th>fixes applied</th><th>commit</th><th>mode</th><th>status</th><th>episodes</th>
    <th>mean</th><th>final</th><th>best</th><th>success</th></tr></thead><tbody>${
    rows.map(r=>`<tr>
      <td><span class="swatch" style="background:${colorFor(r.run_id)}"></span>${escapeHtml(r.label)}</td>
      <td>${(r.fixes && r.fixes.length) ? escapeHtml(r.fixes.join(" + ")) : "<span style='color:var(--muted)'>none (baseline)</span>"}</td>
      <td><code>${shortSha(r.git_sha)}</code></td>
      <td>${r.mode}</td>
      <td>${statusBadge(r)}</td>
      <td>${r.num_episodes}</td>
      <td>${fmt(r.mean_reward)}</td>
      <td>${fmt(r.final_reward)}</td>
      <td>${fmt(r.best_reward)}</td>
      <td>${r.success_rate==null?"—":Math.round(r.success_rate*100)+"%"}</td>
    </tr>`).join("")}</tbody></table>`;
}

// guide panel
$("#guideBtn").addEventListener("click",()=>$("#guide").classList.add("open"));
$("#guideClose").addEventListener("click",()=>$("#guide").classList.remove("open"));
$("#guide").addEventListener("click",e=>{ if(e.target.id==="guide") $("#guide").classList.remove("open"); });

// per-chart toolbar: Back (previous view) · Home (reset zoom) · Maximize
["chartReward","chartEps","chartSteps","chartOutcomes"].forEach(id=>{
  const card=$("#"+id).closest(".card");
  const tb=document.createElement("div"); tb.className="charttools";
  const mk=(txt,title,fn,back)=>{ const b=document.createElement("button"); b.className="maxbtn"; b.textContent=txt; b.title=title;
    if(back) b.setAttribute("data-back",id); b.addEventListener("click",fn); return b; };
  tb.appendChild(mk("←","back (previous view)",()=>backView(id),true));
  tb.appendChild(mk("⟲","reset zoom (home)",()=>homeView(id)));
  tb.appendChild(mk("⤢","maximize",()=>openMaximized(id)));
  card.appendChild(tb);
});
$("#chartClose").addEventListener("click",closeMaximized);
$("#chartModal").addEventListener("click",e=>{ if(e.target.id==="chartModal") closeMaximized(); });
$("#bigBack").addEventListener("click",()=>backView("chartBig"));
$("#bigHome").addEventListener("click",()=>homeView("chartBig"));
syncNav();

document.addEventListener("keydown",e=>{ if(e.key==="Escape"){ $("#guide").classList.remove("open"); closeMaximized(); } });

// controls
$("#smoothChk").addEventListener("change",drawAll);
$("#smoothWin").addEventListener("input",()=>{$("#smoothVal").textContent=$("#smoothWin").value; drawAll();});
$("#selAll").addEventListener("click",()=>{ state.runs.forEach(r=>state.selected.add(r.run_id)); refresh(); });
$("#selNone").addEventListener("click",()=>{ state.selected.clear(); refresh(); });
$("#selLive").addEventListener("click",()=>{ state.selected.clear(); state.runs.filter(r=>r.live).forEach(r=>state.selected.add(r.run_id)); refresh(); });
window.addEventListener("resize", drawAll);

pollRuns();
setInterval(pollRuns, 2000);
