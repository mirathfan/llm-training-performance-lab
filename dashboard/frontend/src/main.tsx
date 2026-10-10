import React, {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity, ArrowDownUp, Download, FlaskConical, Info, RefreshCw} from 'lucide-react';
import {BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, ErrorBar} from 'recharts';
import './style.css';

type Metric = {mean: number | null; std: number | null; unit: string};
type Stage = {name: string; status: string; batch_size: number | null; sequence_length: number | null; successful_repeats: number; repeats: number; settings: Record<string, unknown>; errors: string[]; metrics: Record<string, Metric>; speedup_vs_baseline: number | null; memory_reduction_mib: number | null};
type Run = {run_id: string; model: string | null; origin: string; timestamp_utc: string | null; environment: Record<string, unknown>; stages: Stage[]; artifacts: {path: string; sha256: string}[]; provenance_notes?: string; validation_errors: string[]; config_hash: string | null; git_revision: string | null; git_dirty: boolean | null; config: Record<string, unknown> | null};
type Row = Stage & {run: Run};
const fmt = (v: number | null | undefined, digits = 2) => v == null ? '—' : v.toLocaleString(undefined, {maximumFractionDigits: digits, minimumFractionDigits: digits});
const label = (r: Run) => `${r.model || 'Unknown model'} · ${r.environment.label || 'Unknown environment'} · ${r.run_id}`;
const text = (v: unknown) => v == null ? 'Unknown' : typeof v === 'object' ? JSON.stringify(v) : String(v);
const metrics = [{key:'tokens_per_sec', title:'Training throughput', unit:'tokens/s', color:'#39726b'}, {key:'avg_step_time_ms', title:'Step latency', unit:'ms/step', color:'#657b99'}, {key:'max_gpu_memory_allocated_mb', title:'Peak allocated GPU memory', unit:'MiB', color:'#85806b'}, {key:'speedup', title:'Speedup vs compatible baseline', unit:'×', color:'#79658d'}];

function App() {
  const [runs, setRuns] = useState<Run[]>([]), [selected, setSelected] = useState(''), [loading, setLoading] = useState(true), [error, setError] = useState('');
  const [model, setModel] = useState('all'), [env, setEnv] = useState('all'), [precision, setPrecision] = useState('all'), [status, setStatus] = useState('all');
  const [sort, setSort] = useState('tokens_per_sec'), [asc, setAsc] = useState(false), [detail, setDetail] = useState<Row | null>(null);
  const [compareId, setCompareId] = useState(''), [comparison, setComparison] = useState<{compatible: boolean; reason: string; runs: Run[]} | null>(null), [compareError, setCompareError] = useState('');
  function load() {
    setLoading(true); setError('');
    fetch('/api/runs').then(r => {if (!r.ok) throw new Error(`Results request failed (${r.status})`); return r.json()}).then((data: Run[]) => {
      setRuns(data); setSelected(current => data.some(r => r.run_id === current) ? current : (data.find(r => r.model === 'gpt_small' && r.environment.label === 'WSL2') || data[0])?.run_id || '');
    }).catch(e => setError(e.message)).finally(() => setLoading(false));
  }
  useEffect(load, []);
  const run = runs.find(r => r.run_id === selected);
  useEffect(() => {
    setComparison(null); setCompareError('');
    if (!selected || !compareId) return;
    const controller = new AbortController();
    fetch(`/api/compare?left=${encodeURIComponent(selected)}&right=${encodeURIComponent(compareId)}`, {signal: controller.signal}).then(r => {if(!r.ok) throw new Error(`Comparison failed (${r.status})`); return r.json()}).then(setComparison).catch(e => {if(e.name !== 'AbortError') setCompareError(e.message)});
    return () => controller.abort();
  }, [selected, compareId]);
  const rows = useMemo(() => {
    const displayed = run ? [run, ...(comparison?.compatible ? comparison.runs.filter(r => r.run_id !== selected) : [])] : [];
    return displayed.flatMap(r => r.stages.map(s => ({...s, run:r}))).filter(s =>
      (model === 'all' || (s.run.model || 'Unknown') === model) && (env === 'all' || (s.run.environment.label || 'Unknown') === env) &&
      (precision === 'all' || text(s.settings.precision) === precision) && (status === 'all' || s.status === status));
  }, [run, comparison, selected, model, env, precision, status]);
  const value = (r: Row, key: string): number | string | null => key === 'name' || key === 'status' ? r[key] : key === 'speedup' ? r.speedup_vs_baseline : r.metrics[key]?.mean ?? null;
  const sorted = [...rows].sort((a,b) => {const x=value(a,sort), y=value(b,sort); if(x==null)return y==null?0:1;if(y==null)return -1;return (typeof x==='string' ? x.localeCompare(String(y)) : Number(x)-Number(y))*(asc?1:-1)});
  const ok = rows.filter(s => s.status === 'ok' && !s.run.validation_errors.some(e => e.startsWith(`${s.name}:`)));
  const fastest = [...ok].sort((a,b) => (b.metrics.tokens_per_sec.mean || 0) - (a.metrics.tokens_per_sec.mean || 0))[0];
  const unique = (values: string[]) => ['all', ...Array.from(new Set(values)).sort()];
  function filter(title: string, current: string, options: string[], change: (v:string)=>void) {
    return <label className="filter">{title}<select aria-label={title} value={current} onChange={e=>change(e.target.value)}>{options.map(v=><option key={v} value={v}>{v==='all'?`All ${title==='Status'?'statuses':title.toLowerCase()+'s'}`:v}</option>)}</select></label>;
  }
  function changeSort(key:string) {if(sort===key)setAsc(!asc);else {setSort(key);setAsc(key==='name'||key==='status')}}
  return <><header><div className="brand"><FlaskConical size={22}/><div><strong>LLM Training Performance Lab</strong><span>EXPERIMENT REGISTRY</span></div></div><div className="readOnly"><span className="dot"/> Saved results · Read only</div></header>
  <main><div className="heading"><div><p className="eyebrow">BENCHMARK EXPLORER</p><h1>Training performance</h1><p className="muted">Inspect measured stages, repeat variability, and memory tradeoffs.</p></div><button onClick={load} className="secondary"><RefreshCw size={15}/>Refresh results</button></div>
  {loading && <p role="status">Loading registered results…</p>}{error && <p role="alert" className="notice bad">{error}</p>}
  {!loading && !error && !runs.length && <section className="empty"><FlaskConical/><h2>No registered experiments</h2><p>Import saved artifacts with <code>python experiment.py import</code>. No training is launched here.</p></section>}
  {run && <><section className="selectors panel"><label className="runSelector">Experiment run<select aria-label="Experiment run" value={selected} onChange={e=>{setSelected(e.target.value);setDetail(null);setCompareId('')}}>{runs.map(r=><option key={r.run_id} value={r.run_id}>{label(r)}</option>)}</select></label><label className="runSelector">Compare saved run<select aria-label="Compare saved run" value={compareId} onChange={e=>setCompareId(e.target.value)}><option value="">Select a compatible run</option>{runs.filter(r=>r.run_id!==selected).map(r=><option key={r.run_id} value={r.run_id}>{label(r)}</option>)}</select></label></section>
  {comparison && <p role="status" className={`notice ${comparison.compatible?'':'warning'}`}><Info size={16}/>{comparison.compatible?'Comparison enabled: ':'Comparison unavailable: '}{comparison.reason}. {comparison.compatible?'Both runs appear in the table and charts.':'Runs remain separate.'}</p>}{compareError && <p role="alert">{compareError}</p>}
  <section className="hardware panel"><div><h2><Activity size={16}/>Run disclosure</h2><p>{text(run.environment.gpu)}</p><span className="tag">{text(run.environment.label)}</span><span className="tag">{run.model || 'Unknown model'}</span><span className="tag">{run.origin}</span></div><dl>{['python','pytorch','cuda','os'].map(k=><div key={k}><dt>{k==='pytorch'?'PyTorch':k==='cuda'?'CUDA':k==='os'?'OS':'Python'}</dt><dd>{text(run.environment[k])}</dd></div>)}<div><dt>Recorded UTC</dt><dd>{run.timestamp_utc || 'Unknown · historical artifact'}</dd></div><div><dt>Git revision / dirty</dt><dd>{text(run.git_revision)} / {text(run.git_dirty)}</dd></div></dl><details><summary>Full provenance and configuration</summary><pre>{JSON.stringify({environment:run.environment, config:run.config, config_hash:run.config_hash, notes:run.provenance_notes},null,2)}</pre></details></section>
  {run.validation_errors.length>0 && <details className="notice warning"><summary>{run.validation_errors.length} validation issue(s) · affected stages excluded from chart rankings</summary><ul>{run.validation_errors.map((e,i)=><li key={i}>{e}</li>)}</ul></details>}
  <div className="filters">{filter('Model',model,unique(runs.map(r=>r.model||'Unknown')),setModel)}{filter('Environment',env,unique(runs.map(r=>text(r.environment.label))),setEnv)}{filter('Precision',precision,unique(runs.flatMap(r=>r.stages.map(s=>text(s.settings.precision)))),setPrecision)}{filter('Status',status,['all','ok','failed','skipped'],setStatus)}<button className="clear" onClick={()=>{setModel('all');setEnv('all');setPrecision('all');setStatus('all')}}>Reset filters</button></div>
  <div className="sectionHeading"><h2>Measured stages <span className="count">{rows.length}</span></h2><span className="muted">Mean ± sample standard deviation · click a stage to inspect</span></div>
  {!rows.length?<section className="empty"><h3>No stages match these filters</h3><p>Reset filters or choose another saved run.</p></section>:<div className="tableWrap panel"><table><thead><tr>{[{key:'name',title:'Stage / workload'},{key:'status',title:'Status'},{key:'tokens_per_sec',title:'Tokens/s'},{key:'avg_step_time_ms',title:'Step ms'},{key:'max_gpu_memory_allocated_mb',title:'Allocated MiB'},{key:'speedup',title:'Speedup ×'}].map(c=><th key={c.key} aria-sort={sort===c.key?(asc?'ascending':'descending'):'none'}><button onClick={()=>changeSort(c.key)}>{c.title}<ArrowDownUp size={12}/></button></th>)}<th>Repeats</th></tr></thead><tbody>{sorted.map((s,i)=><tr key={`${s.run.run_id}-${s.name}-${s.batch_size}-${s.sequence_length}-${i}`}><td><button className="stageName" onClick={()=>setDetail(s)}>{s.name}</button><small>batch {s.batch_size} · seq {s.sequence_length} · {text(s.settings.precision)} {comparison?.compatible&&`· ${s.run.run_id}`}</small></td><td><span className={`status ${s.status}`}>{s.status}</span></td>{['tokens_per_sec','avg_step_time_ms','max_gpu_memory_allocated_mb'].map(k=><td key={k}>{fmt(s.metrics[k].mean)}<small>± {fmt(s.metrics[k].std)}</small></td>)}<td>{fmt(s.speedup_vs_baseline)}{s.speedup_vs_baseline!=null?'×':''}</td><td>{s.successful_repeats} / {s.repeats}</td></tr>)}</tbody></table></div>}
  {detail && <section className="panel detail" aria-label="Stage details"><div className="sectionHeading"><h2>{detail.name}</h2><button onClick={()=>setDetail(null)}>Close details</button></div><p>Successful repeats: {detail.successful_repeats} / {detail.repeats}. Memory reduction from compatible baseline: {fmt(detail.memory_reduction_mib)} MiB.</p><dl className="settings">{Object.entries(detail.settings).map(([k,v])=><div key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{text(v)}</dd></div>)}</dl>{detail.errors.length>0?<><h3>Exact saved failure</h3>{detail.errors.map((e,i)=><pre className="failure" key={i}>{e}</pre>)}</>:<p className="muted">No saved errors for this stage.</p>}</section>}
  <div className="sectionHeading chartHeading"><h2>Performance charts</h2><span className="muted">Successful validated stages only · error bars ± 1 SD</span></div>
  {fastest && <p className="muted">Highest visible throughput: <strong>{fastest.name}</strong>, {fmt(fastest.metrics.tokens_per_sec.mean)} tokens/s. Workloads with different batch or sequence sizes are separate measurements.</p>}
  <div className="chartGrid">{metrics.map(m=>{
    const data=ok.filter(s=>m.key!=='speedup'||s.speedup_vs_baseline!=null).map((s,i)=>({name:`${i+1}. ${s.name} [${s.batch_size}/${s.sequence_length}]${comparison?.compatible?' '+s.run.run_id:''}`, value:m.key==='speedup'?s.speedup_vs_baseline:s.metrics[m.key].mean, deviation:m.key==='speedup'?null:s.metrics[m.key].std}));
    return <section className="panel chart" key={m.key}><h3>{m.title}</h3><p className="muted">{m.unit}{m.key==='speedup'?' · mean latency ratio; uncertainty unavailable':''}</p>{data.length?<div aria-label={`${m.title} chart`} role="img"><ResponsiveContainer width="100%" height={280}><BarChart data={data} margin={{top:15,right:15,bottom:65,left:8}}><CartesianGrid strokeDasharray="3 3" vertical={false}/><XAxis dataKey="name" tickFormatter={(_,i)=>String(i+1)} label={{value:'Stage number (hover for name and workload)',position:'bottom',offset:20}}/><YAxis width={70} tickFormatter={v=>Number(v).toLocaleString()} domain={[0,'auto']}/><Tooltip formatter={(v)=>[fmt(Number(v)),m.unit]} contentStyle={{fontSize:12, maxWidth:340}}/><Bar dataKey="value" fill={m.color} radius={[3,3,0,0]} isAnimationActive={false}>{m.key!=='speedup'&&<ErrorBar dataKey="deviation" width={5} strokeWidth={1.5} stroke="#263e4a" isAnimationActive={false}/>}</Bar></BarChart></ResponsiveContainer></div>:<div className="chartEmpty">No successful compatible measurements available.</div>}</section>;
  })}</div>
  <section className="panel downloads"><h2><Download size={17}/>Saved artifacts</h2><p className="muted">Original measurements, plots, and offline reports. Downloads are restricted to registered files.</p><div>{run.artifacts.map((a,i)=><a key={a.path} href={`/api/runs/${encodeURIComponent(run.run_id)}/files/${i}`} download><Download size={14}/>{a.path.split('/').pop()}<span>{a.path.split('.').pop()?.toUpperCase()}</span></a>)}</div>{!run.artifacts.length&&<p>No registered files available for this run.</p>}</section>
  <p className="footnote"><Info size={14}/>Small GPT-style workload · MLPerf-inspired methodology · results do not imply general or production-scale speedups. Unknown metadata is shown explicitly.</p></>}
  </main><footer>LLM Training Performance Lab <span>Saved experiments · no benchmark execution</span></footer></>;
}

createRoot(document.getElementById('root')!).render(<App/>);
