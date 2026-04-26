const state = {
  events: [],
  examples: [],
  selectedExampleId: null,
  selectedLoopId: null,
  loops: [],
  benchmarks: [],
  comparisons: [],
  telemetrySamples: [],
};

const $ = (selector) => document.querySelector(selector);

async function loadInitial() {
  const [events, sessions, telemetry, logs, examples, loops, benchmarks, comparisons] = await Promise.all([
    fetchJson('/api/events?limit=900'),
    fetchJson('/api/sessions'),
    fetchJson('/api/telemetry/gpu'),
    fetchJson('/api/logs?limit=120'),
    fetchJson('/api/examples'),
    fetchJson('/api/loops'),
    fetchJson('/api/benchmarks'),
    fetchJson('/api/comparisons'),
  ]);
  state.events = events.events || [];
  state.examples = examples.examples || [];
  state.loops = loops.loops || [];
  state.benchmarks = benchmarks.benchmarks || [];
  state.comparisons = comparisons.comparisons || [];
  renderSessions(sessions);
  renderExamples();
  renderTelemetry(telemetry.gpus || []);
  renderLogs(logs.lines || []);
  render();
  if (!state.selectedExampleId && state.examples.length) {
    await selectExample(state.examples[0].id);
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${url}: ${response.status} ${detail}`);
  }
  return response.json();
}

function connectStream() {
  const source = new EventSource('/stream');
  source.onopen = () => {
    $('#connection-status').textContent = 'Live';
  };
  source.onerror = () => {
    $('#connection-status').textContent = 'Reconnecting';
  };
  [
    'agent.session_started',
    'agent.episode_started',
    'agent.prompt_received',
    'tool.started',
    'tool.completed',
    'tool.failed',
    'run.requested',
    'run.started',
    'run.completed',
    'run.failed',
    'benchmark.started',
    'benchmark.completed',
    'comparison.created',
    'loop.summary',
    'artifact.created',
    'candidate.proposed',
    'candidate.validated',
    'candidate.judged',
    'hypothesis.recorded',
    'decision_trace.created',
    'log.line',
  ].forEach((kind) => {
    source.addEventListener(kind, (message) => appendEvent(JSON.parse(message.data)));
  });
}

function appendEvent(event) {
  if (state.events.some((item) => item.event_id === event.event_id)) return;
  state.events.push(event);
  state.events = state.events.slice(-1200);
  state.loops = summarizeLoops(state.events);
  if (event.kind === 'benchmark.completed') {
    state.benchmarks = benchmarkRowsFromEvents(state.events);
  }
  if (event.kind === 'comparison.created') {
    state.comparisons = state.events
      .filter((item) => item.kind === 'comparison.created')
      .map((item) => ({ event_id: item.event_id, timestamp: item.timestamp, ...item.payload }));
  }
  render();
}

function render() {
  $('#event-count').textContent = `${state.events.length} events`;
  renderLoops();
  renderTimeline();
  renderBenchmark();
  renderProgress();
  renderExperimentBoard();
  renderArtifacts();
  renderCharts();
}

function renderExamples() {
  const select = $('#example-select');
  select.innerHTML = state.examples
    .map((example) => `<option value="${escapeAttr(example.id)}">${escapeHtml(example.title)}${example.enabled ? '' : ' (disabled)'}</option>`)
    .join('');
  select.addEventListener('change', () => selectExample(select.value));
}

async function selectExample(exampleId) {
  state.selectedExampleId = exampleId;
  const example = state.examples.find((item) => item.id === exampleId);
  if (!example) return;
  $('#example-select').value = exampleId;
  $('#example-description').textContent = `${example.kernel_family}: ${example.description}`;
  setRunControls(example.default_config || {}, example.supported_knobs || []);
  $('#run-button').disabled = !example.enabled;
  $('#run-status').textContent = example.enabled ? 'Ready.' : example.disabled_reason || 'Example disabled.';
  const preview = await fetchJson(`/api/examples/${encodeURIComponent(exampleId)}`);
  $('#source-meta').textContent = preview.language || 'source';
  $('#source-preview').innerHTML = renderCode(preview.source || '', preview.language || 'python');
}

function setRunControls(config, knobs) {
  $('#repetitions-input').value = config.repetitions ?? 20;
  $('#warmup-input').value = config.warmup ?? 5;
  $('#block-sizes-input').value = (config.block_sizes || []).join(',');
  $('#num-warps-input').value = (config.num_warps || []).join(',');
  $('#cache-modifiers-input').value = (config.load_cache_modifiers || ['']).map((item) => item || 'none').join(',');
  $('#cache-modifiers-input').disabled = !knobs.includes('load_cache_modifiers');
}

async function startRun() {
  const exampleId = state.selectedExampleId;
  if (!exampleId) return;
  $('#run-button').disabled = true;
  $('#run-status').textContent = 'Submitting run...';
  try {
    const gpuIndex = $('#gpu-index-input').value.trim();
    const response = await fetchJson('/api/runs', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        example_id: exampleId,
        mode: 'benchmark',
        config: {
          repetitions: Number($('#repetitions-input').value),
          warmup: Number($('#warmup-input').value),
          block_sizes: parseList($('#block-sizes-input').value),
          num_warps: parseList($('#num-warps-input').value),
          load_cache_modifiers: parseStringList($('#cache-modifiers-input').value),
          gpu_index: gpuIndex === '' ? null : Number(gpuIndex),
        },
      }),
    });
    state.selectedLoopId = response.run_id;
    $('#run-status').textContent = `Queued ${response.run_id}`;
  } catch (error) {
    $('#run-status').textContent = error.message;
    $('#run-button').disabled = false;
  }
}

function renderSessions(data) {
  $('#session-summary').innerHTML = `
    <span class="pill">${data.event_count || 0} events</span>
    <span>${(data.sessions || []).length} sessions</span>
    <span>${(data.episodes || []).length} episodes</span>
  `;
}

function renderTelemetry(gpus) {
  const root = $('#gpu-list');
  if (!gpus.length) {
    root.textContent = 'No NVIDIA telemetry available.';
    $('#gpu-sparkline').innerHTML = '';
    return;
  }
  const totalUtil = gpus.reduce((sum, gpu) => sum + (gpu.utilization_gpu_pct || 0), 0) / gpus.length;
  state.telemetrySamples.push(totalUtil);
  state.telemetrySamples = state.telemetrySamples.slice(-36);
  root.innerHTML = gpus
    .map((gpu) => `
      <div class="gpu">
        <strong>GPU ${gpu.index}: ${escapeHtml(gpu.name)}</strong>
        <div>${gpu.utilization_gpu_pct ?? 'n/a'}% util · ${gpu.memory_used_mib ?? 'n/a'} / ${gpu.memory_total_mib ?? 'n/a'} MiB</div>
        <div>${gpu.temperature_c ?? 'n/a'} C · ${gpu.power_w ?? 'n/a'} W</div>
      </div>
    `)
    .join('');
  $('#gpu-sparkline').innerHTML = renderSparkline(state.telemetrySamples);
}

function renderLoops() {
  const loops = state.loops.length ? state.loops : summarizeLoops(state.events);
  state.loops = loops;
  const root = $('#loop-list');
  if (!loops.length) {
    root.textContent = 'No loops yet.';
    return;
  }
  root.innerHTML = loops
    .slice()
    .reverse()
    .slice(0, 14)
    .map((loop) => `
      <button class="loop-item ${state.selectedLoopId === loop.id ? 'selected' : ''}" data-loop="${escapeAttr(loop.id)}">
        <span>${escapeHtml(loop.family || loop.example_id || 'loop')}</span>
        <strong>${escapeHtml(loop.status)}</strong>
        <small>${escapeHtml(loop.best_candidate_id || loop.id)}</small>
      </button>
    `)
    .join('');
}

function renderTimeline() {
  const timeline = $('#timeline');
  const events = filteredEvents().slice().reverse().slice(0, 120);
  $('#selected-loop-label').textContent = state.selectedLoopId || 'all loops';
  if (!events.length) {
    timeline.innerHTML = '<div class="event muted">No events for this loop yet.</div>';
    return;
  }
  timeline.innerHTML = events.map(renderEventCard).join('');
}

function renderEventCard(event) {
  const payload = event.payload || {};
  const title = eventTitle(event);
  const body = eventBody(event);
  return `
    <div class="event event-${event.kind.split('.')[0]}">
      <div class="event-head">
        <span class="pill">${escapeHtml(event.kind)}</span>
        <small>${new Date(event.timestamp).toLocaleTimeString()}</small>
      </div>
      <strong>${escapeHtml(title)}</strong>
      <div class="event-body">${body}</div>
      ${payload.duration_ms ? `<small>${formatNumber(payload.duration_ms, 1)} ms</small>` : ''}
    </div>
  `;
}

function eventTitle(event) {
  const payload = event.payload || {};
  if (event.kind.startsWith('tool.')) return payload.tool || 'tool call';
  if (event.kind.startsWith('run.')) return payload.run_id || 'run';
  if (event.kind.startsWith('benchmark.')) return payload.family || 'benchmark';
  if (event.kind === 'candidate.proposed') return `${payload.count || 0} candidates proposed`;
  if (event.kind === 'hypothesis.recorded') return payload.statement || 'hypothesis';
  if (event.kind === 'comparison.created') return payload.conclusion || 'comparison';
  return event.kind;
}

function eventBody(event) {
  const payload = event.payload || {};
  if (event.kind === 'benchmark.completed' && payload.best) {
    return `
      <div class="mini-metrics">
        <span>best ${escapeHtml(payload.best.candidate_id || 'candidate')}</span>
        <span>${formatNumber(payload.best.median_ms)} ms</span>
        <span>${formatNumber(payload.best.speedup_vs_baseline)}x</span>
      </div>
    `;
  }
  if (event.kind === 'candidate.proposed' && Array.isArray(payload.candidates)) {
    return payload.candidates
      .slice(0, 3)
      .map((candidate) => `<code>${escapeHtml(candidate.id)} ${escapeHtml(JSON.stringify(candidate.changes || {}))}</code>`)
      .join('');
  }
  return `<pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`;
}

function renderBenchmark() {
  const latest = [...state.events].reverse().find((event) => event.kind === 'benchmark.completed');
  const root = $('#benchmark-summary');
  if (!latest) {
    root.textContent = 'No benchmark events yet.';
    return;
  }
  const best = latest.payload?.best || {};
  root.innerHTML = `
    <div><span class="metric">${formatNumber(best.median_ms)}</span><small>median ms</small></div>
    <div><span class="metric">${formatNumber(best.speedup_vs_baseline)}</span><small>speedup</small></div>
    <div><span class="metric">${formatNumber(best.bandwidth_gbps, 1)}</span><small>GB/s</small></div>
    <div><span class="metric">${escapeHtml(String(latest.payload?.candidate_count || 0))}</span><small>candidates</small></div>
  `;
}

function renderProgress() {
  const events = filteredEvents();
  const started = events.find((event) => event.kind === 'run.started' || event.kind === 'benchmark.started');
  const completed = events.find((event) => event.kind === 'run.completed');
  const failed = events.find((event) => event.kind === 'run.failed');
  const benchmark = events.find((event) => event.kind === 'benchmark.completed');
  const root = $('#run-progress');
  if (!started) {
    root.textContent = 'No active run.';
    $('#run-button').disabled = false;
    return;
  }
  const total = benchmark?.payload?.candidate_count || benchmark?.payload?.results?.length || 0;
  const status = failed ? 'failed' : completed ? 'completed' : 'running';
  if (status !== 'running') $('#run-button').disabled = false;
  root.innerHTML = `
    <div class="progress-bar"><span style="width:${status === 'running' ? '55' : '100'}%"></span></div>
    <div>${escapeHtml(status)} · ${total || 'unknown'} candidates · ${escapeHtml(started.payload?.run_id || '')}</div>
  `;
}

function renderExperimentBoard() {
  const interesting = filteredEvents().filter((event) =>
    ['hypothesis.recorded', 'candidate.proposed', 'candidate.validated', 'candidate.judged', 'comparison.created', 'loop.summary'].includes(event.kind)
  ).slice(-16);
  const root = $('#experiment-board');
  if (!interesting.length) {
    root.textContent = 'Hypotheses and candidates will appear here.';
    return;
  }
  root.innerHTML = interesting
    .map((event) => `<div class="evidence-card"><span class="pill">${escapeHtml(event.kind)}</span><pre>${escapeHtml(JSON.stringify(event.payload, null, 2))}</pre></div>`)
    .join('');
}

function renderArtifacts() {
  const artifacts = unique(filteredEvents().flatMap((event) => event.artifact_paths || []).filter(Boolean)).slice(-30);
  const root = $('#artifact-list');
  if (!artifacts.length) {
    root.textContent = 'No artifacts yet.';
    return;
  }
  root.innerHTML = artifacts
    .map((path) => `<button class="artifact" data-artifact="${escapeAttr(path)}">${escapeHtml(shortPath(path))}</button>`)
    .join('');
}

async function previewArtifact(path) {
  $('#artifact-preview').textContent = 'Loading preview...';
  const preview = await fetchJson(`/api/artifacts/preview/${encodeURIComponent(path)}`);
  const header = `<div class="preview-head"><span class="pill">${escapeHtml(preview.render_mode)}</span><small>${escapeHtml(preview.relative_path)}</small></div>`;
  const body = preview.render_mode === 'markdown'
    ? `<div class="markdown-preview">${renderMarkdown(preview.text || '')}</div>`
    : `<pre class="code-block">${renderCode(preview.text || '', preview.language || 'text')}</pre>`;
  $('#artifact-preview').innerHTML = `${header}${body}`;
}

function renderCharts() {
  const rows = rowsForSelectedLoop();
  renderRanking(rows);
  renderIntervals(rows);
  renderThroughput(rows);
  renderComparisons();
}

function renderRanking(rows) {
  const root = $('#ranking-chart');
  const valid = rows.filter((row) => typeof row.median_ms === 'number').sort((a, b) => a.median_ms - b.median_ms).slice(0, 12);
  if (!valid.length) {
    root.textContent = 'No data yet.';
    return;
  }
  const max = Math.max(...valid.map((row) => row.median_ms));
  root.innerHTML = valid.map((row) => chartBar(row.candidate_id, row.median_ms, max, 'ms')).join('');
}

function renderIntervals(rows) {
  const root = $('#interval-chart');
  const valid = rows.filter((row) => typeof row.median_ms === 'number').sort((a, b) => a.median_ms - b.median_ms).slice(0, 10);
  if (!valid.length) {
    root.textContent = 'No data yet.';
    return;
  }
  const max = Math.max(...valid.map((row) => row.p80_ms || row.median_ms));
  root.innerHTML = valid
    .map((row) => {
      const left = ((row.p20_ms || row.median_ms) / max) * 100;
      const width = (((row.p80_ms || row.median_ms) - (row.p20_ms || row.median_ms)) / max) * 100;
      const median = (row.median_ms / max) * 100;
      return `<div class="interval"><small>${escapeHtml(row.candidate_id)}</small><span><i style="left:${left}%;width:${Math.max(width, 1)}%"></i><b style="left:${median}%"></b></span></div>`;
    })
    .join('');
}

function renderThroughput(rows) {
  const root = $('#throughput-chart');
  const valid = rows.filter((row) => typeof row.bandwidth_gbps === 'number').sort((a, b) => b.bandwidth_gbps - a.bandwidth_gbps).slice(0, 12);
  if (!valid.length) {
    root.textContent = 'No throughput data yet.';
    return;
  }
  const max = Math.max(...valid.map((row) => row.bandwidth_gbps));
  root.innerHTML = valid.map((row) => chartBar(row.candidate_id, row.bandwidth_gbps, max, 'GB/s')).join('');
}

function renderComparisons() {
  const root = $('#comparison-board');
  const comparisons = state.comparisons.length
    ? state.comparisons
    : filteredEvents().filter((event) => event.kind === 'comparison.created').map((event) => event.payload);
  const latest = comparisons[comparisons.length - 1];
  if (!latest) {
    root.textContent = 'No comparison yet.';
    return;
  }
  root.innerHTML = `
    <div class="comparison-card">
      <span class="pill">${escapeHtml(latest.conclusion || 'comparison')}</span>
      <strong>${escapeHtml(latest.candidate_id || 'candidate')}</strong>
      <div>${formatNumber(latest.speedup_vs_baseline)}x speedup</div>
      <div>${formatNumber(latest.delta_percent, 2)}% delta</div>
    </div>
  `;
}

function chartBar(label, value, max, unit) {
  const width = max > 0 ? (value / max) * 100 : 0;
  return `
    <div class="bar-row">
      <small>${escapeHtml(label)}</small>
      <span><i style="width:${width}%"></i></span>
      <b>${formatNumber(value, unit === 'GB/s' ? 1 : 4)} ${unit}</b>
    </div>
  `;
}

function rowsForSelectedLoop() {
  const fromEvents = benchmarkRowsFromEvents(filteredEvents());
  if (fromEvents.length) return fromEvents;
  if (!state.selectedLoopId) return state.benchmarks;
  return state.benchmarks.filter((row) => row.run_id === state.selectedLoopId);
}

function benchmarkRowsFromEvents(events) {
  return events.flatMap((event) => {
    if (event.kind !== 'benchmark.completed') return [];
    const runId = event.payload?.run_id || event.event_id;
    const family = event.payload?.family;
    if (Array.isArray(event.payload?.results)) {
      return event.payload.results.map((row) => ({ run_id: runId, family, ...row }));
    }
    return event.payload?.best ? [{ run_id: runId, family, ...event.payload.best }] : [];
  });
}

function filteredEvents() {
  if (!state.selectedLoopId) return state.events;
  return state.events.filter((event) => eventLoopId(event) === state.selectedLoopId || event.episode_id === state.selectedLoopId);
}

function summarizeLoops(events) {
  const grouped = new Map();
  events.forEach((event) => {
    const id = eventLoopId(event);
    if (!id) return;
    if (!grouped.has(id)) grouped.set(id, []);
    grouped.get(id).push(event);
  });
  return [...grouped.entries()].map(([id, eventsForLoop]) => {
    const latest = eventsForLoop[eventsForLoop.length - 1];
    const benchmark = [...eventsForLoop].reverse().find((event) => event.kind === 'benchmark.completed');
    const best = benchmark?.payload?.best || {};
    return {
      id,
      status: loopStatus(eventsForLoop),
      family: firstPayload(eventsForLoop, 'family'),
      example_id: firstPayload(eventsForLoop, 'example_id'),
      best_candidate_id: best.candidate_id,
      best_median_ms: best.median_ms,
      updated_at: latest.timestamp,
    };
  });
}

function loopStatus(events) {
  const kinds = events.map((event) => event.kind);
  if (kinds.includes('run.failed') || kinds.includes('tool.failed')) return 'failed';
  if (kinds.includes('run.completed') || kinds.includes('loop.summary')) return 'completed';
  if (kinds.includes('run.started') || kinds.includes('benchmark.started')) return 'running';
  if (kinds.includes('run.requested')) return 'queued';
  return 'observed';
}

function eventLoopId(event) {
  const payload = event.payload || {};
  return payload.run_id || payload.loop_id || payload.episode_id || event.episode_id || event.session_id || null;
}

function firstPayload(events, key) {
  const event = events.find((item) => item.payload && item.payload[key] !== undefined);
  return event?.payload?.[key] || null;
}

function renderLogs(lines) {
  $('#logs').textContent = lines.length ? lines.join('\n') : 'No logs yet.';
}

function renderSparkline(values) {
  if (!values.length) return '';
  const width = 220;
  const height = 54;
  const points = values
    .map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width;
      const y = height - (Math.max(0, Math.min(value, 100)) / 100) * height;
      return `${x},${y}`;
    })
    .join(' ');
  return `<svg viewBox="0 0 ${width} ${height}" role="img"><polyline points="${points}"></polyline></svg>`;
}

function renderCode(text, language) {
  return text
    .split('\n')
    .map((line, index) => `<span class="code-line"><em>${index + 1}</em><code>${highlight(line, language)}</code></span>`)
    .join('');
}

function renderMarkdown(text) {
  return escapeHtml(text)
    .split('\n')
    .map((line) => {
      if (line.startsWith('# ')) return `<h1>${line.slice(2)}</h1>`;
      if (line.startsWith('## ')) return `<h2>${line.slice(3)}</h2>`;
      if (line.startsWith('### ')) return `<h3>${line.slice(4)}</h3>`;
      if (line.startsWith('- ')) return `<p class="md-bullet">${line.slice(2)}</p>`;
      if (line.startsWith('|')) return `<code>${line}</code>`;
      if (!line.trim()) return '<br />';
      return `<p>${line}</p>`;
    })
    .join('');
}

function highlight(line, language) {
  let html = escapeHtml(line);
  if (language === 'json') {
    return html.replace(/(&quot;[^&]+?&quot;)(\s*:)?/g, '<span class="tok-string">$1</span>$2');
  }
  html = html
    .replace(/(#.*)$/g, '<span class="tok-comment">$1</span>')
    .replace(/(&quot;.*?&quot;|&#39;.*?&#39;)/g, '<span class="tok-string">$1</span>')
    .replace(/\b(@triton\.jit|def|return|if|else|for|in|import|from|class|with|as|raise|try|except)\b/g, '<span class="tok-keyword">$1</span>')
    .replace(/\b(tl\.[A-Za-z_]+|triton\.[A-Za-z_]+)\b/g, '<span class="tok-call">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="tok-number">$1</span>');
  return html;
}

function parseList(value) {
  return value.split(',').map((item) => Number(item.trim())).filter((item) => Number.isFinite(item));
}

function parseStringList(value) {
  return value.split(',').map((item) => item.trim()).filter((item) => item.length);
}

function formatNumber(value, digits = 4) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function shortPath(path) {
  const parts = String(path).split('/');
  return parts.slice(-3).join('/');
}

function unique(values) {
  return [...new Set(values)];
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll('`', '&#96;');
}

$('#refresh-button').addEventListener('click', loadInitial);
$('#run-button').addEventListener('click', startRun);
$('#loop-list').addEventListener('click', (event) => {
  const button = event.target.closest('[data-loop]');
  if (!button) return;
  state.selectedLoopId = button.dataset.loop;
  render();
});
$('#artifact-list').addEventListener('click', (event) => {
  const button = event.target.closest('[data-artifact]');
  if (!button) return;
  previewArtifact(button.dataset.artifact).catch((error) => {
    $('#artifact-preview').textContent = error.message;
  });
});

loadInitial().catch((error) => {
  $('#connection-status').textContent = 'Error';
  $('#timeline').innerHTML = `<div class="event">${escapeHtml(error.message)}</div>`;
});
connectStream();

setInterval(async () => {
  try {
    const telemetry = await fetchJson('/api/telemetry/gpu');
    renderTelemetry(telemetry.gpus || []);
  } catch {
    // Keep the latest known telemetry visible.
  }
}, 5000);
