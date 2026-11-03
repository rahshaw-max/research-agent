/* ── State ── */
const state = {
  phase: 'INPUT',   // INPUT | CLARIFYING | RUNNING | REVIEWING | DONE
  sessionId: null,
  questions: [],
  ws: null,
  cycle: 1,
  depth: 'standard',
};

/* ── Session persistence ── */
const SESSION_KEY = 'researchAgentSessionId';

function saveSessionId(id) {
  try { localStorage.setItem(SESSION_KEY, id); } catch (_) {}
}

function loadSessionId() {
  // URL param takes priority, then localStorage
  const params = new URLSearchParams(location.search);
  return params.get('session') || localStorage.getItem(SESSION_KEY) || null;
}

function clearSessionId() {
  try { localStorage.removeItem(SESSION_KEY); } catch (_) {}
  // Remove ?session= from URL without reload
  const url = new URL(location.href);
  url.searchParams.delete('session');
  history.replaceState({}, '', url.toString());
}

async function tryReconnect() {
  const sessionId = loadSessionId();
  if (!sessionId) return;

  try {
    const res = await fetch(`/api/session/${sessionId}`);
    if (!res.ok) { clearSessionId(); return; }
    const data = await res.json();

    if (data.status === 'done' && data.pptx_path) {
      // Session completed — show download
      state.sessionId = sessionId;
      document.getElementById('download-link').href = `/api/download/${sessionId}`;
      setPhase('DONE');
      return;
    }

    if (data.status === 'awaiting_review') {
      // Reconnect to the review gate
      state.sessionId = sessionId;
      state.cycle = data.cycle_count || 1;
      document.getElementById('running-idea-display').textContent = data.idea || '';
      updateCycleIndicator(state.cycle);
      setPhase('RUNNING');
      connectWebSocket();  // server will send awaiting_review event on WS connect
      return;
    }

    if (data.status === 'interrupted') {
      showInterruptedBanner(data.idea);
      return;
    }
  } catch (_) {
    clearSessionId();
  }
}

function showInterruptedBanner(idea) {
  const banner = document.getElementById('interrupted-banner');
  const msg = document.getElementById('interrupted-msg');
  if (idea) {
    msg.textContent = `Your session "${idea.substring(0, 60)}…" was interrupted when the server restarted. You can start a new research below.`;
  }
  banner.classList.remove('hidden');
}

function dismissInterrupted() {
  clearSessionId();
  document.getElementById('interrupted-banner').classList.add('hidden');
}

// Check on load
document.addEventListener('DOMContentLoaded', () => { tryReconnect(); });

const CYCLE_AGENTS = ['researcher', 'interrogator'];

/* ── Phase management ── */
function setPhase(phase) {
  state.phase = phase;
  document.querySelectorAll('.phase').forEach(el => {
    el.classList.remove('active');
    el.classList.add('hidden');
  });
  const target = document.getElementById(`phase-${phase.toLowerCase()}`);
  if (target) {
    target.classList.remove('hidden');
    target.classList.add('active');
  }
}

/* ── INPUT phase ── */
async function submitIdea() {
  const ideaEl = document.getElementById('idea-input');
  const idea = ideaEl.value.trim();
  if (!idea) {
    showError('input-error', 'Please enter an idea to research.');
    return;
  }
  hideError('input-error');

  const btn = document.querySelector('#phase-input .btn-primary');
  btn.disabled = true;
  btn.textContent = 'Loading…';

  try {
    state.depth = document.getElementById('depth-select')?.value || 'standard';
    const res = await fetch('/api/clarify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ idea }),
    });
    const data = await res.json();
    state.sessionId = data.session_id;
    saveSessionId(data.session_id);
    state.questions = data.questions;
    renderQuestions(data.questions);
    setPhase('CLARIFYING');
  } catch (err) {
    showError('input-error', 'Failed to connect. Is the server running?');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Research This →';
  }
}

/* ── CLARIFYING phase ── */
function renderQuestions(questions) {
  const container = document.getElementById('questions-container');
  container.innerHTML = '';
  questions.forEach((q, i) => {
    const div = document.createElement('div');
    div.className = 'question-item';
    div.innerHTML = `
      <label class="question-label" for="q-${i}">
        <span class="question-number">Q${i + 1}.</span> ${escapeHtml(q)}
      </label>
      <textarea id="q-${i}" rows="2" placeholder="Your answer…"></textarea>
    `;
    container.appendChild(div);
  });
}

async function submitAnswers() {
  const answers = {};
  let allAnswered = true;
  state.questions.forEach((q, i) => {
    const val = document.getElementById(`q-${i}`)?.value?.trim() || '';
    answers[q] = val;
    if (!val) allAnswered = false;
  });

  if (!allAnswered) {
    showError('clarify-error', 'Please answer all questions before continuing.');
    return;
  }
  hideError('clarify-error');

  const btn = document.querySelector('#phase-clarifying .btn-primary');
  btn.disabled = true;
  btn.textContent = 'Starting…';

  try {
    await fetch('/api/submit-answers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, answers, depth: state.depth }),
    });

    // Update running display
    const ideaInput = document.getElementById('idea-input').value.trim();
    document.getElementById('running-idea-display').textContent = ideaInput;
    updateCycleIndicator(1);

    setPhase('RUNNING');
    connectWebSocket();
  } catch (err) {
    showError('clarify-error', 'Failed to start research. Please try again.');
    btn.disabled = false;
    btn.textContent = 'Start Research →';
  }
}

/* ── WebSocket ── */
function connectWebSocket() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${location.host}/ws/${state.sessionId}`);
  state.ws = ws;

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.event) {
      case 'agent_start':    handleAgentStart(msg.agent);                          break;
      case 'agent_chunk':    handleAgentChunk(msg.agent, msg.data.text);           break;
      case 'agent_complete': handleAgentComplete(msg.agent, msg.data);             break;
      case 'agent_status':   handleAgentStatus(msg.agent, msg.data.message);      break;
      case 'awaiting_review':handleAwaitingReview(msg.data); break;
      case 'cycle_start':    handleCycleStart(msg.data.cycle);                     break;
      case 'pipeline_done':  handlePipelineDone(msg.data);                          break;
      case 'stage_timing':   handleStageTiming(msg.data);                          break;
      case 'stage_failed':   handleStageFailed(msg.data);                          break;
      case 'error':          handleGlobalError(msg.data.message);                  break;
    }
  };

  ws.onerror = () => handleGlobalError('WebSocket connection failed.');
}

/* ── Agent card handlers ── */
function handleAgentStatus(agent, message) {
  const el = document.getElementById(`status-${agent}`);
  if (el) {
    el.textContent = message;
    el.classList.remove('hidden');
  }
}

function handleAgentStart(agent) {
  const card = document.getElementById(`card-${agent}`);
  const badge = document.getElementById(`badge-${agent}`);
  const output = document.getElementById(`output-${agent}`);
  if (!card) return;

  card.classList.remove('done');
  card.classList.add('running');
  badge.className = 'status-badge status-running';
  badge.textContent = 'Running…';
  output.textContent = '';
  output.classList.add('visible');
  const statusEl = document.getElementById(`status-${agent}`);
  if (statusEl) { statusEl.textContent = ''; statusEl.classList.add('hidden'); }
}

function handleAgentChunk(agent, text) {
  const output = document.getElementById(`output-${agent}`);
  if (!output) return;
  output.textContent += text;
  output.scrollTop = output.scrollHeight;
}

function handleAgentComplete(agent, data) {
  const card = document.getElementById(`card-${agent}`);
  const badge = document.getElementById(`badge-${agent}`);
  if (!card) return;
  card.classList.remove('running');
  card.classList.add('done');
  badge.className = 'status-badge status-done';
  badge.textContent = 'Done';

  if (agent === 'auditor') {
    const outputEl = document.getElementById('output-auditor');
    if (outputEl) outputEl.textContent = 'Building PowerPoint deck…';
  }

  if (agent === 'interrogator' && data && data.questions) {
    const outputEl = document.getElementById('output-interrogator');
    if (outputEl) {
      outputEl.classList.add('visible');
      outputEl.textContent = data.questions.map((q, i) => `${i + 1}. ${q}`).join('\n\n');
    }
  }
}

/* ── HITL: awaiting review ── */
function handleAwaitingReview(data) {
  const { cycle, questions, gap_report, coverage_score, auto_approve_recommended,
          recommendation_reason, per_sub_question_scores, verification_counts } = data;
  state.cycle = cycle;
  document.getElementById('review-cycle-badge').textContent = `Cycle ${cycle} of 3`;
  const critiquePanel = document.getElementById('critique-panel');
  if (critiquePanel) critiquePanel.classList.add('hidden');

  // Render research quality audit panel
  renderCoveragePanel(coverage_score, gap_report);

  // Render per-sub-question scores table
  renderSubQuestionScores(per_sub_question_scores || []);

  // Render verification counts (only if claims were verified)
  renderVerificationCounts(verification_counts);

  // Auto-approve banner with reason
  const banner = document.getElementById('auto-approve-banner');
  if (auto_approve_recommended) {
    banner.textContent = recommendation_reason
      ? `Coverage assessment: ${recommendation_reason} — further cycles unlikely to help. Consider approving.`
      : 'Research coverage did not increase meaningfully (<0.03) — further cycles are unlikely to help. Consider approving.';
    banner.classList.remove('hidden');
  } else {
    if (recommendation_reason) {
      banner.textContent = recommendation_reason;
      banner.classList.remove('hidden');
    } else {
      banner.classList.add('hidden');
    }
  }

  // Render interrogator questions
  renderInterrogatorQuestions(questions || []);

  // Show/hide cycle limit note
  const noteEl = document.getElementById('cycle-limit-note');
  if (cycle >= 3) {
    noteEl.textContent = 'Maximum refinement cycles reached. You can only approve.';
    document.querySelector('.btn-refine').disabled = true;
  } else {
    noteEl.textContent = '';
    document.querySelector('.btn-refine').disabled = false;
  }

  setPhase('REVIEWING');
}

function renderVerificationCounts(vc) {
  const el = document.getElementById('verification-counts-panel');
  if (!el) return;
  if (!vc || vc.total === 0) {
    el.classList.add('hidden');
    return;
  }
  el.classList.remove('hidden');
  const repairNote = (vc.repaired || vc.removed)
    ? ` <span class="vc-repair">${vc.repaired} repaired · ${vc.removed} removed</span>`
    : '';
  el.innerHTML = `
    <span class="vc-label">Claim Verification</span>
    <span class="vc-stat vc-ok">${vc.supported} ✓</span>
    <span class="vc-stat vc-partial">${vc.partial} ~</span>
    <span class="vc-stat vc-bad">${vc.unsupported} ✗</span>
    ${repairNote}
  `;
}

function renderCoveragePanel(score, gap_report) {
  const el = document.getElementById('coverage-score-panel');
  if (!el) return;
  el.classList.remove('hidden');

  const pct = score != null ? Math.round(score * 100) : '?';
  const gaps = (gap_report && gap_report.gaps) ? gap_report.gaps : [];
  const highSev = gaps.filter(g => g.severity >= 4).length;

  const gapRows = gaps.map(g => `
    <div class="gap-item gap-sev-${g.severity}">
      <span class="gap-type">${escapeHtml(g.type.replace(/_/g, ' '))}</span>
      <span class="gap-desc">${escapeHtml(g.description)}</span>
      <span class="gap-sev">sev ${g.severity}</span>
    </div>`).join('');

  el.innerHTML = `
    <div class="coverage-score-row">
      <span class="coverage-label">Research Coverage Score</span>
      <span class="coverage-value">${pct}%</span>
    </div>
    ${gaps.length > 0 ? `<div class="gap-summary">${gaps.length} gap${gaps.length !== 1 ? 's' : ''} identified${highSev > 0 ? ` · ${highSev} high-severity` : ''}</div>` : '<div class="gap-summary">No gaps identified.</div>'}
    ${gapRows}
  `;
}

function renderSubQuestionScores(scores) {
  let el = document.getElementById('subq-scores-panel');
  if (!el) return;
  if (!scores || scores.length === 0) {
    el.classList.add('hidden');
    return;
  }
  el.classList.remove('hidden');
  const rows = scores.map(s => {
    const raw = typeof s.score === 'number' ? s.score : 0;
    const pct = Math.round(raw * 100);
    const domains = s.n_independent_domains ?? '?';
    const primary = s.has_primary_source ? '✓' : '✗';
    const barClass = pct >= 70 ? 'subq-bar-good' : pct >= 40 ? 'subq-bar-mid' : 'subq-bar-low';
    return `<tr class="subq-row">
      <td class="subq-question" title="${escapeHtml(s.rationale || '')}">${escapeHtml((s.sub_question || '').substring(0, 60))}${(s.sub_question || '').length > 60 ? '…' : ''}</td>
      <td class="subq-score">
        <div class="subq-bar-wrap"><div class="subq-bar ${barClass}" style="width:${pct}%"></div></div>
        <span class="subq-pct">${pct}%</span>
      </td>
      <td class="subq-domains">${escapeHtml(String(domains))}</td>
      <td class="subq-primary ${s.has_primary_source ? 'primary-yes' : 'primary-no'}">${primary}</td>
    </tr>`;
  }).join('');
  el.innerHTML = `
    <div class="subq-title">Per-Sub-Question Coverage</div>
    <table class="subq-table">
      <thead><tr><th>Sub-question</th><th>Score</th><th>Domains</th><th>Primary</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="subq-hint">Hover sub-question for model rationale. Scores are capped deterministically: &lt;2 domains→max 50%, no primary→max 70%.</p>
  `;
}

function renderInterrogatorQuestions(questions) {
  const container = document.getElementById('interrogator-questions-container');
  container.innerHTML = '';
  if (!questions.length) {
    container.innerHTML = '<p class="hint muted">No questions generated.</p>';
    return;
  }
  questions.forEach((q, i) => {
    const div = document.createElement('div');
    div.className = 'iq-item';
    div.innerHTML = `
      <label class="iq-label" for="iq-${i}">
        <span class="iq-num">${i + 1}.</span> ${escapeHtml(q)}
      </label>
      <textarea id="iq-${i}" class="iq-input" rows="2" data-question="${escapeHtml(q)}" placeholder="Your answer (optional)…"></textarea>
    `;
    container.appendChild(div);
  });
}

function toggleRefineForm() {
  const form = document.getElementById('refine-form');
  form.classList.toggle('hidden');
}

async function approveReview() {
  await fetch('/api/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: state.sessionId, action: 'approve' }),
  });
  setPhase('RUNNING');
  updateCycleIndicator(state.cycle);
}

async function submitRefinement() {
  const refinedPrompt = document.getElementById('refined-prompt').value.trim();
  const additionalCtx = document.getElementById('additional-context').value.trim();

  // Collect interrogator question answers
  const interrogatorAnswers = {};
  document.querySelectorAll('.iq-input').forEach(el => {
    const question = el.getAttribute('data-question');
    const answer = el.value.trim();
    if (question && answer) {
      interrogatorAnswers[question] = answer;
    }
  });

  await fetch('/api/review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: state.sessionId,
      action: 'refine',
      refined_prompt: refinedPrompt,
      additional_context: additionalCtx,
      interrogator_answers: interrogatorAnswers,
    }),
  });

  // Reset refine form
  document.getElementById('refined-prompt').value = '';
  document.getElementById('additional-context').value = '';
  document.getElementById('refine-form').classList.add('hidden');

  setPhase('RUNNING');
}

/* ── Cycle start (new loop) ── */
function handleCycleStart(cycle) {
  state.cycle = cycle;
  updateCycleIndicator(cycle);

  // Reset the cycle-scoped agent cards
  CYCLE_AGENTS.forEach(agent => {
    const card = document.getElementById(`card-${agent}`);
    const badge = document.getElementById(`badge-${agent}`);
    const output = document.getElementById(`output-${agent}`);
    if (!card) return;
    card.classList.remove('running', 'done');
    badge.className = 'status-badge status-waiting';
    badge.textContent = 'Waiting';
    output.textContent = '';
    output.classList.remove('visible');

    // Add cycle label to card title
    const titleEl = card.querySelector('.card-title');
    let cycleSpan = titleEl.querySelector('.cycle-label');
    if (!cycleSpan) {
      cycleSpan = document.createElement('span');
      cycleSpan.className = 'cycle-label';
      titleEl.appendChild(cycleSpan);
    }
    cycleSpan.textContent = `Cycle ${cycle}`;
  });
}

function updateCycleIndicator(cycle) {
  document.getElementById('cycle-num').textContent = cycle;
}

/* ── Pipeline done ── */
function handlePipelineDone(data) {
  const downloadUrl = data.download_url;
  document.getElementById('download-link').href = downloadUrl;

  // Add download area to right panel
  const existingArea = document.getElementById('right-download-area');
  if (!existingArea) {
    const area = document.createElement('div');
    area.className = 'right-download-area';
    area.id = 'right-download-area';
    area.innerHTML = `
      <p>Your PowerPoint deck is ready.</p>
      <a href="${escapeHtml(downloadUrl)}" class="btn-download" download="research_summary.pptx">
        &#8595; Download Deck
      </a>
    `;
    const cards = document.getElementById('agent-cards');
    cards.parentNode.insertBefore(area, cards);
  }

  // Render timing summary if provided
  if (data.timing && data.timing.length > 0) {
    renderTimingSummary(data.timing, data.timing_totals);
  }

  setPhase('DONE');
}

/* ── Stage timing ── */
const _stageTiming = [];
function handleStageTiming(data) {
  _stageTiming.push(data);
  // Update timing panel if visible
  renderTimingSummary(_stageTiming, null);
}

function renderTimingSummary(rows, totals) {
  let el = document.getElementById('timing-summary');
  if (!el) {
    el = document.createElement('div');
    el.id = 'timing-summary';
    el.className = 'timing-summary';
    const cards = document.getElementById('agent-cards');
    cards.parentNode.appendChild(el);
  }
  const tot = totals || {
    input_tokens: rows.reduce((s, r) => s + (r.input_tokens || 0), 0),
    output_tokens: rows.reduce((s, r) => s + (r.output_tokens || 0), 0),
    tool_calls: rows.reduce((s, r) => s + (r.tool_calls || 0), 0),
  };
  const totalSec = rows.reduce((s, r) => s + (r.seconds || 0), 0);
  const rowsHtml = rows.map(r => {
    const cyc = r.cycle > 0 ? `c${r.cycle}` : '';
    return `<tr>
      <td>${escapeHtml(r.stage)}</td>
      <td>${escapeHtml(cyc)}</td>
      <td>${(r.seconds || 0).toFixed(1)}s</td>
      <td>${(r.input_tokens || 0).toLocaleString()}</td>
      <td>${(r.output_tokens || 0).toLocaleString()}</td>
      <td>${r.tool_calls || 0}</td>
    </tr>`;
  }).join('');
  el.innerHTML = `
    <h4>Stage Timing</h4>
    <table class="timing-table">
      <thead><tr><th>Stage</th><th>Cyc</th><th>Time</th><th>In tok</th><th>Out tok</th><th>Tools</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
      <tfoot><tr class="timing-total">
        <td colspan="2">TOTAL</td>
        <td>${totalSec.toFixed(1)}s</td>
        <td>${tot.input_tokens.toLocaleString()}</td>
        <td>${tot.output_tokens.toLocaleString()}</td>
        <td>${tot.tool_calls}</td>
      </tr></tfoot>
    </table>
  `;
}

/* ── Error handling ── */
function handleGlobalError(message) {
  document.getElementById('global-error-msg').textContent = message;
  document.getElementById('global-error').classList.remove('hidden');
  // Hide all phase panels
  document.querySelectorAll('.phase').forEach(el => {
    el.classList.remove('active');
    el.classList.add('hidden');
  });
}

/* ── Stage-failed gate ── */
function handleStageFailed(data) {
  const gate = document.getElementById('stage-failed-gate');
  if (!gate) return;
  document.getElementById('stage-failed-stage').textContent = data.stage || 'unknown';
  document.getElementById('stage-failed-reason').textContent = data.reason || '';
  gate.classList.remove('hidden');
}

async function retryStage() {
  const gate = document.getElementById('stage-failed-gate');
  if (gate) gate.classList.add('hidden');
  try {
    await fetch('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, action: 'retry_stage' }),
    });
  } catch (err) {
    handleGlobalError('Failed to send retry command.');
  }
}

async function abortPipeline() {
  const gate = document.getElementById('stage-failed-gate');
  if (gate) gate.classList.add('hidden');
  try {
    await fetch('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: state.sessionId, action: 'abort' }),
    });
  } catch (err) { /* best-effort */ }
  handleGlobalError('Pipeline aborted by user.');
}

function showError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.classList.remove('hidden'); }
}

function hideError(id) {
  const el = document.getElementById(id);
  if (el) { el.classList.add('hidden'); }
}

/* ── Restart ── */
function restart() {
  clearSessionId();
  state.phase = 'INPUT';
  state.sessionId = null;
  state.questions = [];
  state.cycle = 1;
  if (state.ws) { state.ws.close(); state.ws = null; }

  // Reset agent cards
  ['researcher', 'architect_v1', 'critic', 'architect_v2', 'auditor'].forEach(agent => {
    const card = document.getElementById(`card-${agent}`);
    const badge = document.getElementById(`badge-${agent}`);
    const output = document.getElementById(`output-${agent}`);
    if (card) { card.classList.remove('running', 'done'); }
    if (badge) { badge.className = 'status-badge status-waiting'; badge.textContent = 'Waiting'; }
    if (output) { output.textContent = ''; output.classList.remove('visible'); }
    // Remove cycle labels
    if (card) {
      const cl = card.querySelector('.cycle-label');
      if (cl) cl.remove();
    }
  });

  const dl = document.getElementById('right-download-area');
  if (dl) dl.remove();

  document.getElementById('global-error').classList.add('hidden');
  document.getElementById('idea-input').value = '';
  document.getElementById('questions-container').innerHTML = '';
  updateCycleIndicator(1);

  setPhase('INPUT');
}

/* ── Utility ── */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
