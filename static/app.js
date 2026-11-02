/* ── State ── */
const state = {
  phase: 'INPUT',   // INPUT | CLARIFYING | RUNNING | REVIEWING | DONE
  sessionId: null,
  questions: [],
  ws: null,
  cycle: 1,
};

const CYCLE_AGENTS = ['researcher', 'architect_v1', 'critic', 'interrogator'];

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
    const res = await fetch('/api/clarify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ idea }),
    });
    const data = await res.json();
    state.sessionId = data.session_id;
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
      body: JSON.stringify({ session_id: state.sessionId, answers }),
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
      case 'awaiting_review':handleAwaitingReview(msg.data.critique, msg.data.cycle, msg.data.questions); break;
      case 'cycle_start':    handleCycleStart(msg.data.cycle);                     break;
      case 'pipeline_done':  handlePipelineDone(msg.data.download_url);            break;
      case 'error':          handleGlobalError(msg.data.message);                  break;
    }
  };

  ws.onerror = () => handleGlobalError('WebSocket connection failed.');
}

/* ── Agent card handlers ── */
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
function handleAwaitingReview(critique, cycle, questions) {
  state.cycle = cycle;
  document.getElementById('critique-panel').textContent = critique;
  document.getElementById('review-cycle-badge').textContent = `Cycle ${cycle} of 3`;

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
function handlePipelineDone(downloadUrl) {
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

  setPhase('DONE');
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
