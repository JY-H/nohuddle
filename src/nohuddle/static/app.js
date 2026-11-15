const $ = (s) => document.querySelector(s);
const stage = $('#stage');
const frames = new Map(); // game id -> { iframe, label, src, nonce }
let state = null;
let lastEventT = 0;
let dragging = false;

const post = (path, body) => fetch(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
}).then(async (r) => { const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j; });

const fmtDur = (s) => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function badge(g, now) {
  if (g.state === 'break') {
    const what = {
      'no scorebug': 'NO SCOREBUG', 'no overlays': 'NOT GAME', halftime: 'HALFTIME',
      highlight: 'HIGHLIGHTS', 'other game': 'OTHER GAME',
    }[g.detail] || 'BREAK';
    return `<span class="badge break" title="${esc(g.detail)}">${what} ${fmtDur(now - g.since)}</span>`;
  }
  if (g.state === 'stalled') return '<span class="badge stalled">NOT PLAYING</span>';
  if (g.state === 'learning') return '<span class="badge learning" title="Hidden until its scorebug confirms live play">CHECKING</span>';
  return `<span class="badge live">LIVE</span>${g.laggy ? ' <span class="badge laggy">LAGGY</span>' : ''}`;
}

// ---------- stage ----------

// Rectangles (fractions of the stage) for n visible games; the first is the game with audio.
const DRAWER_HEIGHT = 0.2;  // share of the stage the drawer of hidden games takes when open
const drawerOpen = () => localStorage.getItem('drawer') === 'on';
function toggleDrawer() {
  localStorage.setItem('drawer', drawerOpen() ? 'off' : 'on');
  renderStage();
}

// Where the draggable divider sits, per layout (fraction of the stage width); remembered.
const SPLIT_DEFAULTS = { two: 0.5, featured: 2 / 3 };
const splits = { ...SPLIT_DEFAULTS, ...JSON.parse(localStorage.getItem('splits') || '{}') };
const splitKind = (n) => (n === 2 ? 'two' : n >= 3 && n <= 4 ? 'featured' : null);

function layout(n) {
  if (n <= 1) return [[0, 0, 1, 1]];
  if (n === 2) {
    const x = splits.two;
    return [[0, 0, x, 1], [x, 0, 1 - x, 1]];
  }
  if (n <= 4) {
    const x = splits.featured, rest = n - 1;
    return [[0, 0, x, 1], ...Array.from({ length: rest }, (_, i) => [x, i / rest, 1 - x, 1 / rest])];
  }
  const cols = 3, rows = Math.ceil((n + 3) / cols);
  const cells = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (r < 2 && c < 2) continue; // taken by the featured game
    cells.push([c / cols, r / rows, 1 / cols, 1 / rows]);
  }
  return [[0, 0, 2 / cols, 2 / rows], ...cells.slice(0, n - 1)];
}

function place(el, [x, y, w, h]) {
  Object.assign(el.style, { left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` });
}

function renderStage() {
  const ids = new Set(state.games.map((g) => g.id));
  for (const [id, f] of frames) {
    if (!ids.has(id)) { f.iframe.remove(); f.label.remove(); frames.delete(id); }
  }
  for (const g of state.games) {
    let f = frames.get(g.id);
    if (!f) {
      const iframe = document.createElement('iframe');
      iframe.dataset.game = g.id;
      iframe.allow = 'autoplay; fullscreen; encrypted-media';
      iframe.referrerPolicy = 'no-referrer';
      const label = document.createElement('div');
      label.className = 'tile-label';
      label.innerHTML = '<span></span><button class="tile-dbg" title="Something look wrong? See what the detector sees">🐞</button>';
      // Clicking a main tile pins it; clicking a drawer tile does nothing (it's just for a look).
      label.onclick = () => { if (!label.classList.contains('drawer-tile')) post('/api/pin', { id: g.id }); };
      label.querySelector('.tile-dbg').onclick = (e) => { e.stopPropagation(); openDebug(g.id); };
      stage.append(iframe, label);
      f = { iframe, label, src: null, nonce: null };
      frames.set(g.id, f);
    }
    if (g.embed_url && (f.src !== g.embed_url || f.nonce !== g.nonce)) {
      f.iframe.src = g.embed_url; // assigning (even the same URL) reloads the player
      f.src = g.embed_url; f.nonce = g.nonce;
    }
  }

  // Hidden games (on a break, still checking) can be peeked at in a drawer along the bottom:
  // small, muted, labelled with why they're hidden. Otherwise they play under the curtain.
  const hidden = state.games.filter((g) => !state.visible.includes(g.id));
  const D = drawerOpen() && hidden.length ? DRAWER_HEIGHT : 0;
  const rects = layout(state.visible.length).map(([x, y, w, h]) => [x, y * (1 - D), w, h * (1 - D)]);
  const kind = state.mode === 'multi' ? splitKind(state.visible.length) : null;
  $('#splitter').hidden = !kind;
  if (kind) Object.assign($('#splitter').style, { left: `${splits[kind] * 100}%`, bottom: `${D * 100}%` });
  $('#empty').style.bottom = `${D * 100}%`;
  $('#drawer-toggle').classList.toggle('on', drawerOpen());
  const shown = new Map(state.visible.map((id, i) => [id, rects[i]]));
  const tileW = Math.min(0.25, 1 / Math.max(1, hidden.length));
  hidden.forEach((g, i) => { if (D) shown.set(g.id, [i * tileW, 1 - D, tileW, D]); });
  const byId = new Map(state.games.map((g) => [g.id, g]));
  for (const [id, f] of frames) {
    const rect = shown.get(id);
    const inDrawer = !!rect && !state.visible.includes(id);
    f.iframe.classList.toggle('shown', !!rect);
    place(f.iframe, rect || [0, 0, 1, 1]);
    f.label.hidden = !rect;
    if (rect) place(f.label, rect);
    f.label.classList.toggle('audio', id === state.active);
    f.label.classList.toggle('drawer-tile', inDrawer);
    const g = byId.get(id);
    f.label.querySelector('span').innerHTML = inDrawer
      ? `${esc(g.title)} ${badge(g, state.now)}`
      : `${id === state.active ? '🔊 ' : ''}${esc(g.title)}`;
  }

  const empty = state.games.length > 0 && state.visible.length === 0;
  $('#empty').hidden = !empty;
  if (empty) {
    const solo = state.games.length === 1 ? state.games[0] : null;
    const why = solo && { halftime: 'Halftime', highlight: 'Showing highlights', 'other game': 'Showing another game' }[solo.detail];
    $('#empty-title').textContent = solo?.state === 'learning' ? 'Getting a read on the broadcast…'
      : solo ? `${why || 'Commercial break'} · ${fmtDur(state.now - solo.since)}` : 'All games on break';
    $('#empty-sub').textContent = solo?.state === 'learning'
      ? `${solo.title} is shown as soon as its scorebug confirms live play.`
      : solo ? `${solo.title} is muted and hidden until it's back${solo.detail === 'halftime' ? ' for the 3rd quarter' : ''}.`
      : 'Muted. Switching back as soon as one returns.';
    $('#empty-list').innerHTML = solo ? '' : state.games.map((g) => `<li>${esc(g.title)} ${badge(g, state.now)}</li>`).join('');
  }
  if (wasEmpty && !empty && chimeOn()) chime();
  wasEmpty = empty;
}

// A soft two-note chime when play comes back after everything was on a break.
let wasEmpty = false, audio = null;
const chimeOn = () => localStorage.getItem('chime') !== 'off';
function chime() {
  try {
    audio = audio || new AudioContext();
    const now = audio.currentTime;
    [660, 880].forEach((f, i) => {
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = f;
      g.gain.setValueAtTime(0, now + i * 0.18);
      g.gain.linearRampToValueAtTime(0.15, now + i * 0.18 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.001, now + i * 0.18 + 0.5);
      o.connect(g).connect(audio.destination);
      o.start(now + i * 0.18);
      o.stop(now + i * 0.18 + 0.55);
    });
  } catch (e) { /* audio not available yet */ }
}

// Drag the divider between tiles to resize them; double-click to reset.
$('#splitter').addEventListener('pointerdown', (e) => {
  const kind = splitKind(state.visible.length);
  if (!kind) return;
  e.preventDefault();
  const el = e.currentTarget;
  el.setPointerCapture(e.pointerId);
  document.body.classList.add('resizing');
  const move = (ev) => {
    const r = stage.getBoundingClientRect();
    splits[kind] = Math.min(0.85, Math.max(0.15, (ev.clientX - r.left) / r.width));
    renderStage();
  };
  const up = () => {
    el.removeEventListener('pointermove', move);
    document.body.classList.remove('resizing');
    localStorage.setItem('splits', JSON.stringify(splits));
  };
  el.addEventListener('pointermove', move);
  el.addEventListener('pointerup', up, { once: true });
});
$('#splitter').addEventListener('dblclick', () => {
  const kind = splitKind(state.visible.length);
  if (!kind) return;
  splits[kind] = SPLIT_DEFAULTS[kind];
  localStorage.setItem('splits', JSON.stringify(splits));
  renderStage();
});

// ---------- panel ----------

function renderPanel() {
  document.body.classList.toggle('single', state.mode === 'single');
  for (const b of document.querySelectorAll('#mode button')) b.classList.toggle('on', b.dataset.mode === state.mode);
  $('#auto').checked = state.auto;
  const pinned = state.games.find((g) => g.id === state.pinned);
  $('#pinned').hidden = !pinned;
  if (pinned) $('#pinned-title').textContent = pinned.title;

  // Don't rebuild while the user is dragging or has a source dropdown open.
  if (!dragging && !$('#watchlist').contains(document.activeElement?.closest('select'))) {
    $('#watchlist').innerHTML = state.games.map((g, i) => `
      <li draggable="true" data-id="${esc(g.id)}" class="${g.id === state.active ? 'active' : ''}">
        <span class="grip">⋮⋮</span>
        <span class="title">${i + 1}. ${esc(g.title)}${g.clock ? ` <small class="clock">${esc(g.clock)}</small>` : ''}${g.network ? ` <small class="clock">· ${esc(g.network)}</small>` : ''}</span>
        <span>${badge(g, state.now)}</span>
        <span class="actions">
          <button data-act="pin">${g.id === state.pinned ? '📌 Pinned' : 'Watch now'}</button>
          ${g.state === 'break'
            ? '<button data-act="live" title="Wrongly detected as a break: forget what this game looks like and relearn">Not a break</button>'
            : '<button data-act="break" title="It is a commercial but wasn\'t detected">It\'s a break</button>'}
          ${g.embeds.length > 1 ? `<select data-act="mirror">${g.embeds.map((_, j) =>
            `<option value="${j}" ${j === g.embed_index ? 'selected' : ''}>Source ${j + 1}</option>`).join('')}</select>` : ''}
          <button data-act="debug" title="Something look wrong? See what the detector sees">🐞</button>
          <button data-act="remove" title="Remove">✕</button>
        </span>
      </li>`).join('');
  }
  $('#watch-empty').hidden = state.games.length > 0;

  if (document.activeElement?.closest('details') == null) {
    $('#break_after').value = state.settings.break_after;
    $('#live_after').value = state.settings.live_after;
    $('#site').value = state.site;
    $('#fantasycast_url').value = state.fantasycast_url;
  }

  $('#events').innerHTML = [...state.events].reverse().map((e) =>
    `<li><time>${new Date(e.t * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</time>${esc(e.msg)}</li>`).join('');
  const latest = state.events[state.events.length - 1];
  if (latest && latest.t > lastEventT) {
    if (lastEventT) toast(latest.msg);
    lastEventT = latest.t;
  }
}

let toastTimer;
function toast(msg) {
  $('#toast').textContent = msg; $('#toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4000);
}

$('#watchlist').addEventListener('click', (e) => {
  const act = e.target.dataset.act;
  const id = e.target.closest('li')?.dataset.id;
  if (!act || !id) return;
  if (act === 'debug') return openDebug(id);
  if (act === 'pin') post('/api/pin', { id: state.pinned === id ? null : id });
  if (act === 'live' || act === 'break') {
    post('/api/override', { id, state: act }).then(() => toast('Correction noted — report saved for later'));
  }
  if (act === 'remove') post('/api/unwatch', { id });
});
$('#watchlist').addEventListener('change', (e) => {
  if (e.target.dataset.act === 'mirror') post('/api/mirror', { id: e.target.closest('li').dataset.id, index: +e.target.value });
});

// Drag-and-drop priority.
$('#watchlist').addEventListener('dragstart', (e) => { dragging = true; e.target.classList.add('dragging'); });
$('#watchlist').addEventListener('dragover', (e) => {
  e.preventDefault();
  const list = $('#watchlist'), dragged = list.querySelector('.dragging');
  const after = [...list.querySelectorAll('li:not(.dragging)')].find((li) => {
    const r = li.getBoundingClientRect(); return e.clientY < r.top + r.height / 2;
  });
  list.insertBefore(dragged, after || null);
});
$('#watchlist').addEventListener('dragend', async (e) => {
  e.target.classList.remove('dragging');
  const ids = [...document.querySelectorAll('#watchlist li')].map((li) => li.dataset.id);
  await post('/api/order', { ids });
  dragging = false;
});

for (const b of document.querySelectorAll('#mode button')) b.onclick = () => post('/api/auto', { mode: b.dataset.mode });
$('#auto').onchange = (e) => post('/api/auto', { auto: e.target.checked });
$('#unpin').onclick = () => post('/api/pin', { id: null });
$('#chime').checked = chimeOn();
$('#chime').onchange = (e) => localStorage.setItem('chime', e.target.checked ? 'on' : 'off');
$('#panel-toggle').onclick = () => document.body.classList.toggle('panel-hidden');
$('#drawer-toggle').onclick = toggleDrawer;

// Your fantasy matchup, refreshed while the drawer is open.
let fantasyTimer = null;
function toggleFantasy() {
  const open = $('#fantasy').hidden;
  $('#fantasy').hidden = !open;
  $('#fantasy-toggle').classList.toggle('on', open);
  localStorage.setItem('fantasy', open ? 'on' : 'off');
  clearInterval(fantasyTimer);
  if (open) { loadFantasy(); fantasyTimer = setInterval(loadFantasy, 30000); }
}
const pts = (x) => (x == null ? '–' : x.toFixed(1));
function fantasyTeam(side) {
  const rows = (list, bench) => list.map((p) => `
    <div class="fx-row${bench ? ' bench' : ''}"><span class="slot">${esc(p.slot)}</span>
      <span class="who">${esc(p.name)}<small>${esc(p.team)}</small></span>
      <span class="p">${pts(p.points)}<small>${pts(p.projected)}</small></span></div>`).join('');
  return `<div class="fx-side"><h4>${esc(side.name)}</h4>${rows(side.players.filter((p) => p.starter))}
    <h4>Bench</h4>${rows(side.players.filter((p) => !p.starter), true)}</div>`;
}
async function loadFantasy() {
  const d = await (await fetch('/api/fantasy')).json().catch(() => ({ ok: false, error: 'App not reachable.' }));
  if (!d.ok) { $('#fantasy-body').innerHTML = `<p class="hint">${esc(d.error)}</p>`; return; }
  $('#fantasy-title').textContent = `Fantasy · Week ${d.week}`;
  if (!d.me || !d.opp) { $('#fantasy-body').innerHTML = '<p class="hint">No matchup this week.</p>'; return; }
  const line = (s, lead) => `<span class="team${lead ? ' lead' : ''}">${esc(s.name)}</span><span class="pts${lead ? ' lead' : ''}">${pts(s.points)}</span>
    <span class="sub">proj ${pts(s.projected)}${s.win_probability != null ? ` · ${Math.round(s.win_probability * 100)}% to win` : ''}</span><span></span>`;
  const meLead = d.me.points >= d.opp.points;
  $('#fantasy-body').innerHTML = `<div class="fx-score">${line(d.me, meLead)}${line(d.opp, !meLead)}</div>
    ${fantasyTeam(d.me)}${fantasyTeam(d.opp)}<p class="hint">Points · projected. Updates every 30s.</p>`;
}
$('#fantasy-toggle').onclick = toggleFantasy;
$('#espn-open').onclick = () => post('/api/espn-open', {});

// ---------- debug panel ----------

let debugId = null, debugTimer = null, debugInfo = null;

function openDebug(id) {
  debugId = id;
  $('#debug').hidden = false;
  $('#dbg-saved').textContent = '';
  $('#dbg-note').value = '';
  refreshDebug();
  clearInterval(debugTimer);
  debugTimer = setInterval(refreshDebug, 2000);
}

function closeDebug() {
  debugId = null;
  clearInterval(debugTimer);
  $('#debug').hidden = true;
}

async function refreshDebug() {
  if (!debugId) return;
  const id = debugId;
  const r = await fetch(`/api/debug/${id}.json`);
  if (!r.ok || id !== debugId) return;
  debugInfo = await r.json();
  const d = debugInfo, now = Date.now() / 1000;
  $('#dbg-title').textContent = d.title;
  $('#dbg-status').innerHTML = `${badge({ ...d, laggy: d.laggy }, now)} ${d.clock_display ? `<small class="clock">${esc(d.clock_display)}</small>` : ''}`;
  $('#dbg-img').src = `/api/debug/${id}.png?t=${Date.now()}`;
  const furthest = d.clock.max_elapsed;
  $('#dbg-clockinfo').textContent = furthest == null ? '(no position yet)'
    : `furthest point: Q${Math.min(5, Math.floor(furthest / 900) + 1)} ${fmtDur(900 - (furthest % 900))}`;
  $('#dbg-readings').innerHTML = [...d.readings].reverse().slice(0, 12).map((x) => {
    const ago = `${Math.round(now - x.t)}s ago`;
    if (x.resync != null) return `<tr class="resync"><td>${ago}</td><td>resync</td><td>clock ran steadily; position corrected</td></tr>`;
    const parsed = x.clock == null ? '—' : `${x.quarter ? `Q${x.quarter}` : 'Q?'} ${fmtDur(x.clock)}`;
    return `<tr><td>${ago}</td><td>${parsed}</td><td>${esc((x.raw || []).join(' · ')) || '<i>no text</i>'}</td></tr>`;
  }).join('') || '<tr><td>No readings yet.</td></tr>';
}

$('#dbg-close').onclick = closeDebug;
$('#debug').onclick = (e) => { if (e.target.id === 'debug') closeDebug(); };
$('#dbg-save').onclick = async () => {
  try {
    const r = await post('/api/report', { id: debugId, note: $('#dbg-note').value });
    $('#dbg-saved').textContent = `Saved to ${r.path.replace(/^.*\.config/, '~/.config')}`;
    toast('Report saved — tell Claude "I saved a report"');
  } catch (err) { $('#dbg-saved').textContent = `Couldn't save: ${err.message}`; }
};
$('#dbg-copy').onclick = async () => {
  await navigator.clipboard.writeText(JSON.stringify({ note: $('#dbg-note').value, ...debugInfo }, null, 2));
  $('#dbg-saved').textContent = 'Copied.';
};

// ---------- keys ----------

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && debugId) return closeDebug();
  if (e.target.matches('input, select, textarea')) return;
  if ((e.key === 'd' || e.key === 'D') && state?.active) return openDebug(state.active);
  if (e.key === 's' || e.key === 'S') document.body.classList.toggle('panel-hidden');
  else if (e.key === 'm' || e.key === 'M') post('/api/auto', { mode: state.mode === 'single' ? 'multi' : 'single' });
  else if (e.key === 'f' || e.key === 'F') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  else if (e.key === 'a' || e.key === 'A') post('/api/pin', { id: null });
  else if (e.key === 'b' || e.key === 'B') toggleDrawer();
  else if (e.key === 'c' || e.key === 'C') toggleFantasy();
  else if (/^[1-9]$/.test(e.key) && state.games[+e.key - 1]) post('/api/pin', { id: state.games[+e.key - 1].id });
});

// ---------- loop ----------

async function poll() {
  try {
    state = await (await fetch('/api/state')).json();
    renderStage();
    renderPanel();
  } catch (err) { /* server restarting */ }
}

(async () => {
  await poll();
  $('#league').value = state?.league || 'nfl';
  if (localStorage.getItem('fantasy') === 'on') toggleFantasy();
  loadGames();
  setInterval(poll, 500);
})();
