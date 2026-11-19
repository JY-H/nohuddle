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
  // The break screen's tabs: the runner game, and (with NFL games on) the scorigami watch.
  const nfl = state.games.some((g) => g.league === 'nfl');
  const pane = nfl ? breakPane : 'runner';
  $('#break-tabs').hidden = !nfl;
  for (const b of document.querySelectorAll('#break-tabs button')) b.classList.toggle('on', b.dataset.pane === pane);
  $('#runner').hidden = pane !== 'runner';
  $('#scorigami').hidden = pane !== 'scorigami';
  Runner.setActive(empty && pane === 'runner');
  watchScorigami(empty && pane === 'scorigami');
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

// ---------- scorigami ----------

let breakPane = localStorage.getItem('break-pane') || 'runner', scorigamiTimer = null;
document.querySelector('#break-tabs').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-pane]');
  if (!b) return;
  breakPane = b.dataset.pane;
  localStorage.setItem('break-pane', breakPane);
  if (state) renderStage();
});
function watchScorigami(on) {
  if (on && !scorigamiTimer) { loadScorigami(); scorigamiTimer = setInterval(loadScorigami, 30000); }
  if (!on && scorigamiTimer) { clearInterval(scorigamiTimer); scorigamiTimer = null; }
}
const pct = (p) => (p >= 0.999 ? '100%' : p > 0 && p < 0.001 ? '<0.1%' : `${(p * 100).toFixed(1)}%`);
const year = (d) => (d ? d.slice(0, 4) : '');
async function loadScorigami() {
  const d = await (await fetch('/api/scorigami')).json().catch(() => ({ ok: false, error: 'App not reachable.' }));
  const el = $('#sg-games');
  if (!d.ok) { el.innerHTML = `<p class="hint">${esc(d.error)}</p>`; return; }
  const games = d.games.map((g) => {
    if (!g.found) return `<div class="sg-game"><span class="sg-score">${esc(g.title)}</span><span class="sg-p">–</span>
      <span class="sg-now">Not on ESPN's scoreboard today.</span></div>`;
    const [a, b] = g.teams;
    const hi = Math.max(a.score, b.score), lo = Math.min(a.score, b.score);
    const never = g.now.times === 0;
    const now = g.state === 'pre' ? 'Not started yet.'
      : g.state === 'post' ? (never ? `<span class="sg-yes">Final: ${hi}–${lo} — scorigami!</span>`
        : `Final: ${hi}–${lo}, the ${g.now.times + 1}${ordinal(g.now.times + 1)} time.`)
      : never ? `<span class="sg-yes">If it ended now: ${hi}–${lo} has never happened.</span>`
      : `If it ended now: ${hi}–${lo} has happened ${g.now.times}× (last ${year(g.now.last)}).`;
    const shortlist = g.state === 'post' ? [] : g.likeliest.filter((x) => x.p >= 0.005);  // skip long shots
    const likely = shortlist.length
      ? `<span class="sg-likely">Likeliest new final${shortlist.length > 1 ? 's' : ''}: ${shortlist.map((x) => `${x.score} (${pct(x.p)})`).join(' · ')}</span>` : '';
    return `<div class="sg-game">
      <span class="sg-score">${esc(a.abbrev)} ${a.score} – ${b.score} ${esc(b.abbrev)}<small>${esc(g.status)}</small></span>
      <span class="sg-p">${pct(g.p)}<small>scorigami</small></span>
      <span class="sg-now">${now}</span>${likely}</div>`;
  }).join('');
  const latest = d.latest.length
    ? `<p class="hint">Newest scorigami${d.latest.length > 1 ? 's' : ''}: ${d.latest.map((x) => `${esc(x.score)} (${esc(x.game)})`).join(', ')}</p>` : '';
  el.innerHTML = `${games || '<p class="hint">No NFL games on the watchlist.</p>'}${latest}
    <p class="hint">${d.distinct.toLocaleString()} different final scores in NFL history. Chances are approximate.</p>`;
  const mine = d.games.filter((g) => g.found && g.state !== 'pre')
    .map((g) => ({ w: Math.max(...g.teams.map((t) => t.score)), l: Math.min(...g.teams.map((t) => t.score)),
      name: g.teams.map((t) => `${t.abbrev} ${t.score}`).join(' – ') }));
  if (!sgTable || Date.now() - sgTable.at > 600000 || d.distinct !== sgTable.scores.length) {
    const t = await (await fetch('/api/scorigami/table')).json().catch(() => null);
    if (t) sgTable = { at: Date.now(), scores: t.scores };
  }
  if (sgTable) drawScorigamiGrid(sgTable.scores, mine);
}

// The whole table: winning score across the top, losing score down the right, every score
// labelled. Cells are coloured by how often that final has happened (log scale, one blue ramp:
// rare finals sit dark, near the surface; common ones light); finals that never have are empty
// outlines; impossible ones (a loser ahead of the winner, or a team on 1 point) are blocked out.
let sgTable = null, sgHover = null;
const SG = { cell: 13, gap: 1, left: 4, top: 34, right: 46, bottom: 4, maxW: 73, maxL: 51 };
const sgX = (w, step) => SG.left + w * step;  // winning score across
const sgY = (l, step) => SG.top + l * step;  // losing score down
const sgPossible = (w, l) => l <= w && w !== 1 && l !== 1;
const SG_RAMP = ['#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb'];
const SG_BLOCKED = '#1b1f25', SG_NEVER = '#2f3640';
function drawScorigamiGrid(scores, mine) {
  const canvas = $('#sg-canvas'), dpr = window.devicePixelRatio || 1;
  const byKey = new Map(scores.map(([w, l, n, last]) => [w * 1000 + l, { n, last }]));
  const maxW = Math.max(SG.maxW, ...scores.map((s) => s[0]), ...mine.map((m) => m.w));
  const maxL = Math.max(SG.maxL, ...scores.map((s) => s[1]), ...mine.map((m) => m.l));
  const most = Math.max(...scores.map((s) => s[2]));
  const step = SG.cell + SG.gap;
  const W = SG.left + (maxW + 1) * step + SG.right, H = SG.top + (maxL + 1) * step + SG.bottom;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.aspectRatio = `${W} / ${H}`;
  canvas.style.width = `min(100%, ${W}px)`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  // Axis titles and every score.
  const right = sgX(maxW + 1, step);
  ctx.fillStyle = '#c3c9d2'; ctx.font = '600 11px system-ui'; ctx.textAlign = 'center';
  ctx.fillText('Winning team score', (SG.left + right) / 2, 11);
  ctx.save(); ctx.translate(W - 8, (SG.top + sgY(maxL + 1, step)) / 2); ctx.rotate(Math.PI / 2);
  ctx.fillText('Losing team score', 0, 0); ctx.restore();
  ctx.fillStyle = '#8a93a3'; ctx.font = '9px system-ui';
  for (let w = 0; w <= maxW; w++) ctx.fillText(String(w), sgX(w, step) + SG.cell / 2, SG.top - 6);
  ctx.textAlign = 'left';
  for (let l = 0; l <= maxL; l++) ctx.fillText(String(l), right + 4, sgY(l, step) + SG.cell - 3);
  for (let l = 0; l <= maxL; l++) {
    for (let w = 0; w <= maxW; w++) {
      const x = sgX(w, step), y = sgY(l, step), hit = byKey.get(w * 1000 + l);
      if (!sgPossible(w, l)) {
        ctx.fillStyle = SG_BLOCKED;  // gapless, so it reads as one region rather than cells
        ctx.fillRect(x, y, step, step);
      } else if (hit) {
        ctx.fillStyle = SG_RAMP[Math.round(Math.log(hit.n) / Math.log(most) * (SG_RAMP.length - 1))];
        ctx.fillRect(x, y, SG.cell, SG.cell);
      } else {
        ctx.strokeStyle = SG_NEVER; ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, SG.cell - 1, SG.cell - 1);
      }
    }
  }
  for (const m of mine) {  // a 2px ring with a surface gap, so it reads on any cell colour
    const x = sgX(m.w, step), y = sgY(m.l, step);
    ctx.strokeStyle = '#000'; ctx.lineWidth = 4; ctx.strokeRect(x - 2, y - 2, SG.cell + 4, SG.cell + 4);
    ctx.strokeStyle = '#f2f4f7'; ctx.lineWidth = 2; ctx.strokeRect(x - 2, y - 2, SG.cell + 4, SG.cell + 4);
  }
  // Legend ticks where those counts fall on the log scale.
  $('#sg-ticks').innerHTML = [1, 10, 100].filter((n) => Math.log(n) / Math.log(most) < 0.75).concat(most)
    .map((n) => `<span style="left:${(Math.log(n) / Math.log(most)) * 100}%">${n}</span>`).join('');
  sgHover = { byKey, mine, maxW, maxL, step, W };
}
$('#sg-canvas').addEventListener('pointermove', (e) => {
  const tip = $('#sg-tip');
  if (!sgHover) return;
  const r = e.currentTarget.getBoundingClientRect(), k = sgHover.W / r.width;
  const w = Math.floor(((e.clientX - r.left) * k - SG.left) / sgHover.step);
  const l = Math.floor(((e.clientY - r.top) * k - SG.top) / sgHover.step);
  if (w < 0 || l < 0 || w > sgHover.maxW || l > sgHover.maxL || !sgPossible(w, l)) { tip.hidden = true; return; }
  const hit = sgHover.byKey.get(w * 1000 + l);
  const games = sgHover.mine.filter((m) => m.w === w && m.l === l).map((m) => ` · now: ${m.name}`).join('');
  tip.textContent = hit
    ? `${w}–${l}: ${hit.n.toLocaleString()} time${hit.n > 1 ? 's' : ''}, last ${hit.last}${games}`
    : `${w}–${l}: never happened${games}`;
  tip.style.left = `${(sgX(w, sgHover.step) + SG.cell / 2) / k}px`;
  tip.style.top = `${sgY(l, sgHover.step) / k}px`;
  tip.hidden = false;
});
$('#sg-canvas').addEventListener('pointerleave', () => { $('#sg-tip').hidden = true; });
const ordinal = (n) => (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th');

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
  $('#max-tiles').hidden = state.mode !== 'multi';
  if (document.activeElement !== $('#max-tiles')) $('#max-tiles').value = String(state.max_tiles);
  $('#auto').checked = state.auto;
  const pinned = state.games.find((g) => g.id === state.pinned);
  $('#pinned').hidden = !pinned;
  if (pinned) $('#pinned-title').textContent = pinned.title;

  // Don't rebuild while the user is dragging or has a source dropdown open.
  if (!dragging && !$('#watchlist').contains(document.activeElement?.closest('select'))) {
    $('#watchlist').innerHTML = state.games.map((g, i) => `
      <li draggable="true" data-id="${esc(g.id)}" class="${g.id === state.active ? 'active' : ''}">
        <span class="grip">⋮⋮</span>
        <span class="title"><span class="name">${i + 1}. ${esc(g.title)}</span>${g.clock ? ` <small class="clock">${esc(g.clock)}</small>` : ''}${g.network ? ` <small class="clock">· ${esc(g.network)}</small>` : ''}</span>
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
  // Keep the game list's "added" marks in step with the watchlist (e.g. after removing a game).
  const watchedKey = state.games.map((g) => g.id).sort().join('|');
  if (watchedKey !== lastWatchedKey) { lastWatchedKey = watchedKey; if (gameList.length) renderGames(); }

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
$('#max-tiles').onchange = (e) => { post('/api/auto', { max_tiles: +e.target.value }); e.target.blur(); };
$('#auto').onchange = (e) => post('/api/auto', { auto: e.target.checked });
$('#unpin').onclick = () => post('/api/pin', { id: null });
$('#chime').checked = chimeOn();
$('#chime').onchange = (e) => localStorage.setItem('chime', e.target.checked ? 'on' : 'off');
$('#panel-toggle').onclick = () => document.body.classList.toggle('panel-hidden');
$('#drawer-toggle').onclick = toggleDrawer;

// Fantasy: a matchup (yours, or any team's) or the started players in the game with sound.
// Only the tab being looked at is refreshed, every 30s while the drawer is open.
let fantasyTimer = null;
const fx = { tab: localStorage.getItem('fx-tab') || 'matchup', team: localStorage.getItem('fx-team') || '' };
function toggleFantasy() {
  const open = $('#fantasy').hidden;
  $('#fantasy').hidden = !open;
  $('#fantasy-toggle').classList.toggle('on', open);
  localStorage.setItem('fantasy', open ? 'on' : 'off');
  clearInterval(fantasyTimer);
  if (open) { loadFantasy(); fantasyTimer = setInterval(loadFantasy, 30000); }
}
function showFantasyTab(tab, team) {
  fx.tab = tab;
  if (team != null) fx.team = String(team);
  localStorage.setItem('fx-tab', fx.tab);
  localStorage.setItem('fx-team', fx.team);
  for (const b of document.querySelectorAll('.fx-tabs button')) b.classList.toggle('on', b.dataset.tab === tab);
  $('.fx-tabs').classList.toggle('on-screen', tab === 'on-screen');
  $('#fantasy-body').innerHTML = '<p class="hint">Loading…</p>';
  loadFantasy();
}
const pts = (x) => (x == null ? '–' : x.toFixed(1));
const fxRow = (p, extra = '', playing = false) => `
  <div class="fx-row${p.starter ? '' : ' bench'}${p.mine ? ' mine' : ''}${playing ? ' playing' : ''}"><span class="slot">${esc(p.slot)}</span>
    <span class="who">${esc(p.name)}<small>${esc(p.team)}</small>${extra}${p.line ? `<span class="line">${esc(p.line)}</span>` : ''}</span>
    <span class="p">${pts(p.points)}<small>${pts(p.projected)}</small></span></div>`;
// Players in the game being shown are highlighted.
function fantasyTeam(side, onScreen) {
  const row = (p) => fxRow(p, '', onScreen.has(p.team));
  return `<div class="fx-side"><h4>${esc(side.name)}</h4>${side.players.filter((p) => p.starter).map(row).join('')}
    <h4>Bench</h4>${side.players.filter((p) => !p.starter).map(row).join('')}</div>`;
}
async function loadFantasy() {
  const tab = fx.tab;
  const url = tab === 'matchup' ? `/api/fantasy?team=${encodeURIComponent(fx.team)}` : '/api/fantasy/on-screen';
  const d = await (await fetch(url)).json().catch(() => ({ ok: false, error: 'App not reachable.' }));
  if (tab !== fx.tab) return; // switched tabs while this was loading
  if (!d.ok) { $('#fantasy-body').innerHTML = `<p class="hint">${esc(d.error)}</p>`; return; }
  if (d.week) $('#fantasy-title').textContent = `Fantasy · Week ${d.week}`;
  if (tab === 'on-screen') { renderOnScreen(d); return; }
  const sel = $('#fx-team');
  if (document.activeElement !== sel) {
    sel.innerHTML = d.teams.map((t) => `<option value="${t.id}" ${t.id === d.team_id ? 'selected' : ''}>${esc(t.name)}${t.id === d.my_team_id ? ' (you)' : ''}</option>`).join('');
  }
  if (!d.me || !d.opp) { $('#fantasy-body').innerHTML = '<p class="hint">No matchup this week.</p>'; return; }
  const line = (s, lead) => `<span class="team${lead ? ' lead' : ''}">${esc(s.name)}</span><span class="pts${lead ? ' lead' : ''}">${pts(s.points)}</span>
    <span class="sub">proj ${pts(s.projected)}${s.win_probability != null ? ` · ${Math.round(s.win_probability * 100)}% to win` : ''}</span><span></span>`;
  const meLead = d.me.points >= d.opp.points, onScreen = new Set(d.on_screen || []);
  $('#fantasy-body').innerHTML = `<div class="fx-score">${line(d.me, meLead)}${line(d.opp, !meLead)}</div>
    ${fantasyTeam(d.me, onScreen)}${fantasyTeam(d.opp, onScreen)}
    <p class="hint">Points · projected.${onScreen.size ? ` <span class="playing-key">Highlighted</span>: in ${[...onScreen].join('–')}.` : ''} Updates every 30s.</p>`;
}
function renderOnScreen(d) {
  if (!d.game) { $('#fantasy-body').innerHTML = '<p class="hint">No game with sound right now.</p>'; return; }
  const owned = (p) => fxRow(p,
    ` <button class="go" data-team="${p.fantasy_team_id}" title="Show ${esc(p.fantasy_team)}'s matchup">${esc(p.fantasy_team)} →</button>`);
  const group = (title, list, row) => (list.length ? `<h5>${title}</h5>${list.map(row).join('')}` : '');
  const teams = d.teams.map((t) => {
    const body = group('Started', t.started, owned) + group('Bench', t.bench, owned)
      + group('Free agents (10+ pts)', t.free_agents, (p) => fxRow({ ...p, starter: true }, ' <small class="fa">FA</small>'));
    return `<div class="fx-side fx-pro"><h4>${esc(t.name)} <small>${esc(t.abbrev)}</small></h4>
      ${body || '<p class="hint">Nobody rostered, and no free agent past 10 points.</p>'}</div>`;
  }).join('');
  $('#fantasy-body').innerHTML = `${teams || '<p class="hint">Not an NFL game.</p>'}
    <p class="hint">Players rostered in your league (started or bench), and free agents past 10 points. Updates every 30s.</p>`;
}
document.querySelector('.fx-tabs').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-tab]');
  if (b) showFantasyTab(b.dataset.tab);
});
$('#fx-team').onchange = (e) => showFantasyTab('matchup', e.target.value);
for (const b of document.querySelectorAll('.fx-tabs button')) b.classList.toggle('on', b.dataset.tab === fx.tab);
$('.fx-tabs').classList.toggle('on-screen', fx.tab === 'on-screen');
$('#fantasy-body').addEventListener('click', (e) => {
  const b = e.target.closest('button.go');
  if (b) showFantasyTab('matchup', b.dataset.team);
});
$('#fantasy-toggle').onclick = toggleFantasy;
$('#espn-open').onclick = () => post('/api/espn-open', {});

$('#save-settings').onclick = () => post('/api/settings', {
  break_after: +$('#break_after').value, live_after: +$('#live_after').value, site: $('#site').value,
  fantasycast_url: $('#fantasycast_url').value,
}).then(() => { document.activeElement.blur(); toast('Settings saved'); });

// ---------- game list ----------

Runner.mount($('#runner'));

let gameList = [];
let lastWatchedKey = null;
async function loadGames() {
  const league = $('#league').value;
  $('#games').innerHTML = '<li class="hint">Loading…</li>';
  try {
    const r = await fetch(`/api/games?league=${league}`);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    gameList = j.games;
    renderGames();
  } catch (err) {
    $('#games').innerHTML = `<li class="error">${esc(err.message)}</li>`;
  }
}
function renderGames() {
  const q = $('#filter').value.toLowerCase();
  const watched = new Set(state ? state.games.map((g) => g.id) : []);
  const rows = gameList.filter((g) => !q || g.title.toLowerCase().includes(q)).map((g) => {
    const when = g.is_live ? 'LIVE' : new Date(g.starts_at * 1000).toLocaleString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' });
    return `<li class="${g.is_live ? 'live' : ''}"><span class="name">${esc(g.title)}</span><span class="when">${when}</span>
      ${watched.has(g.id) ? '<span class="hint">added</span>' : `<button data-url="${esc(g.url)}" data-title="${esc(g.title)}" data-starts="${g.starts_at}" data-ends="${g.ends_at}">＋</button>`}</li>`;
  });
  $('#games').innerHTML = rows.join('') || '<li class="hint">No games found.</li>';
}
async function addGame(body, button) {
  $('#add-error').hidden = true;
  if (button) { button.disabled = true; button.textContent = '…'; }
  try {
    await post('/api/watch', body);
    await poll();
    renderGames();
  } catch (err) {
    $('#add-error').textContent = err.message; $('#add-error').hidden = false;
    if (button) { button.disabled = false; button.textContent = '＋'; }
  }
}
$('#games').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-url]');
  if (b) addGame({ url: b.dataset.url, title: b.dataset.title, starts_at: +b.dataset.starts, ends_at: +b.dataset.ends }, b);
});
$('#add-url').onsubmit = (e) => { e.preventDefault(); addGame({ url: $('#url').value }); $('#url').value = ''; };
$('#league').onchange = loadGames;
$('#refresh').onclick = loadGames;
$('#filter').oninput = renderGames;

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
  // Keep the page refreshing no matter what: an error in an optional startup step below must
  // never leave the screen frozen on its first snapshot.
  setInterval(poll, 500);
  await poll();
  for (const step of [
    () => { $('#league').value = state?.league || 'nfl'; },
    () => { if (localStorage.getItem('fantasy') === 'on') toggleFantasy(); },
    () => loadGames(),
  ]) {
    try { await step(); } catch (err) { console.error('startup step failed', err); }
  }
})();
