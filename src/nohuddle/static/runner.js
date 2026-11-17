// A little endless runner for when every game is on a break: a football player running down the
// field, hurdling tackling dummies and footballs,
// and staying grounded under the high ones. Space / ↑ to jump, ↓ to duck.
const Runner = (() => {
  const W = 720, H = 200, GROUND = 168;
  const GRAVITY = 0.0028, JUMP = -0.78, FAST_FALL = 0.012;  // px/ms² and px/ms
  let canvas, ctx, raf = null, last = 0, active = false;
  let player, obstacles, speed, distance, nextGap, over, started, keys = { down: false };
  let best = +(localStorage.getItem('runner-best') || 0);

  function reset() {
    player = { x: 60, y: GROUND, vy: 0, ducking: false, stride: 0 };
    obstacles = [];
    speed = 0.32;  // px/ms
    distance = 0;
    nextGap = 500;
    over = false;
  }

  function jump() {
    if (over) { reset(); started = true; return; }
    started = true;
    if (player.y >= GROUND) player.vy = JUMP;
  }

  function spawn() {
    const score = distance / 10;
    if (score > 300 && Math.random() < 0.3) {
      // A football in flight: a low one you jump; one at head height you duck (or jump); a high one
      // sails over your head but sits where a jump would take you (too high to clear), so the only
      // way past is to keep your feet down.
      const lift = [20, 41, 100, 100][Math.floor(Math.random() * 4)];
      obstacles.push({ kind: 'ball', x: W + 20, y: GROUND - lift, w: 26, h: 14 });
    } else {
      const tall = Math.random() < 0.35, pair = score > 150 && Math.random() < 0.3;
      for (let i = 0; i < (pair ? 2 : 1); i++) {
        obstacles.push({ kind: 'dummy', x: W + 20 + i * 22, y: GROUND, w: 16, h: tall ? 46 : 32 });
      }
    }
    nextGap = 260 + Math.random() * 380 + speed * 300;
  }

  function hitbox() {
    const h = player.ducking && player.y >= GROUND ? 22 : 40;
    return { x: player.x + 4, y: player.y - h, w: 18, h };
  }

  function update(dt) {
    if (!started || over) return;
    speed = Math.min(0.32 + distance / 60000, 0.85);
    distance += speed * dt;
    player.ducking = keys.down;
    player.vy += (GRAVITY + (keys.down && player.y < GROUND ? FAST_FALL : 0)) * dt;
    player.y = Math.min(GROUND, player.y + player.vy * dt);
    if (player.y >= GROUND) player.vy = 0;
    player.stride += speed * dt;
    for (const o of obstacles) o.x -= speed * dt * (o.kind === 'ball' ? 1.15 : 1);
    obstacles = obstacles.filter((o) => o.x + o.w > -10);
    nextGap -= speed * dt;
    if (nextGap <= 0) spawn();
    const p = hitbox();
    for (const o of obstacles) {
      const oy = o.kind === 'ball' ? o.y - o.h / 2 : o.y - o.h;
      if (p.x < o.x + o.w - 3 && p.x + p.w > o.x + 3 && p.y < oy + o.h - 2 && p.y + p.h > oy + 2) {
        over = true;
        const score = Math.floor(distance / 10);
        if (score > best) { best = score; localStorage.setItem('runner-best', String(best)); }
      }
    }
  }

  function drawField() {
    ctx.fillStyle = '#1f6b32';
    ctx.fillRect(0, GROUND, W, H - GROUND);
    ctx.strokeStyle = 'rgba(255,255,255,.55)';
    ctx.lineWidth = 2;
    const step = 90, off = distance % step;
    for (let x = -off; x < W; x += step) {
      ctx.beginPath(); ctx.moveTo(x, GROUND); ctx.lineTo(x - 14, H); ctx.stroke();
    }
    ctx.fillStyle = 'rgba(255,255,255,.8)';
    ctx.fillRect(0, GROUND, W, 2);
  }

  const JERSEY = '#2f6fe0', TRIM = '#1b3f8a', PANTS = '#e9c46a', SKIN = '#c68a5a', MASK = '#d8dde6';

  function segment(x1, y1, x2, y2, width, color) {
    ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }

  // A two-part limb hanging from (x, y): `a` swings it forward from straight down, `bend` folds the
  // lower half back. Returns where it ends.
  function limb(x, y, a, bend, len, upper, lower, width) {
    const kx = x + Math.sin(a) * len, ky = y + Math.cos(a) * len;
    const ex = kx + Math.sin(a - bend) * len, ey = ky + Math.cos(a - bend) * len;
    segment(x, y, kx, ky, width, upper);
    segment(kx, ky, ex, ey, width - 1, lower);
    return [ex, ey];
  }

  function leg(a, bend) {
    const [fx, fy] = limb(12, -17, a, Math.min(bend, a + 0.7), 8.5, PANTS, JERSEY, 6);
    ctx.fillStyle = '#111';  // cleat
    ctx.beginPath(); ctx.ellipse(fx + 2, fy + 0.5, 4, 2, 0, 0, Math.PI * 2); ctx.fill();
  }

  // A side-on football player facing right, ball tucked under the front arm.
  function drawPlayer() {
    const { x, y } = player, duck = player.ducking && y >= GROUND, air = y < GROUND;
    const p = player.stride / 9, swing = Math.sin(p);
    ctx.save();
    ctx.translate(x, y);
    if (duck) ctx.scale(1.25, 0.55);
    ctx.lineCap = 'round';
    // Far leg and arm first, so the near ones overlap them.
    if (air) leg(-0.5, 0.2); else leg(-swing * 0.6, 0.15 + 0.9 * Math.max(0, Math.cos(p)));
    limb(12, -31, air ? -0.9 : swing * 0.9, -0.6, 6, JERSEY, SKIN, 4);
    if (air) leg(0.9, 1.3); else leg(swing * 0.6, 0.15 + 0.9 * Math.max(0, -Math.cos(p)));
    // Jersey: leaning into the run, wider up top for the shoulder pads.
    ctx.fillStyle = JERSEY;
    ctx.beginPath();
    ctx.moveTo(7, -16); ctx.lineTo(17, -16); ctx.lineTo(26, -29);
    ctx.quadraticCurveTo(27, -35, 20, -35); ctx.lineTo(12, -35);
    ctx.quadraticCurveTo(6, -35, 6, -29); ctx.closePath(); ctx.fill();
    ctx.fillStyle = TRIM;
    ctx.fillRect(7, -18, 11, 2);  // belt
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px system-ui';
    ctx.fillText('7', 10, -20);
    // Near arm, ball in the crook of the elbow.
    limb(20, -31, 0.25, -1.9, 6, JERSEY, SKIN, 4.5);
    ctx.save();
    ctx.translate(25, -25); ctx.rotate(-0.5);
    ctx.fillStyle = '#7a3e1a';
    ctx.beginPath(); ctx.ellipse(0, 0, 5, 3, 0, 0, Math.PI * 2); ctx.fill();
    segment(-2, -0.5, 2, -0.5, 1, '#fff');
    ctx.restore();
    // Helmet: shell with a center stripe, ear hole, and a cage over the face.
    ctx.fillStyle = JERSEY;
    ctx.beginPath(); ctx.arc(20, -41, 7.5, Math.PI * 0.85, Math.PI * 2.25); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(20, -41, 7.5, Math.PI * 1.15, Math.PI * 1.75); ctx.stroke();
    ctx.fillStyle = TRIM;
    ctx.beginPath(); ctx.arc(18.5, -40, 1.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = SKIN;
    ctx.fillRect(24, -42, 3, 5);
    ctx.lineCap = 'butt';
    segment(24, -43, 29, -43, 1.2, MASK);
    segment(24, -39.5, 29.5, -39.5, 1.2, MASK);
    segment(29, -44, 29, -36, 1.2, MASK);
    segment(24, -36, 29, -36, 1.2, MASK);
    ctx.restore();
  }

  function drawObstacles() {
    for (const o of obstacles) {
      if (o.kind === 'dummy') {
        ctx.fillStyle = '#c0392b';
        ctx.fillRect(o.x, o.y - o.h, o.w, o.h);
        ctx.fillStyle = '#e8e8e8';
        ctx.fillRect(o.x, o.y - o.h + 6, o.w, 4);
        ctx.fillStyle = '#333';
        ctx.fillRect(o.x - 3, o.y - 4, o.w + 6, 4);
      } else {
        ctx.save();
        ctx.translate(o.x + o.w / 2, o.y);
        ctx.rotate(-0.25);
        ctx.fillStyle = '#8b4a22';
        ctx.beginPath(); ctx.ellipse(0, 0, o.w / 2, o.h / 2, 0, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(-5, 0); ctx.lineTo(5, 0); ctx.stroke();
        ctx.restore();
      }
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    drawField();
    drawObstacles();
    drawPlayer();
    ctx.fillStyle = '#8a93a3';
    ctx.font = '13px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`BEST ${String(best).padStart(5, '0')}   ${String(Math.floor(distance / 10)).padStart(5, '0')}`, W - 10, 20);
    ctx.textAlign = 'center';
    ctx.fillStyle = '#e7e9ee';
    if (!started) ctx.fillText('Space / ↑ to run and jump · ↓ to duck', W / 2, 70);
    if (over) ctx.fillText('Tackled! Space to run again', W / 2, 70);
    ctx.textAlign = 'left';
  }

  function frame(t) {
    const dt = Math.min(40, t - (last || t));
    last = t;
    update(dt);
    draw();
    raf = active ? requestAnimationFrame(frame) : null;
  }

  function onKey(e, down) {
    if (!active || e.target.matches('input, select, textarea')) return;
    if (e.key === ' ' || e.key === 'ArrowUp') { e.preventDefault(); if (down && !e.repeat) jump(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); keys.down = down; }
  }

  return {
    mount(el) {
      canvas = el;
      canvas.width = W; canvas.height = H;
      ctx = canvas.getContext('2d');
      canvas.addEventListener('pointerdown', jump);
      document.addEventListener('keydown', (e) => onKey(e, true));
      document.addEventListener('keyup', (e) => onKey(e, false));
      reset();
      started = false;
    },
    // Runs only while the "all games on break" screen is up; a game coming back pauses it.
    setActive(on) {
      if (on === active) return;
      active = on;
      keys.down = false;
      if (on && !raf) { last = 0; raf = requestAnimationFrame(frame); }
    },
  };
})();
