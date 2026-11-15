// A little endless runner for when every game is on a break: a football player running down the
// field, hurdling tackling dummies and ducking under footballs. Space / ↑ to jump, ↓ to duck.
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
      // A football in flight: high ones you duck, low ones you jump.
      const high = Math.random() < 0.6;
      obstacles.push({ kind: 'ball', x: W + 20, y: high ? GROUND - 46 : GROUND - 20, w: 26, h: 14 });
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

  function drawPlayer() {
    const { x, y } = player, duck = player.ducking && y >= GROUND;
    const leg = Math.sin(player.stride / 9) * (y < GROUND ? 0 : 6);
    ctx.save();
    ctx.translate(x, y);
    if (duck) ctx.scale(1.2, 0.6);
    ctx.fillStyle = '#e7e9ee';  // pants and legs
    ctx.fillRect(6, -14, 5, 14 + Math.min(0, leg));
    ctx.fillRect(14, -14, 5, 14 - Math.max(0, leg));
    ctx.fillStyle = '#4c8dff';  // jersey
    ctx.fillRect(4, -32, 18, 18);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px system-ui';
    ctx.fillText('7', 10, -19);
    ctx.fillStyle = '#c9d1dc';  // helmet and facemask
    ctx.beginPath(); ctx.arc(14, -38, 8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#4c8dff';
    ctx.fillRect(10, -46, 3, 16);
    ctx.strokeStyle = '#555'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(20, -40); ctx.lineTo(24, -40); ctx.moveTo(20, -36); ctx.lineTo(24, -36); ctx.stroke();
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
