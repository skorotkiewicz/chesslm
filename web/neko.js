// Neko, the classic desktop cat, chasing your cursor around the chess page.
// Behavior follows NEKO.BAS and xneko: chase, catch, then scratch, groom, yawn, sleep.
'use strict';
(() => {
  const SIZE = 56;
  const SPEED = 2.5;                      // pixels per tick, classic neko pace
  const STOP = 12;                        // close enough to the cursor
  const PHASE_TICKS = 10;                 // ticks per idle animation
  const IDLE_CYCLE = ['stop', 'scratch', 'scratch', 'groom', 'groom', 'yawn', 'sleep'];
  const FUR = '#26292d';
  const EYE = '#f5f3ec';

  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  canvas.style.cssText = 'position:fixed;left:0;top:0;z-index:9999;pointer-events:none;';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.append(canvas);
  const ctx = canvas.getContext('2d');

  const cat = window.neko = {x: innerWidth / 2, y: innerHeight / 2, tx: null, ty: null, state: 'rest', tick: 0, facing: 1};
  let phase = 0;
  // With reduced motion the cat just sits and watches the game.
  const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;

  addEventListener('pointermove', ({clientX: x, clientY: y}) => {
    if (calm) return;
    cat.tx = x;
    cat.ty = y;
    if (cat.state !== 'run' && Math.hypot(cat.tx - cat.x, cat.ty - cat.y) > STOP) {
      cat.state = 'alert';
      phase = 0;
    }
  });

  const puff = (x, y, rx, ry, fill = FUR) => {
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, 7);
    ctx.fillStyle = fill;
    ctx.fill();
  };
  const eye = (x, y, shut) => {
    if (shut) {
      ctx.strokeStyle = EYE;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x - 2, y);
      ctx.lineTo(x + 2, y);
      ctx.stroke();
      return;
    }
    puff(x, y, 2.4, 2.4, EYE);
    puff(x + 0.8, y, 1.1, 1.1, '#101215');
  };
  const ear = (x, y) => {
    ctx.beginPath();
    ctx.moveTo(x - 4, y + 3);
    ctx.lineTo(x, y - 6);
    ctx.lineTo(x + 4, y + 2);
    ctx.fill();
  };
  const leg = (x, swing) => {
    ctx.strokeStyle = FUR;
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x, 4);
    ctx.lineTo(x + swing, 17);
    ctx.stroke();
  };
  const tail = lift => {
    ctx.strokeStyle = FUR;
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(-13, 2);
    ctx.quadraticCurveTo(-24, -lift, -20, -12 - lift);
    ctx.stroke();
  };
  const head = (x, y, shut) => {
    puff(x, y, 10.5, 9.5);
    ear(x - 6, y - 7);
    ear(x + 6, y - 7);
    eye(x + 2, y - 1, shut);
    eye(x + 9, y - 1, shut);
  };

  const poses = {
    rest(t) {
      puff(0, 4, 13, 15);
      tail(2 + (t % 8 < 4 ? 0 : 3));
      head(4, -15, t % 40 > 36);
    },
    stop() {
      puff(0, 4, 13, 15);
      tail(4);
      head(4, -15, false);
    },
    alert() {
      puff(0, 3, 12, 15);
      tail(12);
      head(4, -16, false);
    },
    run(t) {
      const swing = t % 2 ? 3 : -3;
      const hop = t % 2 ? -1 : 0;
      tail(6 + (t % 2) * 4);
      leg(-8, swing);
      leg(-5, -swing);
      leg(6, -swing);
      leg(9, swing);
      puff(0, hop, 16, 10);
      head(15, -7 + hop, false);
    },
    scratch(t) {
      puff(-2, 8, 15, 9);
      tail(10);
      head(12, -2, false);
      leg(-8, 0);
      leg(-4, 0);
      leg(12, t % 2 ? 4 : -2);            // pawing at the ground
    },
    groom(t) {
      puff(0, 4, 13, 15);
      tail(6);
      head(7, -10, t % 4 > 2);            // head turned toward the paw
      leg(8, -6);
    },
    yawn() {
      puff(0, 4, 13, 15);
      tail(3);
      head(4, -16, true);
      puff(9, -9, 3, 4, '#101215');       // open mouth
    },
    sleep(t) {
      puff(0, 8, 18, 8);
      head(13, 3, true);
      ctx.fillStyle = EYE;
      ctx.font = '10px system-ui';
      ctx.fillText('z', 20, -4 - (t % 4) * 3);
    },
  };

  setInterval(() => {
    if (document.hidden) return;
    cat.tick++;
    const chasing = cat.tx !== null;
    const dx = chasing ? cat.tx - cat.x : 0;
    const dy = chasing ? cat.ty - cat.y : 0;
    const distance = chasing ? Math.hypot(dx, dy) : 0;
    if (cat.state === 'run') {
      if (distance <= STOP) {
        cat.state = 'stop';
        phase = 0;
      } else {
        cat.x += Math.min(SPEED, distance) * dx / distance;
        cat.y += Math.min(SPEED, distance) * dy / distance;
        cat.facing = dx >= 0 ? 1 : -1;
      }
    } else if (cat.state === 'alert') {
      cat.state = distance > STOP ? 'run' : 'stop';
    } else if (cat.state !== 'rest' && cat.tick % PHASE_TICKS === 0) {
      cat.state = IDLE_CYCLE[phase++ % IDLE_CYCLE.length];
    }
    cat.x = Math.min(Math.max(cat.x, 8), innerWidth - 8);
    cat.y = Math.min(Math.max(cat.y, 8), innerHeight - 8);
    canvas.style.left = (cat.x - SIZE / 2) + 'px';
    canvas.style.top = (cat.y - SIZE / 2) + 'px';
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.save();
    ctx.translate(SIZE / 2, SIZE / 2);
    puff(0, 20, 13, 3, 'rgba(0,0,0,0.15)');
    ctx.scale(cat.facing, 1);
    poses[cat.state]?.(cat.tick);
    ctx.restore();
  }, 90);
})();
