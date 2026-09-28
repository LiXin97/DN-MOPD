/* DN-MOPD project page, p5c_datamotion: data-driven animated results.
   Every plotted or printed value is read from static/results.json and static/multipliers.json. Vanilla JS, hand-built SVG.
   Progressive enhancement: until a chart has its data it stays hidden and the static fallback in index.html stays visible.
   Motion: each chart animates once, when it scrolls into view; if it leaves the viewport mid-animation it jumps to its
   final state; with prefers-reduced-motion every chart renders its final state immediately.
   Merge (p5_all), one rule for every number: a value label only ever shows its own series, either counting up from 0 to
   its own value on entry or showing its final value at once when a toggle changes the series (the bars still ease).
   - lengths: DN-MOPD's bar and label grow from 0 to DN-MOPD's value (they used to start from Label's value)
   - gains: on a cap / seed toggle the value and interval text switch to the new series at once (they used to count
     through the old series' numbers)
   - the gradient-share chart is dropped (it repeated the bento's 94% / 64% meter); the toggle thumb moves by transform only */
(function () {
  'use strict';
  var doc = document.documentElement;
  var NS = 'http://www.w3.org/2000/svg';
  var mqr = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  function reduced() { return !!(mqr && mqr.matches); }
  doc.classList.add('dm-js');

  /* ================================================================ helpers */
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  var ease = {
    out: function (t) { return 1 - Math.pow(1 - t, 3); },
    inOut: function (t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; },
    sine: function (t) { return -(Math.cos(Math.PI * t) - 1) / 2; }
  };
  function phase(ms, start, dur) { return clamp((ms - start) / dur, 0, 1); }
  function S(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function A(n, attrs) { for (var k in attrs) n.setAttribute(k, attrs[k]); }
  function signed(v, d) {
    var s = Math.abs(v).toFixed(d);
    return (v < 0 && parseFloat(s) !== 0 ? '−' : '+') + s;
  }
  function interval(lo, hi, d) { return '[' + signed(lo, d) + ', ' + signed(hi, d) + ']'; }
  function thousands(v) { return String(Math.round(v)).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function parseNum(s) { return parseFloat(String(s).replace(/,/g, '').replace('−', '-')); }
  function fx(v) { return Math.round(v * 10) / 10; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  // rounded data-end bar, square at the baseline (x0)
  function barPath(x0, x1, y, h, r) {
    var w = x1 - x0;
    if (w <= 0.05) return 'M' + x0 + ',' + y + 'v' + h;
    r = Math.min(r, w, h / 2);
    return 'M' + x0 + ',' + y + 'H' + (x1 - r) + 'A' + r + ',' + r + ' 0 0 1 ' + x1 + ',' + (y + r) +
      'V' + (y + h - r) + 'A' + r + ',' + r + ' 0 0 1 ' + (x1 - r) + ',' + (y + h) + 'H' + x0 + 'Z';
  }

  /* One animation = a frame(msElapsed) callback over a fixed duration. finish() jumps to the final frame. */
  var live = [];
  function play(total, frame, done) {
    var h = { live: true, id: 0 };
    function end() { h.live = false; cancelAnimationFrame(h.id); frame(total); if (done) done(); }
    h.finish = function () { if (h.live) end(); };
    h.stop = function () { h.live = false; cancelAnimationFrame(h.id); };
    if (reduced() || total <= 0) { end(); return h; }
    var t0 = null;
    function step(ts) {
      if (!h.live) return;
      if (t0 === null) t0 = ts;
      var ms = ts - t0;
      if (ms >= total) { end(); return; }
      frame(ms);
      h.id = requestAnimationFrame(step);
    }
    frame(0);
    h.id = requestAnimationFrame(step);
    live.push(h);
    return h;
  }
  window.addEventListener('beforeprint', function () { live.forEach(function (h) { h.finish(); }); });

  /* Start an entry animation once the node is well into view; leaving the viewport completes it at once. */
  function whenSeen(node, start) {
    var handle = null;
    if (reduced() || !('IntersectionObserver' in window)) { handle = start(); if (handle) handle.finish(); return; }
    var th = [];
    for (var i = 0; i <= 20; i++) th.push(i / 20);
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var vh = window.innerHeight || 800;
        var seen = en.isIntersecting && (en.intersectionRatio >= 0.55 || en.intersectionRect.height >= vh * 0.4);
        if (!handle && seen) {
          handle = start();
          if (!handle || !handle.live) io.disconnect();
        } else if (handle && !en.isIntersecting) {
          handle.finish();
          io.disconnect();
        } else if (handle && !handle.live) {
          io.disconnect();
        }
      });
    }, { threshold: th });
    io.observe(node);
  }

  /* Re-layout on width changes (no animation), batched to one frame. */
  function onWidth(node, fn) {
    var last = node.clientWidth, raf = 0;
    function check() {
      raf = 0;
      var w = node.clientWidth;
      if (w > 0 && Math.abs(w - last) > 0.5) { last = w; fn(); }
    }
    if ('ResizeObserver' in window) new ResizeObserver(function () { if (!raf) raf = requestAnimationFrame(check); }).observe(node);
    else window.addEventListener('resize', function () { if (!raf) raf = requestAnimationFrame(check); });
  }

  /* Segmented toggle: buttons with aria-pressed and a sliding thumb. */
  function Seg(el, onChange) {
    var thumb = el.querySelector('.dm-seg-thumb');
    var btns = [].slice.call(el.querySelectorAll('button'));
    function active() { for (var i = 0; i < btns.length; i++) if (btns[i].getAttribute('aria-pressed') === 'true') return btns[i]; return btns[0]; }
    var placed = null;
    // merge: the thumb takes the new button's width at once and FLIPs from the old box with translateX + scaleX,
    // so only transform is transitioned (p5c transitioned width)
    function place(animate) {
      var b = active();
      if (!thumb || !b.offsetWidth) return;
      var L = b.offsetLeft, Wd = b.offsetWidth;
      thumb.style.width = Wd + 'px';
      if (animate && placed && !reduced() && el.classList.contains('dm-seg-on')) {
        thumb.style.transition = 'none';
        thumb.style.transform = 'translateX(' + placed.left + 'px) scaleX(' + (placed.width / Wd).toFixed(4) + ')';
        void thumb.offsetWidth;
        thumb.style.transition = '';
      }
      thumb.style.transform = 'translateX(' + L + 'px)';
      placed = { left: L, width: Wd };
    }
    btns.forEach(function (b) {
      b.addEventListener('click', function () {
        if (b.getAttribute('aria-pressed') === 'true') return;
        btns.forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
        place(true);
        onChange(b.getAttribute('data-v'));
      });
    });
    el.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      var i = btns.indexOf(document.activeElement);
      if (i < 0) return;
      var j = (i + (e.key === 'ArrowRight' ? 1 : btns.length - 1)) % btns.length;
      btns[j].focus();
      btns[j].click();
      e.preventDefault();
    });
    function init() {
      place(false);
      // enable the sliding transition only after the first placement
      requestAnimationFrame(function () { requestAnimationFrame(function () { el.classList.add('dm-seg-on'); }); });
    }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { place(false); });
    window.addEventListener('resize', function () { place(false); });
    return { init: init, place: place };
  }

  /* Tooltip inside a positioned host. */
  function Tip(host) {
    var t = document.createElement('div');
    t.className = 'dm-tip';
    t.setAttribute('aria-hidden', 'true');
    host.appendChild(t);
    return {
      show: function (content, x, y) {
        t.innerHTML = content;
        t.classList.add('on');
        var hw = host.clientWidth, tw = t.offsetWidth, th = t.offsetHeight;
        var left = clamp(x - tw / 2, 0, Math.max(0, hw - tw));
        var top = y - th - 10;
        if (top < -160) top = y + 30;
        t.style.transform = 'translate(' + Math.round(left) + 'px,' + Math.round(top) + 'px)';
      },
      hide: function () { t.classList.remove('on'); },
      node: t
    };
  }

  function ready(fig) { fig.classList.add('dm-ready'); }
  function closeTableView(fig) {
    var d = fig.querySelector('details.dm-tv');
    if (d) d.removeAttribute('open');
  }

  /* ================================================================ 1. headline gains (Results) */
  function initGains(R) {
    var fig = document.getElementById('dm-gains');
    if (!fig) return;
    var mount = fig.querySelector('[data-dm-mount="gains"]');
    var subEl = document.getElementById('dm-gains-sub');
    var status = fig.querySelector('[data-dm-status]');
    var lgCi = fig.querySelector('.dm-lg-ci');
    var SIZES = ['9B', '4B', '2B'], SEEDS = ['42', '43', '44'];
    var XMAX = 4.5, TICKS = [0, 1, 2, 3, 4];
    var st = { cap: 16384, mode: 's42' };
    var SUB = {
      s42: subEl ? subEl.textContent : '',
      mean: 'Mean of student seeds 42, 43 and 44 · two-level bootstrap 95% intervals · dots mark each seed'
    };
    function find(rows, sz, cap) {
      for (var i = 0; i < rows.length; i++) if (rows[i].size === sz && rows[i].eval_cap === cap) return rows[i];
      return null;
    }
    function target() {
      return SIZES.map(function (sz) {
        var a = find(R.dn_vs_label_total.rows, sz, st.cap), m = find(R.three_seeds.rows, sz, st.cap);
        if (st.mode === 'mean') {
          return { v: m.mean, lo: m.ci95[0], hi: m.ci95[1], s: SEEDS.map(function (k) { return m.per_seed[k]; }), sa: 1, a: a, m: m };
        }
        return { v: a.delta, lo: a.ci95[0], hi: a.ci95[1], s: [a.delta, a.delta, a.delta], sa: 0, a: a, m: m };
      });
    }
    function mix(f, t, k) {
      if (k >= 1) return t;
      return { v: lerp(f.v, t.v, k), lo: lerp(f.lo, t.lo, k), hi: lerp(f.hi, t.hi, k),
        s: f.s.map(function (x, i) { return lerp(x, t.s[i], k); }), sa: lerp(f.sa, t.sa, k), a: t.a, m: t.m };
    }
    var cur = target(), tgt = cur, grow = [0, 0, 0], wp = [0, 0, 0], txtO = 1, G = null, tip = null, entryH = null, tr = null;

    function capLabel() { return st.cap === 16384 ? '16K' : '8K'; }
    function rowLabel(i) {
      var d = cur[i], sz = SIZES[i];
      if (st.mode === 'mean') {
        return sz + ', ' + capLabel() + ' evaluation cap, three-seed mean: ' + signed(d.m.mean, 2) + ' pp, 95% interval ' +
          signed(d.m.ci95[0], 2) + ' to ' + signed(d.m.ci95[1], 2) + '; seeds 42, 43, 44: ' +
          SEEDS.map(function (k) { return signed(d.m.per_seed[k], 2); }).join(', ');
      }
      return sz + ', ' + capLabel() + ' evaluation cap, seed 42: ' + signed(d.a.delta, 2) + ' pp, 95% interval ' +
        signed(d.a.ci95[0], 2) + ' to ' + signed(d.a.ci95[1], 2) + '; DN-MOPD ' + d.a.dn_total.toFixed(2) + ', Label ' + d.a.label_total.toFixed(2);
    }
    function tipHtml(i) {
      var d = cur[i], sz = SIZES[i];
      if (st.mode === 'mean') {
        return '<b>' + sz + ' · ' + capLabel() + ' cap · three seeds</b>' +
          SEEDS.map(function (k) { return 'Seed ' + k + ' ' + signed(d.m.per_seed[k], 2); }).join('<br>') +
          '<br>Mean ' + signed(d.m.mean, 2) + ' pp ' + interval(d.m.ci95[0], d.m.ci95[1], 2);
      }
      return '<b>' + sz + ' · ' + capLabel() + ' cap · seed 42</b>DN-MOPD ' + d.a.dn_total.toFixed(2) + ' · Label ' +
        d.a.label_total.toFixed(2) + '<br>Δ ' + signed(d.a.delta, 2) + ' pp ' + interval(d.a.ci95[0], d.a.ci95[1], 2);
    }

    function build() {
      var keepTip = tip && tip.node;
      mount.innerHTML = '';
      var W = Math.max(280, Math.round(mount.clientWidth || 600));
      var narrow = W < 600;
      var g = { W: W, narrow: narrow, rows: [] };
      g.x0 = narrow ? 34 : 60;
      g.val = narrow ? 92 : 190;
      g.x1 = W - g.val - (narrow ? 8 : 18);
      g.top = 26; g.rowH = narrow ? 66 : 68; g.barH = 22;
      g.plotB = g.top + SIZES.length * g.rowH;
      g.H = g.plotB + 46;
      g.X = function (v) { return g.x0 + (v / XMAX) * (g.x1 - g.x0); };
      var svg = S('svg', { class: 'dm-svg', width: W, height: g.H, viewBox: '0 0 ' + W + ' ' + g.H, role: 'group',
        'aria-label': 'DN-MOPD minus Label, six-task Total gain per size' }, mount);
      var grid = S('g', { 'aria-hidden': 'true' }, svg);
      TICKS.forEach(function (t) {
        var x = Math.round(g.X(t)) + 0.5;
        S('line', { class: t === 0 ? 'dm-axis0' : 'dm-gl', x1: x, x2: x, y1: g.top - 6, y2: g.plotB }, grid);
        S('text', { class: 'dm-tick', x: x, y: g.plotB + 18, 'text-anchor': 'middle' }, grid).textContent = String(t);
      });
      S('text', { class: 'dm-note', x: g.X(0) + 6, y: g.top - 10 }, grid).textContent = '0 = Label-routed MOPD';
      S('text', { class: 'dm-axt', x: (g.x0 + g.x1) / 2, y: g.H - 6, 'text-anchor': 'middle' }, grid).textContent =
        narrow ? 'Gain over Label, Total (pp)' : 'DN-MOPD − Label, six-task Total (pp)';
      SIZES.forEach(function (sz, i) {
        var cy = g.top + i * g.rowH + g.rowH / 2;
        var row = S('g', { class: 'dm-row', tabindex: '0', role: 'img' }, svg);
        S('rect', { class: 'dm-hit', x: 0, y: cy - g.rowH / 2 + 3, width: W, height: g.rowH - 6, rx: 10 }, row);
        S('text', { class: 'dm-size', x: 0, y: cy + 6 }, row).textContent = sz;
        var n = { cy: cy, row: row };
        n.bar = S('path', { class: 'dm-bar' }, row);
        n.wr = S('line', { class: 'dm-wk-ring' }, row);
        n.cr1 = S('line', { class: 'dm-wk-ring' }, row);
        n.cr2 = S('line', { class: 'dm-wk-ring' }, row);
        n.w = S('line', { class: 'dm-wk' }, row);
        n.c1 = S('line', { class: 'dm-wk' }, row);
        n.c2 = S('line', { class: 'dm-wk' }, row);
        n.dots = SEEDS.map(function () { return S('circle', { class: 'dm-seed', r: 4.5 }, row); });
        var vx = g.x1 + (narrow ? 12 : 18);
        n.val = S('text', { class: 'dm-val', x: vx, y: narrow ? cy - 1 : cy + 6 }, row);
        n.ci = S('text', { class: 'dm-ci', x: narrow ? vx : vx + 60, y: narrow ? cy + 16 : cy + 5.5 }, row);
        function show() { if (tip) tip.show(tipHtml(i), clamp(g.X(cur[i].v), 90, W - 90), cy - g.barH / 2); }
        row.addEventListener('pointerenter', show);
        row.addEventListener('focus', show);
        row.addEventListener('pointerleave', function () { if (tip) tip.hide(); });
        row.addEventListener('blur', function () { if (tip) tip.hide(); });
        g.rows.push(n);
      });
      G = g;
      if (keepTip) mount.appendChild(keepTip); else tip = Tip(mount);
      draw();
      labels();
    }

    function draw() {
      if (!G) return;
      for (var i = 0; i < SIZES.length; i++) {
        var d = cur[i], n = G.rows[i], cy = n.cy, q = wp[i];
        var vb = d.v * grow[i];
        n.bar.setAttribute('d', barPath(G.x0, G.X(vb), cy - G.barH / 2, G.barH, 4));
        var l = G.X(d.v + (d.lo - d.v) * q), r = G.X(d.v + (d.hi - d.v) * q), ch = 6.5 * q;
        var vis = q > 0.002 ? '1' : '0';
        [n.wr, n.w].forEach(function (ln) { A(ln, { x1: l, x2: r, y1: cy, y2: cy, opacity: vis }); });
        [n.cr1, n.c1].forEach(function (ln) { A(ln, { x1: l, x2: l, y1: cy - ch, y2: cy + ch, opacity: vis }); });
        [n.cr2, n.c2].forEach(function (ln) { A(ln, { x1: r, x2: r, y1: cy - ch, y2: cy + ch, opacity: vis }); });
        // seed dots ride a lane just under the bar: they leave the mean and spread to each seed's value
        for (var k = 0; k < 3; k++) {
          var sx = d.v + (d.s[k] - d.v) * q;
          A(n.dots[k], { cx: G.X(sx), cy: cy + G.barH / 2 + 3 + 6 * d.sa, opacity: (d.sa * q).toFixed(3) });
        }
        // merge: the text is the selected series' own value: a count-up from 0 on entry (grow), and on a toggle the new
        // series' final numbers at once, fading in while the bar and whiskers ease (cur) to them
        var t = tgt[i];
        n.val.textContent = signed(t.v * grow[i], 2);
        n.val.setAttribute('opacity', txtO.toFixed(3));
        n.ci.textContent = interval(t.lo, t.hi, 2);
        n.ci.setAttribute('opacity', (q * txtO).toFixed(3));
      }
    }
    function labels() {
      // accessible names follow the selected state, not the in-between frames
      if (G) for (var i = 0; i < SIZES.length; i++) G.rows[i].row.setAttribute('aria-label', rowLabel(i));
    }

    function finishEntry() { if (entryH) entryH.finish(); else { grow = [1, 1, 1]; wp = [1, 1, 1]; } }
    function change() {
      finishEntry();
      if (entryH === null) entryH = { live: false, finish: function () {} };
      var from = cur.slice(), to = target();
      tgt = to;
      if (tr) tr.stop();
      fig.classList.toggle('dm-mean', st.mode === 'mean');
      if (subEl) subEl.textContent = SUB[st.mode];
      if (lgCi) lgCi.textContent = st.mode === 'mean' ? 'Two-level bootstrap 95% interval' : 'Paired 95% interval';
      if (status) status.textContent = 'Showing the ' + capLabel() + ' evaluation cap, ' + (st.mode === 'mean' ? 'three-seed mean.' : 'student seed 42.');
      tr = play(750, function (ms) {
        txtO = ms >= 750 ? 1 : 0.2 + 0.8 * ease.out(Math.min(1, ms / 420));
        cur = from.map(function (f, i) { return mix(f, to[i], ms >= 750 ? 1 : ease.inOut(ms / 750)); });
        draw();
      });
      labels();
      if (tip) tip.hide();
    }
    var segCap = Seg(fig.querySelector('[data-dm-seg="cap"]'), function (v) { st.cap = +v; change(); });
    var segSeed = Seg(fig.querySelector('[data-dm-seg="seed"]'), function (v) { st.mode = v; change(); });

    ready(fig);
    build();
    segCap.init(); segSeed.init();
    onWidth(mount, build);
    whenSeen(mount, function () {
      if (entryH) return null;
      entryH = play(1450, function (ms) {
        for (var i = 0; i < 3; i++) {
          grow[i] = ease.out(phase(ms, i * 120, 800));
          wp[i] = ease.out(phase(ms, 720 + i * 120, 480));
        }
        draw();
      });
      return entryH;
    });
  }

  /* ================================================================ 2. multiplier trajectories */
  function initMult(M) {
    var fig = document.getElementById('dm-mult');
    if (!fig) return;
    var mount = fig.querySelector('[data-dm-mount="mult"]');
    var ro = {};
    ['upd', 'of', 'phase', 'math', 'code', 'if', 'flag'].forEach(function (k) { ro[k] = fig.querySelector('[data-ro="' + k + '"]'); });
    var KEY = { '9B': '9b', '4B': '4b', '2B': '2b' };
    var DOMS = ['math', 'code', 'if'];
    var NAME = { math: 'math', code: 'code', if: 'IF' };
    var FLOOR = 0.25, CEIL = 4, MAIN = 80, XMAX = 160;
    var LMIN = Math.log(0.08) / Math.LN2, LMAX = Math.log(4.6) / Math.LN2;
    var YT = [0.125, 0.25, 0.5, 1, 2, 4], XT = [1, 40, 80, 120, 160];
    var SER = ['math.raw', 'code.raw', 'if.raw', 'if.pre', 'math.sm', 'code.sm', 'if.sm'];

    function trailing(a, k) {
      var out = [], sum = 0;
      for (var i = 0; i < a.length; i++) {
        sum += a[i];
        if (i >= k) sum -= a[i - k];
        out.push(sum / Math.min(i + 1, k));
      }
      return out;
    }
    var data = {};
    Object.keys(KEY).forEach(function (sz) {
      var z = M.sizes[KEY[sz]], o = { n: z.update.length, w: {}, v: {}, floor: [], z: z };
      DOMS.forEach(function (d) {
        o.w[d] = z[d + '_w'];
        o.v[d + '.raw'] = z[d + '_w'];
        o.v[d + '.sm'] = trailing(z[d + '_w'], 5);
      });
      o.v['if.pre'] = z.if_w_raw;
      o.floor = z.if_w.map(function (x) { return Math.abs(x - FLOOR) < 1e-9; });
      data[sz] = o;
    });
    var st = { size: '9B', u: 0 };
    function exact(sz) {
      var o = data[sz], xs = [], v = {};
      for (var i = 0; i < o.n; i++) xs.push(i + 1);
      SER.forEach(function (k) { v[k] = o.v[k].slice(); });
      return { xs: xs, v: v };
    }
    var disp = exact(st.size);
    var G = null, entryH = null, entered = false, morph = null, clipW = 0, bandO = 0, endO = 0, m80O = 0;

    function lg(w) { return Math.log(w) / Math.LN2; }
    function build() {
      mount.innerHTML = '';
      var W = Math.max(280, Math.round(mount.clientWidth || 600)), narrow = W < 600;
      var g = { W: W, narrow: narrow };
      g.l = narrow ? 40 : 54; g.r = narrow ? 44 : 70; g.t = 34;
      g.H = narrow ? 318 : 392;
      g.px0 = g.l; g.px1 = W - g.r; g.py0 = g.t; g.py1 = g.H - 70;
      g.sy = g.py1 + 12; g.sh = 9;
      g.X = function (u) { return g.px0 + (u - 1) / (XMAX - 1) * (g.px1 - g.px0); };
      g.Y = function (w) { return g.py1 - (lg(w) - LMIN) / (LMAX - LMIN) * (g.py1 - g.py0); };
      var svg = S('svg', { class: 'dm-svg', width: W, height: g.H, viewBox: '0 0 ' + W + ' ' + g.H, 'aria-hidden': 'true' }, mount);
      g.svg = svg;
      var defs = S('defs', null, svg);
      var cid = 'dm-clip-' + Math.random().toString(36).slice(2, 8);
      var cp = S('clipPath', { id: cid }, defs);
      g.clip = S('rect', { x: g.px0 - 4, y: 0, width: 0, height: g.H }, cp);
      // clip band and continuation
      g.band = S('g', null, svg);
      S('rect', { class: 'dm-band', x: g.px0, y: g.Y(CEIL), width: g.px1 - g.px0, height: g.Y(FLOOR) - g.Y(CEIL) }, g.band);
      g.cont = S('rect', { class: 'dm-cont', x: g.X(MAIN), y: g.py0, height: g.py1 - g.py0, width: 0 }, svg);
      var ax = S('g', null, svg);
      YT.forEach(function (w) {
        var y = Math.round(g.Y(w)) + 0.5;
        var cls = (w === FLOOR || w === CEIL) ? 'dm-bandline' : (w === 1 ? 'dm-one' : 'dm-gl');
        S('line', { class: cls, x1: g.px0, x2: g.px1, y1: y, y2: y }, w === FLOOR || w === CEIL ? g.band : ax);
        S('text', { class: 'dm-tick', x: g.px0 - 8, y: y + 4, 'text-anchor': 'end' }, ax).textContent = String(w);
      });
      S('text', { class: 'dm-note', x: g.px1 - 6, y: g.Y(CEIL) + 15, 'text-anchor': 'end' }, g.band).textContent = 'clip band [0.25, 4]';
      S('line', { class: 'dm-gl', x1: g.px0, x2: g.px1, y1: g.py1 + 0.5, y2: g.py1 + 0.5 }, ax);
      g.xt = XT.map(function (u) {
        var t = S('text', { class: 'dm-tick dm-xt', x: g.X(u), y: g.sy + g.sh + 19, 'text-anchor': 'middle' }, ax);
        t.textContent = String(u);
        return { x: g.X(u), hw: String(u).length * 3.6 + 3, node: t };
      });
      S('text', { class: 'dm-axt', x: (g.px0 + g.px1) / 2, y: g.H - 6, 'text-anchor': 'middle' }, ax).textContent = 'Student update';
      S('text', { class: 'dm-note', x: g.px0 - 8, y: g.sy + g.sh - 0.5, 'text-anchor': 'end' }, ax).textContent = narrow ? 'floor' : 'IF floor';
      // series (revealed by the clip)
      var body = S('g', { 'clip-path': 'url(#' + cid + ')' }, svg);
      g.floors = {};
      Object.keys(data).forEach(function (sz) {
        var o = data[sz], fg = S('g', { class: 'dm-floor', opacity: sz === st.size ? 1 : 0 }, body);
        var step = (g.px1 - g.px0) / (XMAX - 1);
        for (var i = 0; i < o.n; i++) if (o.floor[i]) S('rect', { x: g.X(i + 1) - step / 2, y: g.sy, width: step + 0.35, height: g.sh }, fg);
        g.floors[sz] = fg;
      });
      g.paths = {};
      SER.forEach(function (k) {
        var p = k.split('.');
        g.paths[k] = S('path', { class: 'dm-ln ' + p[1] + ' c-' + p[0] }, body);
      });
      // update-80 marker
      var mx = Math.round(g.X(MAIN)) + 0.5;
      g.m80 = S('g', null, svg);
      S('line', { class: 'dm-m80', x1: mx, x2: mx, y1: g.py0 - 12, y2: g.sy + g.sh }, g.m80);
      g.m80l = S('text', { class: 'dm-m80-t', x: mx - 7, y: g.py0 - 16, 'text-anchor': 'end' }, g.m80);
      g.m80l.textContent = 'main runs end';
      g.m80r = S('text', { class: 'dm-m80-t', x: mx + 7, y: g.py0 - 16 }, g.m80);
      g.m80r.textContent = 'continuation →';
      // end labels
      g.ends = {};
      DOMS.forEach(function (d) {
        var eg = S('g', null, svg);
        var key = S('line', { class: 'dm-end-key c-' + d, x1: 0, x2: 8, y1: 0, y2: 0 }, eg);
        var t = S('text', { class: 'dm-end', x: 12, y: 4.5 }, eg);
        t.textContent = NAME[d];
        g.ends[d] = eg;
      });
      // playhead
      g.ph = S('g', null, svg);
      g.phLine = S('line', { class: 'dm-ph-line', y1: g.py0 - 4, y2: g.sy + g.sh + 4 }, g.ph);
      g.phDots = {};
      DOMS.forEach(function (d) { g.phDots[d] = S('circle', { class: 'dm-ph-dot c-' + d, r: 4.5 }, g.ph); });
      // the scrubber badge sits on the x axis, where the update number is read
      g.pill = S('rect', { class: 'dm-pill', y: g.sy + g.sh + 4, height: 20, rx: 10, width: 34 }, g.ph);
      g.pillT = S('text', { class: 'dm-pill-t', y: g.sy + g.sh + 18.2, 'text-anchor': 'middle' }, g.ph);
      // pointer overlay
      g.ov = S('rect', { class: 'dm-overlay', x: g.px0 - 12, y: 0, width: g.px1 - g.px0 + 24, height: g.H }, svg);
      G = g;
      wirePointer();
      draw();
    }

    function pathD(k) {
      var xs = disp.xs, ys = disp.v[k], s = '';
      for (var i = 0; i < xs.length; i++) s += (i ? 'L' : 'M') + G.X(xs[i]).toFixed(1) + ',' + G.Y(ys[i]).toFixed(1);
      return s;
    }
    function draw() {
      if (!G) return;
      SER.forEach(function (k) { G.paths[k].setAttribute('d', pathD(k)); });
      var lastX = disp.xs[disp.xs.length - 1];
      G.cont.setAttribute('width', Math.max(0, G.X(Math.max(MAIN, lastX)) - G.X(MAIN)));
      G.clip.setAttribute('width', clipW === null ? G.W : Math.max(0, clipW));
      G.band.setAttribute('opacity', bandO.toFixed(3));
      G.m80.setAttribute('opacity', m80O.toFixed(3));
      // end labels at the last smoothed value, pushed apart when close
      var L = disp.xs.length - 1, ex = G.X(lastX) + 8, items = DOMS.map(function (d) { return { d: d, y: G.Y(disp.v[d + '.sm'][L]) }; });
      items.sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < 15) items[i].y = items[i - 1].y + 15;
      items.forEach(function (it) {
        G.ends[it.d].setAttribute('transform', 'translate(' + Math.min(ex, G.W - 40).toFixed(1) + ',' + it.y.toFixed(1) + ')');
        G.ends[it.d].setAttribute('opacity', endO.toFixed(3));
      });
      drawPlayhead();
    }
    function drawPlayhead() {
      if (!G) return;
      var n = disp.xs.length, i = clamp(st.u, 0, n - 1);
      var x = G.X(disp.xs[i]);
      A(G.phLine, { x1: x, x2: x });
      DOMS.forEach(function (d) { A(G.phDots[d], { cx: x, cy: G.Y(disp.v[d + '.raw'][i]) }); });
      var label = String(st.u + 1), pw = 14 + label.length * 7.2;
      var pl = clamp(x - pw / 2, G.px0 - 14, G.px1 + 14 - pw);
      A(G.pill, { x: pl, width: pw });
      A(G.pillT, { x: pl + pw / 2 });
      G.pillT.textContent = label;
      // axis numbers step aside while the badge passes over them
      G.xt.forEach(function (t) {
        var hide = t.x + t.hw > pl - 3 && t.x - t.hw < pl + pw + 3;
        t.node.setAttribute('opacity', hide ? '0' : '1');
      });
    }
    function readout() {
      var o = data[st.size], i = st.u, z = o.z;
      ro.upd.textContent = String(i + 1);
      ro.of.textContent = 'of ' + o.n;
      ro.phase.textContent = i + 1 <= MAIN ? 'main run' : 'continuation';
      ro.phase.classList.toggle('is-main', i + 1 <= MAIN);
      ro.math.textContent = z.math_w[i].toFixed(2);
      ro.code.textContent = z.code_w[i].toFixed(2);
      ro['if'].textContent = z.if_w[i].toFixed(2);
      var fl = o.floor[i];
      ro.flag.textContent = fl ? 'at the floor (before clipping ' + z.if_w_raw[i].toFixed(2) + ')' : 'above the floor';
      ro.flag.classList.toggle('is-floor', fl);
      A(mount, { 'aria-valuemax': String(o.n), 'aria-valuenow': String(i + 1),
        'aria-valuetext': st.size + ', update ' + (i + 1) + ' of ' + o.n + (i + 1 <= MAIN ? ' (main run)' : ' (continuation)') +
          ': math ' + z.math_w[i].toFixed(2) + ', code ' + z.code_w[i].toFixed(2) + ', IF ' + z.if_w[i].toFixed(2) +
          (fl ? ', at the 0.25 floor' : ', above the floor') });
    }
    function setU(u) {
      u = clamp(u, 0, data[st.size].n - 1);
      if (u === st.u) return;
      st.u = u;
      drawPlayhead();
      readout();
    }
    function finishEntry() {
      if (entryH) entryH.finish();
      else { entryH = { live: false, finish: function () {} }; clipW = null; bandO = 1; endO = 1; m80O = 1; st.u = data[st.size].n - 1; entered = true; draw(); readout(); }
    }
    var dragging = false;
    function uFromEvent(e) {
      var r = G.svg.getBoundingClientRect();
      var x = (e.clientX - r.left) * (G.W / r.width);
      return Math.round((x - G.px0) / (G.px1 - G.px0) * (XMAX - 1));
    }
    function wirePointer() {
      var ov = G.ov;
      ov.addEventListener('pointerdown', function (e) {
        finishEntry();
        dragging = true;
        try { ov.setPointerCapture(e.pointerId); } catch (err) {}
        setU(uFromEvent(e));
        if (document.activeElement !== mount) mount.focus({ preventScroll: true });
      });
      ov.addEventListener('pointermove', function (e) {
        if (dragging || (e.pointerType === 'mouse' && entered)) setU(uFromEvent(e));
      });
      function up() { dragging = false; }
      ov.addEventListener('pointerup', up);
      ov.addEventListener('pointercancel', up);
      ov.addEventListener('lostpointercapture', up);
    }
    mount.addEventListener('keydown', function (e) {
      var n = data[st.size].n, d = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10 }[e.key];
      if (d === undefined && e.key !== 'Home' && e.key !== 'End') return;
      e.preventDefault();
      finishEntry();
      if (e.key === 'Home') setU(0);
      else if (e.key === 'End') setU(n - 1);
      else setU(st.u + d * (e.shiftKey ? 10 : 1));
    });

    function setSize(sz) {
      finishEntry();
      if (morph) morph.stop();
      var from = disp, to = exact(sz), a = from.xs.length, b = to.xs.length, L = Math.max(a, b);
      function pick(arr, i, n) { return arr[Math.min(i, n - 1)]; }
      var fx0 = [], tx0 = [], fv = {}, tv = {};
      for (var i = 0; i < L; i++) { fx0.push(pick(from.xs, i, a)); tx0.push(pick(to.xs, i, b)); }
      SER.forEach(function (k) {
        fv[k] = []; tv[k] = [];
        for (var j = 0; j < L; j++) { fv[k].push(lg(pick(from.v[k], j, a))); tv[k].push(lg(pick(to.v[k], j, b))); }
      });
      Object.keys(G.floors).forEach(function (k) { G.floors[k].setAttribute('opacity', k === sz ? 1 : 0); });
      st.size = sz;
      st.u = clamp(st.u, 0, b - 1);
      readout();
      morph = play(800, function (ms) {
        if (ms >= 800) { disp = to; draw(); return; }
        var t = ease.inOut(ms / 800), xs = [], v = {};
        for (var i = 0; i < L; i++) xs.push(lerp(fx0[i], tx0[i], t));
        SER.forEach(function (k) { v[k] = fv[k].map(function (x, i) { return Math.pow(2, lerp(x, tv[k][i], t)); }); });
        disp = { xs: xs, v: v };
        draw();
      });
    }
    var seg = Seg(fig.querySelector('[data-dm-seg="size"]'), setSize);

    ready(fig);
    closeTableView(fig);
    build();
    seg.init();
    readout();
    onWidth(mount, build);
    whenSeen(mount, function () {
      if (entryH) return null;
      var n = data[st.size].n;
      entryH = play(2300, function (ms) {
        bandO = ease.out(phase(ms, 0, 500));
        var p = ease.sine(phase(ms, 200, 1900));
        var uf = 1 + p * (n - 1);
        clipW = ms >= 2300 ? null : G.X(uf) - G.px0 + 8;
        m80O = G.X(uf) >= G.X(MAIN) - 2 ? Math.min(1, m80O + 0.12) : 0;
        if (ms >= 2300) m80O = 1;
        endO = ease.out(phase(ms, 1950, 350));
        var u = Math.round(p * (n - 1));
        draw();
        if (u !== st.u || ms === 0) { st.u = u; drawPlayhead(); readout(); }
      }, function () { entered = true; });
      return entryH;
    });
  }

  /* ================================================================ 3. lengths (Table 4) */
  function initLengths(R) {
    var fig = document.getElementById('dm-len');
    if (!fig) return;
    var rows = R.table4_lengths.rows;
    var SZ = [['9B', 'Qwen3.5-9B'], ['4B', 'Qwen3.5-4B'], ['2B', 'Qwen3.5-2B']];
    function get(group, method) {
      for (var i = 0; i < rows.length; i++) if (rows[i].group === group && rows[i].cells[0] === method) return rows[i].cells;
      return null;
    }
    var items = SZ.map(function (s) {
      var l = get(s[1], 'Label-routed'), d = get(s[1], 'DN-MOPD');
      return { size: s[0], group: s[1], lt: l[1], dt: d[1], ltok: l[2], dtok: d[2], lcap: l[5], dcap: d[5] };
    });
    var panels = [
      { mount: fig.querySelector('[data-dm-mount="len-tok"]'), max: 12000, ticks: [0, 4000, 8000, 12000],
        tick: function (t) { return t === 0 ? '0' : (t / 1000) + 'K'; }, lab: 'ltok', dab: 'dtok',
        fmt: function (v) { return thousands(v); }, axis: 'tokens per response' },
      { mount: fig.querySelector('[data-dm-mount="len-cap"]'), max: 25, ticks: [0, 10, 20],
        tick: function (t) { return t + '%'; }, lab: 'lcap', dab: 'dcap',
        fmt: function (v) { return v.toFixed(1) + '%'; }, axis: 'share of responses at the cap' }
    ];
    // merge (required fix): Label and DN-MOPD each grow from 0 to their own value; DN-MOPD starts a beat later, and the
    // dashed outline of Label's value appears on the DN-MOPD row only after DN-MOPD's bar has settled
    var prog = { lab: [0, 0, 0], dn: [0, 0, 0], ghost: [0, 0, 0], dash: 0 };
    var entryH = null;

    function build(P) {
      var keepTip = P.tip && P.tip.node;
      P.mount.innerHTML = '';
      var W = Math.max(260, Math.round(P.mount.clientWidth || 480)), narrow = W < 420;
      var g = { W: W, rows: [] };
      g.x0 = narrow ? 34 : 40; g.val = narrow ? 78 : 104; g.x1 = W - g.val;
      g.bh = 16; g.gap = 4; g.grpH = 2 * g.bh + g.gap + 26; g.top = 6;
      g.plotB = g.top + items.length * g.grpH - 14;
      g.H = g.plotB + 44;
      g.X = function (v) { return g.x0 + v / P.max * (g.x1 - g.x0); };
      var svg = S('svg', { class: 'dm-svg', width: W, height: g.H, viewBox: '0 0 ' + W + ' ' + g.H, 'aria-hidden': 'true' }, P.mount);
      P.ticks.forEach(function (t) {
        var x = Math.round(g.X(t)) + 0.5;
        S('line', { class: t === 0 ? 'dm-axis0' : 'dm-gl', x1: x, x2: x, y1: g.top - 4, y2: g.plotB }, svg);
        S('text', { class: 'dm-tick', x: x, y: g.plotB + 17, 'text-anchor': 'middle' }, svg).textContent = P.tick(t);
      });
      S('text', { class: 'dm-axt', x: (g.x0 + g.x1) / 2, y: g.H - 5, 'text-anchor': 'middle' }, svg).textContent = P.axis;
      items.forEach(function (it, i) {
        var y = g.top + i * g.grpH;
        var row = S('g', { class: 'dm-row' }, svg);
        S('rect', { class: 'dm-hit', x: -2, y: y - 5, width: W + 4, height: 2 * g.bh + g.gap + 10, rx: 8 }, row);
        S('text', { class: 'dm-size', x: 0, y: y + g.bh + g.gap / 2 + 5.5 }, row).textContent = it.size;
        var n = { y: y };
        n.lbar = S('path', { class: 'dm-bar-label' }, row);
        n.ghost = S('rect', { class: 'dm-lghost', y: y + g.bh + g.gap + 0.6, height: g.bh - 1.2, rx: 3 }, row);
        n.dbar = S('path', { class: 'dm-bar' }, row);
        n.lval = S('text', { class: 'dm-lv', y: y + g.bh - 3.5 }, row);
        n.dval = S('text', { class: 'dm-lv', y: y + 2 * g.bh + g.gap - 3.5 }, row);
        function show() {
          if (!P.tip) return;
          P.tip.show('<b>' + it.group + '</b>Label-routed ' + it.ltok + ' math tokens, ' + it.lcap + '% at cap' +
            '<br>DN-MOPD ' + it.dtok + ' math tokens, ' + it.dcap + '% at cap<br>Total ' + it.lt + ' → ' + it.dt, W / 2, y - 4);
        }
        row.addEventListener('pointerenter', show);
        row.addEventListener('pointerleave', function () { if (P.tip) P.tip.hide(); });
        g.rows.push(n);
      });
      P.G = g;
      if (keepTip) P.mount.appendChild(keepTip); else P.tip = Tip(P.mount);
      draw(P);
    }
    function setVal(t, main, sub) {
      t.textContent = main;
      if (sub) { var ts = S('tspan', { class: 'dm-lv-sub', dx: '6' }, t); ts.textContent = sub; }
    }
    function draw(P) {
      var g = P.G;
      if (!g) return;
      items.forEach(function (it, i) {
        var n = g.rows[i], L = parseNum(it[P.lab]), D = parseNum(it[P.dab]);
        var gl = prog.lab[i], gd = prog.dn[i];
        var lv = L * gl, dv = D * gd;
        n.lbar.setAttribute('d', barPath(g.x0, g.X(lv), n.y, g.bh, 4));
        n.dbar.setAttribute('d', barPath(g.x0, g.X(dv), n.y + g.bh + g.gap, g.bh, 4));
        A(n.ghost, { x: g.x0 + 0.6, width: Math.max(0, g.X(L) - g.x0 - 1.2), opacity: prog.ghost[i].toFixed(3), 'stroke-dashoffset': prog.dash.toFixed(2) });
        var vx = g.X(Math.max(lv, dv, 0)) + 9;
        n.lval.setAttribute('x', vx);
        n.dval.setAttribute('x', vx);
        var lt = gl >= 1 ? (P.lab === 'lcap' ? it.lcap + '%' : it.ltok) : P.fmt(P.lab === 'lcap' ? fx(lv) : lv);
        var dt = gd >= 1 ? (P.dab === 'dcap' ? it.dcap + '%' : it.dtok) : P.fmt(P.dab === 'dcap' ? fx(dv) : dv);
        setVal(n.lval, lt, i === 0 ? 'Label' : '');
        setVal(n.dval, dt, i === 0 ? 'DN-MOPD' : '');
        n.lval.setAttribute('opacity', gl > 0 ? 1 : 0);
        n.dval.setAttribute('opacity', gd > 0 ? 1 : 0);
      });
    }
    ready(fig);
    closeTableView(fig);
    panels.forEach(function (P) { build(P); onWidth(P.mount, function () { build(P); }); });
    whenSeen(fig.querySelector('.dm-len-grid'), function () {
      if (entryH) return null;
      entryH = play(2300, function (ms) {
        for (var i = 0; i < 3; i++) {
          prog.lab[i] = ease.out(phase(ms, i * 140, 850));
          prog.dn[i] = ease.out(phase(ms, 380 + i * 140, 850));
          prog.ghost[i] = phase(ms, 1250 + i * 140, 250);
        }
        prog.dash = ms >= 2300 ? 0 : -ms / 60;
        panels.forEach(draw);
      });
      return entryH;
    });
  }

  /* ================================================================ 4. Tables 1-3: staggered rows, DN-MOPD highlight */
  function initTables() {
    if (reduced() || !('IntersectionObserver' in window)) return;
    ['table1', 'table2', 'table3'].forEach(function (id) {
      var fig = document.getElementById(id);
      if (!fig) return;
      var trs = fig.querySelectorAll('tbody tr');
      for (var i = 0; i < trs.length; i++) {
        trs[i].style.setProperty('--dm-i', i);
        if (trs[i].classList.contains('dn-row')) {
          var cells = trs[i].children;
          for (var c = 0; c < cells.length; c++) cells[c].style.setProperty('--dm-c', c);
        }
      }
      var r = fig.getBoundingClientRect();
      if (r.top < (window.innerHeight || 800) * 0.9) return; // already on screen: leave it static
      fig.classList.add('dm-stagger');
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) { fig.classList.add('dm-rows-in'); io.disconnect(); }
        });
      }, { rootMargin: '0px 0px -12% 0px', threshold: 0 });
      io.observe(fig);
    });
    window.addEventListener('beforeprint', function () {
      var s = document.querySelectorAll('.dm-stagger');
      for (var i = 0; i < s.length; i++) s[i].classList.add('dm-rows-in');
    });
  }

  /* ================================================================ boot */
  function getJSON(url) {
    return fetch(url, { credentials: 'same-origin' }).then(function (r) {
      if (!r.ok) throw new Error(url + ' ' + r.status);
      return r.json();
    });
  }
  function safe(fn) {
    return function () {
      try { fn.apply(null, arguments); } catch (e) { if (window.console) console.warn('datamotion:', e); }
    };
  }
  initTables();
  if (!window.fetch) return;
  getJSON('static/results.json').then(function (R) {
    safe(initGains)(R);
    safe(initLengths)(R);
  }, function (e) { if (window.console) console.warn('datamotion: results unavailable', e); });
  getJSON('static/multipliers.json').then(safe(initMult), function (e) { if (window.console) console.warn('datamotion: multipliers unavailable', e); });
})();
