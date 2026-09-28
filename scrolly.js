/* DN-MOPD project page, p5b_scrolly: scrollytelling for the Method section. Loaded after app.js.
   Progressive enhancement only: without this file the five step cards and both diagrams are plain, readable content.
   - The step whose card has crossed the activation line sets data-step on the SVG (CSS highlights that stage, fades the rest).
   - On narrow screens the diagram zooms into the active stage (data-vb) so its labels stay legible.
   - Looping motion runs only while the section is on screen, the tab is visible and reduced motion is not requested (.sy-play).
   - The 9B / 4B / 2B switch swaps in the first-batch numbers of each DN-MOPD run (seed 42) from #sy-data.
   Merge (p5_all): the zoom is a camera transform on one wrapper group, eased by a CSS transition (transform only),
   instead of a per-frame viewBox tween; the on-screen / hidden-tab test is the page's shared gate (DNMOPD.gate, polish.js). */
(function () {
  'use strict';
  var root = document.getElementById('sy');
  var svg = document.getElementById('sy-svg');
  if (!root || !svg) return;
  var steps = Array.prototype.slice.call(root.querySelectorAll('.sy-step'));
  if (!steps.length) return;

  var docEl = document.documentElement;
  var mm = function (q) { return window.matchMedia ? window.matchMedia(q) : { matches: false }; };
  var mqReduce = mm('(prefers-reduced-motion: reduce)');
  var mqNarrow = mm('(max-width: 980px)');
  function onChange(mq, fn) {
    if (mq.addEventListener) mq.addEventListener('change', fn);
    else if (mq.addListener) mq.addListener(fn);
  }
  docEl.classList.add('sy-js');

  var label = document.getElementById('sy-label');
  var note = document.getElementById('sy-note');
  var prog = root.querySelectorAll('.sy-prog i');
  var SIZE = { '9b': '9B', '4b': '4B', '2b': '2B' };
  var size = '4b';

  /* ---- camera zoom: the viewBox stays fixed; one wrapper group is translated and scaled so the target window
     fills the element exactly as a viewBox of that window would (xMidYMid meet). scrolly.css eases the transform. */
  var FULL = svg.getAttribute('viewBox').split(/[\s,]+/).map(Number);
  var cam = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  cam.setAttribute('class', 'sy-cam');
  Array.prototype.slice.call(svg.childNodes).forEach(function (n) {
    if (n.nodeType === 1 && /^(title|desc|defs)$/i.test(n.nodeName)) return;
    cam.appendChild(n);
  });
  svg.appendChild(cam);
  // the first placement is instant; later step changes ease (the transition is keyed to .sy-cam-on)
  requestAnimationFrame(function () { requestAnimationFrame(function () { cam.classList.add('sy-cam-on'); }); });
  var camTarget = FULL;
  function setVB(target) {
    camTarget = target;
    var r = svg.getBoundingClientRect(), W = r.width, H = r.height;
    if (!W || !H) return;
    var s0 = Math.min(W / FULL[2], H / FULL[3]);
    var ox = (W - FULL[2] * s0) / 2 - FULL[0] * s0, oy = (H - FULL[3] * s0) / 2 - FULL[1] * s0;
    var s1 = Math.min(W / target[2], H / target[3]);
    var tx = (W - target[2] * s1) / 2 - target[0] * s1, ty = (H - target[3] * s1) / 2 - target[1] * s1;
    var k = s1 / s0, cx = (tx - ox) / s0, cy = (ty - oy) / s0;
    cam.style.transform = Math.abs(k - 1) < 1e-4 && Math.abs(cx) < 0.01 && Math.abs(cy) < 0.01 ? ''
      : 'translate(' + cx.toFixed(3) + 'px,' + cy.toFixed(3) + 'px) scale(' + k.toFixed(5) + ')';
  }
  // the camera depends on the element's box (40vh-high strip on narrow screens): refit when it changes size
  if ('ResizeObserver' in window) new ResizeObserver(function () { setVB(camTarget); }).observe(svg);
  else window.addEventListener('resize', function () { setVB(camTarget); });
  function frameFor(step) {
    var vb = step.getAttribute('data-vb');
    return mqNarrow.matches && vb ? vb.split(/[\s,]+/).map(Number) : FULL;
  }

  /* ---- step activation */
  var active = null;
  function noteFor(step) {
    return (step.getAttribute('data-note') || '').replace('@SIZE@', SIZE[size]);
  }
  function activate(step, force) {
    if (step === active && !force) return;
    active = step;
    var n = +step.getAttribute('data-step');
    svg.setAttribute('data-step', String(n));
    setVB(frameFor(step));
    for (var i = 0; i < steps.length; i++) steps[i].classList.toggle('is-active', steps[i] === step);
    for (var j = 0; j < prog.length; j++) {
      prog[j].classList.toggle('on', j === n - 1);
      prog[j].classList.toggle('done', j < n - 1);
    }
    if (label) label.textContent = 'Step ' + n + ' of ' + steps.length + ' · ' + (step.getAttribute('data-title') || '');
    if (note) note.textContent = noteFor(step);
  }
  function pick() {
    var line = (window.innerHeight || 800) * (mqNarrow.matches ? 0.72 : 0.6);
    var chosen = steps[0];
    for (var i = 0; i < steps.length; i++) {
      var card = steps[i].querySelector('.sy-card') || steps[i];
      if (card.getBoundingClientRect().top < line) chosen = steps[i];
    }
    activate(chosen);
  }
  var queued = false;
  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; pick(); });
  }
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  onChange(mqNarrow, function () { if (active) setVB(frameFor(active)); schedule(); });
  activate(steps[0], true);
  pick();

  /* ---- motion gate: on screen and visible tab (the page's shared gate), and motion welcome */
  var live = true;
  function gate() { root.classList.toggle('sy-play', live && !mqReduce.matches); }
  if (window.DNMOPD && window.DNMOPD.gate) window.DNMOPD.gate(root, function (on) { live = on; gate(); });
  onChange(mqReduce, gate);
  gate();
  window.addEventListener('beforeprint', function () { svg.removeAttribute('data-step'); cam.style.transform = ''; });
  window.addEventListener('afterprint', function () { if (active) activate(active, true); });

  /* ---- first-batch numbers per model size (values straight from multipliers.json, index 0, seed 42) */
  var dataEl = document.getElementById('sy-data');
  var sw = document.getElementById('sy-size');
  var DATA = null;
  try { DATA = JSON.parse(dataEl.textContent); } catch (e) { DATA = null; }
  if (!DATA || !sw) return;
  var DOMS = ['math', 'code', 'if'];
  var k = DATA._k || 1;
  function txt(id, s) { var el = document.getElementById(id); if (el) el.textContent = s; }
  function fx(v, d) { return Number(v).toFixed(d); }
  function setSize(sz) {
    var d = DATA[sz];
    if (!d) return;
    size = sz;
    DOMS.forEach(function (m) {
      var sig = d[m + '_sigma'], w = d[m + '_w'], raw = d[m + '_w_raw'];
      var clipped = Math.abs(raw - w) > 1e-9;
      txt('sy-sig-' + m, fx(sig, 3));
      txt('sy-w-' + m, '×' + fx(w, 2));
      var band = document.getElementById('sy-band-' + m);
      if (band) {
        band.style.transform = 'scaleX(' + (sig * k).toFixed(4) + ')';
        var norm = band.querySelector('.sy-norm');
        if (norm) norm.style.setProperty('--w', String(w));
      }
      var bars = document.getElementById('sy-bars-' + m);
      if (bars) bars.style.setProperty('--w', String(w));
      var chip = document.getElementById('sy-wc-' + m);
      if (chip) chip.classList.toggle('is-clipped', clipped);
    });
    txt('sy-sig-all', fx(d.sigma_all, 3));
    var bandAll = document.getElementById('sy-band-all');
    if (bandAll) bandAll.style.transform = 'scaleX(' + (d.sigma_all * k).toFixed(4) + ')';
    var guides = document.getElementById('sy-guides');
    if (guides) guides.style.transform = 'scaleX(' + (d.sigma_all * k).toFixed(4) + ')';
    txt('sy-src', 'first batch · ' + SIZE[sz] + ' DN-MOPD run · seed 42');
    txt('sy-tcap', 'First batch, ' + SIZE[sz] + ' DN-MOPD run, seed 42');
    var cells = root.querySelectorAll('.sy-tbl td[data-k]');
    for (var i = 0; i < cells.length; i++) {
      var key = cells[i].getAttribute('data-k');
      var v = d[key];
      cells[i].textContent = /sigma/.test(key) ? fx(v, 3) : fx(v, 2);
      if (/_w$/.test(key)) {
        var m = key.slice(0, -2);
        cells[i].classList.toggle('is-clipped', Math.abs(d[m + '_w_raw'] - d[key]) > 1e-9);
      }
    }
    var binds = root.querySelectorAll('.sy-bind[data-k]');
    for (var b = 0; b < binds.length; b++) binds[b].textContent = '×' + fx(d[binds[b].getAttribute('data-k')], 2);
    var btns = sw.querySelectorAll('button[data-size]');
    for (var j = 0; j < btns.length; j++) btns[j].setAttribute('aria-pressed', String(btns[j].getAttribute('data-size') === sz));
    if (active && note) note.textContent = noteFor(active);
  }
  sw.hidden = false;
  sw.addEventListener('click', function (e) {
    var btn = e.target && e.target.closest ? e.target.closest('button[data-size]') : null;
    if (btn) setSize(btn.getAttribute('data-size'));
  });
})();
