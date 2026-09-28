/* DN-MOPD project page, p5d_polish layer. Runs after app.js; progressive enhancement only.
   - reading-progress bar under the nav, active chapter (aria-current) in the rail, top nav and menu
   - theme toggle label + cross-tab sync (app.js stores the explicit choice; the head script applies it before paint)
   - staggered reveals for siblings that share a row; hover transitions handed back after the reveal
   - figure lightbox on <dialog> (Esc closes, focus returns to the trigger), zoom toggle
   - copy-BibTeX feedback (check icon + glow), Table 3 as cards on phones
   - performance: pause looping motion off-screen or in a hidden tab; content-visibility only until the page has
     loaded (and never during an in-page jump)
   Merge (p5_all): this file also owns the page's single pause gate (DNMOPD.gate), used for the hero (P5 aurora, p5a
   schematic, scroll cue; replaces p5a's hero.js) and by scrolly.js for p5b's looping diagram. */
(function () {
  'use strict';
  var doc = document.documentElement;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var SECTIONS = ['overview', 'method', 'results', 'analysis', 'limitations', 'cite'];
  var raf = window.requestAnimationFrame || function (f) { return setTimeout(f, 16); };
  doc.classList.add('pl-on');
  // the rail's dots settle in after the title block has started
  setTimeout(function () { doc.classList.add('rail-in'); }, 320);

  /* ---------------------------------------------------------------- exact layout before any in-page jump */
  var afterCvOff = [];
  function cvOff() {
    if (doc.classList.contains('cv-off')) return;
    doc.classList.add('cv-off');
    afterCvOff.forEach(function (f) { f(); });
  }
  // capture phase: runs before app.js's click handler calls scrollIntoView
  document.addEventListener('click', function (e) {
    var a = e.target && e.target.closest ? e.target.closest('a[href^="#"]') : null;
    if (a && a.getAttribute('href').length > 1) cvOff();
  }, true);
  window.addEventListener('hashchange', cvOff);
  window.addEventListener('beforeprint', cvOff);
  // content-visibility only speeds up the first render; once the page has loaded, lay everything out in idle time so
  // later jumps, find-in-page, scroll restoration and full-page captures all see real sizes
  function cvOffWhenIdle() {
    (window.requestIdleCallback || function (f) { return setTimeout(f, 300); })(cvOff, { timeout: 1000 });
  }
  if (document.readyState === 'complete') cvOffWhenIdle();
  else window.addEventListener('load', cvOffWhenIdle);

  /* ---------------------------------------------------------------- progress bar + active chapter */
  var nav = document.getElementById('top-nav');
  var bar = null;
  if (nav) {
    bar = document.createElement('div');
    bar.className = 'progress';
    bar.setAttribute('aria-hidden', 'true');
    nav.appendChild(bar);
  }
  var secs = [];
  SECTIONS.forEach(function (id) { var s = document.getElementById(id); if (s) secs.push(s); });
  var links = Array.prototype.slice.call(document.querySelectorAll('.rail a, .nav-links a, .menu-panel a'));
  var rail = document.querySelector('.rail');
  var railLinks = Array.prototype.slice.call(document.querySelectorAll('.rail a'));
  var current;

  function setActive(id) {
    if (id === current) return;
    current = id;
    for (var i = 0; i < links.length; i++) {
      if (id && links[i].getAttribute('href') === '#' + id) links[i].setAttribute('aria-current', 'true');
      else links[i].removeAttribute('aria-current');
    }
    var idx = -1;
    for (var j = 0; j < railLinks.length; j++) if (railLinks[j].getAttribute('href') === '#' + id) idx = j;
    for (var k = 0; k < railLinks.length; k++) railLinks[k].classList.toggle('is-past', idx >= 0 && k < idx);
    if (rail) rail.style.setProperty('--rail-k', idx > 0 ? (idx / (railLinks.length - 1)).toFixed(4) : '0');
  }

  function activeId() {
    if (!secs.length) return null;
    var vh = window.innerHeight || doc.clientHeight;
    var max = doc.scrollHeight - vh;
    if (max > 0 && window.scrollY >= max - 2) return secs[secs.length - 1].id; // bottom of the page: last chapter
    var line = vh * 0.45; // the reading line app.js also uses for its nav highlight
    var id = null;
    for (var i = 0; i < secs.length; i++) {
      if (secs[i].getBoundingClientRect().top <= line) id = secs[i].id;
    }
    return id;
  }

  /* reveal sweep: app.js's IntersectionObserver only sees an element while it crosses the viewport, so a fast
     jump (scrollbar drag, End key, full-page capture) can leave passed cards hidden. On every scroll frame, show
     whatever is in or above the viewport; cards already scrolled past appear without a transition. */
  var revealRoots = Array.prototype.slice.call(document.querySelectorAll('main > section'));
  function sweepReveals() {
    if (!doc.classList.contains('rv-on')) return;
    var vh = window.innerHeight || doc.clientHeight;
    var limit = vh * 0.92;
    var instant = [];
    for (var i = 0; i < revealRoots.length; i++) {
      // a section that starts below the reading limit holds nothing to reveal yet (and is never force-laid-out)
      if (revealRoots[i].getBoundingClientRect().top >= limit) break;
      var pend = revealRoots[i].querySelectorAll('.rv:not(.in)');
      for (var j = 0; j < pend.length; j++) {
        var r = pend[j].getBoundingClientRect();
        if (r.top >= limit) continue;
        // passed already, or brought into view by a resize rather than a scroll: no entrance
        if (r.bottom <= 0 || resized) { pend[j].classList.add('rv-now', 'rv-done'); instant.push(pend[j]); }
        pend[j].classList.add('in');
      }
      // merge: p5c's row-staggered tables own their entry (datamotion.js starts it as they come into view); a table that a
      // fast jump carried past, or a resize brought into view, just appears, like a passed card above
      var tabs = revealRoots[i].querySelectorAll('.dm-stagger:not(.dm-rows-in)');
      for (var t = 0; t < tabs.length; t++) {
        var rt = tabs[t].getBoundingClientRect();
        if (rt.bottom <= 0 || (resized && rt.top < limit)) tabs[t].classList.add('dm-rows-now', 'dm-rows-in');
      }
    }
    if (instant.length) raf(function () { raf(function () { instant.forEach(function (el) { el.classList.remove('rv-now'); }); }); });
  }

  var ticking = false, resized = false;
  function update() {
    ticking = false;
    sweepReveals();
    resized = false;
    var max = doc.scrollHeight - (window.innerHeight || doc.clientHeight);
    var p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
    if (bar) bar.style.transform = 'scaleX(' + p.toFixed(4) + ')';
    setActive(activeId());
  }
  function requestUpdate() { if (!ticking) { ticking = true; raf(update); } }
  window.addEventListener('scroll', requestUpdate, { passive: true });
  // a real resize (new width, or a much taller viewport as in a full-page capture); mobile URL-bar show/hide only
  // nudges the height and keeps the normal entrances
  var lastW = window.innerWidth, lastH = window.innerHeight;
  window.addEventListener('resize', function () {
    var w = window.innerWidth, h = window.innerHeight;
    if (w !== lastW || Math.abs(h - lastH) > lastH * 0.3) resized = true;
    lastW = w; lastH = h;
    requestUpdate();
  });
  window.addEventListener('load', requestUpdate);
  update();

  /* ---------------------------------------------------------------- theme toggle: label, icon turn, other tabs */
  var toggle = document.querySelector('.theme-toggle');
  var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  function effectiveTheme() {
    var t = doc.getAttribute('data-theme');
    if (t === 'light' || t === 'dark') return t;
    return mq && mq.matches ? 'dark' : 'light';
  }
  function labelToggle() {
    if (!toggle) return;
    var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    toggle.setAttribute('aria-label', 'Switch to ' + next + ' theme');
    toggle.setAttribute('title', 'Switch to ' + next + ' theme');
  }
  function syncMeta() {
    var dark = effectiveTheme() === 'dark';
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    for (var i = 0; i < metas.length; i++) metas[i].setAttribute('content', dark ? '#0A131C' : '#FAFBFC');
  }
  if (toggle) {
    toggle.addEventListener('click', function () {
      labelToggle();
      if (!reduce) {
        toggle.classList.remove('flip');
        void toggle.offsetWidth; // restart the icon animation
        toggle.classList.add('flip');
      }
    });
    toggle.addEventListener('animationend', function () { toggle.classList.remove('flip'); });
    labelToggle();
  }
  if (mq) {
    var onScheme = function () { labelToggle(); };
    if (mq.addEventListener) mq.addEventListener('change', onScheme);
    else if (mq.addListener) mq.addListener(onScheme);
  }
  window.addEventListener('storage', function (e) {
    if (e.key !== 'dnmopd-theme') return;
    if (e.newValue === 'light' || e.newValue === 'dark') doc.setAttribute('data-theme', e.newValue);
    else doc.removeAttribute('data-theme');
    labelToggle();
    syncMeta();
  });

  /* ---------------------------------------------------------------- one pause gate for looping motion
     gate(el, fn) calls fn(true) while el is on screen (40px margin) and the tab is visible, fn(false) otherwise, and
     only when that changes. One IntersectionObserver and one visibilitychange listener serve every caller. */
  var gates = [];
  var gateIO = 'IntersectionObserver' in window ? new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      for (var i = 0; i < gates.length; i++) if (gates[i].el === en.target) { gates[i].seen = en.isIntersecting; gates[i].sync(); }
    });
  }, { rootMargin: '40px 0px' }) : null;
  function gate(el, fn) {
    var r = el.getBoundingClientRect(), vh = window.innerHeight || doc.clientHeight;
    var g = { el: el, seen: !gateIO || (r.bottom > -40 && r.top < vh + 40), on: null };
    g.sync = function () {
      var on = g.seen && !document.hidden;
      if (on !== g.on) { g.on = on; fn(on); }
    };
    gates.push(g);
    if (gateIO) gateIO.observe(el);
    g.sync();
    return g;
  }
  document.addEventListener('visibilitychange', function () { gates.forEach(function (g) { g.sync(); }); });
  window.DNMOPD = window.DNMOPD || {};
  window.DNMOPD.gate = gate;

  // hero: html.hero-off pauses the aurora, the schematic (hero.css) and the scroll cue in one go
  var stage = document.querySelector('.hx-stage') || document.querySelector('.hero');
  if (stage) gate(stage, function (on) { doc.classList.toggle('hero-off', !on); });

  /* ---------------------------------------------------------------- staggered reveals */
  // measuring cards inside a section that content-visibility is still skipping would force its layout, so until
  // cv-off only the always-rendered sections are measured
  function stagger() {
    if (!doc.classList.contains('rv-on')) return;
    var parents = [];
    var items = document.querySelectorAll(doc.classList.contains('cv-off') ? '.rv:not(.in)'
      : '.hero .rv:not(.in), #overview .rv:not(.in), #method .rv:not(.in)');
    for (var i = 0; i < items.length; i++) {
      var p = items[i].parentElement;
      if (parents.indexOf(p) < 0) parents.push(p);
    }
    parents.forEach(function (p) {
      var kids = [];
      for (var c = p.firstElementChild; c; c = c.nextElementSibling) if (c.classList.contains('rv') && !c.classList.contains('in')) kids.push(c);
      var rows = [];
      kids.forEach(function (el) {
        var r = el.getBoundingClientRect();
        var row = null;
        for (var j = 0; j < rows.length; j++) if (Math.abs(rows[j].top - r.top) < 8) row = rows[j];
        if (!row) { row = { top: r.top, els: [] }; rows.push(row); }
        row.els.push({ el: el, left: r.left });
      });
      rows.forEach(function (row) {
        row.els.sort(function (a, b) { return a.left - b.left; });
        row.els.forEach(function (o, n) { o.el.style.setProperty('--rv-d', Math.min(n, 4) * 90 + 'ms'); });
      });
    });
  }
  if (!reduce) {
    stagger();
    afterCvOff.push(stagger);
    var st;
    window.addEventListener('resize', function () { clearTimeout(st); st = setTimeout(stagger, 200); });
  }
  // once revealed, cards get their hover transitions back (and lose the entrance delay)
  var shown = document.querySelectorAll('.rv.in');
  for (var s = 0; s < shown.length; s++) shown[s].classList.add('rv-done');
  function doneReveal(e) {
    var t = e.target;
    if (t && t.classList && t.classList.contains('rv') && t.classList.contains('in') && e.propertyName === 'opacity') t.classList.add('rv-done');
  }
  document.addEventListener('transitionend', doneReveal);
  document.addEventListener('transitioncancel', doneReveal);

  /* ---------------------------------------------------------------- figure lightbox */
  var figs = document.querySelectorAll('.float-card');
  var canDialog = typeof HTMLDialogElement === 'function' && 'showModal' in document.createElement('dialog');
  var dlg, lbTitle, lbImg, lbCap, lbNew, lbZoom, lbStage, lastTrigger;
  var ICON_EXPAND = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M14 4h6v6M10 20H4v-6M20 4l-6.5 6.5M4 20l6.5-6.5" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  function buildDialog() {
    dlg = document.createElement('dialog');
    dlg.className = 'lb';
    dlg.setAttribute('aria-labelledby', 'lb-title');
    dlg.innerHTML =
      '<div class="lb-panel">' +
        '<div class="lb-bar">' +
          '<p class="lb-title" id="lb-title"></p>' +
          '<span class="lb-hint">Esc to close</span>' +
          '<button type="button" class="lb-btn lb-zoom" aria-pressed="false">' +
            '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6" fill="none" stroke="currentColor" stroke-width="1.9"/><path d="M15 15l5 5M8 10.5h5M10.5 8v5" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/></svg>' +
            '<span class="lb-zoom-txt">Zoom</span></button>' +
          '<a class="lb-btn lb-newtab" target="_blank" rel="noopener" aria-label="Open the image in a new tab">' +
            '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M14 5h5v5M19 5l-8 8M18 14v4a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h4" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
            '<span class="lb-txt">New tab</span></a>' +
          '<button type="button" class="lb-btn lb-close" aria-label="Close" autofocus>' +
            '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button>' +
        '</div>' +
        '<div class="lb-stage"><div class="lb-paper"><img class="lb-img" alt=""></div></div>' +
        '<p class="lb-cap"></p>' +
      '</div>';
    document.body.appendChild(dlg);
    lbTitle = dlg.querySelector('.lb-title');
    lbImg = dlg.querySelector('.lb-img');
    lbCap = dlg.querySelector('.lb-cap');
    lbNew = dlg.querySelector('.lb-newtab');
    lbZoom = dlg.querySelector('.lb-zoom');
    lbStage = dlg.querySelector('.lb-stage');
    dlg.querySelector('.lb-close').addEventListener('click', function () { dlg.close(); });
    lbZoom.addEventListener('click', function () { setZoom(!dlg.classList.contains('zoomed')); });
    lbImg.addEventListener('click', function (e) { setZoom(!dlg.classList.contains('zoomed'), e.clientX, e.clientY); });
    // a click on the dimmed area (the dialog box itself, outside the panel) closes
    dlg.addEventListener('click', function (e) { if (e.target === dlg) dlg.close(); });
    dlg.addEventListener('close', function () {
      doc.classList.remove('lb-open');
      setZoom(false);
      var t = lastTrigger;
      lastTrigger = null;
      if (t && document.contains(t)) t.focus({ preventScroll: true });
    });
  }

  function setZoom(on, cx, cy) {
    if (!dlg) return;
    var r = lbStage.getBoundingClientRect();
    var fx = 0.5, fy = 0.5;
    if (cx != null && lbStage.scrollWidth > 0) {
      fx = (cx - r.left + lbStage.scrollLeft) / lbStage.scrollWidth;
      fy = (cy - r.top + lbStage.scrollTop) / lbStage.scrollHeight;
    }
    dlg.classList.toggle('zoomed', on);
    lbZoom.setAttribute('aria-pressed', on ? 'true' : 'false');
    lbZoom.querySelector('.lb-zoom-txt').textContent = on ? 'Fit' : 'Zoom';
    if (on) {
      var px = cx != null ? cx - r.left : r.width / 2;
      var py = cy != null ? cy - r.top : r.height / 2;
      lbStage.scrollLeft = fx * lbStage.scrollWidth - px;
      lbStage.scrollTop = fy * lbStage.scrollHeight - py;
    } else {
      lbStage.scrollLeft = 0;
      lbStage.scrollTop = 0;
    }
  }

  // merge: p5b tags Figure 2 "Full diagram · Figure 2"; the lightbox and its button use the figure's own name
  function figName(fig, dflt) {
    var tag = fig.querySelector('.fig-tag');
    var t = tag ? tag.textContent.trim() : '';
    var m = t.match(/Figure\s+\d+/);
    return m ? m[0] : (t || dflt);
  }

  function openFigure(fig, trigger) {
    var img = fig.querySelector('.fig-white img');
    if (!img) return;
    if (!dlg) buildDialog();
    var name = figName(fig, 'Figure');
    lbTitle.innerHTML = '';
    var t = document.createElement('span');
    t.className = 'fig-tag';
    t.textContent = name;
    lbTitle.appendChild(t);
    lbTitle.appendChild(document.createTextNode('Full size'));
    lbImg.src = img.currentSrc || img.src;
    lbImg.alt = img.alt;
    lbNew.href = img.getAttribute('src');
    var cap = fig.querySelector('figcaption');
    lbCap.innerHTML = '';
    if (cap) {
      var c = cap.cloneNode(true);
      var drop = c.querySelectorAll('.fig-tag, .fig-open');
      for (var i = 0; i < drop.length; i++) drop[i].parentNode.removeChild(drop[i]);
      while (c.firstChild) lbCap.appendChild(c.firstChild);
    }
    lbCap.hidden = !lbCap.textContent.trim();
    lastTrigger = trigger;
    setZoom(false);
    dlg.showModal();
    doc.classList.add('lb-open');
  }

  if (canDialog) {
    Array.prototype.forEach.call(figs, function (fig) {
      var wrap = fig.querySelector('.fig-white');
      var img = wrap && wrap.querySelector('img');
      if (!img) return;
      var name = figName(fig, 'figure');
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'fig-expand';
      btn.setAttribute('aria-label', 'View ' + name + ' full size');
      btn.setAttribute('aria-haspopup', 'dialog');
      btn.title = 'View full size';
      btn.innerHTML = ICON_EXPAND;
      wrap.appendChild(btn);
      btn.addEventListener('click', function () { openFigure(fig, btn); });
      img.addEventListener('click', function () { openFigure(fig, btn); });
      var link = fig.querySelector('.fig-open');
      if (link) {
        link.setAttribute('aria-haspopup', 'dialog');
        link.addEventListener('click', function (e) {
          if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return; // keep "open in new tab" gestures
          e.preventDefault();
          openFigure(fig, link);
        });
      }
    });
  }

  /* ---------------------------------------------------------------- copy BibTeX feedback */
  var copyBtn = document.querySelector('.copy-btn');
  var bib = document.querySelector('.bib');
  if (copyBtn) {
    var icon = copyBtn.querySelector('svg');
    if (icon) {
      icon.classList.add('i-copy');
      icon.insertAdjacentHTML('afterend', '<svg class="i-check" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>');
    }
    if (bib && 'MutationObserver' in window) {
      var flashT;
      new MutationObserver(function () {
        if (!copyBtn.classList.contains('done')) return;
        bib.classList.remove('flash');
        void bib.offsetWidth;
        bib.classList.add('flash');
        clearTimeout(flashT);
        flashT = setTimeout(function () { bib.classList.remove('flash'); }, 1200);
      }).observe(copyBtn, { attributes: true, attributeFilter: ['class'] });
    }
  }

  /* ---------------------------------------------------------------- Table 3: labelled cells for the phone card view */
  var t3 = document.querySelector('#table3 .dt');
  if (t3) {
    var heads = Array.prototype.map.call(t3.querySelectorAll('thead th'), function (th) { return th.textContent.trim(); });
    // keep table semantics when phone CSS turns rows into cards
    t3.setAttribute('role', 'table');
    Array.prototype.forEach.call(t3.querySelectorAll('thead, tbody'), function (g) { g.setAttribute('role', 'rowgroup'); });
    Array.prototype.forEach.call(t3.querySelectorAll('tr'), function (tr) {
      tr.setAttribute('role', 'row');
      Array.prototype.forEach.call(tr.children, function (cell, i) {
        if (cell.tagName === 'TD') { cell.setAttribute('role', 'cell'); cell.setAttribute('data-label', heads[i] || ''); }
        else cell.setAttribute('role', cell.getAttribute('scope') === 'row' ? 'rowheader' : 'columnheader');
      });
    });
    t3.classList.add('dt-cards');
    var w3 = t3.closest('.table-wrap');
    if (w3) w3.classList.add('has-cards');
    // let app.js re-measure which tables still scroll sideways
    try { window.dispatchEvent(new Event('resize')); } catch (e) {}
  }
})();
