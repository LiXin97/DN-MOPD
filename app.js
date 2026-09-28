/* DN-MOPD landing page: progressive enhancement only. Every piece of content is in the HTML and visible without this file. */
(function () {
  'use strict';
  var doc = document.documentElement;

  /* ---- math: replace the plain-text fallbacks with KaTeX when it loaded */
  function renderMath() {
    if (!window.katex) return;
    var els = document.querySelectorAll('[data-tex]');
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      try {
        window.katex.render(el.getAttribute('data-tex'), el, {
          displayMode: el.classList.contains('tex-display'),
          throwOnError: false,
          output: 'htmlAndMathml'
        });
      } catch (e) { /* keep the fallback text */ }
    }
  }
  renderMath();

  /* ---- theme toggle (explicit choice overrides the system scheme) */
  var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  function effectiveTheme() {
    var t = doc.getAttribute('data-theme');
    if (t === 'light' || t === 'dark') return t;
    return mq && mq.matches ? 'dark' : 'light';
  }
  function syncThemeColor() {
    var dark = effectiveTheme() === 'dark';
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    for (var i = 0; i < metas.length; i++) {
      if (doc.hasAttribute('data-theme')) {
        metas[i].setAttribute('content', dark ? '#0A131C' : '#FAFBFC');
      }
    }
  }
  var toggle = document.querySelector('.theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
      doc.setAttribute('data-theme', next);
      try { localStorage.setItem('dnmopd-theme', next); } catch (e) {}
      syncThemeColor();
    });
  }
  syncThemeColor();

  /* ---- nav: hairline when scrolled, active section, close mobile menu on navigation */
  var nav = document.getElementById('top-nav');
  function onScroll() {
    if (nav) nav.classList.toggle('scrolled', window.scrollY > 8);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  var menu = document.querySelector('.menu');
  if (menu) {
    menu.addEventListener('click', function (e) {
      if (e.target && e.target.tagName === 'A') menu.removeAttribute('open');
    });
    document.addEventListener('click', function (e) {
      if (menu.hasAttribute('open') && !menu.contains(e.target)) menu.removeAttribute('open');
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && menu.hasAttribute('open')) {
        menu.removeAttribute('open');
        var s = menu.querySelector('summary');
        if (s) s.focus();
      }
    });
  }

  /* (merge: the active chapter in the rail, top nav and menu is owned by polish.js, which sets aria-current;
     the base IntersectionObserver spy that toggled .active here is gone, so the two can never disagree) */

  /* ---- in-page anchors: smooth scroll via JS (CSS smooth scrolling would also slow programmatic scrolls) */
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.addEventListener('click', function (e) {
    var a = e.target && e.target.closest ? e.target.closest('a[href^="#"]') : null;
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    var id = a.getAttribute('href').slice(1);
    var t = id ? document.getElementById(id) : null;
    if (!t) return;
    e.preventDefault();
    t.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    try { history.pushState(null, '', '#' + id); } catch (err) {}
    if (id === 'main' || id === 'top') return;
    if (!t.hasAttribute('tabindex')) t.setAttribute('tabindex', '-1');
    t.focus({ preventScroll: true });
  });

  /* ---- copy BibTeX */
  var copyBtn = document.querySelector('.copy-btn');
  if (copyBtn) {
    var status = document.querySelector('.copy-status');
    var label = copyBtn.querySelector('.copy-label');
    copyBtn.addEventListener('click', function () {
      var src = document.querySelector(copyBtn.getAttribute('data-copy'));
      if (!src) return;
      var text = src.textContent;
      function done(ok) {
        copyBtn.classList.toggle('done', ok);
        if (label) label.textContent = ok ? 'Copied' : 'Press Ctrl+C';
        if (status) status.textContent = ok ? 'BibTeX copied to clipboard' : 'Copy failed; the BibTeX is selected';
        setTimeout(function () {
          copyBtn.classList.remove('done');
          if (label) label.textContent = 'Copy';
        }, 1800);
      }
      function fallback() {
        try {
          var r = document.createRange();
          r.selectNodeContents(src);
          var sel = window.getSelection();
          sel.removeAllRanges();
          sel.addRange(r);
          done(document.execCommand('copy'));
        } catch (e) { done(false); }
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, fallback);
      } else {
        fallback();
      }
    });
  }

  /* ---- wide tables: say so when a table scrolls sideways */
  var wraps = document.querySelectorAll('.table-wrap');
  var hints = [];
  for (var w = 0; w < wraps.length; w++) {
    var h = document.createElement('p');
    h.className = 'scroll-hint';
    h.setAttribute('aria-hidden', 'true');
    h.textContent = 'Scroll sideways for more columns \u2192';
    wraps[w].parentNode.insertBefore(h, wraps[w]);
    hints.push(h);
  }
  function checkWraps() {
    for (var i = 0; i < wraps.length; i++) {
      var over = wraps[i].scrollWidth > wraps[i].clientWidth + 2 && wraps[i].clientWidth > 0;
      wraps[i].classList.toggle('is-scrollable', over);
      hints[i].classList.toggle('on', over);
    }
  }
  checkWraps();
  window.addEventListener('resize', checkWraps);
  var dv = document.querySelector('.data-view');
  if (dv) dv.addEventListener('toggle', checkWraps);

  /* ---- scroll reveal: only when motion is welcome; content stays visible otherwise */
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduce && 'IntersectionObserver' in window) {
    try {
      var items = document.querySelectorAll('.rv');
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) {
            en.target.classList.add('in');
            io.unobserve(en.target);
          }
        });
      }, { rootMargin: '0px 0px -8% 0px', threshold: 0.01 });
      // Anything already on screen is shown at once, so the first paint is never empty.
      var vh = window.innerHeight || 800;
      for (var j = 0; j < items.length; j++) {
        var r = items[j].getBoundingClientRect();
        if (r.top < vh * 0.92) items[j].classList.add('in');
        else io.observe(items[j]);
      }
      doc.classList.add('rv-on');
      // Printing or saving should never capture hidden cards.
      window.addEventListener('beforeprint', function () {
        for (var k = 0; k < items.length; k++) items[k].classList.add('in');
      });
    } catch (e) {
      doc.classList.remove('rv-on');
    }
  }
})();
