// Moves the inline "Contents" + generated kramdown TOC into a sidebar container on post pages.
// On mobile, the sidebar is shown above the content; on wide desktops it becomes a fixed left rail
// (outside the main content column) with a minimize toggle.
(function () {
  const STORAGE_KEY = 'toc-sidebar-collapsed';

  function setCollapsed(collapsed) {
    document.body.classList.toggle('toc-collapsed', collapsed);
    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
    } catch (_) {}
  }

  function getCollapsed() {
    try {
      return localStorage.getItem(STORAGE_KEY) === '1';
    } catch (_) {
      return false;
    }
  }

  function findTocInPost(postEl) {
    const contentEl = postEl.querySelector('.post-content');
    if (!contentEl) return null;

    // kramdown renders {:toc} as <ul id="markdown-toc"> with the class we set.
    const tocList =
      contentEl.querySelector('ul.table-of-content#markdown-toc') ||
      contentEl.querySelector('ul.table-of-content');
    if (!tocList) return null;

    // Often the heading is authored as: <h3>Contents</h3> followed by the TOC <ul>.
    const prev = tocList.previousElementSibling;
    const hasContentsHeading =
      prev &&
      /^H[1-6]$/.test(prev.tagName) &&
      /contents/i.test((prev.textContent || '').trim());

    return { tocList, headingEl: hasContentsHeading ? prev : null };
  }

  function moveTocIntoSidebar() {
    const postEl = document.querySelector('article.post');
    if (!postEl) return;

    const sidebar = postEl.querySelector('.post-toc-sidebar');
    if (!sidebar) return;

    // Avoid doing the move more than once.
    if (postEl.classList.contains('toc-sidebar-ready')) return;

    const toc = findTocInPost(postEl);
    if (!toc) {
      sidebar.innerHTML = '';
      sidebar.setAttribute('hidden', 'hidden');
      return;
    }

    // Clear + unhide sidebar, then move nodes.
    sidebar.innerHTML = '';
    sidebar.removeAttribute('hidden');

    // Build a small header row with a toggle button.
    const header = document.createElement('div');
    header.className = 'post-toc-header';

    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'post-toc-toggle';
    toggleBtn.type = 'button';
    toggleBtn.setAttribute('aria-label', 'Toggle table of contents');
    toggleBtn.setAttribute('title', 'Toggle table of contents');

    header.appendChild(toggleBtn);
    sidebar.appendChild(header);

    const fragment = document.createDocumentFragment();

    if (toc.headingEl) {
      // Normalize the heading text a bit.
      toc.headingEl.classList.add('post-toc-title');
      fragment.appendChild(toc.headingEl);
    }

    toc.tocList.classList.add('post-toc-list');
    fragment.appendChild(toc.tocList);

    sidebar.appendChild(fragment);

    postEl.classList.add('has-toc', 'toc-sidebar-ready');

    // Restore persisted state + wire the toggle.
    setCollapsed(getCollapsed());
    const syncBtn = () => {
      const collapsed = document.body.classList.contains('toc-collapsed');
      toggleBtn.textContent = collapsed ? '»' : '«';
      toggleBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    };
    syncBtn();

    toggleBtn.addEventListener('click', () => {
      const collapsed = document.body.classList.contains('toc-collapsed');
      setCollapsed(!collapsed);
      syncBtn();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', moveTocIntoSidebar);
  } else {
    moveTocIntoSidebar();
  }
})();


