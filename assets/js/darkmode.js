function getInitialMode() {
  const savedMode = localStorage.getItem('dark-mode-storage')
  return savedMode || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

function applyMode(mode) {
  document.body.className = mode;
  localStorage.setItem('dark-mode-storage', mode)
  const isDarkMode = mode === 'dark-mode';
  document.getElementById('switch').checked = isDarkMode;
}

function toggleDarkMode() {
  const mode = document.body.className === 'dark-mode' ? 'light-mode' : 'dark-mode';
  applyMode(mode);
}

// Apply initial mode:
applyMode(getInitialMode());
