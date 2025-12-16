// Dark mode is default, light mode is the toggle option
function getInitialMode() {
  const savedMode = localStorage.getItem('dark-mode-storage');
  if (savedMode) {
    return savedMode;
  }
  // Default to dark mode for our techy theme
  return 'dark-mode';
}

function applyMode(mode) {
  // Remove both classes first
  document.body.classList.remove('dark-mode', 'light-mode');
  // Add the appropriate class
  document.body.classList.add(mode);
  // Save to localStorage
  localStorage.setItem('dark-mode-storage', mode);
  // Update checkbox state (checked = light mode)
  const switchEl = document.getElementById('switch');
  if (switchEl) {
    switchEl.checked = mode === 'light-mode';
  }
}

function toggleDarkMode() {
  const currentMode = document.body.classList.contains('light-mode') ? 'light-mode' : 'dark-mode';
  const newMode = currentMode === 'dark-mode' ? 'light-mode' : 'dark-mode';
  applyMode(newMode);
}

// Apply initial mode when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function() {
    applyMode(getInitialMode());
  });
} else {
  applyMode(getInitialMode());
}

// Also apply on window load to ensure everything is ready
window.addEventListener('load', function() {
  applyMode(getInitialMode());
});
