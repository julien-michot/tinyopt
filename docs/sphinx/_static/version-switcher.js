function addVersionSwitcher() {
  const sidebar = document.querySelector('.sphinxsidebarwrapper');
  const navbar = document.querySelector('.navbar .container');
  const container = sidebar || navbar;
  if (!container) return;

  const currentVersion = window.location.pathname.includes('/dev/') ? 'dev' : 'latest';
  const label = document.createElement('label');
  label.className = sidebar ? 'version-switcher' : 'version-switcher version-switcher-navbar';
  label.append(document.createTextNode(sidebar ? 'Documentation version' : 'Docs:'));

  const select = document.createElement('select');
  select.setAttribute('aria-label', 'Documentation version');
  for (const [version, name, path] of [
    ['dev', 'Development (main)', '/tinyopt/dev/'],
    ['latest', 'Latest release', '/tinyopt/'],
  ]) {
    const option = document.createElement('option');
    option.value = path;
    option.textContent = name;
    option.selected = version === currentVersion;
    select.append(option);
  }
  select.addEventListener('change', () => window.location.assign(select.value));
  label.append(select);
  if (sidebar) sidebar.prepend(label);
  else container.append(label);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', addVersionSwitcher, { once: true });
} else {
  addVersionSwitcher();
}