/** Below 720 px the sidebar is a bottom sheet toggled from the toolbar. */

import { $ } from "./dom";

export const narrow = window.matchMedia("(max-width: 720px)");
const panelsButton = $<HTMLButtonElement>("panels");

export function setPanelsOpen(open: boolean): void {
  document.body.classList.toggle("panels-open", open);
  panelsButton.setAttribute("aria-pressed", String(open));
}

panelsButton.onclick = () => setPanelsOpen(!document.body.classList.contains("panels-open"));
// Start with the sheet open so the Open / sample hints are visible.
setPanelsOpen(narrow.matches);
