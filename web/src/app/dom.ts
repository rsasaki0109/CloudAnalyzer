/** Small DOM and formatting helpers shared by every panel. */

export const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export function setStatus(message: string, isError = false): void {
  const el = $("status");
  el.textContent = message;
  el.classList.toggle("error", isError);
}

/** The message of a thrown value. */
export function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function fmt(v: number): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e4 || a < 1e-3 ? v.toExponential(3) : v.toPrecision(5).replace(/\.?0+$/, "");
}

/** Coordinates with enough decimals for their magnitude. */
export function coord(v: number): string {
  return v.toFixed(Math.abs(v) >= 1e5 ? 3 : 4);
}

export function compact(n: number): string {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);
}

/** `v` rounded up to two significant digits, e.g. 0.0372 -> 0.038. */
export function roundUp(v: number): number {
  const pow = 10 ** Math.floor(Math.log10(v || 1));
  return Number((Math.ceil(v / pow) * pow).toPrecision(2));
}

/** Fill a table body with key / value rows. */
export function fillTable(tbody: HTMLElement, rows: [string, string][]): void {
  tbody.replaceChildren(
    ...rows.map(([k, v]) => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = k;
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(th, td);
      return tr;
    }),
  );
}

/** Offer a blob as a download. */
export function download(data: Blob | Uint8Array, filename: string): void {
  const blob = data instanceof Blob ? data : new Blob([data as BlobPart]);
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  // Give the browser a moment to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** A "✕" button that removes something. */
export function removeButton(onclick: () => void): HTMLButtonElement {
  const remove = document.createElement("button");
  remove.className = "remove";
  remove.textContent = "✕";
  remove.title = "Remove";
  remove.onclick = onclick;
  return remove;
}

/** Set a select and let its listeners react, as if the user had picked it. */
export function choose(id: string, value: string): void {
  const select = $<HTMLSelectElement>(id);
  select.value = value;
  select.dispatchEvent(new Event("change"));
}

/** Whether a key event is typing into a form field rather than a shortcut. */
export function typing(e: KeyboardEvent): boolean {
  return (
    e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement
  );
}
