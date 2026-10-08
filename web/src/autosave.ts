/** Serialize background saves; never acknowledge edits newer than a snapshot. */
export interface AutosaveOptions<T> {
  capture: (signal: AbortSignal) => Promise<T>;
  write: (value: T) => Promise<void>;
  idle: () => boolean;
  render: () => void;
  delay?: number;
}
export class Autosave<T> {
  revision = 0;
  savedRevision = 0;
  exportedRevision = 0;
  enabled = true;
  paused = true;
  saving = false;
  error = "";
  private timer: ReturnType<typeof setTimeout> | undefined;
  private controller: AbortController | undefined;
  private disposed = false;
  private options: AutosaveOptions<T>;
  constructor(options: AutosaveOptions<T>) { this.options = options; }
  get unsaved(): boolean { return this.revision > Math.max(this.savedRevision, this.exportedRevision); }
  changed(): void { this.revision++; this.controller?.abort(); this.options.render(); this.schedule(); }
  schedule(): void {
    clearTimeout(this.timer);
    if (this.disposed || !this.enabled || this.paused || this.saving || this.error || this.revision <= this.savedRevision) return;
    this.timer = setTimeout(() => { void this.save(); }, this.options.delay ?? 1000);
  }
  async save(): Promise<void> {
    clearTimeout(this.timer);
    if (this.disposed || !this.enabled || this.paused || this.saving || this.revision <= this.savedRevision) return;
    if (!this.options.idle()) { this.schedule(); return; }
    const revision = this.revision;
    this.saving = true; this.error = ""; this.options.render();
    const controller = this.controller = new AbortController();
    try {
      const value = await this.options.capture(controller.signal);
      this.controller = undefined;
      if (revision !== this.revision || this.paused || !this.enabled) return;
      await this.options.write(value);
      this.savedRevision = revision;
    } catch (error) { if (!controller.signal.aborted) this.error = error instanceof Error ? error.message : String(error); }
    finally { this.controller = undefined; this.saving = false; this.options.render(); this.schedule(); }
  }
  retry(): void { this.error = ""; this.options.render(); this.schedule(); }
  recovered(): void {
    // Resuming can leave other loaded sources in the workspace; save that state too.
    this.paused = false; this.error = ""; this.changed();
  }
  exported(revision: number): void { this.exportedRevision = revision; this.options.render(); }
  dispose(): void { this.disposed = true; clearTimeout(this.timer); this.controller?.abort(); }
}
