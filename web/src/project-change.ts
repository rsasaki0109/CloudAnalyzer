/** Editing notifications shared without importing UI panels. */
const listeners = new Set<() => void>();
export function projectChanged(): void { for (const listener of listeners) listener(); }
export function onProjectChanged(listener: () => void): void { listeners.add(listener); }
