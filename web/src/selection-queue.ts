/** Keep asynchronous point picks in click order and invalidate abandoned tools. */
export class SelectionQueue {
  private generation = 0;
  private tail: Promise<void> = Promise.resolve();
  private finishing = false;

  enqueue<T>(read: () => Promise<T>, commit: (value: T) => void): Promise<void> {
    if (this.finishing) return Promise.resolve();
    const generation = this.generation;
    // Start the pick now, while the camera still represents this click.
    const result = read().then(
      value => ({ value, error: undefined }),
      error => ({ value: undefined, error }),
    );
    const task = this.tail.then(async () => {
      const picked = await result;
      if (generation !== this.generation) return;
      if (picked.error !== undefined) throw picked.error;
      commit(picked.value as T);
    });
    this.tail = task.catch(() => {});
    return task;
  }

  async finish(commit: () => Promise<void>): Promise<void> {
    if (this.finishing) return;
    this.finishing = true;
    const generation = this.generation;
    try {
      await this.tail;
      if (generation === this.generation) await commit();
    } finally {
      if (generation === this.generation) this.finishing = false;
    }
  }

  cancel(): void {
    this.generation++;
    this.tail = Promise.resolve();
    this.finishing = false;
  }
}
