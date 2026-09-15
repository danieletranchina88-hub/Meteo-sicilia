/* Shared requests and bounded frame storage. No network on a warm frame. */
(function (root) {
  'use strict';
  class ResourceCache {
    constructor() { this.values = new Map(); this.pending = new Map(); this.generation = 0; }
    get(key) { return this.values.get(key); }
    has(key) { return this.values.has(key); }
    clear() { this.generation++; this.values.clear(); this.pending.clear(); }
    load(key, loader) {
      if (this.values.has(key)) return Promise.resolve(this.values.get(key));
      if (this.pending.has(key)) return this.pending.get(key);
      const generation = this.generation;
      const request = Promise.resolve().then(loader).then(value => {
        if (generation === this.generation) this.values.set(key, value);
        return value;
      }).finally(() => { if (this.pending.get(key) === request) this.pending.delete(key); });
      this.pending.set(key, request);
      return request;
    }
  }
  class FrameCache {
    constructor(budget) { this.budget = budget; this.bytes = 0; this.entries = new Map(); }
    get(key) {
      const frame = this.entries.get(key);
      if (frame) { this.entries.delete(key); this.entries.set(key, frame); }
      return frame;
    }
    has(key) { return this.entries.has(key); }
    set(key, frame) {
      if (this.entries.has(key)) this.bytes -= this.entries.get(key).pixels.byteLength;
      this.entries.delete(key);
      if (frame.pixels.byteLength > this.budget) return;
      this.entries.set(key, frame); this.bytes += frame.pixels.byteLength;
      while (this.bytes > this.budget) {
        const oldest = this.entries.keys().next().value;
        this.bytes -= this.entries.get(oldest).pixels.byteLength;
        this.entries.delete(oldest);
      }
    }
    clear() { this.entries.clear(); this.bytes = 0; }
  }
  const api = { ResourceCache, FrameCache };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ForecastCache = api;
})(typeof window !== 'undefined' ? window : this);
