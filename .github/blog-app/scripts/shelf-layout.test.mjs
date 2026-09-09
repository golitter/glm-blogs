import assert from 'node:assert/strict';
import { test } from 'node:test';
import { shelfLayout } from '../src/lib/shelf-layout.ts';

for (const count of [0, 1, 10, 11, 15, 16, 20, 21, 103]) {
  test(`${count} categories: all visible on growing tiers`, () => {
    const items = Array.from({length: count}, (_, id) => id);
    const layout = shelfLayout(items);
    assert.equal(layout.rows, Math.max(2, Math.ceil(count / 5)));
    assert.equal(layout.boards.length, layout.rows + 1);
    assert.deepEqual(layout.entries.map(entry => entry.item), items);
    const positions = new Set();
    for (const {x, y} of layout.entries) {
      assert.ok(y - 1.15 > .6 + .11, 'book rests above bottom shelf');
      assert.ok(y + 1.15 < layout.height - 1, 'book fits below cabinet header');
      positions.add(`${x}:${y}`);
    }
    assert.equal(positions.size, count);
    for (let i = 0; i < layout.entries.length; i++) for (let j = i + 1; j < layout.entries.length; j++) {
      const a = layout.entries[i], b = layout.entries[j];
      assert.ok(Math.abs(a.x - b.x) >= 1.25 || Math.abs(a.y - b.y) >= 2.3);
    }
  });
}
