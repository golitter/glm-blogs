/** Five volumes per tier; additional categories grow the cabinet upward. */
export function shelfLayout<T>(items: readonly T[]) {
  const rows = Math.max(2, Math.ceil(items.length / 5));
  const height = 1.6 + rows * 2.95;
  return {
    rows, height,
    boards: Array.from({length: rows + 1}, (_, row) => .6 + row * 2.95),
    entries: items.map((item, index) => ({
      item,
      x: -3.2 + index % 5 * 1.6,
      y: 1.9 + (rows - 1 - Math.floor(index / 5)) * 2.95,
      z: .12 + index % 2 * .08,
    })),
  };
}
