// High-saturation tones assigned deterministically to top-level categories.

export type CategoryTone = {
  bg: string;
  ink: string;
};

const TONES: CategoryTone[] = [
  { bg: "var(--color-tag-sage-bg)", ink: "#16a34a" },
  { bg: "var(--color-tag-clay-bg)", ink: "#ff493d" },
  { bg: "var(--color-tag-amber-bg)", ink: "#ffdc22" },
  { bg: "var(--color-tag-plum-bg)", ink: "#8b5cf6" },
  { bg: "var(--color-tag-sky-bg)", ink: "#0ea5e9" },
  { bg: "var(--color-tag-slate-bg)", ink: "#64748b" },
];

export function toneFor(index: number): CategoryTone {
  return TONES[index % TONES.length]!;
}
