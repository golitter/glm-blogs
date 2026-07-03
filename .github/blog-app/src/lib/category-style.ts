// Desaturated pastel tones assigned deterministically to top-level categories.
// Color is a scarce resource in this palette — used only as a semantic marker
// on the primary table-of-contents, never as decoration.

export type CategoryTone = {
  bg: string;
  ink: string;
};

const TONES: CategoryTone[] = [
  { bg: "var(--color-tag-sage-bg)", ink: "var(--color-tag-sage-ink)" },
  { bg: "var(--color-tag-clay-bg)", ink: "var(--color-tag-clay-ink)" },
  { bg: "var(--color-tag-amber-bg)", ink: "var(--color-tag-amber-ink)" },
  { bg: "var(--color-tag-plum-bg)", ink: "var(--color-tag-plum-ink)" },
  { bg: "var(--color-tag-sky-bg)", ink: "var(--color-tag-sky-ink)" },
  { bg: "var(--color-tag-slate-bg)", ink: "var(--color-tag-slate-ink)" },
];

export function toneFor(index: number): CategoryTone {
  return TONES[index % TONES.length]!;
}
