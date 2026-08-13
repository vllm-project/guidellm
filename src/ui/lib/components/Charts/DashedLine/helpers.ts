import { DatumValue } from '@nivo/line';

type NumberValue = number | { valueOf(): number };

export const toNumberValue = (value: DatumValue | null | undefined): NumberValue => {
  if (value === null || value === undefined) {
    return 0;
  }
  return value as NumberValue;
};

const allowedMultipliers = [
  1, 1.2, 1.4, 1.5, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 7.5, 8, 9, 10,
];

export function roundDownNice(x: number) {
  if (x <= 0) {
    return x;
  }
  const exponent = Math.floor(Math.log10(x));
  const base = Math.pow(10, exponent);
  const fraction = x / base;
  for (const m of allowedMultipliers.slice().reverse()) {
    if (m <= fraction) {
      return Math.floor(m * base);
    }
  }
  return Math.floor(10 * base);
}

export function roundUpNice(x: number) {
  if (x <= 0) {
    return x;
  }
  const exponent = Math.floor(Math.log10(x));
  const base = Math.pow(10, exponent);
  const fraction = x / base;
  for (const m of allowedMultipliers) {
    if (m >= fraction) {
      return Math.ceil(m * base);
    }
  }
  return Math.ceil(10 * base);
}

export function roundNearestNice(x: number) {
  if (x <= 0) {
    return x;
  }
  const exponent = Math.floor(Math.log10(x));
  const base = Math.pow(10, exponent);
  const fraction = x / base;
  let best = allowedMultipliers[0];
  let bestDiff = Math.abs(fraction - best);
  for (const m of allowedMultipliers) {
    const diff = Math.abs(fraction - m);
    if (diff < bestDiff) {
      best = m;
      bestDiff = diff;
    }
  }
  return Math.round(best * base);
}

export function spacedLogValues(min: number, max: number, steps: number) {
  if (steps < 2) {
    return [];
  }
  if (steps > max - min) {
    steps = max - min + 1;
  }

  if (min === 0) {
    const nonzeroCount = steps - 1;
    const exponent = Math.log10(max) / (nonzeroCount - 1);
    const lowerNonZero = roundDownNice(Math.pow(10, exponent));
    const upperTick = roundUpNice(max);
    const r = Math.pow(upperTick / lowerNonZero, 1 / (nonzeroCount - 1));
    const ticks = [0];
    for (let i = 0; i < nonzeroCount; i++) {
      const value = lowerNonZero * Math.pow(r, i);
      ticks.push(roundNearestNice(value));
    }
    return ticks;
  } else {
    const lowerTick = roundNearestNice(min);
    const upperTick = roundUpNice(max);
    const r = Math.pow(upperTick / lowerTick, 1 / (steps - 1));
    const ticks = [];
    for (let i = 0; i < steps; i++) {
      const value = lowerTick * Math.pow(r, i);
      ticks.push(roundNearestNice(value));
    }
    return ticks;
  }
}
