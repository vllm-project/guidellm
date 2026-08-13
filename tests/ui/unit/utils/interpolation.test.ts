import { createMonotoneSpline } from '@/lib/utils/interpolation';

test('should reproduce points on a straight line', () => {
  const xs = [0, 1, 2, 3];
  const ys = [0, 2, 4, 6];
  const interpolate = createMonotoneSpline(xs, ys);

  [0, 0.5, 1.5, 3].forEach((x) => {
    expect(interpolate(x)).toBeCloseTo(2 * x, 1e-6);
  });
});

test('should return constant data on flat line', () => {
  const xs = [0, 1, 2, 3];
  const ys = [5, 5, 5, 5];
  const interpolate = createMonotoneSpline(xs, ys);

  [0, 1.5, 2].forEach((x) => {
    expect(interpolate(x)).toBeCloseTo(5, 1e-6);
  });
});

test('should hit each point precisely', () => {
  const xs = [0, 2, 5];
  const ys = [1, 4, 2];
  const interpolate = createMonotoneSpline(xs, ys);

  xs.forEach((x, i) => {
    expect(interpolate(x)).toBeCloseTo(ys[i], 1e-6);
  });
});

test('no local extremas added', () => {
  // generate wavy line
  const xs = Array.from(Array(50)).map((_, i) => (i + 1) / 10);
  const ys = xs.map((x) => 1 + Math.sin((3 * Math.PI * x) / 10));
  // check that each interpolated point is between its two bounding points
  const interpolate = createMonotoneSpline(xs, ys);
  const loopedValuesToTest: { expected: number; actual: number }[] = [];
  for (let i = xs[0]; i < xs[xs.length - 1]; i += 0.01) {
    const upperIndex = xs.findIndex((x) => x >= i);
    if (upperIndex === 0) {
      loopedValuesToTest.push({ expected: interpolate(i), actual: ys[0] });
      continue;
    }
    const lowerY = ys[upperIndex - 1];
    const upperY = ys[upperIndex];
    expect(interpolate(i)).toBeLessThanOrEqual(Math.max(lowerY, upperY));
    expect(interpolate(i)).toBeGreaterThanOrEqual(Math.min(lowerY, upperY));
  }
  loopedValuesToTest.forEach((value) => {
    expect(value.expected).toBeCloseTo(value.actual);
  });
});
