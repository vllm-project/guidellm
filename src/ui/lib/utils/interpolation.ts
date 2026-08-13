export function createMonotoneSpline(xs: number[], ys: number[]) {
  const n = xs.length;
  if (n < 2) {
    throw new Error('Need at least two points');
  }
  const dx = new Array<number>(n - 1);
  const dy = new Array<number>(n - 1);
  const m = new Array<number>(n - 1);
  const c1 = new Array<number>(n);

  for (let i = 0; i < n - 1; i++) {
    dx[i] = xs[i + 1] - xs[i];
    if (dx[i] === 0) {
      throw new Error(`xs[${i}] == xs[${i + 1}]`);
    }
    dy[i] = ys[i + 1] - ys[i];
    m[i] = dy[i] / dx[i];
  }

  c1[0] = m[0];
  for (let i = 1; i < n - 1; i++) {
    if (m[i - 1] * m[i] <= 0) {
      c1[i] = 0;
    } else {
      const dx1 = dx[i - 1],
        dx2 = dx[i],
        common = dx1 + dx2;
      c1[i] = (3 * common) / ((common + dx2) / m[i - 1] + (common + dx1) / m[i]);
    }
  }
  c1[n - 1] = m[n - 2];

  return function (x: number) {
    // Binary search for interval i: xs[i] <= x < xs[i+1]
    let lo = 0,
      hi = n - 2;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (x < xs[mid]) {
        hi = mid - 1;
      } else if (x > xs[mid + 1]) {
        lo = mid + 1;
      } else {
        lo = mid;
        break;
      }
    }
    let i = lo;
    if (i < 0) {
      i = 0;
    } else if (i > n - 2) {
      i = n - 2;
    }

    const h = dx[i];
    const t = (x - xs[i]) / h;
    const t2 = t * t,
      t3 = t2 * t;

    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;

    return h00 * ys[i] + h10 * h * c1[i] + h01 * ys[i + 1] + h11 * h * c1[i + 1];
  };
}
