import {
  roundDownNice,
  roundNearestNice,
  roundUpNice,
  spacedLogValues,
} from '@/lib/components/Charts/DashedLine/helpers';

describe('roundNearestNice', () => {
  it('rounds to a nearby nice number', () => {
    expect([10, 12]).toContain(roundNearestNice(11));
    expect([25, 30]).toContain(roundNearestNice(27));
    expect([50]).toContain(roundNearestNice(49));
    expect([75, 80, 85]).toContain(roundNearestNice(81));
    expect([800]).toContain(roundNearestNice(810));
    expect([1300, 1400, 1500]).toContain(roundNearestNice(1342));
  });
  it("doesn't round some nice numbers", () => {
    expect(roundNearestNice(15)).toBe(15);
    expect(roundNearestNice(20)).toBe(20);
    expect(roundNearestNice(30)).toBe(30);
    expect(roundNearestNice(40)).toBe(40);
    expect(roundNearestNice(75)).toBe(75);
    expect(roundNearestNice(100)).toBe(100);
    expect(roundNearestNice(150)).toBe(150);
    expect(roundNearestNice(200)).toBe(200);
    expect(roundNearestNice(400)).toBe(400);
    expect(roundNearestNice(1000)).toBe(1000);
    expect(roundNearestNice(1200)).toBe(1200);
  });
});

describe('roundUpNice', () => {
  it('rounds up to a nearby nice number', () => {
    expect([12, 15]).toContain(roundUpNice(11));
    expect([30]).toContain(roundUpNice(27));
    expect([50]).toContain(roundUpNice(49));
    expect([80, 85, 90]).toContain(roundUpNice(79));
    expect([85, 90]).toContain(roundUpNice(81));
    expect([850, 900, 1000]).toContain(roundUpNice(810));
    expect([1350, 1400, 1500]).toContain(roundUpNice(1342));
  });
  it("doesn't round some nice numbers", () => {
    expect(roundUpNice(15)).toBe(15);
    expect(roundUpNice(20)).toBe(20);
    expect(roundUpNice(30)).toBe(30);
    expect(roundUpNice(40)).toBe(40);
    expect(roundUpNice(75)).toBe(75);
    expect(roundUpNice(100)).toBe(100);
    expect(roundUpNice(150)).toBe(150);
    expect(roundUpNice(200)).toBe(200);
    expect(roundUpNice(400)).toBe(400);
    expect(roundUpNice(1000)).toBe(1000);
    expect(roundUpNice(1200)).toBe(1200);
  });
  it("doesn't unexpectedly round down", () => {
    expect(roundUpNice(1.3)).toBeGreaterThanOrEqual(1.5);
    expect(roundUpNice(3)).toBeGreaterThanOrEqual(3);
    expect(roundUpNice(3.3)).toBeGreaterThanOrEqual(3.5);
    expect(roundUpNice(7.3)).toBeGreaterThanOrEqual(7.5);
    expect(roundUpNice(11)).toBeGreaterThanOrEqual(11);
    expect(roundUpNice(19)).toBeGreaterThanOrEqual(19);
  });
});

describe('roundDownNice', () => {
  it('rounds down to a nearby nice number', () => {
    expect([10]).toContain(roundDownNice(11));
    expect([20, 25]).toContain(roundDownNice(27));
    expect([40, 45, 48]).toContain(roundDownNice(49));
    expect([70, 75]).toContain(roundDownNice(79));
    expect([75, 80]).toContain(roundDownNice(81));
    expect([700, 750, 800]).toContain(roundDownNice(810));
    expect([1200, 1250, 1300]).toContain(roundDownNice(1342));
  });
  it("doesn't round some nice numbers", () => {
    expect(roundDownNice(15)).toBe(15);
    expect(roundDownNice(20)).toBe(20);
    expect(roundDownNice(30)).toBe(30);
    expect(roundDownNice(40)).toBe(40);
    expect(roundDownNice(75)).toBe(75);
    expect(roundDownNice(100)).toBe(100);
    expect(roundDownNice(150)).toBe(150);
    expect(roundDownNice(200)).toBe(200);
    expect(roundDownNice(400)).toBe(400);
    expect(roundDownNice(1000)).toBe(1000);
    expect(roundDownNice(1200)).toBe(1200);
  });
  it("doesn't unexpectedly round up", () => {
    expect(roundDownNice(1.6)).toBeLessThanOrEqual(1.5);
    expect(roundDownNice(3)).toBeLessThanOrEqual(3);
    expect(roundDownNice(3.6)).toBeLessThanOrEqual(3.5);
    expect(roundDownNice(7.6)).toBeLessThanOrEqual(7.5);
    expect(roundDownNice(11)).toBeLessThanOrEqual(11);
    expect(roundDownNice(19)).toBeLessThanOrEqual(19);
  });
});

describe('spacedLogValues', () => {
  const checkValuesRoughlyLogSpaced = (values: number[]) => {
    let i = 1;
    if (values[0] === 0) {
      i++;
    }
    const valuesRatios = [];
    for (i; i < values.length; i++) {
      valuesRatios.push(values[i] / values[i - 1]);
    }
    const valuesRatiosAvg = valuesRatios.reduce((a, b) => a + b) / valuesRatios.length;
    valuesRatios.forEach((ratio) => {
      expect(ratio).toBeCloseTo(valuesRatiosAvg, -0.5);
    });
  };

  it('generates an array of roughly log-scale spaced values', () => {
    expect(spacedLogValues(1, 1000, 4)).toEqual([1, 10, 100, 1000]);
    checkValuesRoughlyLogSpaced(spacedLogValues(1, 1324, 4));
    checkValuesRoughlyLogSpaced(spacedLogValues(123, 12324, 6));
    checkValuesRoughlyLogSpaced(spacedLogValues(1, 122, 6));
    checkValuesRoughlyLogSpaced(spacedLogValues(1, 122, 9));
  });
  it('can handle ticks for small numbers', () => {
    checkValuesRoughlyLogSpaced(spacedLogValues(0, 8, 6));
  });

  it('generates an array of nice round numbers', () => {
    for (const value of spacedLogValues(1, 1000, 4)) {
      expect([roundUpNice(value), roundNearestNice(value)]).toContain(value);
    }
    for (const value of spacedLogValues(1, 1324, 4)) {
      expect([roundUpNice(value), roundNearestNice(value)]).toContain(value);
    }
    for (const value of spacedLogValues(1, 132, 7)) {
      expect([roundUpNice(value), roundNearestNice(value)]).toContain(value);
    }
  });
});
