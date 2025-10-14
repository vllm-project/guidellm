export type Name = 'benchmarks';

export interface Statistics {
  total: number;
  mean: number;
  std: number;
  median: number;
  min: number;
  max: number;
  percentileRows: Percentile[];
  percentiles: Record<PercentileValues, number>;
}

export type PercentileValues = 'p50' | 'p90' | 'p95' | 'p99';

interface Percentile {
  percentile: PercentileValues;
  value: number;
}

export interface BenchmarkMetrics {
  ttft: Statistics;
  tpot: Statistics;
  timePerRequest: Statistics;
  throughput: Statistics;
}

export interface Benchmark extends BenchmarkMetrics {
  requestsPerSecond: number;
}

export type Benchmarks = Benchmark[];
