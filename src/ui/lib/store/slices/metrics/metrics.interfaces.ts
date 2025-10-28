export type Name = 'metrics.state';

export interface MetricsState {
  currentRequestRate: number;
  timePerRequest: SingleMetricsState;
  ttft: SingleMetricsState;
  itl: SingleMetricsState;
  throughput: SingleMetricsState;
}

export type SingleMetricsState = {
  valuesByRps: Record<number, number>;
};
