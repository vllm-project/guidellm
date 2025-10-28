export type Name = 'slo.state';

export interface SloState {
  currentRequestRate: number;
  enforcedPercentile: string;
  current: {
    timePerRequest: number;
    ttft: number;
    itl: number;
    throughput: number;
  };
  tasksDefaults: {
    timePerRequest: number;
    ttft: number;
    itl: number;
    throughput: number;
  };
}
