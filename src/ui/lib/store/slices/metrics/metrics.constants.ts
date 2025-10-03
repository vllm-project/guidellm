import { MetricsState, Name } from './metrics.interfaces';
export const name: Readonly<Name> = 'metrics.state';

export const initialState: MetricsState = {
  currentRequestRate: 0,
  timePerRequest: { valuesByRps: {} },
  ttft: { valuesByRps: {} },
  itl: { valuesByRps: {} },
  throughput: { valuesByRps: {} },
};
