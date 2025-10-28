import { Name, SloState } from './slo.interfaces';

export const name: Readonly<Name> = 'slo.state';

export const defaultPercentile = 'p90';

export const initialState: SloState = {
  currentRequestRate: 0,
  enforcedPercentile: defaultPercentile,
  current: {
    timePerRequest: 0,
    ttft: 0,
    itl: 0,
    throughput: 0,
  },
  tasksDefaults: {
    timePerRequest: 0,
    ttft: 0,
    itl: 0,
    throughput: 0,
  },
};
