import { Name, WorkloadDetails } from './workloadDetails.interfaces';
import { PercentileValues } from '../benchmarks/benchmarks.interfaces';
export const name: Readonly<Name> = 'workloadDetails';

export const initialState: WorkloadDetails = {
  prompts: {
    samples: [],
    tokenDistributions: {
      statistics: {
        total: 0,
        mean: 0,
        std: 0,
        median: 0,
        min: 0,
        max: 0,
        percentiles: {} as Record<PercentileValues, number>,
        percentileRows: [],
      },
      buckets: [],
      bucketWidth: 0,
    },
  },
  generations: {
    samples: [],
    tokenDistributions: {
      statistics: {
        total: 0,
        mean: 0,
        std: 0,
        median: 0,
        min: 0,
        max: 0,
        percentiles: {} as Record<PercentileValues, number>,
        percentileRows: [],
      },
      buckets: [],
      bucketWidth: 0,
    },
  },
  requestsOverTime: {
    numBenchmarks: 0,
    requestsOverTime: {
      statistics: {
        total: 0,
        mean: 0,
        std: 0,
        median: 0,
        min: 0,
        max: 0,
        percentiles: {} as Record<PercentileValues, number>,
        percentileRows: [],
      },
      buckets: [],
      bucketWidth: 0,
    },
  },
  rateType: '',
  server: {
    target: '',
  },
};
