import { Statistics } from '../benchmarks';

export type Name = 'workloadDetails';

interface Bucket {
  value: number;
  count: number;
}

interface Distribution {
  statistics: Statistics;
  buckets: Bucket[];
  bucketWidth: number;
}

interface TokenData {
  samples: string[];
  tokenDistributions: Distribution;
}

interface BenchmarkData {
  numBenchmarks: number;
  requestsOverTime: Distribution;
}

interface Server {
  target: string;
}

export interface WorkloadDetails {
  prompts: TokenData;
  generations: TokenData;
  requestsOverTime: BenchmarkData;
  rateType: string;
  server: Server;
}
