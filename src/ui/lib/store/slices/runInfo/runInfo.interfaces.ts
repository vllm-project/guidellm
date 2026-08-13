export type Name = 'runInfo';

export interface RunInfo {
  model: {
    name: string;
    size: number;
  };
  task: string;
  dataset: {
    name: string;
  };
  timestamp: string;
}
