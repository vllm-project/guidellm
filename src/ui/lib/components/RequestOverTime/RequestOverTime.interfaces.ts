export interface RequestOverTimeProps {
  benchmarksCount: number;
  barData: {
    x: number;
    y: number;
  }[];
  rateType: string;
  lines?: { x: number; y: number; id: string }[];
}
