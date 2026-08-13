export interface CustomTickProps {
  scale: (d: number) => number;
  isXAxis: boolean;
  tick: number;
  withTicks?: boolean;
  isFirst: boolean;
  isLast: boolean;
}
