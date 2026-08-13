export interface CustomGridProps {
  xScale: (d: number) => number;
  yScale: (d: number) => number;
  width: number;
  height: number;
  xTicks: number[];
  yTicks: number[];
  scaledHeight: number;
  fullGrid?: boolean;
}
