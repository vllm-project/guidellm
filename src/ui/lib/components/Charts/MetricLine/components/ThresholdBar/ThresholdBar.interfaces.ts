export interface ThresholdBarProps {
  xScale: (value: number) => number;
  yScale: (value: number) => number;
  threshold?: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}
