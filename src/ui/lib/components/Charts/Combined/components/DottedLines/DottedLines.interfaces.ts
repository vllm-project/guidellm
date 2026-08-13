import { LinesSeries } from '../../../common/interfaces';

export interface DottedLinesProps {
  lines: LinesSeries[];
  leftMargin: number;
  topMargin: number;
  innerHeight: number;
  xScale: (d: number) => number;
}
