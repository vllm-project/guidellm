import { Point } from '../../../common/interfaces';

export interface DottedLinesProps {
  lines: Point[];
  leftMargin: number;
  topMargin: number;
  innerHeight: number;
  xScale: (d: number) => number;
}
