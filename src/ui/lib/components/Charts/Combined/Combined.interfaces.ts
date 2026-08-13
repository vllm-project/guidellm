import { LinesSeries, Margins, Point } from '../common/interfaces';

export interface CombinedProps {
  bars: Point[];
  lines: LinesSeries[];
  width: number;
  height: number;
  margins?: Margins;
  xLegend: string;
  yLegend: string;
}
