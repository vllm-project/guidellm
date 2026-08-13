import { LinesSeries, Margins, Point } from '../common/interfaces';
import { ContainerSize } from './components/ContainerSizeWrapper';

export interface MiniCombinedProps {
  bars: Point[];
  lines: LinesSeries[];
  width: number;
  height: number;
  margins?: Margins;
  xLegend: string;
}

export interface MiniCombinedWithResizeProps extends MiniCombinedProps {
  containerSize: ContainerSize;
}
