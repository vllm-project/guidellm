import { MiniCombinedProps } from '../MiniCombined/MiniCombined.interfaces';

export type Point = {
  x: number;
  y: number;
};

export type LinesSeries = Point & {
  id: string;
};

export type Margins = {
  top?: number;
  bottom?: number;
  left?: number;
  right?: number;
};

type RequiredCombinedProps = Required<MiniCombinedProps>;
type RequiredMargins = Required<Margins>;
type BaseChartScalesProps = Omit<RequiredCombinedProps, 'xLegend' | 'yLegend'>;
export type ChartScalesProps = BaseChartScalesProps & {
  margins: RequiredMargins;
};

export interface CombinedProps {
  bars: Point[];
  lines: LinesSeries[];
  width: number;
  height: number;
  margins?: Margins;
  xLegend: string;
  yLegend: string;
}
