import { LineSvgProps } from '@nivo/line';

import { ScaleType } from '../DashedLine/DashedLine.interfaces';

export enum LineColor {
  Primary,
  Secondary,
  Tertiary,
  Quarternary,
}
export interface MetricLineProps extends LineSvgProps {
  threshold?: number;
  lineColor: LineColor;
  yScaleType?: ScaleType;
}
