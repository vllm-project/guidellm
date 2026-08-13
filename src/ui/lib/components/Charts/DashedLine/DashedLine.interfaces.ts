import { LineSvgProps } from '@nivo/line';

import { Margins } from '../common/interfaces';

export enum ScaleType {
  log = 'symlog',
  linear = 'linear',
}

export interface DashedLineProps extends LineSvgProps {
  margins?: Margins;
  xLegend: string;
  yLegend: string;
  minX?: number;
  yScaleType?: ScaleType;
}
