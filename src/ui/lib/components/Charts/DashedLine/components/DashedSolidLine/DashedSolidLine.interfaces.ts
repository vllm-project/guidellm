import type { CustomLayerProps } from '@nivo/line';
import type { ScaleLinear } from '@nivo/scales';

export type DashedSolidLineProps = Omit<CustomLayerProps, 'xScale' | 'yScale'> & {
  xScale: ScaleLinear<number>;
  yScale: ScaleLinear<number>;
};
