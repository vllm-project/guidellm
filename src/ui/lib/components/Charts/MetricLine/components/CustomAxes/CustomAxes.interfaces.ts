import { CustomLayerProps } from '@nivo/line';
import type { ScaleLinear } from '@nivo/scales';

export type CustomLineLayerProps = Omit<CustomLayerProps, 'xScale' | 'yScale'> & {
  yScale: ScaleLinear<number>;
};
