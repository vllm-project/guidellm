import { BarCustomLayerProps } from '@nivo/bar';

export interface CustomBarsProps<T> extends BarCustomLayerProps<T> {
  xScaleFunc: (d: number) => number;
  yScaleFunc: (d: number) => number;
  heightOffset: number;
}
