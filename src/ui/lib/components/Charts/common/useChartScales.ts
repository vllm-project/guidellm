import { useMemo } from 'react';

import { ChartScalesProps } from './interfaces';

export const calculateStep = (min: number, max: number) => {
  const range = max - min;
  const approxStep = range / 10;
  const magnitude = Math.pow(10, Math.floor(Math.log10(approxStep)));
  return Math.ceil(approxStep / magnitude) * magnitude;
};

const useChartScales = ({ bars, width, height, margins }: ChartScalesProps) => {
  return useMemo(() => {
    const xValues = bars.map((d) => d.x);
    const yValues = bars.map((d) => d.y);

    const xMin = xValues.length === 1 ? 0 : Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues) > 0 ? 0 : Math.min(...yValues);
    const yMax = Math.max(...yValues);

    const xStep = calculateStep(xMin, xMax);
    const yStep = calculateStep(yMin, yMax);

    const xTicks = Array.from(
      { length: Math.ceil((xMax - xMin) / xStep) + 1 },
      (_, i) => xMin + i * xStep
    );
    const yTicks = Array.from(
      { length: Math.ceil((yMax - yMin) / yStep) + 1 },
      (_, i) => yMin + i * yStep
    ).reverse();

    const innerHeight = height - margins.top - margins.bottom;
    const innerWidth = width - margins.left - margins.right;
    const fnScaleX = (d: number) =>
      innerWidth * Math.min(1, (d - xMin) / (xMax - xMin));
    const fnScaleY = (d: number) => (innerHeight * (d - yMin)) / (yMax - yMin);

    return {
      xTicks,
      yTicks,
      fnScaleX,
      fnScaleY,
      innerWidth,
      innerHeight,
    };
  }, [bars, width, height, margins]);
};

export default useChartScales;
