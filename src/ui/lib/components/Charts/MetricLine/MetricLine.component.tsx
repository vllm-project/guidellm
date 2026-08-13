import { useTheme } from '@mui/material';
import { ResponsiveLine } from '@nivo/line';
import React, { FC } from 'react';

import { useColor } from '@/lib/hooks/useColor';

import { MetricLineProps } from '.';
import CustomAxes from './components/CustomAxes';
import ThresholdBar from './components/ThresholdBar';
import { ScaleType } from '../DashedLine/DashedLine.interfaces';

export const Component: FC<MetricLineProps> = ({
  data,
  threshold,
  lineColor,
  yScaleType = ScaleType.log,
}) => {
  const theme = useTheme();
  const selectedColor = useColor(lineColor);
  const lineTheme = {
    axis: {
      legend: {
        text: {
          fill: theme.palette.surface.onSurface,
          fontSize: theme.typography.axisTitle.fontSize,
          fontWeight: theme.typography.axisTitle.fontWeight,
          fontFamily: theme.typography.axisTitle.fontFamily,
        },
      },
      ticks: {
        text: {
          fill: theme.palette.surface.onSurface,
        },
      },
    },
  };
  const xValues = data[0].data.map((d) => d.x) as Array<number>;
  const yValues = data[0].data.map((d) => d.y) as Array<number>;

  const maxX = Math.max(...xValues);
  const minX = Math.min(...xValues);
  const maxY = Math.ceil(Math.max(...yValues));
  const minY = Math.floor(Math.min(...yValues));

  let extraYScaleOptions = {};
  if (yScaleType === ScaleType.linear) {
    extraYScaleOptions = {
      stacked: true,
      reverse: false,
    };
  }

  return (
    <ResponsiveLine
      curve="monotoneX"
      data={data}
      colors={[selectedColor]}
      margin={{ top: 20, right: 10, bottom: 20, left: 35.5 }}
      xScale={{ type: 'linear', min: minX }}
      yScale={{
        type: yScaleType,
        min: minY,
        max: maxY,
        ...extraYScaleOptions,
      }}
      axisBottom={null}
      axisLeft={{
        legendOffset: -30,
        tickRotation: 0,
        tickSize: 5,
        tickPadding: 5,
        tickValues: [minY, maxY],
        renderTick: ({ value, x, y, tickIndex }) => {
          return (
            <g transform={`translate(${x},${y})`} data-id="ticks">
              <text
                x={-4}
                y={tickIndex === 0 ? 0 : 6}
                textAnchor="end"
                style={{
                  fontFamily: theme.typography.axisTitle.fontFamily,
                  fontWeight: theme.typography.axisLabel.fontWeight,
                  fontSize: theme.typography.axisLabel.fontSize,
                  fill: theme.palette.surface.onSurfaceSubdued,
                }}
              >
                {value}
              </text>
            </g>
          );
        },
      }}
      enableGridX={false}
      enableGridY={false}
      pointSize={0}
      useMesh={true}
      layers={[
        CustomAxes,
        ({
          xScale,
          yScale,
        }: {
          xScale: (value: number) => number;
          yScale: (value: number) => number;
        }) => (
          <ThresholdBar
            threshold={threshold}
            yScale={yScale}
            xScale={xScale}
            minX={minX}
            maxX={maxX}
            minY={minY}
            maxY={maxY}
          />
        ),
        'axes',
        'lines',
      ]}
      theme={lineTheme}
    />
  );
};
