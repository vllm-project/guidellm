import { useTheme } from '@mui/material';
import { BarCustomLayerProps, ResponsiveBar } from '@nivo/bar';
import { Point } from '@nivo/core';
import React from 'react';

import { CombinedProps } from './Combined.interfaces';
import CustomBars from './components/CustomBars';
import CustomGrid from './components/CustomGrid';
import { CustomLegendLayer } from './components/CustomLegendLayer';
import CustomTick from './components/CustomTick';
import DottedLines from './components/DottedLines';
import useChartScales from '../common/useChartScales';

export const Component = ({
  bars,
  lines,
  width,
  height,
  margins,
  xLegend,
  yLegend,
}: CombinedProps) => {
  const theme = useTheme();
  const combinedGraphTheme = {
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
  const defaultMargins = { top: 10, left: 10, right: 10, bottom: 10 };
  const finalMargins = {
    ...defaultMargins,
    ...margins,
    bottom: (margins?.bottom || defaultMargins.bottom) + 20,
  };
  const { xTicks, yTicks, fnScaleX, fnScaleY, innerHeight } = useChartScales({
    bars,
    lines,
    width,
    height,
    margins: finalMargins,
  });

  const CustomGridLayer = () => {
    const scaledWidth = fnScaleX(xTicks[xTicks.length - 1]);
    const scaledHeight = innerHeight - fnScaleY(yTicks[0]);

    return (
      <CustomGrid
        xScale={(d: number) => fnScaleX(d)}
        yScale={(d: number) => innerHeight - fnScaleY(d)}
        width={scaledWidth}
        height={innerHeight}
        xTicks={xTicks}
        yTicks={yTicks}
        scaledHeight={scaledHeight}
      />
    );
  };

  const CustomBarLayer = (props: BarCustomLayerProps<Point>) => {
    const heightOffset = innerHeight - fnScaleY(yTicks[0]);
    return (
      <CustomBars
        {...props}
        xScaleFunc={(d: number) => fnScaleX(d)}
        yScaleFunc={(d: number) => innerHeight - fnScaleY(d)}
        heightOffset={heightOffset}
      />
    );
  };
  return (
    <div style={{ width: width + 'px', height: height + 'px', position: 'relative' }}>
      <ResponsiveBar
        animate={true}
        data={bars}
        keys={['y']}
        indexBy="x"
        margin={{
          top: finalMargins.top,
          right: finalMargins.right,
          bottom: finalMargins.bottom,
          left: finalMargins.left,
        }}
        padding={0.5}
        // TODO: change colors scheme
        // colors={{ scheme: 'category10' }}
        colors={[theme.palette.primary.shades.B80]}
        axisLeft={{
          tickValues: yTicks,
          legend: yLegend,
          legendPosition: 'middle',
          legendOffset: -30,
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          renderTick: (tick) => (
            <CustomTick
              isXAxis={false}
              scale={(d: number) => Math.abs(innerHeight - fnScaleY(d))}
              tick={yTicks[tick.tickIndex]}
              isFirst={tick.tickIndex === 0}
              isLast={tick.tickIndex === yTicks.length - 1}
            />
          ),
        }}
        axisBottom={{
          tickValues: xTicks,
          legend: xLegend,
          legendPosition: 'middle',
          legendOffset: 30,
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          renderTick: (tick) => (
            <CustomTick
              isXAxis={true}
              scale={(d: number) => fnScaleX(d)}
              tick={xTicks[tick.tickIndex]}
              isFirst={tick.tickIndex === 0}
              isLast={tick.tickIndex === xTicks.length - 1}
            />
          ),
        }}
        layers={[
          'axes',
          CustomGridLayer,
          CustomBarLayer,
          () => <CustomLegendLayer series={lines} height={height} />,
        ]}
        theme={combinedGraphTheme}
      />
      <DottedLines
        lines={lines}
        leftMargin={finalMargins.left}
        topMargin={finalMargins.top}
        innerHeight={innerHeight}
        xScale={fnScaleX}
      />
    </div>
  );
};
