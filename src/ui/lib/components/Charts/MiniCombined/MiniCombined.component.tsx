import { useTheme } from '@mui/material';
import { BarCustomLayerProps, ResponsiveBar } from '@nivo/bar';
import { Point } from '@nivo/core';
import React from 'react';

import useChartScales from '../common/useChartScales';
import CustomBars from './components/CustomBars';
import CustomGrid from './components/CustomGrid';
import CustomTick from './components/CustomTick';
import { MiniCombinedWithResizeProps } from './MiniCombined.interfaces';

export const Component = ({
  bars,
  lines,
  margins,
  xLegend,
  containerSize,
}: MiniCombinedWithResizeProps) => {
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
    bottom: margins?.bottom || defaultMargins.bottom,
  };
  const { xTicks, yTicks, fnScaleX, fnScaleY, innerHeight } = useChartScales({
    bars,
    lines,
    width: containerSize.width,
    height: containerSize.height,
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
    <div
      style={{
        width: containerSize.width + 'px',
        height: containerSize.height + 'px',
        position: 'relative',
      }}
    >
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
        colors={[theme.palette.primary.shades['0']]}
        axisLeft={null}
        axisBottom={{
          tickValues: xTicks,
          legend: xLegend,
          legendPosition: 'middle',
          legendOffset: 20,
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
        layers={['axes', CustomGridLayer, CustomBarLayer]}
        theme={combinedGraphTheme}
      />
      {/* <DottedLines
        lines={lines}
        leftMargin={finalMargins.left}
        topMargin={finalMargins.top}
        innerHeight={innerHeight}
        xScale={fnScaleX}
      /> */}
    </div>
  );
};
