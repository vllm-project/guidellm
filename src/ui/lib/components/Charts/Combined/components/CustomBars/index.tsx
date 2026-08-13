import { ComputedBarDatum } from '@nivo/bar';
import { Point } from '@nivo/core';
import { useTooltip, BasicTooltip } from '@nivo/tooltip';
import React from 'react';

import { CustomBarsProps } from './CustomBars.interfaces';

const CustomBars = ({
  bars,
  xScaleFunc,
  yScaleFunc,
  heightOffset,
}: CustomBarsProps<Point>) => {
  const { showTooltipFromEvent, hideTooltip } = useTooltip();

  const handleMouseEnter = (
    event: React.MouseEvent<SVGRectElement>,
    bar: ComputedBarDatum<Point>
  ) => {
    showTooltipFromEvent(
      <BasicTooltip
        id={`x: ${bar.data.data.x}, y: ${bar.data.data.y}`}
        enableChip={true}
        color={bar.color}
      />,
      event
    );
  };

  const handleMouseLeave = () => {
    hideTooltip();
  };

  return (
    <g transform={`translate(0,-${heightOffset})`}>
      {bars.map((bar) => {
        return (
          <rect
            key={bar.key}
            x={xScaleFunc(Number(bar.data.data.x)) - bar.width / 2}
            y={yScaleFunc(Number(bar.data.data.y))}
            width={bar.width}
            height={bar.height}
            fill={bar.color}
            onMouseEnter={(event) => handleMouseEnter(event, bar)}
            onMouseLeave={handleMouseLeave}
            rx={bar.height > 8 ? 8 : 1}
          />
        );
      })}
    </g>
  );
};

export default CustomBars;
