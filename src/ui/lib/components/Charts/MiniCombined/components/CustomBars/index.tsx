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
  const minX =
    bars.length === 1
      ? 0
      : Math.min(...bars.map((bar) => xScaleFunc(bar.data.data.x || 0)));
  const maxX = Math.max(...bars.map((bar) => xScaleFunc(bar.data.data.x || 0)));
  const handleMouseEnter = (
    event: React.MouseEvent<SVGPathElement>,
    bar: ComputedBarDatum<Point>
  ) => {
    const x = xScaleFunc(bar.data.data.x || 0) || 0;
    const normalizedPosition = (x - minX) / (maxX - minX);
    const maxPadding = 80;

    let paddingLeft = 0;
    let paddingRight = 0;
    // log scale padding so that tooltip doesn't get cut off by the edges
    if (normalizedPosition < 0.5) {
      const logFactor = -Math.log(2 * normalizedPosition + 0.01) / Math.log(100);
      paddingLeft = Math.min(maxPadding, Math.max(0, logFactor * maxPadding));
    } else {
      const logFactor = -Math.log(2 * (1 - normalizedPosition) + 0.01) / Math.log(100);
      paddingRight = Math.min(maxPadding, Math.max(0, logFactor * maxPadding));
    }
    showTooltipFromEvent(
      <div
        style={{
          paddingLeft: `${paddingLeft}px`,
          paddingRight: `${paddingRight}px`,
        }}
      >
        <BasicTooltip
          id={`x: ${bar.data.data.x || 0}, y: ${bar.data.data.y}`}
          enableChip={true}
          color={bar.color}
        />
      </div>,
      event
    );
  };

  const handleMouseLeave = () => {
    hideTooltip();
  };

  return (
    <g transform={`translate(0,-${heightOffset})`}>
      {bars.map((bar) => {
        const barWidth = Math.min(10, bar.width);
        const maxRadius = Math.floor(barWidth / 2);
        const x = xScaleFunc(Number(bar.data.data.x || 0)) - barWidth / 2;
        const y = yScaleFunc(Number(bar.data.data.y));
        let r;
        if (bar.height < 4) {
          r = Math.min(1, maxRadius);
        } else if (bar.height < barWidth) {
          r = Math.min(Math.floor(bar.height / 2), maxRadius);
        } else {
          r = maxRadius;
        }
        r = Math.min(r, Math.floor(bar.height / 2));
        const v = Math.max(0, bar.height - r);
        const path = `M ${x + r},${y} h ${
          barWidth - 2 * r
        } a ${r},${r} 0 0 1 ${r},${r} v ${v} h -${barWidth} v -${v} a ${r},${r} 0 0 1 ${r},-${r} z`;

        return (
          <path
            key={bar.key}
            d={path}
            fill={bar.color}
            onMouseEnter={(event) => handleMouseEnter(event, bar)}
            onMouseLeave={handleMouseLeave}
          />
        );
      })}
    </g>
  );
};

export default CustomBars;
