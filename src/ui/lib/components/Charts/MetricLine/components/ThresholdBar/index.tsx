import { useTheme } from '@mui/material';
import React, { FC } from 'react';

import { ThresholdBarProps } from './ThresholdBar.interfaces';

const ThresholdBar: FC<ThresholdBarProps> = ({
  xScale,
  yScale,
  threshold,
  minX,
  maxX,
  minY,
  maxY,
}) => {
  const theme = useTheme();

  if (threshold === undefined || threshold <= 0 || threshold > maxY) {
    return null;
  }

  const x0 = xScale(minX);
  const y0 = yScale(minY);
  const xMax = xScale(maxX);
  const yThreshold = yScale(threshold);

  if ([x0, y0, xMax, yThreshold].some((value) => value === undefined)) {
    return null;
  }

  return (
    <g>
      <line
        x1={x0}
        y1={yThreshold}
        x2={xMax - x0}
        y2={yThreshold}
        stroke={theme.palette.outline.main}
        strokeWidth="1.5"
        strokeDasharray="2.4 2.4"
      />
      <rect
        id="threshold"
        x={x0}
        y={yThreshold}
        width={xMax - x0}
        height={y0 - yThreshold}
        fill={theme.palette.surface.surfaceContainerHighest}
      />
    </g>
  );
};

export default ThresholdBar;
