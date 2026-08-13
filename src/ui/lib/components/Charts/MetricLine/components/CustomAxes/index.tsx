import { useTheme } from '@mui/material';
import React from 'react';

import { CustomLineLayerProps } from './CustomAxes.interfaces';

const CustomAxes = ({ yScale }: CustomLineLayerProps) => {
  const theme = useTheme();
  const minY2 = yScale.domain()[0];
  const maxY2 = yScale.domain()[1];
  return (
    <>
      <line
        x1={0}
        x2={0}
        y1={yScale(minY2)}
        y2={yScale(maxY2)}
        stroke={theme.palette.surface.onSurfaceSubdued}
      />
    </>
  );
};

export default CustomAxes;
