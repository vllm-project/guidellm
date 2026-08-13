import { useTheme } from '@mui/material';

import { CustomGridProps } from './CustomGrid.interfaces';

const CustomGrid = ({
  xScale,
  yScale,
  width,
  height,
  xTicks,
  yTicks,
  scaledHeight,
  fullGrid = false,
}: CustomGridProps) => {
  const theme = useTheme();
  // const xTick = xTicks[0];
  const yTick = yTicks[yTicks.length - 1];

  const renderAxlesOnly = (
    <>
      {/*<line*/}
      {/*  key={`x${xTick}`}*/}
      {/*  x1={xScale(xTick)}*/}
      {/*  x2={xScale(xTick)}*/}
      {/*  y1={scaledHeight}*/}
      {/*  y2={height}*/}
      {/*  stroke={theme.palette.outline.subdued}*/}
      {/*/>*/}
      <line
        key={`y${yTick}`}
        x1={0}
        x2={width}
        y1={yScale(yTick)}
        y2={yScale(yTick)}
        stroke={theme.palette.outline.subdued}
      />
    </>
  );

  const renderFullGrid = (
    <>
      {xTicks.map((tick) => (
        <line
          key={`x${tick}`}
          x1={xScale(tick)}
          x2={xScale(tick)}
          y1={scaledHeight}
          y2={height}
          stroke={theme.palette.outline.subdued}
        />
      ))}
      {yTicks.map((tick) => (
        <line
          key={`y${tick}`}
          x1={0}
          x2={width}
          y1={yScale(tick)}
          y2={yScale(tick)}
          stroke={theme.palette.outline.subdued}
        />
      ))}
    </>
  );

  return (
    <g transform={`translate(0,-${scaledHeight})`} id="grid">
      {fullGrid ? renderFullGrid : renderAxlesOnly}
    </g>
  );
};

export default CustomGrid;
