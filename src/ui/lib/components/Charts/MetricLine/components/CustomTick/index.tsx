import { useTheme } from '@mui/material';

import { CustomTickProps } from './CustomTick.interfaces';

const CustomTick = ({
  isXAxis,
  tick,
  scale,
  withTicks = false,
  isFirst,
  isLast,
}: CustomTickProps) => {
  const theme = useTheme();

  const getGroupPosition = () => {
    if (!isXAxis) {
      return { x: 0, y: scale(tick) };
    }

    let x = scale(tick);
    if (isFirst) {
      x = 4;
    }
    if (isLast) {
      x -= 4;
    }
    return { x, y: 0 };
  };

  const renderTickContent = (
    textAnchor: 'middle' | 'end',
    x: number,
    y: number,
    lineX: number,
    lineY: number
  ) => (
    <>
      {withTicks && (
        <line
          x1={0}
          y1={0}
          x2={lineX}
          y2={lineY}
          stroke={theme.palette.surface.onSurface}
        />
      )}
      <text
        textAnchor={textAnchor}
        x={x}
        y={y}
        dominantBaseline={isXAxis ? undefined : 'middle'}
        fontFamily={theme.typography.axisLabel.fontFamily}
        fontWeight={theme.typography.axisLabel.fontWeight}
        fontSize={theme.typography.axisLabel.fontSize}
        fill={theme.palette.surface.onSurface}
      >
        {tick}
      </text>
    </>
  );

  const { x, y } = getGroupPosition();

  return (
    <g key={tick} transform={`translate(${x}, ${y})`}>
      {isXAxis
        ? renderTickContent('middle', 0, 10, 0, 6)
        : renderTickContent('end', -5, 0, -6, 0)}
    </g>
  );
};

export default CustomTick;
