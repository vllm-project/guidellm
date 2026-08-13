import { useTheme } from '@mui/material';

import { formatNumber } from '@/lib/utils/helpers';

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

  if (isXAxis && !isFirst && !isLast) {
    return null;
  }

  const getGroupPosition = () => {
    if (isXAxis) {
      let x = scale(tick);
      if (isFirst) {
        x = 4;
      }
      if (isLast) {
        x -= 4;
      }
      return { x, y: 0 };
    }
    return { x: 0, y: scale(tick) };
  };

  const renderTickContent = (textAnchor: 'middle' | 'end', x: number, y: number) => (
    <>
      {withTicks && (
        <line x1={0} y1={0} x2={x} y2={y} stroke={theme.palette.surface.onSurface} />
      )}
      <text
        textAnchor={textAnchor}
        x={x === 0 ? undefined : x}
        y={y === 0 ? undefined : y}
        dominantBaseline={isXAxis ? undefined : 'middle'}
        fontFamily={theme.typography.axisLabel.fontFamily}
        fontWeight={theme.typography.axisLabel.fontWeight}
        fontSize={theme.typography.axisLabel.fontSize}
        fill={theme.palette.surface.onSurface}
      >
        {formatNumber(tick)}
      </text>
    </>
  );

  const { x, y } = getGroupPosition();

  return (
    <g key={tick} transform={`translate(${x}, ${y})`}>
      {isXAxis ? renderTickContent('middle', 0, 10) : renderTickContent('end', -5, 0)}
    </g>
  );
};

export default CustomTick;
