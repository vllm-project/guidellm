import { useTheme } from '@mui/material';

import { CustomTickProps } from './CustomTick.interfaces';

function CustomTick({
  isXAxis,
  tick,
  scale,
  withTicks = false,
  isFirst,
  isLast,
}: CustomTickProps) {
  const theme = useTheme();

  function getGroupPosition() {
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
  }

  function renderTick() {
    const commonProps = {
      fontFamily: theme.typography.axisLabel.fontFamily,
      fontWeight: theme.typography.axisLabel.fontWeight,
      fontSize: theme.typography.axisLabel.fontSize,
      fill: theme.palette.surface.onSurface,
    };

    return isXAxis ? (
      <>
        {withTicks && (
          <line x1={0} y1={0} x2={0} y2={6} stroke={theme.palette.surface.onSurface} />
        )}
        <text textAnchor="middle" y={10} {...commonProps}>
          {tick}
        </text>
      </>
    ) : (
      <>
        {withTicks && (
          <line x1={0} y1={0} x2={-6} y2={0} stroke={theme.palette.surface.onSurface} />
        )}
        <text textAnchor="end" x={-5} dominantBaseline="middle" {...commonProps}>
          {tick}
        </text>
      </>
    );
  }

  const { x, y } = getGroupPosition();

  return (
    <g key={tick} transform={`translate(${x}, ${y})`}>
      {renderTick()}
    </g>
  );
}

export default CustomTick;
