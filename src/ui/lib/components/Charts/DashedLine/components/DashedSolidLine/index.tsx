import { useTheme } from '@mui/material';

import { DashedSolidLineProps } from './DashedSolidLine.interfaces';
import { toNumberValue } from '../../helpers';

export const DashedSolidLine = ({
  series,
  lineGenerator,
  xScale,
  yScale,
}: DashedSolidLineProps) => {
  const theme = useTheme();
  const palette = theme.palette;

  const colors = [
    palette.surface.onSurface,
    palette.secondary.main,
    palette.tertiary.main,
    palette.quarternary.main,
  ];
  const solidColor = palette.primary.main;

  const getColor = (isSolid: boolean) => {
    if (isSolid) {
      return solidColor;
    }

    if (colors.length === 0) {
      throw new Error('No more colors available');
    }

    return colors.splice(0, 1)[0];
  };

  return series.map(({ id, data, solid = false }) => {
    return (
      <g key={id}>
        <path
          key={id}
          d={(() => {
            const linePath = lineGenerator(
              data.map((d) => ({
                x: xScale(toNumberValue(d.data.x)),
                y: yScale(toNumberValue(d.data.y)),
              }))
            );
            return linePath !== null ? linePath : undefined;
          })()}
          fill="none"
          stroke={getColor(Boolean(solid))}
          style={
            !solid
              ? {
                  strokeDasharray: '2.4 2.4',
                  strokeWidth: 1.5,
                }
              : {
                  strokeWidth: 1.5,
                }
          }
        />
        {solid &&
          data.map((d, pointIndex) => (
            <circle
              key={`${id}-${pointIndex}`}
              cx={xScale(toNumberValue(d.data.x))}
              cy={yScale(toNumberValue(d.data.y))}
              r={4}
              fill={getColor(Boolean(solid))}
            />
          ))}
      </g>
    );
  });
};
