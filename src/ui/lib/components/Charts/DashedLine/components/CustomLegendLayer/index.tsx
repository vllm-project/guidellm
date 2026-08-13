import { useTheme } from '@mui/material';

import { CustomLegendLayerProps } from './CustomLegendLayer.interfaces';

const LEGEND_HEIGHT = 40;

export const CustomLegendLayer = ({ series, ...rest }: CustomLegendLayerProps) => {
  const theme = useTheme();
  const palette = theme.palette;

  const colors = [
    palette.surface.onSurface,
    palette.secondary.main,
    palette.tertiary.main,
    palette.quarternary.main,
  ];
  const solidColor = palette.primary.main;

  const getColor = (isSolid = false) => {
    if (isSolid) {
      return solidColor;
    }

    if (colors.length === 0) {
      throw new Error('No more colors available');
    }

    return colors.splice(0, 1)[0];
  };
  return (
    <g
      transform={`translate(20, ${(rest?.height || rest.innerHeight) - LEGEND_HEIGHT})`}
    >
      {series.map((item, index) => {
        return (
          <g key={item.id} transform={`translate(${index * 100}, 0)`}>
            <line
              x1="0"
              y1="0"
              x2="20"
              y2="0"
              stroke={getColor(Boolean(item.solid))}
              strokeWidth={2}
              strokeDasharray={item?.solid ? '' : '4,4'}
            />
            <text
              x="30"
              y="0"
              fill={theme.palette.surface.onSurface}
              style={{
                fontSize: theme.typography.caption.fontSize,
                fontWeight: theme.typography.caption.fontWeight,
                fontFamily: theme.typography.caption.fontFamily,
              }}
              alignmentBaseline="middle"
            >
              {item.id}
            </text>
          </g>
        );
      })}
    </g>
  );
};
