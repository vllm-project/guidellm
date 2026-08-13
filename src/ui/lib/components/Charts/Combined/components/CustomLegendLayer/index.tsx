import { useTheme } from '@mui/material';

import { CustomLegendLayerProps } from './CustomLegendLayer.interfaces';
import useLineColors from '../../../common/useLineColors';

const LEGEND_HEIGHT = 20;

export const CustomLegendLayer = ({ series, ...rest }: CustomLegendLayerProps) => {
  const theme = useTheme();
  const lineColor = useLineColors();
  return (
    <g transform={`translate(20, ${rest.height - LEGEND_HEIGHT})`}>
      {series.map((item, index) => (
        <g key={item.id} transform={`translate(${index * 100}, 0)`}>
          <line
            x1="0"
            y1="0"
            x2="20"
            y2="0"
            stroke={lineColor[index]}
            strokeWidth={2}
            strokeDasharray={'4,4'}
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
      ))}
    </g>
  );
};
