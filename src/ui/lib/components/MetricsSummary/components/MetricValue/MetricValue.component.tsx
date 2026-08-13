import { Box, Typography } from '@mui/material';
import { FC } from 'react';

import { useColor } from '@/lib/hooks/useColor';
import { CheckCircle, WarningCircle } from '@assets/icons';

import { MetricValueProps } from './MetricValue.interfaces';

export const Component: FC<MetricValueProps> = ({
  label,
  value,
  match,
  valueColor,
}) => {
  const selectedColor = useColor(valueColor);
  return (
    <Box display="flex" flexDirection="column" alignItems="flex-end" gap={1}>
      <Box display="flex" flexDirection="row">
        <Typography
          variant="overline2"
          color="surface.onSurface"
          textTransform="uppercase"
        >
          {label}
        </Typography>
        <Typography variant="overline2" color="surface.onSurfaceSubdued" pl={0.5}>
          (observed)
        </Typography>
      </Box>
      <Typography variant="metric1" color={selectedColor}>
        {value}
      </Typography>
      {match ? <CheckCircle /> : <WarningCircle />}
    </Box>
  );
};
