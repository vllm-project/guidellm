import { Box, Typography } from '@mui/material';

import { Badge } from '../Badge';
import { MeanMetricSummaryProps } from './MeanMetricSummary.interfaces';

export const Component = ({
  meanValue,
  meanUnit,
  rpsValue,
}: MeanMetricSummaryProps) => {
  return (
    <Box flexDirection="column">
      <Box flexDirection="row" display="flex" alignItems="center" gap={'12px'}>
        <Typography
          variant="overline1"
          color="surface.onSurfaceSubdued"
          textTransform="uppercase"
        >
          Mean At
        </Typography>
        <Badge label={`${rpsValue} rps`} />
      </Box>
      <Box flexDirection="row" display="flex">
        <Typography variant="h4" color="surface.onSurfaceAccent" pt={1}>
          {meanValue} {meanUnit}
        </Typography>
      </Box>
    </Box>
  );
};
