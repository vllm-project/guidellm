import { Box, Typography } from '@mui/material';

import { Badge } from '../Badge';
import { Section } from '../Section';
import { DistributionPercentilesProps } from './DistributionPercentiles.interfaces';

export const Component = ({ list, rpsValue, units }: DistributionPercentilesProps) => {
  return (
    <Box>
      <Box flexDirection="row" display="flex" alignItems="center" gap={'12px'}>
        <Typography
          variant="overline1"
          color="surface.onSurfaceSubdued"
          textTransform="uppercase"
        >
          Distribution At
        </Typography>
        <Badge label={`${rpsValue} rps`} />
      </Box>

      {list.map((item) => (
        <Section key={item.label} label={item.label} value={`${item.value} ${units}`} />
      ))}
    </Box>
  );
};
