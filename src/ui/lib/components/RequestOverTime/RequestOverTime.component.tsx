import { Box, Grid, Typography } from '@mui/material';
import { FC } from 'react';

import { RequestOverTimeProps } from './RequestOverTime.interfaces';
import { Badge } from '../../components/Badge';
import { SpecBadge } from '../../components/SpecBadge';
import { MiniCombined } from '../Charts/MiniCombined';
import ContainerSizeWrapper, {
  ContainerSize,
} from '../Charts/MiniCombined/components/ContainerSizeWrapper';

export const Component: FC<RequestOverTimeProps> = ({
  benchmarksCount,
  barData,
  rateType,
  lines = [],
}) => (
  <Box
    display="flex"
    flexDirection="column"
    sx={{ width: '100%' }}
    justifyItems="center"
  >
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <SpecBadge
          label="Number of Benchmarks"
          value={benchmarksCount?.toString()}
          variant="body2"
        />
      </Grid>
      <Grid item>
        <Typography
          variant="overline2"
          color="surface.onSurface"
          textTransform="uppercase"
        >
          Rate Type
        </Typography>
      </Grid>
      <Grid item>
        <Badge label={rateType} />
      </Grid>
    </Grid>
    <div style={{ width: '100%', height: '85px' }}>
      <ContainerSizeWrapper>
        {(containerSize: ContainerSize) => (
          <MiniCombined
            bars={barData}
            lines={lines}
            width={312}
            height={85}
            xLegend="time (sec)"
            margins={{ bottom: 30 }}
            containerSize={containerSize}
          />
        )}
      </ContainerSizeWrapper>
    </div>
  </Box>
);
