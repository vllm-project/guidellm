'use client';
import { Box, Typography } from '@mui/material';

import { useGetRunInfoQuery } from '../../store/slices/runInfo';
import { formateDate } from '../../utils/helpers';
import { SpecBadge } from '../SpecBadge';
import { HeaderCell, HeaderWrapper } from './PageHeader.styles';

export const Component = () => {
  const { data } = useGetRunInfoQuery();
  return (
    <Box py={2}>
      <Typography variant="subtitle2" color="surface.onSurfaceAccent">
        GuideLLM
      </Typography>
      <Typography variant="h4" color="surface.onSurface" my={'12px'}>
        Workload Report
      </Typography>
      <HeaderWrapper container>
        <HeaderCell item xs={5} withDivider sx={{ paddingLeft: 0 }}>
          <SpecBadge
            label="Model"
            value={data?.model?.name || 'N/A'}
            variant="metric2"
            withTooltip
          />
        </HeaderCell>
        <HeaderCell item xs={2} sx={{ paddingRight: 0 }}>
          <SpecBadge
            label="Time Stamp"
            value={data?.timestamp ? formateDate(data?.timestamp) : 'n/a'}
            variant="caption"
          />
        </HeaderCell>
      </HeaderWrapper>
    </Box>
  );
};
