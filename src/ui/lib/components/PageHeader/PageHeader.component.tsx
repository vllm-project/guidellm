'use client';
import { Box, Typography } from '@mui/material';

import { useGetRunInfoQuery } from '../../store/slices/runInfo';
import { formateDate, getFileSize } from '../../utils/helpers';
import { SpecBadge } from '../SpecBadge';
import { HeaderCell, HeaderWrapper } from './PageHeader.styles';

export const Component = () => {
  const { data } = useGetRunInfoQuery();
  const modelSize = getFileSize(data?.model?.size || 0);

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
          <SpecBadge
            label="Model size"
            value={data?.model?.size ? `${modelSize?.size} ${modelSize?.units}` : '0B'}
            variant="body1"
          />
        </HeaderCell>
        <HeaderCell item xs={5} withDivider>
          <SpecBadge
            label="Dataset"
            value={data?.dataset?.name || 'N/A'}
            variant="caption"
            withTooltip
          />
        </HeaderCell>
        <HeaderCell item xs={2} sx={{ paddingRight: 0 }}>
          <SpecBadge
            label="Time Stamp"
            value={data?.timestamp ? formateDate(data?.timestamp) : 'N/A'}
            variant="caption"
          />
        </HeaderCell>
      </HeaderWrapper>
    </Box>
  );
};
