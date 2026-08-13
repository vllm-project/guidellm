'use client';

import { Box, Grid } from '@mui/material';
import { useSelector } from 'react-redux';

import {
  selectGenerationsHistogramBarData,
  selectGenerationsHistogramLineData,
  selectPromptsHistogramBarData,
  selectPromptsHistogramLineData,
  selectRequestOverTimeBarData,
  useGetWorkloadDetailsQuery,
} from '../../store/slices/workloadDetails';
import { formatNumber, parseUrlParts } from '../../utils/helpers';
import { BlockHeader } from '../BlockHeader';
import { Carousel } from '../Carousel';
import { DataPanel } from '../DataPanel';
import { RequestOverTime } from '../RequestOverTime';
import { SpecBadge } from '../SpecBadge';
import { TokenLength } from '../TokenLength';

export const Component = () => {
  const { data } = useGetWorkloadDetailsQuery();
  const promptsBarData = useSelector(selectPromptsHistogramBarData);
  const promptsLineData = useSelector(selectPromptsHistogramLineData);
  const generationsBarData = useSelector(selectGenerationsHistogramBarData);
  const generationsLineData = useSelector(selectGenerationsHistogramLineData);
  const { barChartData: requestOverTimeBarData } = useSelector(
    selectRequestOverTimeBarData
  );
  const { type, target, port } = parseUrlParts(data?.server?.target || '');

  return (
    <>
      <BlockHeader label="Workload Details" />
      <Grid
        container
        spacing={0}
        gap={2}
        data-id="workload-details"
        sx={{ marginLeft: 0, flexWrap: 'nowrap' }}
        justifyContent="space-between"
      >
        <DataPanel
          header="Prompt"
          topContainer={
            <Carousel items={data?.prompts?.samples || []} label="Sample Prompt" />
          }
          bottomContainer={
            <TokenLength
              label={'Mean Prompt Length'}
              tokenCount={formatNumber(
                data?.prompts?.tokenDistributions.statistics.mean ?? 0
              )}
              bars={promptsBarData || []}
              lines={promptsLineData}
            />
          }
          key="dp-1"
        />
        <DataPanel
          header="Server"
          topContainer={
            <Box display="flex" flexDirection="column" sx={{ width: '100%' }}>
              <SpecBadge label="Target" value={target} variant="body2" />
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <SpecBadge label="Type" value={type} variant="body2" />
                </Grid>
                <Grid item xs={6}>
                  <SpecBadge label="Port" value={port} variant="body2" />
                </Grid>
              </Grid>
            </Box>
          }
          bottomContainer={
            <RequestOverTime
              benchmarksCount={data?.requestsOverTime?.numBenchmarks || 0}
              barData={requestOverTimeBarData || []}
              rateType={data?.rateType ?? ''}
            />
          }
          key="dp-2"
        />
        <DataPanel
          header="Generated"
          topContainer={
            <Carousel
              items={data?.generations?.samples || []}
              label="Sample Generated"
            />
          }
          bottomContainer={
            <TokenLength
              label={'Mean Generated Length'}
              tokenCount={formatNumber(
                data?.generations?.tokenDistributions.statistics.mean ?? 0
              )}
              bars={generationsBarData || []}
              lines={generationsLineData}
            />
          }
          key="dp-3"
        />
      </Grid>
    </>
  );
};
