import { Box, Button, Typography, useTheme } from '@mui/material';
import React, { ElementType } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { MetricLine, LineColor } from '@/lib/components/Charts/MetricLine';
import { selectSloState } from '@/lib/store/slices/slo/slo.selectors';
import { setCurrentRequestRate } from '@/lib/store/slices/slo/slo.slice';
import { Expand } from '@assets/icons';

import {
  selectInterpolatedMetrics,
  selectMetricsSummaryLineData,
  useGetBenchmarksQuery,
} from '../../store/slices/benchmarks';
import { selectRunInfo } from '../../store/slices/runInfo';
import { formatNumber } from '../../utils/helpers';
import { BlockHeader } from '../BlockHeader';
import { Input } from '../Input';
import { MetricValue } from './components/MetricValue';
import {
  CustomSlider,
  FieldCell,
  FieldsContainer,
  FooterLeftCell,
  FooterRightCell,
  GraphContainer,
  HeaderLeftCell,
  HeaderRightCell,
  MetricsSummaryContainer,
  MiddleColumn,
  InputContainer,
  StyledSelect,
  StyledInputLabel,
  OptionItem,
  StyledFormControl,
} from './MetricsSummary.styles';
import { useSummary } from './useSummary';

const percentileOptions = ['p50', 'p90', 'p95', 'p99'];

export const Component = () => {
  const theme = useTheme();
  const dispatch = useDispatch();
  const { data } = useGetBenchmarksQuery();

  const lineDataByRps = useSelector(selectMetricsSummaryLineData);
  const interpolatedMetricData = useSelector(selectInterpolatedMetrics);
  const runInfo = useSelector(selectRunInfo);

  const { currentRequestRate } = useSelector(selectSloState);
  const handleSliderChange = (event: Event, newValue: number | number[]) => {
    dispatch(setCurrentRequestRate(newValue as number));
  };

  const {
    ttft: ttftSLO,
    tpot: tpotSLO,
    timePerRequest: timePerRequestSLO,
    throughput: throughputSLO,
    percentile,
    minX,
    maxX,
    errors,
    handleTtft,
    handleTpot,
    handleTimePerRequest,
    handleThroughput,
    handlePercentileChange,
    handleReset,
  } = useSummary();

  const isTtftMatch = Boolean(
    ttftSLO && interpolatedMetricData.ttft.enforcedPercentileValue <= ttftSLO
  );
  const isTpotMatch = Boolean(
    tpotSLO && interpolatedMetricData.tpot.enforcedPercentileValue <= tpotSLO
  );
  const isTprMatch = Boolean(
    timePerRequestSLO &&
      interpolatedMetricData.timePerRequest.enforcedPercentileValue <= timePerRequestSLO
  );
  const isThroughputMatch = Boolean(
    throughputSLO &&
      interpolatedMetricData.throughput.enforcedPercentileValue >= throughputSLO
  );

  const sliderMarks = [
    {
      value: minX,
      label: minX,
    },
    {
      value: maxX,
      label: maxX,
    },
  ];

  if ((data?.length ?? 0) <= 1) {
    return <></>;
  }

  return (
    <>
      <BlockHeader label="Metrics Summary" />
      <MetricsSummaryContainer container>
        <HeaderLeftCell item xs={9}>
          <Box display="flex" flexDirection="row" justifyContent="space-between">
            <Typography variant="h6" color="surface.onSurface" mb={2}>
              Service Level Objectives for{' '}
              <span style={{ color: theme.palette.surface.onSurfaceAccent }}>
                [{runInfo?.task || 'N/A'}]
              </span>
              :
            </Typography>

            <Button onClick={handleReset}>
              <Typography variant="button" color="surface.onSurfaceAccent" mb={2}>
                RESET TO DEFAULT
              </Typography>
            </Button>
          </Box>

          <FieldsContainer data-id="fields-container">
            <FieldCell data-id="field-cell-1">
              <Input
                label="TTFT (ms)"
                value={ttftSLO}
                onChange={handleTtft}
                fullWidth
                fontColor={LineColor.Primary}
                error={errors?.ttft}
              />
            </FieldCell>
            <FieldCell data-id="field-cell-2">
              <Input
                label="TPOT (ms)"
                value={tpotSLO}
                onChange={handleTpot}
                fullWidth
                fontColor={LineColor.Secondary}
                error={errors?.tpot}
              />
            </FieldCell>
            <FieldCell data-id="field-cell-3">
              <Input
                label="TIME PER REQUEST (s)"
                value={timePerRequestSLO}
                onChange={handleTimePerRequest}
                fullWidth
                fontColor={LineColor.Tertiary}
                error={errors?.timePerRequest}
              />
            </FieldCell>
            <FieldCell data-id="field-cell-4">
              <Input
                label="THROUGHPUT (tok/s)"
                value={throughputSLO}
                onChange={handleThroughput}
                fullWidth
                fontColor={LineColor.Quarternary}
                error={errors?.throughput}
              />
            </FieldCell>
          </FieldsContainer>
        </HeaderLeftCell>
        <HeaderRightCell item xs={3}>
          <Typography variant="h6" color="surface.onSurface" mb={'23px'}>
            Observed Values at:
          </Typography>

          <InputContainer>
            <StyledFormControl
              fullWidth
              sx={{ fieldset: { legend: { maxWidth: '100%' } } }}
            >
              <StyledInputLabel shrink={true}>percentile</StyledInputLabel>
              <StyledSelect
                value={percentile}
                onChange={handlePercentileChange}
                autoWidth
                placeholder="Select percentile"
                IconComponent={Expand as ElementType}
                displayEmpty
                MenuProps={{
                  PaperProps: {
                    sx: {
                      backgroundColor: theme.palette.surface.surfaceContainerLow,
                      borderRadius: '8px',
                    },
                  },
                }}
              >
                {percentileOptions.map((value) => (
                  <OptionItem key={value} value={value} sx={{ minWidth: '168px' }}>
                    {value}
                  </OptionItem>
                ))}
              </StyledSelect>
            </StyledFormControl>
          </InputContainer>
        </HeaderRightCell>
        {/* graphs */}

        <MiddleColumn sx={{ paddingLeft: '0px !important' }} item xs={9}>
          <GraphContainer>
            <MetricLine
              data={[{ id: 'ttft', data: lineDataByRps.ttft || [] }]}
              threshold={ttftSLO}
              lineColor={LineColor.Primary}
            />
          </GraphContainer>
        </MiddleColumn>
        <MiddleColumn item xs={3}>
          <MetricValue
            label="TTFT"
            value={`${formatNumber(interpolatedMetricData.ttft.enforcedPercentileValue)} ms`}
            match={isTtftMatch}
            valueColor={LineColor.Primary}
          />
        </MiddleColumn>

        <MiddleColumn sx={{ paddingLeft: '0px !important' }} item xs={9}>
          <GraphContainer>
            <MetricLine
              data={[{ id: 'tpot', data: lineDataByRps.tpot || [] }]}
              threshold={tpotSLO}
              lineColor={LineColor.Secondary}
            />
          </GraphContainer>
        </MiddleColumn>
        <MiddleColumn item xs={3}>
          <MetricValue
            label="TPOT"
            value={`${formatNumber(interpolatedMetricData.tpot.enforcedPercentileValue)} ms`}
            match={isTpotMatch}
            valueColor={LineColor.Secondary}
          />
        </MiddleColumn>

        <MiddleColumn sx={{ paddingLeft: '0px !important' }} item xs={9}>
          <GraphContainer>
            <MetricLine
              data={[
                { id: 'time per request', data: lineDataByRps.timePerRequest || [] },
              ]}
              threshold={timePerRequestSLO}
              lineColor={LineColor.Tertiary}
            />
          </GraphContainer>
        </MiddleColumn>
        <MiddleColumn item xs={3}>
          <MetricValue
            label="time per request"
            value={`${formatNumber(
              interpolatedMetricData.timePerRequest.enforcedPercentileValue
            )} s`}
            match={isTprMatch}
            valueColor={LineColor.Tertiary}
          />
        </MiddleColumn>

        <MiddleColumn sx={{ paddingLeft: '0px !important' }} item xs={9}>
          <GraphContainer>
            <MetricLine
              data={[{ id: 'throughput', data: lineDataByRps.throughput || [] }]}
              threshold={throughputSLO}
              lineColor={LineColor.Quarternary}
            />
          </GraphContainer>
        </MiddleColumn>
        <MiddleColumn item xs={3}>
          <MetricValue
            value={`${formatNumber(
              interpolatedMetricData.throughput.enforcedPercentileValue
            )} tok/s`}
            label="throughput"
            match={isThroughputMatch}
            valueColor={LineColor.Quarternary}
          />
        </MiddleColumn>

        {/* slider */}
        <FooterLeftCell item xs={9}>
          <CustomSlider
            size="medium"
            step={0.01}
            value={formatNumber(currentRequestRate, 2)}
            min={minX}
            max={maxX}
            marks={sliderMarks}
            valueLabelDisplay="on"
            onChange={handleSliderChange}
          />
        </FooterLeftCell>
        <FooterRightCell item xs={3}>
          <Typography
            variant="overline1"
            color="surface.onSurface"
            textTransform="uppercase"
          >
            Maximum RPS per gpu
          </Typography>
          <Typography variant="metric1" color="primary">
            {formatNumber(currentRequestRate)} rps
          </Typography>
        </FooterRightCell>
      </MetricsSummaryContainer>
    </>
  );
};
