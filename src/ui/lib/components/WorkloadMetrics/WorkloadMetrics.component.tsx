'use client';
import { Box } from '@mui/material';
import { useSelector } from 'react-redux';

import { DashedLine } from '../../components/Charts/DashedLine';
import {
  DistributionPercentiles,
  PercentileItem,
} from '../../components/DistributionPercentiles';
import { MeanMetricSummary } from '../../components/MeanMetricSummary';
import {
  selectInterpolatedMetrics,
  selectMetricsDetailsLineData,
  useGetBenchmarksQuery,
} from '../../store/slices/benchmarks';
import { selectSloState } from '../../store/slices/slo/slo.selectors';
import { formatNumber } from '../../utils/helpers';
import { BlockHeader } from '../BlockHeader';
import { GraphTitle } from '../GraphTitle';
import { MetricsContainer } from '../MetricsContainer';
import { GraphsWrapper } from './WorkloadMetrics.styles';

export const columnContent = (
  rpsValue: number,
  percentiles: PercentileItem[],
  units: string
) => <DistributionPercentiles list={percentiles} rpsValue={rpsValue} units={units} />;

export const leftColumn = (rpsValue: number, value: number, units: string) => (
  <MeanMetricSummary meanValue={`${value}`} meanUnit={units} rpsValue={rpsValue} />
);

export const leftColumn3 = (rpsValue: number, value: number, units: string) => (
  <MeanMetricSummary meanValue={`${value}`} meanUnit={units} rpsValue={rpsValue} />
);

export const Component = () => {
  const { data } = useGetBenchmarksQuery();
  const { ttft, itl, timePerRequest, throughput } = useSelector(
    selectMetricsDetailsLineData
  );
  const { currentRequestRate } = useSelector(selectSloState);
  const formattedRequestRate = formatNumber(currentRequestRate);
  const {
    ttft: ttftAtRPS,
    itl: itlAtRPS,
    timePerRequest: timePerRequestAtRPS,
    throughput: throughputAtRPS,
  } = useSelector(selectInterpolatedMetrics);

  const minX = Math.floor(Math.min(...(data?.map((bm) => bm.requestsPerSecond) || [])));
  if ((data?.length ?? 0) <= 1) {
    return <></>;
  }
  return (
    <>
      <BlockHeader label="Metrics Details" />
      <Box display="flex" flexDirection="row" gap={3} mt={3}>
        <MetricsContainer
          header="TTFT"
          leftColumn={leftColumn(
            formattedRequestRate,
            formatNumber(ttftAtRPS.mean),
            'ms'
          )}
          rightColumn={columnContent(formattedRequestRate, ttftAtRPS.percentiles, 'ms')}
        >
          <GraphTitle title="TTFS vs RPS" />
          <GraphsWrapper>
            <DashedLine
              data={ttft}
              margins={{ left: 50, bottom: 50 }}
              xLegend="request per sec"
              yLegend="ttft (ms)"
              minX={minX}
            />
          </GraphsWrapper>
        </MetricsContainer>
        <MetricsContainer
          header="ITL"
          leftColumn={leftColumn3(
            formattedRequestRate,
            formatNumber(itlAtRPS.mean),
            'ms'
          )}
          rightColumn={columnContent(formattedRequestRate, itlAtRPS.percentiles, 'ms')}
        >
          <GraphTitle title="ITL vs RPS" />
          <GraphsWrapper>
            <DashedLine
              data={itl}
              margins={{ left: 50, bottom: 50 }}
              xLegend="request per sec"
              yLegend="itl (ms)"
              minX={minX}
            />
          </GraphsWrapper>
        </MetricsContainer>
      </Box>
      <Box display="flex" flexDirection="row" gap={3} mt={3}>
        <MetricsContainer
          header="E2E Latency"
          leftColumn={leftColumn(
            formattedRequestRate,
            formatNumber(timePerRequestAtRPS.mean),
            's'
          )}
          rightColumn={columnContent(
            formattedRequestRate,
            timePerRequestAtRPS.percentiles,
            's'
          )}
        >
          <GraphTitle title="E2E Latency vs RPS" />
          <GraphsWrapper>
            <DashedLine
              data={timePerRequest}
              margins={{ left: 50, bottom: 50 }}
              xLegend="request per sec"
              yLegend="latency (s)"
              minX={minX}
            />
          </GraphsWrapper>
        </MetricsContainer>
        <MetricsContainer
          header="Throughput"
          leftColumn={leftColumn3(
            formattedRequestRate,
            formatNumber(throughputAtRPS.mean),
            'tok/s'
          )}
        >
          <GraphTitle title="Throughput vs RPS" />
          <GraphsWrapper>
            <DashedLine
              data={throughput}
              margins={{ left: 50, bottom: 50 }}
              xLegend="request per sec"
              yLegend="throughput (tok/s)"
              minX={minX}
            />
          </GraphsWrapper>
        </MetricsContainer>
      </Box>
    </>
  );
};
