import { createSelector } from '@reduxjs/toolkit';

import { Point } from '@/lib/components/Charts/common/interfaces';

import { BenchmarkMetrics, PercentileValues } from './benchmarks.interfaces';
import { PercentileItem } from '../../../components/DistributionPercentiles';
import { formatNumber } from '../../../utils/helpers';
import { createMonotoneSpline } from '../../../utils/interpolation';
import { RootState } from '../../index';
import { selectSloState } from '../slo/slo.selectors';

export const selectBenchmarks = (state: RootState) => state.benchmarks.data;

export const selectMetricsSummaryLineData = createSelector(
  [selectBenchmarks, selectSloState],
  (benchmarks, sloState) => {
    const sortedByRPS = benchmarks
      ?.slice()
      ?.sort((bm1, bm2) => (bm1.requestsPerSecond > bm2.requestsPerSecond ? 1 : -1));
    const selectedPercentile = sloState.enforcedPercentile;

    const lineData: { [K in keyof BenchmarkMetrics]: Point[] } = {
      ttft: [],
      tpot: [],
      timePerRequest: [],
      throughput: [],
    };
    const metrics: (keyof BenchmarkMetrics)[] = [
      'ttft',
      'tpot',
      'timePerRequest',
      'throughput',
    ];
    metrics.forEach((metric) => {
      const data: Point[] = [];
      sortedByRPS?.forEach((benchmark) => {
        const percentile = benchmark[metric].percentileRows.find(
          (p) => p.percentile === selectedPercentile
        );
        data.push({
          x: benchmark.requestsPerSecond,
          y: percentile?.value ?? 0,
        });
      });

      lineData[metric] = data;
    });
    return lineData;
  }
);

const getDefaultMetricValues = () => ({
  enforcedPercentileValue: 0,
  mean: 0,
  percentiles: [],
});

export const selectInterpolatedMetrics = createSelector(
  [selectBenchmarks, selectSloState],
  (benchmarks, sloState) => {
    const metricData: {
      [K in keyof BenchmarkMetrics | 'mean']: {
        enforcedPercentileValue: number;
        mean: number;
        percentiles: PercentileItem[];
      };
    } = {
      ttft: getDefaultMetricValues(),
      tpot: getDefaultMetricValues(),
      timePerRequest: getDefaultMetricValues(),
      throughput: getDefaultMetricValues(),
      mean: getDefaultMetricValues(),
    };
    if ((benchmarks?.length || 0) < 2) {
      return metricData;
    }
    const sortedByRPS = benchmarks
      ?.slice()
      ?.sort((bm1, bm2) => (bm1.requestsPerSecond > bm2.requestsPerSecond ? 1 : -1));
    const requestRates = sortedByRPS?.map((bm) => bm.requestsPerSecond) || [];
    const { enforcedPercentile, currentRequestRate } = sloState;
    const metrics: (keyof BenchmarkMetrics)[] = [
      'ttft',
      'tpot',
      'timePerRequest',
      'throughput',
    ];
    if (!sortedByRPS || sortedByRPS.length === 0) {
      return metricData;
    }
    const invalidRps =
      currentRequestRate < sortedByRPS[0].requestsPerSecond ||
      currentRequestRate > sortedByRPS[sortedByRPS.length - 1].requestsPerSecond;
    if (invalidRps) {
      return metricData;
    }
    metrics.forEach((metric) => {
      const meanValues = sortedByRPS.map((bm) => bm[metric].mean);
      const interpolateMeanAt = createMonotoneSpline(requestRates, meanValues);
      const interpolatedMeanValue: number = interpolateMeanAt(currentRequestRate) || 0;
      const percentiles: PercentileValues[] = ['p50', 'p90', 'p95', 'p99'];
      const valuesByPercentile = percentiles.map((p) => {
        const bmValuesAtP = sortedByRPS.map((bm) => {
          const result = bm[metric].percentiles[p] || 0;
          return result;
        });
        const interpolateValueAtP = createMonotoneSpline(requestRates, bmValuesAtP);
        const interpolatedValueAtP = formatNumber(
          interpolateValueAtP(currentRequestRate)
        );
        return { label: p, value: `${interpolatedValueAtP}` } as PercentileItem;
      });
      const interpolatedPercentileValue =
        Number(valuesByPercentile.find((p) => p.label === enforcedPercentile)?.value) ||
        0;
      metricData[metric] = {
        enforcedPercentileValue: interpolatedPercentileValue,
        mean: interpolatedMeanValue,
        percentiles: valuesByPercentile,
      };
    });
    return metricData;
  }
);

export const selectMetricsDetailsLineData = createSelector(
  [selectBenchmarks],
  (benchmarks) => {
    const sortedByRPS =
      benchmarks
        ?.slice()
        ?.sort((bm1, bm2) =>
          bm1.requestsPerSecond > bm2.requestsPerSecond ? 1 : -1
        ) || [];

    const lineData: {
      [K in keyof BenchmarkMetrics]: { data: Point[]; id: string; solid?: boolean }[];
    } = {
      ttft: [],
      tpot: [],
      timePerRequest: [],
      throughput: [],
    };
    const props: (keyof BenchmarkMetrics)[] = [
      'ttft',
      'tpot',
      'timePerRequest',
      'throughput',
    ];
    props.forEach((prop) => {
      if (sortedByRPS.length === 0) {
        return;
      }
      const data: { [key: string]: { data: Point[]; id: string; solid?: boolean } } =
        {};
      sortedByRPS[0].ttft.percentileRows.forEach((p) => {
        data[p.percentile] = { data: [], id: p.percentile };
      });
      data.mean = { data: [], id: 'mean', solid: true };
      sortedByRPS?.forEach((benchmark) => {
        const rps = benchmark.requestsPerSecond;
        benchmark[prop].percentileRows.forEach((p) => {
          data[p.percentile].data.push({ x: rps, y: p.value });
        });
        const mean = benchmark[prop].mean;
        data.mean.data.push({ x: rps, y: mean });
      });
      lineData[prop] = Object.keys(data).map((key) => {
        return data[key];
      });
    });
    return lineData;
  }
);
