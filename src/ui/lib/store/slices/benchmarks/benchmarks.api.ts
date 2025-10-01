import { ThunkDispatch, UnknownAction } from '@reduxjs/toolkit';
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

import { Benchmarks, Statistics } from './benchmarks.interfaces';
import { formatNumber } from '../../../utils/helpers';
import { defaultPercentile } from '../slo/slo.constants';
import { setSloData } from '../slo/slo.slice';

const USE_MOCK_API = process.env.NEXT_PUBLIC_USE_MOCK_API === 'true';

// currently the injector requires 'window.benchmarks = {};' to be present in the html, but benchmarks is expected to be null or an array
const fetchBenchmarks = () => {
  let benchmarks = window.benchmarks;
  if (!Array.isArray(benchmarks)) {
    benchmarks = [];
  }
  return { data: benchmarks as Benchmarks };
};

const getAverageValueForPercentile = (
  firstMetric: Statistics,
  lastMetric: Statistics,
  percentile: string
) => {
  const firstPercentile = firstMetric?.percentileRows.find(
    (p) => p.percentile === percentile
  );
  const lastPercentile = lastMetric?.percentileRows.find(
    (p) => p.percentile === percentile
  );
  return ((firstPercentile?.value ?? 0) + (lastPercentile?.value ?? 0)) / 2;
};

const setDefaultSLOs = (
  data: Benchmarks,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dispatch: ThunkDispatch<any, any, UnknownAction>
) => {
  // temporarily set default slo values, long term the backend should set default slos that will not just be the avg at the default percentile
  const firstBM = data[0];
  const lastBM = data[data.length - 1];

  const ttftAvg = getAverageValueForPercentile(
    firstBM?.ttft,
    lastBM?.ttft,
    defaultPercentile
  );
  const itlAvg = getAverageValueForPercentile(
    firstBM?.itl,
    lastBM?.itl,
    defaultPercentile
  );
  const timePerRequestAvg = getAverageValueForPercentile(
    firstBM?.timePerRequest,
    lastBM?.timePerRequest,
    defaultPercentile
  );
  const throughputAvg = getAverageValueForPercentile(
    firstBM?.throughput,
    lastBM?.throughput,
    defaultPercentile
  );

  dispatch(
    setSloData({
      currentRequestRate: firstBM?.requestsPerSecond,
      current: {
        ttft: formatNumber(ttftAvg, 0),
        itl: formatNumber(itlAvg, 0),
        timePerRequest: formatNumber(timePerRequestAvg, 0),
        throughput: formatNumber(throughputAvg, 0),
      },
      tasksDefaults: {
        ttft: formatNumber(ttftAvg, 0),
        itl: formatNumber(itlAvg, 0),
        timePerRequest: formatNumber(timePerRequestAvg, 0),
        throughput: formatNumber(throughputAvg, 0),
      },
    })
  );
};

export const benchmarksApi = createApi({
  reducerPath: 'benchmarksApi',
  baseQuery: USE_MOCK_API ? fetchBenchmarks : fetchBaseQuery({ baseUrl: '/api' }),
  endpoints: (builder) => ({
    getBenchmarks: builder.query<Benchmarks, void>({
      query: () => 'benchmarks',
      async onQueryStarted(_, { dispatch, queryFulfilled }) {
        try {
          const { data } = await queryFulfilled;
          setDefaultSLOs(data, dispatch);
        } catch (err) {
          console.error('Failed to fetch benchmarks:', err);
        }
      },
    }),
  }),
});

export const { useGetBenchmarksQuery } = benchmarksApi;
