import { createSelector } from '@reduxjs/toolkit';

import { formatNumber } from '../../../utils/helpers';
import { RootState } from '../../index';

export const selectWorkloadDetails = (state: RootState) => state.workloadDetails.data;

export const selectPromptsHistogramBarData = createSelector(
  [selectWorkloadDetails],
  (workloadDetails) => {
    return workloadDetails?.prompts?.tokenDistributions.buckets.map((bucket) => ({
      x: formatNumber(bucket.value),
      y: bucket.count,
    }));
  }
);

export const selectGenerationsHistogramBarData = createSelector(
  [selectWorkloadDetails],
  (workloadDetails) => {
    return workloadDetails?.generations?.tokenDistributions.buckets.map((bucket) => ({
      x: formatNumber(bucket.value),
      y: bucket.count,
    }));
  }
);

export const selectPromptsHistogramLineData = createSelector(
  [selectWorkloadDetails],
  (workloadDetails) => [
    {
      x: formatNumber(
        workloadDetails?.prompts?.tokenDistributions.statistics.mean ?? 0
      ),
      y: 35,
      id: 'mean',
    },
    {
      x: formatNumber(
        workloadDetails?.prompts?.tokenDistributions.statistics.median ?? 0
      ),
      y: 35,
      id: 'median',
    },
  ]
);

export const selectGenerationsHistogramLineData = createSelector(
  [selectWorkloadDetails],
  (workloadDetails) => [
    {
      x: formatNumber(
        workloadDetails?.generations?.tokenDistributions.statistics.mean ?? 0
      ),
      y: 35,
      id: 'mean',
    },
    {
      x: formatNumber(
        workloadDetails?.generations?.tokenDistributions.statistics.median ?? 0
      ),
      y: 35,
      id: 'median',
    },
  ]
);

export const selectRequestOverTimeBarData = createSelector(
  [selectWorkloadDetails],
  (workloadDetails) => {
    const requestObjs = workloadDetails?.requestsOverTime?.requestsOverTime;
    return {
      barChartData: requestObjs?.buckets?.map((bucket) => ({
        x: formatNumber(bucket.value),
        y: bucket.count,
      })),
    };
  }
);
