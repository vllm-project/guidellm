import { SelectChangeEvent } from '@mui/material';
import { ChangeEvent, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import { Point } from '@/lib/components/Charts/common/interfaces';

import { selectMetricsSummaryLineData } from '../../store/slices/benchmarks';
import { selectSloState } from '../../store/slices/slo/slo.selectors';
import { setEnforcedPercentile, setSloValue } from '../../store/slices/slo/slo.slice';
import { ceil, floor } from '../../utils/helpers';

type Errors = { [key: string]: string | undefined };

const initErrorsState: Errors = {
  ttft: undefined,
  itl: undefined,
  timePerRequest: undefined,
  throughput: undefined,
};

const findMinMax = (lineData: Point[]) => {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < lineData.length; i++) {
    const { x, y } = lineData[i];
    if (x < minX) {
      minX = ceil(x, 2);
    }
    if (x > maxX) {
      maxX = floor(x, 2);
    }
    if (y < minY) {
      minY = ceil(y, 2);
    }
    if (y > maxY) {
      maxY = floor(y, 2);
    }
  }

  return { minX, maxX, minY, maxY };
};

export const useSummary = () => {
  const dispatch = useDispatch();

  const { current, enforcedPercentile, tasksDefaults } = useSelector(selectSloState);
  const { ttft, itl, timePerRequest, throughput } = useSelector(
    selectMetricsSummaryLineData
  );

  const [errors, setErrors] = useState<Errors>(initErrorsState);

  const ttftLimits = findMinMax(ttft || []);
  const itlLimits = findMinMax(itl || []);
  const timePerRequestLimits = findMinMax(timePerRequest || []);
  const throughputLimits = findMinMax(throughput || []);

  const limitsByMetric = {
    ttft: ttftLimits,
    itl: itlLimits,
    timePerRequest: timePerRequestLimits,
    throughput: throughputLimits,
  };

  const validateInput = (field: keyof typeof current, value?: number) => {
    let error: string | undefined;
    const limits = limitsByMetric[field];
    if (value === undefined) {
      error = 'Invalid value';
    } else if (value > limits.maxY) {
      error = 'Error: Larger than maximum';
    } else if (value < limits.minY) {
      error = 'Error: Smaller than minimum';
    }
    setErrors((prev) => ({ ...prev, [field]: error }));
  };

  const sanitizeInput = (value: string) => {
    const sanitizedValue = value.replace(/\D/g, '');
    return sanitizedValue !== '' ? Number(sanitizedValue) : undefined;
  };

  const handleChange =
    (metric: keyof typeof current) => (event: ChangeEvent<HTMLInputElement>) => {
      const newValue = sanitizeInput(event.target.value);
      validateInput(metric, newValue);
      if (newValue !== undefined) {
        dispatch(setSloValue({ metric, value: newValue }));
      }
    };

  const handlePercentileChange = (event: SelectChangeEvent<unknown>) => {
    // TODO: need to validate slos on percentile change
    const newValue = `${event.target.value as string}`;
    dispatch(setEnforcedPercentile(newValue));
  };

  const handleReset = () => {
    Object.entries(tasksDefaults).forEach(([metric, value]) => {
      dispatch(setSloValue({ metric: metric as keyof typeof current, value }));
    });
    setErrors(initErrorsState);
  };

  return {
    ...current,
    percentile: enforcedPercentile,
    minX: ttftLimits.minX,
    maxX: ttftLimits.maxX,
    errors,
    handleTtft: handleChange('ttft'),
    handleItl: handleChange('itl'),
    handleTimePerRequest: handleChange('timePerRequest'),
    handleThroughput: handleChange('throughput'),
    handlePercentileChange,
    handleReset,
  };
};
