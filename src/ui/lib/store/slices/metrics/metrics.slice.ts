import { createSlice, PayloadAction } from '@reduxjs/toolkit';

import * as Constants from './metrics.constants';
import { MetricsState } from './metrics.interfaces';

const metricsSlice = createSlice({
  name: Constants.name,
  initialState: Constants.initialState,
  reducers: {
    setMetricsData: (state, action: PayloadAction<MetricsState>) => {
      return { ...state, ...action.payload };
    },
    setSliderRps: (state, action: PayloadAction<number>) => {
      state.currentRequestRate = action.payload;
    },
  },
});

export const { setMetricsData, setSliderRps } = metricsSlice.actions;
export default metricsSlice.reducer;
