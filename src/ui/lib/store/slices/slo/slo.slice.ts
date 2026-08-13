import { createSlice, PayloadAction } from '@reduxjs/toolkit';

import * as Constants from './slo.constants';
import { SloState } from './slo.interfaces';

const sloSlice = createSlice({
  name: Constants.name,
  initialState: Constants.initialState,
  reducers: {
    setSloData: (state, action: PayloadAction<Partial<SloState>>) => {
      if (action.payload.enforcedPercentile !== undefined) {
        state.enforcedPercentile = action.payload.enforcedPercentile;
      }
      if (action.payload.current) {
        state.current = { ...state.current, ...action.payload.current };
      }
      if (action.payload.tasksDefaults) {
        state.tasksDefaults = {
          ...state.tasksDefaults,
          ...action.payload.tasksDefaults,
        };
      }
      if (action.payload.currentRequestRate) {
        state.currentRequestRate = action.payload.currentRequestRate;
      }
    },
    setEnforcedPercentile: (state, action: PayloadAction<string>) => {
      state.enforcedPercentile = action.payload;
    },
    setCurrentRequestRate: (state, action: PayloadAction<number>) => {
      state.currentRequestRate = action.payload;
    },
    setSloValue: (
      state,
      action: PayloadAction<{ metric: keyof SloState['current']; value: number }>
    ) => {
      const { metric, value } = action.payload;
      if (value >= 0) {
        state.current[metric] = value;
      }
    },
  },
});

export const { setCurrentRequestRate, setEnforcedPercentile, setSloData, setSloValue } =
  sloSlice.actions;
export default sloSlice.reducer;
