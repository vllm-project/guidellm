import { createSlice, PayloadAction } from '@reduxjs/toolkit';

import { benchmarksApi } from './benchmarks.api';
import * as Constants from './benchmarks.constants';
import { Benchmarks } from './benchmarks.interfaces';

interface BenchmarksState {
  data: Benchmarks | null;
}

const initialState: BenchmarksState = {
  data: null,
};

const benchmarksSlice = createSlice({
  name: Constants.name,
  initialState,
  reducers: {
    setBenchmarks: (state, action: PayloadAction<Benchmarks>) => {
      state.data = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      benchmarksApi.endpoints.getBenchmarks.matchFulfilled,
      (state, action) => {
        state.data = action.payload;
      }
    );
  },
});

export const { setBenchmarks } = benchmarksSlice.actions;
export default benchmarksSlice.reducer;
