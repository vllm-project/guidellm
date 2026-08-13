import { createSlice, PayloadAction } from '@reduxjs/toolkit';

import { runInfoApi } from './runInfo.api';
import * as Constants from './runInfo.constants';
import { RunInfo } from './runInfo.interfaces';

interface RunInfoState {
  data: RunInfo | null;
}

const initialState: RunInfoState = {
  data: null,
};

const runInfoSlice = createSlice({
  name: Constants.name,
  initialState,
  reducers: {
    setRunInfo: (state, action: PayloadAction<RunInfo>) => {
      state.data = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      runInfoApi.endpoints.getRunInfo.matchFulfilled,
      (state, action) => {
        state.data = action.payload;
      }
    );
  },
});

export const { setRunInfo } = runInfoSlice.actions;
export default runInfoSlice.reducer;
