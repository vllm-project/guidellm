import { createSlice, PayloadAction } from '@reduxjs/toolkit';

import { workloadDetailsApi } from './workloadDetails.api';
import * as Constants from './workloadDetails.constants';
import { WorkloadDetails } from './workloadDetails.interfaces';

interface WorkloadDetailsState {
  data: WorkloadDetails | null;
}

const initialState: WorkloadDetailsState = {
  data: null,
};

const workloadDetailsSlice = createSlice({
  name: Constants.name,
  initialState,
  reducers: {
    setWorkloadDetails: (state, action: PayloadAction<WorkloadDetails>) => {
      state.data = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(
      workloadDetailsApi.endpoints.getWorkloadDetails.matchFulfilled,
      (state, action) => {
        state.data = action.payload;
      }
    );
  },
});

export const { setWorkloadDetails } = workloadDetailsSlice.actions;
export default workloadDetailsSlice.reducer;
