import { configureStore } from '@reduxjs/toolkit';

import { benchmarksApi, benchmarksReducer } from './slices/benchmarks';
import metricsReducer from './slices/metrics/metrics.slice';
import { runInfoApi, infoReducer } from './slices/runInfo';
import sloReducer from './slices/slo/slo.slice';
import { workloadDetailsApi, workloadDetailsReducer } from './slices/workloadDetails';

export const store = configureStore({
  reducer: {
    metrics: metricsReducer,
    slo: sloReducer,
    runInfo: infoReducer,
    [runInfoApi.reducerPath]: runInfoApi.reducer,
    benchmarks: benchmarksReducer,
    [benchmarksApi.reducerPath]: benchmarksApi.reducer,
    workloadDetails: workloadDetailsReducer,
    [workloadDetailsApi.reducerPath]: workloadDetailsApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(
      runInfoApi.middleware,
      benchmarksApi.middleware,
      workloadDetailsApi.middleware
    ),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
