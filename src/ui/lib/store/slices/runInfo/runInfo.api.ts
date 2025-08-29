import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

import { RunInfo } from './runInfo.interfaces';

const USE_MOCK_API = process.env.NEXT_PUBLIC_USE_MOCK_API === 'true';

const fetchRunInfo = () => {
  return { data: window.runInfo as RunInfo };
};

export const runInfoApi = createApi({
  reducerPath: 'runInfoApi',
  baseQuery: USE_MOCK_API ? fetchRunInfo : fetchBaseQuery({ baseUrl: '/api' }),
  endpoints: (builder) => ({
    getRunInfo: builder.query<RunInfo, void>({
      query: () => 'run-info',
    }),
  }),
});

export const { useGetRunInfoQuery } = runInfoApi;
