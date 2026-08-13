import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

import { WorkloadDetails } from './workloadDetails.interfaces';

const USE_MOCK_API = process.env.NEXT_PUBLIC_USE_MOCK_API === 'true';

const fetchWorkloadDetails = () => {
  return { data: window.workloadDetails as WorkloadDetails };
};

export const workloadDetailsApi = createApi({
  reducerPath: 'workloadDetailsApi',
  baseQuery: USE_MOCK_API ? fetchWorkloadDetails : fetchBaseQuery({ baseUrl: '/api' }),
  endpoints: (builder) => ({
    getWorkloadDetails: builder.query<WorkloadDetails, void>({
      query: () => 'workload-details',
    }),
  }),
});

export const { useGetWorkloadDetailsQuery } = workloadDetailsApi;
