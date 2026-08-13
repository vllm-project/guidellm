import { RootState } from '../../index';

export const selectRunInfo = (state: RootState) => state.runInfo.data;
