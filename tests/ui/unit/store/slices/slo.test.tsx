import { initialState } from '@/lib/store/slices/slo/slo.constants';
import { SloState } from '@/lib/store/slices/slo/slo.interfaces';
import sloReducer, { setSloData } from '@/lib/store/slices/slo/slo.slice';

test('should handle initial state', () => {
  expect(sloReducer(undefined, { type: '' })).toEqual(initialState);
});

test('should set slo data', () => {
  const slo = {
    enforcedPercentile: 'p50',
  } as Partial<SloState>;

  const fullSlos = {
    ...initialState,
    ...slo,
  };
  expect(sloReducer(undefined, setSloData(slo))).toEqual(fullSlos);
});
