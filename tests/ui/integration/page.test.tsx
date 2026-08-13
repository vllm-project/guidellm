import { render, waitFor } from '@testing-library/react';

import Home from '@/app/page';

import { mockBenchmarks } from '../unit/mocks/mockBenchmarks';

const jsonResponse = (data: unknown, status = 200) =>
  Promise.resolve(
    new Response(JSON.stringify(data), {
      status,
      headers: { 'Content-Type': 'application/json' },
    })
  );

const route = (input: RequestInfo) => {
  const url = typeof input === 'string' ? input : input.url;

  if (url.endsWith('/run-info')) return jsonResponse({});
  if (url.endsWith('/workload-details')) return jsonResponse({});
  if (url.endsWith('/benchmarks')) return jsonResponse(mockBenchmarks);

  /* fall-through → 404 */
  return { ok: false, status: 404, json: () => Promise.resolve({}) };
};

beforeEach(() => {
  jest.resetAllMocks();
  (global.fetch as jest.Mock).mockImplementation(route);
});

describe('Home Page', () => {
  it('renders the homepage content', async () => {
    const { getByText } = render(<Home />);
    await waitFor(() => {
      expect(getByText('GuideLLM')).toBeInTheDocument();
    });
  });
});
