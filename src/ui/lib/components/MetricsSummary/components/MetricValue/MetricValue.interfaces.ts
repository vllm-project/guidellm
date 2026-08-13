import { LineColor } from '@/lib/components/Charts/MetricLine';

export interface MetricValueProps {
  label: string;
  value: string;
  match: boolean;
  valueColor: LineColor;
}
