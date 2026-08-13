import { ChangeEvent } from 'react';

import { LineColor } from '@/lib/components/Charts/MetricLine';

export interface InputProps {
  label: string;
  prefix?: string;
  disabled?: boolean;
  value: number | undefined;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  fullWidth?: boolean;
  fontColor?: LineColor;
  error?: string | undefined;
  isNumber?: boolean;
}

export type CustomInputContainer = {
  fullWidth?: boolean;
};

export type CustomInputProps = {
  fontColor: string;
};
