import { Variant } from '@mui/material/styles/createTypography';
import { TypographyPropsVariantOverrides } from '@mui/material/Typography/Typography';
import { OverridableStringUnion } from '@mui/types';
import { ReactNode } from 'react';

export interface SpecBadgeProps {
  label: string;
  value: string;
  variant: OverridableStringUnion<Variant | 'inherit', TypographyPropsVariantOverrides>;
  additionalValue?: ReactNode;
  withTooltip?: boolean;
}
