import { FC } from 'react';

import { BadgeProps } from './Badge.interfaces';
import { StyledTypography } from './Badge.styles';

export const Component: FC<BadgeProps> = ({ label }) => {
  return <StyledTypography variant="body2">{label}</StyledTypography>;
};
