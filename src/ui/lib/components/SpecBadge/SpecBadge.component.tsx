import { Box, Typography, Tooltip } from '@mui/material';
import { FC } from 'react';

import { SpecBadgeProps } from './SpecBadge.interfaces';
import { Container, EllipsisTypography, ValueWrapper } from './SpecBadge.styles';

export const Component: FC<SpecBadgeProps> = ({
  label,
  value,
  variant,
  additionalValue,
  withTooltip = false,
}) => {
  const tooltipContent = (
    <EllipsisTypography variant={variant} color="primary">
      {value}
    </EllipsisTypography>
  );
  return (
    <Container>
      <Typography variant="overline2" color="surface.onSurface">
        {label}
      </Typography>
      <ValueWrapper>
        {withTooltip ? (
          <Tooltip title={value}>{tooltipContent}</Tooltip>
        ) : (
          tooltipContent
        )}
        {additionalValue && <Box ml={1}>{additionalValue}</Box>}
      </ValueWrapper>
    </Container>
  );
};
