import { Box, styled } from '@mui/material';

export const BadgeContainer = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerHigh,
  borderRadius: '6px',
  padding: '4px 6px',
  marginLeft: '12px',
}));
