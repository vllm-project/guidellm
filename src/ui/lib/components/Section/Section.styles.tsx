import { Grid, styled } from '@mui/material';

export const ValueCell = styled(Grid)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerHigh,
  color: theme.palette.surface.onSurface,
  borderRadius: '6px',
  padding: '6px',
  gap: 10,
  marginLeft: '12px',
}));
