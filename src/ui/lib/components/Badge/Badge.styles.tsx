import { styled, Typography } from '@mui/material';

export const StyledTypography = styled(Typography)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerHigh,
  borderRadius: '6px',
  padding: '6px',
  marginRight: '6px',
  display: 'inline-block',
  paddingTop: '8px',
  color: theme.palette.surface.onSurfaceAccent,
}));
