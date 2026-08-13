import { Grid, styled } from '@mui/material';

export const Container = styled(Grid)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerLow,
  borderRadius: '6px',
  padding: '12px',
  width: 'auto',
  margin: '8px',
}));

export const RoundedButton = styled('div')(({ theme }) => ({
  backgroundColor: theme.palette.primary.shades.B80,
  width: '32px',
  height: '32px',
  borderRadius: '16px',
  justifyContent: 'center',
  alignItems: 'center',
  display: 'flex',
  cursor: 'pointer',
  userSelect: 'none',
}));
