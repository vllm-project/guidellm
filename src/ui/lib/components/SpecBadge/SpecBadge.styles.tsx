import { Box, styled, Typography } from '@mui/material';

export const EllipsisTypography = styled(Typography)({
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
});

export const Container = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  marginBottom: '8px',
});

export const ValueWrapper = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  maxWidth: '100%',
  width: 'auto',
});
