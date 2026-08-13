import { Box, styled } from '@mui/material';

export const PromptWrapper = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.primary.container,
  borderRadius: '8px',
  padding: '8px',
  minWidth: '304px',
  minHeight: '84px',
  width: 'auto',
  overflow: 'hidden',
}));
