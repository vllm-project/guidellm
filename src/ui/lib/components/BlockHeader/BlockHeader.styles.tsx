import { Divider, styled } from '@mui/material';

export const CustomDivider = styled(Divider)(({ theme }) => ({
  height: '1px',
  flex: 1,
  marginLeft: '48px',
  backgroundColor: theme.palette.outline.subdued,
}));
