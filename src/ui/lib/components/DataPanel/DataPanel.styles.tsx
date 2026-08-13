import { Grid, styled } from '@mui/material';

import { HeaderContainer as HeaderContainerBase } from '../MetricsContainer/MetricsContainer.styles';

export const InnerContainer = styled(Grid)(({ theme }) => ({
  borderWidth: '1px',
  borderStyle: 'solid',
  borderColor: theme.palette.outline.subdued,
  borderRadius: '8px',
  overflow: 'hidden',
  backgroundColor: theme.palette.surface.surfaceContainerLow,
}));

export const TopCell = styled(Grid)({
  display: 'flex',
  alignItems: 'flex-start',
  height: '180px',
  padding: '24px !important',
});

export const BottomCell = styled(Grid)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  minHeight: '220px',
  height: '180px',
  borderTopWidth: '1px',
  borderTopStyle: 'solid',
  borderTopColor: theme.palette.outline.subdued,
  padding: '24px !important',
}));

export const HeaderContainer = styled(HeaderContainerBase)({
  display: 'flex',
  justifyContent: 'flex-start',
});
