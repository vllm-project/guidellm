import { Grid, styled } from '@mui/material';

export const InnerContainer = styled(Grid)(({ theme }) => ({
  borderWidth: '1px',
  borderStyle: 'solid',
  borderColor: theme.palette.outline.subdued,
  borderRadius: '8px',
  overflow: 'hidden',
  backgroundColor: theme.palette.surface.surfaceContainerLow,
  marginLeft: 0,
  marginTop: 0,
}));

export const HeaderContainer = styled(Grid)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainer,
  borderBottomWidth: '1px',
  borderBottomStyle: 'solid',
  borderBottomColor: theme.palette.outline.subdued,
  padding: '18px 24px',
  margin: 0,
}));

export const RightColumn = styled(Grid)(({ theme }) => ({
  borderLeftWidth: '1px',
  borderLeftStyle: 'solid',
  borderLeftColor: theme.palette.outline.subdued,
}));

export const MainContainer = styled(Grid)(({ theme }) => ({
  borderBottomWidth: '1px',
  borderBottomStyle: 'solid',
  borderBottomColor: theme.palette.outline.subdued,
}));
