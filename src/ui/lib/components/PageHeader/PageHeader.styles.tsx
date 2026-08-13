import { Grid, styled } from '@mui/material';

import { HeaderCellProps } from './PageHeader.interfaces';

export const HeaderWrapper = styled(Grid)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerLow,
  padding: '0 16px',
  borderRadius: '8px',
}));

export const HeaderCell = styled(Grid, {
  shouldForwardProp: (propName) => propName !== 'withDivider',
})<HeaderCellProps>(({ theme, withDivider }) => ({
  overflow: 'hidden',
  padding: '8px',
  ...(withDivider && {
    borderRight: `1px solid ${theme.palette.outline.subdued}`,
  }),
}));
