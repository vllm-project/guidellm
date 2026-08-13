import { Grid, Typography } from '@mui/material';

import { MetricsContainerProps } from './MetricsContainer.interfaces';
import {
  HeaderContainer,
  InnerContainer,
  MainContainer,
  RightColumn,
} from './MetricsContainer.styles';

export const Component = ({
  header,
  leftColumn,
  rightColumn,
  children,
}: MetricsContainerProps) => {
  return (
    <InnerContainer container spacing={2}>
      <HeaderContainer
        xs={12}
        item
        flexDirection="row"
        display="flex"
        justifyContent="space-between"
      >
        <Typography variant="overline1" color="surface.onSurface">
          {header}
        </Typography>
      </HeaderContainer>

      <MainContainer
        xs={12}
        item
        padding="24px"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        display="flex"
      >
        {children}
      </MainContainer>

      {/* middle containers  */}
      <Grid
        xs={rightColumn ? 6 : 12}
        item
        display="flex"
        alignItems="center"
        padding="24px"
        minHeight="232px"
      >
        {leftColumn}
      </Grid>
      {rightColumn && (
        <RightColumn
          xs={6}
          item
          display="flex"
          alignItems="center"
          padding="24px"
          minHeight="232px"
        >
          {rightColumn}
        </RightColumn>
      )}
    </InnerContainer>
  );
};
