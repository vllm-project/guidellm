import { Grid, Typography } from '@mui/material';

import { RowContainerProps } from './RowContainer.interfaces';

export const Component = ({ label, children }: RowContainerProps) => {
  return (
    <Grid container alignItems="center">
      <Grid item xs={2}>
        <Typography variant="h6" color="surface.onSurface">
          {label}
        </Typography>
      </Grid>
      <Grid item xs={10}>
        {children}
      </Grid>
    </Grid>
  );
};
