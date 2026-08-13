import { Grid, Typography } from '@mui/material';

import { SectionProps } from './Section.interfaces';
import { ValueCell } from './Section.styles';

export const Component = ({ label, value }: SectionProps) => {
  return (
    <Grid container display="flex" flexDirection="row" m="6px" sx={{ width: 'auto' }}>
      <Grid item alignContent="center">
        <Typography variant="subtitle2" color="surface.onSurfaceSubdued">
          {label}
        </Typography>
      </Grid>
      <ValueCell alignContent="center" item>
        {value}
      </ValueCell>
    </Grid>
  );
};
