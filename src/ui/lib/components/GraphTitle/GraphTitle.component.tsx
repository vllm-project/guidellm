import { Typography, useTheme } from '@mui/material';

import { GraphTitleProps } from './GraphTitle.interfaces';

export const Component = ({ title }: GraphTitleProps) => {
  const theme = useTheme();
  return (
    <Typography
      variant="subtitle2"
      color={theme.palette.surface.onSurfaceSubdued}
      mb={2}
    >
      {title}
    </Typography>
  );
};
