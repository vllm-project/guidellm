'use client';
import { Box, Typography, useTheme } from '@mui/material';

import { SvgContainer } from '@/lib/utils/SvgContainer';
import { Info } from '@assets/icons';

import { BlockHeaderProps } from './BlockHeader.interfaces';
import { CustomDivider } from './BlockHeader.styles';

export const Component = ({ label, withDivider = false }: BlockHeaderProps) => {
  const theme = useTheme();
  return (
    <Box display="flex" alignItems="center" my={3}>
      <Typography variant="h5" color="surface.onSurface">
        {label}
      </Typography>
      <Box ml={2}>
        <SvgContainer color={theme.palette.surface.onSurfaceAccent}>
          <Info />
        </SvgContainer>
      </Box>
      {withDivider && <CustomDivider />}
    </Box>
  );
};
