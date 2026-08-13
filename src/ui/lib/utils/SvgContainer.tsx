import { styled } from '@mui/material';

export const SvgContainer = styled('span')(({ color }) => ({
  path: {
    fill: color,
  },
}));
