import { useTheme } from '@mui/material';

const useLineColors = () => {
  const theme = useTheme();
  const palette = theme.palette;
  return [
    palette.primary.shades.W100,
    palette.secondary.main,
    palette.tertiary.main,
    palette.quarternary.main,
  ];
};

export default useLineColors;
