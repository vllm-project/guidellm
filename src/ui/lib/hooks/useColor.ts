import { useTheme } from '@mui/material';

import { LineColor } from '@/lib/components/Charts/MetricLine';

export const useColor = (colorType: LineColor | undefined) => {
  const theme = useTheme();
  switch (colorType) {
    case LineColor.Primary:
      return theme.palette.primary.main;
    case LineColor.Secondary:
      return theme.palette.secondary.main;
    case LineColor.Tertiary:
      return theme.palette.tertiary.main;
    case LineColor.Quarternary:
      return theme.palette.quarternary.main;
    default:
      return theme.palette.surface.onSurfaceAccent;
  }
};
