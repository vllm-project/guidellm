import { useMediaQuery, useTheme } from '@mui/material';

/**
 * Custom hook that returns the current MUI breakpoint based on screen width.
 *
 * Uses Material-UI's breakpoint system to determine which breakpoint matches
 * the current viewport size.
 *
 * @returns {'xs' | 'sm' | 'md' | 'lg' | 'xl'} The current breakpoint name
 */
export function useCurrentBreakpoint() {
  const theme = useTheme();
  const isXs = useMediaQuery(theme.breakpoints.only('xs'));
  const isSm = useMediaQuery(theme.breakpoints.only('sm'));
  const isMd = useMediaQuery(theme.breakpoints.only('md'));
  const isLg = useMediaQuery(theme.breakpoints.only('lg'));
  const isXl = useMediaQuery(theme.breakpoints.only('xl'));

  if (isXs) {
    return 'xs';
  } else if (isSm) {
    return 'sm';
  } else if (isMd) {
    return 'md';
  } else if (isLg) {
    return 'lg';
  } else if (isXl) {
    return 'xl';
  }
  return 'xs';
}
