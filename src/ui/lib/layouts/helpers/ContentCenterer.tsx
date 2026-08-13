import { ReactNode } from 'react';

import { useCurrentBreakpoint } from './useCurrentBreakpoint';

type HasChildren = {
  children: ReactNode;
};

/**
 * Helper function that returns appropriate padding based on breakpoint.
 *
 * @param breakpoint - The current MUI breakpoint name
 * @returns Padding value as a CSS string
 */
const getPaddingForBreakpoint = (breakpoint: string) =>
  breakpoint === 'xs' ? '16px' : '32px';

/**
 * Component that centers content with responsive padding and max-width constraints.
 *
 * Automatically adjusts padding and max-width based on the current screen size
 * using MUI breakpoints for optimal display across devices.
 *
 * @param children - React nodes to be centered and constrained
 */
export const ContentCenterer = ({ children }: HasChildren) => {
  const breakpoint = useCurrentBreakpoint();
  const padding = getPaddingForBreakpoint(breakpoint);
  const maxWidth = breakpoint === 'sm' ? '600px' : '1440px';
  return (
    <div
      style={{
        margin: '0 auto',
        padding: `0 ${padding}`,
        maxWidth: maxWidth,
        boxSizing: 'border-box',
        height: '100%',
        width: '100%',
      }}
    >
      {children}
    </div>
  );
};
