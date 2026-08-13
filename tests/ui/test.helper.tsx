import { ThemeProvider } from '@mui/material/styles';
import React, { ReactNode } from 'react';

import { muiThemeV3Dark } from '@/app/theme';
import { ReduxProvider } from '@/lib/store/provider';

interface TestProvidersProps {
  children: ReactNode;
}

export const MockedWrapper = ({ children }: TestProvidersProps) => {
  return (
    <ReduxProvider>
      <ThemeProvider theme={muiThemeV3Dark}>{children}</ThemeProvider>
    </ReduxProvider>
  );
};
