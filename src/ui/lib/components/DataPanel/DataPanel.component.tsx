import { Typography } from '@mui/material';

import { DataPanelProps } from './DataPanel.interfaces';
import {
  BottomCell,
  InnerContainer,
  TopCell,
  HeaderContainer,
} from './DataPanel.styles';

export const Component = ({
  header,
  topContainer,
  bottomContainer,
}: DataPanelProps) => {
  return (
    <InnerContainer item xs={4}>
      <HeaderContainer item xs={12}>
        <Typography variant="overline1" color="surface.onSurface">
          {header}
        </Typography>
      </HeaderContainer>

      <TopCell item xs={12} data-id="top-cell">
        {topContainer}
      </TopCell>

      {bottomContainer && (
        <BottomCell item xs={12} data-id="bottom-cell">
          {bottomContainer}
        </BottomCell>
      )}
    </InnerContainer>
  );
};
