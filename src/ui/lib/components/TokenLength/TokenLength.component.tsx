import { Box, Typography } from '@mui/material';
import { FC } from 'react';

import { TokenLengthProps } from './TokenLength.interfaces';
import { MiniCombined } from '../../components/Charts/MiniCombined';
import ContainerSizeWrapper, {
  ContainerSize,
} from '../../components/Charts/MiniCombined/components/ContainerSizeWrapper';

export const Component: FC<TokenLengthProps> = ({ label, tokenCount, bars, lines }) => (
  <Box
    display="flex"
    flexDirection="column"
    sx={{ width: '100%' }}
    justifyItems="center"
    alignItems="space-between"
  >
    <Box display="flex" flexDirection="column">
      <Box>
        <Typography variant="overline2" color="surface.onSurface">
          {label}
        </Typography>
      </Box>
      <Box mt="8px" mb="16px">
        <Typography variant="metric1" color="primary">
          {`${tokenCount} token${tokenCount !== 1 ? 's' : ''}`}
        </Typography>
      </Box>
    </Box>
    <div style={{ width: '100%', height: '85px' }}>
      <ContainerSizeWrapper>
        {(containerSize: ContainerSize) => (
          <MiniCombined
            bars={bars}
            lines={lines}
            width={312}
            height={85}
            xLegend="length (tokens)"
            margins={{ bottom: 30 }}
            containerSize={containerSize}
          />
        )}
      </ContainerSizeWrapper>
    </div>
  </Box>
);
