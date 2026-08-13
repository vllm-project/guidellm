import { Box, Typography, useTheme } from '@mui/material';
import dynamic from 'next/dynamic';
import { FC } from 'react';

import { CarouselProps } from './Carousel.interfaces';
import { PromptWrapper } from './Carousel.styles';

const Carousel = dynamic(
  () => import('react-material-ui-carousel').then((mod) => mod.default),
  {
    ssr: false,
  }
);

function truncateString(str: string, limit: number) {
  if (str.length <= limit) {
    return str;
  }
  let cutIndex = limit;
  while (
    cutIndex < str.length &&
    (str[cutIndex] !== ' ' || /[.,!?]/.test(str[cutIndex - 1]))
  ) {
    cutIndex++;
  }
  return cutIndex < str.length ? str.slice(0, cutIndex) + '...' : str;
}

export const Component: FC<CarouselProps> = ({ label, items }) => {
  const theme = useTheme();
  return (
    <Box sx={{ width: '100%' }}>
      <Typography
        variant="overline2"
        color="surface.onSurface"
        textTransform="uppercase"
      >
        {label}
      </Typography>
      <Carousel
        interval={10000}
        duration={1000}
        animation="fade"
        IndicatorIcon={null}
        navButtonsAlwaysInvisible={true}
        sx={{ marginTop: '6px' }}
      >
        {items.map((item, i) => (
          <PromptWrapper key={i} data-id="prompt-wrapper">
            <Typography variant="body2" color={theme.palette.primary.main}>
              {truncateString(item, 200)}
            </Typography>
          </PromptWrapper>
        ))}
      </Carousel>
    </Box>
  );
};
