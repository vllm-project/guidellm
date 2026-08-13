'use client';
import { Grid, useTheme } from '@mui/material';
import { MutableRefObject, useEffect, useRef, useState } from 'react';

import { ArrowDown, ArrowUp } from '@assets/icons';

import { SectionContainerProps } from './SectionContainer.interfaces';
import { Container, RoundedButton } from './SectionContainer.styles';
import { SvgContainer } from '../../utils/SvgContainer';

export const Component = ({ children }: SectionContainerProps) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(false);
  const [showButton, setShowButton] = useState(false);
  const containerRef: MutableRefObject<HTMLDivElement | null> = useRef(null);

  useEffect(() => {
    const checkOverflow = () => {
      const container = containerRef.current;
      if (container) {
        const hasOverflow = container.scrollHeight > container.clientHeight;
        setShowButton(hasOverflow);
      }
    };

    checkOverflow();
    window.addEventListener('resize', checkOverflow);

    return () => {
      window.removeEventListener('resize', checkOverflow);
    };
  }, []);

  return (
    <Container display="flex">
      <Grid
        flexGrow={1}
        flexWrap="wrap"
        direction="row"
        display="flex"
        ref={containerRef}
        sx={{ maxHeight: expanded ? 'none' : '42px', overflow: 'hidden' }}
      >
        {children}
      </Grid>
      <Grid
        sx={{ width: '72px' }}
        justifyContent="flex-end"
        alignItems="flex-start"
        display="flex"
      >
        {showButton && (
          <RoundedButton
            onClick={() => setExpanded(!expanded)}
            sx={{ marginTop: '4px' }}
          >
            <SvgContainer color={theme.palette.primary.onContainer}>
              {expanded ? (
                <SvgContainer color={theme.palette.primary.onContainer}>
                  <ArrowDown />
                </SvgContainer>
              ) : (
                <SvgContainer color={theme.palette.primary.onContainer}>
                  <ArrowUp />
                </SvgContainer>
              )}
            </SvgContainer>
          </RoundedButton>
        )}
      </Grid>
    </Container>
  );
};
