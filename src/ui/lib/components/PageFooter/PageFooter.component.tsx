'use client';
import { Box, Link, Typography } from '@mui/material';
import React from 'react';

import { guideLLMLogoLight } from '@assets/icons';

export const Component = () => {
  return (
    <Box
      component="footer"
      sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '16px',
      }}
    >
      <Box sx={{ flexGrow: 1, flexDirection: 'row' }} display="flex">
        <Typography variant="subtitle2" color="surface.onSurfaceSubdued">
          Got questions?
        </Typography>
        <Link onClick={() => alert('Not implemented')} underline="none" ml={1}>
          <Typography variant="subtitle2" color="surface.onSurfaceAccent">
            Send us a message on Slack
          </Typography>
        </Link>
      </Box>
      <Box>
        <img width="150" alt="guidellm logo" src={guideLLMLogoLight.src} />
      </Box>
    </Box>
  );
};
