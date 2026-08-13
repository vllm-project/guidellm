import { Box, Typography } from '@mui/material';
import React, { FC } from 'react';

import { useColor } from '@/lib/hooks/useColor';

import { InputProps } from './Input.interfaces';
import { StyledTextField, InputContainer, ErrorMessage } from './Input.styles';

export const Component: FC<InputProps> = ({
  label,
  prefix,
  disabled = false,
  value,
  onChange,
  fullWidth = false,
  fontColor,
  error,
  isNumber = false,
}) => {
  const selectedColor = useColor(fontColor);
  return (
    <Box>
      <InputContainer
        className={disabled ? 'disabled' : ''}
        fullWidth={fullWidth}
        data-id="input-wrapper"
      >
        <Typography variant="overline2" color="surface.onSurface">
          {label}
        </Typography>
        <Box display="flex" alignItems="baseline">
          {prefix && (
            <Typography
              variant="body1"
              color="surface.onSurfaceAccent"
              sx={{ marginRight: 1 }}
            >
              {prefix}
            </Typography>
          )}
          <StyledTextField
            fontColor={selectedColor}
            disabled={disabled}
            placeholder={label}
            variant="standard"
            type="number"
            value={value}
            onChange={onChange}
            {...(isNumber && {
              inputProps: {
                type: 'number',
                step: '1',
                min: '0',
              },
            })}
          />
        </Box>
      </InputContainer>
      <ErrorMessage variant="caption" color="error.main">
        {error || ' '}
      </ErrorMessage>
    </Box>
  );
};
