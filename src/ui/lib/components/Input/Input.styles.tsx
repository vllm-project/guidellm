import { Box, styled, TextField, Typography } from '@mui/material';

import { CustomInputContainer, CustomInputProps } from './Input.interfaces';

export const StyledTextField = styled(TextField, {
  shouldForwardProp: (propName) => propName !== 'fontColor',
})<CustomInputProps>(({ theme, fontColor }) => ({
  '& .MuiInputBase-input': {
    color: fontColor,
    fontSize: theme.typography.body1.fontSize,
    fontWeight: theme.typography.body1.fontWeight,
    padding: '0',
  },
  '& .MuiInput-underline:before': {
    borderBottomColor: 'transparent',
  },
  '& .MuiInput-underline:hover:not(.Mui-disabled):before': {
    borderBottomColor: 'transparent',
  },
  '& .MuiInput-underline:after': {
    borderBottomColor: 'transparent',
  },
  '& input[type=number]': {
    MozAppearance: 'textfield',
  },
  '& input[type=number]::-webkit-outer-spin-button': {
    WebkitAppearance: 'none',
    margin: 0,
  },
  '& input[type=number]::-webkit-inner-spin-button': {
    WebkitAppearance: 'none',
    margin: 0,
  },
  '& .Mui-disabled': {
    color: theme.palette.surface.onSurfaceSubdued + ' !important',
    WebkitTextFillColor: theme.palette.surface.onSurfaceSubdued + ' !important',
  },
}));

export const InputContainer = styled(Box, {
  shouldForwardProp: (propName) => propName !== 'fullWidth',
})<CustomInputContainer>(({ theme, fullWidth }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerHigh,
  marginBottom: '8px',
  padding: '8px 12px 11px 12px',
  display: fullWidth ? 'block' : 'inline-block',
  borderRadius: '8px',
  minWidth: '168px',
  width: 'auto',
  flex: 1,
  '&.disabled': { opacity: 0.4 },
}));

export const ErrorMessage = styled(Typography)({
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  textAlign: 'center',
  minHeight: '15px',
});
