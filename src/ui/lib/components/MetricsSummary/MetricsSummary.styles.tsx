import {
  Box,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  styled,
} from '@mui/material';

import { CustomSelectProps } from './MetricSummary.interfaces';

export const MetricsSummaryContainer = styled(Grid)(({ theme }) => ({
  borderWidth: '1px',
  borderStyle: 'solid',
  borderColor: theme.palette.outline.subdued,
  borderRadius: '8px',
  overflow: 'hidden',
  backgroundColor: theme.palette.surface.surfaceContainerLow,
}));

export const MiddleColumn = styled(Grid)(({ theme }) => ({
  borderBottomWidth: 1,
  borderBottomColor: theme.palette.outline.subdued,
  borderBottomStyle: 'solid',
  padding: '16px',
}));

export const FieldsContainer = styled('div')({
  display: 'flex',
  justifyContent: 'space-between',
  gap: '16px',
});

export const FieldCell = styled('div')({
  flex: 1,
  minWidth: 'calc((100% - 3 * 16px) / 4)',
  boxSizing: 'border-box',
});

export const HeaderLeftCell = styled(Grid)(({ theme }) => ({
  borderBottomWidth: 1,
  borderBottomColor: theme.palette.outline.subdued,
  borderBottomStyle: 'solid',
  paddingTop: '16px',
  paddingLeft: '16px',
  paddingBottom: '16px',
  paddingRight: '8px',
}));

export const HeaderRightCell = styled(Grid)(({ theme }) => ({
  borderBottomWidth: 1,
  borderBottomColor: theme.palette.outline.subdued,
  borderBottomStyle: 'solid',
  borderLeftWidth: 1,
  borderLeftColor: theme.palette.outline.subdued,
  borderLeftStyle: 'solid',
  paddingTop: '16px',
  paddingLeft: '8px',
  paddingBottom: '16px',
  paddingRight: '24px',
}));

export const FooterRightCell = styled(Grid)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerHigh,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-end',
  padding: '16px',
}));

export const FooterLeftCell = styled(Grid)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerHigh,
  paddingTop: '32px',
  paddingBottom: '16px',
  paddingLeft: '36px',
  paddingRight: '26px',
}));

export const CustomSlider = styled(Slider)(({ theme }) => ({
  '& .MuiSlider-valueLabel': {
    backgroundColor: theme.palette.surface.surfaceContainer,
    color: theme.palette.surface.onSurface,
    opacity: 1,
    fontSize: theme.typography.caption.fontSize,
    fontFamily: theme.typography.caption.fontFamily,
    fontWeight: theme.typography.caption.fontWeight,
    lineHeight: theme.typography.caption.lineHeight,
  },
  '& .MuiSlider-markLabel': {
    color: theme.palette.surface.onSurface,
    opacity: 1,
    fontSize: theme.typography.caption.fontSize,
    fontFamily: theme.typography.caption.fontFamily,
    fontWeight: theme.typography.caption.fontWeight,
    lineHeight: theme.typography.caption.lineHeight,
  },
  '& .MuiSlider-thumb': {
    position: 'relative',
    '&::before, &::after': {
      borderRadius: 0,
      content: '""',
      position: 'absolute',
      top: '-510px',
      left: '50%',
      transform: 'translateX(-50%)',
      width: '1px',
      height: '470px',
      display: 'block',
      borderLeftColor: theme.palette.primary.main,
      borderLeftWidth: '1px',
      borderLeftStyle: 'dashed',
    },
    // '&::after': {
    //   left: 'calc(50% - 1px)',
    // },
  },
}));

export const GraphContainer = styled('div')({
  width: '100%',
  height: '90px',
  overflow: 'visible',
});

export const InputContainer = styled(Box)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerHigh,
  padding: '8px 12px',
  display: 'block',
  borderRadius: '8px',
  minWidth: '168px',
  width: 'auto',
  height: '50px',
  overflow: 'hidden',
  '&.disabled': { opacity: 0.4 },
}));

export const StyledSelect = styled(Select, {
  shouldForwardProp: (propName) => propName !== 'placeholder',
})<CustomSelectProps>(({ theme, placeholder }) => ({
  '& .MuiSelect-select .notranslate::after': placeholder
    ? {
        content: `"${placeholder}"`,
        opacity: 0.42,
      }
    : {},
  '& .MuiOutlinedInput-root': {
    '& fieldset': {
      borderColor: 'transparent',
    },
    '&:hover fieldset': {
      borderColor: 'transparent',
    },
    '&.Mui-focused fieldset': {
      borderColor: 'transparent',
    },
  },
  '& .MuiSelect-select': {
    padding: '24px 14px',
  },
  '& .MuiInputLabel-root': {
    color: theme.palette.text.secondary,
    marginTop: '8px',
    marginBottom: '8px',
  },
  '& .MuiInputLabel-root.Mui-focused': {
    color: theme.palette.surface.onSurfaceSubdued,
  },
  '& .MuiInputLabel-root.MuiFormLabel-filled': {
    color: theme.palette.surface.onSurfaceSubdued,
  },
  '& .MuiInputBase-input': {
    color: theme.palette.surface.onSurfaceSubdued,
  },
}));

export const StyledInputLabel = styled(InputLabel)(({ theme }) => ({
  color: theme.palette.surface.onSurface,
  textTransform: 'uppercase',
  marginTop: '6px',
  '&.MuiInputLabel-shrink': {
    transform: 'translate(14px, 0px) scale(0.75)',
    color: theme.palette.surface.onSurface,
  },
}));

export const StyledFormControl = styled(FormControl)({
  '& .MuiOutlinedInput-root': {
    '& fieldset': {
      borderColor: 'transparent',
    },
    '&:hover fieldset': {
      borderColor: 'transparent',
    },
    '&.Mui-focused fieldset': {
      borderColor: 'transparent',
    },
  },
});

export const OptionItem = styled(MenuItem)(({ theme }) => ({
  backgroundColor: theme.palette.surface.surfaceContainerLow,
  color: theme.palette.surface.onSurface,
  '&:hover': {
    backgroundColor: theme.palette.surface.surfaceContainerHigh,
    color: theme.palette.surface.onSurface,
  },
  '&.Mui-selected': {
    backgroundColor: theme.palette.surface.surfaceContainerHigh,
    color: theme.palette.surface.onSurface,
  },
}));
