import { Palette, createTheme } from '@mui/material/styles';
import {
  TypographyOptions,
  TypographyStyleOptions,
} from '@mui/material/styles/createTypography';

import {
  BLACK,
  WHITE,
  NEURAL_BLUE,
  NEURAL_BLUE_SHADES,
  CARIBBEAN_GREEN_SHADES,
  CARIBBEAN_GREEN,
  YELLOW,
  YELLOW_SHADES,
  ORANGE,
  ORANGE_SHADES,
  SURFACE_SHADE_1,
  SURFACE_SHADE_2,
  SURFACE_SHADE_3,
  SURFACE_CONTAINER_SHADE_1,
  SURFACE_CONTAINER_SHADE_2,
  SURFACE_CONTAINER_SHADE_3,
  SURFACE_CONTAINER_SHADE_4,
  SURFACE_CONTAINER_SHADE_5,
  DARK_SURFACE_SHADE_1,
  DARK_SURFACE_SHADE_2,
  DARK_SURFACE_SHADE_3,
  DARK_SURFACE_CONTAINER_SHADE_1,
  DARK_SURFACE_CONTAINER_SHADE_2,
  DARK_SURFACE_CONTAINER_SHADE_3,
  DARK_SURFACE_CONTAINER_SHADE_4,
  DARK_SURFACE_CONTAINER_SHADE_5,
  CHARCOAL_NAVY,
  SEA_LION,
  MIDNIGHT_BLUE,
  DEEP_FOREST_GREEN,
  DARK_BRONZE,
  CRIMSON_FLAME,
  BLUSH_PINK,
  DEEP_MAROON,
  LIME_GREEN,
  MINT_CREAM,
  FOREST_GREEN,
  STORM_GRAY,
  LIGHT_SLATE_GRAY,
  LAVENDER_GRAY,
  ASH_GRAY,
  VERY_DARK_GREEN,
  RED_SHADES,
  OUTER_SPACE_GRAY,
} from '../lib/utils/Colors';
// Spezia
import SpeziaMedium from './assets/fonts/spezia/Spezia-Medium.otf';
import SpeziaRegular from './assets/fonts/spezia/Spezia-Regular.otf';
import SpeziaMonoMedium from './assets/fonts/spezia/SpeziaMono-Medium.otf';
export const FONT_FAMILY_SPEZIA = 'Spezia, sans-serif';
export const FONT_FAMILY_SPEZIA_MONO = 'Spezia Mono, sans-serif';

// Update the Typography's variant prop options
declare module '@mui/material/Typography' {
  interface TypographyPropsVariantOverrides {
    metric: true;
    metric1: true;
    metric2: true;
    overline1: true;
    overline2: true;
    axisTitle: true;
    axisLabel: true;
  }
}

declare module '@mui/material/styles' {
  interface TypographyVariants {
    metric: TypographyStyleOptions;
    metric1: TypographyStyleOptions;
    metric2: TypographyStyleOptions;
    overline1: TypographyStyleOptions;
    overline2: TypographyStyleOptions;
    axisTitle: TypographyStyleOptions;
    axisLabel: TypographyStyleOptions;
  }

  // allow configuration using `createTheme`
  interface TypographyVariantsOptions {
    metric?: TypographyStyleOptions;
    metric1?: TypographyStyleOptions;
    metric2?: TypographyStyleOptions;
    overline1?: TypographyStyleOptions;
    overline2?: TypographyStyleOptions;
    axisTitle?: TypographyStyleOptions;
    axisLabel?: TypographyStyleOptions;
  }

  interface PaletteColor {
    subdued: string;
    accent: string;
    main: string;
    onMain: string;
    shades: {
      [key: string]: string;
    };
    container: string;
    onContainer: string;
  }

  interface SimplePaletteColorOptions {
    subdued?: string;
    accent?: string;
    onMain?: string;
    shades?: Record<string, string>;
    container?: string;
    onContainer?: string;
  }

  interface Palette {
    outline: PaletteColor;
    quarternary: PaletteColor;
    scrim: PaletteColor;
    shadow: PaletteColor;
    surface: {
      onSurface: string;
      onSurfaceAccent: string;
      onSurfaceSubdued: string;
      surface: string;
      surfaceContainer: string;
      surfaceContainerHigh: string;
      surfaceContainerHighest: string;
      surfaceContainerLow: string;
      surfaceContainerLowest: string;
      surfaceSubdued: string;
      surfaceSubduedContainer: string;
      surfaceAccent: string;
    };
    tertiary: PaletteColor;
  }

  interface PaletteOptions {
    outline: SimplePaletteColorOptions;
    quarternary: SimplePaletteColorOptions;
    scrim: SimplePaletteColorOptions;
    shadow: SimplePaletteColorOptions;
    surface: Palette['surface'];
    tertiary: SimplePaletteColorOptions;
  }
}

const PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS = {
  main: '#000',
  onMain: '#000',
  subdued: '#000',
  accent: '#000',
  shades: {},
  container: '#000',
  onContainer: '#000',
};

const themeV3Typography:
  | TypographyOptions
  | ((palette: Palette) => TypographyOptions)
  | undefined = {
  fontFamily: FONT_FAMILY_SPEZIA,
  fontWeightRegular: 400,
  fontWeightMedium: 500,
  h1: {
    color: BLACK,
    fontSize: 96,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '120px',
    '@media (max-width:600px)': {
      fontSize: 56,
      lineHeight: '70px',
    },
  },
  h2: {
    color: BLACK,
    fontSize: 60,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '75px',
    '@media (max-width:600px)': {
      fontSize: 38,
      lineHeight: '48px',
    },
  },
  h3: {
    color: BLACK,
    fontSize: 48,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '60px',
    '@media (max-width:600px)': {
      fontSize: 32,
      lineHeight: '40px',
    },
  },
  h4: {
    color: BLACK,
    fontSize: 32,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '40px',
    '@media (max-width:600px)': {
      fontSize: 24,
      lineHeight: '30px',
    },
  },
  h5: {
    color: BLACK,
    fontSize: 24,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '30px',
    '@media (max-width:600px)': {
      fontSize: 20,
      lineHeight: '25px',
    },
  },
  h6: {
    color: BLACK,
    fontSize: 20,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '25px',
    '@media (max-width:600px)': {
      fontSize: 18,
      lineHeight: '25px',
    },
  },
  body1: {
    color: BLACK,
    fontSize: 18,
    fontFamily: FONT_FAMILY_SPEZIA,
    fontWeight: 400,
    letterSpacing: '0.25px',
    lineHeight: '22px',
    '@media (max-width:600px)': {
      fontSize: 16,
      lineHeight: '20px',
    },
  },
  body2: {
    color: BLACK,
    fontSize: 16,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    letterSpacing: '0.25px',
    lineHeight: '20px',
    '@media (max-width:600px)': {
      fontSize: 14,
      lineHeight: '17px',
    },
  },
  subtitle1: {
    color: BLACK,
    fontSize: 22,
    lineHeight: '26px',
    fontWeight: 500,
    fontFamily: FONT_FAMILY_SPEZIA,
    '@media (max-width:600px)': {
      fontSize: 20,
      lineHeight: '25px',
    },
  },
  subtitle2: {
    color: BLACK,
    fontSize: 16,
    fontWeight: 500,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '20px',
    '@media (max-width:600px)': {
      fontSize: 14,
      lineHeight: '17px',
    },
  },
  caption: {
    color: BLACK,
    fontSize: 12,
    fontWeight: 400,
    lineHeight: '15px',
    fontFamily: FONT_FAMILY_SPEZIA,
    letterSpacing: '0.25px',
    '@media (max-width:600px)': {
      fontSize: 10,
      lineHeight: '12px',
    },
  },
  button: {
    fontSize: 16,
    fontWeight: 500,
    fontFamily: FONT_FAMILY_SPEZIA_MONO,
    lineHeight: '20px',
    textTransform: 'uppercase',
    '@media (max-width:600px)': {
      fontSize: 14,
      lineHeight: '17px',
    },
  },
  overline1: {
    color: BLACK,
    fontSize: 14,
    fontWeight: 500,
    fontFamily: FONT_FAMILY_SPEZIA_MONO,
    lineHeight: '20px',
    textTransform: 'uppercase',
    '@media (max-width:600px)': {
      fontSize: 10,
      lineHeight: '12px',
    },
  },
  overline2: {
    color: BLACK,
    fontSize: 12,
    fontWeight: 500,
    fontFamily: FONT_FAMILY_SPEZIA_MONO,
    lineHeight: '20px',
    textTransform: 'uppercase',
    '@media (max-width:600px)': {
      fontSize: 10,
      lineHeight: '12px',
    },
  },
  metric1: {
    color: BLACK,
    fontSize: 32,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '26px',
    '@media (max-width:600px)': {
      fontSize: 10,
      lineHeight: '12px',
    },
  },
  metric2: {
    color: BLACK,
    fontSize: 22,
    fontWeight: 400,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '26px',
    '@media (max-width:600px)': {
      fontSize: 10,
      lineHeight: '12px',
    },
  },
  axisTitle: {
    color: BLACK,
    fontSize: 10,
    fontWeight: 500,
    fontFamily: FONT_FAMILY_SPEZIA_MONO,
    lineHeight: '14px',
    '@media (max-width:600px)': {
      fontSize: 10,
      lineHeight: '12px',
    },
  },
  axisLabel: {
    color: BLACK,
    fontSize: 8,
    fontWeight: 500,
    fontFamily: FONT_FAMILY_SPEZIA,
    lineHeight: '8px',
    '@media (max-width:600px)': {
      fontSize: 10,
      lineHeight: '12px',
    },
  },
};

const themeV3FontStyles = {
  fallbacks: [
    // -- Spezia --
    {
      '@font-face': {
        fontFamily: 'Spezia',
        fontStyle: 'normal',
        fontWeight: 400,
        src: `url(${SpeziaRegular}) format('truetype')`,
      },
    },
    {
      '@font-face': {
        fontFamily: 'Spezia',
        fontStyle: 'normal',
        fontWeight: 500,
        src: `url(${SpeziaMedium}) format('truetype')`,
      },
    },
    {
      '@font-face': {
        fontFamily: 'SpeziaMono',
        fontStyle: 'normal',
        fontWeight: 500,
        src: `url(${SpeziaMonoMedium}) format('truetype')`,
      },
    },
  ],
};

export const muiThemeV3Light = createTheme({
  palette: {
    primary: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: NEURAL_BLUE,
      shades: NEURAL_BLUE_SHADES,
      container: NEURAL_BLUE_SHADES.W80,
      onContainer: MIDNIGHT_BLUE,
    },
    secondary: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: CARIBBEAN_GREEN,
      shades: CARIBBEAN_GREEN_SHADES,
      container: CARIBBEAN_GREEN_SHADES.W80,
      onContainer: DEEP_FOREST_GREEN,
    },
    tertiary: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: YELLOW,
      shades: YELLOW_SHADES,
      container: YELLOW_SHADES.W80,
      onContainer: DARK_BRONZE,
    },
    quarternary: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: ORANGE,
      shades: ORANGE_SHADES,
    },
    surface: {
      onSurface: CHARCOAL_NAVY,
      onSurfaceAccent: NEURAL_BLUE,
      onSurfaceSubdued: SEA_LION,
      surface: SURFACE_SHADE_2,
      surfaceContainer: SURFACE_CONTAINER_SHADE_3,
      surfaceContainerHigh: SURFACE_CONTAINER_SHADE_4,
      surfaceContainerHighest: SURFACE_CONTAINER_SHADE_5,
      surfaceContainerLow: SURFACE_CONTAINER_SHADE_2,
      surfaceContainerLowest: SURFACE_CONTAINER_SHADE_1,
      surfaceSubdued: SURFACE_SHADE_3,
      surfaceSubduedContainer: SURFACE_SHADE_2,
      surfaceAccent: SURFACE_SHADE_1,
    },
    error: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: CRIMSON_FLAME,
      onMain: WHITE,
      container: BLUSH_PINK,
      onContainer: DEEP_MAROON,
    },
    success: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: LIME_GREEN,
      onMain: WHITE,
      container: MINT_CREAM,
      onContainer: FOREST_GREEN,
    },
    outline: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: STORM_GRAY,
      subdued: LIGHT_SLATE_GRAY,
    },
    scrim: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: BLACK,
    },
    shadow: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: BLACK,
    },
  },
  spacing: 8,
  typography: themeV3Typography,
  components: {
    MuiCssBaseline: {
      styleOverrides: themeV3FontStyles,
    },
  },
});

export const themeV3LightName = 'muiThemeV3Light';

const DEFAULT_SURFACE_OPTIONS = {
  onSurface: CHARCOAL_NAVY,
  onSurfaceAccent: NEURAL_BLUE,
  onSurfaceSubdued: SEA_LION,
  surface: SURFACE_SHADE_2,
  surfaceContainer: SURFACE_CONTAINER_SHADE_3,
  surfaceContainerHigh: SURFACE_CONTAINER_SHADE_4,
  surfaceContainerHighest: SURFACE_CONTAINER_SHADE_5,
  surfaceContainerLow: SURFACE_CONTAINER_SHADE_2,
  surfaceContainerLowest: SURFACE_CONTAINER_SHADE_1,
  surfaceSubdued: SURFACE_SHADE_3,
  surfaceAccent: SURFACE_SHADE_1,
};

export const muiThemeV3Dark = createTheme({
  palette: {
    primary: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: NEURAL_BLUE,
      shades: NEURAL_BLUE_SHADES,
      container: NEURAL_BLUE_SHADES.B99,
      onContainer: NEURAL_BLUE_SHADES.W80,
    },
    secondary: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: CARIBBEAN_GREEN,
      shades: CARIBBEAN_GREEN_SHADES,
      container: CARIBBEAN_GREEN_SHADES.B99,
      onContainer: CARIBBEAN_GREEN_SHADES.W80,
    },
    tertiary: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: YELLOW,
      shades: YELLOW_SHADES,
      container: YELLOW_SHADES.B99,
      onContainer: YELLOW_SHADES.W90,
    },
    quarternary: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: ORANGE,
      shades: ORANGE_SHADES,
    },
    surface: {
      ...DEFAULT_SURFACE_OPTIONS,
      onSurface: LAVENDER_GRAY,
      onSurfaceAccent: NEURAL_BLUE,
      onSurfaceSubdued: ASH_GRAY,
      surface: DARK_SURFACE_SHADE_1,
      surfaceContainer: DARK_SURFACE_CONTAINER_SHADE_3,
      surfaceContainerHigh: DARK_SURFACE_CONTAINER_SHADE_4,
      surfaceContainerHighest: DARK_SURFACE_CONTAINER_SHADE_5,
      surfaceContainerLow: DARK_SURFACE_CONTAINER_SHADE_2,
      surfaceContainerLowest: DARK_SURFACE_CONTAINER_SHADE_1,
      surfaceSubdued: DARK_SURFACE_SHADE_3,
      surfaceSubduedContainer: SURFACE_SHADE_2,
      surfaceAccent: DARK_SURFACE_SHADE_2,
    },
    error: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: CRIMSON_FLAME,
      onMain: WHITE,
      container: RED_SHADES.B90,
      onContainer: RED_SHADES.W90,
    },
    success: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: LIME_GREEN,
      onMain: WHITE,
      container: VERY_DARK_GREEN,
      onContainer: MINT_CREAM,
    },
    outline: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: STORM_GRAY,
      subdued: OUTER_SPACE_GRAY,
    },
    scrim: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: BLACK,
    },
    shadow: {
      ...PALETTE_COLOR_OPTIONS_PLACEHOLDER_TOKENS,
      main: BLACK,
    },
  },
  spacing: 8,
  typography: themeV3Typography,
  components: {
    MuiCssBaseline: {
      styleOverrides: themeV3FontStyles,
    },
  },
});

export const themeV3DarkName = 'muiThemeV3Dark';
