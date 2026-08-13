/**
 * Converts a hex color string to RGB values.
 *
 * @param hex - Hex color string (e.g., "#FF5733")
 * @returns Object with r, g, b numeric values
 */
function hexToRGB(hex: string) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return { r, g, b };
}

/**
 * Converts RGB values to a hex color string.
 *
 * @param r - Red value (0-255)
 * @param g - Green value (0-255)
 * @param b - Blue value (0-255)
 * @returns Hex color string (e.g., "#FF5733")
 */
function rgbToHex(r: number, g: number, b: number) {
  return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

/**
 * Blends two hex colors by a specified step amount.
 *
 * @param colorA - First hex color
 * @param colorB - Second hex color
 * @param step - Blend ratio (0 = colorA, 1 = colorB)
 * @returns Blended hex color
 */
function mixColors(colorA: string, colorB: string, step: number) {
  const colorARGB = hexToRGB(colorA);
  const colorBRGB = hexToRGB(colorB);

  const r = Math.round(colorARGB.r * (1 - step) + colorBRGB.r * step);
  const g = Math.round(colorARGB.g * (1 - step) + colorBRGB.g * step);
  const b = Math.round(colorARGB.b * (1 - step) + colorBRGB.b * step);

  return rgbToHex(r, g, b);
}

const NEURAL_WHITE = '#FFFFFF';
const NEURAL_BLACK = '#0F161F';

/**
 * Generates a complete shade palette from a base color.
 *
 * Creates lighter (W10-W100) and darker (B10-B100) variations by mixing
 * the base color with neural white and black at different ratios.
 *
 * @param color - Base hex color to generate shades from
 * @returns Object mapping shade keys to hex color values
 */
export const generateShades = (color: string) => {
  const shades: { [key: string]: string } = { '0': color };
  for (let i = 1; i <= 9; i++) {
    const lightened = mixColors(color, NEURAL_WHITE, i * 0.1);
    shades[`W${i * 10}`] = lightened;
    const darkened = mixColors(color, NEURAL_BLACK, i * 0.1);
    shades[`B${i * 10}`] = darkened;
  }
  shades.W99 = mixColors(color, NEURAL_WHITE, 0.99);
  shades.W100 = mixColors(color, NEURAL_WHITE, 1);
  shades.B99 = mixColors(color, NEURAL_BLACK, 0.99);
  shades.B100 = mixColors(color, NEURAL_BLACK, 1);
  return shades;
};
