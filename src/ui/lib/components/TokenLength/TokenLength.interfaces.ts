export interface TokenLengthProps {
  label: string;
  tokenCount: number;
  bars: { x: number; y: number }[];
  lines: { x: number; y: number; id: string }[];
}
