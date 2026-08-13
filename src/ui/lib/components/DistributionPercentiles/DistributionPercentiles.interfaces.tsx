export type PercentileItem = {
  label: string;
  value: string;
};

export interface DistributionPercentilesProps {
  list: PercentileItem[];
  rpsValue: number;
  units: string;
}
