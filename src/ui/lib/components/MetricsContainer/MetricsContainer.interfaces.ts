import { ReactNode } from 'react';

export interface MetricsContainerProps {
  header: string;
  leftColumn: ReactNode;
  rightColumn?: ReactNode;
  children: ReactNode;
}
