import { ReactNode } from 'react';

export interface DataPanelProps {
  header: string;
  topContainer: ReactNode;
  bottomContainer?: ReactNode;
}
