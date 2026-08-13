import { Box, BoxProps } from '@mui/material';
import { FC, ReactNode } from 'react';

export interface Props extends BoxProps {
  body: ReactNode;
  footer: ReactNode;
  header: ReactNode;
}

export const FullPageWithHeaderAndFooterLayout: FC<Props> = ({
  body,
  footer,
  header,
  ...rest
}) => {
  return (
    <Box minHeight="100vh" display="grid" gridTemplateRows="auto 1fr auto" {...rest}>
      {header}
      {body}
      {footer}
    </Box>
  );
};
