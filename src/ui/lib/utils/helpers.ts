import { filesize } from 'filesize';

export const formatValue = (value: number, fractionDigits = 2) =>
  `$ ${value.toFixed(fractionDigits)}`;

export const ceil = (number: number, precision = 0) => {
  const n = number * Math.pow(10, precision);
  return Math.ceil(n) / Math.pow(10, precision);
};

export const floor = (number: number, precision = 0) => {
  const n = number * Math.pow(10, precision);
  return Math.floor(n) / Math.pow(10, precision);
};

export const formatNumber = (number: number, precision = 2) =>
  Number(number.toFixed(precision));

export const parseUrlParts = (urlString: string) => {
  try {
    const url = new URL(urlString);
    return {
      type: url.protocol.replace(':', ''),
      target: url.hostname,
      port: url.port || '',
      path: url.pathname,
    };
  } catch (_) {
    return {
      type: '',
      target: '',
      port: '',
      path: '',
    };
  }
};

export const getFileSize = (
  size?: string | number | null,
  roundDecimal?: number,
  bits?: boolean
): { size: string; units: string } | undefined => {
  if (size) {
    const round = roundDecimal === 0 ? 0 : roundDecimal || 1;
    const fileSize = `${filesize(size, { round, bits })}`.split(' ');

    return {
      size: fileSize[0],
      units: fileSize[1].toUpperCase(),
    };
  }
};

export const formateDate = (timestamp: string) => {
  const date = new Date(Number(timestamp) * 1000);

  const options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  };

  return date.toLocaleString('en-US', options).replace(',', '');
};
