import { useTheme } from '@mui/material';
import { ResponsiveLine, Serie } from '@nivo/line';

import { CustomLegendLayer } from './components/CustomLegendLayer';
import { DashedSolidLine } from './components/DashedSolidLine';
import { DashedLineProps, ScaleType } from './DashedLine.interfaces';
import { spacedLogValues } from './helpers';

export const getMinTick = (data: readonly Serie[]) => {
  return Math.min(
    ...data.map((lineData) =>
      Math.min(...lineData.data.map((point) => point.y as number))
    )
  );
};

export const getMaxTick = (data: readonly Serie[]) => {
  return Math.max(
    ...data.map((lineData) =>
      Math.max(...lineData.data.map((point) => point.y as number))
    )
  );
};

export const Component = ({
  data,
  xLegend,
  yLegend,
  margins,
  minX,
  yScaleType = ScaleType.log,
}: DashedLineProps) => {
  const theme = useTheme();
  const defaultMargins = { top: 10, left: 10, right: 10, bottom: 10 };
  const finalMargins = {
    ...defaultMargins,
    ...margins,
    bottom: (margins?.bottom || defaultMargins.bottom) + 50,
  };

  const dashedLineTheme = {
    textColor: theme.palette.surface.onSurface,
    fontSize: 14,
    axis: {
      domain: {
        line: {
          stroke: theme.palette.outline.subdued,
          strokeWidth: 1,
        },
      },
      ticks: {
        line: {
          stroke: theme.palette.outline.subdued,
          strokeWidth: 1,
        },
        text: {
          fill: theme.palette.surface.onSurface,
          fontSize: theme.typography.axisTitle.fontSize,
          fontFamily: theme.typography.axisTitle.fontFamily,
          fontWeight: theme.typography.axisTitle.fontWeight,
        },
      },
      legend: {
        text: {
          fill: theme.palette.surface.onSurface,
          fontSize: theme.typography.axisTitle.fontSize,
          fontFamily: theme.typography.axisTitle.fontFamily,
          fontWeight: theme.typography.axisTitle.fontWeight,
        },
      },
    },
    grid: {
      line: {
        stroke: theme.palette.outline.subdued,
        strokeWidth: 1,
      },
    },
  };

  let extraLeftAxisOptions = {};
  let extraYScaleOptions = {};
  if (yScaleType === ScaleType.log) {
    const ticks = spacedLogValues(
      Math.floor(getMinTick(data)),
      Math.ceil(getMaxTick(data)),
      6
    );
    extraLeftAxisOptions = {
      tickValues: ticks,
    };
    extraYScaleOptions = {
      min: ticks[0],
      max: ticks[ticks.length - 1],
    };
  }

  return (
    <div style={{ height: '100%', width: '100%' }}>
      <ResponsiveLine
        data={data}
        enablePoints={false}
        curve="monotoneX"
        margin={finalMargins}
        axisTop={null}
        axisRight={null}
        axisBottom={{
          tickSize: 0,
          tickPadding: 5,
          tickRotation: 0,
          legend: xLegend,
          legendOffset: 36,
          legendPosition: 'middle',
        }}
        axisLeft={{
          tickSize: 0,
          tickPadding: 5,
          tickRotation: 0,
          legend: yLegend,
          legendOffset: -40,
          legendPosition: 'middle',
          ...extraLeftAxisOptions,
        }}
        xScale={{
          min: minX,
          type: 'linear',
        }}
        yScale={{
          type: yScaleType,
          ...extraYScaleOptions,
        }}
        colors={{ scheme: 'category10' }}
        layers={[
          'markers',
          'axes',
          'points',
          'legends',
          DashedSolidLine,
          CustomLegendLayer,
        ]}
        theme={dashedLineTheme}
      />
    </div>
  );
};
