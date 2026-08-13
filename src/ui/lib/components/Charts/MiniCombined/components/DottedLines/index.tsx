import { DottedLinesProps } from './DottedLines.interfaces';
import useLineColors from '../../../common/useLineColors';

const DottedLines = ({
  lines,
  leftMargin,
  topMargin,
  xScale,
  innerHeight,
}: DottedLinesProps) => {
  const lineColor = useLineColors();
  return (
    <svg
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
      width="100%"
      height="100%"
    >
      <g transform={`translate(${leftMargin}, ${topMargin})`}>
        {lines.map((line, i) => (
          <line
            key={i}
            x1={xScale(line.x)}
            y1="0"
            x2={xScale(line.x)}
            y2={innerHeight}
            stroke={lineColor[i]}
            strokeWidth="1.5"
            strokeDasharray="3,3"
          />
        ))}
      </g>
    </svg>
  );
};

export default DottedLines;
