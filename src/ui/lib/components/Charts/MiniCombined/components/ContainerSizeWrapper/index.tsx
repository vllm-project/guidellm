import React, { useState, useEffect, useRef } from 'react';

interface ContainerSizeWrapperProps {
  children: (containerSize: ContainerSize) => React.ReactNode;
}

export interface ContainerSize {
  width: number;
  height: number;
}

const ContainerSizeWrapper: React.FC<ContainerSizeWrapperProps> = ({ children }) => {
  const [containerSize, setContainerSize] = useState<ContainerSize>({
    width: 0,
    height: 0,
  });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setContainerSize({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);

    return () => window.removeEventListener('resize', updateSize);
  }, []);

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
      {children(containerSize)}
    </div>
  );
};

export default ContainerSizeWrapper;
